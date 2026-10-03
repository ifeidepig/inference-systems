"""A/B/C benchmark for demand-driven Hybrid Prefix Cache promotion."""

import argparse
import gc
import json
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams


def build_prompts(
    vocab_size: int,
    shared_length: int,
    suffix_length: int,
    num_requests: int = 3,
) -> list[list[int]]:
    safe_vocab = min(vocab_size, 200_000)
    shared = [
        (index * 37 + 11) % safe_vocab
        for index in range(shared_length)
    ]
    prompts = []
    for request_index in range(num_requests):
        seed = 101 + request_index * 107
        suffix = [
            (index * (17 + request_index * 2) + seed) % safe_vocab
            for index in range(suffix_length)
        ]
        prompts.append(shared + suffix)
    return prompts


def run_request(engine: LLM, prompt: list[int], sampling) -> dict:
    engine.reset_runtime_metrics()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    output = engine.generate([prompt], sampling, use_tqdm=False)[0]
    elapsed = perf_counter() - started
    runtime = engine.get_runtime_metrics()
    request = engine.get_request_metrics()[-1]
    scheduler = runtime["scheduler"]
    model_runner = runtime["model_runner"]
    return {
        "elapsed_s": elapsed,
        "token_ids": output["token_ids"],
        "ttft_ms": request["ttft_ms"],
        "tpot_ms": request["tpot_ms"],
        "schedule_count": request["schedule_count"],
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "kv_candidate_tokens": scheduler.get(
            "hybrid_prefix_kv_candidate_tokens", 0
        ),
        "committed_hit_tokens": scheduler.get(
            "hybrid_prefix_committed_hit_tokens", 0
        ),
        "kv_only_misses": scheduler.get(
            "hybrid_prefix_kv_only_miss_count", 0
        ),
        "alignment_lost_tokens": scheduler.get(
            "hybrid_prefix_alignment_lost_tokens", 0
        ),
        "promotions_planned": scheduler.get(
            "hybrid_prefix_shared_junction_promotions_planned", 0
        ),
        "promotions_published": scheduler.get(
            "hybrid_prefix_shared_junction_promotions_published", 0
        ),
        "checkpoint_entries": scheduler.get(
            "hybrid_prefix_cache_entries", 0
        ),
        "checkpoint_dtype": model_runner.get(
            "hybrid_prefix_checkpoint_dtype", "none"
        ),
        "checkpoint_slots_total": model_runner.get(
            "hybrid_prefix_checkpoint_slots_total", 0
        ),
        "checkpoint_bytes_per_slot": model_runner.get(
            "hybrid_prefix_checkpoint_bytes_per_slot", 0
        ),
        "checkpoint_bytes": model_runner.get(
            "hybrid_prefix_checkpoint_bytes", 0
        ),
        "capture_ms": model_runner.get("hybrid_prefix_capture_ms", 0.0),
        "restore_ms": model_runner.get("hybrid_prefix_restore_ms", 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--shared-prefix-length", type=int, default=496)
    parser.add_argument("--unique-suffix-length", type=int, default=64)
    parser.add_argument("--output-tokens", type=int, default=2)
    parser.add_argument(
        "--setup-output-tokens",
        type=int,
        default=None,
        help="A/B output length; defaults to --output-tokens.",
    )
    parser.add_argument("--prefix-match-unit", type=int, default=16)
    parser.add_argument("--checkpoint-memory-mib", type=int, default=128)
    parser.add_argument(
        "--checkpoint-dtype",
        choices=("fp32", "bf16", "int8"),
        default="fp32",
    )
    parser.add_argument(
        "--promotion-min-sightings",
        type=int,
        default=2,
    )
    parser.add_argument("--max-num-kvcache-blocks", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--split-checkpoints", action="store_true")
    parser.add_argument(
        "--gdn-decode-backend",
        choices=("torch", "cuda", "auto"),
        default="torch",
    )
    args = parser.parse_args()
    if 256 % args.prefix_match_unit:
        parser.error("prefix match unit must divide physical block size 256")
    if args.shared_prefix_length % args.prefix_match_unit:
        parser.error("shared prefix must align to prefix match unit")
    if args.promotion_min_sightings < 2:
        parser.error("promotion min sightings must be at least 2")

    prompt_length = (
        args.shared_prefix_length + args.unique_suffix_length
    )
    engine = LLM(
        args.model,
        max_num_batched_tokens=prompt_length + 8,
        max_num_seqs=1,
        max_model_len=prompt_length + args.output_tokens + 8,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=not args.cuda_graph,
        max_num_kvcache_blocks=args.max_num_kvcache_blocks,
        max_num_state_slots=1,
        prefix_match_unit=args.prefix_match_unit,
        enable_prefix_cache=True,
        enable_hybrid_prefix_cache=True,
        hybrid_prefix_checkpoint_interval_tokens=4096,
        hybrid_prefix_checkpoint_memory_bytes=(
            args.checkpoint_memory_mib * 1024 * 1024
        ),
        hybrid_prefix_checkpoint_dtype=args.checkpoint_dtype,
        hybrid_prefix_promotion_min_sightings=(
            args.promotion_min_sightings
        ),
        hybrid_prefix_retention_policy="adaptive",
        hybrid_prefix_eviction_policy="cost_aware",
        enable_hybrid_internal_checkpoints=(
            not args.split_checkpoints
        ),
        gdn_decode_backend=args.gdn_decode_backend,
    )
    prompts = build_prompts(
        engine.config.hf_config.vocab_size,
        args.shared_prefix_length,
        args.unique_suffix_length,
        num_requests=args.promotion_min_sightings + 1,
    )
    setup_sampling = SamplingParams(
        temperature=0.0,
        max_tokens=(
            args.setup_output_tokens
            if args.setup_output_tokens is not None
            else args.output_tokens
        ),
        ignore_eos=True,
    )
    reuse_sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.output_tokens,
        ignore_eos=True,
    )
    try:
        stages = []
        for index, prompt in enumerate(prompts):
            sighting = index + 1
            if sighting == 1:
                label = "producer"
            elif sighting == args.promotion_min_sightings:
                label = "promoter"
            elif sighting > args.promotion_min_sightings:
                label = "reuser"
            else:
                label = "observer"
            sampling = (
                reuse_sampling
                if sighting > args.promotion_min_sightings
                else setup_sampling
            )
            result = run_request(engine, prompt, sampling)
            result["stage"] = f"request_{sighting}_{label}"
            stages.append(result)
        promoter = stages[args.promotion_min_sightings - 1]
        reuser = stages[args.promotion_min_sightings]
        if promoter["promotions_published"] < 1:
            raise SystemExit(
                "threshold request did not publish a shared-junction checkpoint"
            )
        if reuser["committed_hit_tokens"] < args.shared_prefix_length:
            raise SystemExit(
                "post-threshold request did not restore the promoted boundary"
            )
        print(json.dumps(stages, indent=2))
    finally:
        engine.exit()
        del engine
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
