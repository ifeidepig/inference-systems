"""Cold versus aligned Hybrid Prefix Cache benchmark for Qwen3.5."""

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
    *,
    no_share: bool = False,
):
    safe_vocab = min(vocab_size, 200_000)
    shared = [(index * 37 + 11) % safe_vocab for index in range(shared_length)]
    source = shared + [
        (index * 13 + 7) % safe_vocab for index in range(suffix_length)
    ]
    target_shared = (
        [(index * 41 + 23) % safe_vocab for index in range(shared_length)]
        if no_share
        else shared
    )
    target = target_shared + [
        (index * 17 + 19) % safe_vocab for index in range(suffix_length)
    ]
    return source, target


def run_case(
    args,
    enabled: bool,
    *,
    internal_checkpoints: bool | None = None,
) -> dict:
    if internal_checkpoints is None:
        internal_checkpoints = args.internal_checkpoints
    checkpoint_bytes = args.checkpoint_memory_mib * 1024 * 1024 if enabled else 0
    engine = LLM(
        args.model,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=1,
        max_model_len=args.shared_prefix_length
        + args.unique_suffix_length
        + args.output_tokens
        + 8,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=not args.cuda_graph,
        max_num_kvcache_blocks=args.max_num_kvcache_blocks,
        max_num_state_slots=1,
        prefix_match_unit=args.prefix_match_unit,
        enable_prefix_cache=enabled,
        enable_hybrid_prefix_cache=enabled,
        hybrid_prefix_checkpoint_interval_blocks=args.interval_blocks,
        hybrid_prefix_checkpoint_interval_tokens=args.interval_tokens,
        hybrid_prefix_checkpoint_memory_bytes=checkpoint_bytes,
        hybrid_prefix_retention_policy=args.retention_policy,
        hybrid_prefix_eviction_policy=args.eviction_policy,
        enable_hybrid_internal_checkpoints=(
            enabled and internal_checkpoints
        ),
        gdn_decode_backend=getattr(args, "gdn_decode_backend", "torch"),
    )
    source, target = build_prompts(
        engine.config.hf_config.vocab_size,
        args.shared_prefix_length,
        args.unique_suffix_length,
        no_share=getattr(args, "no_share", False),
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.output_tokens,
        ignore_eos=True,
    )
    try:
        # Matching-shape producer/warmup. Only the enabled case publishes
        # recurrent checkpoints; both cases warm the same model shapes.
        engine.reset_runtime_metrics()
        engine.generate([source], sampling, use_tqdm=False)
        producer_metrics = engine.get_runtime_metrics()
        producer_request = engine.get_request_metrics()[-1]
        entries_after_source = producer_metrics["scheduler"].get(
            "hybrid_prefix_cache_entries", 0
        )
        engine.reset_runtime_metrics()
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        output = engine.generate([target], sampling, use_tqdm=False)[0]
        elapsed = perf_counter() - started
        metrics = engine.get_runtime_metrics()
        request = engine.get_request_metrics()[-1]
        return {
            "enabled": enabled,
            "gdn_decode_backend": getattr(
                args, "gdn_decode_backend", "torch"
            ),
            "internal_checkpoints": (
                internal_checkpoints if enabled else False
            ),
            "retention_policy": args.retention_policy if enabled else "none",
            "eviction_policy": args.eviction_policy if enabled else "none",
            "elapsed_s": elapsed,
            "throughput_tok_s": len(output["token_ids"]) / elapsed,
            "ttft_ms": request["ttft_ms"],
            "tpot_ms": request["tpot_ms"],
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "entries_after_source": entries_after_source,
            "producer_request": producer_request,
            "producer_scheduler": producer_metrics["scheduler"],
            "producer_model_runner": producer_metrics["model_runner"],
            "token_ids": output["token_ids"],
            "scheduler": metrics["scheduler"],
            "model_runner": metrics["model_runner"],
        }
    finally:
        engine.exit()
        del engine
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--shared-prefix-length", type=int, default=2048)
    parser.add_argument("--unique-suffix-length", type=int, default=64)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--interval-blocks", type=int, default=8)
    parser.add_argument("--interval-tokens", type=int, default=None)
    parser.add_argument("--prefix-match-unit", type=int, default=256)
    parser.add_argument("--checkpoint-memory-mib", type=int, default=512)
    parser.add_argument(
        "--retention-policy",
        choices=("periodic", "adaptive"),
        default="periodic",
    )
    parser.add_argument(
        "--eviction-policy",
        choices=("lru", "cost_aware"),
        default="lru",
    )
    parser.add_argument("--internal-checkpoints", action="store_true")
    parser.add_argument("--compare-internal", action="store_true")
    parser.add_argument("--no-share", action="store_true")
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-kvcache-blocks", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument(
        "--gdn-decode-backend",
        choices=("torch", "cuda", "auto"),
        default="torch",
    )
    args = parser.parse_args()
    if 256 % args.prefix_match_unit:
        parser.error("prefix match unit must divide physical block size 256")
    if args.checkpoint_memory_mib <= 0:
        parser.error("checkpoint memory budget must be positive")

    baseline = run_case(args, False, internal_checkpoints=False)
    variants = []
    if args.compare_internal:
        variants.append(run_case(args, True, internal_checkpoints=False))
        variants.append(run_case(args, True, internal_checkpoints=True))
    else:
        variants.append(run_case(args, True))
    for hybrid in variants:
        hybrid["tokens_match_baseline"] = (
            hybrid["token_ids"] == baseline["token_ids"]
        )
        hybrid["ttft_change_percent"] = (
            hybrid["ttft_ms"] / baseline["ttft_ms"] - 1
        ) * 100
        if not hybrid["tokens_match_baseline"]:
            raise SystemExit(
                "Hybrid Prefix Cache output differs from cold baseline"
            )
    print(json.dumps([baseline, *variants], indent=2))


if __name__ == "__main__":
    main()
