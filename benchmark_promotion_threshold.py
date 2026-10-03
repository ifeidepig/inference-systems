"""Frequency-aware benchmark for shared-junction promotion thresholds."""

import argparse
import gc
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams


def build_trace(
    workload: str,
    *,
    num_prefixes: int,
    hot_frequency: int,
    zipf_requests: int,
    seed: int,
) -> list[tuple[int, int]]:
    if workload == "singleton":
        counts = [1] * num_prefixes
    elif workload == "pair":
        counts = [2] * num_prefixes
    elif workload == "triple":
        counts = [3] * num_prefixes
    elif workload == "hot":
        counts = [hot_frequency] * max(1, num_prefixes // 4)
    elif workload == "zipf":
        rng = random.Random(seed)
        population = list(range(num_prefixes))
        weights = [1.0 / (rank + 1) for rank in population]
        sampled = rng.choices(
            population,
            weights=weights,
            k=zipf_requests,
        )
        counts_by_prefix = Counter(sampled)
        counts = [counts_by_prefix[index] for index in population]
    else:
        raise ValueError(f"unsupported workload: {workload}")

    trace = []
    for prefix_id, frequency in enumerate(counts):
        trace.extend((prefix_id, occurrence) for occurrence in range(frequency))
    random.Random(seed + 1).shuffle(trace)
    return trace


def build_prompt(
    vocab_size: int,
    prefix_id: int,
    occurrence: int,
    shared_length: int,
    suffix_length: int,
    prompt_seed: int = 0,
) -> list[int]:
    safe_vocab = min(vocab_size, 200_000)
    shared_seed = 1009 + prompt_seed * 104729 + prefix_id * 7919
    shared = [
        (shared_seed + index * 37) % safe_vocab
        for index in range(shared_length)
    ]
    suffix_seed = (
        50021
        + prompt_seed * 13007
        + prefix_id * 1543
        + occurrence * 3571
    )
    suffix = [
        (suffix_seed + index * 43) % safe_vocab
        for index in range(suffix_length)
    ]
    return shared + suffix


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(int(len(ordered) * fraction), len(ordered) - 1)
    return ordered[index]


def run_threshold(args, threshold: int, trace) -> dict:
    prompt_length = args.shared_prefix_length + args.unique_suffix_length
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
        hybrid_prefix_promotion_min_sightings=threshold,
        hybrid_prefix_retention_policy="adaptive",
        hybrid_prefix_eviction_policy="cost_aware",
        enable_hybrid_internal_checkpoints=True,
        gdn_decode_backend=args.gdn_decode_backend,
        request_metrics_history_size=max(len(trace) + 8, 64),
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.output_tokens,
        ignore_eos=True,
    )
    seen = defaultdict(int)
    request_rows = []
    token_digest = hashlib.sha256()
    try:
        for warmup_index in range(args.warmup_requests):
            warmup_prompt = build_prompt(
                engine.config.hf_config.vocab_size,
                100_000 + warmup_index,
                1,
                args.shared_prefix_length,
                args.unique_suffix_length,
                prompt_seed=args.seed,
            )
            engine.generate([warmup_prompt], sampling, use_tqdm=False)
        engine.reset_runtime_metrics()
        started = perf_counter()
        for request_index, (prefix_id, _) in enumerate(trace):
            occurrence = seen[prefix_id] + 1
            seen[prefix_id] = occurrence
            prompt = build_prompt(
                engine.config.hf_config.vocab_size,
                prefix_id,
                occurrence,
                args.shared_prefix_length,
                args.unique_suffix_length,
                prompt_seed=args.seed,
            )
            output = engine.generate([prompt], sampling, use_tqdm=False)[0]
            token_ids = output["token_ids"]
            token_digest.update(
                json.dumps(token_ids, separators=(",", ":")).encode()
            )
            request = engine.get_request_metrics()[-1]
            request_rows.append(
                {
                    "request_index": request_index,
                    "prefix_id": prefix_id,
                    "occurrence": occurrence,
                    "ttft_ms": request["ttft_ms"],
                    "tpot_ms": request["tpot_ms"],
                    "token_ids": token_ids,
                }
            )
        elapsed = perf_counter() - started
        runtime = engine.get_runtime_metrics()
        scheduler = runtime["scheduler"]
        model_runner = runtime["model_runner"]
        ttfts = [row["ttft_ms"] for row in request_rows]
        ttft_by_occurrence = {}
        for occurrence in sorted({row["occurrence"] for row in request_rows}):
            values = [
                row["ttft_ms"]
                for row in request_rows
                if row["occurrence"] == occurrence
            ]
            ttft_by_occurrence[str(occurrence)] = {
                "count": len(values),
                "median_ms": statistics.median(values),
                "p95_ms": percentile(values, 0.95),
            }
        return {
            "threshold": threshold,
            "workload": args.workload,
            "request_count": len(trace),
            "prefix_frequencies": dict(sorted(Counter(x[0] for x in trace).items())),
            "elapsed_s": elapsed,
            "token_digest": token_digest.hexdigest(),
            "ttft_median_ms": statistics.median(ttfts),
            "ttft_p95_ms": percentile(ttfts, 0.95),
            "ttft_by_occurrence": ttft_by_occurrence,
            "promotions_published": scheduler.get(
                "hybrid_prefix_shared_junction_promotions_published", 0
            ),
            "useful_promotions": scheduler.get(
                "hybrid_prefix_useful_promotion_count", 0
            ),
            "useful_promotion_ratio": scheduler.get(
                "hybrid_prefix_useful_promotion_ratio", 0.0
            ),
            "promotions_not_yet_reused": scheduler.get(
                "hybrid_prefix_promotions_not_yet_reused", 0
            ),
            "unused_promotion_evictions": scheduler.get(
                "hybrid_prefix_unused_promotion_eviction_count", 0
            ),
            "alignment_lost_tokens": scheduler.get(
                "hybrid_prefix_alignment_lost_tokens", 0
            ),
            "replay_due_to_missing_checkpoint_tokens": scheduler.get(
                "hybrid_prefix_replay_due_to_missing_checkpoint_tokens", 0
            ),
            "shared_junction_saved_replay_tokens": scheduler.get(
                "hybrid_prefix_shared_junction_saved_replay_tokens", 0
            ),
            "checkpoint_entries": scheduler.get(
                "hybrid_prefix_cache_entries", 0
            ),
            "checkpoint_peak_entries": scheduler.get(
                "hybrid_prefix_peak_entries", 0
            ),
            "shared_junction_entries": scheduler.get(
                "hybrid_prefix_entries_shared_junction", 0
            ),
            "shared_junction_peak_entries": scheduler.get(
                "hybrid_prefix_peak_shared_junction_entries", 0
            ),
            "checkpoint_evictions": scheduler.get(
                "hybrid_prefix_eviction_count", 0
            ),
            "checkpoint_slots_total": model_runner.get(
                "hybrid_prefix_checkpoint_slots_total", 0
            ),
            "checkpoint_bytes": model_runner.get(
                "hybrid_prefix_checkpoint_bytes", 0
            ),
            "capture_ms": model_runner.get(
                "hybrid_prefix_capture_ms", 0.0
            ),
            "restore_ms": model_runner.get(
                "hybrid_prefix_restore_ms", 0.0
            ),
            "requests": request_rows if args.include_requests else None,
        }
    finally:
        engine.exit()
        del engine
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--workload",
        choices=("singleton", "pair", "triple", "hot", "zipf"),
        required=True,
    )
    parser.add_argument("--thresholds", type=int, nargs="+", default=(2, 3))
    parser.add_argument("--num-prefixes", type=int, default=8)
    parser.add_argument("--hot-frequency", type=int, default=8)
    parser.add_argument("--zipf-requests", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shared-prefix-length", type=int, default=496)
    parser.add_argument("--unique-suffix-length", type=int, default=64)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--prefix-match-unit", type=int, default=16)
    parser.add_argument("--checkpoint-memory-mib", type=int, default=128)
    parser.add_argument(
        "--checkpoint-dtype",
        choices=("fp32", "bf16", "int8"),
        default="int8",
    )
    parser.add_argument("--max-num-kvcache-blocks", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--include-requests", action="store_true")
    parser.add_argument(
        "--gdn-decode-backend",
        choices=("torch", "cuda", "auto"),
        default="torch",
    )
    args = parser.parse_args()
    if args.num_prefixes <= 0:
        parser.error("num prefixes must be positive")
    if args.warmup_requests < 0:
        parser.error("warmup requests must be non-negative")
    if any(threshold < 2 for threshold in args.thresholds):
        parser.error("demand-driven thresholds must be at least 2")
    if 256 % args.prefix_match_unit:
        parser.error("prefix match unit must divide physical block size 256")
    if args.shared_prefix_length % args.prefix_match_unit:
        parser.error("shared prefix must align to prefix match unit")
    prompt_length = (
        args.shared_prefix_length + args.unique_suffix_length
    )
    prompt_tail = (
        (prompt_length - 1) // args.prefix_match_unit
    ) * args.prefix_match_unit
    if prompt_tail <= args.shared_prefix_length:
        parser.error(
            "workload does not exercise demand promotion: prompt-tail "
            "checkpoint is already at or before the shared boundary; "
            "increase unique suffix length"
        )

    trace = build_trace(
        args.workload,
        num_prefixes=args.num_prefixes,
        hot_frequency=args.hot_frequency,
        zipf_requests=args.zipf_requests,
        seed=args.seed,
    )
    results = [
        run_threshold(args, threshold, trace)
        for threshold in args.thresholds
    ]
    baseline_digest = results[0]["token_digest"]
    for result in results:
        result["tokens_match_first_threshold"] = (
            result["token_digest"] == baseline_digest
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
