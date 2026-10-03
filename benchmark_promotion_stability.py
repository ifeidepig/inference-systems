"""Repeated real-model threshold benchmark with median/IQR aggregation."""

import argparse
import json
import statistics
from collections import defaultdict
from types import SimpleNamespace

from benchmark_promotion_threshold import build_trace, run_threshold


def quantiles(values):
    ordered = sorted(float(value) for value in values)
    return {
        "median": statistics.median(ordered),
        "q1": ordered[len(ordered) // 4],
        "q3": ordered[(3 * len(ordered)) // 4],
    }


def percentile(values, fraction):
    ordered = sorted(float(value) for value in values)
    return ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=("hot",),
        choices=("pair", "triple", "hot", "zipf"),
    )
    parser.add_argument(
        "--prefix-lengths",
        nargs="+",
        type=int,
        default=(256, 496, 1024, 2048),
    )
    parser.add_argument("--thresholds", nargs="+", type=int, default=(2, 3))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--num-prefixes", type=int, default=4)
    parser.add_argument("--hot-frequency", type=int, default=8)
    parser.add_argument("--zipf-requests", type=int, default=32)
    parser.add_argument("--unique-suffix-length", type=int, default=64)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--prefix-match-unit", type=int, default=16)
    parser.add_argument("--checkpoint-memory-mib", type=int, default=128)
    parser.add_argument(
        "--checkpoint-dtype",
        choices=("fp32", "bf16", "int8"),
        default="int8",
    )
    parser.add_argument("--max-num-kvcache-blocks", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument(
        "--gdn-decode-backend",
        choices=("torch", "cuda", "auto"),
        default="torch",
    )
    args = parser.parse_args()
    if args.seeds < 5:
        parser.error("stable benchmark requires at least 5 seeds")

    scalar_metrics = (
        "ttft_median_ms",
        "ttft_p95_ms",
        "promotions_published",
        "useful_promotions",
        "useful_promotion_ratio",
        "promotions_not_yet_reused",
        "unused_promotion_evictions",
        "alignment_lost_tokens",
        "shared_junction_saved_replay_tokens",
        "checkpoint_peak_entries",
        "shared_junction_peak_entries",
        "checkpoint_evictions",
        "capture_ms",
        "restore_ms",
    )
    summaries = []
    raw_runs = []
    for workload in args.workloads:
        for prefix_length in args.prefix_lengths:
            grouped = defaultdict(list)
            tokens_exact = True
            for seed in range(args.seeds):
                trace = build_trace(
                    workload,
                    num_prefixes=args.num_prefixes,
                    hot_frequency=args.hot_frequency,
                    zipf_requests=args.zipf_requests,
                    seed=seed,
                )
                threshold_order = (
                    list(args.thresholds)
                    if seed % 2 == 0
                    else list(reversed(args.thresholds))
                )
                seed_results = []
                for threshold in threshold_order:
                    run_args = SimpleNamespace(
                        **vars(args),
                        workload=workload,
                        shared_prefix_length=prefix_length,
                        seed=seed,
                        include_requests=True,
                    )
                    result = run_threshold(run_args, threshold, trace)
                    result["seed"] = seed
                    raw_runs.append(result)
                    grouped[threshold].append(result)
                    seed_results.append(result)
                tokens_exact &= len(
                    {result["token_digest"] for result in seed_results}
                ) == 1

            for threshold in args.thresholds:
                runs = grouped[threshold]
                pooled_ttft = [
                    request["ttft_ms"]
                    for run in runs
                    for request in run["requests"]
                ]
                occurrence_values = defaultdict(list)
                for run in runs:
                    for request in run["requests"]:
                        occurrence_values[request["occurrence"]].append(
                            request["ttft_ms"]
                        )
                summaries.append(
                    {
                        "workload": workload,
                        "prefix_length": prefix_length,
                        "threshold": threshold,
                        "seeds": args.seeds,
                        "tokens_exact_across_thresholds_per_seed": tokens_exact,
                        "pooled_ttft_p50_ms": percentile(pooled_ttft, 0.50),
                        "pooled_ttft_p95_ms": percentile(pooled_ttft, 0.95),
                        "metrics": {
                            metric: quantiles([run[metric] for run in runs])
                            for metric in scalar_metrics
                        },
                        "occurrence_ttft_ms": {
                            str(occurrence): quantiles(values)
                            for occurrence, values in sorted(
                                occurrence_values.items()
                            )
                        },
                    }
                )
    print(json.dumps({"summary": summaries, "runs": raw_runs}, indent=2))


if __name__ == "__main__":
    main()
