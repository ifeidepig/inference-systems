"""Long-horizon control-plane study using the production checkpoint manager."""

import argparse
import json
import statistics
from collections import defaultdict

from benchmark_promotion_threshold import build_trace
from nanovllm.engine.hybrid_prefix_cache import (
    GDNCheckpointManager,
    PendingHybridPrefixCapture,
    PrefixKVCandidate,
)


def stable_summary(values: list[float | int]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    return {
        "median": statistics.median(ordered),
        "q1": ordered[len(ordered) // 4],
        "q3": ordered[(3 * len(ordered)) // 4],
        "min": ordered[0],
        "max": ordered[-1],
    }


def run_trace(
    trace: list[tuple[int, int]],
    *,
    threshold: int,
    prefix_length: int,
    suffix_length: int,
    capacity: int,
) -> dict:
    cache = GDNCheckpointManager(capacity, eviction_policy="cost_aware")
    seen = defaultdict(int)
    demand_observations = defaultdict(int)
    promotions_published = 0
    alignment_replay_tokens = 0
    next_checkpoint_slot = 0

    def publish(pending: PendingHybridPrefixCapture) -> None:
        nonlocal next_checkpoint_slot
        cache.reserve_capture(pending)
        cache.publish(pending, checkpoint_slot=next_checkpoint_slot)
        next_checkpoint_slot += 1

    for request_index, (prefix_id, _) in enumerate(trace):
        prefix_hash = prefix_id + 1
        candidate = PrefixKVCandidate(
            num_cached_blocks=0,
            boundary_tokens=prefix_length,
            prefix_hash=prefix_hash,
            tail_block_id=prefix_id + 1,
            tail_valid_tokens=prefix_length,
        )
        if seen[prefix_id] > 0:
            hit = cache.find_hit([candidate])
            if hit is None:
                alignment_replay_tokens += prefix_length
                demand_observations[prefix_id] += 1
                total_sightings = 1 + demand_observations[prefix_id]
                if total_sightings >= threshold:
                    publish(
                        PendingHybridPrefixCapture(
                            prefix_hash=prefix_hash,
                            boundary_tokens=prefix_length,
                            tail_block_id=prefix_id + 1,
                            state_slot=0,
                            reason="shared_junction",
                            replay_saved_tokens=prefix_length,
                            demand_count=demand_observations[prefix_id],
                        )
                    )
                    promotions_published += 1
                    demand_observations[prefix_id] = 0

        seen[prefix_id] += 1
        prompt_tail = prefix_length + suffix_length - 16
        publish(
            PendingHybridPrefixCapture(
                prefix_hash=1_000_000 + request_index,
                boundary_tokens=prompt_tail,
                tail_block_id=1_000_000 + request_index,
                state_slot=0,
                reason="prompt_tail",
                replay_saved_tokens=prompt_tail,
            )
        )

    metrics = cache.get_metrics()
    useful = metrics["hybrid_prefix_useful_promotion_count"]
    shared_entries = [
        entry
        for entry in cache.entries.values()
        if entry.reason == "shared_junction"
    ]
    unused_resident = sum(entry.hit_count == 0 for entry in shared_entries)
    return {
        "request_count": len(trace),
        "promotions_published": promotions_published,
        "useful_promotions": useful,
        "useful_promotion_ratio": (
            useful / promotions_published if promotions_published else 0.0
        ),
        "unused_promotions_resident": unused_resident,
        "unused_promotion_evictions": metrics[
            "hybrid_prefix_unused_promotion_eviction_count"
        ],
        "alignment_replay_tokens": alignment_replay_tokens,
        "saved_replay_tokens": metrics[
            "hybrid_prefix_shared_junction_saved_replay_tokens"
        ],
        "peak_entries": metrics["hybrid_prefix_peak_entries"],
        "peak_shared_junction_entries": metrics[
            "hybrid_prefix_peak_shared_junction_entries"
        ],
        "checkpoint_evictions": metrics["hybrid_prefix_eviction_count"],
        "shared_junction_evictions": metrics[
            "hybrid_prefix_shared_junction_eviction_count"
        ],
        "promotion_churn_per_1k": (
            (promotions_published + metrics[
                "hybrid_prefix_shared_junction_eviction_count"
            ])
            * 1000
            / len(trace)
        ),
        "net_replay_tokens_saved": (
            metrics["hybrid_prefix_shared_junction_saved_replay_tokens"]
            - alignment_replay_tokens
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=("pair", "triple", "hot", "zipf"),
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
    parser.add_argument("--num-prefixes", type=int, default=128)
    parser.add_argument("--hot-frequency", type=int, default=64)
    parser.add_argument("--zipf-requests", type=int, default=1024)
    parser.add_argument("--suffix-length", type=int, default=64)
    parser.add_argument("--checkpoint-capacity", type=int, default=24)
    args = parser.parse_args()
    if args.seeds < 5:
        parser.error("stable study requires at least 5 seeds")
    if any(value < 2 for value in args.thresholds):
        parser.error("thresholds must be at least 2")

    rows = []
    for workload in args.workloads:
        for prefix_length in args.prefix_lengths:
            for threshold in args.thresholds:
                runs = []
                for seed in range(args.seeds):
                    trace = build_trace(
                        workload,
                        num_prefixes=args.num_prefixes,
                        hot_frequency=args.hot_frequency,
                        zipf_requests=args.zipf_requests,
                        seed=seed,
                    )
                    runs.append(
                        run_trace(
                            trace,
                            threshold=threshold,
                            prefix_length=prefix_length,
                            suffix_length=args.suffix_length,
                            capacity=args.checkpoint_capacity,
                        )
                    )
                metric_names = [
                    key for key in runs[0] if key != "request_count"
                ]
                rows.append(
                    {
                        "workload": workload,
                        "prefix_length": prefix_length,
                        "threshold": threshold,
                        "seeds": args.seeds,
                        "request_count_per_seed": runs[0]["request_count"],
                        "metrics": {
                            name: stable_summary([run[name] for run in runs])
                            for name in metric_names
                        },
                    }
                )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
