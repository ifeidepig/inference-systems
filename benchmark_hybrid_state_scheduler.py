"""Deterministic control-plane benchmark for Hybrid State-Aware scheduling.

This benchmark deliberately excludes GPU/model time.  It reuses the exact
Scheduler admission and preemption selectors with synthetic prefix snapshots,
so policy overhead, ordering, fairness and recompute cost can be studied before
running the end-to-end Qwen3.5 benchmark described in the accompanying doc.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from statistics import median
from types import SimpleNamespace

from nanovllm import SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


BASE_NS = 1_000_000_000


@dataclass(frozen=True, slots=True)
class RequestSpec:
    name: str
    prompt_tokens: int
    reusable_tokens: int
    arrival_ms: float


class SnapshotScheduler(Scheduler):
    """Scheduler whose cache snapshot is supplied by a deterministic trace."""

    def __init__(self, config):
        super().__init__(config)
        self.snapshots: dict[int, tuple[int, int, bool]] = {}

    def set_snapshot(
        self,
        sequence: Sequence,
        *,
        reusable_tokens: int,
        kv_candidate_tokens: int | None = None,
        feasible: bool = True,
    ) -> None:
        self.snapshots[sequence.seq_id] = (
            reusable_tokens,
            reusable_tokens if kv_candidate_tokens is None else kv_candidate_tokens,
            feasible,
        )

    def _probe_prefix(self, seq, *, durable_only=False):
        del durable_only
        return self.snapshots.get(seq.seq_id, (0, 0, True))


def scheduler_config(
    admission_policy: str,
    preemption_policy: str = "lifo",
    *,
    candidate_window: int,
    aging_tokens_per_ms: float,
    max_wait_ms: float,
    min_saved_tokens: int,
):
    return SimpleNamespace(
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        eos=0,
        kvcache_block_size=256,
        prefix_match_unit=16,
        num_kvcache_blocks=128,
        scheduling_policy="prefill_first",
        waiting_admission_policy=admission_policy,
        preemption_policy=preemption_policy,
        hybrid_scheduler_candidate_window=candidate_window,
        hybrid_scheduler_aging_tokens_per_ms=aging_tokens_per_ms,
        hybrid_scheduler_max_wait_ms=max_wait_ms,
        hybrid_scheduler_min_saved_tokens=min_saved_tokens,
        hybrid_scheduler_preemption_penalty=128.0,
        enable_prefix_cache=True,
        enable_chunked_prefill=True,
        enable_hybrid_prefix_cache=False,
        num_speculative_tokens=0,
        request_metrics_history_size=0,
    )


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(math.ceil(probability * len(ordered)) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]


def make_sequence(spec: RequestSpec) -> Sequence:
    sequence = Sequence(
        [sequence_token(spec.name)] * spec.prompt_tokens,
        SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True),
        arrival_time_ns=BASE_NS + int(spec.arrival_ms * 1e6),
    )
    sequence.benchmark_name = spec.name
    return sequence


def sequence_token(name: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(name)) % 32000


def workloads() -> dict[str, list[RequestSpec]]:
    return {
        "shared_prefix": [
            RequestSpec("cold-0", 560, 0, 0),
            RequestSpec("shared-0", 560, 496, 0),
            RequestSpec("cold-1", 560, 0, 0),
            RequestSpec("shared-1", 560, 496, 0),
            RequestSpec("shared-2", 560, 496, 0),
        ],
        "multi_session": [
            RequestSpec("cold-a", 816, 0, 0),
            RequestSpec("session-a0", 816, 752, 0),
            RequestSpec("session-b0", 560, 496, 0),
            RequestSpec("cold-b", 560, 0, 0),
            RequestSpec("session-a1", 816, 752, 0),
            RequestSpec("session-b1", 560, 496, 0),
        ],
        "unique_prompt": [
            RequestSpec(f"unique-{index}", 560, 0, 0)
            for index in range(8)
        ],
        "kv_pressure": [
            RequestSpec("large-cold", 2048, 0, 0),
            RequestSpec("hot-0", 2048, 1792, 0),
            RequestSpec("medium-cold", 1024, 0, 0),
            RequestSpec("hot-1", 2048, 1792, 0),
        ],
        "multi_turn": [
            RequestSpec("cold-0", 1024, 0, 0),
            RequestSpec("turn-1", 320, 256, 0),
            RequestSpec("cold-1", 1024, 0, 0),
            RequestSpec("turn-2", 576, 512, 0),
            RequestSpec("cold-2", 1024, 0, 0),
            RequestSpec("turn-3", 832, 768, 0),
        ],
    }


def run_admission_trace(
    specs: list[RequestSpec],
    admission_policy: str,
    args,
) -> dict:
    Sequence.block_size = 256
    scheduler = SnapshotScheduler(
        scheduler_config(
            admission_policy,
            candidate_window=args.candidate_window,
            aging_tokens_per_ms=args.aging_tokens_per_ms,
            max_wait_ms=args.max_wait_ms,
            min_saved_tokens=args.min_saved_tokens,
        )
    )
    pending = sorted(specs, key=lambda item: item.arrival_ms)
    current_ms = min((item.arrival_ms for item in pending), default=0.0)
    records = []
    while pending or scheduler.waiting:
        arrived = [item for item in pending if item.arrival_ms <= current_ms]
        for spec in arrived:
            sequence = make_sequence(spec)
            scheduler.set_snapshot(
                sequence,
                reusable_tokens=spec.reusable_tokens,
            )
            scheduler.add(sequence)
            pending.remove(spec)
        if not scheduler.waiting:
            current_ms = pending[0].arrival_ms
            continue
        now_ns = BASE_NS + int(current_ms * 1e6)
        selected = scheduler._select_waiting_candidate(now_ns)
        assert scheduler.waiting.popleft() is selected
        reusable, _, _ = scheduler._probe_prefix(selected)
        uncached = selected.num_prompt_tokens - reusable
        queue_ms = current_ms - (
            (selected.arrival_time_ns - BASE_NS) / 1e6
        )
        service_ms = uncached / args.prefill_tokens_per_ms + args.decode_ms
        current_ms += service_ms
        selected.mark_scheduled(BASE_NS + int(current_ms * 1e6))
        records.append(
            {
                "request": selected.benchmark_name,
                "queue_ms": queue_ms,
                "service_ms": service_ms,
                "uncached_tokens": uncached,
                "reusable_tokens": reusable,
                "bypasses": selected.scheduler_bypass_count,
            }
        )
    queue_ms = [record["queue_ms"] for record in records]
    total_uncached = sum(record["uncached_tokens"] for record in records)
    metrics = scheduler.get_metrics()
    return {
        "policy": admission_policy,
        "order": [record["request"] for record in records],
        "requests": len(records),
        "makespan_ms": current_ms,
        "request_throughput_per_s": (
            len(records) / current_ms * 1000 if current_ms else 0.0
        ),
        "prefill_tokens_executed": total_uncached,
        "queue_p50_ms": median(queue_ms) if queue_ms else 0.0,
        "queue_p95_ms": percentile(queue_ms, 0.95),
        "max_queue_ms": max(queue_ms, default=0.0),
        "max_bypasses": max(
            (record["bypasses"] for record in records), default=0
        ),
        "reorders": metrics["hybrid_scheduler_reorders"],
        "aging_overrides": metrics["hybrid_scheduler_aging_overrides"],
        "probe_ms": metrics["hybrid_scheduler_probe_ms"],
    }


def run_preemption_scenarios(policy: str, args) -> dict:
    scenarios = [
        # (current recompute/reclaim, older, LIFO/newest)
        ((64, 2), (640, 3), (1024, 4)),
        ((256, 1), (512, 4), (768, 2)),
        ((128, 4), (256, 2), (896, 3)),
        ((384, 3), (448, 4), (1536, 5)),
    ]
    total_recompute = 0
    total_reclaimed = 0
    choices = []
    for scenario_index, scenario in enumerate(scenarios):
        scheduler = SnapshotScheduler(
            scheduler_config(
                "fcfs",
                policy,
                candidate_window=args.candidate_window,
                aging_tokens_per_ms=args.aging_tokens_per_ms,
                max_wait_ms=args.max_wait_ms,
                min_saved_tokens=args.min_saved_tokens,
            )
        )
        sequences = []
        reclaimable = {}
        for candidate_index, (recompute, blocks) in enumerate(scenario):
            total_tokens = recompute + 2048
            sequence = make_sequence(
                RequestSpec(
                    f"s{scenario_index}-c{candidate_index}",
                    total_tokens,
                    total_tokens - recompute,
                    0,
                )
            )
            sequence.num_cached_tokens = total_tokens
            sequence.status = SequenceStatus.RUNNING
            scheduler.set_snapshot(
                sequence,
                reusable_tokens=total_tokens - recompute,
            )
            reclaimable[sequence.seq_id] = blocks
            sequences.append(sequence)
        scheduler.block_manager.reclaimable_blocks = (
            lambda sequence: reclaimable[sequence.seq_id]
        )
        current, older, newest = sequences
        scheduler.running.extend([older, newest])
        victim = scheduler._select_preemption_victim(current)
        total_recompute += victim.recompute_tokens
        total_reclaimed += victim.reclaimable_blocks
        choices.append(
            {
                "scenario": scenario_index,
                "victim": victim.sequence.benchmark_name,
                "recompute_tokens": victim.recompute_tokens,
                "reclaimable_blocks": victim.reclaimable_blocks,
                "cost_per_block": victim.cost_per_block,
            }
        )
    return {
        "policy": policy,
        "scenarios": choices,
        "total_recompute_tokens": total_recompute,
        "total_reclaimable_blocks": total_reclaimed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-window", type=int, default=8)
    parser.add_argument("--aging-tokens-per-ms", type=float, default=0.5)
    parser.add_argument("--max-wait-ms", type=float, default=200.0)
    parser.add_argument("--min-saved-tokens", type=int, default=16)
    parser.add_argument("--prefill-tokens-per-ms", type=float, default=4.0)
    parser.add_argument("--decode-ms", type=float, default=8.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.candidate_window <= 0 or args.prefill_tokens_per_ms <= 0:
        parser.error("candidate window and prefill rate must be positive")

    results = {
        "kind": "control_plane_policy_benchmark",
        "methodology": {
            "gpu_time_included": False,
            "candidate_window": args.candidate_window,
            "aging_tokens_per_ms": args.aging_tokens_per_ms,
            "max_wait_ms": args.max_wait_ms,
            "min_saved_tokens": args.min_saved_tokens,
            "prefill_tokens_per_ms": args.prefill_tokens_per_ms,
            "decode_ms": args.decode_ms,
        },
        "admission": {},
        "preemption": {},
    }
    for name, specs in workloads().items():
        baseline = run_admission_trace(specs, "fcfs", args)
        aware = run_admission_trace(specs, "hybrid_state_aware", args)
        aware["delta_vs_fcfs"] = {
            "queue_p95_ms": aware["queue_p95_ms"] - baseline["queue_p95_ms"],
            "max_queue_ms": aware["max_queue_ms"] - baseline["max_queue_ms"],
            "makespan_ms": aware["makespan_ms"] - baseline["makespan_ms"],
        }
        results["admission"][name] = {
            "fcfs": baseline,
            "hybrid_state_aware": aware,
        }
    for policy in ("lifo", "recompute_aware"):
        results["preemption"][policy] = run_preemption_scenarios(policy, args)

    payload = json.dumps(results, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
