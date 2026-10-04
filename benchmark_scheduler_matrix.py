"""Run paired baseline/ablation/full scheduler experiments in fresh engines."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter

from aggregate_scheduler_results import aggregate_results


@dataclass(frozen=True, slots=True)
class Ablation:
    name: str
    admission: str
    preemption: str
    score_source: str = "joint"
    aging: bool = True
    hysteresis: bool = True
    sticky: bool = True


ABLATIONS = {
    item.name: item
    for item in (
        Ablation("baseline", "fcfs", "lifo"),
        Ablation(
            "topw_kv_only", "hybrid_state_aware", "lifo", "kv_only",
            False, False, False,
        ),
        Ablation(
            "topw_joint", "hybrid_state_aware", "lifo", "joint",
            False, False, False,
        ),
        Ablation(
            "joint_aging", "hybrid_state_aware", "lifo", "joint",
            True, False, False,
        ),
        Ablation(
            "guarded_admission", "hybrid_state_aware", "lifo", "joint",
            True, True, True,
        ),
        Ablation("preemption_only", "fcfs", "recompute_aware"),
        Ablation("full", "hybrid_state_aware", "recompute_aware"),
    )
}


PRESSURE_BLOCKS = {"low": 8, "medium": 6, "high": 5}


def workload_variants(workload: str, pressure_levels: list[str]):
    if workload in ("kv_pressure", "victim_choice"):
        return [(f"{workload}_{level}", level) for level in pressure_levels]
    return [(workload, None)]


def build_command(
    args,
    ablation: Ablation,
    workload: str,
    pressure: str | None,
    repeat_index: int,
    seed: int,
    output: Path,
) -> list[str]:
    if pressure is None:
        profile = workload
    elif workload.startswith("victim_choice"):
        profile = "kv_pressure_victim_choice"
    else:
        profile = "kv_pressure"
    max_num_seqs = 4 if pressure is not None else 2
    max_num_batched_tokens = (
        max(args.max_num_batched_tokens, 1024)
        if workload.startswith("victim_choice")
        else args.max_num_batched_tokens
    )
    max_model_len = 1024
    max_blocks = PRESSURE_BLOCKS[pressure] if pressure is not None else 16
    shared_prefix_length = 240 if pressure is not None else 496
    prefix_seeds = 0 if workload == "unique_prompt" else 2
    command = [
        sys.executable,
        str(Path(__file__).with_name("benchmark_online.py")),
        "--model", str(args.model),
        "--output", str(output),
        "--workload-profile", profile,
        "--workload-name", workload,
        "--run-label", ablation.name,
        "--repeat-index", str(repeat_index),
        "--seed", str(seed),
        "--temperature", "0",
        "--max-model-len", str(max_model_len),
        "--max-num-batched-tokens", str(max_num_batched_tokens),
        "--max-num-seqs", str(max_num_seqs),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--scheduling-policy", "decode_first",
        "--waiting-admission-policy", ablation.admission,
        "--preemption-policy", ablation.preemption,
        "--hybrid-scheduler-score-source", ablation.score_source,
        "--hybrid-scheduler-candidate-window", str(args.candidate_window),
        "--hybrid-scheduler-aging-tokens-per-ms", str(args.aging_tokens_per_ms),
        "--hybrid-scheduler-max-wait-ms", str(args.max_wait_ms),
        "--hybrid-scheduler-min-saved-tokens", str(args.min_saved_tokens),
        "--hybrid-scheduler-preemption-penalty", str(args.preemption_penalty),
        "--max-num-kvcache-blocks", str(max_blocks),
        "--prefix-match-unit", "16",
        "--enable-hybrid-prefix-cache",
        "--hybrid-prefix-checkpoint-memory-mib", str(args.checkpoint_memory_mib),
        "--hybrid-prefix-checkpoint-dtype", args.checkpoint_dtype,
        "--hybrid-prefix-checkpoint-interval-tokens", "256",
        "--hybrid-prefix-retention-policy", "adaptive",
        "--hybrid-prefix-eviction-policy", "cost_aware",
        "--prefix-seed-requests", str(prefix_seeds),
        "--shared-prefix-length", str(shared_prefix_length),
        "--gdn-decode-backend", args.gdn_decode_backend,
        "--enable-scheduler-profiling",
        "--scheduler-decision-history-size", "8192",
    ]
    if not ablation.aging:
        command.append("--disable-hybrid-scheduler-aging")
    if not ablation.hysteresis:
        command.append("--disable-hybrid-scheduler-hysteresis")
    if not ablation.sticky:
        command.append("--disable-hybrid-scheduler-sticky-recovery")
    return command


def parse_selection(selected: list[str], available: dict[str, object]) -> list[str]:
    if selected == ["all"]:
        return list(available)
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"unknown selection: {sorted(unknown)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("benchmark_results/scheduler_validation"),
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--configs", nargs="+", default=["all"])
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=[
            "shared_prefix", "multi_session", "unique_prompt",
            "kv_pressure", "multi_turn",
        ],
    )
    parser.add_argument(
        "--pressure-levels", nargs="+", default=["low", "medium", "high"]
    )
    parser.add_argument("--candidate-window", type=int, default=8)
    parser.add_argument("--aging-tokens-per-ms", type=float, default=0.5)
    parser.add_argument("--max-wait-ms", type=float, default=200.0)
    parser.add_argument("--min-saved-tokens", type=int, default=16)
    parser.add_argument("--preemption-penalty", type=float, default=128.0)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--checkpoint-memory-mib", type=int, default=64)
    parser.add_argument(
        "--checkpoint-dtype", choices=("fp32", "bf16"), default="bf16"
    )
    parser.add_argument(
        "--gdn-decode-backend", choices=("torch", "cuda", "auto"),
        default="torch",
    )
    parser.add_argument("--order-seed", type=int, default=20261004)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip raw JSON files that already completed successfully.",
    )
    args = parser.parse_args()
    if args.runs <= 0 or not args.seeds:
        parser.error("runs and seeds must be non-empty and positive")

    configs = parse_selection(args.configs, ABLATIONS)
    workloads = parse_selection(
        args.workloads,
        {name: None for name in (
            "shared_prefix", "multi_session", "unique_prompt",
            "kv_pressure", "multi_turn", "victim_choice",
        )},
    )
    pressure_levels = parse_selection(args.pressure_levels, PRESSURE_BLOCKS)
    raw_dir = args.output_root / "raw"
    summary_dir = args.output_root / "summary"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.jsonl"
    rng = random.Random(args.order_seed)

    failures = []
    for workload in workloads:
        for workload_name, pressure in workload_variants(
            workload, pressure_levels
        ):
            for repeat_index in range(args.runs):
                seed = args.seeds[repeat_index % len(args.seeds)]
                run_order = list(configs)
                rng.shuffle(run_order)
                for order_index, config_name in enumerate(run_order):
                    output = (
                        raw_dir / workload_name / config_name
                        / f"run_{repeat_index:02d}_seed_{seed}.json"
                    )
                    output.parent.mkdir(parents=True, exist_ok=True)
                    if args.resume and output.is_file():
                        print(
                            f"[{workload_name}] run={repeat_index} "
                            f"config={config_name} already complete",
                            flush=True,
                        )
                        continue
                    command = build_command(
                        args,
                        ABLATIONS[config_name],
                        workload_name,
                        pressure,
                        repeat_index,
                        seed,
                        output,
                    )
                    manifest = {
                        "workload": workload_name,
                        "configuration": config_name,
                        "repeat_index": repeat_index,
                        "seed": seed,
                        "order_index": order_index,
                        "command": command,
                    }
                    if args.dry_run:
                        print(" ".join(command))
                        continue
                    started = perf_counter()
                    log_path = output.with_suffix(".log")
                    print(
                        f"[{workload_name}] run={repeat_index} "
                        f"config={config_name}",
                        flush=True,
                    )
                    with log_path.open("w", encoding="utf-8") as log_handle:
                        completed = subprocess.run(
                            command,
                            text=True,
                            stdout=log_handle,
                            stderr=subprocess.STDOUT,
                        )
                    manifest["wall_s"] = perf_counter() - started
                    manifest["returncode"] = completed.returncode
                    manifest["log"] = str(log_path)
                    with manifest_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(manifest) + "\n")
                    if completed.returncode:
                        failures.append(manifest)
                        if not args.keep_going:
                            raise SystemExit(completed.returncode)

    if args.dry_run:
        return
    report = aggregate_results(raw_dir, summary_dir)
    if failures:
        (summary_dir / "failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8"
        )
    print(json.dumps(report["correctness"], indent=2))
    if report["correctness"]["mismatches"]:
        raise SystemExit("request-level greedy output digest mismatch")


if __name__ == "__main__":
    main()
