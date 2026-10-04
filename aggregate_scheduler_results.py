"""Aggregate raw Hybrid State-Aware Scheduler benchmark JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median, stdev


METRICS = {
    "ttft_mean_ms": ("summary", "ttft_ms", "mean"),
    "ttft_p50_ms": ("summary", "ttft_ms", "p50"),
    "ttft_p95_ms": ("summary", "ttft_ms", "p95"),
    "ttft_p99_ms": ("summary", "ttft_ms", "p99"),
    "tpot_mean_ms": ("summary", "tpot_ms", "mean"),
    "tpot_p50_ms": ("summary", "tpot_ms", "p50"),
    "tpot_p95_ms": ("summary", "tpot_ms", "p95"),
    "tpot_p99_ms": ("summary", "tpot_ms", "p99"),
    "request_latency_p95_ms": ("summary", "request_latency_ms", "p95"),
    "request_latency_p99_ms": ("summary", "request_latency_ms", "p99"),
    "makespan_ms": ("summary", "total_makespan_ms"),
    "request_throughput_rps": ("summary", "request_throughput_rps"),
    "input_throughput_tok_s": ("summary", "input_throughput_tokens_per_s"),
    "output_throughput_tok_s": ("summary", "output_throughput_tokens_per_s"),
    "peak_gpu_gib": ("summary", "peak_allocated_gib"),
    "scheduler_cpu_ms": ("summary", "scheduler_cpu_ms"),
    "scheduler_cpu_fraction": ("summary", "scheduler_cpu_fraction"),
    "admission_decisions": (
        "runtime_metrics", "scheduler", "scheduler_admission_decisions"
    ),
    "reorders": ("runtime_metrics", "scheduler", "hybrid_scheduler_reorders"),
    "reorder_rate": (
        "runtime_metrics", "scheduler", "hybrid_scheduler_reorder_rate"
    ),
    "selected_kv_tokens": (
        "runtime_metrics", "scheduler",
        "hybrid_scheduler_selected_kv_candidate_tokens",
    ),
    "selected_gdn_tokens": (
        "runtime_metrics", "scheduler",
        "hybrid_scheduler_selected_gdn_checkpoint_tokens",
    ),
    "selected_joint_tokens": (
        "runtime_metrics", "scheduler",
        "hybrid_scheduler_selected_joint_recoverable_tokens",
    ),
    "estimated_remaining_prefill_tokens": (
        "runtime_metrics", "scheduler",
        "hybrid_scheduler_estimated_remaining_prefill_tokens",
    ),
    "actual_reused_tokens": (
        "runtime_metrics", "scheduler", "prefix_tokens_reused_actual"
    ),
    "actual_prefill_tokens": (
        "runtime_metrics", "scheduler", "prefill_tokens_scheduled_total"
    ),
    "prefill_observed_ms": (
        "runtime_metrics", "scheduler", "prefill_observed_ms"
    ),
    "preemptions": ("runtime_metrics", "scheduler", "preemption_count"),
    "victim_recompute_tokens": (
        "runtime_metrics", "scheduler", "scheduler_victim_recompute_tokens"
    ),
    "victim_reclaimable_blocks": (
        "runtime_metrics", "scheduler", "scheduler_victim_reclaimable_blocks"
    ),
    "victim_logical_blocks": (
        "runtime_metrics", "scheduler", "scheduler_victim_logical_blocks"
    ),
    "victim_cost_per_block": (
        "runtime_metrics", "scheduler",
        "scheduler_victim_mean_cost_per_reclaimed_block",
    ),
    "hysteresis_rejections": (
        "runtime_metrics", "scheduler",
        "hybrid_scheduler_hysteresis_rejections",
    ),
    "sticky_triggers": (
        "runtime_metrics", "scheduler",
        "hybrid_scheduler_sticky_recovery_triggers",
    ),
    "aging_promotions": (
        "runtime_metrics", "scheduler", "hybrid_scheduler_aging_overrides"
    ),
    "starvation_count": (
        "runtime_metrics", "scheduler", "hybrid_scheduler_starvation_count"
    ),
    "admission_p95_us": (
        "runtime_metrics", "scheduler",
        "hybrid_scheduler_admission_latency", "p95_us",
    ),
    "admission_mean_us": (
        "runtime_metrics", "scheduler",
        "hybrid_scheduler_admission_latency", "mean_us",
    ),
    "admission_p50_us": (
        "runtime_metrics", "scheduler",
        "hybrid_scheduler_admission_latency", "p50_us",
    ),
    "admission_p99_us": (
        "runtime_metrics", "scheduler",
        "hybrid_scheduler_admission_latency", "p99_us",
    ),
    "probe_p95_us": (
        "runtime_metrics", "scheduler",
        "hybrid_scheduler_probe_latency", "p95_us",
    ),
    "victim_selection_p95_us": (
        "runtime_metrics", "scheduler",
        "recompute_aware_selection_latency", "p95_us",
    ),
    "victim_selection_p50_us": (
        "runtime_metrics", "scheduler",
        "recompute_aware_selection_latency", "p50_us",
    ),
    "victim_selection_p99_us": (
        "runtime_metrics", "scheduler",
        "recompute_aware_selection_latency", "p99_us",
    ),
    "bookkeeping_p95_us": (
        "runtime_metrics", "scheduler",
        "scheduler_metrics_bookkeeping_latency", "p95_us",
    ),
}


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(
        max(math.ceil(probability * len(ordered)) - 1, 0),
        len(ordered) - 1,
    )
    return ordered[index]


def describe(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "median": median(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


def nested(result: dict, path: tuple[str, ...]):
    value = result
    for key in path:
        value = value[key]
    return value


def load_results(raw_dir: Path) -> list[tuple[Path, dict]]:
    loaded = []
    for path in sorted(raw_dir.rglob("*.json")):
        loaded.append((path, json.loads(path.read_text(encoding="utf-8"))))
    if not loaded:
        raise ValueError(f"no raw result JSON files found under {raw_dir}")
    return loaded


def correctness_report(results: list[tuple[Path, dict]]) -> dict:
    paired: dict[tuple, dict[str, tuple[str, Path]]] = {}
    for path, result in results:
        config = result["config"]
        key = (
            config["workload_name"],
            config["repeat_index"],
            config["seed"],
        )
        paired.setdefault(key, {})[config["run_label"]] = (
            result["output_digest"], path
        )
    mismatches = []
    checked = 0
    for key, configs in paired.items():
        baseline = configs.get("baseline")
        if baseline is None:
            continue
        for label, (digest, path) in configs.items():
            if label == "baseline":
                continue
            checked += 1
            if digest != baseline[0]:
                mismatches.append(
                    {
                        "pair": key,
                        "configuration": label,
                        "baseline_digest": baseline[0],
                        "candidate_digest": digest,
                        "baseline_path": str(baseline[1]),
                        "candidate_path": str(path),
                    }
                )
    return {"comparisons": checked, "mismatches": mismatches}


def aggregate_results(raw_dir: Path, summary_dir: Path) -> dict:
    results = load_results(raw_dir)
    grouped: dict[tuple[str, str], list[dict]] = {}
    for _, result in results:
        config = result["config"]
        key = (config["workload_name"], config["run_label"])
        grouped.setdefault(key, []).append(result)

    groups = []
    grouped_summaries = {}
    csv_rows = []
    for (workload, configuration), items in sorted(grouped.items()):
        metric_summary = {}
        for metric, path in METRICS.items():
            values = [float(nested(item, path)) for item in items]
            metric_summary[metric] = describe(values)
            csv_rows.append(
                {
                    "workload": workload,
                    "configuration": configuration,
                    "metric": metric,
                    **metric_summary[metric],
                }
            )
        groups.append(
            {
                "workload": workload,
                "configuration": configuration,
                "runs": len(items),
                "metrics": metric_summary,
            }
        )
        grouped_summaries[(workload, configuration)] = metric_summary

    comparisons = []
    for (workload, configuration), candidate in sorted(
        grouped_summaries.items()
    ):
        if configuration == "baseline":
            continue
        baseline = grouped_summaries.get((workload, "baseline"))
        if baseline is None:
            continue
        deltas = {}
        for metric in METRICS:
            baseline_value = baseline[metric]["median"]
            candidate_value = candidate[metric]["median"]
            deltas[metric] = {
                "baseline_median": baseline_value,
                "candidate_median": candidate_value,
                "absolute": candidate_value - baseline_value,
                "percent": (
                    (candidate_value / baseline_value - 1) * 100
                    if baseline_value
                    else None
                ),
            }
        comparisons.append(
            {
                "workload": workload,
                "configuration": configuration,
                "deltas_vs_baseline": deltas,
            }
        )

    report = {
        "raw_dir": str(raw_dir),
        "raw_files": len(results),
        "correctness": correctness_report(results),
        "groups": groups,
        "comparisons": comparisons,
    }
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (summary_dir / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "workload", "configuration", "metric", "count", "mean",
                "std", "median", "p95", "p99", "min", "max",
            ),
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    headline = (
        "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms", "tpot_p95_ms",
        "request_throughput_rps", "output_throughput_tok_s",
        "preemptions", "victim_recompute_tokens",
        "victim_reclaimable_blocks", "reorder_rate",
        "scheduler_cpu_fraction",
    )
    markdown = [
        "# Scheduler Benchmark Summary",
        "",
        f"Raw files: {len(results)}",
        "",
        "## Correctness",
        "",
        f"Paired digest comparisons: {report['correctness']['comparisons']}",
        f"Mismatches: {len(report['correctness']['mismatches'])}",
        "",
        "## Median across fresh-engine runs",
        "",
        "| Workload | Configuration | Metric | Median | P95 across runs |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for group in groups:
        for metric in headline:
            stats = group["metrics"][metric]
            markdown.append(
                f"| {group['workload']} | {group['configuration']} | "
                f"{metric} | {stats['median']:.6g} | {stats['p95']:.6g} |"
            )
    markdown.extend(
        [
            "",
            "## Delta versus baseline (median)",
            "",
            "| Workload | Configuration | Metric | Delta % |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for comparison in comparisons:
        for metric in headline:
            percent = comparison["deltas_vs_baseline"][metric]["percent"]
            rendered = "n/a" if percent is None else f"{percent:.3f}%"
            markdown.append(
                f"| {comparison['workload']} | "
                f"{comparison['configuration']} | {metric} | {rendered} |"
            )
    (summary_dir / "summary.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--require-digest-match", action="store_true")
    args = parser.parse_args()
    report = aggregate_results(args.raw_dir, args.summary_dir)
    print(json.dumps(report["correctness"], indent=2))
    if args.require_digest_match and report["correctness"]["mismatches"]:
        raise SystemExit("request-level greedy output digest mismatch")


if __name__ == "__main__":
    main()
