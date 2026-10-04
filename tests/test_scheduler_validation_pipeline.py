import json
from pathlib import Path
from types import SimpleNamespace

from aggregate_scheduler_results import METRICS, aggregate_results
from benchmark_scheduler_matrix import ABLATIONS, build_command


def _matrix_args(tmp_path):
    return SimpleNamespace(
        model=tmp_path,
        max_num_batched_tokens=256,
        gpu_memory_utilization=0.9,
        candidate_window=8,
        aging_tokens_per_ms=0.5,
        max_wait_ms=200.0,
        min_saved_tokens=16,
        preemption_penalty=128.0,
        checkpoint_memory_mib=64,
        checkpoint_dtype="bf16",
        gdn_decode_backend="torch",
    )


def test_matrix_command_encodes_ablation_and_pressure(tmp_path):
    command = build_command(
        _matrix_args(tmp_path),
        ABLATIONS["topw_kv_only"],
        "kv_pressure_high",
        "high",
        repeat_index=2,
        seed=7,
        output=tmp_path / "raw.json",
    )
    joined = " ".join(command)

    assert "--workload-profile kv_pressure" in joined
    assert "--max-num-kvcache-blocks 5" in joined
    assert "--hybrid-scheduler-score-source kv_only" in joined
    assert "--disable-hybrid-scheduler-aging" in command
    assert "--disable-hybrid-scheduler-hysteresis" in command
    assert "--disable-hybrid-scheduler-sticky-recovery" in command


def test_victim_choice_command_admits_multiple_prefills(tmp_path):
    command = build_command(
        _matrix_args(tmp_path),
        ABLATIONS["preemption_only"],
        "victim_choice_high",
        "high",
        repeat_index=0,
        seed=0,
        output=tmp_path / "victim.json",
    )
    joined = " ".join(command)

    assert "--workload-profile kv_pressure_victim_choice" in joined
    assert "--max-num-batched-tokens 1024" in joined


def _set_nested(target, path, value):
    current = target
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def _fake_result(label: str, digest: str, value: float):
    result = {
        "config": {
            "workload_name": "shared_prefix",
            "run_label": label,
            "repeat_index": 0,
            "seed": 3,
        },
        "output_digest": digest,
    }
    for path in METRICS.values():
        _set_nested(result, path, value)
    return result


def test_aggregator_writes_summary_and_checks_paired_digest(tmp_path):
    raw = tmp_path / "raw"
    summary = tmp_path / "summary"
    raw.mkdir()
    (raw / "baseline.json").write_text(
        json.dumps(_fake_result("baseline", "same", 1.0)),
        encoding="utf-8",
    )
    (raw / "full.json").write_text(
        json.dumps(_fake_result("full", "same", 2.0)),
        encoding="utf-8",
    )

    report = aggregate_results(raw, summary)

    assert report["correctness"] == {"comparisons": 1, "mismatches": []}
    assert (summary / "summary.json").is_file()
    assert (summary / "summary.csv").is_file()
    assert (summary / "summary.md").is_file()
    groups = {
        (item["workload"], item["configuration"]): item
        for item in report["groups"]
    }
    assert groups[("shared_prefix", "full")]["metrics"]["ttft_p50_ms"][
        "median"
    ] == 2.0
    comparison = report["comparisons"][0]
    assert comparison["configuration"] == "full"
    assert comparison["deltas_vs_baseline"]["ttft_p50_ms"]["percent"] == 100.0
