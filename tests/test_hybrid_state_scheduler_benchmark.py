from types import SimpleNamespace

from benchmark_hybrid_state_scheduler import (
    RequestSpec,
    run_admission_trace,
    run_joint_boundary_ablation,
    run_preemption_scenarios,
    workloads,
)


def _args():
    return SimpleNamespace(
        candidate_window=8,
        aging_tokens_per_ms=0.5,
        max_wait_ms=200.0,
        min_saved_tokens=16,
        prefill_tokens_per_ms=4.0,
        decode_ms=8.0,
    )


def test_shared_prefix_trace_reorders_without_changing_total_work():
    args = _args()
    trace = workloads()["shared_prefix"]

    fcfs = run_admission_trace(trace, "fcfs", args)
    aware = run_admission_trace(trace, "hybrid_state_aware", args)

    assert fcfs["prefill_tokens_executed"] == aware["prefill_tokens_executed"]
    assert fcfs["order"] != aware["order"]
    assert aware["order"][0].startswith("shared")
    assert aware["reorders"] > 0


def test_unique_trace_preserves_fcfs_order():
    args = _args()
    trace = workloads()["unique_prompt"]

    fcfs = run_admission_trace(trace, "fcfs", args)
    aware = run_admission_trace(trace, "hybrid_state_aware", args)

    assert aware["order"] == fcfs["order"]
    assert aware["reorders"] == 0


def test_aging_caps_repeated_bypass():
    args = _args()
    args.max_wait_ms = 20.0
    trace = [
        RequestSpec("cold", 1024, 0, 0),
        *[
            RequestSpec(f"hot-{index}", 1024, 1008, index * 2)
            for index in range(12)
        ],
    ]

    aware = run_admission_trace(trace, "hybrid_state_aware", args)

    assert "cold" in aware["order"]
    assert aware["aging_overrides"] > 0
    assert aware["max_bypasses"] <= len(trace) - 1


def test_recompute_aware_preemption_reduces_estimated_replay():
    args = _args()

    lifo = run_preemption_scenarios("lifo", args)
    aware = run_preemption_scenarios("recompute_aware", args)

    assert aware["total_recompute_tokens"] < lifo["total_recompute_tokens"]
    assert aware["total_reclaimable_blocks"] > 0


def test_joint_boundary_scoring_avoids_kv_only_overestimate():
    args = _args()

    kv_only = run_joint_boundary_ablation("kv_only", args)
    joint = run_joint_boundary_ablation("joint", args)

    assert kv_only["kv_only_overestimate_tokens"] == 904
    assert kv_only["selected"] == "kv5000-gdn4096"
    assert joint["selected"] == "kv4500-gdn4500"
