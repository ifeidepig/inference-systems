from types import SimpleNamespace

import pytest

from nanovllm import SamplingParams
from nanovllm.engine.hybrid_prefix_cache import PendingHybridPrefixCapture
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


NOW_NS = 10_000_000_000


@pytest.fixture(autouse=True)
def _stable_sequence_block_size():
    """Isolate scheduler tests from legacy tests that mutate this class field."""
    Sequence.block_size = 256
    yield
    Sequence.block_size = 256


def _config(**overrides):
    values = dict(
        max_num_seqs=4,
        max_num_batched_tokens=1024,
        eos=0,
        kvcache_block_size=256,
        prefix_match_unit=256,
        num_kvcache_blocks=32,
        scheduling_policy="prefill_first",
        waiting_admission_policy="hybrid_state_aware",
        preemption_policy="recompute_aware",
        hybrid_scheduler_candidate_window=4,
        hybrid_scheduler_aging_tokens_per_ms=0.5,
        hybrid_scheduler_max_wait_ms=200.0,
        hybrid_scheduler_min_saved_tokens=16,
        hybrid_scheduler_preemption_penalty=0.0,
        enable_prefix_cache=True,
        enable_chunked_prefill=True,
        enable_hybrid_prefix_cache=False,
        num_speculative_tokens=0,
        request_metrics_history_size=16,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _sequence(tokens, *, age_ms=0.0):
    return Sequence(
        list(tokens),
        SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True),
        arrival_time_ns=NOW_NS - int(age_ms * 1e6),
    )


def _seed_prefix(scheduler, tokens, boundary=512):
    source = _sequence(tokens)
    scheduler.block_manager.allocate(source, num_cached_blocks=0)
    source.num_scheduled_tokens = boundary
    scheduler.block_manager.hash_blocks(source)
    scheduler.block_manager.deallocate(source)


def test_bounded_admission_prefers_jointly_reusable_work():
    scheduler = Scheduler(_config())
    shared = list(range(512))
    _seed_prefix(scheduler, shared + [700] * 88)
    cold = _sequence([900] * 600, age_ms=10)
    warm = _sequence(shared + [701] * 88, age_ms=5)
    scheduler.add(cold)
    scheduler.add(warm)

    selected = scheduler._select_waiting_candidate(NOW_NS)

    assert selected is warm
    assert list(scheduler.waiting) == [warm, cold]
    assert warm.last_reusable_tokens == 512
    assert cold.scheduler_bypass_count == 1
    metrics = scheduler.get_metrics()
    assert metrics["hybrid_scheduler_reorders"] == 1
    assert metrics["hybrid_scheduler_estimated_saved_prefill_tokens"] == 512


def test_candidate_outside_bounded_window_cannot_jump_queue():
    scheduler = Scheduler(
        _config(hybrid_scheduler_candidate_window=1)
    )
    shared = list(range(512))
    _seed_prefix(scheduler, shared + [700] * 88)
    cold = _sequence([900] * 600)
    warm = _sequence(shared + [701] * 88)
    scheduler.add(cold)
    scheduler.add(warm)

    assert scheduler._select_waiting_candidate(NOW_NS) is cold
    assert list(scheduler.waiting) == [cold, warm]


def test_zero_prefill_budget_does_not_probe_or_reorder_waiting_queue():
    scheduler = Scheduler(_config())
    cold = _sequence([900] * 600)
    warm = _sequence([901] * 600)
    scheduler.add(cold)
    scheduler.add(warm)

    assert scheduler._schedule_prefill(0, 1) == []
    assert list(scheduler.waiting) == [cold, warm]
    assert scheduler.get_metrics()["hybrid_scheduler_admission_decisions"] == 0


def test_hysteresis_keeps_feasible_fcfs_head():
    scheduler = Scheduler(
        _config(hybrid_scheduler_min_saved_tokens=600)
    )
    shared = list(range(512))
    _seed_prefix(scheduler, shared + [700] * 88)
    cold = _sequence([900] * 600)
    warm = _sequence(shared + [701] * 88)
    scheduler.add(cold)
    scheduler.add(warm)

    assert scheduler._select_waiting_candidate(NOW_NS) is cold
    assert scheduler.last_hybrid_scheduler_decision["reason"] == (
        "benefit_below_hysteresis"
    )


def test_hysteresis_can_be_disabled_for_ablation():
    scheduler = Scheduler(
        _config(
            hybrid_scheduler_min_saved_tokens=600,
            hybrid_scheduler_enable_hysteresis=False,
        )
    )
    shared = list(range(512))
    _seed_prefix(scheduler, shared + [700] * 88)
    cold = _sequence([900] * 600)
    warm = _sequence(shared + [701] * 88)
    scheduler.add(cold)
    scheduler.add(warm)

    assert scheduler._select_waiting_candidate(NOW_NS) is warm
    assert scheduler.get_metrics()["hybrid_scheduler_hysteresis_rejections"] == 0


def test_aging_deadline_prevents_starvation():
    scheduler = Scheduler(_config(hybrid_scheduler_max_wait_ms=100.0))
    shared = list(range(512))
    _seed_prefix(scheduler, shared + [700] * 88)
    overdue = _sequence([900] * 600, age_ms=150)
    warm = _sequence(shared + [701] * 88, age_ms=5)
    scheduler.add(overdue)
    scheduler.add(warm)

    assert scheduler._select_waiting_candidate(NOW_NS) is overdue
    assert scheduler.last_hybrid_scheduler_decision["reason"] == "aging_deadline"
    assert scheduler.get_metrics()["hybrid_scheduler_aging_overrides"] == 1


def test_aging_can_be_disabled_for_ablation():
    scheduler = Scheduler(
        _config(
            hybrid_scheduler_max_wait_ms=100.0,
            hybrid_scheduler_enable_aging=False,
        )
    )
    shared = list(range(512))
    _seed_prefix(scheduler, shared + [700] * 88)
    overdue = _sequence([900] * 600, age_ms=150)
    warm = _sequence(shared + [701] * 88, age_ms=5)
    scheduler.add(overdue)
    scheduler.add(warm)

    assert scheduler._select_waiting_candidate(NOW_NS) is warm
    assert scheduler.get_metrics()["hybrid_scheduler_aging_overrides"] == 0


def test_chunk_continuation_and_preempted_head_are_sticky():
    scheduler = Scheduler(_config())
    continuation = _sequence([1] * 600)
    scheduler.block_manager.allocate(continuation, num_cached_blocks=0)
    continuation.num_cached_tokens = 256
    warm = _sequence([2] * 600)
    scheduler.add(continuation)
    scheduler.add(warm)

    assert scheduler._select_waiting_candidate(NOW_NS) is continuation
    assert scheduler.get_metrics()["hybrid_scheduler_admission_decisions"] == 0

    scheduler.block_manager.deallocate(continuation)
    continuation.preemption_count = 1
    assert scheduler._select_waiting_candidate(NOW_NS) is continuation


def test_sticky_recovery_can_be_disabled_for_ablation():
    scheduler = Scheduler(
        _config(hybrid_scheduler_enable_sticky_recovery=False)
    )
    shared = list(range(512))
    _seed_prefix(scheduler, shared + [700] * 88)
    continuation = _sequence([900] * 600)
    scheduler.block_manager.allocate(continuation, num_cached_blocks=0)
    continuation.num_cached_tokens = 256
    warm = _sequence(shared + [701] * 88)
    scheduler.add(continuation)
    scheduler.add(warm)

    assert scheduler._select_waiting_candidate(NOW_NS) is warm
    assert scheduler.get_metrics()[
        "hybrid_scheduler_sticky_recovery_triggers"
    ] == 0


def test_reclaimable_blocks_excludes_shared_physical_pages():
    scheduler = Scheduler(_config())
    sequence = _sequence([1] * 600)
    scheduler.block_manager.allocate(sequence, num_cached_blocks=0)
    shared_block = sequence.block_table[0]
    scheduler.block_manager.blocks[shared_block].ref_count += 1

    assert scheduler.block_manager.reclaimable_blocks(sequence) == 2


def test_recompute_aware_preemption_beats_lifo_baseline():
    scheduler = Scheduler(_config())

    cheap = _sequence(list(range(512)) + [11] * 88)
    scheduler.block_manager.allocate(cheap, num_cached_blocks=0)
    cheap.num_scheduled_tokens = 512
    scheduler.block_manager.hash_blocks(cheap)
    cheap.num_cached_tokens = 600
    cheap.num_scheduled_tokens = 0
    cheap.status = SequenceStatus.RUNNING

    expensive = _sequence([99] * 600)
    scheduler.block_manager.allocate(expensive, num_cached_blocks=0)
    expensive.num_cached_tokens = 600
    expensive.status = SequenceStatus.RUNNING
    scheduler.running.append(expensive)

    victim = scheduler._select_preemption_victim(cheap)

    assert victim.sequence is cheap
    assert victim.reusable_tokens == 512
    assert victim.recompute_tokens == 88
    assert victim.reclaimable_blocks == 3
    metrics = scheduler.get_metrics()
    assert metrics["recompute_aware_estimated_recompute_tokens"] == 88
    assert metrics["recompute_aware_avoided_recompute_tokens"] == 512


def test_lifo_preemption_policy_preserves_legacy_victim_order():
    scheduler = Scheduler(_config(preemption_policy="lifo"))
    current = _sequence([1] * 300)
    older = _sequence([2] * 300)
    newest = _sequence([3] * 300)
    for sequence in (current, older, newest):
        scheduler.block_manager.allocate(sequence, num_cached_blocks=0)
        sequence.num_cached_tokens = 300
        sequence.status = SequenceStatus.RUNNING
    scheduler.running.extend([older, newest])

    victim = scheduler._select_preemption_victim(current)

    assert victim.sequence is newest
    assert scheduler.get_metrics()["recompute_aware_preemptions"] == 0


def test_decode_pressure_preempts_cheaper_request_and_continues_current():
    scheduler = Scheduler(_config(num_kvcache_blocks=4))
    current = _sequence([90] * 257)
    scheduler.block_manager.allocate(current, num_cached_blocks=0)
    current.num_cached_tokens = 256
    current.status = SequenceStatus.RUNNING

    cheap = _sequence(list(range(256)) + [11] * 44)
    scheduler.block_manager.allocate(cheap, num_cached_blocks=0)
    cheap.num_scheduled_tokens = 256
    scheduler.block_manager.hash_blocks(cheap)
    cheap.num_scheduled_tokens = 0
    cheap.num_cached_tokens = 256
    cheap.status = SequenceStatus.RUNNING
    scheduler.running.extend([current, cheap])

    scheduled = scheduler._schedule_decode(sequence_budget=1)

    assert scheduled == [current]
    assert current in scheduler.running
    assert cheap in scheduler.waiting
    assert cheap.preemption_count == 1
    assert len(current.block_table) == 3
    assert scheduler.get_metrics()["preemption_count"] == 1


def test_hybrid_admission_uses_kv_gdn_intersection_not_kv_only_hit():
    old_block_size = Sequence.block_size
    Sequence.block_size = 4
    try:
        scheduler = Scheduler(
            _config(
                kvcache_block_size=4,
                prefix_match_unit=4,
                num_kvcache_blocks=12,
                enable_hybrid_prefix_cache=True,
                hybrid_prefix_checkpoint_interval_blocks=1,
            ),
            state_manager=SimpleNamespace(can_allocate=True),
            prefix_checkpoint_pool=SimpleNamespace(capacity=4),
        )
        source = _sequence(list(range(9)))
        scheduler.block_manager.allocate(source, num_cached_blocks=0)
        source.num_scheduled_tokens = 8
        scheduler.block_manager.hash_blocks(source)
        boundary = scheduler.block_manager.prefix_metadata_at_boundary(source, 4)
        scheduler.block_manager.deallocate(source)
        scheduler.hybrid_prefix_cache.publish(
            PendingHybridPrefixCapture(
                prefix_hash=boundary.prefix_hash,
                boundary_tokens=4,
                tail_block_id=boundary.tail_block_id,
                state_slot=0,
            ),
            checkpoint_slot=0,
        )
        incoming = _sequence(list(range(8)) + [99])

        probe = scheduler._admission_probe(incoming, 0, NOW_NS)

        assert probe.kv_candidate_tokens == 8
        assert probe.reusable_tokens == 4
        assert probe.uncached_tokens == 5
    finally:
        Sequence.block_size = old_block_size


def test_hybrid_kv_only_scoring_is_available_as_safe_ablation():
    old_block_size = Sequence.block_size
    Sequence.block_size = 4
    try:
        scheduler = Scheduler(
            _config(
                kvcache_block_size=4,
                prefix_match_unit=4,
                num_kvcache_blocks=12,
                enable_hybrid_prefix_cache=True,
                hybrid_prefix_checkpoint_interval_blocks=1,
                hybrid_scheduler_score_source="kv_only",
            ),
            state_manager=SimpleNamespace(can_allocate=True),
            prefix_checkpoint_pool=SimpleNamespace(capacity=4),
        )
        source = _sequence(list(range(9)))
        scheduler.block_manager.allocate(source, num_cached_blocks=0)
        source.num_scheduled_tokens = 8
        scheduler.block_manager.hash_blocks(source)
        boundary = scheduler.block_manager.prefix_metadata_at_boundary(source, 4)
        scheduler.block_manager.deallocate(source)
        scheduler.hybrid_prefix_cache.publish(
            PendingHybridPrefixCapture(
                boundary.prefix_hash,
                4,
                boundary.tail_block_id,
                state_slot=0,
            ),
            checkpoint_slot=0,
        )
        probe = scheduler._admission_probe(
            _sequence(list(range(8)) + [99]), 0, NOW_NS
        )

        assert probe.kv_candidate_tokens == 8
        assert probe.gdn_checkpoint_tokens == 4
        assert probe.joint_recoverable_tokens == 4
        assert probe.reusable_tokens == 8
        # This changes ranking only; actual restore still uses the joint plan.
    finally:
        Sequence.block_size = old_block_size


def test_scheduler_profiling_exports_latency_and_decision_trace():
    scheduler = Scheduler(_config(enable_scheduler_profiling=True))
    first = _sequence([1] * 300)
    second = _sequence([2] * 300)
    scheduler.add(first)
    scheduler.add(second)

    scheduler._select_waiting_candidate(NOW_NS)
    metrics = scheduler.get_metrics()

    assert metrics["hybrid_scheduler_admission_latency"]["count"] == 1
    assert metrics["hybrid_scheduler_probe_latency"]["count"] == 1
    assert scheduler.get_scheduler_decision_events()[0]["type"] == "admission"
