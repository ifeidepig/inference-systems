from types import SimpleNamespace

import torch

from nanovllm import SamplingParams
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.hybrid_prefix_cache import (
    FullAttentionPrefixManager,
    GDNCheckpointManager,
    HybridCheckpointRetentionPolicy,
    HybridPrefixCoordinator,
    HybridPrefixCache,
    PendingHybridPrefixCapture,
    PrefixKVCandidate,
)
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.state_manager import (
    HybridPrefixCheckpointPool,
    HybridStateManager,
)


def _manager(max_num_seqs=2):
    return HybridStateManager(
        max_num_seqs=max_num_seqs,
        num_linear_layers=2,
        num_value_heads=1,
        key_head_dim=2,
        value_head_dim=3,
        conv_dim=4,
        conv_kernel_size=4,
        conv_dtype=torch.bfloat16,
        device="cpu",
    )


def test_checkpoint_pool_capture_restore_free_isolation():
    manager = _manager()
    pool = HybridPrefixCheckpointPool(
        manager,
        memory_budget_bytes=manager.bytes_per_slot() * 2,
    )
    first = SimpleNamespace(state_slot=None)
    second = SimpleNamespace(state_slot=None)
    first_slot = manager.allocate(first)
    second_slot = manager.allocate(second)
    manager.recurrent_states[:, first_slot].fill_(3.5)
    manager.conv_states[:, first_slot].fill_(2.0)

    checkpoint_slot = pool.capture(first_slot)
    manager.recurrent_states[:, second_slot].fill_(99)
    manager.conv_states[:, second_slot].fill_(99)
    pool.restore(checkpoint_slot, second_slot)

    torch.testing.assert_close(
        manager.recurrent_states[:, second_slot],
        manager.recurrent_states[:, first_slot],
    )
    torch.testing.assert_close(
        manager.conv_states[:, second_slot],
        manager.conv_states[:, first_slot],
    )
    pool.free(checkpoint_slot)
    assert checkpoint_slot not in pool.used_checkpoint_slots
    assert not pool.recurrent_checkpoints[checkpoint_slot].any()
    assert not pool.conv_checkpoints[checkpoint_slot].any()


def test_checkpoint_pool_fp32_bf16_int8_storage_and_roundtrip():
    expected_dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "int8": torch.int8,
    }
    capacities = {}
    for checkpoint_dtype in expected_dtype:
        manager = HybridStateManager(
            max_num_seqs=1,
            num_linear_layers=2,
            num_value_heads=2,
            key_head_dim=8,
            value_head_dim=16,
            conv_dim=4,
            conv_kernel_size=4,
            conv_dtype=torch.bfloat16,
            device="cpu",
        )
        request = SimpleNamespace(state_slot=None)
        request_slot = manager.allocate(request)
        torch.manual_seed(7)
        original_recurrent = torch.randn_like(
            manager.recurrent_states[:, request_slot]
        )
        original_conv = torch.randn_like(
            manager.conv_states[:, request_slot]
        )
        manager.recurrent_states[:, request_slot].copy_(original_recurrent)
        manager.conv_states[:, request_slot].copy_(original_conv)
        pool = HybridPrefixCheckpointPool(
            manager,
            memory_budget_bytes=manager.bytes_per_slot() * 4,
            checkpoint_dtype=checkpoint_dtype,
        )

        checkpoint_slot = pool.capture(request_slot)
        manager.recurrent_states[:, request_slot].zero_()
        manager.conv_states[:, request_slot].zero_()
        pool.restore(checkpoint_slot, request_slot)

        assert pool.recurrent_checkpoints.dtype == expected_dtype[checkpoint_dtype]
        assert pool.memory_bytes() == pool.capacity * pool.bytes_per_checkpoint
        torch.testing.assert_close(
            manager.conv_states[:, request_slot],
            original_conv,
            rtol=0,
            atol=0,
        )
        restored = manager.recurrent_states[:, request_slot]
        if checkpoint_dtype == "fp32":
            torch.testing.assert_close(restored, original_recurrent, rtol=0, atol=0)
            assert pool.recurrent_scales is None
        elif checkpoint_dtype == "bf16":
            torch.testing.assert_close(
                restored,
                original_recurrent.to(torch.bfloat16).to(torch.float32),
                rtol=0,
                atol=0,
            )
            assert pool.recurrent_scales is None
        else:
            assert pool.recurrent_scales is not None
            assert pool.recurrent_scales.shape[1:] == (
                manager.num_linear_layers,
                2,
                8,
                1,
            )
            error = (restored - original_recurrent).abs()
            bound = pool.recurrent_scales[checkpoint_slot] / 2 + 1e-6
            assert torch.all(error <= bound)
        capacities[checkpoint_dtype] = pool.capacity

    assert capacities["fp32"] < capacities["bf16"] < capacities["int8"]


def test_int8_checkpoint_free_clears_qdata_scale_and_conv():
    manager = _manager(max_num_seqs=1)
    request = SimpleNamespace(state_slot=None)
    request_slot = manager.allocate(request)
    manager.recurrent_states[:, request_slot].zero_()
    manager.conv_states[:, request_slot].fill_(3)
    pool = HybridPrefixCheckpointPool(
        manager,
        memory_budget_bytes=manager.bytes_per_slot(),
        checkpoint_dtype="int8",
    )

    checkpoint_slot = pool.capture(request_slot)

    assert not pool.recurrent_checkpoints[checkpoint_slot].any()
    assert torch.all(pool.recurrent_scales[checkpoint_slot] == 1e-8)
    pool.free(checkpoint_slot)
    assert not pool.recurrent_checkpoints[checkpoint_slot].any()
    assert not pool.recurrent_scales[checkpoint_slot].any()
    assert not pool.conv_checkpoints[checkpoint_slot].any()


def test_hybrid_metadata_falls_back_to_shorter_aligned_candidate():
    old_block_size = Sequence.block_size
    Sequence.block_size = 4
    try:
        blocks = BlockManager(num_blocks=8, block_size=4)
        source = Sequence(
            list(range(9)),
            SamplingParams(temperature=0.0, max_tokens=1),
        )
        blocks.allocate(source, num_cached_blocks=0)
        source.num_scheduled_tokens = 8
        blocks.hash_blocks(source)
        first_boundary = blocks.prefix_metadata_at_boundary(source, 4)
        blocks.deallocate(source)

        cache = HybridPrefixCache(capacity=1)
        pending = PendingHybridPrefixCapture(
            prefix_hash=first_boundary.prefix_hash,
            boundary_tokens=4,
            tail_block_id=first_boundary.tail_block_id,
            state_slot=0,
        )
        cache.publish(pending, checkpoint_slot=0)

        incoming = Sequence(
            list(range(8)) + [99],
            SamplingParams(temperature=0.0, max_tokens=1),
        )
        candidates = blocks.find_prefix_candidates(incoming)
        assert [candidate.boundary_tokens for candidate in candidates] == [8, 4]
        hit = cache.find_hit(candidates)
        assert hit is not None
        assert hit.candidate.boundary_tokens == 4
        assert hit.checkpoint_slot == 0
    finally:
        Sequence.block_size = old_block_size


def test_hybrid_metadata_lru_eviction_is_bounded():
    cache = HybridPrefixCache(capacity=1)
    first = PendingHybridPrefixCapture(11, 4, 2, 0)
    second = PendingHybridPrefixCapture(22, 8, 3, 1)
    cache.publish(first, checkpoint_slot=0)

    evicted = cache.reserve_capture(second)

    assert len(evicted) == 1
    assert evicted[0].prefix_hash == 11
    assert not cache.entries
    cache.publish(second, checkpoint_slot=0)
    assert list(cache.entries) == [(22, 8)]
    assert cache.get_metrics()["hybrid_prefix_eviction_count"] == 1


def test_coordinator_plans_longest_jointly_restorable_prefix():
    old_block_size = Sequence.block_size
    Sequence.block_size = 4
    try:
        blocks = BlockManager(num_blocks=8, block_size=4)
        source = Sequence(
            list(range(9)),
            SamplingParams(temperature=0.0, max_tokens=1),
        )
        blocks.allocate(source, num_cached_blocks=0)
        source.num_scheduled_tokens = 8
        blocks.hash_blocks(source)
        first = blocks.prefix_metadata_at_boundary(source, 4)
        blocks.deallocate(source)

        checkpoints = GDNCheckpointManager(capacity=1)
        checkpoints.publish(
            PendingHybridPrefixCapture(
                first.prefix_hash,
                first.boundary_tokens,
                first.tail_block_id,
                state_slot=0,
            ),
            checkpoint_slot=0,
        )
        coordinator = HybridPrefixCoordinator(
            FullAttentionPrefixManager(blocks),
            checkpoints,
        )
        incoming = Sequence(
            list(range(8)) + [99],
            SamplingParams(temperature=0.0, max_tokens=1),
        )

        plan = coordinator.plan(incoming)

        assert plan is not None
        assert plan.boundary_tokens == 4
        assert plan.num_cached_blocks == 1
        assert plan.hit is not None
        assert plan.hit.checkpoint_slot == 0
    finally:
        Sequence.block_size = old_block_size


def test_coordinator_returns_cold_plan_when_only_kv_is_resident():
    old_block_size = Sequence.block_size
    Sequence.block_size = 4
    try:
        blocks = BlockManager(num_blocks=8, block_size=4)
        source = Sequence(
            list(range(9)),
            SamplingParams(temperature=0.0, max_tokens=1),
        )
        blocks.allocate(source, num_cached_blocks=0)
        source.num_scheduled_tokens = 8
        blocks.hash_blocks(source)
        blocks.deallocate(source)
        coordinator = HybridPrefixCoordinator(
            FullAttentionPrefixManager(blocks),
            GDNCheckpointManager(capacity=1),
        )
        incoming = Sequence(
            list(range(8)) + [99],
            SamplingParams(temperature=0.0, max_tokens=1),
        )

        plan = coordinator.plan(incoming)

        assert plan is not None
        assert plan.hit is None
        assert plan.boundary_tokens == 0
        assert plan.num_cached_blocks == 0
        assert coordinator.get_metrics()["hybrid_prefix_fallback_count"] == 1
    finally:
        Sequence.block_size = old_block_size


def test_adaptive_coordinator_promotes_longest_kv_only_junction_once():
    old_block_size = Sequence.block_size
    Sequence.block_size = 4
    try:
        blocks = BlockManager(num_blocks=8, block_size=4)
        source = Sequence(
            list(range(9)),
            SamplingParams(temperature=0.0, max_tokens=1),
        )
        blocks.allocate(source, 0)
        source.num_scheduled_tokens = 8
        blocks.hash_blocks(source)
        blocks.deallocate(source)
        coordinator = HybridPrefixCoordinator(
            FullAttentionPrefixManager(blocks),
            GDNCheckpointManager(capacity=2),
            promote_shared_junctions=True,
        )
        incoming = Sequence(
            list(range(8)) + [99],
            SamplingParams(temperature=0.0, max_tokens=1),
        )

        plan = coordinator.plan(incoming)

        assert plan is not None
        assert plan.hit is None
        assert plan.promotion_candidate is not None
        assert plan.promotion_candidate.boundary_tokens == 8
        coordinator.commit(plan)
        duplicate = coordinator.plan(incoming)
        assert duplicate is not None
        assert duplicate.promotion_candidate is None
        metrics = coordinator.get_metrics()
        assert metrics["hybrid_prefix_kv_only_miss_count"] == 2
        assert metrics["hybrid_prefix_alignment_lost_tokens"] == 16
        assert metrics[
            "hybrid_prefix_shared_junction_promotions_planned"
        ] == 1
    finally:
        Sequence.block_size = old_block_size


def test_promotion_threshold_three_waits_for_third_total_sighting():
    old_block_size = Sequence.block_size
    Sequence.block_size = 4
    try:
        blocks = BlockManager(num_blocks=8, block_size=4)
        source = Sequence(
            list(range(9)),
            SamplingParams(temperature=0.0, max_tokens=1),
        )
        blocks.allocate(source, 0)
        source.num_scheduled_tokens = 8
        blocks.hash_blocks(source)
        blocks.deallocate(source)
        checkpoints = GDNCheckpointManager(capacity=2)
        coordinator = HybridPrefixCoordinator(
            FullAttentionPrefixManager(blocks),
            checkpoints,
            promote_shared_junctions=True,
            promotion_min_sightings=3,
        )

        second = Sequence(
            list(range(8)) + [99],
            SamplingParams(temperature=0.0, max_tokens=1),
        )
        second_plan = coordinator.plan(second)
        assert second_plan is not None
        assert second_plan.promotion_candidate is None

        third = Sequence(
            list(range(8)) + [100],
            SamplingParams(temperature=0.0, max_tokens=1),
        )
        third_plan = coordinator.plan(third)
        assert third_plan is not None
        assert third_plan.promotion_candidate is not None
        coordinator.commit(third_plan)
        candidate = third_plan.promotion_candidate
        assert coordinator.demand_observations(candidate) == 2
        pending = PendingHybridPrefixCapture(
            candidate.prefix_hash,
            candidate.boundary_tokens,
            candidate.tail_block_id,
            state_slot=0,
            reason="shared_junction",
            replay_saved_tokens=candidate.boundary_tokens,
            demand_count=2,
        )
        coordinator.publish(pending, checkpoint_slot=0)

        fourth = Sequence(
            list(range(8)) + [101],
            SamplingParams(temperature=0.0, max_tokens=1),
        )
        fourth_plan = coordinator.plan(fourth)
        assert fourth_plan is not None
        assert fourth_plan.hit is not None
        coordinator.commit(fourth_plan)
        metrics = coordinator.get_metrics()
        assert metrics["hybrid_prefix_promotion_min_sightings"] == 3
        assert metrics["hybrid_prefix_kv_only_miss_count"] == 2
        assert metrics["hybrid_prefix_useful_promotion_count"] == 1
        assert metrics["hybrid_prefix_useful_promotion_ratio"] == 1.0
        assert metrics["hybrid_prefix_promotions_not_yet_reused"] == 0
        assert metrics[
            "hybrid_prefix_shared_junction_saved_replay_tokens"
        ] == 8
    finally:
        Sequence.block_size = old_block_size


def test_failed_shared_junction_capture_can_be_promoted_again():
    old_block_size = Sequence.block_size
    Sequence.block_size = 4
    try:
        blocks = BlockManager(num_blocks=8, block_size=4)
        source = Sequence(
            list(range(9)),
            SamplingParams(temperature=0.0, max_tokens=1),
        )
        blocks.allocate(source, 0)
        source.num_scheduled_tokens = 8
        blocks.hash_blocks(source)
        blocks.deallocate(source)
        coordinator = HybridPrefixCoordinator(
            FullAttentionPrefixManager(blocks),
            GDNCheckpointManager(capacity=2),
            promote_shared_junctions=True,
        )
        incoming = Sequence(
            list(range(8)) + [99],
            SamplingParams(temperature=0.0, max_tokens=1),
        )
        first = coordinator.plan(incoming)
        assert first is not None and first.promotion_candidate is not None
        coordinator.commit(first)
        candidate = first.promotion_candidate
        coordinator.resolve_capture(
            PendingHybridPrefixCapture(
                candidate.prefix_hash,
                candidate.boundary_tokens,
                candidate.tail_block_id,
                state_slot=0,
                reason="shared_junction",
            ),
            published=False,
        )

        retry = coordinator.plan(incoming)

        assert retry is not None
        assert retry.promotion_candidate is not None
        assert coordinator.get_metrics()[
            "hybrid_prefix_shared_junction_promotions_cancelled"
        ] == 1
    finally:
        Sequence.block_size = old_block_size


def test_adaptive_scheduler_promotes_shared_junction_for_next_request():
    class StateController:
        def __init__(self):
            self.next_slot = 0

        def call(self, method, *args):
            if method == "allocate_state_slot":
                args[0].state_slot = self.next_slot
                self.next_slot += 1
                return args[0].state_slot
            if method == "evict_prefix_checkpoint":
                return None
            raise AssertionError(f"unexpected state call: {method}")

    old_block_size = Sequence.block_size
    Sequence.block_size = 4
    try:
        config = SimpleNamespace(
            max_num_seqs=4,
            max_num_batched_tokens=64,
            eos=-1,
            kvcache_block_size=4,
            prefix_match_unit=4,
            num_kvcache_blocks=16,
            scheduling_policy="prefill_first",
            enable_prefix_cache=True,
            enable_chunked_prefill=True,
            enable_hybrid_prefix_cache=True,
            enable_hybrid_internal_checkpoints=False,
            hybrid_prefix_checkpoint_interval_blocks=4,
            hybrid_prefix_retention_policy="adaptive",
            hybrid_prefix_eviction_policy="cost_aware",
        )
        scheduler = Scheduler(
            config,
            state_manager=SimpleNamespace(can_allocate=True),
            state_controller=StateController(),
            prefix_checkpoint_pool=SimpleNamespace(capacity=4),
        )
        sampling = SamplingParams(temperature=0.0, max_tokens=1)

        # Request A has a prompt-tail checkpoint at 8, inside its unique
        # suffix, while its first four KV tokens remain reusable.
        source = Sequence([0, 1, 2, 3, 10, 11, 12, 13, 14], sampling)
        scheduler.block_manager.allocate(source, 0)
        source.num_scheduled_tokens = 8
        scheduler.block_manager.hash_blocks(source)
        tail = scheduler.block_manager.prefix_metadata_at_boundary(source, 8)
        scheduler.hybrid_prefix_cache.publish(
            PendingHybridPrefixCapture(
                tail.prefix_hash,
                8,
                tail.tail_block_id,
                state_slot=0,
                reason="prompt_tail",
            ),
            checkpoint_slot=0,
        )
        scheduler.block_manager.deallocate(source)

        # Request B sees KV@4 but no GDN@4. It must replay from zero and
        # captures the shared junction while crossing it.
        second = Sequence([0, 1, 2, 3, 20, 21, 22, 23, 24], sampling)
        scheduler.add(second)
        batch = scheduler._schedule_prefill(64, 1)
        assert batch == [second]
        assert second.num_cached_tokens == 0
        assert second.num_scheduled_tokens == 4
        pending, hashed = scheduler.prepare_prefix_captures(batch, True)
        assert hashed
        assert len(pending) == 1
        assert pending[0].boundary_tokens == 4
        assert pending[0].reason == "shared_junction"
        assert pending[0].replay_saved_tokens == 4
        scheduler.publish_prefix_capture(pending[0], checkpoint_slot=1)
        scheduler.postprocess(batch, [99], True, blocks_already_hashed=True)

        # Request C can now restore the jointly available KV/GDN boundary.
        third = Sequence([0, 1, 2, 3, 30, 31, 32, 33, 34], sampling)
        plan = scheduler.hybrid_prefix_coordinator.plan(third)
        assert plan is not None
        assert plan.hit is not None
        assert plan.boundary_tokens == 4
        assert plan.promotion_candidate is None
        metrics = scheduler.get_metrics()
        assert metrics[
            "hybrid_prefix_shared_junction_promotions_published"
        ] == 1
        assert metrics["hybrid_prefix_demand_entries"] == 0
    finally:
        Sequence.block_size = old_block_size


def test_adaptive_retention_adds_prompt_tail_without_densifying_blocks():
    periodic = HybridCheckpointRetentionPolicy(
        block_size=4,
        interval_tokens=16,
        mode="periodic",
    )
    adaptive = HybridCheckpointRetentionPolicy(
        block_size=4,
        interval_tokens=16,
        mode="adaptive",
    )

    assert periodic.next_retained_boundary(16, 27) is None
    retained = adaptive.next_retained_boundary(16, 27)
    assert retained is not None
    assert retained.boundary_tokens == 24
    assert retained.reason == "prompt_tail"
    assert adaptive.classify(20, 27) is None


def test_cost_aware_eviction_keeps_reused_shorter_checkpoint():
    cache = GDNCheckpointManager(capacity=2, eviction_policy="cost_aware")
    short = PendingHybridPrefixCapture(11, 4, 1, 0)
    long = PendingHybridPrefixCapture(22, 8, 2, 1)
    incoming = PendingHybridPrefixCapture(33, 12, 3, 2)
    cache.publish(short, checkpoint_slot=0)
    cache.publish(long, checkpoint_slot=1)
    short_candidate = PrefixKVCandidate(1, 4, 11, 1)
    assert cache.find_hit([short_candidate]) is not None
    assert cache.find_hit([short_candidate]) is not None

    evicted = cache.reserve_capture(incoming)

    assert len(evicted) == 1
    assert evicted[0].prefix_hash == 22
    assert (11, 4) in cache.entries
    assert cache.get_metrics()["hybrid_prefix_evicted_retained_value"] == 8


def test_metrics_count_unused_shared_junction_eviction_as_pollution():
    cache = GDNCheckpointManager(capacity=1)
    shared = PendingHybridPrefixCapture(
        11,
        8,
        1,
        0,
        reason="shared_junction",
        replay_saved_tokens=8,
        demand_count=1,
    )
    cache.publish(shared, checkpoint_slot=0)

    evicted = cache.reserve_capture(
        PendingHybridPrefixCapture(22, 16, 2, 1, reason="prompt_tail")
    )

    assert len(evicted) == 1
    metrics = cache.get_metrics()
    assert metrics["hybrid_prefix_shared_junction_eviction_count"] == 1
    assert metrics["hybrid_prefix_unused_promotion_eviction_count"] == 1
    assert metrics["hybrid_prefix_entries_shared_junction"] == 0
    assert metrics["hybrid_prefix_peak_shared_junction_entries"] == 1


def test_adaptive_scheduler_splits_at_prompt_tail_boundary():
    class StateController:
        def call(self, method, *args):
            if method == "allocate_state_slot":
                args[0].state_slot = 0
                return 0
            raise AssertionError(f"unexpected state call: {method}")

    old_block_size = Sequence.block_size
    Sequence.block_size = 4
    try:
        config = SimpleNamespace(
            max_num_seqs=2,
            max_num_batched_tokens=64,
            eos=-1,
            kvcache_block_size=4,
            num_kvcache_blocks=16,
            scheduling_policy="prefill_first",
            enable_prefix_cache=True,
            enable_chunked_prefill=True,
            enable_hybrid_prefix_cache=True,
            hybrid_prefix_checkpoint_interval_blocks=4,
            hybrid_prefix_retention_policy="adaptive",
            hybrid_prefix_eviction_policy="lru",
        )
        scheduler = Scheduler(
            config,
            state_manager=SimpleNamespace(can_allocate=True),
            state_controller=StateController(),
            prefix_checkpoint_pool=SimpleNamespace(capacity=2),
        )
        seq = Sequence(
            list(range(27)),
            SamplingParams(temperature=0.0, max_tokens=1),
        )
        scheduler.add(seq)

        first = scheduler._schedule_prefill(64, 1)
        assert first[0].num_scheduled_tokens == 16
        scheduler.postprocess(first, [99], is_prefill=True)
        second = scheduler._schedule_prefill(64, 1)

        assert second[0].num_scheduled_tokens == 8
        assert seq.num_cached_tokens + seq.num_scheduled_tokens == 24
        pending, blocks_hashed = scheduler.prepare_prefix_captures(
            second,
            is_prefill=True,
        )
        assert blocks_hashed
        assert len(pending) == 1
        assert pending[0].boundary_tokens == 24
        assert pending[0].reason == "prompt_tail"
    finally:
        Sequence.block_size = old_block_size


def test_fine_grained_index_finds_sub_block_boundary_and_cow_plan():
    old_block_size = Sequence.block_size
    Sequence.block_size = 256
    try:
        blocks = BlockManager(
            num_blocks=12,
            block_size=256,
            prefix_match_unit=16,
        )
        shared = list(range(500))
        source = Sequence(
            shared + [700 + index for index in range(64)],
            SamplingParams(temperature=0.0, max_tokens=1),
        )
        blocks.allocate(source, 0)
        source.num_scheduled_tokens = source.num_tokens
        blocks.hash_blocks(source)
        blocks.deallocate(source)
        incoming = Sequence(
            shared + [900 + index for index in range(64)],
            SamplingParams(temperature=0.0, max_tokens=1),
        )

        candidate = blocks.find_prefix_candidates(incoming)[0]

        assert candidate.boundary_tokens == 496
        assert candidate.num_cached_blocks == 1
        assert candidate.tail_valid_tokens == 240
        assert blocks.can_allocate_candidate(incoming, candidate)
        page_copy = blocks.allocate_candidate(incoming, candidate)
        assert page_copy is not None
        assert page_copy.source_block_id == candidate.tail_block_id
        assert page_copy.destination_block_id == incoming.block_table[1]
        assert page_copy.destination_block_id != page_copy.source_block_id
        assert page_copy.num_tokens == 240
        assert incoming.num_cached_tokens == 496
        blocks.release_cow_source(page_copy)
        blocks.deallocate(incoming)
        assert not blocks.used_block_ids
    finally:
        Sequence.block_size = old_block_size


def test_partial_cow_failure_rolls_back_kv_and_state_ownership():
    class FailingController:
        def call(self, method, *args):
            if method == "allocate_state_slot":
                args[0].state_slot = 0
                return 0
            if method == "copy_prefix_kv":
                raise RuntimeError("injected COW failure")
            if method == "free_state_slot":
                args[0].state_slot = None
                return None
            raise AssertionError(f"unexpected state call: {method}")

    old_block_size = Sequence.block_size
    Sequence.block_size = 8
    try:
        config = SimpleNamespace(
            max_num_seqs=2,
            max_num_batched_tokens=32,
            eos=-1,
            kvcache_block_size=8,
            prefix_match_unit=2,
            num_kvcache_blocks=12,
            scheduling_policy="prefill_first",
            enable_prefix_cache=True,
            enable_chunked_prefill=True,
            enable_hybrid_prefix_cache=True,
            hybrid_prefix_checkpoint_interval_blocks=1,
            hybrid_prefix_checkpoint_interval_tokens=2,
            hybrid_prefix_retention_policy="periodic",
            hybrid_prefix_eviction_policy="lru",
        )
        scheduler = Scheduler(
            config,
            state_manager=SimpleNamespace(can_allocate=True),
            state_controller=FailingController(),
            prefix_checkpoint_pool=SimpleNamespace(capacity=2),
        )
        source = Sequence(
            [0, 1, 2, 3, 4, 5, 80, 81, 82],
            SamplingParams(temperature=0.0, max_tokens=1),
        )
        scheduler.block_manager.allocate(source, 0)
        source.num_scheduled_tokens = 8
        scheduler.block_manager.hash_blocks(source)
        metadata = scheduler.block_manager.prefix_metadata_at_boundary(source, 6)
        scheduler.hybrid_prefix_cache.publish(
            PendingHybridPrefixCapture(
                metadata.prefix_hash,
                6,
                metadata.tail_block_id,
                state_slot=0,
            ),
            checkpoint_slot=0,
        )
        scheduler.block_manager.deallocate(source)
        incoming = Sequence(
            [0, 1, 2, 3, 4, 5, 90, 91, 92],
            SamplingParams(temperature=0.0, max_tokens=1),
        )
        scheduler.add(incoming)

        try:
            scheduler._schedule_prefill(32, 1)
        except RuntimeError as exc:
            assert "injected COW failure" in str(exc)
        else:
            raise AssertionError("injected COW failure did not propagate")

        assert incoming.block_table == []
        assert incoming.state_slot is None
        assert incoming.num_cached_tokens == 0
        assert not scheduler.block_manager.used_block_ids
        assert all(
            block.ref_count == 0 for block in scheduler.block_manager.blocks
        )
    finally:
        Sequence.block_size = old_block_size


def test_fine_match_rounds_n_minus_one_n_and_n_plus_one_safely():
    old_block_size = Sequence.block_size
    Sequence.block_size = 256
    try:
        blocks = BlockManager(12, 256, prefix_match_unit=16)
        source_tokens = list(range(240)) + [800 + i for i in range(80)]
        source = Sequence(
            source_tokens,
            SamplingParams(temperature=0.0, max_tokens=1),
        )
        blocks.allocate(source, 0)
        source.num_scheduled_tokens = source.num_tokens
        blocks.hash_blocks(source)
        blocks.deallocate(source)

        def longest(shared_tokens: int) -> int:
            incoming = Sequence(
                source_tokens[:shared_tokens]
                + [1200 + i for i in range(320 - shared_tokens)],
                SamplingParams(temperature=0.0, max_tokens=1),
            )
            return blocks.find_prefix_candidates(incoming)[0].boundary_tokens

        assert longest(239) == 224
        assert longest(240) == 240
        assert longest(241) == 240
    finally:
        Sequence.block_size = old_block_size


def test_reassigning_page_invalidates_sub_block_tail_hashes():
    old_block_size = Sequence.block_size
    Sequence.block_size = 8
    try:
        blocks = BlockManager(2, 8, prefix_match_unit=2)
        sampling = SamplingParams(temperature=0.0, max_tokens=1)
        source = Sequence(list(range(9)), sampling)
        blocks.allocate(source, 0)
        source.num_scheduled_tokens = source.num_tokens
        blocks.hash_blocks(source)
        assert blocks.find_prefix_candidates(source)
        blocks.deallocate(source)

        replacement = Sequence([50 + i for i in range(9)], sampling)
        blocks.allocate(replacement, 0)

        assert blocks.find_prefix_candidates(source) == []
        assert blocks.prefix_cache_evictions > 0
    finally:
        Sequence.block_size = old_block_size


def test_internal_checkpoint_scheduler_keeps_one_prefill_and_lists_boundaries():
    class StateController:
        def call(self, method, *args):
            if method == "allocate_state_slot":
                args[0].state_slot = 0
                return 0
            raise AssertionError(f"unexpected state call: {method}")

    old_block_size = Sequence.block_size
    Sequence.block_size = 8
    try:
        config = SimpleNamespace(
            max_num_seqs=1,
            max_num_batched_tokens=32,
            eos=-1,
            kvcache_block_size=8,
            prefix_match_unit=2,
            num_kvcache_blocks=8,
            scheduling_policy="prefill_first",
            enable_prefix_cache=True,
            enable_chunked_prefill=True,
            enable_hybrid_prefix_cache=True,
            enable_hybrid_internal_checkpoints=True,
            hybrid_prefix_checkpoint_interval_blocks=1,
            hybrid_prefix_checkpoint_interval_tokens=8,
            hybrid_prefix_retention_policy="periodic",
            hybrid_prefix_eviction_policy="lru",
        )
        scheduler = Scheduler(
            config,
            state_manager=SimpleNamespace(can_allocate=True),
            state_controller=StateController(),
            prefix_checkpoint_pool=SimpleNamespace(capacity=4),
        )
        seq = Sequence(
            list(range(19)),
            SamplingParams(temperature=0.0, max_tokens=1),
        )
        scheduler.add(seq)

        batch = scheduler._schedule_prefill(32, 1)

        assert batch[0].num_scheduled_tokens == 19
        boundaries = scheduler.internal_prefix_boundaries(batch, True)
        assert boundaries == {seq.seq_id: (8, 16)}
        pending, hashed = scheduler.prepare_prefix_captures(batch, True)
        assert hashed
        assert [item.boundary_tokens for item in pending] == [8, 16]
        assert all(item.internal_state for item in pending)
    finally:
        Sequence.block_size = old_block_size


def test_internal_adaptive_checkpoint_captures_promoted_junction_in_one_prefill():
    class StateController:
        def call(self, method, *args):
            if method == "allocate_state_slot":
                args[0].state_slot = 0
                return 0
            raise AssertionError(f"unexpected state call: {method}")

    old_block_size = Sequence.block_size
    Sequence.block_size = 4
    try:
        config = SimpleNamespace(
            max_num_seqs=1,
            max_num_batched_tokens=32,
            eos=-1,
            kvcache_block_size=4,
            prefix_match_unit=4,
            num_kvcache_blocks=12,
            scheduling_policy="prefill_first",
            enable_prefix_cache=True,
            enable_chunked_prefill=True,
            enable_hybrid_prefix_cache=True,
            enable_hybrid_internal_checkpoints=True,
            hybrid_prefix_checkpoint_interval_blocks=4,
            hybrid_prefix_retention_policy="adaptive",
            hybrid_prefix_eviction_policy="lru",
        )
        scheduler = Scheduler(
            config,
            state_manager=SimpleNamespace(can_allocate=True),
            state_controller=StateController(),
            prefix_checkpoint_pool=SimpleNamespace(capacity=4),
        )
        sampling = SamplingParams(temperature=0.0, max_tokens=1)
        source = Sequence([0, 1, 2, 3, 10, 11, 12, 13, 14], sampling)
        scheduler.block_manager.allocate(source, 0)
        source.num_scheduled_tokens = 8
        scheduler.block_manager.hash_blocks(source)
        scheduler.block_manager.deallocate(source)

        incoming = Sequence([0, 1, 2, 3, 20, 21, 22, 23, 24], sampling)
        scheduler.add(incoming)
        batch = scheduler._schedule_prefill(32, 1)

        assert batch == [incoming]
        assert incoming.num_scheduled_tokens == incoming.num_tokens
        assert scheduler.internal_prefix_boundaries(batch, True) == {
            incoming.seq_id: (4, 8)
        }
        pending, hashed = scheduler.prepare_prefix_captures(batch, True)
        assert hashed
        assert [item.boundary_tokens for item in pending] == [4, 8]
        assert [item.reason for item in pending] == [
            "shared_junction",
            "prompt_tail",
        ]
        assert all(item.internal_state for item in pending)
    finally:
        Sequence.block_size = old_block_size


def test_checkpoint_pool_captures_sparse_internal_tensors():
    manager = _manager(max_num_seqs=1)
    pool = HybridPrefixCheckpointPool(
        manager,
        memory_budget_bytes=manager.bytes_per_slot(),
    )
    recurrent = torch.randn_like(pool.recurrent_checkpoints[0])
    conv = torch.randn_like(pool.conv_checkpoints[0])

    slot = pool.capture_tensors(recurrent, conv)

    torch.testing.assert_close(pool.recurrent_checkpoints[slot], recurrent)
    torch.testing.assert_close(pool.conv_checkpoints[slot], conv)


def test_int8_checkpoint_pool_captures_sparse_internal_tensors():
    manager = _manager(max_num_seqs=1)
    request_slot = manager.allocate()
    pool = HybridPrefixCheckpointPool(
        manager,
        memory_budget_bytes=manager.bytes_per_slot(),
        checkpoint_dtype="int8",
    )
    recurrent = torch.randn_like(manager.recurrent_states[:, request_slot])
    conv = torch.randn_like(manager.conv_states[:, request_slot])

    checkpoint_slot = pool.capture_tensors(recurrent, conv)
    manager.recurrent_states[:, request_slot].zero_()
    manager.conv_states[:, request_slot].zero_()
    pool.restore(checkpoint_slot, request_slot)

    error = (
        manager.recurrent_states[:, request_slot] - recurrent
    ).abs()
    assert torch.all(
        error <= pool.recurrent_scales[checkpoint_slot] / 2 + 1e-6
    )
    torch.testing.assert_close(
        manager.conv_states[:, request_slot], conv, rtol=0, atol=0
    )
