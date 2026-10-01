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
