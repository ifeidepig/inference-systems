from types import SimpleNamespace

import torch

from nanovllm import SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.state_manager import HybridStateManager


def _manager(max_num_seqs=2):
    return HybridStateManager(
        max_num_seqs=max_num_seqs,
        num_linear_layers=1,
        num_value_heads=1,
        key_head_dim=2,
        value_head_dim=2,
        conv_dim=4,
        conv_kernel_size=4,
        conv_dtype=torch.bfloat16,
        device="cpu",
    )


def _scheduler(manager, token_budget=256, num_speculative_tokens=0):
    config = SimpleNamespace(
        max_num_seqs=2,
        max_num_batched_tokens=token_budget,
        eos=0,
        kvcache_block_size=256,
        num_kvcache_blocks=8,
        scheduling_policy="prefill_first",
        enable_prefix_cache=True,
        enable_chunked_prefill=True,
        scheduler_target_ttft_ms=200.0,
        scheduler_target_tpot_ms=50.0,
        slo_prefill_priority_threshold=0.8,
        slo_min_prefill_tokens=64,
        slo_kv_pressure_threshold=0.9,
        slo_queue_pressure_threshold=3,
        slo_latency_safety_margin_ms=5.0,
        scheduler_ewma_alpha=0.2,
        request_metrics_history_size=16,
        num_speculative_tokens=num_speculative_tokens,
    )
    return Scheduler(config, state_manager=manager)


def _sequence(prompt_length=1, max_tokens=2):
    return Sequence(
        [1] * prompt_length,
        SamplingParams(temperature=0.1, max_tokens=max_tokens, ignore_eos=True),
    )


def test_hybrid_scheduler_allocates_one_slot_and_disables_prefix_cache():
    manager = _manager()
    scheduler = _scheduler(manager)
    sequence = _sequence(prompt_length=300)
    scheduler.add(sequence)

    first_batch = scheduler.schedule()[0]
    slot = sequence.state_slot
    assert slot is not None
    assert slot in manager.used_slot_ids
    assert not scheduler.enable_prefix_cache
    scheduler.postprocess(first_batch.seqs, [9], is_prefill=True)

    second_batch = scheduler.schedule()[0]
    assert sequence.state_slot == slot
    scheduler.postprocess(second_batch.seqs, [9], is_prefill=True)
    assert sequence.state_slot == slot


def test_preemption_frees_and_zeros_hybrid_state_for_recompute():
    manager = _manager(max_num_seqs=1)
    scheduler = _scheduler(manager)
    sequence = _sequence()
    scheduler.add(sequence)
    scheduler.schedule()
    slot = sequence.state_slot
    manager.recurrent_states[:, slot].fill_(5.0)
    manager.conv_states[:, slot].fill_(6.0)

    scheduler.preempt(sequence)
    assert sequence.state_slot is None
    assert sequence.num_cached_tokens == 0
    assert sequence in scheduler.waiting
    assert not manager.recurrent_states[:, slot].any()
    assert not manager.conv_states[:, slot].any()


def test_abort_and_finish_release_hybrid_state():
    manager = _manager(max_num_seqs=1)
    scheduler = _scheduler(manager)
    aborted = _sequence()
    scheduler.add(aborted)
    scheduler.schedule()
    aborted_slot = aborted.state_slot
    assert scheduler.abort(aborted.seq_id)
    assert aborted.state_slot is None
    assert aborted_slot in manager.free_slot_ids

    finished = _sequence(max_tokens=1)
    scheduler.add(finished)
    batch = scheduler.schedule()[0]
    finished_slot = finished.state_slot
    scheduler.postprocess(batch.seqs, [9], is_prefill=True)
    assert finished.is_finished
    assert finished.state_slot is None
    assert finished_slot in manager.free_slot_ids


def test_speculative_postprocess_keeps_one_pending_token_and_finishes():
    manager = _manager(max_num_seqs=1)
    scheduler = _scheduler(manager, num_speculative_tokens=2)
    sequence = _sequence(max_tokens=5)
    scheduler.add(sequence)
    prefill = scheduler.schedule()[0]
    scheduler.postprocess(prefill.seqs, [9], is_prefill=True)

    decode = scheduler.schedule()[0]
    scheduler.postprocess_speculative(decode.seqs, [[10, 11, 12]])
    assert sequence.completion_token_ids == [9, 10, 11, 12]
    assert sequence.num_cached_tokens == len(sequence) - 1
    assert not sequence.is_finished

    decode = scheduler.schedule()[0]
    slot = sequence.state_slot
    scheduler.postprocess_speculative(decode.seqs, [[13, 14]])
    assert sequence.completion_token_ids == [9, 10, 11, 12, 13]
    assert sequence.is_finished
    assert sequence.state_slot is None
    assert slot in manager.free_slot_ids


def test_speculative_decode_reserves_kv_block_across_boundary():
    manager = _manager(max_num_seqs=1)
    scheduler = _scheduler(
        manager,
        token_budget=256,
        num_speculative_tokens=2,
    )
    sequence = _sequence(prompt_length=256, max_tokens=8)
    scheduler.add(sequence)
    prefill = scheduler.schedule()[0]
    scheduler.postprocess(prefill.seqs, [9], is_prefill=True)
    assert len(sequence.block_table) == 1

    scheduler.schedule()
    assert len(sequence.block_table) == 2
