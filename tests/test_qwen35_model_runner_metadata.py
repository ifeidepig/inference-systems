import torch

from nanovllm import SamplingParams
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import get_context, reset_context


def _runner():
    runner = ModelRunner.__new__(ModelRunner)
    runner.block_size = 256
    # prepare_* only needs presence/absence; state storage is tested separately.
    runner.state_manager = object()
    return runner


def _sequence():
    sequence = Sequence(
        [11, 12],
        SamplingParams(temperature=0.1, max_tokens=2, ignore_eos=True),
    )
    sequence.block_table = [3]
    sequence.state_slot = 5
    return sequence


def test_prepare_prefill_carries_state_slot_and_query_boundaries():
    runner = _runner()
    sequence = _sequence()
    sequence.num_scheduled_tokens = 2
    try:
        input_ids, positions = runner.prepare_prefill([sequence])
        context = get_context()
        assert input_ids.tolist() == [11, 12]
        assert positions.tolist() == [0, 1]
        assert context.cu_seqlens_q.tolist() == [0, 2]
        assert context.state_slot_ids.tolist() == [5]
    finally:
        reset_context()


def test_prepare_decode_carries_one_query_per_sequence_and_state_slot():
    runner = _runner()
    first = _sequence()
    second = _sequence()
    second.last_token = 21
    second.token_ids[-1] = 21
    second.block_table = [4]
    second.state_slot = 7
    try:
        input_ids, positions = runner.prepare_decode([first, second])
        context = get_context()
        assert input_ids.tolist() == [12, 21]
        assert positions.tolist() == [1, 1]
        assert context.cu_seqlens_q.tolist() == [0, 1, 2]
        assert context.state_slot_ids.tolist() == [5, 7]
        assert context.slot_mapping.tolist() == [3 * 256 + 1, 4 * 256 + 1]
    finally:
        reset_context()
