import pytest
import torch

from nanovllm.kernels._extension import load_cuda_ops
from nanovllm.kernels.gdn import gdn_replay_commit_torch_


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("num_steps", [1, 2, 4])
def test_gdn_replay_commit_cuda_matches_torch(batch_size, num_steps):
    if not torch.cuda.is_available():
        return
    load_cuda_ops()
    torch.manual_seed(31 + batch_size * 7 + num_steps)
    num_slots = batch_size + 3
    num_layers = 2
    num_heads = 4
    key_dim = 16
    value_dim = 16
    slot_ids = torch.arange(
        1,
        batch_size + 1,
        dtype=torch.long,
        device="cuda",
    )
    commit_lengths = torch.arange(
        1,
        batch_size + 1,
        dtype=torch.long,
        device="cuda",
    ).remainder(num_steps) + 1
    state = torch.randn(
        num_layers,
        num_slots,
        num_heads,
        key_dim,
        value_dim,
        device="cuda",
    )
    keys = torch.randn(
        num_layers,
        batch_size * num_steps,
        num_heads,
        key_dim,
        device="cuda",
    )
    deltas = torch.randn(
        num_layers,
        batch_size * num_steps,
        num_heads,
        value_dim,
        device="cuda",
    )
    log_decays = -torch.rand(
        num_layers,
        batch_size * num_steps,
        num_heads,
        device="cuda",
    )

    expected = state.clone()
    gdn_replay_commit_torch_(
        expected,
        slot_ids,
        keys,
        deltas,
        log_decays,
        commit_lengths,
        num_steps,
    )
    actual = state.clone()
    torch.ops.nanovllm.gdn_replay_commit(
        actual,
        slot_ids,
        keys,
        deltas,
        log_decays,
        commit_lengths,
        num_steps,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("num_steps", [1, 2, 4])
def test_gdn_conv_commit_cuda_selects_variable_boundaries(num_steps):
    if not torch.cuda.is_available():
        return
    load_cuda_ops()
    num_layers = 2
    batch_size = 3
    num_slots = 6
    channels = 20
    window = 3
    slot_ids = torch.tensor([1, 3, 4], device="cuda")
    boundaries = torch.arange(batch_size, device="cuda").remainder(num_steps)
    conv_states = torch.randn(
        num_layers,
        num_slots,
        channels,
        window,
        dtype=torch.bfloat16,
        device="cuda",
    )
    history = torch.randn(
        num_layers,
        batch_size * num_steps,
        channels,
        window,
        dtype=torch.bfloat16,
        device="cuda",
    )
    indices = torch.arange(batch_size, device="cuda") * num_steps + boundaries
    expected = conv_states.clone()
    expected.index_copy_(1, slot_ids, history.index_select(1, indices))
    actual = conv_states.clone()
    torch.ops.nanovllm.gdn_conv_commit(
        actual,
        history,
        slot_ids,
        boundaries,
        num_steps,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
