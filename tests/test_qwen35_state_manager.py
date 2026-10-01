from types import SimpleNamespace

import torch

from nanovllm.engine.state_manager import HybridStateManager


def _manager(max_num_seqs=2):
    return HybridStateManager(
        max_num_seqs=max_num_seqs,
        num_linear_layers=2,
        num_value_heads=2,
        key_head_dim=3,
        value_head_dim=4,
        conv_dim=5,
        conv_kernel_size=4,
        conv_dtype=torch.bfloat16,
        device="cpu",
    )


def test_allocate_and_free_zero_state_before_slot_reuse():
    manager = _manager(max_num_seqs=1)
    first = SimpleNamespace(state_slot=None)
    slot = manager.allocate(first)
    assert slot == 0
    manager.recurrent_states[:, slot].fill_(3.0)
    manager.conv_states[:, slot].fill_(4.0)
    manager.free(first)
    assert first.state_slot is None
    assert not manager.recurrent_states[:, slot].any()
    assert not manager.conv_states[:, slot].any()

    second = SimpleNamespace(state_slot=None)
    assert manager.allocate(second) == slot
    assert not manager.recurrent_states[:, slot].any()
    assert not manager.conv_states[:, slot].any()


def test_scratch_slot_is_reserved_and_never_allocated():
    manager = _manager(max_num_seqs=2)
    slots = {manager.allocate(), manager.allocate()}
    assert slots == {0, 1}
    assert manager.scratch_slot_id == 2
    try:
        manager.allocate()
    except RuntimeError:
        pass
    else:
        raise AssertionError("state manager allocated beyond its real slot capacity")


def test_gather_scatter_maps_full_layer_indices_to_compact_state_pool():
    manager = _manager(max_num_seqs=2)
    slot_a = manager.allocate()
    slot_b = manager.allocate()
    slot_ids = torch.tensor([slot_b, slot_a])
    linear_indices = (0, 2)

    recurrent, conv = manager.gather(
        slot_ids,
        num_total_layers=4,
        linear_layer_indices=linear_indices,
    )
    assert recurrent[1] is None and recurrent[3] is None
    assert conv[1] is None and conv[3] is None
    recurrent[0].fill_(10.0)
    recurrent[2].fill_(20.0)
    conv[0].fill_(30.0)
    conv[2].fill_(40.0)
    manager.scatter(
        slot_ids,
        recurrent,
        conv,
        linear_layer_indices=linear_indices,
    )

    torch.testing.assert_close(
        manager.recurrent_states[0, slot_b],
        torch.full_like(manager.recurrent_states[0, slot_b], 10.0),
    )
    torch.testing.assert_close(
        manager.recurrent_states[1, slot_a],
        torch.full_like(manager.recurrent_states[1, slot_a], 20.0),
    )
    torch.testing.assert_close(
        manager.conv_states[0, slot_b],
        torch.full_like(manager.conv_states[0, slot_b], 30.0),
    )
    torch.testing.assert_close(
        manager.conv_states[1, slot_a],
        torch.full_like(manager.conv_states[1, slot_a], 40.0),
    )


def test_memory_bytes_includes_reserved_scratch_slot():
    manager = _manager(max_num_seqs=2)
    expected = (
        manager.recurrent_states.numel() * manager.recurrent_states.element_size()
        + manager.conv_states.numel() * manager.conv_states.element_size()
    )
    assert manager.memory_bytes() == expected


def test_speculative_snapshots_restore_per_request_accepted_boundary():
    manager = _manager(max_num_seqs=2)
    first_slot = manager.allocate()
    second_slot = manager.allocate()
    slots = torch.tensor([first_slot, second_slot])
    snapshots = [manager.snapshot(slots)]

    manager.recurrent_states[:, first_slot].fill_(10.0)
    manager.recurrent_states[:, second_slot].fill_(11.0)
    manager.conv_states[:, first_slot].fill_(12.0)
    manager.conv_states[:, second_slot].fill_(13.0)
    snapshots.append(manager.snapshot(slots))

    manager.recurrent_states[:, first_slot].fill_(20.0)
    manager.recurrent_states[:, second_slot].fill_(21.0)
    manager.conv_states[:, first_slot].fill_(22.0)
    manager.conv_states[:, second_slot].fill_(23.0)
    snapshots.append(manager.snapshot(slots))

    manager.restore_accepted_prefixes(snapshots, torch.tensor([2, 0]))
    assert torch.all(manager.recurrent_states[:, first_slot] == 20.0)
    assert torch.all(manager.conv_states[:, first_slot] == 22.0)
    assert not manager.recurrent_states[:, second_slot].any()
    assert not manager.conv_states[:, second_slot].any()
