from collections import deque
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True, slots=True)
class HybridStateSnapshot:
    slot_ids: torch.Tensor
    recurrent_states: torch.Tensor
    conv_states: torch.Tensor


class HybridStateManager:
    """Central fixed-size state pool for Qwen3.5 linear-attention layers."""

    def __init__(
        self,
        *,
        max_num_seqs: int,
        num_linear_layers: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_dim: int,
        conv_kernel_size: int,
        conv_dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        if max_num_seqs <= 0 or num_linear_layers <= 0:
            raise ValueError("state pool dimensions must be positive")
        if conv_kernel_size < 2:
            raise ValueError("conv_kernel_size must be at least two")

        self.max_num_seqs = max_num_seqs
        self.num_linear_layers = num_linear_layers
        self.scratch_slot_id = max_num_seqs
        num_slots_with_scratch = max_num_seqs + 1
        self.recurrent_states = torch.zeros(
            num_linear_layers,
            num_slots_with_scratch,
            num_value_heads,
            key_head_dim,
            value_head_dim,
            dtype=torch.float32,
            device=device,
        )
        self.conv_states = torch.zeros(
            num_linear_layers,
            num_slots_with_scratch,
            conv_dim,
            conv_kernel_size - 1,
            dtype=conv_dtype,
            device=device,
        )
        self.free_slot_ids: deque[int] = deque(range(max_num_seqs))
        self.used_slot_ids: set[int] = set()

    @property
    def can_allocate(self) -> bool:
        return bool(self.free_slot_ids)

    def _zero_slot(self, slot_id: int) -> None:
        self.recurrent_states[:, slot_id].zero_()
        self.conv_states[:, slot_id].zero_()

    def allocate(self, sequence=None) -> int:
        if not self.free_slot_ids:
            raise RuntimeError("no free Qwen3.5 state slots")
        if sequence is not None and getattr(sequence, "state_slot", None) is not None:
            raise ValueError("sequence already owns a state slot")
        slot_id = self.free_slot_ids.popleft()
        self._zero_slot(slot_id)
        self.used_slot_ids.add(slot_id)
        if sequence is not None:
            sequence.state_slot = slot_id
        return slot_id

    def free(self, sequence_or_slot) -> None:
        if isinstance(sequence_or_slot, int):
            slot_id = sequence_or_slot
            sequence = None
        else:
            sequence = sequence_or_slot
            slot_id = getattr(sequence, "state_slot", None)
            if slot_id is None:
                return
        if slot_id not in self.used_slot_ids:
            raise ValueError(f"state slot {slot_id} is not allocated")
        self._zero_slot(slot_id)
        self.used_slot_ids.remove(slot_id)
        self.free_slot_ids.append(slot_id)
        if sequence is not None:
            sequence.state_slot = None

    def gather(
        self,
        slot_ids: torch.Tensor,
        *,
        num_total_layers: int,
        linear_layer_indices: tuple[int, ...] | list[int],
    ) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None]]:
        if len(linear_layer_indices) != self.num_linear_layers:
            raise ValueError("linear layer index count does not match state pool")
        slot_ids = slot_ids.to(device=self.recurrent_states.device, dtype=torch.long)
        recurrent = self.recurrent_states.index_select(1, slot_ids)
        conv = self.conv_states.index_select(1, slot_ids)
        recurrent_by_layer: list[torch.Tensor | None] = [None] * num_total_layers
        conv_by_layer: list[torch.Tensor | None] = [None] * num_total_layers
        for compact_index, layer_index in enumerate(linear_layer_indices):
            recurrent_by_layer[layer_index] = recurrent[compact_index]
            conv_by_layer[layer_index] = conv[compact_index]
        return recurrent_by_layer, conv_by_layer

    def scatter(
        self,
        slot_ids: torch.Tensor,
        recurrent_by_layer: list[torch.Tensor | None],
        conv_by_layer: list[torch.Tensor | None],
        *,
        linear_layer_indices: tuple[int, ...] | list[int],
    ) -> None:
        if len(linear_layer_indices) != self.num_linear_layers:
            raise ValueError("linear layer index count does not match state pool")
        slot_ids = slot_ids.to(device=self.recurrent_states.device, dtype=torch.long)
        for compact_index, layer_index in enumerate(linear_layer_indices):
            recurrent = recurrent_by_layer[layer_index]
            conv = conv_by_layer[layer_index]
            if recurrent is None or conv is None:
                raise ValueError(f"missing updated state for linear layer {layer_index}")
            self.recurrent_states[compact_index].index_copy_(
                0,
                slot_ids,
                recurrent.to(self.recurrent_states.dtype),
            )
            self.conv_states[compact_index].index_copy_(
                0,
                slot_ids,
                conv.to(self.conv_states.dtype),
            )

    def reset_scratch(self) -> None:
        self._zero_slot(self.scratch_slot_id)

    def snapshot(self, slot_ids: torch.Tensor) -> HybridStateSnapshot:
        slot_ids = slot_ids.to(device=self.recurrent_states.device, dtype=torch.long)
        return HybridStateSnapshot(
            slot_ids=slot_ids.clone(),
            recurrent_states=self.recurrent_states.index_select(1, slot_ids).clone(),
            conv_states=self.conv_states.index_select(1, slot_ids).clone(),
        )

    def restore(self, snapshot: HybridStateSnapshot) -> None:
        slot_ids = snapshot.slot_ids.to(
            device=self.recurrent_states.device,
            dtype=torch.long,
        )
        self.recurrent_states.index_copy_(
            1,
            slot_ids,
            snapshot.recurrent_states.to(self.recurrent_states.dtype),
        )
        self.conv_states.index_copy_(
            1,
            slot_ids,
            snapshot.conv_states.to(self.conv_states.dtype),
        )

    def restore_accepted_prefixes(
        self,
        snapshots: list[HybridStateSnapshot],
        accepted_lengths: torch.Tensor,
    ) -> None:
        """Restore each request from its own accepted draft boundary.

        ``snapshots[0]`` is the state before drafting and ``snapshots[i]`` is
        the state after ``i`` draft tokens.
        """
        if not snapshots:
            raise ValueError("at least the pre-draft snapshot is required")
        base_slots = snapshots[0].slot_ids
        batch_size = base_slots.numel()
        if accepted_lengths.numel() != batch_size:
            raise ValueError("accepted_lengths must have one value per snapshotted slot")
        for snapshot in snapshots:
            if not torch.equal(snapshot.slot_ids, base_slots):
                raise ValueError("all speculative snapshots must use the same slot order")

        accepted_cpu = accepted_lengths.to("cpu", dtype=torch.long).tolist()
        selected_recurrent = []
        selected_conv = []
        for batch_index, accepted in enumerate(accepted_cpu):
            if accepted < 0 or accepted >= len(snapshots):
                raise ValueError(
                    f"accepted length {accepted} has no matching state snapshot"
                )
            selected_recurrent.append(
                snapshots[accepted].recurrent_states[:, batch_index : batch_index + 1]
            )
            selected_conv.append(
                snapshots[accepted].conv_states[:, batch_index : batch_index + 1]
            )
        self.restore(
            HybridStateSnapshot(
                slot_ids=base_slots,
                recurrent_states=torch.cat(selected_recurrent, dim=1),
                conv_states=torch.cat(selected_conv, dim=1),
            )
        )

    def memory_bytes(self) -> int:
        return (
            self.recurrent_states.numel() * self.recurrent_states.element_size()
            + self.conv_states.numel() * self.conv_states.element_size()
        )

    def bytes_per_slot(self) -> int:
        return (
            self.recurrent_states[:, 0].numel()
            * self.recurrent_states.element_size()
            + self.conv_states[:, 0].numel()
            * self.conv_states.element_size()
        )


class HybridPrefixCheckpointPool:
    """Fixed-budget immutable copies of aligned Hybrid request states."""

    _SUPPORTED_DTYPES = ("fp32", "bf16", "int8")

    def __init__(
        self,
        state_manager: HybridStateManager,
        memory_budget_bytes: int,
        checkpoint_dtype: str = "fp32",
    ) -> None:
        if checkpoint_dtype not in self._SUPPORTED_DTYPES:
            raise ValueError(
                "checkpoint dtype must be one of "
                f"{self._SUPPORTED_DTYPES}, got {checkpoint_dtype}"
            )
        self.state_manager = state_manager
        self.checkpoint_dtype = checkpoint_dtype
        recurrent_shape = state_manager.recurrent_states.shape
        conv_shape = state_manager.conv_states.shape
        self.recurrent_shape = (
            recurrent_shape[0],
            *recurrent_shape[2:],
        )
        self.conv_shape = (conv_shape[0], *conv_shape[2:])
        recurrent_elements = state_manager.recurrent_states[:, 0].numel()
        conv_bytes = (
            state_manager.conv_states[:, 0].numel()
            * state_manager.conv_states.element_size()
        )
        self.scale_shape = None
        scale_bytes = 0
        if checkpoint_dtype == "fp32":
            recurrent_storage_dtype = torch.float32
            recurrent_bytes = recurrent_elements * 4
        elif checkpoint_dtype == "bf16":
            recurrent_storage_dtype = torch.bfloat16
            recurrent_bytes = recurrent_elements * 2
        else:
            recurrent_storage_dtype = torch.int8
            recurrent_bytes = recurrent_elements
            # Active recurrent layout is [L, slot, H, K, V]. Quantize over
            # V so every (layer, head, key-channel) owns one FP32 scale.
            self.scale_shape = (
                self.recurrent_shape[0],
                self.recurrent_shape[1],
                self.recurrent_shape[2],
                1,
            )
            scale_elements = 1
            for size in self.scale_shape:
                scale_elements *= size
            scale_bytes = scale_elements * 4
        self.recurrent_bytes_per_checkpoint = recurrent_bytes
        self.scale_bytes_per_checkpoint = scale_bytes
        self.conv_bytes_per_checkpoint = conv_bytes
        self.bytes_per_checkpoint = recurrent_bytes + scale_bytes + conv_bytes
        self.capacity = memory_budget_bytes // self.bytes_per_checkpoint
        if self.capacity <= 0:
            raise ValueError(
                "hybrid prefix checkpoint budget is smaller than one checkpoint: "
                f"budget={memory_budget_bytes}, "
                f"required={self.bytes_per_checkpoint}"
            )
        self.recurrent_checkpoints = torch.zeros(
            self.capacity,
            *self.recurrent_shape,
            dtype=recurrent_storage_dtype,
            device=state_manager.recurrent_states.device,
        )
        self.recurrent_scales = (
            torch.zeros(
                self.capacity,
                *self.scale_shape,
                dtype=torch.float32,
                device=state_manager.recurrent_states.device,
            )
            if self.scale_shape is not None
            else None
        )
        self.conv_checkpoints = torch.zeros(
            self.capacity,
            *self.conv_shape,
            dtype=state_manager.conv_states.dtype,
            device=state_manager.conv_states.device,
        )
        self.free_checkpoint_slots: deque[int] = deque(range(self.capacity))
        self.used_checkpoint_slots: set[int] = set()

    @property
    def can_allocate(self) -> bool:
        return bool(self.free_checkpoint_slots)

    def _store_recurrent(
        self,
        checkpoint_slot: int,
        recurrent_state: torch.Tensor,
    ) -> None:
        if self.checkpoint_dtype != "int8":
            self.recurrent_checkpoints[checkpoint_slot].copy_(recurrent_state)
            return
        state_fp32 = recurrent_state.to(torch.float32)
        amax = state_fp32.abs().amax(dim=-1, keepdim=True)
        scale = (amax / 127.0).clamp(min=1e-8)
        quantized = torch.round(state_fp32 / scale).clamp(-127, 127)
        self.recurrent_checkpoints[checkpoint_slot].copy_(
            quantized.to(torch.int8)
        )
        self.recurrent_scales[checkpoint_slot].copy_(scale)

    def _restore_recurrent(
        self,
        checkpoint_slot: int,
        request_state_slot: int,
    ) -> None:
        destination = self.state_manager.recurrent_states[:, request_state_slot]
        if self.checkpoint_dtype != "int8":
            destination.copy_(self.recurrent_checkpoints[checkpoint_slot])
            return
        restored = (
            self.recurrent_checkpoints[checkpoint_slot].to(torch.float32)
            * self.recurrent_scales[checkpoint_slot]
        )
        destination.copy_(restored)

    def _zero_checkpoint(self, checkpoint_slot: int) -> None:
        self.recurrent_checkpoints[checkpoint_slot].zero_()
        if self.recurrent_scales is not None:
            self.recurrent_scales[checkpoint_slot].zero_()
        self.conv_checkpoints[checkpoint_slot].zero_()

    def capture(self, request_state_slot: int) -> int:
        if request_state_slot not in self.state_manager.used_slot_ids:
            raise ValueError("cannot checkpoint an unallocated request state slot")
        if not self.free_checkpoint_slots:
            raise RuntimeError("no free hybrid prefix checkpoint slots")
        checkpoint_slot = self.free_checkpoint_slots.popleft()
        try:
            self._store_recurrent(
                checkpoint_slot,
                self.state_manager.recurrent_states[:, request_state_slot]
            )
            self.conv_checkpoints[checkpoint_slot].copy_(
                self.state_manager.conv_states[:, request_state_slot]
            )
        except Exception:
            self._zero_checkpoint(checkpoint_slot)
            self.free_checkpoint_slots.appendleft(checkpoint_slot)
            raise
        self.used_checkpoint_slots.add(checkpoint_slot)
        return checkpoint_slot

    def capture_tensors(
        self,
        recurrent_state: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> int:
        expected_recurrent = self.recurrent_shape
        expected_conv = self.conv_shape
        if tuple(recurrent_state.shape) != tuple(expected_recurrent):
            raise ValueError(
                f"internal recurrent checkpoint must have shape {expected_recurrent}"
            )
        if tuple(conv_state.shape) != tuple(expected_conv):
            raise ValueError(
                f"internal conv checkpoint must have shape {expected_conv}"
            )
        if not self.free_checkpoint_slots:
            raise RuntimeError("no free hybrid prefix checkpoint slots")
        checkpoint_slot = self.free_checkpoint_slots.popleft()
        try:
            self._store_recurrent(checkpoint_slot, recurrent_state)
            self.conv_checkpoints[checkpoint_slot].copy_(conv_state)
        except Exception:
            self._zero_checkpoint(checkpoint_slot)
            self.free_checkpoint_slots.appendleft(checkpoint_slot)
            raise
        self.used_checkpoint_slots.add(checkpoint_slot)
        return checkpoint_slot

    def restore(self, checkpoint_slot: int, request_state_slot: int) -> None:
        if checkpoint_slot not in self.used_checkpoint_slots:
            raise ValueError("hybrid prefix checkpoint slot is not allocated")
        if request_state_slot not in self.state_manager.used_slot_ids:
            raise ValueError("request state slot is not allocated")
        self._restore_recurrent(checkpoint_slot, request_state_slot)
        self.state_manager.conv_states[:, request_state_slot].copy_(
            self.conv_checkpoints[checkpoint_slot]
        )

    def free(self, checkpoint_slot: int) -> None:
        if checkpoint_slot not in self.used_checkpoint_slots:
            raise ValueError("hybrid prefix checkpoint slot is not allocated")
        self._zero_checkpoint(checkpoint_slot)
        self.used_checkpoint_slots.remove(checkpoint_slot)
        self.free_checkpoint_slots.append(checkpoint_slot)

    def rollback_capture(self, checkpoint_slot: int) -> None:
        """Undo an unpublished allocation and restore deterministic order."""
        if checkpoint_slot not in self.used_checkpoint_slots:
            raise ValueError("hybrid prefix checkpoint slot is not allocated")
        self._zero_checkpoint(checkpoint_slot)
        self.used_checkpoint_slots.remove(checkpoint_slot)
        free_slots = [*self.free_checkpoint_slots, checkpoint_slot]
        self.free_checkpoint_slots = deque(sorted(free_slots))

    def memory_bytes(self) -> int:
        return (
            self.recurrent_checkpoints.numel()
            * self.recurrent_checkpoints.element_size()
            + (
                self.recurrent_scales.numel()
                * self.recurrent_scales.element_size()
                if self.recurrent_scales is not None
                else 0
            )
            + self.conv_checkpoints.numel()
            * self.conv_checkpoints.element_size()
        )


def _transaction_device(reference: torch.Tensor) -> torch.device:
    if dist.is_initialized() and dist.get_backend() == "nccl":
        return reference.device
    return torch.device("cpu")


def transactional_capture_prefix_checkpoint(
    pool: HybridPrefixCheckpointPool,
    request_state_slot: int,
    *,
    recurrent_state: torch.Tensor | None = None,
    conv_state: torch.Tensor | None = None,
) -> int:
    """All-rank prepare/capture with rollback before scheduler publication."""
    checkpoint_slot = None
    local_error = None
    try:
        if recurrent_state is None and conv_state is None:
            checkpoint_slot = pool.capture(request_state_slot)
        elif recurrent_state is not None and conv_state is not None:
            checkpoint_slot = pool.capture_tensors(recurrent_state, conv_state)
        else:
            raise ValueError("both internal checkpoint tensors are required")
    except Exception as exc:  # participate in consensus before raising
        local_error = exc

    if not dist.is_initialized() or dist.get_world_size() == 1:
        if local_error is not None:
            raise local_error
        return checkpoint_slot

    device = _transaction_device(pool.recurrent_checkpoints)
    success = torch.tensor(
        0 if local_error is not None else 1,
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(success, op=dist.ReduceOp.MIN)
    if int(success.item()) == 0:
        if checkpoint_slot is not None:
            pool.rollback_capture(checkpoint_slot)
        if dist.get_rank() == 0:
            raise RuntimeError("hybrid prefix capture rolled back on all ranks")
        return -1

    slot = torch.tensor(checkpoint_slot, dtype=torch.int64, device=device)
    minimum = slot.clone()
    maximum = slot.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if int(minimum.item()) != int(maximum.item()):
        pool.rollback_capture(checkpoint_slot)
        if dist.get_rank() == 0:
            raise RuntimeError(
                "hybrid prefix checkpoint slot IDs diverged across ranks"
            )
        return -1
    return checkpoint_slot


def transactional_restore_prefix_checkpoint(
    pool: HybridPrefixCheckpointPool,
    checkpoint_slot: int,
    request_state_slot: int,
) -> None:
    """Restore all local shards or roll every rank back to its prior state."""
    manager = pool.state_manager
    recurrent_backup = manager.recurrent_states[:, request_state_slot].clone()
    conv_backup = manager.conv_states[:, request_state_slot].clone()
    local_error = None
    try:
        pool.restore(checkpoint_slot, request_state_slot)
    except Exception as exc:  # participate in consensus before raising
        local_error = exc

    if dist.is_initialized() and dist.get_world_size() > 1:
        device = _transaction_device(pool.recurrent_checkpoints)
        success = torch.tensor(
            0 if local_error is not None else 1,
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(success, op=dist.ReduceOp.MIN)
        global_success = int(success.item()) == 1
    else:
        global_success = local_error is None

    if not global_success:
        manager.recurrent_states[:, request_state_slot].copy_(recurrent_backup)
        manager.conv_states[:, request_state_slot].copy_(conv_backup)
        if not dist.is_initialized() or dist.get_rank() == 0:
            raise RuntimeError("hybrid prefix restore rolled back on all ranks")
