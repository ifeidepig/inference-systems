from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


GDNDecodeBackend = Literal["torch", "cuda", "auto"]
SUPPORTED_GDN_DECODE_BACKENDS = frozenset(("torch", "cuda", "auto"))


def validate_gdn_decode_backend(backend: str) -> GDNDecodeBackend:
    normalized = backend.strip().lower()
    if normalized not in SUPPORTED_GDN_DECODE_BACKENDS:
        choices = ", ".join(sorted(SUPPORTED_GDN_DECODE_BACKENDS))
        raise ValueError(
            f"unsupported GDN decode backend {backend!r}; expected one of {choices}"
        )
    return normalized  # type: ignore[return-value]


def gdn_decode_core_torch(
    projected_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_states: torch.Tensor,
    conv_states: torch.Tensor,
    *,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference one-token GDN state transition for a packed decode batch.

    This function is intentionally kept as the differential oracle for the
    custom CUDA backend.  Linear projections, gated RMSNorm, out_proj, and TP
    collectives live outside this contract.
    """
    key_dim = num_key_heads * key_head_dim
    value_dim = num_value_heads * value_head_dim
    conv_dim = 2 * key_dim + value_dim
    if projected_qkv.ndim != 2 or projected_qkv.shape[1] != conv_dim:
        raise ValueError(
            f"projected_qkv must have shape [batch, {conv_dim}]"
        )
    batch_size = projected_qkv.shape[0]
    if tuple(a.shape) != (batch_size, num_value_heads):
        raise ValueError(
            f"a must have shape {(batch_size, num_value_heads)}"
        )
    if tuple(b.shape) != (batch_size, num_value_heads):
        raise ValueError(
            f"b must have shape {(batch_size, num_value_heads)}"
        )
    if conv_weight.ndim != 2 or conv_weight.shape[0] != conv_dim:
        raise ValueError("conv_weight must have shape [conv_dim, kernel_size]")
    kernel_size = conv_weight.shape[1]
    if tuple(conv_states.shape) != (batch_size, conv_dim, kernel_size - 1):
        raise ValueError(
            "conv_states must have shape "
            f"{(batch_size, conv_dim, kernel_size - 1)}"
        )
    expected_recurrent_shape = (
        batch_size,
        num_value_heads,
        key_head_dim,
        value_head_dim,
    )
    if tuple(recurrent_states.shape) != expected_recurrent_shape:
        raise ValueError(
            f"recurrent_states must have shape {expected_recurrent_shape}"
        )
    if tuple(A_log.shape) != (num_value_heads,):
        raise ValueError(f"A_log must have shape {(num_value_heads,)}")
    if tuple(dt_bias.shape) != (num_value_heads,):
        raise ValueError(f"dt_bias must have shape {(num_value_heads,)}")

    window = torch.cat((conv_states, projected_qkv.unsqueeze(-1)), dim=-1)
    mixed_qkv = F.silu(
        (window.float() * conv_weight.float().unsqueeze(0))
        .sum(dim=-1)
        .to(projected_qkv.dtype)
    )
    new_conv_states = window[:, :, 1:]
    query, key, value = mixed_qkv.split(
        (key_dim, key_dim, value_dim),
        dim=-1,
    )
    query = query.reshape(-1, num_key_heads, key_head_dim)
    key = key.reshape(-1, num_key_heads, key_head_dim)
    value = value.reshape(-1, num_value_heads, value_head_dim)
    if num_value_heads % num_key_heads:
        raise ValueError("num_value_heads must be divisible by num_key_heads")
    repetition = num_value_heads // num_key_heads
    if repetition > 1:
        query = query.repeat_interleave(repetition, dim=1)
        key = key.repeat_interleave(repetition, dim=1)

    query_fp32 = query.float()
    key_fp32 = key.float()
    query_fp32 = query_fp32 * torch.rsqrt(
        (query_fp32 * query_fp32).sum(dim=-1, keepdim=True) + 1e-6
    )
    key_fp32 = key_fp32 * torch.rsqrt(
        (key_fp32 * key_fp32).sum(dim=-1, keepdim=True) + 1e-6
    )
    query_fp32 = query_fp32 / (key_head_dim**0.5)

    beta = torch.sigmoid(b.float()).unsqueeze(-1)
    log_decay = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())
    new_recurrent_states = recurrent_states.float() * log_decay.exp()[
        ..., None, None
    ]
    predicted_value = torch.einsum(
        "bhd,bhdv->bhv",
        key_fp32,
        new_recurrent_states,
    )
    delta = (value.float() - predicted_value) * beta
    new_recurrent_states = new_recurrent_states + torch.einsum(
        "bhd,bhv->bhdv",
        key_fp32,
        delta,
    )
    core_output = torch.einsum(
        "bhd,bhdv->bhv",
        query_fp32,
        new_recurrent_states,
    ).to(projected_qkv.dtype)
    return core_output, new_recurrent_states, new_conv_states


def _cuda_contract_reason(
    projected_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_states: torch.Tensor,
    conv_states: torch.Tensor,
    *,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
) -> str | None:
    tensors = (
        projected_qkv,
        a,
        b,
        conv_weight,
        A_log,
        dt_bias,
        recurrent_states,
        conv_states,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        return "all inputs must be CUDA tensors"
    if len({tensor.device for tensor in tensors}) != 1:
        return "all inputs must be on the same CUDA device"
    if projected_qkv.dtype not in (torch.float16, torch.bfloat16):
        return "projected_qkv must use float16 or bfloat16"
    if any(
        tensor.dtype != projected_qkv.dtype
        for tensor in (a, b, conv_weight, conv_states)
    ):
        return "projection, convolution, and conv-state dtypes must match"
    if recurrent_states.dtype != torch.float32:
        return "recurrent state must use float32"
    if A_log.dtype != torch.float32:
        return "A_log must use float32"
    if dt_bias.dtype != projected_qkv.dtype:
        return "dt_bias dtype must match projected_qkv"
    if key_head_dim != 128 or value_head_dim != 128:
        return "CUDA v0 supports key/value head dimensions of 128"
    if num_key_heads != num_value_heads:
        return "CUDA v0 requires equal local key and value head counts"
    if conv_weight.ndim != 2 or conv_weight.shape[1] != 4:
        return "CUDA v0 supports convolution width 4"
    if not all(tensor.is_contiguous() for tensor in tensors):
        return "all CUDA v0 inputs must be contiguous"
    return None


def _cuda_op_available() -> bool:
    try:
        from nanovllm.kernels._extension import load_cuda_ops

        load_cuda_ops()
        getattr(torch.ops.nanovllm, "gdn_decode")
    except (AttributeError, ImportError, OSError, RuntimeError):
        return False
    return True


def gdn_decode_core(
    projected_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_states: torch.Tensor,
    conv_states: torch.Tensor,
    *,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    backend: str = "torch",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch the GDN decode core while retaining a safe Torch fallback."""
    backend = validate_gdn_decode_backend(backend)
    arguments = (
        projected_qkv,
        a,
        b,
        conv_weight,
        A_log,
        dt_bias,
        recurrent_states,
        conv_states,
    )
    dimensions = dict(
        num_key_heads=num_key_heads,
        num_value_heads=num_value_heads,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
    )
    if backend == "torch":
        return gdn_decode_core_torch(*arguments, **dimensions)

    reason = _cuda_contract_reason(*arguments, **dimensions)
    if reason is not None:
        if backend == "auto":
            return gdn_decode_core_torch(*arguments, **dimensions)
        raise RuntimeError(f"GDN CUDA decode backend is unavailable: {reason}")

    if not _cuda_op_available():
        if backend == "auto":
            return gdn_decode_core_torch(*arguments, **dimensions)
        raise RuntimeError(
            "GDN CUDA decode backend was requested, but nanovllm::gdn_decode "
            "is not built"
        )

    return torch.ops.nanovllm.gdn_decode(
        *arguments,
        num_key_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
    )


def gdn_replay_commit_torch_(
    recurrent_states: torch.Tensor,
    slot_ids: torch.Tensor,
    replay_keys: torch.Tensor,
    replay_deltas: torch.Tensor,
    replay_log_decays: torch.Tensor,
    commit_lengths: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    """Fold compact accepted GDN transitions into active state in place."""
    batch_size = slot_ids.numel()
    state = recurrent_states.index_select(1, slot_ids.to(torch.long))
    keys = replay_keys.view(
        replay_keys.shape[0],
        batch_size,
        num_steps,
        *replay_keys.shape[2:],
    ).float()
    deltas = replay_deltas.view(
        replay_deltas.shape[0],
        batch_size,
        num_steps,
        *replay_deltas.shape[2:],
    ).float()
    log_decays = replay_log_decays.view(
        replay_log_decays.shape[0],
        batch_size,
        num_steps,
        *replay_log_decays.shape[2:],
    ).float()
    lengths = commit_lengths.to(device=state.device, dtype=torch.long)
    for step in range(num_steps):
        retain = (
            log_decays[:, :, step].exp().unsqueeze(-1).unsqueeze(-1)
        )
        updated = state * retain + torch.einsum(
            "lbhd,lbhv->lbhdv",
            keys[:, :, step],
            deltas[:, :, step],
        )
        state = torch.where(
            (lengths > step).view(1, batch_size, 1, 1, 1),
            updated,
            state,
        )
    recurrent_states.index_copy_(1, slot_ids.to(torch.long), state)
    return recurrent_states


def gdn_replay_commit_(
    recurrent_states: torch.Tensor,
    slot_ids: torch.Tensor,
    replay_keys: torch.Tensor,
    replay_deltas: torch.Tensor,
    replay_log_decays: torch.Tensor,
    commit_lengths: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    """Use the fused CUDA replay fold when available, else the Torch oracle."""
    if recurrent_states.is_cuda and _cuda_op_available():
        try:
            op = getattr(torch.ops.nanovllm, "gdn_replay_commit")
        except AttributeError:
            pass
        else:
            return op(
                recurrent_states,
                slot_ids.to(device=recurrent_states.device, dtype=torch.long),
                replay_keys.contiguous(),
                replay_deltas.contiguous(),
                replay_log_decays.contiguous(),
                commit_lengths.to(
                    device=recurrent_states.device,
                    dtype=torch.long,
                ),
                num_steps,
            )
    return gdn_replay_commit_torch_(
        recurrent_states,
        slot_ids,
        replay_keys,
        replay_deltas,
        replay_log_decays,
        commit_lengths,
        num_steps,
    )


def gdn_conv_commit_(
    conv_states: torch.Tensor,
    conv_history: torch.Tensor,
    slot_ids: torch.Tensor,
    state_boundaries: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    """Select and commit one compact conv window per request."""
    slots = slot_ids.to(device=conv_states.device, dtype=torch.long)
    boundaries = state_boundaries.to(
        device=conv_states.device,
        dtype=torch.long,
    )
    if conv_states.is_cuda and _cuda_op_available():
        try:
            op = getattr(torch.ops.nanovllm, "gdn_conv_commit")
        except AttributeError:
            pass
        else:
            return op(
                conv_states,
                conv_history.contiguous(),
                slots,
                boundaries,
                num_steps,
            )
    batch_size = slots.numel()
    history_indices = (
        torch.arange(batch_size, device=conv_states.device) * num_steps
        + boundaries
    )
    selected = conv_history.index_select(1, history_indices)
    conv_states.index_copy_(1, slots, selected)
    return conv_states
