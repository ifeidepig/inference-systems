"""Slow, explicit Qwen3.5 Gated DeltaNet correctness references.

This module is intentionally independent from the production model path.  It
provides a small FP32 golden implementation for validating future prefill,
decode, chunked, and fused kernels.  The formulas follow the Hugging Face
Qwen3.5 reference, but the implementation here is written specifically for
nano-vLLM's tests.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GatedDeltaCoreOutput:
    output: torch.Tensor
    conv_state: torch.Tensor
    recurrent_state: torch.Tensor
    conv_state_history: torch.Tensor | None = None
    recurrent_state_history: torch.Tensor | None = None


def _l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _checkpoint_index_set(
    checkpoint_indices: tuple[int, ...] | None,
    sequence_length: int,
) -> set[int] | None:
    if checkpoint_indices is None:
        return None
    if tuple(sorted(set(checkpoint_indices))) != checkpoint_indices:
        raise ValueError("checkpoint_indices must be sorted and unique")
    if any(index < 0 or index >= sequence_length for index in checkpoint_indices):
        raise ValueError("checkpoint index is outside the sequence")
    return set(checkpoint_indices)


def short_causal_conv_reference(
    projected_qkv: torch.Tensor,
    weight: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    return_state_history: bool = False,
    checkpoint_indices: tuple[int, ...] | None = None,
):
    """Apply depthwise causal convolution one token at a time.

    Args:
        projected_qkv: ``[batch, sequence, conv_dim]`` raw packed Q/K/V.
        weight: ``[conv_dim, kernel_size]`` depthwise convolution weights.
        initial_state: optional ``[batch, conv_dim, kernel_size - 1]``
            projected-input history.

    Returns:
        SiLU-activated convolution output ``[batch, sequence, conv_dim]`` and
        final minimal history state ``[batch, conv_dim, kernel_size - 1]``.
    """
    if projected_qkv.ndim != 3:
        raise ValueError("projected_qkv must have shape [batch, sequence, conv_dim]")
    if weight.ndim != 2:
        raise ValueError("weight must have shape [conv_dim, kernel_size]")

    batch_size, sequence_length, conv_dim = projected_qkv.shape
    checkpoint_set = _checkpoint_index_set(
        checkpoint_indices,
        sequence_length,
    )
    capture_states = return_state_history or checkpoint_set is not None
    if weight.shape[0] != conv_dim:
        raise ValueError(
            f"conv_dim mismatch: input has {conv_dim}, weight has {weight.shape[0]}"
        )
    kernel_size = weight.shape[1]
    if kernel_size < 2:
        raise ValueError("Qwen3.5 short convolution expects kernel_size >= 2")

    x = projected_qkv.float()
    w = weight.float()
    expected_state_shape = (batch_size, conv_dim, kernel_size - 1)
    if initial_state is None:
        state = torch.zeros(expected_state_shape, dtype=torch.float32, device=x.device)
    else:
        if tuple(initial_state.shape) != expected_state_shape:
            raise ValueError(
                f"initial_state must have shape {expected_state_shape}, "
                f"got {tuple(initial_state.shape)}"
            )
        state = initial_state.float().clone()

    outputs = []
    state_history = []
    for token_idx in range(sequence_length):
        current = x[:, token_idx].unsqueeze(-1)
        window = torch.cat((state, current), dim=-1)
        convolved = (window * w.unsqueeze(0)).sum(dim=-1)
        outputs.append(F.silu(convolved))
        state = window[:, :, 1:]
        if capture_states and (
            return_state_history or token_idx in checkpoint_set
        ):
            state_history.append(state.clone())

    if outputs:
        output = torch.stack(outputs, dim=1)
    else:
        output = x.new_empty((batch_size, 0, conv_dim))
    if capture_states:
        history = (
            torch.stack(state_history, dim=1)
            if state_history
            else state[:, None, :, :][:, :0]
        )
        return output, state, history
    return output, state


def recurrent_gated_delta_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    normalize_qk: bool = True,
    return_state_history: bool = False,
    checkpoint_indices: tuple[int, ...] | None = None,
):
    """Run the gated delta rule as an explicit token-by-token FP32 scan.

    Q/K heads must already be repeated to match the number of value heads.

    Shapes:
        query/key: ``[batch, sequence, heads, key_head_dim]``
        value: ``[batch, sequence, heads, value_head_dim]``
        log_decay/beta: ``[batch, sequence, heads]``
        state: ``[batch, heads, key_head_dim, value_head_dim]``
    """
    if query.shape != key.shape or query.ndim != 4:
        raise ValueError("query and key must have the same 4-D shape")
    if value.ndim != 4 or value.shape[:3] != query.shape[:3]:
        raise ValueError("value must match query in batch, sequence, and heads")
    if log_decay.shape != query.shape[:3] or beta.shape != query.shape[:3]:
        raise ValueError("log_decay and beta must have shape [batch, sequence, heads]")

    batch_size, sequence_length, num_heads, key_head_dim = query.shape
    checkpoint_set = _checkpoint_index_set(
        checkpoint_indices,
        sequence_length,
    )
    capture_states = return_state_history or checkpoint_set is not None
    value_head_dim = value.shape[-1]
    state_shape = (batch_size, num_heads, key_head_dim, value_head_dim)
    if initial_state is None:
        state = torch.zeros(state_shape, dtype=torch.float32, device=query.device)
    else:
        if tuple(initial_state.shape) != state_shape:
            raise ValueError(
                f"initial_state must have shape {state_shape}, "
                f"got {tuple(initial_state.shape)}"
            )
        state = initial_state.float().clone()

    q = query.float()
    k = key.float()
    v = value.float()
    decay = log_decay.float()
    update_rate = beta.float()
    if normalize_qk:
        q = _l2_normalize(q)
        k = _l2_normalize(k)
    q = q / (key_head_dim**0.5)

    outputs = []
    state_history = []
    for token_idx in range(sequence_length):
        q_t = q[:, token_idx]
        k_t = k[:, token_idx]
        v_t = v[:, token_idx]

        retain_t = decay[:, token_idx].exp().unsqueeze(-1).unsqueeze(-1)
        state = state * retain_t

        predicted_value = torch.einsum("bhd,bhdv->bhv", k_t, state)
        delta = (v_t - predicted_value) * update_rate[:, token_idx].unsqueeze(-1)
        state = state + torch.einsum("bhd,bhv->bhdv", k_t, delta)
        outputs.append(torch.einsum("bhd,bhdv->bhv", q_t, state))
        if capture_states and (
            return_state_history or token_idx in checkpoint_set
        ):
            state_history.append(state.clone())

    if outputs:
        output = torch.stack(outputs, dim=1)
    else:
        output = v.new_empty((batch_size, 0, num_heads, value_head_dim))
    if capture_states:
        history = (
            torch.stack(state_history, dim=1)
            if state_history
            else state[:, None, ...][:, :0]
        )
        return output, state, history
    return output, state


def gated_delta_core_reference(
    projected_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    num_key_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    conv_state: torch.Tensor | None = None,
    recurrent_state: torch.Tensor | None = None,
    return_state_history: bool = False,
    checkpoint_indices: tuple[int, ...] | None = None,
) -> GatedDeltaCoreOutput:
    """Reference the convolution + recurrent core before z-gated output.

    The inputs are projection results rather than model weights so tests can
    isolate state transitions from Linear/TP/loader behavior.
    """
    if num_value_heads % num_key_heads != 0:
        raise ValueError("num_value_heads must be divisible by num_key_heads")

    key_dim = num_key_heads * key_head_dim
    value_dim = num_value_heads * value_head_dim
    expected_conv_dim = 2 * key_dim + value_dim
    if projected_qkv.shape[-1] != expected_conv_dim:
        raise ValueError(
            f"projected_qkv last dim must be {expected_conv_dim}, "
            f"got {projected_qkv.shape[-1]}"
        )
    expected_gate_shape = (*projected_qkv.shape[:2], num_value_heads)
    if tuple(a.shape) != expected_gate_shape or tuple(b.shape) != expected_gate_shape:
        raise ValueError(f"a and b must have shape {expected_gate_shape}")
    if tuple(a_log.shape) != (num_value_heads,) or tuple(dt_bias.shape) != (
        num_value_heads,
    ):
        raise ValueError("a_log and dt_bias must have shape [num_value_heads]")

    conv_result = short_causal_conv_reference(
        projected_qkv,
        conv_weight,
        conv_state,
        return_state_history=return_state_history,
        checkpoint_indices=checkpoint_indices,
    )
    capture_states = return_state_history or checkpoint_indices is not None
    if capture_states:
        mixed_qkv, final_conv_state, conv_state_history = conv_result
    else:
        mixed_qkv, final_conv_state = conv_result
        conv_state_history = None
    query, key, value = mixed_qkv.split((key_dim, key_dim, value_dim), dim=-1)
    batch_size, sequence_length = projected_qkv.shape[:2]
    query = query.reshape(
        batch_size, sequence_length, num_key_heads, key_head_dim
    )
    key = key.reshape(batch_size, sequence_length, num_key_heads, key_head_dim)
    value = value.reshape(
        batch_size, sequence_length, num_value_heads, value_head_dim
    )

    repetition = num_value_heads // num_key_heads
    if repetition > 1:
        query = query.repeat_interleave(repetition, dim=2)
        key = key.repeat_interleave(repetition, dim=2)

    beta = torch.sigmoid(b.float())
    log_decay = -a_log.float().exp() * F.softplus(a.float() + dt_bias.float())
    recurrent_result = recurrent_gated_delta_reference(
        query,
        key,
        value,
        log_decay,
        beta,
        recurrent_state,
        return_state_history=return_state_history,
        checkpoint_indices=checkpoint_indices,
    )
    if capture_states:
        output, final_recurrent_state, recurrent_state_history = recurrent_result
    else:
        output, final_recurrent_state = recurrent_result
        recurrent_state_history = None
    return GatedDeltaCoreOutput(
        output=output,
        conv_state=final_conv_state,
        recurrent_state=final_recurrent_state,
        conv_state_history=conv_state_history,
        recurrent_state_history=recurrent_state_history,
    )
