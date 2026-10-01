"""Qwen3.5 Gated DeltaNet layer.

The first implementation deliberately uses the slow FP32 reference core.  It
establishes projection, ragged-batch, state, and output-gating semantics before
any Triton or fused backend is introduced.
"""

import os

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.layernorm import Qwen3_5RMSNormGated
from nanovllm.kernels.gdn import gdn_decode_core, validate_gdn_decode_backend
from nanovllm.models.qwen35_reference import gated_delta_core_reference


class GatedDeltaNet(nn.Module):

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.tp_size = dist.get_world_size() if dist.is_initialized() else 1
        self.tp_rank = dist.get_rank() if dist.is_initialized() else 0
        self.total_num_key_heads = config.linear_num_key_heads
        self.total_num_value_heads = config.linear_num_value_heads
        if (
            self.total_num_key_heads % self.tp_size
            or self.total_num_value_heads % self.tp_size
        ):
            raise ValueError("GDN head counts must be divisible by tensor parallel size")
        self.num_key_heads = self.total_num_key_heads // self.tp_size
        self.num_value_heads = self.total_num_value_heads // self.tp_size
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.decode_backend = validate_gdn_decode_backend(
            getattr(
                config,
                "gdn_decode_backend",
                os.environ.get("NANOVLLM_GDN_DECODE_BACKEND", "torch"),
            )
        )

        if self.num_value_heads % self.num_key_heads:
            raise ValueError("linear_num_value_heads must divide by linear_num_key_heads")

        self.key_dim = self.num_key_heads * self.key_head_dim
        self.value_dim = self.num_value_heads * self.value_head_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_value_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_value_heads, bias=False)
        self.in_proj_qkv.weight.weight_loader = self._qkv_weight_loader
        self.in_proj_z.weight.weight_loader = self._value_projection_weight_loader
        self.in_proj_b.weight.weight_loader = self._value_head_weight_loader
        self.in_proj_a.weight.weight_loader = self._value_head_weight_loader
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
        )
        self.conv1d.weight.weight_loader = self._qkv_weight_loader
        # Preserve the checkpoint's case-sensitive parameter name.
        # Strict checkpoint loading overwrites this parameter in production.
        # Use a deterministic finite value so standalone layer/reference tests
        # never depend on uninitialized allocator contents.
        self.A_log = nn.Parameter(
            torch.zeros(self.num_value_heads, dtype=torch.float32)
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_value_heads))
        self.A_log.weight_loader = self._value_head_weight_loader
        self.dt_bias.weight_loader = self._value_head_weight_loader
        self.norm = Qwen3_5RMSNormGated(
            self.value_head_dim,
            eps=config.rms_norm_eps,
        )
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self.out_proj.weight.weight_loader = self._out_proj_weight_loader

    def _qkv_weight_loader(self, parameter, loaded_weight):
        total_key_dim = self.total_num_key_heads * self.key_head_dim
        total_value_dim = self.total_num_value_heads * self.value_head_dim
        q_start = self.tp_rank * self.key_dim
        k_start = total_key_dim + self.tp_rank * self.key_dim
        v_start = 2 * total_key_dim + self.tp_rank * self.value_dim
        q = loaded_weight.narrow(0, q_start, self.key_dim)
        k = loaded_weight.narrow(0, k_start, self.key_dim)
        v = loaded_weight.narrow(0, v_start, self.value_dim)
        parameter.data.copy_(torch.cat((q, k, v), dim=0))

    def _value_head_weight_loader(self, parameter, loaded_weight):
        start = self.tp_rank * self.num_value_heads
        parameter.data.copy_(loaded_weight.narrow(0, start, self.num_value_heads))

    def _value_projection_weight_loader(self, parameter, loaded_weight):
        start = self.tp_rank * self.value_dim
        parameter.data.copy_(loaded_weight.narrow(0, start, self.value_dim))

    def _out_proj_weight_loader(self, parameter, loaded_weight):
        start = self.tp_rank * self.value_dim
        parameter.data.copy_(loaded_weight.narrow(1, start, self.value_dim))

    def _validate_inputs(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        recurrent_states: torch.Tensor | None,
        conv_states: torch.Tensor | None,
    ) -> int:
        if hidden_states.ndim != 2 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden_states must have shape [tokens, {self.hidden_size}]"
            )
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must contain packed-sequence boundaries")
        num_sequences = cu_seqlens.numel() - 1
        is_capturing = (
            cu_seqlens.is_cuda
            and torch.cuda.is_available()
            and torch.cuda.is_current_stream_capturing()
        )
        if not is_capturing:
            if int(cu_seqlens[0]) != 0 or int(cu_seqlens[-1]) != hidden_states.shape[0]:
                raise ValueError("cu_seqlens must start at zero and end at num_tokens")

        if recurrent_states is not None:
            expected = (
                num_sequences,
                self.num_value_heads,
                self.key_head_dim,
                self.value_head_dim,
            )
            if tuple(recurrent_states.shape) != expected:
                raise ValueError(f"recurrent_states must have shape {expected}")
        if conv_states is not None:
            expected = (
                num_sequences,
                self.conv_dim,
                self.conv_kernel_size - 1,
            )
            if tuple(conv_states.shape) != expected:
                raise ValueError(f"conv_states must have shape {expected}")
        return num_sequences

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        recurrent_states: torch.Tensor | None = None,
        conv_states: torch.Tensor | None = None,
        return_state_history: bool = False,
        state_checkpoint_indices: tuple[int, ...] | None = None,
    ):
        """Run a packed ragged batch and return updated per-sequence states."""
        num_sequences = self._validate_inputs(
            hidden_states,
            cu_seqlens,
            recurrent_states,
            conv_states,
        )

        if (
            hidden_states.shape[0] == num_sequences
            and recurrent_states is not None
            and conv_states is not None
        ):
            result = self._forward_batched_decode(
                hidden_states,
                recurrent_states,
                conv_states,
            )
            if return_state_history or state_checkpoint_indices is not None:
                output, recurrent, conv = result
                return output, recurrent, conv, recurrent, conv
            return result

        projected_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        a = self.in_proj_a(hidden_states)
        b = self.in_proj_b(hidden_states)
        conv_weight = self.conv1d.weight.squeeze(1)

        outputs = []
        new_conv_states = []
        new_recurrent_states = []
        conv_state_histories = []
        recurrent_state_histories = []
        is_capturing = (
            hidden_states.is_cuda
            and torch.cuda.is_available()
            and torch.cuda.is_current_stream_capturing()
        )
        fixed_sequence_length = (
            hidden_states.shape[0] // num_sequences if is_capturing else None
        )
        for sequence_idx in range(num_sequences):
            if is_capturing:
                start = sequence_idx * fixed_sequence_length
                end = start + fixed_sequence_length
            else:
                start = int(cu_seqlens[sequence_idx])
                end = int(cu_seqlens[sequence_idx + 1])
            sequence_conv_state = (
                None if conv_states is None else conv_states[sequence_idx : sequence_idx + 1]
            )
            sequence_recurrent_state = (
                None
                if recurrent_states is None
                else recurrent_states[sequence_idx : sequence_idx + 1]
            )
            core = gated_delta_core_reference(
                projected_qkv[start:end].unsqueeze(0),
                conv_weight,
                a[start:end].unsqueeze(0),
                b[start:end].unsqueeze(0),
                self.A_log,
                self.dt_bias,
                num_key_heads=self.num_key_heads,
                num_value_heads=self.num_value_heads,
                key_head_dim=self.key_head_dim,
                value_head_dim=self.value_head_dim,
                conv_state=sequence_conv_state,
                recurrent_state=sequence_recurrent_state,
                return_state_history=return_state_history,
                checkpoint_indices=state_checkpoint_indices,
            )
            core_output = core.output.to(hidden_states.dtype)
            gate = z[start:end].reshape(
                1,
                end - start,
                self.num_value_heads,
                self.value_head_dim,
            )
            gated = self.norm(core_output, gate)
            local_output = self.out_proj(gated.reshape(end - start, self.value_dim))
            if self.tp_size > 1:
                dist.all_reduce(local_output)
            outputs.append(local_output)
            new_conv_states.append(core.conv_state.to(hidden_states.dtype))
            new_recurrent_states.append(core.recurrent_state)
            if return_state_history or state_checkpoint_indices is not None:
                conv_state_histories.append(
                    core.conv_state_history.squeeze(0).to(hidden_states.dtype)
                )
                recurrent_state_histories.append(
                    core.recurrent_state_history.squeeze(0)
                )

        result = (
            torch.cat(outputs, dim=0),
            torch.cat(new_recurrent_states, dim=0),
            torch.cat(new_conv_states, dim=0),
        )
        if return_state_history or state_checkpoint_indices is not None:
            return result + (
                torch.cat(recurrent_state_histories, dim=0),
                torch.cat(conv_state_histories, dim=0),
            )
        return result

    def _forward_batched_decode(
        self,
        hidden_states: torch.Tensor,
        recurrent_states: torch.Tensor,
        conv_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Graph-friendly single-token decode for every packed sequence."""
        projected_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states).reshape(
            -1,
            self.num_value_heads,
            self.value_head_dim,
        )
        a = self.in_proj_a(hidden_states)
        b = self.in_proj_b(hidden_states)

        core_output, new_recurrent_states, new_conv_states = gdn_decode_core(
            projected_qkv,
            a,
            b,
            self.conv1d.weight.squeeze(1),
            self.A_log,
            self.dt_bias,
            recurrent_states,
            conv_states,
            num_key_heads=self.num_key_heads,
            num_value_heads=self.num_value_heads,
            key_head_dim=self.key_head_dim,
            value_head_dim=self.value_head_dim,
            backend=self.decode_backend,
        )
        gated = self.norm(core_output, z)
        output = self.out_proj(gated.reshape(-1, self.value_dim))
        if self.tp_size > 1:
            dist.all_reduce(output)
        return output, new_recurrent_states, new_conv_states
