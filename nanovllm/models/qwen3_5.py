"""Dense Qwen3.5 text-only model for nano-vLLM.

The initial Gated DeltaNet backend is correctness-first and uses the explicit
FP32 reference core. Vision, MTP execution, quantization, and MoE are outside
this model's scope.
"""

import torch
from torch import nn
import torch.distributed as dist

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.gated_delta_net import GatedDeltaNet
from nanovllm.layers.layernorm import Qwen3_5RMSNorm
from nanovllm.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.models.registry import ModelCapabilities


class Qwen3_5Attention(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_heads % tp_size or self.total_num_kv_heads % tp_size:
            raise ValueError("Q/KV head counts must be divisible by tensor parallel size")
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_heads * self.head_dim * 2,
            bias=config.attention_bias,
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        rope_parameters = getattr(config, "rope_parameters", None) or {}
        rotary_dim = int(
            self.head_dim * rope_parameters.get("partial_rotary_factor", 1.0)
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
            base=rope_parameters.get("rope_theta", 10_000_000),
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        q_gate = self.q_proj(hidden_states).view(
            -1,
            self.num_heads,
            self.head_dim * 2,
        )
        query, gate = q_gate.chunk(2, dim=-1)
        key = self.k_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = self.rotary_emb(positions, query, key)
        output = self.attn(query, key, value)
        if output.ndim == 4:
            output = output.squeeze(1)
        output = output * torch.sigmoid(gate)
        return self.o_proj(output.flatten(1, -1))


class Qwen3_5MLP(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError("Qwen3.5 MLP currently supports only SiLU")
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(hidden_states)))


class Qwen3_5DecoderLayer(nn.Module):

    def __init__(self, config, layer_idx: int, layer_type: str) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.is_linear_attention = layer_type == "linear_attention"
        if self.is_linear_attention:
            self.linear_attn = GatedDeltaNet(config, layer_idx)
            self.self_attn = None
        else:
            self.self_attn = Qwen3_5Attention(config)
            self.linear_attn = None
        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        recurrent_state: torch.Tensor | None = None,
        conv_state: torch.Tensor | None = None,
        return_state_history: bool = False,
        state_checkpoint_indices: tuple[int, ...] | None = None,
    ):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.is_linear_attention:
            linear_result = self.linear_attn(
                hidden_states,
                cu_seqlens,
                recurrent_state,
                conv_state,
                return_state_history=return_state_history,
                state_checkpoint_indices=state_checkpoint_indices,
            )
            if return_state_history or state_checkpoint_indices is not None:
                (
                    hidden_states,
                    recurrent_state,
                    conv_state,
                    recurrent_history,
                    conv_history,
                ) = linear_result
            else:
                hidden_states, recurrent_state, conv_state = linear_result
                recurrent_history = conv_history = None
        else:
            hidden_states = self.self_attn(positions, hidden_states)
            recurrent_history = conv_history = None

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states,
            residual,
        )
        hidden_states = self.mlp(hidden_states)
        return (
            hidden_states,
            residual,
            recurrent_state,
            conv_state,
            recurrent_history,
            conv_history,
        )


class Qwen3_5Model(nn.Module):

    def __init__(self, config, capabilities: ModelCapabilities) -> None:
        super().__init__()
        self.layer_types = capabilities.layer_types
        self.linear_attention_layer_indices = capabilities.linear_attention_layer_indices
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            Qwen3_5DecoderLayer(config, index, layer_type)
            for index, layer_type in enumerate(self.layer_types)
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        recurrent_states: list[torch.Tensor | None] | None = None,
        conv_states: list[torch.Tensor | None] | None = None,
        return_state_history: bool = False,
        state_checkpoint_indices: tuple[int, ...] | None = None,
    ):
        num_layers = len(self.layers)
        if recurrent_states is None:
            recurrent_states = [None] * num_layers
        if conv_states is None:
            conv_states = [None] * num_layers
        if len(recurrent_states) != num_layers or len(conv_states) != num_layers:
            raise ValueError("state lists must have one entry per decoder layer")

        hidden_states = self.embed_tokens(input_ids)
        residual = None
        new_recurrent_states: list[torch.Tensor | None] = []
        new_conv_states: list[torch.Tensor | None] = []
        recurrent_histories: list[torch.Tensor | None] = []
        conv_histories: list[torch.Tensor | None] = []
        for index, layer in enumerate(self.layers):
            (
                hidden_states,
                residual,
                recurrent_state,
                conv_state,
                recurrent_history,
                conv_history,
            ) = layer(
                positions,
                hidden_states,
                residual,
                cu_seqlens,
                recurrent_states[index],
                conv_states[index],
                return_state_history=return_state_history,
                state_checkpoint_indices=state_checkpoint_indices,
            )
            new_recurrent_states.append(recurrent_state)
            new_conv_states.append(conv_state)
            recurrent_histories.append(recurrent_history)
            conv_histories.append(conv_history)
        hidden_states, _ = self.norm(hidden_states, residual)
        result = hidden_states, new_recurrent_states, new_conv_states
        if return_state_history or state_checkpoint_indices is not None:
            return result + (recurrent_histories, conv_histories)
        return result


class Qwen3_5ForCausalLM(nn.Module):
    # Checkpoint q/k/v projections are already separate for Qwen3.5.
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    checkpoint_prefix_mapping = (
        ("model.language_model.", "model."),
        ("language_model.", "model."),
    )
    explicitly_skipped_prefixes = ("model.visual.", "mtp.")

    def __init__(self, config, capabilities: ModelCapabilities) -> None:
        super().__init__()
        self.config = config
        self.capabilities = capabilities
        self.model = Qwen3_5Model(config, capabilities)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            # Tie the Parameter object itself, not only its storage.  Official
            # tied checkpoints contain only ``embed_tokens.weight`` and the
            # strict loader discovers aliases by Parameter identity.
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        recurrent_states: list[torch.Tensor | None] | None = None,
        conv_states: list[torch.Tensor | None] | None = None,
        return_state_history: bool = False,
        state_checkpoint_indices: tuple[int, ...] | None = None,
    ):
        return self.model(
            input_ids,
            positions,
            cu_seqlens,
            recurrent_states,
            conv_states,
            return_state_history,
            state_checkpoint_indices,
        )

    def compute_logits(self, hidden_states_or_tuple) -> torch.Tensor:
        hidden_states = (
            hidden_states_or_tuple[0]
            if isinstance(hidden_states_or_tuple, tuple)
            else hidden_states_or_tuple
        )
        return self.lm_head(hidden_states)
