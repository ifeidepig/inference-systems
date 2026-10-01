"""Qwen3.5 one-layer Multi-Token Prediction draft head."""

import torch
from torch import nn

from nanovllm.layers.layernorm import Qwen3_5RMSNorm
from nanovllm.models.qwen3_5 import Qwen3_5DecoderLayer


class Qwen3_5MTP(nn.Module):
    """Produce draft hidden states from target hidden state and token embedding.

    This module mirrors the checkpoint's ``mtp.*`` tensors. Embedding and LM
    head weights are owned by the target model and passed in/shared rather than
    duplicated because Qwen3.5-9B sets ``mtp_use_dedicated_embeddings=false``.
    """

    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    checkpoint_prefix_mapping = (("mtp.", ""),)
    checkpoint_included_prefixes = ("mtp.",)

    def __init__(self, config, embed_tokens: nn.Module) -> None:
        super().__init__()
        num_layers = int(getattr(config, "mtp_num_hidden_layers", 1))
        if num_layers != 1:
            raise NotImplementedError("the first MTP implementation supports one layer")
        # The target owns this module. Avoid registering it again so strict
        # MTP loading only expects the 15 checkpoint tensors under `mtp.*`.
        object.__setattr__(self, "_embed_tokens", embed_tokens)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, config.num_hidden_layers, "full_attention")]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.pre_fc_norm_embedding(self._embed_tokens(input_ids))
        target_hidden_states = self.pre_fc_norm_hidden(target_hidden_states)
        hidden_states = self.fc(
            torch.cat((embeddings, target_hidden_states), dim=-1)
        )
        hidden_states, residual, _, _, _, _ = self.layers[0](
            positions,
            hidden_states,
            None,
            cu_seqlens,
        )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
