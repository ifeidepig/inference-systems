"""Model/config normalization for nano-vLLM model construction."""

from dataclasses import dataclass
from typing import Any

from torch import nn


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    architecture: str
    model_type: str
    layer_types: tuple[str, ...]
    full_attention_layer_indices: tuple[int, ...]
    linear_attention_layer_indices: tuple[int, ...]
    has_recurrent_state: bool
    supports_mrope: bool
    mtp_num_hidden_layers: int

    @property
    def is_hybrid(self) -> bool:
        return bool(self.linear_attention_layer_indices)


def normalize_hf_config(full_config: Any) -> tuple[Any, ModelCapabilities]:
    """Return the executable text config and explicit model capabilities."""
    text_config = getattr(full_config, "text_config", full_config)
    architecture_list = getattr(full_config, "architectures", None) or getattr(
        text_config,
        "architectures",
        None,
    )
    model_type = getattr(text_config, "model_type", getattr(full_config, "model_type", ""))

    if model_type == "qwen3_5_text" or getattr(full_config, "model_type", "") == "qwen3_5":
        architecture = (
            architecture_list[0]
            if architecture_list
            else "Qwen3_5ForCausalLM"
        )
        # A multimodal checkpoint still maps to the text-only runtime class.
        if architecture == "Qwen3_5ForConditionalGeneration":
            architecture = "Qwen3_5ForCausalLM"
    elif model_type == "qwen3":
        architecture = architecture_list[0] if architecture_list else "Qwen3ForCausalLM"
    else:
        raise ValueError(f"unsupported model_type: {model_type!r}")

    num_layers = int(text_config.num_hidden_layers)
    raw_layer_types = getattr(text_config, "layer_types", None)
    if raw_layer_types is None:
        layer_types = ("full_attention",) * num_layers
    else:
        layer_types = tuple(raw_layer_types)
        if len(layer_types) != num_layers:
            raise ValueError(
                f"layer_types has {len(layer_types)} entries for {num_layers} layers"
            )
    allowed = {"full_attention", "linear_attention", "sliding_attention"}
    unknown = sorted(set(layer_types) - allowed)
    if unknown:
        raise ValueError(f"unsupported layer types: {unknown}")

    full_indices = tuple(
        index
        for index, layer_type in enumerate(layer_types)
        if layer_type in ("full_attention", "sliding_attention")
    )
    linear_indices = tuple(
        index
        for index, layer_type in enumerate(layer_types)
        if layer_type == "linear_attention"
    )
    rope_parameters = getattr(text_config, "rope_parameters", None) or {}
    capabilities = ModelCapabilities(
        architecture=architecture,
        model_type=model_type,
        layer_types=layer_types,
        full_attention_layer_indices=full_indices,
        linear_attention_layer_indices=linear_indices,
        has_recurrent_state=bool(linear_indices),
        supports_mrope=bool(
            rope_parameters.get("mrope_interleaved", False)
            or rope_parameters.get("mrope_section")
        ),
        mtp_num_hidden_layers=int(getattr(text_config, "mtp_num_hidden_layers", 0)),
    )
    return text_config, capabilities


def create_model(text_config: Any, capabilities: ModelCapabilities) -> nn.Module:
    if capabilities.architecture == "Qwen3ForCausalLM":
        from nanovllm.models.qwen3 import Qwen3ForCausalLM

        return Qwen3ForCausalLM(text_config)
    if capabilities.architecture == "Qwen3_5ForCausalLM":
        from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM

        return Qwen3_5ForCausalLM(text_config, capabilities)
    raise ValueError(f"unsupported architecture: {capabilities.architecture!r}")
