from types import SimpleNamespace

import torch.distributed as dist

from nanovllm.models.registry import create_model, normalize_hf_config
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM


def _qwen3_config():
    return SimpleNamespace(
        model_type="qwen3",
        architectures=["Qwen3ForCausalLM"],
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        hidden_act="silu",
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        attention_bias=False,
        rope_theta=10_000,
        rope_scaling=None,
        tie_word_embeddings=True,
    )


def _qwen35_full_config():
    text = SimpleNamespace(
        model_type="qwen3_5_text",
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        hidden_act="silu",
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        attention_bias=False,
        tie_word_embeddings=False,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=3,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        mtp_num_hidden_layers=1,
        rope_parameters={
            "partial_rotary_factor": 0.5,
            "rope_theta": 10_000,
            "mrope_interleaved": True,
            "mrope_section": [1, 0, 0],
        },
    )
    return SimpleNamespace(
        model_type="qwen3_5",
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config=text,
    )


def _ensure_process_group():
    if dist.is_initialized():
        return False
    dist.init_process_group(
        "gloo",
        init_method="tcp://127.0.0.1:29637",
        rank=0,
        world_size=1,
    )
    return True


def test_registry_normalizes_qwen3_without_changing_architecture():
    text, capabilities = normalize_hf_config(_qwen3_config())
    assert text.model_type == "qwen3"
    assert capabilities.architecture == "Qwen3ForCausalLM"
    assert capabilities.full_attention_layer_indices == (0, 1)
    assert capabilities.linear_attention_layer_indices == ()
    assert not capabilities.is_hybrid


def test_registry_normalizes_qwen35_multimodal_wrapper_to_text_runtime():
    text, capabilities = normalize_hf_config(_qwen35_full_config())
    assert text.model_type == "qwen3_5_text"
    assert capabilities.architecture == "Qwen3_5ForCausalLM"
    assert capabilities.linear_attention_layer_indices == (0, 1, 2)
    assert capabilities.full_attention_layer_indices == (3,)
    assert capabilities.is_hybrid
    assert capabilities.supports_mrope
    assert capabilities.mtp_num_hidden_layers == 1


def test_registry_constructs_qwen3_and_tiny_qwen35_models():
    owns_group = _ensure_process_group()
    try:
        qwen3_config = _qwen3_config()
        qwen3_text, qwen3_capabilities = normalize_hf_config(qwen3_config)
        qwen3 = create_model(qwen3_text, qwen3_capabilities)
        assert isinstance(qwen3, Qwen3ForCausalLM)

        qwen35_text, qwen35_capabilities = normalize_hf_config(
            _qwen35_full_config()
        )
        qwen35 = create_model(qwen35_text, qwen35_capabilities)
        assert isinstance(qwen35, Qwen3_5ForCausalLM)
        assert [layer.layer_type for layer in qwen35.model.layers] == [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]
    finally:
        if owns_group:
            dist.destroy_process_group()
