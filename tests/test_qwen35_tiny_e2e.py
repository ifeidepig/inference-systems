from types import SimpleNamespace

import torch
import torch.distributed as dist

from nanovllm.layers.gated_delta_net import GatedDeltaNet
from nanovllm.layers.layernorm import Qwen3_5RMSNorm, Qwen3_5RMSNormGated
from nanovllm.models.registry import create_model, normalize_hf_config
from nanovllm.utils.context import reset_context, set_context


def _full_config():
    text = SimpleNamespace(
        model_type="qwen3_5_text",
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        hidden_act="silu",
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        attention_bias=False,
        tie_word_embeddings=False,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        mtp_num_hidden_layers=1,
        rope_parameters={
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000,
            "mrope_interleaved": True,
            "mrope_section": [2, 2, 4],
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
        init_method="tcp://127.0.0.1:29638",
        rank=0,
        world_size=1,
    )
    return True


def _initialize_model(model):
    torch.manual_seed(123)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.02)
        for module in model.modules():
            if isinstance(module, Qwen3_5RMSNorm):
                module.weight.zero_()
            elif isinstance(module, Qwen3_5RMSNormGated):
                module.weight.fill_(1.0)
            elif isinstance(module, GatedDeltaNet):
                module.A_log.zero_()
                module.dt_bias.fill_(1.0)


def _install_single_block_kv_cache(model, dtype):
    attention_modules = [
        module
        for module in model.modules()
        if hasattr(module, "k_cache") and hasattr(module, "v_cache")
    ]
    assert len(attention_modules) == 1
    for module in attention_modules:
        module.k_cache = torch.zeros(
            1,
            256,
            module.num_kv_heads,
            module.head_dim,
            dtype=dtype,
            device="cuda",
        )
        module.v_cache = torch.zeros_like(module.k_cache)
    return attention_modules


def _prefill_context(num_tokens):
    cu = torch.tensor([0, num_tokens], dtype=torch.int32, device="cuda")
    set_context(
        True,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=num_tokens,
        max_seqlen_k=num_tokens,
        slot_mapping=torch.arange(num_tokens, dtype=torch.int32, device="cuda"),
    )
    return cu


def test_tiny_hybrid_prefill_decode_matches_one_shot():
    if not torch.cuda.is_available():
        return
    owns_group = _ensure_process_group()
    try:
        text_config, capabilities = normalize_hf_config(_full_config())
        model = create_model(text_config, capabilities).cuda().to(torch.bfloat16).eval()
        _initialize_model(model)
        attention_modules = _install_single_block_kv_cache(model, torch.bfloat16)
        input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long, device="cuda")
        positions = torch.arange(4, dtype=torch.long, device="cuda")

        with torch.inference_mode():
            cu_full = _prefill_context(4)
            full_hidden, full_recurrent, full_conv = model(
                input_ids,
                positions,
                cu_full,
            )
            full_logits = model.compute_logits(full_hidden)
            reset_context()

            for module in attention_modules:
                module.k_cache.zero_()
                module.v_cache.zero_()

            cu_prefix = _prefill_context(3)
            _, prefix_recurrent, prefix_conv = model(
                input_ids[:3],
                positions[:3],
                cu_prefix,
            )
            reset_context()

            set_context(
                False,
                cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
                slot_mapping=torch.tensor([3], dtype=torch.int32, device="cuda"),
                context_lens=torch.tensor([4], dtype=torch.int32, device="cuda"),
                block_tables=torch.tensor([[0]], dtype=torch.int32, device="cuda"),
            )
            decode_hidden, decode_recurrent, decode_conv = model(
                input_ids[3:],
                positions[3:],
                torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
                prefix_recurrent,
                prefix_conv,
            )
            decode_logits = model.compute_logits(decode_hidden)
            reset_context()

        torch.testing.assert_close(
            decode_hidden.float(),
            full_hidden[-1:].float(),
            rtol=2e-2,
            atol=2e-2,
        )
        torch.testing.assert_close(
            decode_logits.float(),
            full_logits.float(),
            rtol=2e-2,
            atol=2e-2,
        )
        assert decode_logits.argmax(dim=-1).item() == full_logits.argmax(dim=-1).item()
        for layer_index in capabilities.linear_attention_layer_indices:
            torch.testing.assert_close(
                decode_recurrent[layer_index],
                full_recurrent[layer_index],
                rtol=2e-2,
                atol=2e-2,
            )
            torch.testing.assert_close(
                decode_conv[layer_index].float(),
                full_conv[layer_index].float(),
                rtol=2e-2,
                atol=2e-2,
            )
    finally:
        reset_context()
        if owns_group:
            dist.destroy_process_group()


def test_tiny_hybrid_three_decode_steps_match_one_shot_hidden_and_logits():
    if not torch.cuda.is_available():
        return
    owns_group = _ensure_process_group()
    try:
        text_config, capabilities = normalize_hf_config(_full_config())
        model = create_model(text_config, capabilities).cuda().to(torch.bfloat16).eval()
        _initialize_model(model)
        attention_modules = _install_single_block_kv_cache(model, torch.bfloat16)
        input_ids = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long, device="cuda")
        positions = torch.arange(6, dtype=torch.long, device="cuda")

        with torch.inference_mode():
            cu_full = _prefill_context(6)
            full_hidden, full_recurrent, full_conv = model(
                input_ids,
                positions,
                cu_full,
            )
            reset_context()
            full_logits = model.compute_logits(full_hidden)

            for module in attention_modules:
                module.k_cache.zero_()
                module.v_cache.zero_()

            cu_prefix = _prefill_context(3)
            _, recurrent, conv = model(
                input_ids[:3],
                positions[:3],
                cu_prefix,
            )
            reset_context()

            decoded_hidden = []
            for token_index in range(3, 6):
                cu_decode = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
                set_context(
                    False,
                    cu_seqlens_q=cu_decode,
                    slot_mapping=torch.tensor(
                        [token_index], dtype=torch.int32, device="cuda"
                    ),
                    context_lens=torch.tensor(
                        [token_index + 1], dtype=torch.int32, device="cuda"
                    ),
                    block_tables=torch.tensor([[0]], dtype=torch.int32, device="cuda"),
                )
                hidden, recurrent, conv = model(
                    input_ids[token_index : token_index + 1],
                    positions[token_index : token_index + 1],
                    cu_decode,
                    recurrent,
                    conv,
                )
                decoded_hidden.append(hidden)
                reset_context()

            decoded_hidden = torch.cat(decoded_hidden, dim=0)
            decoded_logits = model.compute_logits(decoded_hidden)

        torch.testing.assert_close(
            decoded_hidden.float(),
            full_hidden[3:].float(),
            rtol=2e-2,
            atol=2e-2,
        )
        torch.testing.assert_close(
            decoded_logits.float(),
            full_logits[3:].float(),
            rtol=2e-2,
            atol=2e-2,
        )
        assert torch.equal(
            decoded_logits.argmax(dim=-1),
            full_logits[3:].argmax(dim=-1),
        )
        for layer_index in capabilities.linear_attention_layer_indices:
            torch.testing.assert_close(
                recurrent[layer_index],
                full_recurrent[layer_index],
                rtol=2e-2,
                atol=2e-2,
            )
            torch.testing.assert_close(
                conv[layer_index].float(),
                full_conv[layer_index].float(),
                rtol=2e-2,
                atol=2e-2,
            )
    finally:
        reset_context()
        if owns_group:
            dist.destroy_process_group()
