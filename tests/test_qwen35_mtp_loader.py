from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from safetensors.torch import save_file

from nanovllm.models.qwen3_5_mtp import Qwen3_5MTP
from nanovllm.models.registry import create_model, normalize_hf_config
from nanovllm.utils.loader import load_model


def _config():
    text = SimpleNamespace(
        model_type="qwen3_5_text",
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        hidden_act="silu",
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        attention_bias=False,
        tie_word_embeddings=True,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
        rope_parameters={"partial_rotary_factor": 0.25, "rope_theta": 10_000},
    )
    full = SimpleNamespace(
        model_type="qwen3_5",
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config=text,
    )
    return text, normalize_hf_config(full)[1]


def test_mtp_strict_loader_consumes_only_mtp_checkpoint_tensors(tmp_path: Path):
    owns_group = False
    if not dist.is_initialized():
        dist.init_process_group(
            "gloo",
            init_method="tcp://127.0.0.1:29643",
            rank=0,
            world_size=1,
        )
        owns_group = True
    try:
        config, capabilities = _config()
        target = create_model(config, capabilities).float()
        source = Qwen3_5MTP(config, target.model.embed_tokens).float()
        torch.manual_seed(47)
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.normal_(0.0, 0.02)

        tensors = {"model.language_model.embed_tokens.weight": torch.randn(32, 64)}
        for name, tensor in source.state_dict().items():
            tensor = tensor.detach().cpu().contiguous()
            if name.endswith("gate_up_proj.weight"):
                gate, up = tensor.chunk(2, dim=0)
                tensors["mtp." + name.replace("gate_up_proj", "gate_proj")] = gate.contiguous()
                tensors["mtp." + name.replace("gate_up_proj", "up_proj")] = up.contiguous()
            else:
                tensors["mtp." + name] = tensor
        save_file(tensors, tmp_path / "model.safetensors")

        loaded = Qwen3_5MTP(config, target.model.embed_tokens).float()
        report = load_model(loaded, str(tmp_path), strict=True)
        assert report.ok
        assert "model.language_model.embed_tokens.weight" in report.skipped
        assert len(report.loaded) + len(report.packed) == 15
        for name, expected in source.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[name], expected)
    finally:
        if owns_group:
            dist.destroy_process_group()
