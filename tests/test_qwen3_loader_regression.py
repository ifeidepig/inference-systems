from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from safetensors.torch import save_file

from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model


def _config():
    return SimpleNamespace(
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
        tie_word_embeddings=False,
    )


def test_strict_loader_preserves_qwen3_packed_checkpoint_path(tmp_path: Path):
    owns_group = False
    if not dist.is_initialized():
        dist.init_process_group(
            "gloo",
            init_method="tcp://127.0.0.1:29641",
            rank=0,
            world_size=1,
        )
        owns_group = True
    try:
        config = _config()
        source = Qwen3ForCausalLM(config).float()
        torch.manual_seed(41)
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.normal_(0.0, 0.02)

        checkpoint = {}
        for name, tensor in source.state_dict().items():
            tensor = tensor.detach().cpu().contiguous()
            if name.endswith("qkv_proj.weight"):
                q, k, v = tensor.split((8, 4, 4), dim=0)
                checkpoint[name.replace("qkv_proj", "q_proj")] = q.contiguous()
                checkpoint[name.replace("qkv_proj", "k_proj")] = k.contiguous()
                checkpoint[name.replace("qkv_proj", "v_proj")] = v.contiguous()
            elif name.endswith("gate_up_proj.weight"):
                gate, up = tensor.chunk(2, dim=0)
                checkpoint[name.replace("gate_up_proj", "gate_proj")] = gate.contiguous()
                checkpoint[name.replace("gate_up_proj", "up_proj")] = up.contiguous()
            else:
                checkpoint[name] = tensor
        save_file(checkpoint, tmp_path / "model.safetensors")

        target = Qwen3ForCausalLM(config).float()
        report = load_model(target, str(tmp_path), strict=True)
        assert report.ok
        assert len(report.packed) == config.num_hidden_layers * 5
        for name, expected in source.state_dict().items():
            torch.testing.assert_close(target.state_dict()[name], expected)
    finally:
        if owns_group:
            dist.destroy_process_group()
