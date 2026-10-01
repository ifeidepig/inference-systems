import os
import tempfile
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from nanovllm.layers.gated_delta_net import GatedDeltaNet
from nanovllm.models.qwen35_reference import gated_delta_core_reference


def _config():
    return SimpleNamespace(
        hidden_size=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=3,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
    )


def _worker(rank, world_size, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        module = GatedDeltaNet(_config(), layer_idx=0).float()
        assert module.num_key_heads == 1
        assert module.num_value_heads == 2
        assert module.conv_dim == 10

        full_qkv = torch.arange(20 * 8, dtype=torch.float32).reshape(20, 8)
        module.in_proj_qkv.weight.weight_loader(module.in_proj_qkv.weight, full_qkv)
        q = full_qkv[rank * 3 : (rank + 1) * 3]
        k = full_qkv[6 + rank * 3 : 6 + (rank + 1) * 3]
        v = full_qkv[12 + rank * 4 : 12 + (rank + 1) * 4]
        torch.testing.assert_close(
            module.in_proj_qkv.weight,
            torch.cat((q, k, v), dim=0),
        )

        full_z = torch.arange(8 * 8, dtype=torch.float32).reshape(8, 8)
        module.in_proj_z.weight.weight_loader(module.in_proj_z.weight, full_z)
        torch.testing.assert_close(
            module.in_proj_z.weight,
            full_z[rank * 4 : (rank + 1) * 4],
        )

        full_heads = torch.arange(4, dtype=torch.float32)
        module.A_log.weight_loader(module.A_log, full_heads)
        torch.testing.assert_close(
            module.A_log,
            full_heads[rank * 2 : (rank + 1) * 2],
        )

        full_out = torch.arange(8 * 8, dtype=torch.float32).reshape(8, 8)
        module.out_proj.weight.weight_loader(module.out_proj.weight, full_out)
        torch.testing.assert_close(
            module.out_proj.weight,
            full_out[:, rank * 4 : (rank + 1) * 4],
        )
    finally:
        dist.destroy_process_group()


def _forward_worker(rank, world_size, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        module = GatedDeltaNet(_config(), layer_idx=0).float().eval()
        torch.manual_seed(29)
        qkv = torch.randn(20, 8) * 0.1
        z_weight = torch.randn(8, 8) * 0.1
        a_weight = torch.randn(4, 8) * 0.1
        b_weight = torch.randn(4, 8) * 0.1
        conv_weight = torch.randn(20, 1, 4) * 0.1
        a_log = torch.randn(4) * 0.1
        dt_bias = torch.randn(4) * 0.1
        out_weight = torch.randn(8, 8) * 0.1
        hidden = torch.randn(3, 8)

        module.in_proj_qkv.weight.weight_loader(module.in_proj_qkv.weight, qkv)
        module.in_proj_z.weight.weight_loader(module.in_proj_z.weight, z_weight)
        module.in_proj_a.weight.weight_loader(module.in_proj_a.weight, a_weight)
        module.in_proj_b.weight.weight_loader(module.in_proj_b.weight, b_weight)
        module.conv1d.weight.weight_loader(module.conv1d.weight, conv_weight)
        module.A_log.weight_loader(module.A_log, a_log)
        module.dt_bias.weight_loader(module.dt_bias, dt_bias)
        module.out_proj.weight.weight_loader(module.out_proj.weight, out_weight)
        module.norm.weight.data.fill_(1.0)

        actual, _, _ = module(hidden, torch.tensor([0, 3]))

        projected = F.linear(hidden, qkv).unsqueeze(0)
        a = F.linear(hidden, a_weight).unsqueeze(0)
        b = F.linear(hidden, b_weight).unsqueeze(0)
        core = gated_delta_core_reference(
            projected,
            conv_weight.squeeze(1),
            a,
            b,
            a_log,
            dt_bias,
            num_key_heads=2,
            num_value_heads=4,
            key_head_dim=3,
            value_head_dim=2,
        ).output
        z = F.linear(hidden, z_weight).reshape(1, 3, 4, 2)
        variance = core.pow(2).mean(dim=-1, keepdim=True)
        gated = core * torch.rsqrt(variance + 1e-6) * F.silu(z)
        expected = F.linear(gated.reshape(3, 8), out_weight)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    finally:
        dist.destroy_process_group()


def test_gdn_tp2_weight_shards_and_local_state_shapes():
    rendezvous = tempfile.NamedTemporaryFile(delete=False)
    rendezvous.close()
    try:
        mp.start_processes(
            _worker,
            args=(2, rendezvous.name),
            nprocs=2,
            join=True,
            start_method="fork",
        )
    finally:
        try:
            os.unlink(rendezvous.name)
        except FileNotFoundError:
            pass


def test_gdn_tp2_forward_matches_global_reference():
    rendezvous = tempfile.NamedTemporaryFile(delete=False)
    rendezvous.close()
    try:
        mp.start_processes(
            _forward_worker,
            args=(2, rendezvous.name),
            nprocs=2,
            join=True,
            start_method="fork",
        )
    finally:
        try:
            os.unlink(rendezvous.name)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    test_gdn_tp2_weight_shards_and_local_state_shapes()
    test_gdn_tp2_forward_matches_global_reference()
    print("Qwen3.5 TP=2 shard and forward tests passed")
