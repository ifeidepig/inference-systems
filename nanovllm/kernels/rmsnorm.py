from __future__ import annotations

import torch

from nanovllm.kernels._extension import load_cuda_ops


load_cuda_ops()


@torch.library.register_fake("nanovllm::rmsnorm")
def _rmsnorm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    torch._check(weight.ndim == 1)
    torch._check(weight.shape[0] == x.shape[-1])
    torch._check(x.dtype == weight.dtype)
    return torch.empty_like(x)


@torch.library.register_fake("nanovllm::fused_add_rmsnorm")
def _fused_add_rmsnorm_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch._check(x.shape == residual.shape)
    torch._check(weight.ndim == 1)
    torch._check(weight.shape[0] == x.shape[-1])
    torch._check(x.dtype == residual.dtype)
    torch._check(x.dtype == weight.dtype)
    return torch.empty_like(x), torch.empty_like(residual)


def fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fused residual add + RMSNorm with nano-vLLM's rounding order."""
    return torch.ops.nanovllm.fused_add_rmsnorm(
        x,
        residual,
        weight,
        epsilon,
    )


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Run RMSNorm with nano-vLLM's current rounding order."""
    return torch.ops.nanovllm.rmsnorm(x, weight, epsilon)
