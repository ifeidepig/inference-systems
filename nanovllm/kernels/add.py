from __future__ import annotations

import torch

from nanovllm.kernels._extension import load_cuda_ops


load_cuda_ops()


@torch.library.register_fake("nanovllm::add")
def _add_fake(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    torch._check(x.shape == residual.shape)
    torch._check(x.dtype == residual.dtype)
    torch._check(x.device == residual.device)
    return torch.empty_like(x)


def add(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Add two contiguous FP16/BF16 CUDA tensors on PyTorch's current stream."""
    return torch.ops.nanovllm.add(x, residual)
