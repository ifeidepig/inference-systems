import os

import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.use_custom_cuda = os.environ.get(
            "NANOVLLM_USE_CUSTOM_RMSNORM",
            "0",
        ) == "1"

    def can_use_custom_fused(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> bool:
        return (
            self.use_custom_cuda
            and x.is_cuda
            and residual.is_cuda
            and self.weight.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16)
            and x.dtype == residual.dtype
            and x.dtype == self.weight.dtype
            and x.shape == residual.shape
            and x.shape[-1] in (128, 1024)
            and x.is_contiguous()
            and residual.is_contiguous()
            and self.weight.is_contiguous()
        )

    def can_use_custom_rms(self, x: torch.Tensor) -> bool:
        return (
            self.use_custom_cuda
            and x.is_cuda
            and self.weight.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16)
            and x.dtype == self.weight.dtype
            and x.shape[-1] in (128, 1024)
            and x.is_contiguous()
            and self.weight.is_contiguous()
        )

    def custom_rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        from nanovllm.kernels import rmsnorm

        return rmsnorm(x, self.weight, self.eps)

    def custom_add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Import lazily so the extension is compiled during model warmup, not
        # for users who keep the default torch.compile implementation.
        from nanovllm.kernels import fused_add_rmsnorm

        return fused_add_rmsnorm(x, residual, self.weight, self.eps)

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None and self.can_use_custom_rms(x):
            return self.custom_rms_forward(x)
        elif residual is None:
            return self.rms_forward(x)
        elif self.can_use_custom_fused(x, residual):
            return self.custom_add_rms_forward(x, residual)
        else:
            return self.add_rms_forward(x, residual)


class Qwen3_5RMSNorm(nn.Module):
    """Qwen3.5 zero-centered RMSNorm.

    Qwen3.5 checkpoints store a zero-centered scale parameter and apply
    ``1 + weight`` in FP32.  Keep this separate from ``RMSNorm`` so existing
    Qwen3 parameters and the custom CUDA dispatch retain their current
    semantics.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def _normalize(self, x: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        normalized = x_fp32 * torch.rsqrt(variance + self.eps)
        normalized = normalized * (1.0 + self.weight.float())
        return normalized.to(output_dtype)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        output_dtype = x.dtype
        if residual is None:
            return self._normalize(x, output_dtype)

        summed_fp32 = x.float() + residual.float()
        new_residual = summed_fp32.to(output_dtype)
        return self._normalize(summed_fp32, output_dtype), new_residual


class Qwen3_5RMSNormGated(nn.Module):
    """Per-head RMSNorm followed by the Qwen3.5 SiLU output gate."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape != gate.shape:
            raise ValueError("hidden_states and gate must have the same shape")
        output_dtype = hidden_states.dtype
        x_fp32 = hidden_states.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        normalized = x_fp32 * torch.rsqrt(variance + self.eps)
        normalized = normalized * self.weight.float()
        return (normalized * torch.nn.functional.silu(gate.float())).to(output_dtype)
