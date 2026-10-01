import os
import unittest

import torch

from nanovllm.layers.layernorm import RMSNorm


def reference(
    module: RMSNorm,
    x: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    original_dtype = x.dtype
    h = x.float() + residual.float()
    residual_output = h.to(original_dtype)
    variance = h.pow(2).mean(dim=-1, keepdim=True)
    normalized = h * torch.rsqrt(variance + module.eps)
    return normalized.to(original_dtype) * module.weight, residual_output


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class RMSNormModuleDispatchTest(unittest.TestCase):

    def setUp(self) -> None:
        self.previous = os.environ.get("NANOVLLM_USE_CUSTOM_RMSNORM")

    def tearDown(self) -> None:
        if self.previous is None:
            os.environ.pop("NANOVLLM_USE_CUSTOM_RMSNORM", None)
        else:
            os.environ["NANOVLLM_USE_CUSTOM_RMSNORM"] = self.previous

    def test_opt_in_custom_path_matches_reference(self) -> None:
        os.environ["NANOVLLM_USE_CUSTOM_RMSNORM"] = "1"
        module = RMSNorm(1024).cuda().bfloat16()
        x = torch.randn(17, 1024, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn_like(x)

        self.assertTrue(module.can_use_custom_fused(x, residual))
        actual, actual_residual = module(x, residual)
        expected, expected_residual = reference(module, x, residual)

        torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_opt_in_standalone_path_matches_reference(self) -> None:
        os.environ["NANOVLLM_USE_CUSTOM_RMSNORM"] = "1"
        module = RMSNorm(128).cuda().bfloat16()
        x = torch.randn(3, 7, 128, device="cuda", dtype=torch.bfloat16)

        self.assertTrue(module.can_use_custom_rms(x))
        actual = module(x)

        value = x.float()
        variance = value.pow(2).mean(dim=-1, keepdim=True)
        expected = (
            value * torch.rsqrt(variance + module.eps)
        ).to(x.dtype) * module.weight
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_default_keeps_compiled_fallback(self) -> None:
        os.environ.pop("NANOVLLM_USE_CUSTOM_RMSNORM", None)
        module = RMSNorm(1024).cuda().bfloat16()
        x = torch.randn(3, 1024, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn_like(x)

        self.assertFalse(module.can_use_custom_fused(x, residual))
        actual, actual_residual = module(x, residual)
        expected, expected_residual = reference(module, x, residual)
        torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_unsupported_hidden_size_falls_back(self) -> None:
        os.environ["NANOVLLM_USE_CUSTOM_RMSNORM"] = "1"
        module = RMSNorm(256).cuda().bfloat16()
        x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        self.assertFalse(module.can_use_custom_fused(x, residual))


if __name__ == "__main__":
    unittest.main()
