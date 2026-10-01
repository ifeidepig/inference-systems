import unittest

import torch


def reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    original_dtype = x.dtype
    h = x.float() + residual.float()
    residual_output = h.to(original_dtype)
    variance = h.pow(2).mean(dim=-1, keepdim=True)
    normalized = h * torch.rsqrt(variance + epsilon)
    output = normalized.to(original_dtype) * weight
    return output, residual_output


def rmsnorm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    original_dtype = x.dtype
    value = x.float()
    variance = value.pow(2).mean(dim=-1, keepdim=True)
    normalized = value * torch.rsqrt(variance + epsilon)
    return normalized.to(original_dtype) * weight


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class FusedAddRMSNormTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from nanovllm.kernels import fused_add_rmsnorm, rmsnorm

        cls.op = staticmethod(fused_add_rmsnorm)
        cls.rmsnorm = staticmethod(rmsnorm)

    def test_standalone_rmsnorm(self) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            for shape in ((7, 1024), (2, 3, 128)):
                with self.subTest(dtype=dtype, shape=shape):
                    x = torch.randn(*shape, device="cuda", dtype=dtype)
                    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)
                    actual = self.rmsnorm(x, weight, 1e-6)
                    expected = rmsnorm_reference(x, weight, 1e-6)
                    tolerance = 2e-3 if dtype == torch.float16 else 2e-2
                    torch.testing.assert_close(
                        actual,
                        expected,
                        rtol=tolerance,
                        atol=tolerance,
                    )

    def test_standalone_rmsnorm_torch_compile(self) -> None:
        compiled = torch.compile(self.rmsnorm, fullgraph=True)
        x = torch.randn(5, 1024, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
        actual = compiled(x, weight, 1e-6)
        expected = rmsnorm_reference(x, weight, 1e-6)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def assert_matches_reference(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        torch.manual_seed(7)
        x = torch.randn(*shape, device="cuda", dtype=dtype)
        residual = torch.randn_like(x)
        weight = torch.randn(shape[-1], device="cuda", dtype=dtype)
        epsilon = 1e-6

        actual, actual_residual = self.op(x, residual, weight, epsilon)
        expected, expected_residual = reference(x, residual, weight, epsilon)

        torch.testing.assert_close(
            actual_residual,
            expected_residual,
            rtol=0,
            atol=0,
        )
        tolerance = 2e-3 if dtype == torch.float16 else 2e-2
        torch.testing.assert_close(
            actual,
            expected,
            rtol=tolerance,
            atol=tolerance,
        )

    def test_supported_shapes_and_dtypes(self) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            for shape in ((1, 1024), (37, 1024), (2, 5, 128)):
                with self.subTest(dtype=dtype, shape=shape):
                    self.assert_matches_reference(shape, dtype)

    def test_zero_input(self) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.zeros(4, 1024, device="cuda", dtype=dtype)
            residual = torch.zeros_like(x)
            weight = torch.ones(1024, device="cuda", dtype=dtype)
            output, residual_output = self.op(x, residual, weight, 1e-6)
            self.assertEqual(torch.count_nonzero(output).item(), 0)
            self.assertEqual(torch.count_nonzero(residual_output).item(), 0)

    def test_rejects_unsupported_hidden_size(self) -> None:
        x = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        weight = torch.ones(256, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "hidden_size 128 and 1024"):
            self.op(x, residual, weight, 1e-6)

    def test_torch_compile_fullgraph(self) -> None:
        compiled = torch.compile(self.op, fullgraph=True)
        x = torch.randn(11, 1024, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        weight = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
        actual, actual_residual = compiled(x, residual, weight, 1e-6)
        expected, expected_residual = reference(x, residual, weight, 1e-6)
        torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    def test_cuda_graph_capture_and_replay(self) -> None:
        x = torch.randn(8, 1024, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        weight = torch.randn(1024, device="cuda", dtype=torch.bfloat16)

        for _ in range(3):
            self.op(x, residual, weight, 1e-6)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output, residual_output = self.op(x, residual, weight, 1e-6)

        x.copy_(torch.randn_like(x))
        residual.copy_(torch.randn_like(residual))
        graph.replay()
        torch.cuda.synchronize()

        expected, expected_residual = reference(x, residual, weight, 1e-6)
        torch.testing.assert_close(residual_output, expected_residual, rtol=0, atol=0)
        torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
