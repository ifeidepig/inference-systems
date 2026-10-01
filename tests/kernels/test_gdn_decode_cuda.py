import unittest

import torch

from nanovllm.kernels.gdn import gdn_decode_core


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class GDNDecodeCUDATest(unittest.TestCase):

    def make_inputs(self, batch_size: int, dtype: torch.dtype):
        torch.manual_seed(53)
        num_heads = 16
        head_dim = 128
        conv_dim = 3 * num_heads * head_dim
        return (
            (
                torch.randn(batch_size, conv_dim, device="cuda", dtype=dtype),
                torch.randn(batch_size, num_heads, device="cuda", dtype=dtype),
                torch.randn(batch_size, num_heads, device="cuda", dtype=dtype),
                torch.randn(conv_dim, 4, device="cuda", dtype=dtype) * 0.1,
                torch.randn(num_heads, device="cuda", dtype=torch.float32) * 0.1,
                torch.randn(num_heads, device="cuda", dtype=dtype) * 0.1,
                torch.randn(
                    batch_size,
                    num_heads,
                    head_dim,
                    head_dim,
                    device="cuda",
                    dtype=torch.float32,
                )
                * 0.1,
                torch.randn(batch_size, conv_dim, 3, device="cuda", dtype=dtype),
            ),
            dict(
                num_key_heads=num_heads,
                num_value_heads=num_heads,
                key_head_dim=head_dim,
                value_head_dim=head_dim,
            ),
        )

    def assert_matches_torch(self, batch_size: int, dtype: torch.dtype) -> None:
        arguments, dimensions = self.make_inputs(batch_size, dtype)
        expected = gdn_decode_core(
            *arguments,
            **dimensions,
            backend="torch",
        )
        actual = gdn_decode_core(
            *arguments,
            **dimensions,
            backend="cuda",
        )
        tolerance = 3e-3 if dtype == torch.float16 else 3e-2
        torch.testing.assert_close(
            actual[0], expected[0], rtol=tolerance, atol=tolerance
        )
        torch.testing.assert_close(
            actual[1], expected[1], rtol=tolerance, atol=tolerance
        )
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)

    def test_supported_batches_and_dtypes(self) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            for batch_size in (1, 2, 4):
                with self.subTest(dtype=dtype, batch_size=batch_size):
                    self.assert_matches_torch(batch_size, dtype)

    def test_cuda_graph_capture_and_replay(self) -> None:
        arguments, dimensions = self.make_inputs(2, torch.bfloat16)
        for _ in range(3):
            gdn_decode_core(*arguments, **dimensions, backend="cuda")
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = gdn_decode_core(*arguments, **dimensions, backend="cuda")
        arguments[0].copy_(torch.randn_like(arguments[0]))
        graph.replay()
        torch.cuda.synchronize()

        expected = gdn_decode_core(*arguments, **dimensions, backend="torch")
        torch.testing.assert_close(actual[0], expected[0], rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(actual[1], expected[1], rtol=3e-2, atol=3e-2)
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)

    def test_32_step_state_drift_stays_bounded(self) -> None:
        batch_size = 8
        dtype = torch.bfloat16
        arguments, dimensions = self.make_inputs(batch_size, dtype)
        (
            _,
            _,
            _,
            conv_weight,
            A_log,
            dt_bias,
            recurrent,
            conv,
        ) = arguments
        torch_recurrent = recurrent.clone()
        cuda_recurrent = recurrent.clone()
        torch_conv = conv.clone()
        cuda_conv = conv.clone()
        max_core_error = 0.0
        max_state_error = 0.0
        conv_dim = conv.shape[1]
        num_heads = dimensions["num_value_heads"]

        for step in range(32):
            torch.manual_seed(1000 + step)
            projected_qkv = torch.randn(
                batch_size, conv_dim, device="cuda", dtype=dtype
            )
            a = torch.randn(batch_size, num_heads, device="cuda", dtype=dtype)
            b = torch.randn(batch_size, num_heads, device="cuda", dtype=dtype)
            torch_output, torch_recurrent, torch_conv = gdn_decode_core(
                projected_qkv,
                a,
                b,
                conv_weight,
                A_log,
                dt_bias,
                torch_recurrent,
                torch_conv,
                **dimensions,
                backend="torch",
            )
            cuda_output, cuda_recurrent, cuda_conv = gdn_decode_core(
                projected_qkv,
                a,
                b,
                conv_weight,
                A_log,
                dt_bias,
                cuda_recurrent,
                cuda_conv,
                **dimensions,
                backend="cuda",
            )
            max_core_error = max(
                max_core_error,
                (torch_output.float() - cuda_output.float()).abs().max().item(),
            )
            max_state_error = max(
                max_state_error,
                (torch_recurrent - cuda_recurrent).abs().max().item(),
            )

        self.assertLessEqual(max_core_error, 2e-5)
        self.assertLessEqual(max_state_error, 1e-6)
        torch.testing.assert_close(cuda_conv, torch_conv, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
