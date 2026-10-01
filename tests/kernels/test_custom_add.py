import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class CustomAddTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from nanovllm.kernels import add

        cls.add = staticmethod(add)

    def test_float16_and_bfloat16(self) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = torch.randn(37, 1024, device="cuda", dtype=dtype)
                residual = torch.randn_like(x)
                actual = self.add(x, residual)
                expected = (x.float() + residual.float()).to(dtype)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_uses_current_stream(self) -> None:
        stream = torch.cuda.Stream()
        x = torch.randn(1024, device="cuda", dtype=torch.float16)
        residual = torch.randn_like(x)

        with torch.cuda.stream(stream):
            actual = self.add(x, residual)
            expected = (x.float() + residual.float()).half()

        stream.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rejects_non_contiguous_input(self) -> None:
        x = torch.randn(8, 16, device="cuda", dtype=torch.float16).t()
        residual = torch.randn_like(x)
        with self.assertRaisesRegex(RuntimeError, "x must be contiguous"):
            self.add(x, residual)

    def test_rejects_shape_mismatch(self) -> None:
        x = torch.randn(8, 16, device="cuda", dtype=torch.float16)
        residual = torch.randn(8, 8, device="cuda", dtype=torch.float16)
        with self.assertRaisesRegex(RuntimeError, "same shape"):
            self.add(x, residual)

    def test_torch_compile_fullgraph(self) -> None:
        compiled = torch.compile(self.add, fullgraph=True)
        x = torch.randn(13, 128, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        actual = compiled(x, residual)
        expected = (x.float() + residual.float()).bfloat16()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
