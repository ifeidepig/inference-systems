from types import SimpleNamespace
import unittest

import torch

from nanovllm.kernels.gdn import (
    gdn_decode_core,
    gdn_decode_core_torch,
    validate_gdn_decode_backend,
)
from nanovllm.layers.gated_delta_net import GatedDeltaNet


def inputs():
    torch.manual_seed(41)
    batch_size = 3
    num_key_heads = 2
    num_value_heads = 4
    key_head_dim = 3
    value_head_dim = 2
    conv_dim = 2 * num_key_heads * key_head_dim + num_value_heads * value_head_dim
    arguments = (
        torch.randn(batch_size, conv_dim),
        torch.randn(batch_size, num_value_heads),
        torch.randn(batch_size, num_value_heads),
        torch.randn(conv_dim, 4),
        torch.randn(num_value_heads),
        torch.randn(num_value_heads),
        torch.randn(
            batch_size,
            num_value_heads,
            key_head_dim,
            value_head_dim,
        ),
        torch.randn(batch_size, conv_dim, 3),
    )
    dimensions = dict(
        num_key_heads=num_key_heads,
        num_value_heads=num_value_heads,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
    )
    return arguments, dimensions


class GDNDecodeDispatchTest(unittest.TestCase):

    def test_backend_validation(self) -> None:
        self.assertEqual(validate_gdn_decode_backend(" Torch "), "torch")
        self.assertEqual(validate_gdn_decode_backend("AUTO"), "auto")
        with self.assertRaisesRegex(ValueError, "unsupported GDN decode backend"):
            validate_gdn_decode_backend("triton")

    def test_auto_falls_back_to_torch_for_cpu_inputs(self) -> None:
        arguments, dimensions = inputs()
        expected = gdn_decode_core_torch(*arguments, **dimensions)
        actual = gdn_decode_core(*arguments, **dimensions, backend="auto")
        for actual_tensor, expected_tensor in zip(actual, expected):
            torch.testing.assert_close(actual_tensor, expected_tensor)

    def test_explicit_cuda_rejects_unsupported_inputs(self) -> None:
        arguments, dimensions = inputs()
        with self.assertRaisesRegex(RuntimeError, "all inputs must be CUDA tensors"):
            gdn_decode_core(*arguments, **dimensions, backend="cuda")

    def test_layer_accepts_explicit_backend_setting(self) -> None:
        config = SimpleNamespace(
            hidden_size=8,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=3,
            linear_value_head_dim=2,
            linear_conv_kernel_dim=4,
            rms_norm_eps=1e-6,
            gdn_decode_backend="auto",
        )
        module = GatedDeltaNet(config, layer_idx=0)
        self.assertEqual(module.decode_backend, "auto")


if __name__ == "__main__":
    unittest.main()
