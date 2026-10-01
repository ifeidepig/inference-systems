from types import SimpleNamespace

import torch

from nanovllm.layers.gated_delta_net import GatedDeltaNet


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


def test_gated_delta_net_chunk_continuation_matches_full_sequence():
    torch.manual_seed(11)
    module = GatedDeltaNet(_config(), layer_idx=0).float().eval()
    hidden_states = torch.randn(5, 8)

    full_output, full_recurrent, full_conv = module(
        hidden_states,
        torch.tensor([0, 5]),
    )
    prefix_output, prefix_recurrent, prefix_conv = module(
        hidden_states[:3],
        torch.tensor([0, 3]),
    )
    continuation_output, continuation_recurrent, continuation_conv = module(
        hidden_states[3:],
        torch.tensor([0, 2]),
        prefix_recurrent,
        prefix_conv,
    )

    torch.testing.assert_close(
        torch.cat((prefix_output, continuation_output), dim=0),
        full_output,
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(continuation_recurrent, full_recurrent)
    # The two paths invoke Linear with different GEMM shapes (5 tokens versus
    # 3+2), so their projected values can differ by a few FP32 ulps even though
    # the state transition is identical. Exact equality is tested separately
    # at the fixed-projection reference level.
    torch.testing.assert_close(continuation_conv, full_conv, rtol=1e-5, atol=1e-6)


def test_gated_delta_net_packed_sequences_match_independent_execution():
    torch.manual_seed(13)
    module = GatedDeltaNet(_config(), layer_idx=0).float().eval()
    first = torch.randn(2, 8)
    second = torch.randn(3, 8)
    packed = torch.cat((first, second), dim=0)

    packed_output, packed_recurrent, packed_conv = module(
        packed,
        torch.tensor([0, 2, 5]),
    )
    first_output, first_recurrent, first_conv = module(first, torch.tensor([0, 2]))
    second_output, second_recurrent, second_conv = module(second, torch.tensor([0, 3]))

    torch.testing.assert_close(
        packed_output,
        torch.cat((first_output, second_output), dim=0),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        packed_recurrent,
        torch.cat((first_recurrent, second_recurrent), dim=0),
    )
    torch.testing.assert_close(
        packed_conv,
        torch.cat((first_conv, second_conv), dim=0),
        rtol=1e-5,
        atol=1e-6,
    )


def test_gated_delta_net_state_shapes():
    module = GatedDeltaNet(_config(), layer_idx=7).float().eval()
    output, recurrent, conv = module(torch.randn(1, 8), torch.tensor([0, 1]))

    assert output.shape == (1, 8)
    assert recurrent.shape == (1, 4, 3, 2)
    assert recurrent.dtype == torch.float32
    assert conv.shape == (1, 20, 3)


def test_batched_single_token_decode_matches_independent_full_scans():
    torch.manual_seed(17)
    module = GatedDeltaNet(_config(), layer_idx=0).float().eval()
    prefixes = torch.randn(3, 2, 8)
    decode_inputs = torch.randn(3, 8)
    packed_prefix = prefixes.reshape(6, 8)
    _, recurrent, conv = module(
        packed_prefix,
        torch.tensor([0, 2, 4, 6]),
    )
    batched_output, batched_recurrent, batched_conv = module(
        decode_inputs,
        torch.tensor([0, 1, 2, 3]),
        recurrent,
        conv,
    )

    expected_outputs = []
    expected_recurrent = []
    expected_conv = []
    for index in range(3):
        full_input = torch.cat((prefixes[index], decode_inputs[index : index + 1]))
        output, final_recurrent, final_conv = module(
            full_input,
            torch.tensor([0, 3]),
        )
        expected_outputs.append(output[-1:])
        expected_recurrent.append(final_recurrent)
        expected_conv.append(final_conv)

    torch.testing.assert_close(
        batched_output,
        torch.cat(expected_outputs),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        batched_recurrent,
        torch.cat(expected_recurrent),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        batched_conv,
        torch.cat(expected_conv),
        rtol=1e-5,
        atol=1e-6,
    )
