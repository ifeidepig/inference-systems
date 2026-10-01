import torch

from nanovllm.models.qwen35_reference import gated_delta_core_reference


def _make_inputs(sequence_length: int = 5):
    torch.manual_seed(7)
    batch_size = 1
    num_key_heads = 2
    num_value_heads = 4
    key_head_dim = 3
    value_head_dim = 2
    kernel_size = 4
    key_dim = num_key_heads * key_head_dim
    value_dim = num_value_heads * value_head_dim
    conv_dim = 2 * key_dim + value_dim

    inputs = {
        "projected_qkv": torch.randn(batch_size, sequence_length, conv_dim),
        "conv_weight": torch.randn(conv_dim, kernel_size),
        "a": torch.randn(batch_size, sequence_length, num_value_heads),
        "b": torch.randn(batch_size, sequence_length, num_value_heads),
        "a_log": torch.randn(num_value_heads),
        "dt_bias": torch.randn(num_value_heads),
        "num_key_heads": num_key_heads,
        "num_value_heads": num_value_heads,
        "key_head_dim": key_head_dim,
        "value_head_dim": value_head_dim,
    }
    return inputs


def _run_slice(
    inputs,
    start,
    end,
    conv_state=None,
    recurrent_state=None,
    *,
    return_state_history=False,
    checkpoint_indices=None,
):
    return gated_delta_core_reference(
        inputs["projected_qkv"][:, start:end],
        inputs["conv_weight"],
        inputs["a"][:, start:end],
        inputs["b"][:, start:end],
        inputs["a_log"],
        inputs["dt_bias"],
        num_key_heads=inputs["num_key_heads"],
        num_value_heads=inputs["num_value_heads"],
        key_head_dim=inputs["key_head_dim"],
        value_head_dim=inputs["value_head_dim"],
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        return_state_history=return_state_history,
        checkpoint_indices=checkpoint_indices,
    )


def test_arbitrary_chunk_split_matches_full_scan():
    inputs = _make_inputs()
    full = _run_slice(inputs, 0, 5)

    first = _run_slice(inputs, 0, 2)
    second = _run_slice(
        inputs,
        2,
        3,
        first.conv_state,
        first.recurrent_state,
    )
    third = _run_slice(
        inputs,
        3,
        5,
        second.conv_state,
        second.recurrent_state,
    )
    chunked_output = torch.cat((first.output, second.output, third.output), dim=1)

    torch.testing.assert_close(chunked_output, full.output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(third.conv_state, full.conv_state, rtol=0, atol=0)
    torch.testing.assert_close(
        third.recurrent_state,
        full.recurrent_state,
        rtol=1e-5,
        atol=1e-5,
    )


def test_prefill_then_single_token_decode_matches_full_scan():
    inputs = _make_inputs()
    full = _run_slice(inputs, 0, 5)

    prefill = _run_slice(inputs, 0, 4)
    decode = _run_slice(
        inputs,
        4,
        5,
        prefill.conv_state,
        prefill.recurrent_state,
    )

    torch.testing.assert_close(decode.output, full.output[:, 4:5], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(decode.conv_state, full.conv_state, rtol=0, atol=0)
    torch.testing.assert_close(
        decode.recurrent_state,
        full.recurrent_state,
        rtol=1e-5,
        atol=1e-5,
    )


def test_reference_state_shapes_cover_qk_head_replication():
    inputs = _make_inputs(sequence_length=3)
    result = _run_slice(inputs, 0, 3)

    assert result.output.shape == (1, 3, 4, 2)
    assert result.conv_state.shape == (1, 20, 3)
    assert result.recurrent_state.shape == (1, 4, 3, 2)


def test_per_token_state_history_matches_every_prefix_boundary():
    inputs = _make_inputs(sequence_length=5)
    result = _run_slice(
        inputs,
        0,
        5,
        return_state_history=True,
    )

    assert result.conv_state_history.shape == (1, 5, 20, 3)
    assert result.recurrent_state_history.shape == (1, 5, 4, 3, 2)
    for prefix_length in range(1, 6):
        prefix = _run_slice(inputs, 0, prefix_length)
        torch.testing.assert_close(
            result.conv_state_history[:, prefix_length - 1],
            prefix.conv_state,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            result.recurrent_state_history[:, prefix_length - 1],
            prefix.recurrent_state,
            rtol=1e-5,
            atol=1e-5,
        )


def test_sparse_internal_checkpoints_match_full_history_and_split_prefill():
    inputs = _make_inputs(sequence_length=6)
    full_history = _run_slice(
        inputs,
        0,
        6,
        return_state_history=True,
    )
    sparse = _run_slice(
        inputs,
        0,
        6,
        checkpoint_indices=(1, 4),
    )

    assert sparse.conv_state_history.shape == (1, 2, 20, 3)
    assert sparse.recurrent_state_history.shape == (1, 2, 4, 3, 2)
    torch.testing.assert_close(
        sparse.conv_state_history,
        full_history.conv_state_history[:, (1, 4)],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        sparse.recurrent_state_history,
        full_history.recurrent_state_history[:, (1, 4)],
        rtol=1e-5,
        atol=1e-5,
    )
    split_at_two = _run_slice(inputs, 0, 2)
    split_at_five = _run_slice(inputs, 0, 5)
    torch.testing.assert_close(
        sparse.conv_state_history[:, 0],
        split_at_two.conv_state,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        sparse.recurrent_state_history[:, 1],
        split_at_five.recurrent_state,
        rtol=1e-5,
        atol=1e-5,
    )
