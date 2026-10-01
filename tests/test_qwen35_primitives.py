import torch

from nanovllm.layers.layernorm import Qwen3_5RMSNorm
from nanovllm.layers.rotary_embedding import RotaryEmbedding, apply_rotary_emb


def test_qwen35_rmsnorm_uses_zero_centered_scale():
    module = Qwen3_5RMSNorm(4, eps=1e-6)
    x = torch.tensor([[1.0, -2.0, 3.0, -4.0]])

    actual = module(x)
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    expected = x.float() * torch.rsqrt(variance + module.eps)
    torch.testing.assert_close(actual, expected)

    with torch.no_grad():
        module.weight.copy_(torch.tensor([0.0, 0.5, -0.25, 1.0]))
    actual = module(x)
    expected = expected * (1.0 + module.weight.float())
    torch.testing.assert_close(actual, expected)


def test_qwen35_add_rmsnorm_preserves_unscaled_residual_stream():
    module = Qwen3_5RMSNorm(4, eps=1e-6)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16)
    residual = torch.tensor([[0.5, -1.0, 1.5, -2.0]], dtype=torch.bfloat16)

    actual, new_residual = module(x, residual)
    summed_fp32 = x.float() + residual.float()
    expected_residual = summed_fp32.to(torch.bfloat16)
    variance = summed_fp32.pow(2).mean(dim=-1, keepdim=True)
    expected = (summed_fp32 * torch.rsqrt(variance + module.eps)).to(torch.bfloat16)

    torch.testing.assert_close(new_residual, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_partial_rope_rotates_prefix_and_preserves_suffix():
    rope = RotaryEmbedding(
        head_size=8,
        rotary_dim=4,
        max_position_embeddings=16,
        base=10_000,
    )
    positions = torch.tensor([0, 3])
    query = torch.arange(16, dtype=torch.float32).reshape(2, 1, 8)
    key = query + 100

    actual_query, actual_key = rope(positions, query, key)
    cos_sin = rope.cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    expected_query_prefix = apply_rotary_emb(query[..., :4], cos, sin)
    expected_key_prefix = apply_rotary_emb(key[..., :4], cos, sin)

    torch.testing.assert_close(actual_query[..., :4], expected_query_prefix)
    torch.testing.assert_close(actual_key[..., :4], expected_key_prefix)
    torch.testing.assert_close(actual_query[..., 4:], query[..., 4:], rtol=0, atol=0)
    torch.testing.assert_close(actual_key[..., 4:], key[..., 4:], rtol=0, atol=0)


def test_full_dimension_rope_keeps_existing_behavior():
    rope = RotaryEmbedding(
        head_size=4,
        rotary_dim=4,
        max_position_embeddings=16,
        base=10_000,
    )
    positions = torch.tensor([1, 2])
    query = torch.randn(2, 2, 4)
    key = torch.randn(2, 1, 4)

    actual_query, actual_key = rope(positions, query, key)
    cos_sin = rope.cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    torch.testing.assert_close(actual_query, apply_rotary_emb(query, cos, sin))
    torch.testing.assert_close(actual_key, apply_rotary_emb(key, cos, sin))
