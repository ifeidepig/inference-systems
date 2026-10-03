import torch

from nanovllm.engine.state_manager import (
    HybridPrefixCheckpointPool,
    HybridStateManager,
)
from nanovllm.models.qwen35_reference import (
    recurrent_gated_delta_reference,
)


def _compressed_state(initial_state: torch.Tensor, checkpoint_dtype: str):
    _, heads, key_dim, value_dim = initial_state.shape
    manager = HybridStateManager(
        max_num_seqs=1,
        num_linear_layers=1,
        num_value_heads=heads,
        key_head_dim=key_dim,
        value_head_dim=value_dim,
        conv_dim=4,
        conv_kernel_size=4,
        conv_dtype=torch.bfloat16,
        device="cpu",
    )
    slot = manager.allocate()
    manager.recurrent_states[:, slot].copy_(initial_state)
    pool = HybridPrefixCheckpointPool(
        manager,
        memory_budget_bytes=manager.bytes_per_slot(),
        checkpoint_dtype=checkpoint_dtype,
    )
    checkpoint_slot = pool.capture(slot)
    manager.recurrent_states[:, slot].zero_()
    pool.restore(checkpoint_slot, slot)
    return manager.recurrent_states[:, slot].clone()


def test_compressed_checkpoint_recurrent_drift_stays_bounded_for_1024_steps():
    torch.manual_seed(11)
    heads, key_dim, value_dim = 2, 8, 8
    prefix_tokens, continuation_tokens = 64, 1024
    total_tokens = prefix_tokens + continuation_tokens
    query = torch.randn(1, total_tokens, heads, key_dim)
    key = torch.randn_like(query)
    value = torch.randn(1, total_tokens, heads, value_dim) * 0.25
    log_decay = torch.full((1, total_tokens, heads), -0.05)
    beta = torch.full((1, total_tokens, heads), 0.2)
    _, initial_state = recurrent_gated_delta_reference(
        query[:, :prefix_tokens],
        key[:, :prefix_tokens],
        value[:, :prefix_tokens],
        log_decay[:, :prefix_tokens],
        beta[:, :prefix_tokens],
    )
    continuation = slice(prefix_tokens, None)

    for checkpoint_dtype in ("bf16", "int8"):
        reference_state = initial_state
        compressed_state = _compressed_state(
            initial_state, checkpoint_dtype
        )
        start = 0
        for end in (1, 32, 128, 1024):
            local = slice(start, end)
            reference_output, reference_state = (
                recurrent_gated_delta_reference(
                    query[:, continuation][:, local],
                    key[:, continuation][:, local],
                    value[:, continuation][:, local],
                    log_decay[:, continuation][:, local],
                    beta[:, continuation][:, local],
                    reference_state,
                )
            )
            compressed_output, compressed_state = (
                recurrent_gated_delta_reference(
                    query[:, continuation][:, local],
                    key[:, continuation][:, local],
                    value[:, continuation][:, local],
                    log_decay[:, continuation][:, local],
                    beta[:, continuation][:, local],
                    compressed_state,
                )
            )
            assert torch.isfinite(compressed_state).all()
            assert (
                compressed_state - reference_state
            ).abs().max().item() < 1e-3
            assert (
                compressed_output - reference_output
            ).abs().max().item() < 3e-4
            start = end
