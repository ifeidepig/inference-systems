from types import SimpleNamespace

import torch
import torch.distributed as dist

from nanovllm.engine.speculative import (
    HybridSpeculativeTransaction,
    build_greedy_commit_plan,
    greedy_verify,
)
from nanovllm.engine.state_manager import HybridStateManager
from nanovllm.models.qwen3_5_mtp import Qwen3_5MTP
from nanovllm.models.registry import create_model, normalize_hf_config
from nanovllm.utils.context import reset_context, set_context


def _config():
    text = SimpleNamespace(
        model_type="qwen3_5_text",
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        hidden_act="silu",
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        attention_bias=False,
        tie_word_embeddings=True,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
        rope_parameters={
            "partial_rotary_factor": 0.25,
            "rope_theta": 10_000,
        },
    )
    full = SimpleNamespace(
        model_type="qwen3_5",
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config=text,
    )
    return text, normalize_hf_config(full)[1]


def _ensure_group():
    if dist.is_initialized():
        return False
    dist.init_process_group(
        "gloo",
        init_method="tcp://127.0.0.1:29642",
        rank=0,
        world_size=1,
    )
    return True


def test_mtp_head_uses_shared_embedding_and_produces_hidden_states():
    if not torch.cuda.is_available():
        return
    owns_group = _ensure_group()
    try:
        config, capabilities = _config()
        target = create_model(config, capabilities).cuda().to(torch.bfloat16).eval()
        mtp = Qwen3_5MTP(config, target.model.embed_tokens).cuda().to(torch.bfloat16).eval()
        torch.manual_seed(43)
        with torch.no_grad():
            for parameter in mtp.parameters():
                parameter.normal_(0.0, 0.02)
        assert mtp._embed_tokens is target.model.embed_tokens
        input_ids = torch.tensor([1, 2, 3], device="cuda")
        positions = torch.arange(3, device="cuda")
        hidden = torch.randn(3, config.hidden_size, dtype=torch.bfloat16, device="cuda")
        cu = torch.tensor([0, 3], dtype=torch.int32, device="cuda")
        set_context(
            True,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=3,
            max_seqlen_k=3,
            slot_mapping=torch.empty(0, dtype=torch.int32, device="cuda"),
        )
        output = mtp(input_ids, positions, hidden, cu)
        assert output.shape == hidden.shape
    finally:
        reset_context()
        if owns_group:
            dist.destroy_process_group()


def test_greedy_verification_handles_different_accept_lengths():
    proposals = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    target_tokens = torch.tensor([[1, 2, 3], [4, 0, 6], [0, 8, 9]])
    logits = torch.full((3, 3, 10), -100.0)
    logits.scatter_(2, target_tokens.unsqueeze(-1), 100.0)

    result = greedy_verify(proposals, logits)
    assert result.accepted_lengths.tolist() == [3, 1, 0]
    assert result.output_tokens == [[1, 2, 3], [4, 0], [0]]

    plan = build_greedy_commit_plan(torch.tensor([10, 20, 30]), result)
    assert plan.committed_context_lengths.tolist() == [13, 21, 30]
    assert plan.replacement_tokens_require_decode.tolist() == [False, True, True]


def test_hybrid_speculative_transaction_commits_variable_state_boundaries():
    manager = HybridStateManager(
        max_num_seqs=2,
        num_linear_layers=1,
        num_value_heads=1,
        key_head_dim=2,
        value_head_dim=2,
        conv_dim=3,
        conv_kernel_size=4,
        conv_dtype=torch.bfloat16,
        device="cpu",
    )
    slots = torch.tensor([manager.allocate(), manager.allocate()])
    transaction = HybridSpeculativeTransaction(manager, slots)
    manager.recurrent_states[:, slots[0]].fill_(10)
    manager.recurrent_states[:, slots[1]].fill_(11)
    transaction.capture_step()
    manager.recurrent_states[:, slots[0]].fill_(20)
    manager.recurrent_states[:, slots[1]].fill_(21)
    transaction.capture_step()

    transaction.commit(torch.tensor([2, 0]))
    assert torch.all(manager.recurrent_states[:, slots[0]] == 20)
    assert not manager.recurrent_states[:, slots[1]].any()


def test_hybrid_speculative_transaction_full_rollback():
    manager = HybridStateManager(
        max_num_seqs=1,
        num_linear_layers=1,
        num_value_heads=1,
        key_head_dim=2,
        value_head_dim=2,
        conv_dim=3,
        conv_kernel_size=4,
        conv_dtype=torch.bfloat16,
        device="cpu",
    )
    slot = manager.allocate()
    manager.recurrent_states[:, slot].fill_(3)
    transaction = HybridSpeculativeTransaction(manager, torch.tensor([slot]))
    manager.recurrent_states[:, slot].fill_(9)
    transaction.capture_step()
    transaction.rollback()
    assert torch.all(manager.recurrent_states[:, slot] == 3)
