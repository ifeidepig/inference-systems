from dataclasses import dataclass

import torch

from nanovllm.engine.state_manager import HybridStateManager, HybridStateSnapshot


@dataclass(frozen=True, slots=True)
class GreedyVerificationResult:
    accepted_lengths: torch.Tensor
    output_tokens: list[list[int]]


@dataclass(frozen=True, slots=True)
class GreedyCommitPlan:
    accepted_lengths: torch.Tensor
    committed_context_lengths: torch.Tensor
    emitted_tokens: list[list[int]]
    replacement_tokens_require_decode: torch.Tensor


class HybridSpeculativeTransaction:
    """Collect and commit/rollback per-step GDN state snapshots."""

    def __init__(self, state_manager: HybridStateManager, slot_ids: torch.Tensor):
        self.state_manager = state_manager
        self.slot_ids = slot_ids
        self.snapshots: list[HybridStateSnapshot] = [
            state_manager.snapshot(slot_ids)
        ]
        self.closed = False

    def capture_step(self) -> None:
        if self.closed:
            raise RuntimeError("speculative transaction is already closed")
        self.snapshots.append(self.state_manager.snapshot(self.slot_ids))

    def commit(self, accepted_lengths: torch.Tensor) -> None:
        if self.closed:
            raise RuntimeError("speculative transaction is already closed")
        self.state_manager.restore_accepted_prefixes(
            self.snapshots,
            accepted_lengths,
        )
        self.closed = True

    def rollback(self) -> None:
        if self.closed:
            raise RuntimeError("speculative transaction is already closed")
        self.state_manager.restore(self.snapshots[0])
        self.closed = True


def greedy_verify(
    proposal_tokens: torch.Tensor,
    target_logits: torch.Tensor,
) -> GreedyVerificationResult:
    """Verify a ragged batch of fixed-width proposals under greedy decoding.

    ``target_logits[:, i]`` predicts the token at proposal position ``i``.
    The returned token list contains every accepted proposal and, on the first
    rejection, the target model's replacement token.
    """
    if proposal_tokens.ndim != 2 or target_logits.ndim != 3:
        raise ValueError("expected proposals [batch, steps] and logits [batch, steps, vocab]")
    if proposal_tokens.shape != target_logits.shape[:2]:
        raise ValueError("proposal and target step dimensions must match")

    target_tokens = target_logits.argmax(dim=-1)
    return greedy_verify_tokens(proposal_tokens, target_tokens)


def greedy_verify_tokens(
    proposal_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
) -> GreedyVerificationResult:
    """Greedy verification when target argmax tokens are already available."""
    if proposal_tokens.ndim != 2 or target_tokens.ndim != 2:
        raise ValueError("expected proposal and target tokens [batch, steps]")
    if proposal_tokens.shape != target_tokens.shape:
        raise ValueError("proposal and target token dimensions must match")
    batch_size, num_steps = proposal_tokens.shape
    accepted_lengths = torch.zeros(batch_size, dtype=torch.int32, device=proposal_tokens.device)
    output_tokens: list[list[int]] = []
    for batch_index in range(batch_size):
        accepted = 0
        tokens = []
        for step in range(num_steps):
            proposal = int(proposal_tokens[batch_index, step])
            target = int(target_tokens[batch_index, step])
            if proposal != target:
                tokens.append(target)
                break
            tokens.append(proposal)
            accepted += 1
        accepted_lengths[batch_index] = accepted
        output_tokens.append(tokens)
    return GreedyVerificationResult(accepted_lengths, output_tokens)


def build_greedy_commit_plan(
    base_context_lengths: torch.Tensor,
    verification: GreedyVerificationResult,
) -> GreedyCommitPlan:
    """Create logical KV/state commit metadata after greedy verification.

    The replacement target token emitted on a rejection has not yet passed
    through the model, so state and context are committed only through the
    accepted proposal prefix. That replacement is the next decode input.
    """
    if base_context_lengths.shape != verification.accepted_lengths.shape:
        raise ValueError("base_context_lengths must match accepted_lengths")
    committed = base_context_lengths + verification.accepted_lengths.to(
        base_context_lengths.dtype
    )
    replacement_required = torch.tensor(
        [len(tokens) > int(accepted) for tokens, accepted in zip(
            verification.output_tokens,
            verification.accepted_lengths.to("cpu").tolist(),
        )],
        dtype=torch.bool,
        device=base_context_lengths.device,
    )
    return GreedyCommitPlan(
        accepted_lengths=verification.accepted_lengths,
        committed_context_lengths=committed,
        emitted_tokens=verification.output_tokens,
        replacement_tokens_require_decode=replacement_required,
    )
