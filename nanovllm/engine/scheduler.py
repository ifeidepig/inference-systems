from collections import deque
from dataclasses import dataclass

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


@dataclass
class ScheduledBatch:
    seqs: list[Sequence]
    is_prefill: bool

    @property
    def num_tokens(self) -> int:
        if self.is_prefill:
            return sum(seq.num_scheduled_tokens for seq in self.seqs)
        return len(self.seqs)


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.scheduling_policy = config.scheduling_policy
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> list[ScheduledBatch]:
        if self.scheduling_policy == "prefill_first":
            prefill = self._schedule_prefill(
                self.max_num_batched_tokens, self.max_num_seqs
            )
            if prefill:
                return [ScheduledBatch(prefill, is_prefill=True)]
            decode = self._schedule_decode(self.max_num_seqs)
            assert decode
            return [ScheduledBatch(decode, is_prefill=False)]

        decode = self._schedule_decode(self.max_num_seqs)
        remaining_tokens = self.max_num_batched_tokens - len(decode)
        remaining_seqs = self.max_num_seqs - len(decode)
        prefill = self._schedule_prefill(remaining_tokens, remaining_seqs)
        batches = []
        if decode:
            batches.append(ScheduledBatch(decode, is_prefill=False))
        if prefill:
            batches.append(ScheduledBatch(prefill, is_prefill=True))
        assert batches
        return batches

    def _schedule_prefill(
        self, token_budget: int, sequence_budget: int
    ) -> list[Sequence]:
        scheduled_seqs = []
        num_batched_tokens = 0

        while self.waiting and len(scheduled_seqs) < sequence_budget:
            seq = self.waiting[0]
            remaining = token_budget - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        return scheduled_seqs

    def _schedule_decode(self, sequence_budget: int) -> list[Sequence]:
        scheduled_seqs = []
        while self.running and len(scheduled_seqs) < sequence_budget:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
