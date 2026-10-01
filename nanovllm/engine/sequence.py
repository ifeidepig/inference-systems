from copy import copy
from enum import Enum, auto
from itertools import count
from time import perf_counter_ns

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
    ABORTED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(
        self,
        token_ids: list[int],
        sampling_params=SamplingParams(),
        arrival_time_ns: int | None = None,
        admitted_time_ns: int | None = None,
    ):
        if not token_ids:
            raise ValueError("prompt must contain at least one token")
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.state_slot: int | None = None
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        now_ns = perf_counter_ns()
        self.arrival_time_ns = (
            arrival_time_ns if arrival_time_ns is not None else now_ns
        )
        self.admitted_time_ns = (
            admitted_time_ns if admitted_time_ns is not None else now_ns
        )
        self.first_scheduled_time_ns = None
        self.last_scheduled_time_ns = None
        self.finished_time_ns = None
        self.token_timestamps_ns: list[int] = []
        self.schedule_count = 0
        self.preemption_count = 0
        self.prefix_cache_hit_blocks = 0
        self.peak_kv_blocks = 0

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status in (SequenceStatus.FINISHED, SequenceStatus.ABORTED)

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def mark_scheduled(self, scheduled_at_ns: int | None = None) -> None:
        scheduled_at_ns = (
            scheduled_at_ns if scheduled_at_ns is not None else perf_counter_ns()
        )
        if self.first_scheduled_time_ns is None:
            self.first_scheduled_time_ns = scheduled_at_ns
        self.last_scheduled_time_ns = scheduled_at_ns
        self.schedule_count += 1
        self.peak_kv_blocks = max(self.peak_kv_blocks, len(self.block_table))

    def mark_preempted(self) -> None:
        self.preemption_count += 1

    def append_token(
        self, token_id: int, generated_at_ns: int | None = None
    ) -> None:
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
        self.token_timestamps_ns.append(
            generated_at_ns if generated_at_ns is not None else perf_counter_ns()
        )

    def mark_finished(self, finished_at_ns: int | None = None) -> None:
        self.finished_time_ns = (
            finished_at_ns if finished_at_ns is not None else perf_counter_ns()
        )

    @property
    def first_token_time_ns(self) -> int | None:
        return self.token_timestamps_ns[0] if self.token_timestamps_ns else None

    @property
    def last_token_time_ns(self) -> int | None:
        return self.token_timestamps_ns[-1] if self.token_timestamps_ns else None

    def lifecycle_metrics(self, now_ns: int | None = None) -> dict:
        now_ns = now_ns if now_ns is not None else perf_counter_ns()
        reference_time_ns = self.finished_time_ns or now_ns

        def elapsed_ms(start: int | None, end: int | None) -> float | None:
            if start is None or end is None:
                return None
            return (end - start) / 1e6

        token_gaps_ms = [
            (current - previous) / 1e6
            for previous, current in zip(
                self.token_timestamps_ns, self.token_timestamps_ns[1:]
            )
        ]
        return {
            "request_id": self.seq_id,
            "status": self.status.name.lower(),
            "prompt_tokens": self.num_prompt_tokens,
            "output_tokens": self.num_completion_tokens,
            "admission_delay_ms": elapsed_ms(
                self.arrival_time_ns, self.admitted_time_ns
            ),
            "queue_ms": elapsed_ms(
                self.arrival_time_ns, self.first_scheduled_time_ns
            ),
            "ttft_ms": elapsed_ms(self.arrival_time_ns, self.first_token_time_ns),
            "e2e_ms": elapsed_ms(self.arrival_time_ns, self.finished_time_ns),
            "tpot_ms": (
                sum(token_gaps_ms) / len(token_gaps_ms)
                if token_gaps_ms
                else None
            ),
            "max_token_gap_ms": max(token_gaps_ms) if token_gaps_ms else None,
            "time_since_last_token_ms": elapsed_ms(
                self.last_token_time_ns, reference_time_ns
            ),
            "schedule_count": self.schedule_count,
            "preemption_count": self.preemption_count,
            "prefix_cache_hit_blocks": self.prefix_cache_hit_blocks,
            "peak_kv_blocks": self.peak_kv_blocks,
        }

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.num_scheduled_tokens,
            self.block_table,
            self.state_slot,
            last_state,
        )

    def __setstate__(self, state):
        if len(state) == 6:  # Backward compatibility with older worker payloads.
            (
                self.num_tokens,
                self.num_prompt_tokens,
                self.num_cached_tokens,
                self.num_scheduled_tokens,
                self.block_table,
                last_state,
            ) = state
            self.state_slot = None
        else:
            (
                self.num_tokens,
                self.num_prompt_tokens,
                self.num_cached_tokens,
                self.num_scheduled_tokens,
                self.block_table,
                self.state_slot,
                last_state,
            ) = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
