from collections import deque
from dataclasses import dataclass
from time import perf_counter_ns

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.phase_profiler import PhaseProfiler
from nanovllm.engine.hybrid_prefix_cache import (
    CheckpointBoundary,
    FullAttentionPrefixManager,
    GDNCheckpointManager,
    HybridCheckpointRetentionPolicy,
    HybridPrefixCoordinator,
    HybridPrefixHit,
    PendingHybridPrefixCapture,
    PrefixKVCandidate,
)


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

    def __init__(
        self,
        config: Config,
        state_manager=None,
        state_controller=None,
        prefix_checkpoint_pool=None,
    ):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.scheduling_policy = config.scheduling_policy
        self.target_ttft_ms = getattr(config, "scheduler_target_ttft_ms", 200.0)
        self.target_tpot_ms = getattr(config, "scheduler_target_tpot_ms", 50.0)
        self.slo_prefill_priority_threshold = getattr(
            config, "slo_prefill_priority_threshold", 0.8
        )
        self.slo_min_prefill_tokens = getattr(
            config, "slo_min_prefill_tokens", 64
        )
        self.slo_kv_pressure_threshold = getattr(
            config, "slo_kv_pressure_threshold", 0.9
        )
        self.slo_queue_pressure_threshold = getattr(
            config, "slo_queue_pressure_threshold", 3
        )
        self.slo_latency_safety_margin_ms = getattr(
            config, "slo_latency_safety_margin_ms", 5.0
        )
        self.ewma_alpha = getattr(config, "scheduler_ewma_alpha", 0.2)
        self.state_manager = state_manager
        self.state_controller = state_controller
        self.enable_hybrid_prefix_cache = bool(
            getattr(config, "enable_hybrid_prefix_cache", False)
            and state_manager is not None
        )
        self.enable_hybrid_internal_checkpoints = bool(
            self.enable_hybrid_prefix_cache
            and getattr(config, "enable_hybrid_internal_checkpoints", False)
        )
        self.hybrid_prefix_cache = None
        self.hybrid_prefix_coordinator = None
        self.hybrid_prefix_promotions: dict[
            int, tuple[CheckpointBoundary, int, PrefixKVCandidate, int]
        ] = {}
        self.hybrid_prefix_checkpoint_interval_tokens = 0
        if self.enable_hybrid_prefix_cache:
            if prefix_checkpoint_pool is None:
                raise ValueError("hybrid prefix cache requires a checkpoint pool")
            self.hybrid_prefix_cache = GDNCheckpointManager(
                prefix_checkpoint_pool.capacity,
                eviction_policy=getattr(
                    config, "hybrid_prefix_eviction_policy", "lru"
                ),
            )
            self.hybrid_prefix_checkpoint_interval_tokens = (
                getattr(
                    config,
                    "hybrid_prefix_checkpoint_interval_tokens",
                    None,
                )
                or config.hybrid_prefix_checkpoint_interval_blocks
                * config.kvcache_block_size
            )
            self.hybrid_prefix_retention_policy = (
                HybridCheckpointRetentionPolicy(
                    block_size=getattr(
                        config, "prefix_match_unit", config.kvcache_block_size
                    ),
                    interval_tokens=(
                        self.hybrid_prefix_checkpoint_interval_tokens
                    ),
                    mode=getattr(
                        config,
                        "hybrid_prefix_retention_policy",
                        "periodic",
                    ),
                )
            )
        else:
            self.hybrid_prefix_retention_policy = None
        # Prefix hits are invalid until conv/recurrent snapshots are stored at
        # the same token boundary as Paged KV.
        self.enable_prefix_cache = (
            getattr(config, "enable_prefix_cache", True)
            and (state_manager is None or self.enable_hybrid_prefix_cache)
        )
        self.enable_chunked_prefill = getattr(
            config, "enable_chunked_prefill", True
        )
        self.num_speculative_tokens = getattr(
            config, "num_speculative_tokens", 0
        )
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            prefix_match_unit=getattr(
                config, "prefix_match_unit", config.kvcache_block_size
            ),
        )
        self.full_attention_prefix_manager = FullAttentionPrefixManager(
            self.block_manager
        )
        if self.hybrid_prefix_cache is not None:
            self.hybrid_prefix_coordinator = HybridPrefixCoordinator(
                self.full_attention_prefix_manager,
                self.hybrid_prefix_cache,
                promote_shared_junctions=(
                    getattr(
                        config,
                        "hybrid_prefix_retention_policy",
                        "periodic",
                    )
                    == "adaptive"
                ),
                promotion_min_sightings=getattr(
                    config,
                    "hybrid_prefix_promotion_min_sightings",
                    2,
                ),
                demand_capacity=max(
                    prefix_checkpoint_pool.capacity * 16,
                    64,
                ),
            )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        history_size = getattr(config, "request_metrics_history_size", 1024)
        self.completed_request_metrics: deque[dict] = deque(maxlen=history_size)
        self.prefill_ms_per_token_ewma = None
        self.prefill_base_ms = None
        self.decode_step_ms_ewma = None
        self.mtp_phase_profiler = PhaseProfiler(
            getattr(config, "enable_mtp_phase_profiling", False)
        )
        self.reset_metrics()

    def _allocate_state(self, seq: Sequence) -> None:
        if self.state_manager is None:
            return
        if self.state_controller is not None:
            self.state_controller.call("allocate_state_slot", seq)
        else:
            self.state_manager.allocate(seq)

    def _free_state(self, seq: Sequence) -> None:
        if self.state_manager is None or seq.state_slot is None:
            return
        if self.state_controller is not None:
            self.state_controller.call("free_state_slot", seq)
        else:
            self.state_manager.free(seq)

    def _restore_prefix_state(
        self,
        seq: Sequence,
        hit: HybridPrefixHit,
    ) -> None:
        if self.state_controller is None:
            raise RuntimeError("hybrid prefix restore requires a state controller")
        self.state_controller.call(
            "restore_prefix_checkpoint",
            seq,
            hit.checkpoint_slot,
        )

    def _copy_prefix_kv(self, page_copy) -> None:
        if self.state_controller is None:
            raise RuntimeError("partial prefix COW requires a model controller")
        self.state_controller.call(
            "copy_prefix_kv",
            page_copy.source_block_id,
            page_copy.destination_block_id,
            page_copy.num_tokens,
        )

    def reset_metrics(self) -> None:
        self.mtp_phase_profiler.reset()
        self.block_manager.reset_metrics()
        self.prefix_cache_queries = 0
        self.prefix_cache_eligible_blocks = 0
        self.prefix_cache_hit_blocks = 0
        self.chunked_prefill_steps = 0
        self.preemption_count = 0
        self.aborted_request_count = 0
        self.completed_request_count = 0
        self.completed_request_metrics.clear()
        self.slo_prefill_priority_steps = 0
        self.slo_decode_priority_steps = 0
        self.slo_admission_deferred_steps = 0
        self.slo_dynamic_prefill_tokens = 0
        self.slo_rejected_latency_observations = 0
        self.slo_infeasible_budget_steps = 0
        self.slo_throughput_priority_steps = 0
        self.last_slo_decision = None
        if self.hybrid_prefix_coordinator is not None:
            self.hybrid_prefix_coordinator.reset_metrics()

    def get_metrics(self) -> dict[str, object]:
        eligible = self.prefix_cache_eligible_blocks
        metrics = {
            "prefix_cache_enabled": self.enable_prefix_cache,
            "prefix_cache_queries": self.prefix_cache_queries,
            "prefix_cache_eligible_blocks": eligible,
            "prefix_cache_hit_blocks": self.prefix_cache_hit_blocks,
            "prefix_cache_hit_rate": (
                self.prefix_cache_hit_blocks / eligible if eligible else 0.0
            ),
            "chunked_prefill_enabled": self.enable_chunked_prefill,
            "chunked_prefill_steps": self.chunked_prefill_steps,
            "preemption_count": self.preemption_count,
            "aborted_request_count": self.aborted_request_count,
            "completed_request_count": self.completed_request_count,
            "slo_prefill_priority_steps": self.slo_prefill_priority_steps,
            "slo_decode_priority_steps": self.slo_decode_priority_steps,
            "slo_admission_deferred_steps": self.slo_admission_deferred_steps,
            "slo_dynamic_prefill_tokens": self.slo_dynamic_prefill_tokens,
            "slo_rejected_latency_observations": (
                self.slo_rejected_latency_observations
            ),
            "slo_infeasible_budget_steps": self.slo_infeasible_budget_steps,
            "slo_throughput_priority_steps": self.slo_throughput_priority_steps,
            "last_slo_decision": self.last_slo_decision,
            "prefill_ms_per_token_ewma": self.prefill_ms_per_token_ewma,
            "prefill_base_ms": self.prefill_base_ms,
            "decode_step_ms_ewma": self.decode_step_ms_ewma,
            "kv_blocks_total": len(self.block_manager.blocks),
            "kv_blocks_used": len(self.block_manager.used_block_ids),
            "kv_blocks_free": len(self.block_manager.free_block_ids),
            "kv_blocks_cached": self.block_manager.num_cached_blocks,
            "kv_blocks_cached_reachable": (
                self.block_manager.num_reachable_cached_blocks
            ),
            "prefix_match_unit": self.block_manager.prefix_match_unit,
            "partial_prefix_cow_allocations": (
                self.block_manager.partial_prefix_cow_allocations
            ),
            "prefix_cache_evictions": (
                self.block_manager.prefix_cache_evictions
            ),
            "prefix_cache_cached_block_reassignments": (
                self.block_manager.cached_block_reassignments
            ),
            "prefix_cache_duplicate_registrations": (
                self.block_manager.duplicate_cache_registrations
            ),
        }
        if self.hybrid_prefix_coordinator is not None:
            metrics.update(self.hybrid_prefix_coordinator.get_metrics())
        metrics["mtp_scheduler_phase_profile"] = (
            self.mtp_phase_profiler.metrics()
        )
        return metrics

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def abort(self, seq_id: int) -> bool:
        for queue in (self.waiting, self.running):
            for seq in queue:
                if seq.seq_id != seq_id:
                    continue
                queue.remove(seq)
                self._cancel_sequence_promotion(seq)
                if seq.block_table:
                    self.block_manager.deallocate(seq)
                if self.state_manager is not None:
                    self._free_state(seq)
                seq.num_scheduled_tokens = 0
                seq.status = SequenceStatus.ABORTED
                seq.mark_finished()
                self.aborted_request_count += 1
                self.completed_request_count += 1
                self.completed_request_metrics.append(seq.lifecycle_metrics())
                return True
        return False

    def schedule(self) -> list[ScheduledBatch]:
        if self.scheduling_policy == "slo_aware":
            return self._schedule_slo_aware()
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

    def _schedule_slo_aware(
        self, now_ns: int | None = None
    ) -> list[ScheduledBatch]:
        now_ns = now_ns if now_ns is not None else perf_counter_ns()
        if not self.running:
            waiting_urgency = self._waiting_urgency(now_ns)
            prefill = self._schedule_prefill(
                self.max_num_batched_tokens, self.max_num_seqs
            )
            assert prefill
            self.slo_prefill_priority_steps += 1
            self._record_slo_decision(
                "prefill_only", 0.0, waiting_urgency,
                self.max_num_batched_tokens,
            )
            return [ScheduledBatch(prefill, is_prefill=True)]
        if not self.waiting:
            decode = self._schedule_decode(self.max_num_seqs)
            assert decode
            self.slo_decode_priority_steps += 1
            self._record_slo_decision(
                "decode_only", self._decode_urgency(now_ns), 0.0, 0
            )
            return [ScheduledBatch(decode, is_prefill=False)]

        decode_urgency = self._decode_urgency(now_ns)
        waiting_urgency = self._waiting_urgency(now_ns)
        decode = self._schedule_decode(self.max_num_seqs)
        remaining_tokens = self.max_num_batched_tokens - len(decode)
        remaining_seqs = self.max_num_seqs - len(decode)
        prefill = []
        prefill_budget = 0
        kv_pressure = self._kv_pressure()
        queue_pressure = len(self.waiting) >= self.slo_queue_pressure_threshold
        should_defer = (
            kv_pressure >= self.slo_kv_pressure_threshold
            and waiting_urgency < self.slo_prefill_priority_threshold
        )
        if queue_pressure and remaining_tokens > 0 and remaining_seqs > 0:
            prefill_budget = remaining_tokens
            prefill = self._schedule_prefill(prefill_budget, remaining_seqs)
            self.slo_dynamic_prefill_tokens += sum(
                seq.num_scheduled_tokens for seq in prefill
            )
            self.slo_throughput_priority_steps += 1
        elif should_defer:
            self.slo_admission_deferred_steps += 1
        elif remaining_tokens > 0 and remaining_seqs > 0:
            prefill_budget = self._latency_bounded_prefill_budget(
                remaining_tokens, waiting_urgency
            )
            if prefill_budget:
                prefill = self._schedule_prefill(prefill_budget, remaining_seqs)
                self.slo_dynamic_prefill_tokens += sum(
                    seq.num_scheduled_tokens for seq in prefill
                )
            else:
                self.slo_admission_deferred_steps += 1

        self.slo_decode_priority_steps += 1
        self._record_slo_decision(
            "decode_first" if prefill else "decode_only",
            decode_urgency,
            waiting_urgency,
            prefill_budget,
        )
        batches = [ScheduledBatch(decode, is_prefill=False)] if decode else []
        if prefill:
            batches.append(ScheduledBatch(prefill, is_prefill=True))
        assert batches
        return batches

    def _latency_bounded_prefill_budget(
        self, remaining_tokens: int, waiting_urgency: float
    ) -> int:
        if (
            self.prefill_ms_per_token_ewma is None
            or self.decode_step_ms_ewma is None
        ):
            return remaining_tokens
        available_ms = max(
            self.target_tpot_ms
            - self.decode_step_ms_ewma
            - self.slo_latency_safety_margin_ms,
            0.0,
        )
        full_chunk_ms = (
            self.prefill_ms_per_token_ewma * self.max_num_batched_tokens
        )
        if full_chunk_ms <= available_ms or self.prefill_base_ms is None:
            return remaining_tokens
        if available_ms <= self.prefill_base_ms:
            self.slo_infeasible_budget_steps += 1
            if waiting_urgency >= self.slo_prefill_priority_threshold:
                return remaining_tokens
            return 0
        incremental_ms = full_chunk_ms - self.prefill_base_ms
        if incremental_ms <= 0:
            latency_budget = 0
        else:
            incremental_ms_per_token = (
                incremental_ms / self.max_num_batched_tokens
            )
            latency_budget = int(
                (available_ms - self.prefill_base_ms)
                / incremental_ms_per_token
            )
        return min(
            remaining_tokens,
            max(min(self.slo_min_prefill_tokens, remaining_tokens), latency_budget),
        )

    def observe_batch(
        self, is_prefill: bool, num_tokens: int, duration_ms: float
    ) -> None:
        if not is_prefill:
            phase_name = (
                "mtp.engine_model_runner_wall"
                if self.num_speculative_tokens
                else "target_only.engine_model_runner_wall"
            )
            self.mtp_phase_profiler.record_cpu(phase_name, duration_ms)
        if is_prefill:
            self.prefill_base_ms = (
                duration_ms
                if self.prefill_base_ms is None
                else min(self.prefill_base_ms, duration_ms)
            )
            full_chunk_threshold = self.max_num_batched_tokens * 0.8
            if num_tokens >= full_chunk_threshold:
                observation = duration_ms / num_tokens
                if (
                    self.prefill_ms_per_token_ewma is not None
                    and observation > self.prefill_ms_per_token_ewma * 2
                ):
                    self.slo_rejected_latency_observations += 1
                else:
                    self.prefill_ms_per_token_ewma = self._update_ewma(
                        self.prefill_ms_per_token_ewma, observation
                    )
        else:
            self.decode_step_ms_ewma = self._update_ewma(
                self.decode_step_ms_ewma, duration_ms
            )

    def _update_ewma(
        self, previous: float | None, observation: float
    ) -> float:
        if previous is None:
            return observation
        return self.ewma_alpha * observation + (1 - self.ewma_alpha) * previous

    def _waiting_urgency(self, now_ns: int) -> float:
        if not self.waiting:
            return 0.0
        oldest_arrival = min(seq.arrival_time_ns for seq in self.waiting)
        return max((now_ns - oldest_arrival) / 1e6 / self.target_ttft_ms, 0.0)

    def _decode_urgency(self, now_ns: int) -> float:
        urgencies = []
        for seq in self.running:
            reference_ns = (
                seq.last_token_time_ns
                or seq.last_scheduled_time_ns
                or seq.arrival_time_ns
            )
            urgencies.append(
                max((now_ns - reference_ns) / 1e6 / self.target_tpot_ms, 0.0)
            )
        return max(urgencies, default=0.0)

    def _kv_pressure(self) -> float:
        total_blocks = len(self.block_manager.blocks)
        return len(self.block_manager.used_block_ids) / total_blocks

    def _record_slo_decision(
        self,
        decision: str,
        decode_urgency: float,
        waiting_urgency: float,
        prefill_budget: int,
    ) -> None:
        self.last_slo_decision = {
            "decision": decision,
            "decode_urgency": decode_urgency,
            "waiting_urgency": waiting_urgency,
            "prefill_budget": prefill_budget,
            "kv_pressure": self._kv_pressure(),
        }

    def _schedule_prefill(
        self, token_budget: int, sequence_budget: int
    ) -> list[Sequence]:
        scheduled_seqs = []
        num_batched_tokens = 0

        while self.waiting and len(scheduled_seqs) < sequence_budget:
            seq = self.waiting[0]
            hybrid_prefix_hit = None
            plan = None
            remaining = token_budget - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                if self.enable_hybrid_prefix_cache:
                    self.prefix_cache_queries += 1
                    self.prefix_cache_eligible_blocks += max(
                        seq.num_blocks - 1, 0
                    )
                    plan = self.hybrid_prefix_coordinator.plan(seq)
                    if plan is None:
                        break
                    hybrid_prefix_hit = plan.hit
                    num_cached_blocks = plan.num_cached_blocks
                else:
                    if self.enable_prefix_cache:
                        self.prefix_cache_queries += 1
                        self.prefix_cache_eligible_blocks += max(seq.num_blocks - 1, 0)
                    num_cached_blocks = self.block_manager.can_allocate(
                        seq, use_prefix_cache=self.enable_prefix_cache
                    )
                    if num_cached_blocks == -1:
                        break
                if (
                    self.state_manager is not None
                    and seq.state_slot is None
                    and not self.state_manager.can_allocate
                ):
                    break
                num_tokens = (
                    seq.num_tokens - plan.boundary_tokens
                    if plan is not None
                    else seq.num_tokens - num_cached_blocks * self.block_size
                )
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens:
                if scheduled_seqs:
                    break
                if not self.enable_chunked_prefill:
                    remaining = num_tokens
            if not seq.block_table:
                page_copy = None
                if plan is not None:
                    try:
                        page_copy = self.full_attention_prefix_manager.allocate(
                            seq,
                            plan.candidate,
                        )
                        if self.state_manager is not None:
                            self._allocate_state(seq)
                        if page_copy is not None:
                            self._copy_prefix_kv(page_copy)
                        if hybrid_prefix_hit is not None:
                            self._restore_prefix_state(seq, hybrid_prefix_hit)
                        self.hybrid_prefix_coordinator.commit(plan)
                        if plan.promotion_candidate is not None:
                            self.hybrid_prefix_promotions[seq.seq_id] = (
                                CheckpointBoundary(
                                    plan.promotion_candidate.boundary_tokens,
                                    "shared_junction",
                                ),
                                plan.promotion_candidate.boundary_tokens
                                - plan.boundary_tokens,
                                plan.promotion_candidate,
                                self.hybrid_prefix_coordinator
                                .demand_observations(
                                    plan.promotion_candidate
                                ),
                            )
                    except Exception:
                        self.hybrid_prefix_coordinator.cancel_promotion(
                            plan.promotion_candidate
                        )
                        if page_copy is not None:
                            self.block_manager.release_cow_source(page_copy)
                        if seq.state_slot is not None:
                            self._free_state(seq)
                        if seq.block_table:
                            self.block_manager.deallocate(seq)
                        raise
                    if page_copy is not None:
                        self.block_manager.release_cow_source(page_copy)
                else:
                    self.block_manager.allocate(seq, num_cached_blocks)
                    if self.state_manager is not None:
                        self._allocate_state(seq)
                self.prefix_cache_hit_blocks += num_cached_blocks
                seq.prefix_cache_hit_blocks += num_cached_blocks
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            if (
                self.enable_hybrid_prefix_cache
                and not self.enable_hybrid_internal_checkpoints
            ):
                start = seq.num_cached_tokens
                retained = self._next_retained_boundary(seq, start)
                if retained is not None:
                    distance = retained.boundary_tokens - start
                    if 0 < distance < seq.num_scheduled_tokens:
                        seq.num_scheduled_tokens = distance
            seq.mark_scheduled()
            if seq.num_scheduled_tokens < num_tokens:
                self.chunked_prefill_steps += 1
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        return scheduled_seqs

    def _schedule_decode(self, sequence_budget: int) -> list[Sequence]:
        profiling_started = perf_counter_ns()
        scheduled_seqs = []
        append_reservation = self.num_speculative_tokens + 1
        while self.running and len(scheduled_seqs) < sequence_budget:
            seq = self.running.popleft()
            def has_capacity() -> bool:
                if self.num_speculative_tokens:
                    return self.block_manager.can_reserve_append(
                        seq, append_reservation
                    )
                return self.block_manager.can_append(seq)

            while not has_capacity():
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                if self.num_speculative_tokens:
                    with self.mtp_phase_profiler.phase(
                        "scheduler.kv_reserve", gpu=False
                    ):
                        self.block_manager.reserve_append(
                            seq, append_reservation
                        )
                else:
                    self.block_manager.may_append(seq)
                seq.mark_scheduled()
                scheduled_seqs.append(seq)
        self.running.extendleft(reversed(scheduled_seqs))
        self.mtp_phase_profiler.record_cpu(
            "scheduler.decode_admission",
            (perf_counter_ns() - profiling_started) / 1e6,
        )
        return scheduled_seqs

    def preempt(self, seq: Sequence):
        self.preemption_count += 1
        seq.mark_preempted()
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self._cancel_sequence_promotion(seq)
        self.block_manager.deallocate(seq)
        if self.state_manager is not None:
            self._free_state(seq)
        self.waiting.appendleft(seq)

    def _cancel_sequence_promotion(self, seq: Sequence) -> None:
        promotion = self.hybrid_prefix_promotions.pop(seq.seq_id, None)
        if promotion is None or self.hybrid_prefix_coordinator is None:
            return
        _, _, candidate, _ = promotion
        self.hybrid_prefix_coordinator.cancel_promotion(candidate)

    def _pending_promotion(
        self,
        seq: Sequence,
    ) -> tuple[
        CheckpointBoundary, int, PrefixKVCandidate, int
    ] | None:
        return self.hybrid_prefix_promotions.get(seq.seq_id)

    def _next_retained_boundary(
        self,
        seq: Sequence,
        start_tokens: int,
    ) -> CheckpointBoundary | None:
        retained = self.hybrid_prefix_retention_policy.next_retained_boundary(
            start_tokens,
            seq.num_tokens,
        )
        promotion = self._pending_promotion(seq)
        if promotion is None:
            return retained
        promoted_boundary, _, _, _ = promotion
        if promoted_boundary.boundary_tokens <= start_tokens:
            return retained
        if (
            retained is None
            or promoted_boundary.boundary_tokens <= retained.boundary_tokens
        ):
            return promoted_boundary
        return retained

    def _classify_capture_boundary(
        self,
        seq: Sequence,
        boundary_tokens: int,
    ) -> CheckpointBoundary | None:
        promotion = self._pending_promotion(seq)
        if (
            promotion is not None
            and promotion[0].boundary_tokens == boundary_tokens
        ):
            return promotion[0]
        return self.hybrid_prefix_retention_policy.classify(
            boundary_tokens,
            seq.num_tokens,
        )

    def prepare_prefix_captures(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
    ) -> tuple[list[PendingHybridPrefixCapture], bool]:
        if not is_prefill or not self.enable_hybrid_prefix_cache:
            return [], False
        eligible = []
        retained_by_sequence = {}
        for seq in seqs:
            end = seq.num_cached_tokens + seq.num_scheduled_tokens
            if self.enable_hybrid_internal_checkpoints:
                retained = self._internal_retained_boundaries(seq, end)
            else:
                item = self._classify_capture_boundary(seq, end)
                retained = [item] if item is not None else []
            if retained and seq.num_scheduled_tokens > 0:
                eligible.append(seq)
                retained_by_sequence[seq.seq_id] = retained
        if not eligible:
            return [], False
        for seq in seqs:
            self.block_manager.hash_blocks(seq)
        pending = []
        for seq in eligible:
            step_end = seq.num_cached_tokens + seq.num_scheduled_tokens
            for retained in retained_by_sequence[seq.seq_id]:
                boundary = retained.boundary_tokens
                metadata = self.full_attention_prefix_manager.metadata_at_boundary(
                    seq, boundary
                )
                capture = PendingHybridPrefixCapture(
                    prefix_hash=metadata.prefix_hash,
                    boundary_tokens=boundary,
                    tail_block_id=metadata.tail_block_id,
                    state_slot=seq.state_slot,
                    reason=retained.reason,
                    internal_state=boundary < step_end,
                    sequence_id=seq.seq_id,
                    replay_saved_tokens=(
                        self.hybrid_prefix_promotions[seq.seq_id][1]
                        if retained.reason == "shared_junction"
                        else boundary
                    ),
                        demand_count=(
                            self.hybrid_prefix_promotions[seq.seq_id][3]
                            if retained.reason == "shared_junction"
                            else 0
                        ),
                )
                if self.hybrid_prefix_coordinator.contains(
                    metadata.prefix_hash, boundary
                ):
                    self._resolve_pending_promotion(
                        capture, published=False
                    )
                    continue
                pending.append(capture)
        return pending, True

    def _internal_retained_boundaries(
        self,
        seq: Sequence,
        end_tokens: int,
    ):
        retained = self.hybrid_prefix_retention_policy.retained_boundaries(
            seq.num_cached_tokens,
            end_tokens,
            seq.num_tokens,
        )
        promotion = self._pending_promotion(seq)
        if promotion is not None:
            promoted_boundary, _, _, _ = promotion
            if (
                seq.num_cached_tokens
                < promoted_boundary.boundary_tokens
                <= end_tokens
            ):
                retained = [
                    item
                    for item in retained
                    if item.boundary_tokens
                    != promoted_boundary.boundary_tokens
                ]
                retained.append(promoted_boundary)
                retained.sort(key=lambda item: item.boundary_tokens)
        capacity = self.hybrid_prefix_cache.capacity
        if len(retained) > capacity:
            if promotion is None:
                retained = retained[-capacity:]
            else:
                promoted_boundary = promotion[0]
                others = [
                    item
                    for item in retained
                    if item.boundary_tokens
                    != promoted_boundary.boundary_tokens
                ]
                retained = sorted(
                    others[-max(capacity - 1, 0):] + [promoted_boundary],
                    key=lambda item: item.boundary_tokens,
                )
        return retained

    def internal_prefix_boundaries(
        self,
        seqs: list[Sequence],
        is_prefill: bool,
    ) -> dict[int, tuple[int, ...]]:
        if not is_prefill or not self.enable_hybrid_internal_checkpoints:
            return {}
        result = {}
        for seq in seqs:
            start = seq.num_cached_tokens
            end = start + seq.num_scheduled_tokens
            boundaries = self._internal_retained_boundaries(seq, end)
            internal = tuple(
                boundary.boundary_tokens
                for boundary in boundaries
                if boundary.boundary_tokens < end
            )
            if internal:
                result[seq.seq_id] = internal
        return result

    def reserve_prefix_capture(
        self,
        pending: PendingHybridPrefixCapture,
    ) -> bool:
        if self.hybrid_prefix_coordinator.contains(
            pending.prefix_hash, pending.boundary_tokens
        ):
            self._resolve_pending_promotion(pending, published=False)
            return False
        evicted = self.hybrid_prefix_coordinator.reserve_capture(pending)
        for entry in evicted:
            self.state_controller.call(
                "evict_prefix_checkpoint", entry.checkpoint_slot
            )
        return True

    def publish_prefix_capture(
        self,
        pending: PendingHybridPrefixCapture,
        checkpoint_slot: int,
    ) -> None:
        self.hybrid_prefix_coordinator.publish(pending, checkpoint_slot)
        if pending.reason == "shared_junction":
            self.hybrid_prefix_promotions.pop(pending.sequence_id, None)

    def cancel_prefix_capture(
        self,
        pending: PendingHybridPrefixCapture,
    ) -> None:
        self._resolve_pending_promotion(pending, published=False)

    def _resolve_pending_promotion(
        self,
        pending: PendingHybridPrefixCapture,
        *,
        published: bool,
    ) -> None:
        if pending.reason != "shared_junction":
            return
        self.hybrid_prefix_coordinator.resolve_capture(
            pending, published=published
        )
        self.hybrid_prefix_promotions.pop(pending.sequence_id, None)

    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids: list[int],
        is_prefill: bool,
        *,
        blocks_already_hashed: bool = False,
    ):
        profiling_started = perf_counter_ns()
        generated_at_ns = perf_counter_ns()
        for seq, token_id in zip(seqs, token_ids):
            if self.enable_prefix_cache and not blocks_already_hashed:
                self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id, generated_at_ns)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                seq.mark_finished(generated_at_ns)
                self.block_manager.deallocate(seq)
                if self.state_manager is not None:
                    self._free_state(seq)
                self.running.remove(seq)
                self.completed_request_count += 1
                self.completed_request_metrics.append(
                    seq.lifecycle_metrics(generated_at_ns)
                )
        if not is_prefill:
            self.mtp_phase_profiler.record_cpu(
                "target_only.scheduler_commit",
                (perf_counter_ns() - profiling_started) / 1e6,
            )

    def postprocess_speculative(
        self,
        seqs: list[Sequence],
        output_token_ids: list[list[int]],
    ) -> None:
        """Commit greedy speculative outputs while keeping one pending token."""
        profiling_started = perf_counter_ns()
        generated_at_ns = perf_counter_ns()
        for seq, tokens in zip(seqs, output_token_ids):
            if not tokens:
                raise ValueError("speculative decode must emit at least one token")
            # The runner processed the old pending token and every emitted
            # token except the final one.
            seq.num_cached_tokens += len(tokens)
            seq.num_scheduled_tokens = 0
            for token_id in tokens:
                if seq.num_completion_tokens == seq.max_tokens:
                    break
                seq.append_token(token_id, generated_at_ns)
                if not seq.ignore_eos and token_id == self.eos:
                    break

            finished = (
                seq.num_completion_tokens == seq.max_tokens
                or (
                    not seq.ignore_eos
                    and seq.completion_token_ids
                    and seq.completion_token_ids[-1] == self.eos
                )
            )
            if finished:
                seq.status = SequenceStatus.FINISHED
                seq.mark_finished(generated_at_ns)
                self.block_manager.deallocate(seq)
                if self.state_manager is not None:
                    self._free_state(seq)
                self.running.remove(seq)
                self.completed_request_count += 1
                self.completed_request_metrics.append(
                    seq.lifecycle_metrics(generated_at_ns)
                )
            else:
                seq.num_cached_tokens = len(seq) - 1
                with self.mtp_phase_profiler.phase(
                    "scheduler.kv_tail_reclaim", gpu=False
                ):
                    self.block_manager.trim_to_sequence_length(seq)
        self.mtp_phase_profiler.record_cpu(
            "scheduler.speculative_commit",
            (perf_counter_ns() - profiling_started) / 1e6,
        )

    def get_request_metrics(self) -> list[dict]:
        return list(self.completed_request_metrics)
