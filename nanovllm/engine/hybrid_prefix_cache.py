"""Control-plane metadata for block-aligned Qwen3.5 Hybrid Prefix Cache."""

from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter_ns


@dataclass(frozen=True, slots=True)
class PrefixKVCandidate:
    num_cached_blocks: int
    boundary_tokens: int
    prefix_hash: int
    tail_block_id: int
    tail_valid_tokens: int = 0


@dataclass(frozen=True, slots=True)
class HybridPrefixHit:
    candidate: PrefixKVCandidate
    checkpoint_slot: int


@dataclass(frozen=True, slots=True)
class HybridPrefixAllocationPlan:
    """Immutable admission plan produced before mutating request ownership."""

    hit: HybridPrefixHit | None
    num_cached_blocks: int
    kv_candidate_tokens: int = 0
    promotion_candidate: PrefixKVCandidate | None = None

    @property
    def boundary_tokens(self) -> int:
        return self.hit.candidate.boundary_tokens if self.hit is not None else 0

    @property
    def candidate(self) -> PrefixKVCandidate | None:
        return self.hit.candidate if self.hit is not None else None


@dataclass(frozen=True, slots=True)
class PendingHybridPrefixCapture:
    prefix_hash: int
    boundary_tokens: int
    tail_block_id: int
    state_slot: int
    reason: str = "periodic"
    internal_state: bool = False
    sequence_id: int = -1
    replay_saved_tokens: int = 0
    demand_count: int = 0


@dataclass(slots=True)
class HybridPrefixEntry:
    prefix_hash: int
    boundary_tokens: int
    tail_block_id: int
    checkpoint_slot: int
    last_access_ns: int
    reason: str
    hit_count: int = 0
    replay_saved_tokens: int = 0
    demand_count: int = 0

    @property
    def retained_value(self) -> int:
        """Cheap benefit proxy used by count-bounded cost-aware eviction."""
        saved_tokens = self.replay_saved_tokens or self.boundary_tokens
        return saved_tokens * (self.hit_count + self.demand_count + 1)


@dataclass(slots=True)
class PrefixDemandEntry:
    """Bounded CPU-only evidence for a KV prefix missing GDN state."""

    prefix_hash: int
    boundary_tokens: int
    observations: int
    last_seen_ns: int
    promotion_pending: bool = False


@dataclass(frozen=True, slots=True)
class CheckpointBoundary:
    boundary_tokens: int
    reason: str


class HybridCheckpointRetentionPolicy:
    """Separate restorable block boundaries from checkpoint retention.

    Every full KV block is a valid candidate boundary. ``periodic`` retains
    only configured intervals. ``adaptive`` additionally retains the final
    reusable full-block boundary before the prompt tail.
    """

    def __init__(
        self,
        *,
        block_size: int,
        interval_tokens: int,
        mode: str = "periodic",
    ) -> None:
        if block_size <= 0 or interval_tokens <= 0:
            raise ValueError("checkpoint boundary sizes must be positive")
        if interval_tokens % block_size:
            raise ValueError("checkpoint interval must align to KV blocks")
        if mode not in ("periodic", "adaptive"):
            raise ValueError(f"unsupported hybrid retention policy: {mode}")
        self.block_size = block_size
        self.interval_tokens = interval_tokens
        self.mode = mode

    def prompt_tail_boundary(self, prompt_tokens: int) -> int:
        if prompt_tokens <= 1:
            return 0
        return ((prompt_tokens - 1) // self.block_size) * self.block_size

    def classify(
        self,
        boundary_tokens: int,
        prompt_tokens: int,
    ) -> CheckpointBoundary | None:
        if boundary_tokens <= 0 or boundary_tokens % self.block_size:
            return None
        if boundary_tokens % self.interval_tokens == 0:
            return CheckpointBoundary(boundary_tokens, "periodic")
        if (
            self.mode == "adaptive"
            and boundary_tokens == self.prompt_tail_boundary(prompt_tokens)
        ):
            return CheckpointBoundary(boundary_tokens, "prompt_tail")
        return None

    def next_retained_boundary(
        self,
        start_tokens: int,
        prompt_tokens: int,
    ) -> CheckpointBoundary | None:
        boundary = (start_tokens // self.block_size + 1) * self.block_size
        safe_limit = self.prompt_tail_boundary(prompt_tokens)
        while boundary <= safe_limit:
            candidate = self.classify(boundary, prompt_tokens)
            if candidate is not None:
                return candidate
            boundary += self.block_size
        return None

    def retained_boundaries(
        self,
        start_tokens: int,
        end_tokens: int,
        prompt_tokens: int,
    ) -> list[CheckpointBoundary]:
        retained = []
        boundary = (
            start_tokens // self.block_size + 1
        ) * self.block_size
        safe_limit = min(
            end_tokens,
            self.prompt_tail_boundary(prompt_tokens),
        )
        while boundary <= safe_limit:
            candidate = self.classify(boundary, prompt_tokens)
            if candidate is not None:
                retained.append(candidate)
            boundary += self.block_size
        return retained


class GDNCheckpointManager:
    """Bounded GDN checkpoint index; tensors live in ModelRunner."""

    def __init__(self, capacity: int, eviction_policy: str = "lru") -> None:
        if capacity <= 0:
            raise ValueError("hybrid prefix cache capacity must be positive")
        self.capacity = capacity
        if eviction_policy not in ("lru", "cost_aware"):
            raise ValueError(
                f"unsupported checkpoint eviction policy: {eviction_policy}"
            )
        self.eviction_policy = eviction_policy
        self.entries: dict[tuple[int, int], HybridPrefixEntry] = {}
        self.lru: OrderedDict[tuple[int, int], None] = OrderedDict()
        self.reset_metrics()

    def reset_metrics(self) -> None:
        self.queries = 0
        self.kv_candidate_tokens = 0
        self.committed_hit_tokens = 0
        self.captures = 0
        self.restores = 0
        self.evictions = 0
        self.evicted_retained_value = 0
        self.fallbacks = 0
        self.shared_junction_hits = 0
        self.useful_shared_junction_promotions = 0
        self.shared_junction_saved_replay_tokens = 0
        self.shared_junction_evictions = 0
        self.unused_shared_junction_evictions = 0
        self.peak_entries = len(self.entries)
        self.peak_shared_junction_entries = sum(
            entry.reason == "shared_junction"
            for entry in self.entries.values()
        )

    def find_hit(
        self,
        candidates: list[PrefixKVCandidate],
    ) -> HybridPrefixHit | None:
        hit = self.peek_hit(candidates)
        self.record_lookup(candidates, hit)
        return hit

    def peek_hit(
        self,
        candidates: list[PrefixKVCandidate],
    ) -> HybridPrefixHit | None:
        for candidate in candidates:
            key = (candidate.prefix_hash, candidate.boundary_tokens)
            entry = self.entries.get(key)
            if entry is not None and entry.tail_block_id == candidate.tail_block_id:
                return HybridPrefixHit(candidate, entry.checkpoint_slot)
        return None

    def record_lookup(
        self,
        candidates: list[PrefixKVCandidate],
        hit: HybridPrefixHit | None,
    ) -> None:
        self.queries += 1
        if candidates:
            self.kv_candidate_tokens += candidates[0].boundary_tokens
        if hit is not None:
            candidate = hit.candidate
            key = (candidate.prefix_hash, candidate.boundary_tokens)
            entry = self.entries[key]
            entry.last_access_ns = perf_counter_ns()
            if entry.reason == "shared_junction":
                self.shared_junction_hits += 1
                self.shared_junction_saved_replay_tokens += (
                    entry.replay_saved_tokens or entry.boundary_tokens
                )
                if entry.hit_count == 0:
                    self.useful_shared_junction_promotions += 1
            entry.hit_count += 1
            self.lru.move_to_end(key)
            self.committed_hit_tokens += candidate.boundary_tokens
            self.restores += 1
        elif candidates:
            self.fallbacks += 1

    def contains(self, prefix_hash: int, boundary_tokens: int) -> bool:
        return (prefix_hash, boundary_tokens) in self.entries

    def reserve_capture(
        self,
        pending: PendingHybridPrefixCapture,
    ) -> list[HybridPrefixEntry]:
        if self.contains(pending.prefix_hash, pending.boundary_tokens):
            return []
        evicted = []
        while len(self.entries) >= self.capacity:
            if self.eviction_policy == "lru":
                key, _ = self.lru.popitem(last=False)
            else:
                key = min(
                    self.entries,
                    key=lambda item: (
                        self.entries[item].retained_value,
                        self.entries[item].last_access_ns,
                    ),
                )
                del self.lru[key]
            entry = self.entries.pop(key)
            evicted.append(entry)
            if entry.reason == "shared_junction":
                self.shared_junction_evictions += 1
                if entry.hit_count == 0:
                    self.unused_shared_junction_evictions += 1
            self.evicted_retained_value += entry.retained_value
            self.evictions += 1
        return evicted

    def publish(
        self,
        pending: PendingHybridPrefixCapture,
        checkpoint_slot: int,
    ) -> None:
        if self.contains(pending.prefix_hash, pending.boundary_tokens):
            raise ValueError("hybrid prefix checkpoint was already published")
        entry = HybridPrefixEntry(
            prefix_hash=pending.prefix_hash,
            boundary_tokens=pending.boundary_tokens,
            tail_block_id=pending.tail_block_id,
            checkpoint_slot=checkpoint_slot,
            last_access_ns=perf_counter_ns(),
            reason=pending.reason,
            replay_saved_tokens=pending.replay_saved_tokens,
            demand_count=pending.demand_count,
        )
        key = (pending.prefix_hash, pending.boundary_tokens)
        self.entries[key] = entry
        self.lru[key] = None
        self.captures += 1
        self.peak_entries = max(self.peak_entries, len(self.entries))
        shared_entries = sum(
            item.reason == "shared_junction"
            for item in self.entries.values()
        )
        self.peak_shared_junction_entries = max(
            self.peak_shared_junction_entries,
            shared_entries,
        )

    def get_metrics(self) -> dict[str, int | float]:
        periodic_entries = 0
        prompt_tail_entries = 0
        shared_junction_entries = 0
        unused_shared_junction_entries = 0
        for entry in self.entries.values():
            if entry.reason == "periodic":
                periodic_entries += 1
            elif entry.reason == "prompt_tail":
                prompt_tail_entries += 1
            elif entry.reason == "shared_junction":
                shared_junction_entries += 1
                if entry.hit_count == 0:
                    unused_shared_junction_entries += 1
        return {
            "hybrid_prefix_cache_capacity": self.capacity,
            "hybrid_prefix_cache_entries": len(self.entries),
            "hybrid_prefix_cache_queries": self.queries,
            "hybrid_prefix_kv_candidate_tokens": self.kv_candidate_tokens,
            "hybrid_prefix_committed_hit_tokens": self.committed_hit_tokens,
            "hybrid_prefix_state_alignment_lost_tokens": (
                self.kv_candidate_tokens - self.committed_hit_tokens
            ),
            "hybrid_prefix_capture_count": self.captures,
            "hybrid_prefix_restore_count": self.restores,
            "hybrid_prefix_eviction_count": self.evictions,
            "hybrid_prefix_evicted_retained_value": (
                self.evicted_retained_value
            ),
            "hybrid_prefix_eviction_policy": self.eviction_policy,
            "hybrid_prefix_fallback_count": self.fallbacks,
            "hybrid_prefix_hit_rate": (
                self.restores / self.queries if self.queries else 0.0
            ),
            "hybrid_prefix_entries_periodic": periodic_entries,
            "hybrid_prefix_entries_prompt_tail": prompt_tail_entries,
            "hybrid_prefix_entries_shared_junction": shared_junction_entries,
            "hybrid_prefix_unused_shared_junction_entries": (
                unused_shared_junction_entries
            ),
            "hybrid_prefix_peak_entries": self.peak_entries,
            "hybrid_prefix_peak_shared_junction_entries": (
                self.peak_shared_junction_entries
            ),
            "hybrid_prefix_shared_junction_hit_count": (
                self.shared_junction_hits
            ),
            "hybrid_prefix_useful_promotion_count": (
                self.useful_shared_junction_promotions
            ),
            "hybrid_prefix_shared_junction_saved_replay_tokens": (
                self.shared_junction_saved_replay_tokens
            ),
            "hybrid_prefix_shared_junction_eviction_count": (
                self.shared_junction_evictions
            ),
            "hybrid_prefix_unused_promotion_eviction_count": (
                self.unused_shared_junction_evictions
            ),
        }


# Backwards-compatible name retained for downstream users and existing tests.
HybridPrefixCache = GDNCheckpointManager


class FullAttentionPrefixManager:
    """Typed prefix-cache view over the existing paged-KV BlockManager."""

    def __init__(self, block_manager) -> None:
        self.block_manager = block_manager

    def find_candidates(self, sequence) -> list[PrefixKVCandidate]:
        return self.block_manager.find_prefix_candidates(sequence)

    def can_allocate(
        self,
        sequence,
        candidate: PrefixKVCandidate | None,
    ) -> bool:
        return self.block_manager.can_allocate_candidate(
            sequence,
            candidate,
        )

    def allocate(self, sequence, candidate: PrefixKVCandidate | None):
        return self.block_manager.allocate_candidate(sequence, candidate)

    def metadata_at_boundary(
        self,
        sequence,
        boundary_tokens: int,
    ) -> PrefixKVCandidate:
        return self.block_manager.prefix_metadata_at_boundary(
            sequence,
            boundary_tokens,
        )


class HybridPrefixCoordinator:
    """Intersect Full-Attention KV and GDN checkpoint availability.

    M1 intentionally preserves full-block matching. Later milestones can
    replace the two managers independently without growing Scheduler policy.
    """

    def __init__(
        self,
        full_attention: FullAttentionPrefixManager,
        gdn_checkpoints: GDNCheckpointManager,
        *,
        promote_shared_junctions: bool = False,
        promotion_min_sightings: int = 2,
        demand_capacity: int = 1024,
    ) -> None:
        if demand_capacity <= 0:
            raise ValueError("prefix demand capacity must be positive")
        if promotion_min_sightings < 2:
            raise ValueError(
                "demand-driven promotion requires at least two sightings"
            )
        self.full_attention = full_attention
        self.gdn_checkpoints = gdn_checkpoints
        self.promote_shared_junctions = promote_shared_junctions
        self.promotion_min_sightings = promotion_min_sightings
        self.demand_capacity = demand_capacity
        self.demands: OrderedDict[
            tuple[int, int], PrefixDemandEntry
        ] = OrderedDict()
        self.reset_promotion_metrics()

    def reset_promotion_metrics(self) -> None:
        self.kv_only_misses = 0
        self.alignment_lost_tokens = 0
        self.promotions_planned = 0
        self.promotions_published = 0
        self.promotions_cancelled = 0

    def _observe_alignment_gap(
        self,
        candidate: PrefixKVCandidate,
        committed_boundary: int,
    ) -> PrefixKVCandidate | None:
        lost_tokens = candidate.boundary_tokens - committed_boundary
        if lost_tokens <= 0:
            return None
        self.kv_only_misses += 1
        self.alignment_lost_tokens += lost_tokens
        if not self.promote_shared_junctions:
            return None
        key = (candidate.prefix_hash, candidate.boundary_tokens)
        now = perf_counter_ns()
        demand = self.demands.get(key)
        if demand is None:
            while len(self.demands) >= self.demand_capacity:
                evictable = next(
                    (
                        item_key
                        for item_key, item in self.demands.items()
                        if not item.promotion_pending
                    ),
                    None,
                )
                if evictable is None:
                    return None
                self.demands.pop(evictable)
            demand = PrefixDemandEntry(
                prefix_hash=candidate.prefix_hash,
                boundary_tokens=candidate.boundary_tokens,
                observations=0,
                last_seen_ns=now,
            )
            self.demands[key] = demand
        demand.observations += 1
        demand.last_seen_ns = now
        self.demands.move_to_end(key)
        if demand.promotion_pending or self.gdn_checkpoints.contains(*key):
            return None
        # A resident KV candidate proves a previous producer existed. This
        # request is therefore the second sighting of the prefix and may
        # promote its recurrent state while replaying the missing suffix.
        total_sightings = 1 + demand.observations
        if total_sightings < self.promotion_min_sightings:
            return None
        return candidate

    def _activate_promotion(
        self,
        candidate: PrefixKVCandidate | None,
    ) -> None:
        if candidate is None:
            return
        key = (candidate.prefix_hash, candidate.boundary_tokens)
        demand = self.demands.get(key)
        if demand is None or demand.promotion_pending:
            return
        demand.promotion_pending = True
        self.promotions_planned += 1

    def cancel_promotion(
        self,
        candidate: PrefixKVCandidate | None,
    ) -> None:
        if candidate is None:
            return
        key = (candidate.prefix_hash, candidate.boundary_tokens)
        demand = self.demands.get(key)
        if demand is not None and demand.promotion_pending:
            demand.promotion_pending = False
            self.promotions_cancelled += 1

    def demand_observations(
        self,
        candidate: PrefixKVCandidate,
    ) -> int:
        demand = self.demands.get(
            (candidate.prefix_hash, candidate.boundary_tokens)
        )
        return demand.observations if demand is not None else 0

    def resolve_capture(
        self,
        pending: PendingHybridPrefixCapture,
        *,
        published: bool,
    ) -> None:
        if pending.reason != "shared_junction":
            return
        key = (pending.prefix_hash, pending.boundary_tokens)
        demand = self.demands.get(key)
        if demand is not None:
            demand.promotion_pending = False
            if published:
                self.demands.pop(key, None)
        if published:
            self.promotions_published += 1
        else:
            self.promotions_cancelled += 1

    def plan(self, sequence) -> HybridPrefixAllocationPlan | None:
        candidates = self.full_attention.find_candidates(sequence)
        probe = self._probe_from_candidates(candidates)
        hit = probe.hit
        num_cached_blocks = probe.num_cached_blocks
        if not self.full_attention.can_allocate(sequence, hit.candidate if hit else None):
            return None
        committed_boundary = hit.candidate.boundary_tokens if hit else 0
        promotion_candidate = (
            self._observe_alignment_gap(candidates[0], committed_boundary)
            if candidates
            else None
        )
        if hit is None:
            self.gdn_checkpoints.record_lookup(candidates, None)
        return HybridPrefixAllocationPlan(
            hit,
            num_cached_blocks,
            probe.kv_candidate_tokens,
            promotion_candidate,
        )

    def probe(self, sequence) -> HybridPrefixAllocationPlan:
        """Read-only KV/GDN intersection for scheduler scoring.

        This method does not allocate blocks, update LRU/hit metrics, observe
        demand, or publish promotion metadata.
        """
        return self._probe_from_candidates(
            self.full_attention.find_candidates(sequence)
        )

    def _probe_from_candidates(
        self,
        candidates: list[PrefixKVCandidate],
    ) -> HybridPrefixAllocationPlan:
        hit = self.gdn_checkpoints.peek_hit(candidates)
        return HybridPrefixAllocationPlan(
            hit=hit,
            num_cached_blocks=(
                hit.candidate.num_cached_blocks if hit is not None else 0
            ),
            kv_candidate_tokens=(
                candidates[0].boundary_tokens if candidates else 0
            ),
        )

    def commit(self, plan: HybridPrefixAllocationPlan) -> None:
        self._activate_promotion(plan.promotion_candidate)
        if plan.hit is not None:
            self.gdn_checkpoints.record_lookup(
                [
                    PrefixKVCandidate(
                        num_cached_blocks=plan.hit.candidate.num_cached_blocks,
                        boundary_tokens=plan.kv_candidate_tokens,
                        prefix_hash=plan.hit.candidate.prefix_hash,
                        tail_block_id=plan.hit.candidate.tail_block_id,
                        tail_valid_tokens=plan.hit.candidate.tail_valid_tokens,
                    )
                ],
                plan.hit,
            )

    def contains(self, prefix_hash: int, boundary_tokens: int) -> bool:
        return self.gdn_checkpoints.contains(prefix_hash, boundary_tokens)

    def reserve_capture(
        self,
        pending: PendingHybridPrefixCapture,
    ) -> list[HybridPrefixEntry]:
        return self.gdn_checkpoints.reserve_capture(pending)

    def publish(
        self,
        pending: PendingHybridPrefixCapture,
        checkpoint_slot: int,
    ) -> None:
        self.gdn_checkpoints.publish(pending, checkpoint_slot)
        self.resolve_capture(pending, published=True)

    def reset_metrics(self) -> None:
        self.gdn_checkpoints.reset_metrics()
        self.reset_promotion_metrics()

    def get_metrics(self) -> dict[str, int | float]:
        metrics = self.gdn_checkpoints.get_metrics()
        metrics.update(
            {
                "hybrid_prefix_kv_only_miss_count": self.kv_only_misses,
                "hybrid_prefix_alignment_lost_tokens": (
                    self.alignment_lost_tokens
                ),
                "hybrid_prefix_shared_junction_promotions_planned": (
                    self.promotions_planned
                ),
                "hybrid_prefix_shared_junction_promotions_published": (
                    self.promotions_published
                ),
                "hybrid_prefix_shared_junction_promotions_cancelled": (
                    self.promotions_cancelled
                ),
                "hybrid_prefix_demand_entries": len(self.demands),
                "hybrid_prefix_promotion_min_sightings": (
                    self.promotion_min_sightings
                ),
                "hybrid_prefix_replay_due_to_missing_checkpoint_tokens": (
                    self.alignment_lost_tokens
                ),
                "hybrid_prefix_useful_promotion_ratio": (
                    self.gdn_checkpoints.useful_shared_junction_promotions
                    / self.promotions_published
                    if self.promotions_published
                    else 0.0
                ),
                "hybrid_prefix_promotions_not_yet_reused": max(
                    sum(
                        entry.reason == "shared_junction"
                        and entry.hit_count == 0
                        for entry in self.gdn_checkpoints.entries.values()
                    ),
                    0,
                ),
            }
        )
        return metrics
