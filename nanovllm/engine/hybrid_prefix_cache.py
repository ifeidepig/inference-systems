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


@dataclass(slots=True)
class HybridPrefixEntry:
    prefix_hash: int
    boundary_tokens: int
    tail_block_id: int
    checkpoint_slot: int
    last_access_ns: int
    reason: str
    hit_count: int = 0

    @property
    def retained_value(self) -> int:
        """Cheap benefit proxy used by count-bounded cost-aware eviction."""
        return self.boundary_tokens * (self.hit_count + 1)


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
        )
        key = (pending.prefix_hash, pending.boundary_tokens)
        self.entries[key] = entry
        self.lru[key] = None
        self.captures += 1

    def get_metrics(self) -> dict[str, int | float]:
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
    ) -> None:
        self.full_attention = full_attention
        self.gdn_checkpoints = gdn_checkpoints

    def plan(self, sequence) -> HybridPrefixAllocationPlan | None:
        candidates = self.full_attention.find_candidates(sequence)
        hit = self.gdn_checkpoints.peek_hit(candidates)
        num_cached_blocks = hit.candidate.num_cached_blocks if hit else 0
        if not self.full_attention.can_allocate(sequence, hit.candidate if hit else None):
            return None
        if hit is None:
            self.gdn_checkpoints.record_lookup(candidates, None)
        return HybridPrefixAllocationPlan(
            hit,
            num_cached_blocks,
            candidates[0].boundary_tokens if candidates else 0,
        )

    def commit(self, plan: HybridPrefixAllocationPlan) -> None:
        if plan.hit is None:
            return
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

    def reset_metrics(self) -> None:
        self.gdn_checkpoints.reset_metrics()

    def get_metrics(self) -> dict[str, int | float]:
        return self.gdn_checkpoints.get_metrics()
