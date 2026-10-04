from collections import deque
from dataclasses import dataclass
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence
from nanovllm.engine.hybrid_prefix_cache import PrefixKVCandidate


@dataclass(frozen=True, slots=True)
class PrefixIndexEntry:
    prefix_hash: int
    unit_token_ids: tuple[int, ...]
    tail_block_id: int
    tail_valid_tokens: int


@dataclass(frozen=True, slots=True)
class KVPageCopy:
    source_block_id: int
    destination_block_id: int
    num_tokens: int


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []
        self.prefix_hashes: set[int] = set()

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
        self.prefix_hashes.clear()


class BlockManager:

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        prefix_match_unit: int | None = None,
    ):
        self.block_size = block_size
        self.prefix_match_unit = prefix_match_unit or block_size
        if self.prefix_match_unit <= 0 or block_size % self.prefix_match_unit:
            raise ValueError("prefix_match_unit must divide block_size")
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.hash_to_prefix_entry: dict[int, PrefixIndexEntry] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.reset_metrics()

    def reset_metrics(self):
        self.prefix_cache_evictions = 0
        self.cached_block_reassignments = 0
        self.duplicate_cache_registrations = 0
        self.partial_prefix_cow_allocations = 0

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1:
            self.cached_block_reassignments += 1
            if self.hash_to_block_id.get(block.hash) == block_id:
                del self.hash_to_block_id[block.hash]
                self.prefix_cache_evictions += 1
        for prefix_hash in tuple(block.prefix_hashes):
            entry = self.hash_to_prefix_entry.get(prefix_hash)
            if entry is not None and entry.tail_block_id == block_id:
                del self.hash_to_prefix_entry[prefix_hash]
                self.prefix_cache_evictions += 1
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence, use_prefix_cache: bool = True) -> int:
        if not use_prefix_cache:
            return 0 if len(self.free_block_ids) >= seq.num_blocks else -1
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def find_prefix_candidates(self, seq: Sequence) -> list[PrefixKVCandidate]:
        """Return valid prefix hits at the configured match unit, longest first."""
        if self.prefix_match_unit < self.block_size:
            return self._find_fine_prefix_candidates(seq)
        prefix_hash = -1
        candidates = []
        # Preserve one uncached prompt block/range so the model produces logits.
        for block_index in range(seq.num_blocks - 1):
            token_ids = seq.block(block_index)
            prefix_hash = self.compute_hash(token_ids, prefix_hash)
            block_id = self.hash_to_block_id.get(prefix_hash, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            candidates.append(
                PrefixKVCandidate(
                    num_cached_blocks=block_index + 1,
                    boundary_tokens=(block_index + 1) * self.block_size,
                    prefix_hash=prefix_hash,
                    tail_block_id=block_id,
                    tail_valid_tokens=self.block_size,
                )
            )
        candidates.reverse()
        return candidates

    def _find_fine_prefix_candidates(
        self,
        seq: Sequence,
    ) -> list[PrefixKVCandidate]:
        prefix_hash = -1
        candidates = []
        # Fine-grained COW makes a hit inside the final physical page safe.
        # Keep only the final match range uncached so prefill still produces
        # logits; the legacy full-block path must keep the whole final page.
        safe_limit = (
            (seq.num_tokens - 1) // self.prefix_match_unit
        ) * self.prefix_match_unit
        for start in range(0, safe_limit, self.prefix_match_unit):
            end = start + self.prefix_match_unit
            unit_tokens = tuple(seq.token_ids[start:end])
            prefix_hash = self.compute_hash(list(unit_tokens), prefix_hash)
            entry = self.hash_to_prefix_entry.get(prefix_hash)
            if entry is None or entry.unit_token_ids != unit_tokens:
                break
            partial_tokens = end % self.block_size
            candidates.append(
                PrefixKVCandidate(
                    num_cached_blocks=end // self.block_size,
                    boundary_tokens=end,
                    prefix_hash=prefix_hash,
                    tail_block_id=entry.tail_block_id,
                    tail_valid_tokens=(partial_tokens or self.block_size),
                )
            )
        candidates.reverse()
        return candidates

    def can_allocate_with_prefix(
        self,
        seq: Sequence,
        num_cached_blocks: int,
    ) -> bool:
        if num_cached_blocks < 0 or num_cached_blocks >= seq.num_blocks:
            raise ValueError("invalid cached block count")
        candidates = self.find_prefix_candidates(seq)
        by_count = {
            candidate.num_cached_blocks: candidate for candidate in candidates
        }
        if num_cached_blocks:
            candidate = by_count.get(num_cached_blocks)
            if candidate is None:
                return False
        num_new_blocks = seq.num_blocks
        prefix_hash = -1
        for block_index in range(num_cached_blocks):
            token_ids = seq.block(block_index)
            prefix_hash = self.compute_hash(token_ids, prefix_hash)
            block_id = self.hash_to_block_id[prefix_hash]
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        return len(self.free_block_ids) >= num_new_blocks

    def _reference_block(self, block_id: int) -> None:
        block = self.blocks[block_id]
        if block_id in self.used_block_ids:
            block.ref_count += 1
        else:
            block.ref_count = 1
            self.free_block_ids.remove(block_id)
            self.used_block_ids.add(block_id)

    def _release_block(self, block_id: int) -> None:
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)

    def _full_prefix_block_ids(
        self,
        seq: Sequence,
        num_full_blocks: int,
    ) -> list[int]:
        if self.prefix_match_unit == self.block_size:
            prefix_hash = -1
            result = []
            for block_index in range(num_full_blocks):
                prefix_hash = self.compute_hash(seq.block(block_index), prefix_hash)
                result.append(self.hash_to_block_id[prefix_hash])
            return result
        prefix_hash = -1
        result = []
        units_per_block = self.block_size // self.prefix_match_unit
        for unit_index in range(num_full_blocks * units_per_block):
            start = unit_index * self.prefix_match_unit
            token_ids = seq.token_ids[start : start + self.prefix_match_unit]
            prefix_hash = self.compute_hash(token_ids, prefix_hash)
            if (unit_index + 1) % units_per_block == 0:
                result.append(
                    self.hash_to_prefix_entry[prefix_hash].tail_block_id
                )
        return result

    def can_allocate_candidate(
        self,
        seq: Sequence,
        candidate: PrefixKVCandidate | None,
    ) -> bool:
        if candidate is None:
            return len(self.free_block_ids) >= seq.num_blocks
        full_block_ids = self._full_prefix_block_ids(
            seq,
            candidate.num_cached_blocks,
        )
        required_free = seq.num_blocks
        for block_id in full_block_ids:
            if block_id in self.used_block_ids:
                required_free -= 1
        is_partial = candidate.tail_valid_tokens != self.block_size
        if is_partial and candidate.tail_block_id not in self.used_block_ids:
            # Pin the COW source until its GPU rows have been copied.
            required_free += 1
        return len(self.free_block_ids) >= required_free

    def allocate_candidate(
        self,
        seq: Sequence,
        candidate: PrefixKVCandidate | None,
    ) -> KVPageCopy | None:
        """Allocate a request and pin a partial-page COW source if needed."""
        assert not seq.block_table
        if candidate is None:
            self.allocate(seq, 0)
            return None
        full_block_ids = self._full_prefix_block_ids(
            seq,
            candidate.num_cached_blocks,
        )
        is_partial = candidate.tail_valid_tokens != self.block_size
        pinned_source = False
        try:
            for block_id in full_block_ids:
                self._reference_block(block_id)
                seq.block_table.append(block_id)
            if is_partial:
                self._reference_block(candidate.tail_block_id)
                pinned_source = True
            while len(seq.block_table) < seq.num_blocks:
                seq.block_table.append(self._allocate_block())
            seq.num_cached_tokens = candidate.boundary_tokens
            if not is_partial:
                return None
            self.partial_prefix_cow_allocations += 1
            return KVPageCopy(
                source_block_id=candidate.tail_block_id,
                destination_block_id=(
                    seq.block_table[candidate.num_cached_blocks]
                ),
                num_tokens=candidate.tail_valid_tokens,
            )
        except Exception:
            if pinned_source:
                self._release_block(candidate.tail_block_id)
            self.deallocate(seq)
            raise

    def release_cow_source(self, page_copy: KVPageCopy) -> None:
        self._release_block(page_copy.source_block_id)

    def prefix_metadata_at_boundary(
        self,
        seq: Sequence,
        boundary_tokens: int,
    ) -> PrefixKVCandidate:
        if boundary_tokens <= 0 or boundary_tokens % self.prefix_match_unit:
            raise ValueError("prefix boundary must align to prefix_match_unit")
        block_index = (boundary_tokens - 1) // self.block_size
        if block_index >= len(seq.block_table):
            raise ValueError("prefix boundary has no allocated block")
        block_id = seq.block_table[block_index]
        if self.prefix_match_unit < self.block_size:
            prefix_hash = -1
            for start in range(0, boundary_tokens, self.prefix_match_unit):
                prefix_hash = self.compute_hash(
                    seq.token_ids[start : start + self.prefix_match_unit],
                    prefix_hash,
                )
            entry = self.hash_to_prefix_entry.get(prefix_hash)
            if entry is None or entry.tail_block_id != block_id:
                raise ValueError("prefix boundary has not been indexed")
            return PrefixKVCandidate(
                num_cached_blocks=boundary_tokens // self.block_size,
                boundary_tokens=boundary_tokens,
                prefix_hash=prefix_hash,
                tail_block_id=block_id,
                tail_valid_tokens=(
                    boundary_tokens % self.block_size or self.block_size
                ),
            )
        block = self.blocks[block_id]
        if block.hash == -1:
            raise ValueError("prefix boundary block has not been hashed")
        return PrefixKVCandidate(
            num_cached_blocks=block_index + 1,
            boundary_tokens=boundary_tokens,
            prefix_hash=block.hash,
            tail_block_id=block_id,
            tail_valid_tokens=self.block_size,
        )

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def reclaimable_blocks(self, seq: Sequence) -> int:
        """Physical blocks that become free if this request releases refs."""
        return sum(
            self.blocks[block_id].ref_count == 1
            for block_id in set(seq.block_table)
        )

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def additional_blocks_for_append(self, seq: Sequence, num_tokens: int) -> int:
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        final_num_blocks = (
            len(seq) + num_tokens + self.block_size - 1
        ) // self.block_size
        return max(final_num_blocks - len(seq.block_table), 0)

    def can_reserve_append(self, seq: Sequence, num_tokens: int) -> bool:
        return len(self.free_block_ids) >= self.additional_blocks_for_append(
            seq, num_tokens
        )

    def reserve_append(self, seq: Sequence, num_tokens: int) -> None:
        required = self.additional_blocks_for_append(seq, num_tokens)
        if len(self.free_block_ids) < required:
            raise RuntimeError("insufficient KV blocks for speculative append")
        for _ in range(required):
            seq.block_table.append(self._allocate_block())

    def trim_to_sequence_length(self, seq: Sequence) -> None:
        """Release speculative tail blocks beyond the committed token list."""
        while len(seq.block_table) > seq.num_blocks:
            block_id = seq.block_table.pop()
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count != 0:
                raise RuntimeError("speculative tail block is unexpectedly shared")
            self._deallocate_block(block_id)

    def hash_blocks(self, seq: Sequence):
        if self.prefix_match_unit < self.block_size:
            self._hash_prefix_units(seq)
            return
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            existing_block_id = self.hash_to_block_id.get(h)
            if (
                existing_block_id is not None
                and existing_block_id != block.block_id
                and self.blocks[existing_block_id].token_ids == token_ids
            ):
                self.duplicate_cache_registrations += 1
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id

    def _hash_prefix_units(self, seq: Sequence) -> None:
        completed_tokens = seq.num_cached_tokens + seq.num_scheduled_tokens
        completed_units = completed_tokens // self.prefix_match_unit
        first_new_unit = seq.num_cached_tokens // self.prefix_match_unit
        prefix_hash = -1
        for unit_index in range(completed_units):
            start = unit_index * self.prefix_match_unit
            end = start + self.prefix_match_unit
            token_ids = tuple(seq.token_ids[start:end])
            prefix_hash = self.compute_hash(list(token_ids), prefix_hash)
            if unit_index < first_new_unit:
                continue
            boundary = end
            block_index = (boundary - 1) // self.block_size
            block_id = seq.block_table[block_index]
            entry = PrefixIndexEntry(
                prefix_hash=prefix_hash,
                unit_token_ids=token_ids,
                tail_block_id=block_id,
                tail_valid_tokens=(boundary % self.block_size or self.block_size),
            )
            existing = self.hash_to_prefix_entry.get(prefix_hash)
            if existing is not None and existing.tail_block_id != block_id:
                self.duplicate_cache_registrations += 1
            self.hash_to_prefix_entry[prefix_hash] = entry
            self.blocks[block_id].prefix_hashes.add(prefix_hash)
            if boundary % self.block_size == 0:
                block = self.blocks[block_id]
                block.update(prefix_hash, seq.block(block_index))
                self.hash_to_block_id[prefix_hash] = block_id

    @property
    def num_cached_blocks(self) -> int:
        return sum(block.hash != -1 for block in self.blocks)

    @property
    def num_reachable_cached_blocks(self) -> int:
        if self.prefix_match_unit < self.block_size:
            return len(
                {entry.tail_block_id for entry in self.hash_to_prefix_entry.values()}
            )
        return len(set(self.hash_to_block_id.values()))
