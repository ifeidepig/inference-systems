# Hybrid Prefix Cache Design for Qwen3.5

Date: 2026-09-27

Status: experimental fine-grained implementation complete; MTP coexistence off,
real multi-GPU NCCL/9B performance pending

## Executive decision

Do not enable the existing KV-only prefix-cache switch for Qwen3.5. Implement
a correctness-first, block-aligned hybrid cache whose reusable unit is:

```text
(prefix hash, token boundary, full-attention KV blocks, GDN state checkpoint)
```

Use a separately budgeted, sparse GDN checkpoint pool. The reusable prefix
length is the greatest boundary supported by both the full-attention KV cache
and a valid recurrent/conv checkpoint. Begin without simultaneous MTP prefix
reuse and without partial-block hits; add those only after the aligned path has
passed differential and lifecycle tests.

## Local reference audit

### Yuezheng-Ling-0412/nanovllm-qwen3.5

The repository retains the ordinary hash/block prefix-cache implementation for
attention-only models, but its hybrid scheduler calls:

```python
self.block_manager.allocate(seq, disable_prefix_cache=self.is_hybrid)
```

It does not store recurrent/conv checkpoints or restore them on a hit. Its
`StateSlotManager` only allocates integer slots and does not provide an aligned
prefix-state cache.

Conclusion: useful as evidence that KV-only reuse must be disabled, not as a
Hybrid Prefix Cache implementation.

### RLS-ResearchLab/qLLM

The README explicitly lists prefix caching as permanently disabled while a
`StateManager` is active. `BlockManager(disable_prefix_cache=True)` skips all
prefix lookup for hybrid requests. Its code comments document an empirical
failure: reusing a shared KV prefix while resetting recurrent/conv state to
zero produced a degenerate repetition loop.

Conclusion: it supplies strong negative evidence and lifecycle tests, but no
checkpointed-prefix design. The repository has no declared license, so use it
as a design reference only.

## External implementation lessons

Current vLLM models recurrent caches as separate cache groups and uses a
`HybridKVCacheCoordinator`. Each cache type computes its own longest reusable
prefix; the coordinator monotonically reduces the candidate until every group
agrees on one boundary. For Mamba/GDN-like state, `mamba_cache_mode="align"`
stores state only at reusable boundaries. Full-attention and recurrent groups
must both hit the chosen length.

Useful ideas to adopt:

- separate managers/pools for full-attention KV and recurrent state;
- an explicit token boundary attached to every state checkpoint;
- intersection/fixed-point reconciliation of per-group hit lengths;
- block-aligned mode first;
- independent hash granularity only after copy-on-write is implemented;
- sparse retention intervals and explicit reachable boundaries;
- offload hand-off must carry the exact boundary, not just bytes/block ID.

Known failure modes to design against:

- a state checkpoint at boundary `N` combined with replay from `N-1` applies a
  token twice and can silently change output;
- retaining only the latest request checkpoint can yield a zero hybrid hit
  when it lands in request-unique suffix tokens;
- coarse recurrent blocks cause a prefix-hit cliff for shorter shared prompts;
- publishing a cache key for a null/unmaterialized state produces silent
  corruption even when data transfer itself succeeds;
- MTP/EAGLE lookahead changes the finalized-token and replay boundary.

SGLang's Unified Radix Cache demonstrates coordinated component eviction and
session-aware retention for Full Attention, SWA and Mamba components. It is a
useful later reference for hierarchy/eviction, but the first nano-vLLM
implementation should remain a small block-hash design rather than importing a
full radix tree and recurrent-cache system at once.

## Implemented MVP status

Implemented in the current worktree:

- default-off Config flags and pre-NCCL snapshot byte validation;
- fixed-budget GPU `HybridPrefixCheckpointPool`;
- longest-first KV candidate lookup and state-boundary intersection;
- sparse scheduler splits at configured block intervals;
- synchronous capture followed by metadata publish;
- immutable checkpoint copy into request-owned state slots;
- bounded LRU checkpoint metadata and GPU-slot eviction;
- safe fallback when KV exists but its state checkpoint was evicted;
- server CLI flags, metrics, and reproducible benchmark harness;
- tiny Hybrid E2E cold/warm equality in eager and CUDA Graph decode modes;
- hit -> preempt -> re-hit -> abort lifecycle isolation;
- eviction/fallback equality and checkpoint-pool unit tests.

Still gated:

- repeated/interval official 0.8B benchmark;
- TP transactional publication;
- MTP shifted-cache coexistence;
- adaptive/shared-junction and partial-block checkpoints;
- CPU checkpoint tier.

## Fine-grained roadmap: M1 control-plane split

The first fine-grained milestone is implemented without changing cache-hit
semantics:

- `FullAttentionPrefixManager` exposes typed paged-KV candidate, admission,
  allocation, and boundary-metadata operations over `BlockManager`;
- `GDNCheckpointManager` owns bounded recurrent/conv checkpoint metadata;
- `HybridPrefixCoordinator` intersects the managers and returns an immutable
  `HybridPrefixAllocationPlan` before Scheduler mutates request ownership;
- the former `HybridPrefixCache` name remains as a compatibility alias;
- `prefix_match_unit` is now an explicit Config/CLI field, but M1 rejects a
  value different from the physical KV block size. This prevents a finer
  value from being silently accepted before partial-page COW exists.

M1 deliberately preserves full-block matching, fixed-interval retention, and
the existing GPU checkpoint pool. Fine-grained hashing, partial-page KV COW,
adaptive retention, and TP transactions were left to subsequent milestones;
the following section records the completed first M2 increment.

## M2 retention and eviction policy split

M2 adds explicit, opt-in policies while preserving the M1 defaults:

- every completed full KV block is a *candidate* checkpoint boundary;
- `periodic` retention keeps the original fixed interval behavior;
- `adaptive` additionally keeps the final reusable full-block boundary before
  the prompt tail, which captures a system/shared-prefix junction without
  making every intermediate block dense;
- `lru` remains the default eviction policy;
- `cost_aware` evicts the entry with the smallest
  `boundary_tokens * (hit_count + 1)` benefit proxy, breaking ties by age;
- capture metadata records whether the checkpoint was retained because it was
  `periodic` or a `prompt_tail` boundary;
- the benchmark exposes both policies so a no-share workload can measure their
  cost instead of assuming extra checkpoints are beneficial.

The cost score is deliberately a transparent baseline. Later benchmark data
can replace it with measured saved-prefill milliseconds per retained byte.

## M3 fine-grained matching and partial-page COW

M3 is implemented for TP=1:

- `prefix_match_unit` may be a proper divisor of the 256-token physical KV
  block size;
- `BlockManager` maintains a chained hash-unit index independently of the
  legacy full-block index;
- each index entry identifies its tail physical block and the number of valid
  prefix rows in that block;
- a hit ending inside a physical page reuses preceding full pages, pins the
  cached source page, allocates a private destination page, and returns a
  `KVPageCopy` plan;
- `ModelRunner.copy_prefix_kv` copies K and V rows across every Full Attention
  layer before suffix execution;
- the source page is immutable during the copy and is released afterward;
- KV allocation, request-state allocation, page copy, checkpoint restore, and
  metadata commit form one admission transaction. Injected copy failures free
  the request state, all allocated/referenced KV pages, and the pinned source;
- hit metrics are committed only after GPU COW and GDN restore succeed;
- optional `hybrid_prefix_checkpoint_interval_tokens` permits checkpoint
  intervals aligned to `prefix_match_unit`, rather than only whole KV blocks.

Reassigning a physical page invalidates every sub-block hash whose tail points
to that page. Earlier hash units in other pages remain reachable.

## M4 sparse internal GDN checkpoints

The experimental TP=1 internal-checkpoint path is implemented for a single
prefill request per batch:

- the GDN reference scan accepts sorted sparse token indices and clones state
  only at those indices instead of materializing per-token state history;
- every GDN layer returns recurrent and convolution checkpoints for the same
  sparse positions while still producing the final running state;
- Scheduler no longer clips a prefill at retained boundaries when internal
  checkpoints are enabled;
- ModelRunner executes one full prefill, assembles compact all-GDN-layer
  snapshots for each requested boundary, and temporarily stages them;
- the fixed-budget checkpoint pool can capture directly from staged tensors;
- staged tensors are discarded after publication or a duplicate/unused
  boundary, preventing request-lifetime leaks;
- `max_num_seqs=1` is enforced for this experimental path until ragged batches
  carry per-sequence checkpoint-position metadata.

This path is opt-in with `enable_hybrid_internal_checkpoints`; the default
remains the previously validated split-prefill behavior.

## M5 tensor-parallel checkpoint transaction

Hybrid Prefix Cache is no longer rejected solely because TP is greater than
one. Checkpoint capture and restore now use an all-rank transaction:

```text
local prepare/copy
-> all-reduce success
-> verify identical checkpoint slot IDs
-> rank 0 returns success
-> Scheduler publishes CPU metadata
```

If any rank fails capture, every successful rank frees and zeroes its local
checkpoint slot and reconstructs the free queue in deterministic slot order.
Slot-ID divergence is itself a rollback condition. If any rank fails restore,
every rank copies its private
request state backup back into place. Partial KV-page COW also performs an
all-rank success consensus; stale bytes on a failed private destination page
remain unreachable after Scheduler rolls back block ownership.

Only rank 0 raises the consensus failure to Scheduler. Worker ranks finish
rollback and return to their command loop, avoiding a worker-process crash.
Two-process Gloo fault-injection tests prove capture rollback, restore rollback,
slot-ID agreement, successful restore, COW failure consensus, and successful
work after a failed transaction. Existing TP=2 shard/forward tests also pass.
Real multi-GPU NCCL performance remains a hardware-platform validation item.

## Memory analysis

For one request boundary, the state bytes are:

```text
recurrent = L_gdn * H_value * D_key * D_value * 4
conv      = L_gdn * conv_dim * (kernel_size - 1) * model_dtype_bytes
```

At block size 256:

| Model | GDN snapshot | One full-attention KV block | Ratio |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 18.63 MiB | 3.00 MiB | 6.21x |
| Qwen3.5-9B | 49.13 MiB | 8.00 MiB | 6.14x |

Both scale down per rank under TP by similar head sharding factors, so the
ratio remains the central issue. Caching one state snapshot at every 256-token
block would make recurrent snapshots dominate cache capacity.

## Boundary convention

Define one convention and assert it everywhere:

```text
checkpoint(boundary=N)
= state after tokens [0, N) have been processed

resume input
= token at position N

seq.num_cached_tokens
= N
```

The lookup must never return all prompt tokens when logits for the next token
have not been cached. For the first implementation, preserve nano-vLLM's rule
of leaving the final prompt block/token range to recompute. Add exact-boundary
tests for `N-1`, `N`, and `N+1` to prevent double application.

## Proposed components

### 1. HybridPrefixCheckpointPool (per TP rank)

Preallocate a fixed-capacity GPU pool:

```text
recurrent_checkpoints[
  checkpoint_slot,
  compact_gdn_layer,
  value_head,
  key_dim,
  value_dim
]

conv_checkpoints[
  checkpoint_slot,
  compact_gdn_layer,
  channel,
  kernel_size - 1
]
```

Responsibilities:

- allocate/free immutable checkpoint slots;
- capture from a request state slot;
- restore by copying into a newly allocated request-owned state slot;
- maintain generation/version and last-access metadata;
- expose bytes, hits, copies, evictions and restore latency;
- use deterministic checkpoint IDs across TP ranks, with all-rank ack before
  publishing a cache entry.

Do not store arbitrary `torch.clone()` tensors in a Python dictionary; that
provides no hard memory budget and fragments the allocator.

### 2. HybridPrefixEntry (CPU metadata)

Attach to a prefix hash/boundary:

```text
prefix_hash
boundary_tokens
tail_kv_block_id / KV block chain identity
checkpoint_slot_id
cache_signature
last_access / policy metadata
```

The cache signature must include model/checkpoint revision, adapter/LoRA,
dtype/quantization/backend layout, TP world size/shard layout, and any other
input that changes hidden-state semantics.

### 3. HybridPrefixCoordinator

Lookup algorithm:

```text
kv_hit_length = longest valid full-attention block prefix
state_boundaries = resident checkpoint boundaries for matching hashes

hit_length = max(
    boundary <= min(kv_hit_length, prompt_safe_limit)
    for boundary in state_boundaries
)
```

Return zero if no common boundary exists. KV-only hits are not valid Hybrid
hits.

On hit:

1. increment/recover cached KV block references;
2. allocate request state slot;
3. restore checkpoint into that slot on every TP rank;
4. set `num_cached_tokens=hit_length`;
5. schedule the suffix from exactly `hit_length`.

Publish/restore must be transactional: no scheduler-visible hit until all rank
checkpoint copies are complete.

## Capture policy

### Phase 1: sparse, block-aligned checkpoints

Add a configurable checkpoint interval in full KV blocks, for example 4 or 8.
The scheduler caps a prefill chunk at the next selected boundary so the current
terminal state is exactly the checkpoint state. Also capture an aligned prompt
tail when policy allows.

Advantages:

- no per-token state-history allocation during long prefill;
- boundary is easy to prove;
- cache lookup and rollback remain block-aligned.

Cost:

- extra prefill launches at selected boundaries;
- shared prefixes shorter than the interval cannot hit;
- a system-prompt junction inside an interval reuses only the previous
  checkpoint and recomputes its suffix.

Start with interval 8 for Qwen3.5-9B/block-size 256 as a conservative memory
baseline (one ~49 MiB state per 2048 tokens per TP=1 rank), then benchmark
intervals 1/2/4/8. This is a starting experiment, not a universal default.

### Phase 2: adaptive/reachable boundaries

After Phase 1 correctness:

- always retain explicitly declared application/session prefix boundaries;
- retain observed shared-prefix junctions;
- optionally recompute a newly discovered junction once to seed future hits;
- support GPU-hot / CPU-cold checkpoint tiers;
- use benefit/cost eviction rather than plain LRU alone.

Suggested value score:

```text
estimated_saved_prefill_ms * observed_hit_frequency
---------------------------------------------------
checkpoint_bytes + pinned_kv_bytes
```

### Phase 3: decouple hash and physical block granularity

Only after Phase 1/2:

- allow a smaller prefix-match unit than the physical recurrent page;
- add partial-block copy-on-write;
- preserve exact state boundary metadata;
- add MTP finalization/lookahead rules.

This addresses short shared prompts and arbitrary chat/tool boundaries, but it
substantially expands correctness surface.

## MTP policy

For the first Hybrid Prefix Cache milestone, reject
`enable_prefix_cache && num_speculative_tokens > 0` with a clear error. Native
MTP owns shifted Full-Attention KV whose boundary differs from target KV/GDN
state. Once the target-only cache is proven, extend the atomic entry to include
or reconstruct MTP shifted KV and test accepted/rejected lookahead boundaries.

## Eviction policy

Treat KV and state as one logical entry, even if they use separate pools.

- If the checkpoint is evicted, that boundary is no longer a Hybrid hit.
- If a tail KV block is reassigned, evict/invalidate the attached checkpoint.
- Copy-on-hit means active request state no longer pins the immutable
  checkpoint after restore; KV block refs remain request-owned.
- Never publish an entry whose checkpoint slot is null, incomplete or from a
  different generation.
- Coordinate duplicate concurrent prefix computation so only complete entries
  win publication.

The first policy should be LRU among refcount-zero complete entries with a hard
snapshot budget. Add cost-aware retention only after metrics exist.

## Required correctness tests

1. cold versus warm: hidden/logits/greedy/state equality;
2. exact boundaries `N-1`, `N`, `N+1` and no double application;
3. one-shot versus chunked capture;
4. two concurrent requests hit one immutable snapshot and then diverge;
5. request slot reuse after hit has no contamination;
6. checkpoint eviction falls back to an earlier boundary or cold prefill;
7. KV eviction invalidates the associated checkpoint;
8. cancellation during capture does not publish a partial entry;
9. TP ranks publish/restore the same checkpoint ID and boundary;
10. hash collision still validates token content/cache signature;
11. Prefix Cache + preemption lifecycle;
12. MTP is rejected until its shifted cache has dedicated tests.

## Required metrics and workloads

Metrics:

```text
kv_candidate_hit_tokens
hybrid_committed_hit_tokens
tokens_lost_to_state_alignment
checkpoint_pool_bytes / slots / evictions
checkpoint_capture_ms / restore_ms
prefill_tokens_recomputed
TTFT / throughput / preemption count
```

Workloads:

- long shared system prompt + unique user suffix;
- multi-turn conversations;
- shared prefix shorter than checkpoint interval (negative control);
- no-shared-prefix traffic (overhead control);
- snapshot-pool pressure and eviction;
- chunked prefill crossing multiple selected boundaries.

## Implementation sequence in this repository

```text
HPC-0  DONE: config flags, memory estimator, metrics, reject MTP coexistence
HPC-1  DONE: fixed-budget checkpoint pool
HPC-2  DONE: sparse aligned boundary split/capture
HPC-3  DONE: KV/state hit intersection and synchronous restore
HPC-4  DONE: finish/preempt/abort/eviction/fallback/graph and TP transactions
HPC-5  DONE (single-run gate): official 0.8B equality + normal batched TTFT measured
HPC-6  DONE: adaptive retention, dense/adaptive/internal and no-share matrix
HPC-7  DONE: fine-grained partial hits and internal checkpoints
HPC-8  PENDING: optional CPU tier, ragged internal checkpoints, MTP coexistence
```

Do not begin with a radix tree rewrite. The correctness risk is the aligned
state transaction, not the choice of hash-table versus tree index. Prove that
on the existing `BlockManager` first; replace the lookup structure only when
workload evidence shows it is the bottleneck.

## Concrete patch map for the current codebase

### `nanovllm/config.py`

Add default-off controls:

```python
enable_hybrid_prefix_cache: bool = False
hybrid_prefix_checkpoint_interval_blocks: int = 8
hybrid_prefix_checkpoint_memory_bytes: int = 0
```

Validation for the first milestone:

```text
hybrid prefix cache requires recurrent-state model
memory budget > 0
interval > 0
num_speculative_tokens == 0
```

Use a byte budget rather than only a slot count because snapshot size changes
substantially across 0.8B/9B and TP sizes. ModelRunner computes bytes per local
snapshot and derives capacity.

### `nanovllm/engine/state_manager.py`

Add `HybridPrefixCheckpointPool`, separate from request slots:

```python
class HybridPrefixCheckpointPool:
    allocate(checkpoint_id) -> checkpoint_slot
    capture(checkpoint_slot, request_state_slot)
    restore(checkpoint_slot, request_state_slot)
    free(checkpoint_slot)
    contains(checkpoint_id) -> bool
    memory_bytes() -> int
```

Checkpoint tensors should use checkpoint as the leading dimension so one
checkpoint copy is contiguous across the compact GDN layer dimension where
practical. Request state and checkpoint state must never alias.

ModelRunner RPCs:

```python
capture_prefix_checkpoint(checkpoint_id, state_slot)
restore_prefix_checkpoint(checkpoint_id, state_slot)
evict_prefix_checkpoint(checkpoint_id)
```

The scheduler chooses deterministic IDs; every TP rank executes the same RPC.

### `nanovllm/engine/block_manager.py`

Do not overload the current integer `can_allocate()` result further. Introduce
an explicit result:

```python
@dataclass
class PrefixKVCandidate:
    num_cached_blocks: int
    boundary_tokens: int
    tail_hash: int
```

Add:

```python
find_kv_prefix_candidates(seq) -> list[PrefixKVCandidate]
allocate_from_prefix(seq, candidate)
```

Return candidates longest-first at complete block boundaries, excluding the
unsafe final prompt range just as the existing implementation does. Keep token
content validation after hash lookup.

### New `nanovllm/engine/hybrid_prefix_cache.py`

Own CPU metadata and policy:

```python
@dataclass
class HybridPrefixEntry:
    checkpoint_id: int
    prefix_hash: int
    boundary_tokens: int
    tail_block_id: int
    last_access_ns: int
    generation: int

class HybridPrefixCache:
    find_hit(seq, kv_candidates) -> HybridPrefixHit | None
    reserve_capture(prefix_hash, boundary) -> PendingCapture
    publish(pending, checkpoint_id)
    abort(pending)
    evict_until_fit(required_bytes)
```

Only `publish()` makes an entry visible. A pending capture is never a hit.

### `nanovllm/engine/scheduler.py`

Admission path:

```text
1. ask BlockManager for KV candidates
2. ask HybridPrefixCache for the longest candidate with a resident checkpoint
3. reserve/increment KV block references
4. allocate a fresh request state slot
5. RPC restore checkpoint into that slot
6. set num_cached_tokens to the exact shared boundary
```

If restore fails on any rank, roll back KV refs/state slot and cold-prefill.

Prefill scheduling path:

```python
next_boundary = next_interval_boundary(seq.num_cached_tokens)
seq.num_scheduled_tokens = min(
    normal_budget,
    next_boundary - seq.num_cached_tokens,
)
```

Only force this split when a checkpoint slot can be reserved or policy marks
the boundary valuable; otherwise do not add a useless launch.

### `nanovllm/engine/llm_engine.py`

Add a two-phase boundary commit around prefill execution:

```text
runner.run(prefill)                    # KV and request state now materialized
scheduler.prepare_prefix_captures()    # hashes/boundaries, not visible yet
runner.capture_prefix_checkpoint(...)  # all ranks copy state
scheduler.publish_prefix_captures()    # entry becomes visible atomically
scheduler.postprocess(...)
```

RPC acknowledgement already exists in ModelRunner shared-memory control and
should be reused. An exception calls `abort()` and leaves no visible entry.

### `nanovllm/engine/model_runner.py`

Create the checkpoint pool after the request state manager and before KV
budgeting. Subtract its hard byte budget from memory available to Paged KV.

Expose metrics:

```text
hybrid_checkpoint_slots_total/used
hybrid_checkpoint_bytes
capture_count / capture_ms
restore_count / restore_ms
eviction_count
```

### Boundary state capture

For Phase 1, scheduler splitting ensures the request state slot itself is the
state exactly at the selected boundary, so capture is a direct pool copy. Do
not enable long-prefill `return_state_history=True`: retaining per-token FP32
histories for long prompts defeats the memory goal.

Later, add selected-boundary kernel outputs rather than full per-token history
if launch splitting becomes measurable overhead.

## MVP acceptance gate

The feature is not complete merely because warm prompts run. It must prove:

```text
MTP off, default-off, safe cold fallback
block-aligned and sub-block COW paths
cold and warm hidden/logits/greedy equality
recurrent and conv states equal at resume boundary
hit followed by divergent decode remains isolated
snapshot eviction safely falls back
no-share workload overhead bounded and reported
snapshot budget visibly reduces available KV capacity
```

The gate is satisfied locally for TP=1 and with two-process Gloo transaction
fault injection. Real multi-GPU NCCL/9B validation and MTP coexistence remain
separate follow-up work.
