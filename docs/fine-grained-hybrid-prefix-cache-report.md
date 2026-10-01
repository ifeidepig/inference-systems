# Fine-Grained Hybrid Prefix Cache Report

Date: 2026-09-30

## Outcome

The Qwen3.5 text runtime now has an opt-in Fine-Grained Hybrid Prefix Cache
that coordinates Full-Attention Paged KV and GDN recurrent/conv checkpoints.
The implementation keeps the original default-off behavior and cold-prefill
fallback.

Implemented milestones:

1. typed Full-Attention/GDN managers and `HybridPrefixCoordinator`;
2. periodic/adaptive checkpoint retention and LRU/cost-aware eviction;
3. sub-block chained hashes, partial-page all-layer KV COW, atomic admission;
4. sparse internal GDN checkpoints from one prefill forward;
5. all-rank capture/restore/COW consensus with rollback and rank-0 publish.

## Key invariants

```text
checkpoint(N) = GDN state after tokens [0,N)
resume input  = token position N

committed hit = longest boundary with both:
  reachable Full-Attention KV prefix
  resident GDN recurrent + conv checkpoint
```

A sub-block hit never writes into its cached source page. It pins the source,
allocates a private destination, copies every Full-Attention layer's K/V rows,
restores GDN state, and only then commits hit metadata.

## Reproduction

Shared-prefix policy matrix:

```bash
PYTHONSAFEPATH=1 PYTHONPATH=. \
python benchmark_fine_grained_prefix_cache.py \
  --model /path/to/Qwen3.5-0.8B-Base \
  --shared-prefix-length 496 \
  --unique-suffix-length 1 \
  --output-tokens 2 \
  --checkpoint-memory-mib 640 \
  --max-num-batched-tokens 512 \
  --max-num-kvcache-blocks 32 \
  --summary-only
```

Negative control: add `--no-share`.

## Local official-0.8B evidence

| Variant | Hit | TTFT | TPOT | Output throughput | Producer forwards | No-share TTFT delta |
|---|---:|---:|---:|---:|---:|---:|
| no cache | 0 | 2438.69 ms | 40.16 ms | 0.81 tok/s | 1 | baseline |
| block aligned | 256 | 1186.91 ms | 39.28 ms | 1.63 tok/s | 2 | +5.47% |
| fine dense | 496 | 38.23 ms | 38.44 ms | 25.99 tok/s | 32 | +61.14% |
| fine adaptive | 496 | 39.15 ms | 37.68 ms | 25.93 tok/s | 2 | +1.97% |
| fine internal | 496 | 38.83 ms | 38.93 ms | 25.63 tok/s | 1 | +1.35% |

All variants produced identical greedy tokens. Fine hits copied 2,949,120 KV
bytes with a measured local COW latency of 0.078-0.103 ms. These are single
local runs intended to establish mechanism and direction, not production
confidence intervals.

Every enabled variant used the same 664,289,280-byte preallocated checkpoint
budget in this matrix. The summary distinguishes allocated checkpoint bytes
from COW bytes and committed hit tokens.

CUDA Graph decode was separately validated with both split and internal
checkpoint producers; output matched cold execution and decode graph replay
was observed.

## Transaction evidence

Two-process Gloo tests inject failures on rank 1 and verify:

- successful rank-local capture slots are freed and zeroed;
- request states are restored after a partial restore;
- slot IDs agree before rank-0 publication;
- worker ranks remain alive after rollback;
- a successful capture/restore/COW can run after an injected failure.

The existing TP=2 GDN shard and forward-equivalence tests also pass. The host
has one GPU, so real NCCL and Qwen3.5-9B performance remain cloud validation.

## Resume-ready contribution

> Designed and implemented a Qwen3.5 Fine-Grained Hybrid Prefix Cache for
> nano-vLLM, decoupling physical KV pages, prefix hash granularity, and GDN
> checkpoint retention. Added all-layer partial-page KV copy-on-write,
> adaptive/cost-aware checkpointing, sparse internal recurrent checkpoints,
> and all-rank transactional publication/rollback. On an official 0.8B
> checkpoint, increased a controlled shared-prefix hit from 256 to 496 tokens,
> while reducing no-share overhead from dense checkpointing's +61.1% TTFT to
> +1.35% with adaptive internal checkpoints; verified exact tokens, CUDA Graph
> decode, lifecycle isolation, and distributed fault rollback.

Use the exact hardware/workload qualifiers above when presenting the numbers.
