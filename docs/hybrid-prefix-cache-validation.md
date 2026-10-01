# Hybrid Prefix Cache Validation

Date: 2026-09-27

## Scope

Experimental MVP constraints:

```text
Qwen3.5 dense hybrid text runtime
TP=1 real-GPU performance; TP=2 transaction logic validated with Gloo
MTP disabled
full-block baseline plus fine-grained checkpoints
fixed GPU checkpoint byte budget
default off
```

## Tiny model correctness

The tiny 3-GDN + 1-Full-Attention test uses two 300-token prompts with the
same first 256 tokens and different 44-token suffixes.

Validated:

- producer split and checkpoint capture at token boundary 256;
- second request restores one KV block plus recurrent/conv checkpoint;
- four greedy output tokens equal an independent cold baseline;
- equality holds with eager and CUDA Graph decode;
- a restored request can be preempted, re-admitted from the immutable
  checkpoint, then aborted without deleting or contaminating the checkpoint;
- a one-slot checkpoint pool evicts the old snapshot under a second prefix;
- when the old KV candidate remains but its checkpoint was evicted, lookup
  falls back to cold prefill and output remains equal;
- pool capture/restore/free copies are isolated and freed slots are zeroed.

## Official Qwen3.5-0.8B checkpoint

Command:

```bash
PYTHONSAFEPATH=1 PYTHONPATH=. \
python benchmark_hybrid_prefix_cache.py \
  --model /path/to/Qwen3.5-0.8B-Base \
  --shared-prefix-length 256 \
  --unique-suffix-length 4 \
  --output-tokens 2 \
  --interval-blocks 1 \
  --checkpoint-memory-mib 64 \
  --max-num-batched-tokens 1 \
  --max-num-kvcache-blocks 4 \
  --gpu-memory-utilization 0.95
```

The unusually small token budget is intentional: desktop graphics processes
currently occupy about 3 GiB of the 6 GiB GPU, and the normal 8-token warmup
causes the conservative KV budget check to reject initialization. This run is
a real-checkpoint correctness/observability gate, not a production benchmark.

Results:

```text
cold and warm output tokens: exact match
KV candidate tokens:         256
committed Hybrid hit tokens: 256
state-alignment lost tokens: 0
prefix hit blocks:           1
snapshot pool capacity:      3
snapshot bytes per slot:     19,537,920
allocated checkpoint bytes:  58,613,760
restore count:               1
restore latency:             0.187 ms

cold prefill model runs:     260
warm prefill model runs:     4
cold TTFT:                   9244.49 ms
warm TTFT:                   139.87 ms
```

The 98.5% TTFT reduction reflects an artificial one-token chunk schedule and
only proves that 256 tokens were skipped. Representative TTFT/throughput and
checkpoint-interval comparisons remain pending a GPU window with normal
headroom and realistic batched prefill.

### Normal 300-token batched prefill window

When desktop GPU usage later fell enough for the conservative budget check,
the same official checkpoint was rerun with `max_num_batched_tokens=300`, a
256-token shared prefix, 44-token divergent suffix, and 4 output tokens:

```text
cold output == warm output:       true
cold prefill model runs:          1 (300 tokens)
warm prefill model runs:          1 (44 suffix tokens)
committed Hybrid hit:             256 tokens / 1 KV block
state-alignment lost tokens:      0
checkpoint pool:                  3 slots / 58,613,760 bytes
restore latency:                  0.189 ms

cold TTFT:                        1459.32 ms
warm TTFT:                         250.14 ms
TTFT delta:                         -82.86%
cold TPOT:                          35.36 ms
warm TPOT:                          35.34 ms
cold peak allocated:                 1.606 GiB
warm peak allocated:                 1.637 GiB
```

This is a single short run after a matching producer request, so it proves the
expected prefill-only performance direction rather than a statistically stable
production speedup. TPOT remaining unchanged is consistent with Prefix Cache
not accelerating decode.

## Fine-grained 240/256 partial-page validation

The M3 path was validated with the official 0.8B checkpoint using a physical
KV block size of 256, `prefix_match_unit=16`, a 240-token shared prefix, and
different 64-token suffixes. The retained GDN checkpoint is at token 240, so
the second request must copy the first 240 rows of the cached physical page to
a private page before writing its suffix.

Eager result:

```text
cold output == warm output:       true
committed Hybrid hit:             240 tokens
COW operations:                   1
COW bytes across all KV layers:   2,949,120
COW latency:                      0.077 ms
GDN restore latency:              0.166 ms
cold TTFT:                        1493.27 ms
warm TTFT:                         353.36 ms
TTFT delta:                        -76.34%
```

CUDA Graph decode result:

```text
cold output == warm output:       true
committed Hybrid hit:             240 tokens
COW operations:                   1
COW bytes:                        2,949,120
COW latency:                      0.122 ms
GDN restore latency:              0.174 ms
decode CUDA Graph replays:        1
cold TTFT:                        1482.97 ms
warm TTFT:                         353.47 ms
TTFT delta:                        -76.16%
```

These are single local runs, not statistical claims. They prove that the
fine-grained index, all-layer KV COW, GDN restore, suffix prefill, and CUDA
Graph decode compose correctly on a real checkpoint.

Additional deterministic tests cover:

- a 500-token shared prefix resolving to 496 for block=256/match=16;
- `N-1/N/N+1` matching around a 240-token boundary: 224/240/240;
- source and destination physical pages being distinct;
- injected COW failure restoring an empty request block table, null state slot,
  zero block references, and no used physical blocks;
- the prior eager/graph hit -> preempt -> re-hit -> abort lifecycle test.

### Internal checkpoint versus split prefill

On the same official 0.8B, 304-token producer workload with one retained
checkpoint at token 240:

```text
split prefill:
  scheduler steps:       2
  prefill model runs:    2
  producer TTFT:         1528.39 ms

internal checkpoint:
  scheduler steps:       1
  prefill model runs:    1
  sparse checkpoints:    1
  staged state bytes:    19,537,920
  producer TTFT:         1504.80 ms

relative producer TTFT:  -1.54% (single controlled A/B run)
```

Both variants subsequently committed the same 240-token fine-grained hit,
performed one partial-page COW, and produced the same greedy token as the cold
baseline. The timing is directional rather than statistically stable; the
strong evidence is removal of one scheduler step and one full model forward.

## Evidence boundary

Proven:

- official checkpoint compatibility;
- KV/state aligned hit correctness at one complete block;
- state restoration before suffix prefill;
- output equivalence;
- byte-budget accounting and restore timing.

Not yet proven:

- real multi-GPU NCCL performance and 9B behavior;
- MTP coexistence;
- TP fine-grained publication/restore;
- repeated performance and interval 1/2/4/8 comparisons;
- cancellation during a multi-rank capture;

## Five-variant workload matrix

`benchmark_fine_grained_prefix_cache.py` runs no-cache, 256-token
block-aligned, fine-dense, fine-adaptive, and fine-internal variants with a
fixed checkpoint byte budget. A local official-0.8B single-run matrix used a
496-token shared prefix, one divergent suffix token, two output tokens, a
16-token match unit, and a 640 MiB checkpoint budget.

Shared-prefix results:

| Variant | Hit | TTFT | TPOT | Throughput | Delta | Producer forwards |
|---|---:|---:|---:|---:|---:|---:|
| no cache | 0 | 2438.69 ms | 40.16 ms | 0.81 tok/s | baseline | 1 |
| block aligned | 256 | 1186.91 ms | 39.28 ms | 1.63 tok/s | -51.33% | 2 |
| fine dense | 496 | 38.23 ms | 38.44 ms | 25.99 tok/s | -98.43% | 32 |
| fine adaptive | 496 | 39.15 ms | 37.68 ms | 25.93 tok/s | -98.39% | 2 |
| fine internal | 496 | 38.83 ms | 38.93 ms | 25.63 tok/s | -98.41% | 1 |

All variants produced exactly the same tokens. Fine variants copied 2,949,120
KV bytes; measured COW latency was 0.078-0.103 ms. The large TTFT percentage is
expected for a target with only one uncached prompt token and is not a general
serving claim.

No-share negative-control results:

| Variant | Hit | Target TTFT delta | Producer forwards |
|---|---:|---:|---:|
| block aligned | 0 | +5.47% | 2 |
| fine dense | 0 | +61.14% | 32 |
| fine adaptive | 0 | +1.97% | 2 |
| fine internal | 0 | +1.35% | 1 |

This demonstrates the central policy result: fine matching alone is not
enough. Dense checkpoint retention destroys producer/no-share performance;
adaptive retention removes most of that cost, and internal checkpoints remove
the extra model forward.
