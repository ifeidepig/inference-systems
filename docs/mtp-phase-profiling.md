# Qwen3.5 MTP-2 Phase Profiling

## Scope

This study first profiles the pre-optimization correctness-first native MTP-2
implementation, then records the 2026-10-05 follow-up implementation and A/B.
It does not implement ReplaySSM, change the recurrence, alter the workload, or
remove full recurrent/conv state history.

Primary configuration:

```text
GPU: RTX 3060
Model: official Qwen3.5-0.8B-Base
Model dtype: BF16
GDN decode backend: fused CUDA
Speculative width: 2
Batch: 2 fixed prompts
Output: 32 tokens/request
CUDA Graph: enabled
Parallel verifier: enabled
```

Every measured run uses a shape-matched warmup. End-to-end results use 20
fresh engines with phase profiling disabled. Phase results use 10 fresh engines
and 11 MTP rounds per engine. All output streams matched target-only greedy
decoding.

## Execution graph

```text
Scheduler KV reservation (N_draft + 1)
-> Target decode of old pending token -> guaranteed base token
-> MTP draft 0
-> MTP draft 1
-> packed verify input [base, proposal_0]
-> parallel target verify + recurrent/conv history
-> greedy accept/reject
-> keep final active GDN state or selectively roll back an earlier boundary
-> build emitted tokens/final pending token
-> extra MTP-layer-only forward for shifted-KV alignment, without LM head
-> Scheduler logical KV commit and tail-page reclaim
```

Before the follow-up optimization, final alignment was a complete third MTP
step and computed an LM-head result that the caller discarded. Rejected target
KV rows are not copied back; logical lengths make them unreachable and complete
tail pages are reclaimed.

## End-to-end controlled result

20 fresh-engine repetitions:

| Metric | Target-only | MTP-2 |
| --- | ---: | ---: |
| Throughput median | 163.90 tok/s | 133.19 tok/s |
| Throughput IQR | 157.70-166.02 | 127.85-136.89 |
| Throughput P95 | 167.22 tok/s | 139.55 tok/s |
| Mean TPOT | 9.71 ms | 12.34 ms |
| Mean TTFT | 95.93 ms | 102.76 ms |
| Peak allocated | 1.647 GiB | 1.757 GiB |
| Draft acceptance | n/a | 65.91% |
| Token equality | reference | exact, 20/20 |

MTP throughput is 18.18% below target-only in this controlled workload.

Across one MTP run:

```text
proposed drafts: 44
accepted drafts: 29
acceptance: 65.91%
MTP rounds: 11
emitted tokens: 64 across batch=2
average accepted drafts/request/round: 29 / 22 = 1.318
average emitted tokens/request/round: 64 / 22 = 2.909
```

## Phase breakdown

CUDA Event medians aggregated across 10 fresh engines. Each engine contains 11
MTP rounds; the table reports the median of each engine's per-round median.

| Phase | GPU median/round | Share of measured GPU sum |
| --- | ---: | ---: |
| Target decode | 8.90 ms | 27.6% |
| Draft 0, including context/LM head/argmax | 2.25 ms | 7.0% |
| Draft 1, including context/LM head/argmax | 2.22 ms | 6.9% |
| Verify prepare + forward + argmax | 15.22 ms | 47.2% |
| Accepted state boundary/select/scatter/hidden select | 1.20 ms | 3.7% |
| Final MTP alignment | 2.22 ms | 6.9% |
| Remaining measured GPU metadata | 0.27 ms | 0.8% |
| Total phase sum | 32.27 ms | 100% |

ModelRunner wall time was about 35.99 ms/round. Target-only ModelRunner wall
time was about 10.04 ms/token, so 2.909 serial target tokens cost approximately
29.2 ms. The speculative round still costs roughly 6.8 ms more than the serial
work it replaces.

The three MTP steps cost about 6.69 ms together. Their LM heads alone account
for roughly 4.76 ms; final alignment's unused LM head costs about 1.58 ms/round.

## CPU and synchronization

Important CPU wall medians:

| Range | CPU wall |
| --- | ---: |
| Accept/reject | 14.49 ms |
| Initial metadata | 8.27 ms |
| Verify metadata preparation | 1.56 ms |
| State select enqueue | 0.40 ms |
| State scatter enqueue | 0.53 ms |
| Scheduler speculative commit | 0.031 ms |
| Scheduler KV reservation | 0.003 ms |

The large accept/reject and initial-metadata wall times are synchronization
locations, not independent pure-Python compute. They wait for queued GPU work
through CUDA scalar reads and tensor construction. They must not be added to
the GPU phase sum as separate non-overlapping costs.

Production-mode Nsight Systems node tracing, after subtracting a matching
prefill-only trace:

| Per unit | Target-only token | MTP round |
| --- | ---: | ---: |
| CUDA Graph launches | 1 | 5 |
| Host kernel launch APIs | 14 | 127 |
| GPU kernel activities | 1,281 | 5,274 |
| `cudaStreamSynchronize` | 1 | 30 |
| Async memcpy calls | 14 | 71 |

At 2.909 emitted tokens/round, MTP still performs about 10.3 stream
synchronizations per emitted token versus one for target-only. The principal
sources are GPU-scalar reads in greedy verification and GPU-to-CPU metadata
conversion.

## GDN state-history cost

Official 0.8B TP1 state boundary:

```text
recurrent: [18, 16, 128, 128] FP32 = 18.000 MiB
conv:      [18, 6144, 3] BF16    = 0.633 MiB
total:                               18.633 MiB
```

MTP-2, batch 2 materializes 74.53 MiB/round. The fixed-width verify graph also
owns a 74.53 MiB persistent history buffer at max batch 2. Analytical scaling:

| Batch | MTP-2 history |
| ---: | ---: |
| 1 | 37.27 MiB |
| 4 | 149.06 MiB |
| 8 | 298.13 MiB |
| 16 | 596.25 MiB |
| 32 | 1,192.50 MiB |

Explicit accepted-state select and scatter cost approximately 1.05 ms GPU per
round. An eager diagnostic with instrumentation-only history labels measured:

```text
recurrent clone + stack: ~0.75 ms GPU
conv clone + stack:      ~0.24 ms GPU
layer pack + cat:        ~0.68 ms GPU
total labelled history:  ~1.67 ms GPU/round
```

This eager diagnostic is not substituted for CUDA Graph latency. It bounds the
order of magnitude of clone/stack/pack operations; graph verify remains the
authoritative 14.91 ms phase measurement.

The MTP peak-allocation increase over target-only is about 0.110 GiB. The
74.53 MiB persistent history buffer explains a large fraction of the memory
delta, even though history manipulation is not the first latency bottleneck at
batch 2.

## Cost model

For the measured MTP-2 round:

```text
Cost_MTP_GPU
= Target decode                  8.90 ms
 + 2 draft steps                 4.47 ms
 + Parallel verify              15.22 ms
 + State select/scatter          1.20 ms
 + Final MTP alignment           2.22 ms
 + Other GPU metadata            0.27 ms
= 32.27 ms/round
```

Benefit:

```text
average emitted tokens = 2.909/request/round
serial target wall equivalent ~= 2.909 * 10.04 = 29.2 ms
```

Therefore high acceptance does not create a speedup: packed verify saves some
serial target work, but verify itself is the largest phase and the extra draft,
alignment, state, and synchronization work pushes the round above break-even.

## Bottleneck ranking

1. **Parallel target verify** - 15.22 ms GPU, 47% of phase sum. It includes
   target compute and correctness-first history generation.
2. **Target decode** - 8.90 ms GPU, required once per round.
3. **Three MTP steps** - 6.69 ms GPU. The extra alignment LM head is unused and
   is a higher-confidence optimization target than ReplaySSM.
4. **CPU/GPU synchronization and metadata** - 30 stream synchronizations per
   round; accept/reject is the largest visible synchronization wall.
5. **State-history selection/scatter and materialization** - about 1.05 ms
   explicit graph-path selection/scatter plus ~1.67 ms eager diagnostic history
   clone/stack/pack. Important for memory scaling, but not the first latency
   bottleneck at batch 2.
6. **Scheduler bookkeeping** - tens of microseconds; not material.

## ReplaySSM decision

```text
Not recommended now
```

For the current 0.8B, MTP-2, batch-2 workload, Verify and the three MTP steps
dominate. Even an optimistic removal of all labelled history manipulation plus
state select/scatter would not clearly reduce the 32.27 ms round below the
~29.2 ms serial-target break-even point.

ReplaySSM should be reconsidered when:

- batch 16/32 persistent history buffers materially reduce concurrency;
- a larger model makes history traffic a larger portion of verify;
- graph-path history materialization is isolated and measured above the eager
  diagnostic estimate;
- draft/verify/alignment and synchronization optimizations have already been
  exhausted.

Higher-priority measured opportunities are:

1. skip the unused final-alignment LM head;
2. replace Python CUDA-scalar accept/reject with a vectorized GPU result plus
   one compact host transfer;
3. remove repeated GPU-to-CPU position conversions from MTP context metadata;
4. only then re-profile history before designing ReplaySSM.

## Implemented follow-up: alignment projection removal and selective rollback

On 2026-10-05 the first optimization above was implemented. `_mtp_step` now
accepts `compute_logits=False`; final alignment still executes the MTP layer so
its shifted KV cache is correct, but it no longer runs the discarded
vocabulary projection. Draft steps continue to compute logits normally.

The state commit path was also tightened without introducing a second active
state pool. Parallel verify already leaves each request's active GDN slot at
the final verify boundary. The runner now keeps that state in place whenever
it is the selected boundary and gathers/scatters history only for requests
that must roll back to an earlier boundary. This is deliberately not a raw
pointer switch into the verify-history buffer: that buffer is graph-owned and
reused on the next replay, while active request state must remain mutable and
valid across decode, preemption, prefix capture, cancellation, and TP ranks.

Same-machine before/after A/B, RTX 3060, official 0.8B BF16, batch 2,
32 output tokens/request, CUDA Graph, fused CUDA GDN backend, five fresh
engines per arm:

| Metric | Before | After |
| --- | ---: | ---: |
| Target-only throughput median | 164.46 tok/s | 165.19 tok/s |
| MTP-2 throughput median | 136.16 tok/s | 144.97 tok/s |
| MTP vs target-only | -17.21% | -12.24% |
| Token equality | exact, 5/5 | exact, 5/5 |

The MTP median improved by 6.47% relative to the pre-change MTP path. This is
still a negative result versus target-only and must not be presented as a
general MTP speedup.

One shape-matched phase-profile run measured:

| Phase | Before | After, no rollback rows |
| --- | ---: | ---: |
| Final alignment LM head | 1.64 ms | removed |
| Final alignment total | 2.37 ms | 0.71 ms |
| State select | 0.49 ms | 0.14 ms |
| State scatter | 0.58 ms | 0.05 ms |
| Selected/scattered state bytes over 11 rounds | 410 MiB | 0 MiB |

The no-rollback trace reused 22/22 active rows. A controlled two-request test
also covers mixed boundaries in one batch: one request reuses the active final
state while one request restores an earlier history row. Eager and CUDA Graph
outputs remain identical to target-only greedy decoding.

## Implemented follow-up: minimal rollback snapshot

Selective rollback removed the final-boundary select/scatter, but the verifier
still cloned and stored every boundary, including the state already resident in
the active slot. The next change replaces dense history with fixed sparse
checkpoint indices:

```text
MTP width K
rollback snapshots = H0 ... H(K-2)
active final state = H(K-1)
```

For MTP-2, only `H0` is captured. Accepted length zero restores `H0`; accepted
length one or two keeps active `H1`. For K=1 no rollback history is needed.
The same rule generalizes to K=4 as three rollback snapshots plus one active
final boundary. Eager verification uses `state_checkpoint_indices`, while the
CUDA Graph owns a fixed `batch x (K-1)` history buffer. This removes the final
state clone/stack and graph-buffer copy rather than merely ignoring it later.

Focused correctness coverage now includes:

- K=1/2/4 eager and CUDA Graph parity against target-only greedy;
- forced immediate rejection under parallel eager and graph verification;
- two requests with different accepted lengths in one batch;
- sparse recurrent/conv checkpoints against full history and independent
  prefix scans.

Official 0.8B BF16, RTX 3060, batch 2, MTP-2, 32 output tokens/request,
CUDA Graph and fused CUDA GDN. Five fresh profiling engines per implementation:

| Metric | Dense K-boundary history | Minimal K-1 history | Change |
| --- | ---: | ---: | ---: |
| Verify-forward GPU median | 15.79 ms | 13.98 ms | -11.4% |
| Profiled ModelRunner wall median | 34.83 ms | 33.13 ms | -4.9% |
| History bytes over 11 rounds | 820 MiB | 410 MiB | -50% |
| Persistent verify history buffer | 74.53 MiB | 37.27 MiB | -50% |
| Peak allocated median | 1.751 GiB | 1.716 GiB | about -36 MiB |

Profiling-off five-engine MTP A/B:

| Metric | Dense history | Minimal history | Change |
| --- | ---: | ---: | ---: |
| Throughput median | 141.90 tok/s | 145.96 tok/s | +2.86% |
| Throughput IQR | 137.97-143.45 | 144.80-146.63 | — |
| Mean TPOT median | 11.24 ms | 10.98 ms | -2.31% |
| Peak allocated median | 1.751 GiB | 1.722 GiB | about -30 MiB |

The matched target-only median in the minimal-history run was 165.39 tok/s,
so MTP remains 11.75% slower despite the improvement. All five MTP streams
matched target-only greedy. Separate official-model K=1 and K=4 CUDA-Graph
smokes also matched target-only tokens.

This optimization removes one snapshot per request, not a constant fraction:
MTP-2 history falls by 50%, MTP-4 by 25%, and MTP-8 by 12.5%. Its absolute
saving still scales with batch size, number of GDN layers and state size.

## Tool boundary

Nsight Compute returned:

```text
ERR_NVGPUCTRPERM
NCU hardware counters unavailable
```

No claims are made about achieved occupancy, bandwidth utilization, cache hit
rate, or memory-bound percentage.

## Reproduction

```bash
# End-to-end, profiling disabled
python benchmark_qwen35_mtp.py \
  --model /path/to/Qwen3.5-0.8B-Base \
  --mode both --output-tokens 32 --cuda-graph \
  --repetitions 20 --gdn-decode-backend cuda

# Phase timing
python benchmark_qwen35_mtp.py \
  --model /path/to/Qwen3.5-0.8B-Base \
  --mode both --output-tokens 32 --cuda-graph \
  --repetitions 10 --phase-profile --gdn-decode-backend cuda

# Torch profiler diagnostic
python profile_qwen35_mtp_torch.py \
  --model /path/to/Qwen3.5-0.8B-Base \
  --output-tokens 4 --eager
```
