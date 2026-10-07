# Compact ReplaySSM Final Batch/Concurrency Sweep

## Decision

```text
ReplaySSM feature freeze
```

No further ReplaySSM kernel or runtime features will be added in the current
project phase. The compact path remains default-off as a research prototype.
It must not be presented as a general throughput acceleration or a
quality-safe replacement for the default MTP state path.

Reasons:

1. speculative-state memory decreases consistently as concurrency grows;
2. throughput is workload/batch dependent rather than monotonic;
3. exact-token parity is prompt- and batch-shape-sensitive in the current BF16
   GDN/MTP stack;
4. the complete output-only periodic-flush kernel would substantially expand
   scope beyond the evidence available on the local GPU.

## Method

```text
GPU: RTX 3060 Laptop 6 GiB
Model: official Qwen3.5-0.8B-Base BF16
MTP width: 2
Batch/concurrency: 1 / 2 / 4 / 8
Output: 32 tokens/request
CUDA Graph: enabled
GDN backend: fused CUDA
Runs: 5 fresh engines per mode and batch
Order: alternating Minimal/Replay by repetition
```

Each batch uses deterministic four-token prompts. Prompts are identical across
the Minimal Snapshot and Compact Replay arms. The sweep reports medians and IQR
instead of a single run.

Reproduction:

```bash
python benchmark_replayssm_sweep.py \
  --model /path/to/Qwen3.5-0.8B-Base \
  --batch-sizes 1 2 4 8 \
  --repetitions 5 \
  --output-tokens 32 \
  --num-speculative-tokens 2 \
  --gdn-decode-backend cuda
```

## Serving results

| Batch | Minimal tok/s | Replay tok/s | Throughput | Minimal TPOT | Replay TPOT | TPOT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 92.26 | 94.27 | +2.18% | 9.22 ms | 9.02 ms | -2.14% |
| 2 | 143.89 | 146.89 | +2.08% | 10.96 ms | 10.69 ms | -2.50% |
| 4 | 200.73 | 210.02 | +4.63% | 14.92 ms | 13.62 ms | -8.70% |
| 8 | 274.11 | 263.07 | -4.03% | 20.19 ms | 19.63 ms | -2.77% |

The batch-4 point is positive, but the batch-8 throughput reverses sign. The
result is not evidence of a general concurrency crossover. On this short
workload, total throughput includes TTFT/engine scheduling effects, while TPOT
isolates only post-first-token intervals. More workloads would be required to
claim a stable speed regime, and this project explicitly stops here.

## Memory scaling

| Batch | Minimal peak | Replay peak | Approx. saved |
| ---: | ---: | ---: | ---: |
| 1 | 1.736 GiB | 1.720 GiB | 16.1 MiB |
| 2 | 1.810 GiB | 1.776 GiB | 35.0 MiB |
| 4 | 1.957 GiB | 1.893 GiB | 65.5 MiB |
| 8 | 2.252 GiB | 2.120 GiB | 134.4 MiB |

The savings grow approximately with active request count, which is the expected
benefit of replacing one full recurrent snapshot per request with compact
records plus conv windows.

The model-level peak difference includes graph/runtime allocations and is not
identical to the analytical Verify buffer difference. The controlled batch-2
buffer calculation remains:

```text
Minimal Snapshot: 37.27 MiB
Compact Replay:    3.66 MiB
Reduction:         10.18x
```

## Token parity boundary

Within each arm, every five-run token digest was stable. Across arms:

| Batch | Minimal vs Replay | Minimal vs target-only | Replay vs target-only |
| ---: | --- | --- | --- |
| 1 | different | different | exact |
| 2 | exact | exact | exact |
| 4 | different | different | exact |
| 8 | different | different | different |

Observed first paired divergences:

- batch 1: request 0, output index 8, `57308` vs `11957`;
- batch 4: request 0, output index 8, same branch;
- batch 8: request 7, output index 16, `58` vs `8`.

This does not establish that Compact Replay is always less accurate: at batch 1
and 4 it matched target-only while Minimal Snapshot selected the other branch.
Instead, it demonstrates that the current BF16 GDN/MTP stack is sensitive to
batch shape and floating-point association near an argmax boundary. Since the
project requires conservative token-parity evidence, the feature cannot be
promoted to default or claimed quality-safe.

## Final claim boundary

Allowed:

> Implemented a default-off Compact ReplaySSM prototype that replaces full GDN
> speculative recurrent snapshots with compact transition records and fused
> commit kernels; reduced the batch-2 Verify buffer from 37.27 MiB to 3.66 MiB
> and demonstrated request-count-scaled peak-memory savings.

Not allowed:

- “ReplaySSM improved throughput by 4.6%” without batch/workload qualifiers;
- “ReplaySSM is token exact” as a general claim;
- “implemented full output-only ReplaySSM”;
- “production-ready ReplaySSM backend”.

Raw aggregate:

`benchmark_results/mtp_replayssm_batch_sweep_20261007.json`

