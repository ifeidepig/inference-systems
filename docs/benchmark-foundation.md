# Serving Foundation Benchmark

Date: 2026-09-12

## Environment

- GPU: NVIDIA GeForce RTX 3060 Laptop GPU, 6 GiB
- Model: Qwen3-0.6B, BF16
- Runtime: Python 3.11, PyTorch 2.8.0+cu128
- Scheduling policy: decode first
- CUDA Graph workload: 8 requests, 128 input tokens, 32 output tokens
- Prefix Cache workload: 768 input tokens, 8 output tokens
- Chunked Prefill workload: 768-token prefill injected into an active decode

## Results

| Feature and metric | Baseline | Enabled | Change |
| --- | ---: | ---: | ---: |
| Prefix Cache warm TTFT | 75.45 +/- 1.55 ms | 24.63 +/- 0.38 ms | -67.36% (3.06x) |
| CUDA Graph mean TPOT | 25.70 +/- 0.95 ms | 8.86 +/- 0.66 ms | -65.52% |
| CUDA Graph output throughput | 285.84 +/- 9.36 tok/s | 754.12 +/- 15.10 tok/s | +163.82% |
| CUDA Graph measured duration | 896.59 +/- 30.25 ms | 339.61 +/- 6.97 ms | -62.12% |
| Chunked Prefill decode interference gap | 64.86 +/- 2.17 ms | 48.57 +/- 0.88 ms | -25.12% |
| Chunked Prefill total duration | 771.48 +/- 8.13 ms | 841.00 +/- 12.10 ms | +9.01% |

Values are population mean +/- standard deviation.

## Tail Latency

| Metric | P50 | P95 | P99 |
| --- | ---: | ---: | ---: |
| Prefix Cache warm TTFT, disabled | 74.98 ms | 78.74 ms | 80.22 ms |
| Prefix Cache warm TTFT, enabled | 24.52 ms | 25.44 ms | 25.79 ms |
| CUDA Graph TPOT, eager | 25.68 ms | 27.43 ms | 28.64 ms |
| CUDA Graph TPOT, replay | 9.05 ms | 9.66 ms | 10.16 ms |
| Chunked Prefill gap, disabled | 64.32 ms | 69.57 ms | 70.85 ms |
| Chunked Prefill gap, enabled | 48.28 ms | 50.27 ms | 50.88 ms |

## Methodology

CUDA Graph and eager execution were each measured for 30 in-process trials after
shape-matched warmup. Every trial used distinct prompt prefixes to prevent
unintended prefix-cache hits. The enabled path recorded 33 graph replays per
trial; the baseline recorded 33 eager decode calls per trial.

Prefix Cache used one cold request followed by 30 identical warm requests. The
enabled run reused 60 of 62 eligible prompt blocks. The disabled run recorded no
cache queries, hashes, or hits.

Chunked Prefill used 30 trials per configuration. The enabled configuration split
the injected prefill into three partial-prefill steps per trial. The disabled
configuration executed one full prefill and recorded no partial-prefill steps.

## Correctness Checks

- All 27 unit tests passed.
- Every serving and interference trial ended with zero active KV blocks.
- Completion lengths matched the configured output lengths in all trials.
- Real health, chat completion, text completion streaming, and metrics HTTP
  requests returned status 200.
- Streaming emitted incremental text and a final `[DONE]` event.
- The API server completed graceful shutdown with exit code 0.

## Interpretation

CUDA Graph is the clearest latency and throughput win for this workload because
decode repeatedly launches the same model shape. Prefix Cache substantially
reduces repeated-prompt TTFT by skipping full prompt blocks. Chunked Prefill
reduces the stall observed by an active decode request, but it increases total
completion time because the prefill is split across more model executions. This
is a latency-throughput tradeoff rather than a universal speedup.

Raw JSON files are stored locally under `benchmark_results/foundation-30/`.
