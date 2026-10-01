# Qwen3.5 Fused GDN Decode Kernel

## Status

The repository now contains a first CUDA implementation of the one-token
Gated DeltaNet recurrent core. It is deliberately **opt-in**. The default
backend remains the previously validated PyTorch path.

The CUDA source builds successfully for SM 8.6 and registers
`nanovllm::gdn_decode`. Local RTX 3060 Laptop validation now covers numerical
differential tests, Compute Sanitizer, CUDA Graph replay, and an official
Qwen3.5-0.8B end-to-end A/B. The backend remains opt-in until 9B and real
TP/NCCL validation are complete.

## Fusion boundary

The custom operation consumes already projected `qkv`, `a`, and `b` tensors
and fuses:

```text
conv history shift + width-4 depthwise convolution
-> SiLU
-> Q/K normalization
-> decay and beta
-> FP32 recurrent-state transition
-> per-value-head core output
```

The projection GEMMs, gated RMSNorm, output projection, and TP all-reduce stay
outside the operation.

## CUDA v0 contract

The first specialization targets the shapes shared by the official Qwen3.5
0.8B and planned 9B validation path:

- FP16 or BF16 projection/conv tensors;
- FP32 recurrent state and `A_log`;
- 128-dimensional key and value heads;
- equal local key/value head counts;
- convolution width 4;
- contiguous tensors on one CUDA device.

One CUDA block owns one `(request, local_value_head)` pair. It uses 128
threads, one per value dimension, and stages the 128x128 FP32 recurrent matrix
in dynamic shared memory. Q/K reductions are block-local; every recurrent
element is read from and written to global memory once.

The cubin compiled for SM 8.6 reports 39 registers/thread for FP16 and BF16.
The initial fully unrolled implementation used 168 registers/thread; limiting
the three 128-step loops to four-way unrolling removed that avoidable pressure.
The dynamic shared-memory function attribute is configured through
`std::call_once`, rather than issuing `cudaFuncSetAttribute` on every eager
decode token.

## Backend selection

Python API:

```python
llm = LLM(
    model_path,
    gdn_decode_backend="torch",  # torch | cuda | auto
)
```

Server CLI:

```bash
nanovllm-serve \
  --model /path/to/Qwen3.5-0.8B-Base \
  --gdn-decode-backend cuda
```

Behavior:

- `torch`: always use the established reference/vectorized path;
- `cuda`: require the CUDA v0 contract and built operator, otherwise fail;
- `auto`: use CUDA only when the contract and operator are available, otherwise
  fall back to Torch.

The explicit `cuda` mode intentionally fails instead of hiding an unsupported
shape or missing extension.

## Build

```bash
conda activate dl
CUDA_HOME="$CONDA_PREFIX" \
TORCH_CUDA_ARCH_LIST=8.6 \
python setup.py build_ext --inplace
```

## Local validation results

Environment:

```text
GPU: NVIDIA GeForce RTX 3060 Laptop 6 GiB, SM 8.6
PyTorch: 2.8.0+cu128
Model: official Qwen3.5-0.8B-Base, BF16
```

Differential coverage includes FP16/BF16 and batch 1/2/4/8/16. Across the
recorded random cases, maximum absolute errors were:

```text
core output:      <= 1.5258789e-05
recurrent state: <= 9.0673566e-06
conv state:       exactly equal
```

CUDA Graph capture/replay passed. Compute Sanitizer reported zero memcheck
errors and zero racecheck hazards.

BF16 microbenchmark medians, 20 warmups, 100 iterations, 7 repeats:

| Batch | Torch core us | CUDA core us | Core speedup | Torch layer us | CUDA layer us | Layer speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 385.09 | 27.37 | 14.07x | 605.92 | 246.96 | 2.45x |
| 2 | 383.70 | 40.85 | 9.39x | 592.67 | 244.91 | 2.42x |
| 4 | 383.71 | 63.17 | 6.07x | 598.97 | 248.22 | 2.41x |
| 8 | 408.66 | 108.87 | 3.75x | 606.63 | 250.25 | 2.42x |
| 16 | 559.01 | 200.87 | 2.78x | 687.06 | 307.97 | 2.23x |

Official 0.8B CUDA Graph A/B used two requests, 32 output tokens per request,
and three fresh-engine repetitions:

| Backend | Throughput tok/s | Mean TPOT ms | Mean TTFT ms | Tokens stable |
| --- | ---: | ---: | ---: | --- |
| Torch | 148.29 +/- 9.11 | 11.00 | 92.07 | yes |
| CUDA | 164.72 +/- 5.17 | 9.58 | 91.73 | yes |

The CUDA backend improved throughput by about 11.1% and reduced mean TPOT by
about 12.9%; TTFT stayed effectively flat, as expected for a decode-only
kernel. CUDA and Torch output token IDs were exactly equal. An eager-mode
official-model run also matched the same greedy token sequence.

MTP-2 with the fused backend remained correct: target-only and MTP output IDs
matched, acceptance was 65.9%, and state recompute forwards stayed at zero.
It was still 14.95% slower than the now-faster target-only path, so MTP remains
a measured negative optimization for this workload.

### Hybrid state lifecycle

The official-model Fine-Grained Hybrid Prefix Cache matrix was repeated with
the CUDA backend and CUDA Graph enabled. All five variants matched the cold
baseline. The fine-grained paths restored a 496-token prefix, copied 2,949,120
bytes through partial-page COW, and then decoded through the fused backend.

A dedicated forced lifecycle validation uses a 240-token internal checkpoint:

```text
initial admission -> restore + partial-page COW
first completion token
forced preemption -> release mutable KV/state
re-admission -> restore + partial-page COW
CUDA Graph decode -> finish
```

It produced the same four output tokens as the Torch cold run, recorded one
preemption, two restores, two COW operations (5,898,240 copied bytes total),
and two CUDA Graph replays.

For CUDA Graph padding, a real batch of 3 was run against a captured batch of
4. The extra row was mapped to the scratch state slot on every replay. A
128-token-per-request run accumulated 254 padded rows across two repetitions;
all real request outputs stayed equal between Torch and CUDA.

### Interleaved end-to-end A/B

`benchmark_gdn_backend.py` alternates backend order by cycle:

```text
cycle 0: Torch -> CUDA
cycle 1: CUDA -> Torch
```

Selected official 0.8B results:

| Execution | Concurrency | Output/request | Throughput change | TPOT change | Token result |
| --- | ---: | ---: | ---: | ---: | --- |
| eager | 1 | 32 | +29.47% | -25.33% | exact |
| eager | 3 | 32 | +25.77% | -25.98% | exact |
| graph | 1 | 128 | +12.40% | -11.90% | exact |
| graph4, real batch 3 | 3 | 128 | +15.06% | -15.70% | exact |
| eager | 8 | 32 | +17.64% | -23.60% | one tie-sensitive divergence |
| graph | 8 | 32 | +9.62% | -20.78% | same tie-sensitive divergence |

The concurrency-8 divergence is deterministic in both backends and was traced
to request 4, output position 10. Torch produced an exact BF16 logit tie:

```text
token 16 = 15.4375
token 17 = 15.4375
```

Torch `argmax` selected token 16 by index order. The fused reduction order
produced token 17 = 15.5 and token 16 = 15.4375, selecting token 17 and causing
the later sequence to diverge. A 32-step batch-8 recurrence differential test
remained bounded at `7.63e-6` maximum core-output error and `3.35e-8` maximum
state error, with exact conv state. This is a greedy tie-breaking sensitivity,
not state corruption, but it means the project must not claim universal token
identity for the optimized backend. The default therefore remains `torch`.

### Nsight evidence

Nsight Systems profiled the same batch-1 microbenchmark for both backends. The
Torch trace contained 2,169 `cudaLaunchKernel` calls; the fused trace contained
421, a reduction of about 80.6%. The 46 captured fused-kernel instances averaged
about 17.3 us of GPU time.

Nsight Compute could not access hardware performance counters on this host
(`ERR_NVGPUCTRPERM`). Consequently, DRAM bandwidth and achieved occupancy are
not reported. Static evidence is limited to 39 registers/thread and about 66
KiB dynamic shared memory per block. On SM 8.6 that shared-memory footprint is
expected to limit this kernel to one resident block per SM; this is an
inference, not an NCU measurement.

## Validation commands

CPU dispatch/reference tests:

```bash
PYTHONPATH=. python -m unittest -v \
  tests.kernels.test_gdn_decode_dispatch
```

CUDA differential and graph tests:

```bash
PYTHONPATH=. python -m unittest -v \
  tests.kernels.test_gdn_decode_cuda
```

Microbenchmark:

```bash
PYTHONPATH=. python benchmark_gdn_decode.py \
  --device cuda --dtype bfloat16 --backend torch --batch-size 1

PYTHONPATH=. python benchmark_gdn_decode.py \
  --device cuda --dtype bfloat16 --backend cuda --batch-size 1
```

Interleaved end-to-end A/B:

```bash
PYTHONPATH=. python benchmark_gdn_backend.py \
  --model /path/to/Qwen3.5-0.8B-Base \
  --concurrency 1,3,8 \
  --max-num-seqs 8 \
  --output-tokens 32,128 \
  --execution both \
  --cycles 2
```

Prefix/COW/preemption lifecycle:

```bash
PYTHONPATH=. python tests/validate_qwen35_gdn_cuda_lifecycle.py \
  --model /path/to/Qwen3.5-0.8B-Base
```

Required remaining validation:

1. Nsight Compute bandwidth/occupancy analysis on a host that permits GPU
   performance counters;
2. optional reduction-order work if strict greedy identity at exact BF16 ties
   is a product requirement;
3. 9B and real TP/NCCL validation on the compute platform.
