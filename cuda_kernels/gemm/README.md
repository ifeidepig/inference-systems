# CUDA GEMM Optimization Ladder

A compact CUDA project that progresses from scalar FP32 GEMM to shared/register tiling and FP16/BF16 Tensor Core WMMA, then builds a measured shape/dtype-aware dispatch against an equal-contract cuBLAS baseline.

The goal is not to claim a general replacement for cuBLAS. It is to show the complete performance-engineering loop:

```text
correctness contract
-> memory/layout design
-> CUDA Core tiling
-> Tensor Core WMMA
-> fair cuBLAS baseline
-> multi-shape measurement
-> dispatch only where the custom path wins stably
-> retain negative results elsewhere
```

## Implemented Paths

### FP32 CUDA Core

- one-thread-per-output naive kernel;
- 16x16 shared-memory tiled kernel;
- 32x32 CTA tile with 2x2 per-thread register accumulation;
- irregular `127x193x65` boundary validation;
- row-major cuBLAS SGEMM baseline using the same FP32 contract.

### FP16/BF16 Tensor Core

- row-major `A[M,K]`, row-major `B[K,N]`, FP32 accumulator/output;
- direct WMMA: one warp computes one 16x16 output tile;
- staged WMMA: eight warps share 32x16 A and 16x64 B tiles;
- FP16 and BF16 template specializations;
- row-major cuBLAS `GemmEx` baseline with FP32 compute/output;
- 11 Qwen3.5/square shapes per dtype, 22 combinations total;
- dtype/shape-aware dispatch with cuBLAS fallback.

Both WMMA paths lower to Tensor Core instructions on the RTX 3060:

```text
HMMA.16816.F32
HMMA.16816.F32.BF16
```

## Shape-Aware Dispatch

The benchmark measures every candidate but dispatches to the custom kernel only in the stable winning regime:

```text
BF16 && M == 16 && K == 1024 && N >= 4096
    -> direct WMMA
otherwise
    -> cuBLAS
```

Why the policy is narrow:

- BF16 QKV (`16x1024x6144`) was 1.89x-2.02x cuBLAS across five fresh-process runs;
- BF16 Gate-Up (`16x1024x7168`) was 1.71x-1.74x cuBLAS;
- FP16 QKV ranged from 0.91x to 1.11x, so it is not a safe custom dispatch;
- larger M and narrow-N projections strongly favor cuBLAS;
- staged WMMA was slower on every measured shape and remains a documented negative experiment.

## Local Methodology

```text
GPU:       NVIDIA RTX 3060 Laptop, 6 GiB, SM 8.6
Toolkit:   CUDA 13.4
Driver:    595.91.07
Inputs:    deterministic FP16/BF16 matrices
Output:    FP32
Reference: cuBLAS with matching input/output/compute types and layout
Timing:    CUDA Events; 10 warmups; adaptive repeats; 7 median trials
```

The row-major cuBLAS mapping computes:

```text
C = A * B
```

as the column-major equivalent:

```text
C^T = B^T * A^T
```

No transpose kernel or host/device transfer is included in the timed region.

## Selected Results

One recorded 7-trial run from [`results/rtx3060_gemm_matrix.csv`](results/rtx3060_gemm_matrix.csv):

| Dtype / shape | cuBLAS | Direct WMMA | Direct/cuBLAS | Dispatch |
| --- | ---: | ---: | ---: | --- |
| BF16 `16x1024x6144` QKV | 0.0933 ms | 0.0500 ms | 1.87x | Direct WMMA |
| BF16 `16x1024x7168` Gate-Up | 0.0901 ms | 0.0525 ms | 1.71x | Direct WMMA |
| FP16 `16x1024x6144` QKV | 0.0540 ms | 0.0462 ms | 1.17x in this run, unstable across processes | cuBLAS |
| BF16 `64x1024x6144` QKV | 0.0725 ms | 0.1570 ms | 0.46x | cuBLAS |
| BF16 `512x512x512` | 0.0241 ms | 0.0458 ms | 0.53x | cuBLAS |

All 22 FP16/BF16 combinations passed the cuBLAS differential check. For the deterministic matrix suite, direct and staged outputs were exactly equal to the recorded cuBLAS FP32 output.

The FP32 irregular-shape ladder produced approximately:

```text
naive:          0.00863 ms
shared tiled:   0.00846 ms
register tiled: 0.00817 ms
cuBLAS SGEMM:   0.00818 ms
```

This tiny irregular case is launch-limited and is retained as a tiling/boundary exercise, not evidence of general cuBLAS parity.

## Profiling and Safety

- Compute Sanitizer memcheck: 0 errors.
- Compute Sanitizer racecheck: 0 hazards.
- SASS confirms FP16 and BF16 HMMA instructions.
- Static resource report:
  - direct FP16/BF16: 42 registers/thread, 0 B shared memory;
  - staged FP16: 48 registers/thread, 3,072 B shared memory;
  - staged BF16: 56 registers/thread, 3,072 B shared memory.
- Nsight Systems on BF16 QKV under profiling:
  - direct WMMA average GPU time: about 48.2 us;
  - cuBLAS main GEMM + split-K reduction: about 112.6 us total;
  - staged WMMA: about 143.4 us.

Nsight Compute hardware counters are unavailable on this host (`ERR_NVGPUCTRPERM`), so achieved bandwidth and Tensor Core utilization are not reported.

## Build

```bash
cmake -S . -B build/release \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.4/bin/nvcc
cmake --build build/release --parallel
```

The default target architecture is `sm_86`. Override `CMAKE_CUDA_ARCHITECTURES` when building for another GPU.

## Run

Full FP16/BF16 matrix:

```bash
./build/release/gemm_bench --dtype both --trials 7
```

One shape:

```bash
./build/release/gemm_bench \
  --dtype bf16 \
  --label qwen35_prefill16_qkv \
  --trials 7
```

Short sanitizer run:

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 \
  ./build/release/gemm_bench \
  --dtype both --label qwen35_prefill16_qkv --trials 1 --repeats 1

compute-sanitizer --tool racecheck --error-exitcode 1 \
  ./build/release/gemm_bench \
  --dtype both --label qwen35_prefill16_qkv --trials 1 --repeats 1
```

FP32 ladder:

```bash
./build/release/gemm_tiled
```

## Files

```text
src/gemm.cu         minimal FP32 GEMM and transfer timing
src/gemm_tiled.cu   FP32 naive/shared/register ladder + cuBLAS
src/gemm_wmma.cu    original direct/staged WMMA learning program
src/gemm_bench.cu   multi-dtype, multi-shape cuBLAS benchmark and dispatch
results/             recorded local result matrix
```

## Known Limitations

- The custom dispatch is intentionally specialized for BF16 small-M/wide-N projections.
- Decode M=1, larger prefill M, FP16, and narrow-N shapes fall back to cuBLAS.
- The staged WMMA implementation is correctness-complete but not performance competitive.
- There is no `cp.async`, double buffering, CUTLASS/CuTe implementation, or fused epilogue yet.
- The dispatch is a standalone CUDA backend prototype and is not wired into nano-vLLM Linear layers.
- Results are from one Ampere laptop GPU and are not generalized to other architectures.
