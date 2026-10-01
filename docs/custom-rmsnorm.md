# Custom CUDA RMSNorm

## Scope

The optional custom extension implements the two RMSNorm semantics used by
`nanovllm.layers.RMSNorm`:

```text
rmsnorm(x, weight, eps) -> output

fused_add_rmsnorm(x, residual, weight, eps)
    -> output, residual_output
```

The correctness-first implementation supports:

- CUDA FP16 and BF16;
- hidden sizes 128 and 1024;
- arbitrary contiguous leading dimensions;
- PyTorch's current CUDA stream;
- `torch.compile(fullgraph=True)`;
- CUDA Graph capture and replay.

Unsupported dtype, shape, device, or layout falls back to the existing
`torch.compile` implementation at the `RMSNorm` module boundary.

## Numerical contract

The kernels intentionally match the current Python implementation's rounding
order. Fused add computes the residual sum in FP32, writes a low-precision
residual output, and computes the variance from the unrounded FP32 sum:

```text
h_fp32 = fp32(x) + fp32(residual)
residual_output = cast_input_dtype(h_fp32)
variance = mean(h_fp32 * h_fp32)
normalized_fp32 = h_fp32 * rsqrt(variance + epsilon)
normalized_lowp = cast_input_dtype(normalized_fp32)
output = normalized_lowp * weight
```

The first kernel version therefore caches `h_fp32` in registers. Packed
low-precision cache, global reread, shared-memory cache, and in-place mutation
are separate performance experiments because they change either rounding,
resource usage, aliasing, or all three.

## Build

The loader first tries the ahead-of-time module `nanovllm._C`, then falls back
to a JIT extension for development. The Conda environment contains the CUDA
12.8 compiler that matches PyTorch 2.8.0+cu128, while `/usr/local/cuda` points
to CUDA 13.4. Select the matching toolkit explicitly:

```bash
conda activate dl
CUDA_HOME="$CONDA_PREFIX" \
TORCH_CUDA_ARCH_LIST=8.6 \
python setup.py build_ext --inplace
```

An editable installation can use:

```bash
CUDA_HOME="$CONDA_PREFIX" \
TORCH_CUDA_ARCH_LIST=8.6 \
pip install -e . --no-build-isolation
```

The extension loader also discovers headers installed by split NVIDIA Python
packages such as `nvidia-cusparse-cu12`.

## Enable

The custom path is opt-in. Default behavior remains unchanged.

```bash
export NANOVLLM_USE_CUSTOM_RMSNORM=1
python example.py
```

The model used by the local benchmark is Qwen3-0.6B BF16:

```text
hidden_size = 1024
head_dim = 128
num_hidden_layers = 28
```

Consequently, the current specializations cover all 113 RMSNorm modules: two
layer norms and q/k norms in each decoder layer, plus the final norm.

## Tests

Run the focused suite:

```bash
PYTHONPATH=. python -m unittest -v \
  tests.kernels.test_custom_add \
  tests.kernels.test_fused_add_rmsnorm \
  tests.kernels.test_rmsnorm_module_dispatch
```

The suite covers:

- FP16 and BF16;
- H=128 and H=1024;
- 2-D and 3-D inputs;
- exact residual output;
- zero input and unsupported-H errors;
- current-stream execution;
- `torch.compile(fullgraph=True)`;
- CUDA Graph capture/replay;
- module-level opt-in and fallback.

Compute Sanitizer commands:

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 \
  python -c "import torch; from nanovllm.kernels import fused_add_rmsnorm; \
x=torch.randn(2,1024,device='cuda',dtype=torch.bfloat16); \
r=torch.randn_like(x); w=torch.randn(1024,device='cuda',dtype=torch.bfloat16); \
fused_add_rmsnorm(x,r,w,1e-6); torch.cuda.synchronize()"

compute-sanitizer --tool racecheck --error-exitcode 1 \
  python -c "import torch; from nanovllm.kernels import fused_add_rmsnorm; \
x=torch.randn(2,1024,device='cuda',dtype=torch.bfloat16); \
r=torch.randn_like(x); w=torch.randn(1024,device='cuda',dtype=torch.bfloat16); \
fused_add_rmsnorm(x,r,w,1e-6); torch.cuda.synchronize()"
```

## Benchmarks

Microbenchmark:

```bash
PYTHONPATH=. python benchmark_fused_rmsnorm.py \
  --rows 32 --hidden-size 1024 --dtype bfloat16
```

Model-level A/B with CUDA Graph:

```bash
PYTHONPATH=. python benchmark_model_rmsnorm.py \
  --trials 3 --batch-size 4 --max-tokens 16

PYTHONPATH=. python benchmark_model_rmsnorm.py \
  --custom --trials 3 --batch-size 4 --max-tokens 16
```

Initial local validation on RTX 3060 Laptop showed:

```text
BF16 H=1024 rows=32 microbenchmark:
torch.compile baseline: 76.37 us
custom CUDA:           10.72 us

Qwen3-0.6B CUDA Graph, batch=4, 16 generated tokens/request:
baseline median: 162.96 ms
custom median:   163.02 ms
```

The model-level result is effectively neutral. It is not valid to promote the
microkernel speedup as an end-to-end speedup: CUDA Graph removes much launch
overhead and GEMM/attention dominate the model step. The three-trial result is
an integration check, not a final statistical performance claim.

## Verified resources

`cuobjdump --dump-resource-usage` reported:

| Kernel specialization | Registers/thread | Shared memory | Local memory |
|---|---:|---:|---:|
| H=128 FP16/BF16 | 17 | 20 B | 0 |
| H=1024 FP16/BF16 | 22 | 36 B | 0 |

## Remaining optimization work

The following are intentionally outside the correctness-first integration:

- vectorized `half2` / `bfloat162` IO;
- packed low-precision versus FP32 register cache ablation;
- global-reread and shared-memory-cache variants;
- in-place mutation schema and alias analysis;
- larger hidden-size specializations;
- NCU counter collection and multi-order statistical benchmarking;
- broader model and GPU coverage.
