import argparse

import torch

from nanovllm.kernels import fused_add_rmsnorm
from nanovllm.layers.layernorm import RMSNorm


def benchmark(function, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(1)
    module = RMSNorm(args.hidden_size).cuda().to(dtype)
    x = torch.randn(args.rows, args.hidden_size, device="cuda", dtype=dtype)
    residual = torch.randn_like(x)

    baseline = lambda: module.add_rms_forward(x, residual)
    custom = lambda: fused_add_rmsnorm(
        x,
        residual,
        module.weight,
        module.eps,
    )

    baseline_us = benchmark(baseline, args.warmup, args.iterations)
    custom_us = benchmark(custom, args.warmup, args.iterations)

    output, residual_output = custom()
    reference_output, reference_residual = baseline()
    absolute_error = (output.float() - reference_output.float()).abs()

    print(f"rows:                 {args.rows}")
    print(f"hidden_size:          {args.hidden_size}")
    print(f"dtype:                {args.dtype}")
    print(f"torch.compile:        {baseline_us:.3f} us")
    print(f"custom CUDA:          {custom_us:.3f} us")
    print(f"speedup:              {baseline_us / custom_us:.3f}x")
    print(f"max absolute error:   {absolute_error.max().item():.8f}")
    print(f"mean absolute error:  {absolute_error.mean().item():.8f}")
    print(f"residual exact:       {torch.equal(residual_output, reference_residual)}")


if __name__ == "__main__":
    main()
