from __future__ import annotations

import argparse
import json
import statistics
import time
from types import SimpleNamespace

import torch

from nanovllm.kernels.gdn import gdn_decode_core
from nanovllm.layers.gated_delta_net import GatedDeltaNet


def measure(
    function,
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
    repeats: int,
) -> list[float]:
    with torch.inference_mode():
        for _ in range(warmup):
            function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        samples = []
        for _ in range(repeats):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iterations):
                    function()
                stop.record()
                stop.synchronize()
                elapsed_us = start.elapsed_time(stop) * 1000.0 / iterations
            else:
                start_time = time.perf_counter_ns()
                for _ in range(iterations):
                    function()
                elapsed_us = (
                    (time.perf_counter_ns() - start_time) / 1000.0 / iterations
                )
            samples.append(elapsed_us)
    return samples


def summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    p90_index = min(len(ordered) - 1, int(0.9 * len(ordered)))
    return {
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "p90_us": ordered[p90_index],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the one-token Qwen3.5 GDN decode core and layer."
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--num-key-heads", type=int, default=16)
    parser.add_argument("--num-value-heads", type=int, default=16)
    parser.add_argument("--key-head-dim", type=int, default=128)
    parser.add_argument("--value-head-dim", type=int, default=128)
    parser.add_argument("--conv-width", type=int, default=4)
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--backend", choices=("torch", "cuda", "auto"), default="torch")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.iterations <= 0 or args.repeats <= 0:
        parser.error("batch-size, iterations, and repeats must be positive")
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    device = torch.device(device_name)
    dtype = getattr(torch, args.dtype)
    if device.type == "cpu" and dtype != torch.float32:
        parser.error("CPU baseline currently requires --dtype float32")

    config = SimpleNamespace(
        hidden_size=args.hidden_size,
        linear_num_key_heads=args.num_key_heads,
        linear_num_value_heads=args.num_value_heads,
        linear_key_head_dim=args.key_head_dim,
        linear_value_head_dim=args.value_head_dim,
        linear_conv_kernel_dim=args.conv_width,
        rms_norm_eps=1e-6,
        gdn_decode_backend=args.backend,
    )
    torch.manual_seed(47)
    module = GatedDeltaNet(config, layer_idx=0).eval().to(device=device, dtype=dtype)
    # Production constructs A_log explicitly in FP32 even when the surrounding
    # model uses BF16/FP16. Restore that invariant after the benchmark's bulk
    # module conversion.
    module.A_log.data = module.A_log.data.float()
    batch_size = args.batch_size
    hidden_states = torch.randn(
        batch_size, args.hidden_size, device=device, dtype=dtype
    )
    recurrent_states = torch.randn(
        batch_size,
        args.num_value_heads,
        args.key_head_dim,
        args.value_head_dim,
        device=device,
        dtype=torch.float32,
    )
    conv_states = torch.randn(
        batch_size,
        module.conv_dim,
        args.conv_width - 1,
        device=device,
        dtype=dtype,
    )

    with torch.inference_mode():
        projected_qkv = module.in_proj_qkv(hidden_states)
        a = module.in_proj_a(hidden_states)
        b = module.in_proj_b(hidden_states)

    core = lambda: gdn_decode_core(
        projected_qkv,
        a,
        b,
        module.conv1d.weight.squeeze(1),
        module.A_log,
        module.dt_bias,
        recurrent_states,
        conv_states,
        num_key_heads=module.num_key_heads,
        num_value_heads=module.num_value_heads,
        key_head_dim=module.key_head_dim,
        value_head_dim=module.value_head_dim,
        backend=args.backend,
    )
    layer = lambda: module._forward_batched_decode(
        hidden_states,
        recurrent_states,
        conv_states,
    )
    core_samples = measure(
        core,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    layer_samples = measure(
        layer,
        device=device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    report = {
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "dtype": args.dtype,
        "backend": args.backend,
        "batch_size": batch_size,
        "hidden_size": args.hidden_size,
        "num_key_heads": args.num_key_heads,
        "num_value_heads": args.num_value_heads,
        "key_head_dim": args.key_head_dim,
        "value_head_dim": args.value_head_dim,
        "conv_width": args.conv_width,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "core": summarize(core_samples),
        "layer": summarize(layer_samples),
    }
    if args.json:
        print(json.dumps(report, indent=2))
        return
    for key, value in report.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for metric, number in value.items():
                print(f"  {metric}: {number:.3f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
