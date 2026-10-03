"""Microbenchmark compressed Hybrid Prefix Checkpoint storage."""

import argparse
import json
import statistics

import torch

from nanovllm.engine.state_manager import (
    HybridPrefixCheckpointPool,
    HybridStateManager,
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(int(len(ordered) * fraction), len(ordered) - 1)
    return ordered[index]


def timed_cuda(callable_) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    callable_()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def run_format(args, checkpoint_dtype: str) -> dict:
    manager = HybridStateManager(
        max_num_seqs=1,
        num_linear_layers=args.num_linear_layers,
        num_value_heads=args.num_value_heads,
        key_head_dim=args.key_head_dim,
        value_head_dim=args.value_head_dim,
        conv_dim=args.conv_dim,
        conv_kernel_size=args.conv_kernel_size,
        conv_dtype=torch.bfloat16,
        device="cuda",
    )
    request_slot = manager.allocate()
    manager.recurrent_states[:, request_slot].normal_()
    manager.conv_states[:, request_slot].normal_()
    pool = HybridPrefixCheckpointPool(
        manager,
        memory_budget_bytes=args.memory_budget_mib * 1024 * 1024,
        checkpoint_dtype=checkpoint_dtype,
    )

    for _ in range(args.warmup):
        slot = pool.capture(request_slot)
        pool.restore(slot, request_slot)
        pool.free(slot)
    torch.cuda.synchronize()

    capture_ms = []
    restore_ms = []
    peak_temporary_bytes = 0
    for _ in range(args.iterations):
        torch.cuda.reset_peak_memory_stats()
        baseline_bytes = torch.cuda.memory_allocated()
        holder = []
        capture_ms.append(
            timed_cuda(lambda: holder.append(pool.capture(request_slot)))
        )
        peak_temporary_bytes = max(
            peak_temporary_bytes,
            torch.cuda.max_memory_allocated() - baseline_bytes,
        )
        checkpoint_slot = holder[0]
        manager.recurrent_states[:, request_slot].zero_()
        manager.conv_states[:, request_slot].zero_()
        restore_ms.append(
            timed_cuda(lambda: pool.restore(checkpoint_slot, request_slot))
        )
        pool.free(checkpoint_slot)
        torch.cuda.synchronize()

    return {
        "checkpoint_dtype": checkpoint_dtype,
        "bytes_per_checkpoint": pool.bytes_per_checkpoint,
        "capacity": pool.capacity,
        "pool_bytes": pool.memory_bytes(),
        "capture_ms_median": statistics.median(capture_ms),
        "capture_ms_p95": percentile(capture_ms, 0.95),
        "restore_ms_median": statistics.median(restore_ms),
        "restore_ms_p95": percentile(restore_ms, 0.95),
        "peak_temporary_bytes": peak_temporary_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-budget-mib", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--num-linear-layers", type=int, default=18)
    parser.add_argument("--num-value-heads", type=int, default=16)
    parser.add_argument("--key-head-dim", type=int, default=128)
    parser.add_argument("--value-head-dim", type=int, default=128)
    parser.add_argument("--conv-dim", type=int, default=6144)
    parser.add_argument("--conv-kernel-size", type=int, default=4)
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 0:
        parser.error("iterations must be positive and warmup non-negative")

    results = [
        run_format(args, checkpoint_dtype)
        for checkpoint_dtype in ("fp32", "bf16", "int8")
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
