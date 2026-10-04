"""Torch profiler capture for the current correctness-first MTP-2 path."""

import argparse
import json

import torch
from torch.profiler import ProfilerActivity, profile

from benchmark_qwen35_mtp import DEFAULT_PROMPTS
from nanovllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--trace", default="/tmp/mtp-torch-trace.json")
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()

    engine = LLM(
        args.model,
        max_num_batched_tokens=16,
        max_num_seqs=len(DEFAULT_PROMPTS),
        max_model_len=64,
        gpu_memory_utilization=0.8,
        enforce_eager=args.eager,
        max_num_kvcache_blocks=4,
        max_num_state_slots=len(DEFAULT_PROMPTS),
        num_speculative_tokens=2,
        speculative_parallel_verify=True,
        gdn_decode_backend="cuda",
        enable_mtp_phase_profiling=True,
    )
    try:
        engine.generate(
            DEFAULT_PROMPTS,
            SamplingParams(
                temperature=0.0,
                max_tokens=4,
                ignore_eos=True,
            ),
            use_tqdm=False,
        )
        engine.reset_runtime_metrics()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            outputs = engine.generate(
                DEFAULT_PROMPTS,
                SamplingParams(
                    temperature=0.0,
                    max_tokens=args.output_tokens,
                    ignore_eos=True,
                ),
                use_tqdm=False,
            )
        prof.export_chrome_trace(args.trace)
        interesting = (
            "mtp.",
            "gdn.history.",
            "aten::clone",
            "aten::stack",
            "aten::copy_",
            "aten::index_select",
            "aten::index_copy_",
            "aten::cat",
        )
        rows = []
        for event in prof.key_averages():
            if not any(event.key.startswith(prefix) for prefix in interesting):
                continue
            rows.append(
                {
                    "name": event.key,
                    "count": event.count,
                    "self_cpu_ms": event.self_cpu_time_total / 1000,
                    "cpu_total_ms": event.cpu_time_total / 1000,
                    "self_device_ms": event.self_device_time_total / 1000,
                    "device_total_ms": event.device_time_total / 1000,
                    "cpu_memory_bytes": event.cpu_memory_usage,
                    "device_memory_bytes": event.device_memory_usage,
                }
            )
        print(
            json.dumps(
                {
                    "output_tokens": [item["token_ids"] for item in outputs],
                    "trace": args.trace,
                    "events": sorted(
                        rows,
                        key=lambda row: row["device_total_ms"],
                        reverse=True,
                    ),
                },
                indent=2,
            )
        )
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
