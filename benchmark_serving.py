"""Measure request-level latency and throughput for nano-vLLM.

This benchmark intentionally uses the engine step API instead of timing only
``LLM.generate``.  That lets it observe when every request produces its first
token and when it finishes.
"""

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams


@dataclass
class RequestMetrics:
    request_id: int
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float
    tpot_ms: float
    e2e_ms: float


def percentile(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated percentile without extra dependencies."""
    if not values:
        raise ValueError("cannot compute a percentile of an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values),
        "stddev": statistics.pstdev(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def make_prompt(token_id: int, unique_token_id: int, length: int) -> list[int]:
    if length <= 0:
        raise ValueError("prompt length must be positive")
    return [unique_token_id] + [token_id] * (length - 1)


def make_prompts(
    base_token_id: int,
    length: int,
    count: int,
    vocab_size: int,
    unique_offset: int = 1,
) -> list[list[int]]:
    return [
        make_prompt(
            base_token_id,
            (base_token_id + unique_offset + request_index) % vocab_size,
            length,
        )
        for request_index in range(count)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("benchmark.json"))
    parser.add_argument("--num-requests", type=int, default=4)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--output-length", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--scheduling-policy",
        choices=("prefill_first", "decode_first", "slo_aware"),
        default="prefill_first",
    )
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument("--disable-prefix-cache", action="store_true")
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.prompt_length + args.output_length > args.max_model_len:
        raise ValueError("prompt and output lengths exceed --max-model-len")
    if args.num_requests <= 0:
        raise ValueError("--num-requests must be positive")
    if args.output_length <= 0:
        raise ValueError("--output-length must be positive")
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")


def run_trial(
    llm: LLM,
    args: argparse.Namespace,
    sampling: SamplingParams,
    base_token_id: int,
    vocab_size: int,
    unique_offset: int,
) -> dict:
    torch.cuda.reset_peak_memory_stats()
    llm.reset_runtime_metrics()
    submitted_at: dict[int, float] = {}
    sequences = {}
    first_token_at: dict[int, float] = {}
    finished_at: dict[int, float] = {}

    benchmark_started = perf_counter()
    prompts = make_prompts(
        base_token_id,
        args.prompt_length,
        args.num_requests,
        vocab_size,
        unique_offset=unique_offset,
    )
    for prompt in prompts:
        submitted = perf_counter()
        llm.add_request(prompt, sampling)
        sequence = llm.scheduler.waiting[-1]
        submitted_at[sequence.seq_id] = submitted
        sequences[sequence.seq_id] = sequence

    prefill_time = 0.0
    decode_time = 0.0
    prefill_steps = 0
    decode_steps = 0
    mixed_steps = 0
    mixed_time = 0.0
    peak_kv_blocks_used = 0

    while not llm.is_finished():
        step_started = perf_counter()
        _, stats = llm.step()
        torch.cuda.synchronize()
        step_finished = perf_counter()
        step_time = step_finished - step_started

        if stats.prefill_tokens and stats.decode_tokens:
            mixed_time += step_time
            mixed_steps += 1
        elif stats.prefill_tokens:
            prefill_time += step_time
            prefill_steps += 1
        else:
            decode_time += step_time
            decode_steps += 1

        blocks_used = len(llm.scheduler.block_manager.used_block_ids)
        peak_kv_blocks_used = max(peak_kv_blocks_used, blocks_used)

        for request_id, sequence in sequences.items():
            if sequence.num_completion_tokens and request_id not in first_token_at:
                first_token_at[request_id] = step_finished
            if sequence.is_finished and request_id not in finished_at:
                finished_at[request_id] = step_finished

    torch.cuda.synchronize()
    benchmark_finished = perf_counter()
    active_kv_blocks = len(llm.scheduler.block_manager.used_block_ids)
    if active_kv_blocks:
        raise RuntimeError(f"KV block leak: {active_kv_blocks} blocks remain active")

    requests = []
    for request_id, sequence in sequences.items():
        first = first_token_at[request_id]
        finished = finished_at[request_id]
        output_tokens = sequence.num_completion_tokens
        requests.append(
            RequestMetrics(
                request_id=request_id,
                prompt_tokens=sequence.num_prompt_tokens,
                output_tokens=output_tokens,
                ttft_ms=(first - submitted_at[request_id]) * 1000,
                tpot_ms=(finished - first) * 1000 / max(output_tokens - 1, 1),
                e2e_ms=(finished - submitted_at[request_id]) * 1000,
            )
        )

    duration = benchmark_finished - benchmark_started
    total_output_tokens = sum(request.output_tokens for request in requests)
    return {
        "summary": {
            "duration_s": duration,
            "request_throughput_rps": len(requests) / duration,
            "output_throughput_tokens_per_s": total_output_tokens / duration,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            "total_kv_blocks": len(llm.scheduler.block_manager.blocks),
            "peak_kv_blocks_used": peak_kv_blocks_used,
            "peak_kv_block_utilization": (
                peak_kv_blocks_used / len(llm.scheduler.block_manager.blocks)
            ),
            "prefill_steps": prefill_steps,
            "prefill_time_s": prefill_time,
            "decode_steps": decode_steps,
            "decode_time_s": decode_time,
            "mixed_steps": mixed_steps,
            "mixed_time_s": mixed_time,
            "ttft_ms": summarize([request.ttft_ms for request in requests]),
            "tpot_ms": summarize([request.tpot_ms for request in requests]),
            "e2e_ms": summarize([request.e2e_ms for request in requests]),
        },
        "runtime_metrics": llm.get_runtime_metrics(),
        "requests": [asdict(request) for request in requests],
    }


def aggregate_trials(trials: list[dict]) -> dict:
    summaries = [trial["summary"] for trial in trials]
    requests = [request for trial in trials for request in trial["requests"]]
    return {
        "duration_s": summarize([summary["duration_s"] for summary in summaries]),
        "request_throughput_rps": summarize(
            [summary["request_throughput_rps"] for summary in summaries]
        ),
        "output_throughput_tokens_per_s": summarize(
            [summary["output_throughput_tokens_per_s"] for summary in summaries]
        ),
        "peak_allocated_gib": max(
            summary["peak_allocated_gib"] for summary in summaries
        ),
        "peak_reserved_gib": max(
            summary["peak_reserved_gib"] for summary in summaries
        ),
        "ttft_ms": summarize([request["ttft_ms"] for request in requests]),
        "tpot_ms": summarize([request["tpot_ms"] for request in requests]),
        "e2e_ms": summarize([request["e2e_ms"] for request in requests]),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    llm = LLM(
        str(args.model),
        enforce_eager=not args.use_cuda_graph,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        scheduling_policy=args.scheduling_policy,
        enable_prefix_cache=not args.disable_prefix_cache,
        enable_chunked_prefill=not args.disable_chunked_prefill,
    )

    candidate_ids = llm.tokenizer.encode(" benchmark", add_special_tokens=False)
    if not candidate_ids:
        raise RuntimeError("tokenizer did not produce a benchmark token")
    base_token_id = candidate_ids[0]
    vocab_size = llm.tokenizer.vocab_size
    sampling = SamplingParams(
        temperature=0.1,
        max_tokens=args.output_length,
        ignore_eos=True,
    )

    # Match measured request shapes so lazy kernel initialization is excluded.
    # A separate unique-token range prevents prefix-cache hits in the real run.
    warmup_prompts = make_prompts(
        base_token_id,
        args.prompt_length,
        args.num_requests,
        vocab_size,
        unique_offset=args.num_requests + 1024,
    )
    llm.generate(
        warmup_prompts,
        SamplingParams(temperature=0.1, max_tokens=2, ignore_eos=True),
        use_tqdm=False,
    )
    torch.cuda.synchronize()
    trials = [
        run_trial(
            llm,
            args,
            sampling,
            base_token_id,
            vocab_size,
            unique_offset=1 + trial_index * args.num_requests,
        )
        for trial_index in range(args.repetitions)
    ]
    result = {
        "config": {
            "model": str(args.model),
            "num_requests": args.num_requests,
            "prompt_length": args.prompt_length,
            "output_length": args.output_length,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "repetitions": args.repetitions,
            "scheduling_policy": args.scheduling_policy,
            "cuda_graph_enabled": args.use_cuda_graph,
            "prefix_cache_enabled": not args.disable_prefix_cache,
            "chunked_prefill_enabled": not args.disable_chunked_prefill,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        },
        "aggregate": aggregate_trials(trials),
        "trials": trials,
    }
    if args.repetitions == 1:
        result.update(trials[0])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
