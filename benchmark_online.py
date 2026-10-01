"""Replay dynamic request arrivals against nano-vLLM."""

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter_ns

import torch

from benchmark_serving import make_prompt, summarize
from nanovllm import LLM, SamplingParams


@dataclass(frozen=True)
class RequestSpec:
    request_index: int
    arrival_offset_ms: float
    prompt_length: int
    output_length: int
    shared_prefix_group: int | None


def generate_arrival_offsets(
    num_requests: int,
    request_rate: float,
    pattern: str,
    seed: int,
    burst_size: int,
) -> list[float]:
    if num_requests <= 0:
        raise ValueError("num_requests must be positive")
    if request_rate <= 0:
        raise ValueError("request_rate must be positive")
    if burst_size <= 0:
        raise ValueError("burst_size must be positive")

    if pattern == "bursty":
        burst_interval_ms = burst_size / request_rate * 1000
        return [
            (request_index // burst_size) * burst_interval_ms
            for request_index in range(num_requests)
        ]
    if pattern != "poisson":
        raise ValueError(f"unknown arrival pattern: {pattern}")

    rng = random.Random(seed)
    offsets = [0.0]
    for _ in range(1, num_requests):
        offsets.append(offsets[-1] + rng.expovariate(request_rate) * 1000)
    return offsets


def generate_workload(
    num_requests: int,
    request_rate: float,
    arrival_pattern: str,
    prompt_lengths: list[int],
    output_lengths: list[int],
    shared_prefix_ratio: float,
    shared_prefix_groups: int,
    seed: int,
    burst_size: int,
) -> list[RequestSpec]:
    if not prompt_lengths or min(prompt_lengths) <= 0:
        raise ValueError("prompt lengths must be positive")
    if not output_lengths or min(output_lengths) <= 0:
        raise ValueError("output lengths must be positive")
    if not 0 <= shared_prefix_ratio <= 1:
        raise ValueError("shared_prefix_ratio must be between 0 and 1")
    if shared_prefix_groups <= 0:
        raise ValueError("shared_prefix_groups must be positive")

    arrivals = generate_arrival_offsets(
        num_requests, request_rate, arrival_pattern, seed, burst_size
    )
    rng = random.Random(seed + 1)
    workload = []
    for request_index, arrival_offset_ms in enumerate(arrivals):
        uses_shared_prefix = rng.random() < shared_prefix_ratio
        workload.append(
            RequestSpec(
                request_index=request_index,
                arrival_offset_ms=arrival_offset_ms,
                prompt_length=rng.choice(prompt_lengths),
                output_length=rng.choice(output_lengths),
                shared_prefix_group=(
                    request_index % shared_prefix_groups
                    if uses_shared_prefix
                    else None
                ),
            )
        )
    return workload


def build_prompt(
    spec: RequestSpec,
    base_token_id: int,
    vocab_size: int,
    shared_prefix_length: int,
) -> list[int]:
    unique_token_id = (base_token_id + 2000 + spec.request_index) % vocab_size
    if spec.shared_prefix_group is None:
        return make_prompt(base_token_id, unique_token_id, spec.prompt_length)

    prefix_token_id = (
        base_token_id + 100 + spec.shared_prefix_group
    ) % vocab_size
    prefix_length = min(shared_prefix_length, spec.prompt_length - 1)
    return [prefix_token_id] * prefix_length + [unique_token_id] * (
        spec.prompt_length - prefix_length
    )


def prompt_class(prompt_tokens: int) -> str:
    if prompt_tokens <= 128:
        return "short"
    if prompt_tokens <= 512:
        return "medium"
    return "long"


def summarize_requests(
    requests: list[dict], target_ttft_ms: float, target_tpot_ms: float
) -> dict:
    ttft_values = [request["ttft_ms"] for request in requests]
    tpot_values = [
        request["tpot_ms"]
        for request in requests
        if request["tpot_ms"] is not None
    ]
    max_gap_values = [
        request["max_token_gap_ms"]
        for request in requests
        if request["max_token_gap_ms"] is not None
    ]
    queue_values = [request["queue_ms"] for request in requests]
    admission_values = [request["admission_delay_ms"] for request in requests]
    ttft_violations = sum(value > target_ttft_ms for value in ttft_values)
    tpot_violations = sum(value > target_tpot_ms for value in max_gap_values)
    request_violations = sum(
        request["ttft_ms"] > target_ttft_ms
        or (
            request["max_token_gap_ms"] is not None
            and request["max_token_gap_ms"] > target_tpot_ms
        )
        for request in requests
    )
    return {
        "ttft_ms": summarize(ttft_values),
        "tpot_ms": summarize(tpot_values),
        "max_token_gap_ms": summarize(max_gap_values),
        "queue_ms": summarize(queue_values),
        "admission_delay_ms": summarize(admission_values),
        "ttft_slo_violation_rate": ttft_violations / len(requests),
        "tpot_slo_violation_rate": tpot_violations / len(max_gap_values),
        "request_slo_violation_rate": request_violations / len(requests),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/online.json"),
    )
    parser.add_argument("--num-requests", type=int, default=24)
    parser.add_argument("--request-rate", type=float, default=8.0)
    parser.add_argument(
        "--arrival-pattern", choices=("poisson", "bursty"), default="poisson"
    )
    parser.add_argument("--burst-size", type=int, default=4)
    parser.add_argument(
        "--prompt-lengths", type=int, nargs="+", default=[64, 256, 768]
    )
    parser.add_argument("--output-lengths", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--shared-prefix-ratio", type=float, default=0.0)
    parser.add_argument("--shared-prefix-groups", type=int, default=2)
    parser.add_argument("--shared-prefix-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-ttft-ms", type=float, default=200.0)
    parser.add_argument("--target-tpot-ms", type=float, default=50.0)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument(
        "--scheduling-policy",
        choices=("prefill_first", "decode_first", "slo_aware"),
        default="decode_first",
    )
    parser.add_argument("--slo-min-prefill-tokens", type=int, default=64)
    parser.add_argument("--slo-kv-pressure-threshold", type=float, default=0.9)
    parser.add_argument("--slo-queue-pressure-threshold", type=int, default=3)
    parser.add_argument("--slo-latency-safety-margin-ms", type=float, default=5.0)
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument("--disable-prefix-cache", action="store_true")
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.target_ttft_ms <= 0 or args.target_tpot_ms <= 0:
        raise ValueError("SLO targets must be positive")
    if args.shared_prefix_length < 0:
        raise ValueError("shared prefix length cannot be negative")
    if min(args.output_lengths) < 2:
        raise ValueError("online SLO measurement requires at least two output tokens")
    if max(args.prompt_lengths) + max(args.output_lengths) > args.max_model_len:
        raise ValueError("a configured request shape exceeds --max-model-len")


def main() -> None:
    args = parse_args()
    validate_args(args)
    workload = generate_workload(
        args.num_requests,
        args.request_rate,
        args.arrival_pattern,
        args.prompt_lengths,
        args.output_lengths,
        args.shared_prefix_ratio,
        args.shared_prefix_groups,
        args.seed,
        args.burst_size,
    )
    llm = LLM(
        str(args.model),
        enforce_eager=not args.use_cuda_graph,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        scheduling_policy=args.scheduling_policy,
        scheduler_target_ttft_ms=args.target_ttft_ms,
        scheduler_target_tpot_ms=args.target_tpot_ms,
        slo_min_prefill_tokens=args.slo_min_prefill_tokens,
        slo_kv_pressure_threshold=args.slo_kv_pressure_threshold,
        slo_queue_pressure_threshold=args.slo_queue_pressure_threshold,
        slo_latency_safety_margin_ms=args.slo_latency_safety_margin_ms,
        enable_prefix_cache=not args.disable_prefix_cache,
        enable_chunked_prefill=not args.disable_chunked_prefill,
        request_metrics_history_size=max(args.num_requests, 1),
    )
    candidate_ids = llm.tokenizer.encode(" benchmark", add_special_tokens=False)
    if not candidate_ids:
        raise RuntimeError("tokenizer did not produce a benchmark token")
    base_token_id = candidate_ids[0]
    vocab_size = llm.tokenizer.vocab_size

    for warmup_index, prompt_length in enumerate(sorted(set(args.prompt_lengths))):
        warmup = make_prompt(
            base_token_id,
            (base_token_id + 1000 + warmup_index) % vocab_size,
            prompt_length,
        )
        llm.generate(
            [warmup],
            SamplingParams(temperature=0.1, max_tokens=2, ignore_eos=True),
            use_tqdm=False,
        )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    llm.reset_runtime_metrics()

    benchmark_started_ns = perf_counter_ns()
    next_request = 0
    sequences = {}
    specs_by_request_id = {}
    trace = []
    peak_waiting = peak_running = peak_kv_blocks = 0

    while next_request < len(workload) or not llm.is_finished():
        now_ns = perf_counter_ns()
        while next_request < len(workload):
            spec = workload[next_request]
            arrival_ns = benchmark_started_ns + int(spec.arrival_offset_ms * 1e6)
            if arrival_ns > now_ns:
                break
            prompt = build_prompt(
                spec, base_token_id, vocab_size, args.shared_prefix_length
            )
            request_id = llm.add_request(
                prompt,
                SamplingParams(
                    temperature=0.1,
                    max_tokens=spec.output_length,
                    ignore_eos=True,
                ),
                arrival_time_ns=arrival_ns,
            )
            sequence = llm.scheduler.waiting[-1]
            sequences[request_id] = sequence
            specs_by_request_id[request_id] = spec
            next_request += 1
            now_ns = perf_counter_ns()

        peak_waiting = max(peak_waiting, len(llm.scheduler.waiting))
        peak_running = max(peak_running, len(llm.scheduler.running))
        peak_kv_blocks = max(
            peak_kv_blocks, len(llm.scheduler.block_manager.used_block_ids)
        )

        if not llm.is_finished():
            step_started_ns = perf_counter_ns()
            _, stats = llm.step()
            torch.cuda.synchronize()
            step_finished_ns = perf_counter_ns()
            trace.append(
                {
                    "time_ms": (step_finished_ns - benchmark_started_ns) / 1e6,
                    "duration_ms": (step_finished_ns - step_started_ns) / 1e6,
                    "prefill_tokens": stats.prefill_tokens,
                    "decode_tokens": stats.decode_tokens,
                    "waiting": len(llm.scheduler.waiting),
                    "running": len(llm.scheduler.running),
                    "kv_blocks_used": len(
                        llm.scheduler.block_manager.used_block_ids
                    ),
                    "slo_decision": (
                        dict(llm.scheduler.last_slo_decision)
                        if llm.scheduler.last_slo_decision is not None
                        else None
                    ),
                }
            )
            peak_waiting = max(peak_waiting, len(llm.scheduler.waiting))
            peak_running = max(peak_running, len(llm.scheduler.running))
            peak_kv_blocks = max(
                peak_kv_blocks, len(llm.scheduler.block_manager.used_block_ids)
            )
            continue

        next_arrival_ns = benchmark_started_ns + int(
            workload[next_request].arrival_offset_ms * 1e6
        )
        sleep_seconds = max((next_arrival_ns - perf_counter_ns()) / 1e9, 0)
        time.sleep(min(sleep_seconds, 0.01))

    benchmark_finished_ns = perf_counter_ns()
    requests = []
    for request_id, sequence in sequences.items():
        metrics = sequence.lifecycle_metrics(benchmark_finished_ns)
        spec = specs_by_request_id[request_id]
        metrics.update(
            {
                "request_index": spec.request_index,
                "arrival_offset_ms": spec.arrival_offset_ms,
                "shared_prefix_group": spec.shared_prefix_group,
                "prompt_class": prompt_class(spec.prompt_length),
            }
        )
        requests.append(metrics)
    requests.sort(key=lambda request: request["request_index"])

    breakdown = {}
    for class_name in ("short", "medium", "long"):
        selected = [
            request for request in requests if request["prompt_class"] == class_name
        ]
        if selected:
            breakdown[class_name] = summarize_requests(
                selected, args.target_ttft_ms, args.target_tpot_ms
            )

    duration_s = (benchmark_finished_ns - benchmark_started_ns) / 1e9
    total_output_tokens = sum(request["output_tokens"] for request in requests)
    result = {
        "config": vars(args) | {
            "model": str(args.model),
            "output": str(args.output),
            "cuda_graph_enabled": args.use_cuda_graph,
            "prefix_cache_enabled": not args.disable_prefix_cache,
            "chunked_prefill_enabled": not args.disable_chunked_prefill,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        },
        "summary": summarize_requests(
            requests, args.target_ttft_ms, args.target_tpot_ms
        ) | {
            "duration_s": duration_s,
            "request_throughput_rps": len(requests) / duration_s,
            "output_throughput_tokens_per_s": total_output_tokens / duration_s,
            "peak_waiting_requests": peak_waiting,
            "peak_running_requests": peak_running,
            "peak_kv_blocks_used": peak_kv_blocks,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        },
        "breakdown": breakdown,
        "runtime_metrics": llm.get_runtime_metrics(),
        "workload": [asdict(spec) for spec in workload],
        "requests": requests,
        "trace": trace,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(json.dumps(result["runtime_metrics"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
