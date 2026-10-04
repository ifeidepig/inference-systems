"""Replay dynamic request arrivals against nano-vLLM."""

import argparse
import hashlib
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


def generate_scheduler_profile(profile: str) -> list[RequestSpec]:
    """Fixed traces used by the scheduler validation matrix."""
    if profile == "shared_prefix":
        groups = [None, 0, 0, 0, 0, 0]
        lengths = [560] * len(groups)
        outputs = [4] * len(groups)
    elif profile == "multi_session":
        groups = [None, 0, 1, None, 0, 1, 0, 1]
        lengths = [560] * len(groups)
        outputs = [4] * len(groups)
    elif profile == "unique_prompt":
        groups = [None] * 8
        lengths = [560] * len(groups)
        outputs = [4] * len(groups)
    elif profile == "kv_pressure":
        groups = [None, 0, None, 0, None, 0, None, 0]
        lengths = [256] * len(groups)
        outputs = [8] * len(groups)
    elif profile == "kv_pressure_victim_choice":
        # LIFO sees a cold newest victim; cost-aware preemption can instead
        # choose a request with a durable 240-token joint checkpoint.
        groups = [0, None, 0, None, 0, None, 0, None]
        lengths = [256] * len(groups)
        outputs = [8] * len(groups)
    elif profile == "multi_turn":
        groups = [None, 0, None, 0, None, 0]
        lengths = [832, 320, 832, 576, 832, 832]
        outputs = [4] * len(groups)
    else:
        raise ValueError(f"unknown scheduler workload profile: {profile}")
    return [
        RequestSpec(
            request_index=index,
            arrival_offset_ms=0.0,
            prompt_length=lengths[index],
            output_length=outputs[index],
            shared_prefix_group=groups[index],
        )
        for index in range(len(groups))
    ]


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


def output_digest(token_ids_by_request: list[list[int]]) -> str:
    payload = json.dumps(token_ids_by_request, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    e2e_values = [request["e2e_ms"] for request in requests]
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
        "request_latency_ms": summarize(e2e_values),
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
    parser.add_argument(
        "--prefix-seed-requests",
        type=int,
        default=0,
        help=(
            "Warm each shared-prefix group with this many requests before "
            "metrics reset; use 2 for second-sighting hybrid promotion."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--workload-profile",
        choices=(
            "custom",
            "shared_prefix",
            "multi_session",
            "unique_prompt",
            "kv_pressure",
            "kv_pressure_victim_choice",
            "multi_turn",
        ),
        default="custom",
    )
    parser.add_argument("--run-label", default="standalone")
    parser.add_argument("--workload-name", default="custom")
    parser.add_argument("--repeat-index", type=int, default=0)
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
    parser.add_argument(
        "--waiting-admission-policy",
        choices=("fcfs", "hybrid_state_aware"),
        default="fcfs",
    )
    parser.add_argument(
        "--preemption-policy",
        choices=("lifo", "recompute_aware"),
        default="lifo",
    )
    parser.add_argument("--hybrid-scheduler-candidate-window", type=int, default=8)
    parser.add_argument(
        "--hybrid-scheduler-aging-tokens-per-ms", type=float, default=0.5
    )
    parser.add_argument("--hybrid-scheduler-max-wait-ms", type=float, default=200.0)
    parser.add_argument("--hybrid-scheduler-min-saved-tokens", type=int, default=16)
    parser.add_argument(
        "--hybrid-scheduler-preemption-penalty", type=float, default=128.0
    )
    parser.add_argument(
        "--hybrid-scheduler-score-source",
        choices=("joint", "kv_only"),
        default="joint",
    )
    parser.add_argument("--disable-hybrid-scheduler-aging", action="store_true")
    parser.add_argument(
        "--disable-hybrid-scheduler-hysteresis", action="store_true"
    )
    parser.add_argument(
        "--disable-hybrid-scheduler-sticky-recovery", action="store_true"
    )
    parser.add_argument("--enable-scheduler-profiling", action="store_true")
    parser.add_argument("--scheduler-decision-history-size", type=int, default=4096)
    parser.add_argument("--slo-min-prefill-tokens", type=int, default=64)
    parser.add_argument("--slo-kv-pressure-threshold", type=float, default=0.9)
    parser.add_argument("--slo-queue-pressure-threshold", type=int, default=3)
    parser.add_argument("--slo-latency-safety-margin-ms", type=float, default=5.0)
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument("--max-num-kvcache-blocks", type=int, default=None)
    parser.add_argument("--prefix-match-unit", type=int, default=256)
    parser.add_argument("--disable-prefix-cache", action="store_true")
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--enable-hybrid-prefix-cache", action="store_true")
    parser.add_argument("--hybrid-prefix-checkpoint-memory-mib", type=int, default=128)
    parser.add_argument(
        "--hybrid-prefix-checkpoint-dtype",
        choices=("fp32", "bf16", "int8"),
        default="bf16",
    )
    parser.add_argument(
        "--hybrid-prefix-checkpoint-interval-tokens", type=int, default=256
    )
    parser.add_argument(
        "--hybrid-prefix-retention-policy",
        choices=("periodic", "adaptive"),
        default="adaptive",
    )
    parser.add_argument(
        "--hybrid-prefix-eviction-policy",
        choices=("lru", "cost_aware"),
        default="cost_aware",
    )
    parser.add_argument(
        "--gdn-decode-backend",
        choices=("torch", "cuda", "auto"),
        default="auto",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.target_ttft_ms <= 0 or args.target_tpot_ms <= 0:
        raise ValueError("SLO targets must be positive")
    if args.temperature < 0:
        raise ValueError("temperature cannot be negative")
    if args.shared_prefix_length < 0:
        raise ValueError("shared prefix length cannot be negative")
    if args.prefix_seed_requests < 0:
        raise ValueError("prefix seed request count cannot be negative")
    if min(args.output_lengths) < 2:
        raise ValueError("online SLO measurement requires at least two output tokens")
    if (
        args.workload_profile == "custom"
        and max(args.prompt_lengths) + max(args.output_lengths)
        > args.max_model_len
    ):
        raise ValueError("a configured request shape exceeds --max-model-len")
    if 256 % args.prefix_match_unit:
        raise ValueError("prefix match unit must divide the 256-token KV page")
    if args.hybrid_scheduler_candidate_window <= 0:
        raise ValueError("candidate window must be positive")
    if args.enable_hybrid_prefix_cache and args.disable_prefix_cache:
        raise ValueError("hybrid prefix cache requires prefix caching")
    if (
        args.enable_hybrid_prefix_cache
        and args.hybrid_prefix_checkpoint_memory_mib <= 0
    ):
        raise ValueError("hybrid checkpoint memory must be positive")


def main() -> None:
    args = parse_args()
    validate_args(args)
    workload = (
        generate_workload(
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
        if args.workload_profile == "custom"
        else generate_scheduler_profile(args.workload_profile)
    )
    if max(
        spec.prompt_length + spec.output_length for spec in workload
    ) > args.max_model_len:
        raise ValueError("scheduler profile exceeds --max-model-len")
    llm = LLM(
        str(args.model),
        enforce_eager=not args.use_cuda_graph,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        max_num_state_slots=args.max_num_seqs,
        max_num_kvcache_blocks=args.max_num_kvcache_blocks,
        scheduling_policy=args.scheduling_policy,
        waiting_admission_policy=args.waiting_admission_policy,
        preemption_policy=args.preemption_policy,
        hybrid_scheduler_candidate_window=(
            args.hybrid_scheduler_candidate_window
        ),
        hybrid_scheduler_aging_tokens_per_ms=(
            args.hybrid_scheduler_aging_tokens_per_ms
        ),
        hybrid_scheduler_max_wait_ms=args.hybrid_scheduler_max_wait_ms,
        hybrid_scheduler_min_saved_tokens=(
            args.hybrid_scheduler_min_saved_tokens
        ),
        hybrid_scheduler_preemption_penalty=(
            args.hybrid_scheduler_preemption_penalty
        ),
        hybrid_scheduler_score_source=args.hybrid_scheduler_score_source,
        hybrid_scheduler_enable_aging=(
            not args.disable_hybrid_scheduler_aging
        ),
        hybrid_scheduler_enable_hysteresis=(
            not args.disable_hybrid_scheduler_hysteresis
        ),
        hybrid_scheduler_enable_sticky_recovery=(
            not args.disable_hybrid_scheduler_sticky_recovery
        ),
        enable_scheduler_profiling=args.enable_scheduler_profiling,
        scheduler_decision_history_size=args.scheduler_decision_history_size,
        scheduler_target_ttft_ms=args.target_ttft_ms,
        scheduler_target_tpot_ms=args.target_tpot_ms,
        slo_min_prefill_tokens=args.slo_min_prefill_tokens,
        slo_kv_pressure_threshold=args.slo_kv_pressure_threshold,
        slo_queue_pressure_threshold=args.slo_queue_pressure_threshold,
        slo_latency_safety_margin_ms=args.slo_latency_safety_margin_ms,
        enable_prefix_cache=not args.disable_prefix_cache,
        enable_chunked_prefill=not args.disable_chunked_prefill,
        prefix_match_unit=args.prefix_match_unit,
        enable_hybrid_prefix_cache=args.enable_hybrid_prefix_cache,
        hybrid_prefix_checkpoint_interval_tokens=(
            args.hybrid_prefix_checkpoint_interval_tokens
        ),
        hybrid_prefix_checkpoint_memory_bytes=(
            args.hybrid_prefix_checkpoint_memory_mib * 1024 * 1024
            if args.enable_hybrid_prefix_cache
            else 0
        ),
        hybrid_prefix_checkpoint_dtype=(
            args.hybrid_prefix_checkpoint_dtype
        ),
        hybrid_prefix_retention_policy=(
            args.hybrid_prefix_retention_policy
        ),
        hybrid_prefix_eviction_policy=args.hybrid_prefix_eviction_policy,
        gdn_decode_backend=args.gdn_decode_backend,
        request_metrics_history_size=max(args.num_requests, 1),
    )
    candidate_ids = llm.tokenizer.encode(" benchmark", add_special_tokens=False)
    if not candidate_ids:
        raise RuntimeError("tokenizer did not produce a benchmark token")
    base_token_id = candidate_ids[0]
    vocab_size = llm.tokenizer.vocab_size
    workload_base_token_id = (
        base_token_id + args.seed * 7919
    ) % vocab_size

    workload_prompt_lengths = sorted(
        {spec.prompt_length for spec in workload}
    )
    for warmup_index, prompt_length in enumerate(workload_prompt_lengths):
        warmup = make_prompt(
            workload_base_token_id,
            (workload_base_token_id + 1000 + warmup_index) % vocab_size,
            prompt_length,
        )
        llm.generate(
            [warmup],
            SamplingParams(
                temperature=args.temperature,
                max_tokens=2,
                ignore_eos=True,
            ),
            use_tqdm=False,
        )
    if args.prefix_seed_requests:
        seed_prompt_length = max(workload_prompt_lengths)
        shared_groups = sorted(
            {
                spec.shared_prefix_group
                for spec in workload
                if spec.shared_prefix_group is not None
            }
        )
        for group in shared_groups:
            for seed_index in range(args.prefix_seed_requests):
                seed_spec = RequestSpec(
                    request_index=(
                        args.num_requests + group * args.prefix_seed_requests
                        + seed_index
                    ),
                    arrival_offset_ms=0.0,
                    prompt_length=seed_prompt_length,
                    output_length=2,
                    shared_prefix_group=group,
                )
                llm.generate(
                    [
                        build_prompt(
                            seed_spec,
                            workload_base_token_id,
                            vocab_size,
                            args.shared_prefix_length,
                        )
                    ],
                    SamplingParams(
                        temperature=args.temperature,
                        max_tokens=2,
                        ignore_eos=True,
                    ),
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
    finished_outputs = {}
    peak_waiting = peak_running = peak_kv_blocks = 0

    while next_request < len(workload) or not llm.is_finished():
        now_ns = perf_counter_ns()
        while next_request < len(workload):
            spec = workload[next_request]
            arrival_ns = benchmark_started_ns + int(spec.arrival_offset_ms * 1e6)
            if arrival_ns > now_ns:
                break
            prompt = build_prompt(
                spec,
                workload_base_token_id,
                vocab_size,
                args.shared_prefix_length,
            )
            request_id = llm.add_request(
                prompt,
                SamplingParams(
                    temperature=args.temperature,
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
            outputs, stats = llm.step()
            finished_outputs.update(dict(outputs))
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
    total_input_tokens = sum(request["prompt_tokens"] for request in requests)
    ordered_output_tokens = [
        finished_outputs[request_id]
        for request_id, _ in sorted(
            specs_by_request_id.items(),
            key=lambda item: item[1].request_index,
        )
    ]
    runtime_metrics = llm.get_runtime_metrics()
    scheduler_metrics = runtime_metrics["scheduler"]
    scheduler_cpu_ms = (
        scheduler_metrics.get("scheduler_admission_decision_ms", 0.0)
        + scheduler_metrics.get("scheduler_preemption_selection_ms", 0.0)
        + scheduler_metrics.get("scheduler_metrics_bookkeeping_ms", 0.0)
    )
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
            "total_makespan_ms": duration_s * 1000,
            "request_throughput_rps": len(requests) / duration_s,
            "input_throughput_tokens_per_s": total_input_tokens / duration_s,
            "output_throughput_tokens_per_s": total_output_tokens / duration_s,
            "scheduler_cpu_ms": scheduler_cpu_ms,
            "scheduler_cpu_fraction": scheduler_cpu_ms / (duration_s * 1000),
            "peak_waiting_requests": peak_waiting,
            "peak_running_requests": peak_running,
            "peak_kv_blocks_used": peak_kv_blocks,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        },
        "breakdown": breakdown,
        "runtime_metrics": runtime_metrics,
        "scheduler_decision_events": (
            llm.scheduler.get_scheduler_decision_events()
        ),
        "workload": [asdict(spec) for spec in workload],
        "requests": requests,
        "output_token_ids": ordered_output_tokens,
        "output_digest": output_digest(ordered_output_tokens),
        "trace": trace,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(json.dumps(result["runtime_metrics"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
