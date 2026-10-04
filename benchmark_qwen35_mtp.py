"""Reproducible Qwen3.5 target-vs-native-MTP microbenchmark.

Example:
    PYTHONSAFEPATH=1 PYTHONPATH=. python benchmark_qwen35_mtp.py \
        --model /path/to/Qwen3.5-0.8B-Base --mode both

The benchmark is deliberately small and reports negative results as-is.  It
measures the complete nano-vLLM engine path, including scheduling and state
transactions, rather than timing an isolated model kernel.
"""

import argparse
import gc
import json
from statistics import mean, median, pstdev
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams


DEFAULT_PROMPTS = [
    [9707, 11, 1879, 0],
    [151643, 8948, 198, 17, 18],
]


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    return {
        "median": median(ordered),
        "p95": ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)],
        "q1": ordered[len(ordered) // 4],
        "q3": ordered[(3 * len(ordered)) // 4],
    }


def summarize_phase_profiles(runs: list[dict], owner: str) -> dict:
    profile_key = (
        "mtp_phase_profile"
        if owner == "model_runner"
        else "mtp_scheduler_phase_profile"
    )
    profiles = [run[owner][profile_key] for run in runs]
    if not profiles or not profiles[0]["enabled"]:
        return {"enabled": False}
    phase_names = sorted(
        {
            name
            for profile in profiles
            for name in profile["phases"]
        }
    )
    return {
        "enabled": True,
        "phases": {
            name: {
                "cpu_median_ms_across_runs": distribution(
                    [
                        profile["phases"][name]["cpu_ms"]["median"]
                        for profile in profiles
                        if name in profile["phases"]
                    ]
                ),
                "cpu_p95_ms_across_runs": distribution(
                    [
                        profile["phases"][name]["cpu_ms"]["p95"]
                        for profile in profiles
                        if name in profile["phases"]
                    ]
                ),
                "gpu_median_ms_across_runs": distribution(
                    [
                        profile["phases"][name]["gpu_ms"]["median"]
                        for profile in profiles
                        if name in profile["phases"]
                    ]
                ),
                "gpu_p95_ms_across_runs": distribution(
                    [
                        profile["phases"][name]["gpu_ms"]["p95"]
                        for profile in profiles
                        if name in profile["phases"]
                    ]
                ),
            }
            for name in phase_names
        },
        "counters": {
            name: distribution(
                [
                    profile["counters"].get(name, 0.0)
                    for profile in profiles
                ]
            )
            for name in sorted(
                {
                    name
                    for profile in profiles
                    for name in profile["counters"]
                }
            )
        },
    }


def run_once(args, num_speculative_tokens: int) -> dict:
    engine = LLM(
        args.model,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=len(DEFAULT_PROMPTS),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        max_num_kvcache_blocks=args.max_num_kvcache_blocks,
        max_num_state_slots=len(DEFAULT_PROMPTS),
        num_speculative_tokens=num_speculative_tokens,
        speculative_parallel_verify=args.parallel_verify,
        gdn_decode_backend=args.gdn_decode_backend,
        enable_mtp_phase_profiling=args.phase_profile,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.output_tokens,
        ignore_eos=True,
    )
    try:
        if args.warmup:
            engine.generate(
                DEFAULT_PROMPTS,
                SamplingParams(
                    temperature=0.0,
                    max_tokens=min(4, args.output_tokens),
                    ignore_eos=True,
                ),
                use_tqdm=False,
            )
        engine.reset_runtime_metrics()
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        torch.cuda.nvtx.range_push("benchmark.measured")
        try:
            outputs = engine.generate(
                DEFAULT_PROMPTS,
                sampling,
                use_tqdm=False,
            )
        finally:
            torch.cuda.nvtx.range_pop()
        elapsed = perf_counter() - started
        request_metrics = engine.get_request_metrics()[-len(DEFAULT_PROMPTS) :]
        tpot_values = [
            item["tpot_ms"]
            for item in request_metrics
            if item["tpot_ms"] is not None
        ]
        output_count = sum(len(item["token_ids"]) for item in outputs)
        runtime_metrics = engine.get_runtime_metrics()
        return {
            "num_speculative_tokens": num_speculative_tokens,
            "gdn_decode_backend": args.gdn_decode_backend,
            "elapsed_s": elapsed,
            "output_tokens": output_count,
            "throughput_tok_s": output_count / elapsed,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "mean_ttft_ms": mean(item["ttft_ms"] for item in request_metrics),
            "mean_tpot_ms": mean(tpot_values) if tpot_values else None,
            "model_runner": runtime_metrics["model_runner"],
            "scheduler": runtime_metrics["scheduler"],
            "token_ids": [item["token_ids"] for item in outputs],
        }
    finally:
        engine.exit()
        del engine
        gc.collect()
        torch.cuda.empty_cache()


def summarize_runs(runs: list[dict]) -> dict:
    if len(runs) == 1:
        return runs[0]
    token_ids = runs[0]["token_ids"]
    tpot_values = [
        run["mean_tpot_ms"]
        for run in runs
        if run["mean_tpot_ms"] is not None
    ]
    return {
        "num_speculative_tokens": runs[0]["num_speculative_tokens"],
        "gdn_decode_backend": runs[0]["gdn_decode_backend"],
        "repetitions": len(runs),
        "throughput_tok_s_mean": mean(
            run["throughput_tok_s"] for run in runs
        ),
        "throughput_tok_s_std": pstdev(
            run["throughput_tok_s"] for run in runs
        ),
        "throughput_tok_s_distribution": distribution(
            [run["throughput_tok_s"] for run in runs]
        ),
        "mean_ttft_ms": mean(run["mean_ttft_ms"] for run in runs),
        "mean_tpot_ms": mean(tpot_values) if tpot_values else None,
        "peak_allocated_gib_max": max(
            run["peak_allocated_gib"] for run in runs
        ),
        "tokens_stable_across_repetitions": all(
            run["token_ids"] == token_ids for run in runs
        ),
        "model_runner_phase_summary": summarize_phase_profiles(
            runs, "model_runner"
        ),
        "scheduler_phase_summary": summarize_phase_profiles(
            runs, "scheduler"
        ),
        "token_ids": token_ids,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--mode", choices=("baseline", "mtp", "both"), default="both"
    )
    parser.add_argument("--num-speculative-tokens", type=int, default=2)
    parser.add_argument("--phase-profile", action="store_true")
    parser.add_argument(
        "--gdn-decode-backend",
        choices=("torch", "cuda", "auto"),
        default="torch",
    )
    parser.add_argument(
        "--sequential-verify",
        dest="parallel_verify",
        action="store_false",
        help="Use the slow stepwise verifier as a correctness oracle.",
    )
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=64)
    parser.add_argument("--max-num-kvcache-blocks", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--cuda-graph", dest="enforce_eager", action="store_false"
    )
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.set_defaults(
        warmup=True,
        enforce_eager=True,
        parallel_verify=True,
    )
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")

    results = []
    if args.mode in ("baseline", "both"):
        results.append(
            summarize_runs(
                [run_once(args, 0) for _ in range(args.repetitions)]
            )
        )
    if args.mode in ("mtp", "both"):
        results.append(
            summarize_runs(
                [
                    run_once(args, args.num_speculative_tokens)
                    for _ in range(args.repetitions)
                ]
            )
        )
    if len(results) == 2:
        results[1]["tokens_match_baseline"] = (
            results[1]["token_ids"] == results[0]["token_ids"]
        )
        baseline_throughput = results[0].get(
            "throughput_tok_s_mean", results[0].get("throughput_tok_s")
        )
        mtp_throughput = results[1].get(
            "throughput_tok_s_mean", results[1].get("throughput_tok_s")
        )
        results[1]["throughput_change_percent"] = (
            mtp_throughput
            / baseline_throughput
            - 1
        ) * 100
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
