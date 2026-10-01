"""Interleaved end-to-end A/B for the Torch and CUDA GDN decode backends."""

from __future__ import annotations

import argparse
import gc
import json
from statistics import mean, pstdev
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams


def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(",") if item)
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def make_prompts(concurrency: int, prompt_tokens: int, vocab_size: int):
    safe_vocab = min(vocab_size, 200_000)
    return [
        [
            (request_index * 1009 + token_index * 37 + 11) % safe_vocab
            for token_index in range(prompt_tokens)
        ]
        for request_index in range(concurrency)
    ]


def run_once(
    args,
    *,
    backend: str,
    concurrency: int,
    output_tokens: int,
    cuda_graph: bool,
) -> dict:
    engine = LLM(
        args.model,
        max_num_batched_tokens=max(
            args.max_num_batched_tokens,
            concurrency * args.prompt_tokens,
        ),
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.prompt_tokens + output_tokens + 8,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=not cuda_graph,
        max_num_kvcache_blocks=max(args.max_num_kvcache_blocks, concurrency * 2),
        max_num_state_slots=args.max_num_seqs,
        enable_prefix_cache=False,
        gdn_decode_backend=backend,
    )
    prompts = make_prompts(
        concurrency,
        args.prompt_tokens,
        engine.config.hf_config.vocab_size,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=output_tokens,
        ignore_eos=True,
    )
    try:
        if args.warmup:
            engine.generate(
                prompts,
                SamplingParams(
                    temperature=0.0,
                    max_tokens=min(4, output_tokens),
                    ignore_eos=True,
                ),
                use_tqdm=False,
            )
        engine.reset_runtime_metrics()
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        outputs = engine.generate(prompts, sampling, use_tqdm=False)
        elapsed = perf_counter() - started
        request_metrics = engine.get_request_metrics()[-concurrency:]
        total_output_tokens = sum(len(item["token_ids"]) for item in outputs)
        return {
            "backend": backend,
            "concurrency": concurrency,
            "output_tokens_per_request": output_tokens,
            "execution": "graph" if cuda_graph else "eager",
            "elapsed_s": elapsed,
            "throughput_tok_s": total_output_tokens / elapsed,
            "mean_ttft_ms": mean(item["ttft_ms"] for item in request_metrics),
            "mean_tpot_ms": mean(item["tpot_ms"] for item in request_metrics),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "token_ids": [item["token_ids"] for item in outputs],
            "model_runner": engine.get_runtime_metrics()["model_runner"],
        }
    finally:
        engine.exit()
        del engine
        gc.collect()
        torch.cuda.empty_cache()


def summarize(runs: list[dict]) -> dict:
    return {
        "runs": len(runs),
        "throughput_tok_s_mean": mean(run["throughput_tok_s"] for run in runs),
        "throughput_tok_s_std": pstdev(
            run["throughput_tok_s"] for run in runs
        ),
        "mean_ttft_ms": mean(run["mean_ttft_ms"] for run in runs),
        "mean_tpot_ms": mean(run["mean_tpot_ms"] for run in runs),
        "peak_allocated_gib_max": max(run["peak_allocated_gib"] for run in runs),
        "decode_eager_runs": sum(
            run["model_runner"]["decode_eager_runs"] for run in runs
        ),
        "decode_cudagraph_replays": sum(
            run["model_runner"]["decode_cudagraph_replays"] for run in runs
        ),
        "decode_cudagraph_padded_rows": sum(
            run["model_runner"]["decode_cudagraph_padded_rows"] for run in runs
        ),
        "decode_cudagraph_max_padding": max(
            run["model_runner"]["decode_cudagraph_max_padding"] for run in runs
        ),
        "decode_cudagraph_last_real_batch_size": runs[-1]["model_runner"][
            "decode_cudagraph_last_real_batch_size"
        ],
        "decode_cudagraph_last_graph_batch_size": runs[-1]["model_runner"][
            "decode_cudagraph_last_graph_batch_size"
        ],
    }


def token_mismatches(reference: list[list[int]], runs_by_backend: dict) -> list[dict]:
    mismatches = []
    for backend, runs in runs_by_backend.items():
        for run_index, run in enumerate(runs):
            for request_index, (expected, actual) in enumerate(
                zip(reference, run["token_ids"])
            ):
                for token_index, (expected_token, actual_token) in enumerate(
                    zip(expected, actual)
                ):
                    if expected_token != actual_token:
                        mismatches.append(
                            {
                                "backend": backend,
                                "run_index": run_index,
                                "request_index": request_index,
                                "token_index": token_index,
                                "expected": expected_token,
                                "actual": actual_token,
                            }
                        )
                        break
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=parse_int_list, default=(1, 2, 3, 4))
    parser.add_argument("--output-tokens", type=parse_int_list, default=(32,))
    parser.add_argument("--prompt-tokens", type=int, default=16)
    parser.add_argument(
        "--execution",
        choices=("eager", "graph", "both"),
        default="graph",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=2,
        help="Each cycle runs both backends; odd cycles reverse their order.",
    )
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-num-kvcache-blocks", type=int, default=16)
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Engine capacity; defaults to the largest requested concurrency.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.set_defaults(warmup=True)
    args = parser.parse_args()
    if args.cycles <= 0 or args.prompt_tokens <= 0:
        parser.error("cycles and prompt-tokens must be positive")
    if args.max_num_seqs is None:
        args.max_num_seqs = max(args.concurrency)
    if args.max_num_seqs < max(args.concurrency):
        parser.error("max-num-seqs must cover every requested concurrency")

    execution_modes = (
        (False, True)
        if args.execution == "both"
        else (args.execution == "graph",)
    )
    results = []
    for cuda_graph in execution_modes:
        for concurrency in args.concurrency:
            for output_tokens in args.output_tokens:
                runs_by_backend = {"torch": [], "cuda": []}
                reference_tokens = None
                token_exact = True
                execution_order = []
                for cycle in range(args.cycles):
                    order = ("torch", "cuda") if cycle % 2 == 0 else ("cuda", "torch")
                    for backend in order:
                        execution_order.append(backend)
                        run = run_once(
                            args,
                            backend=backend,
                            concurrency=concurrency,
                            output_tokens=output_tokens,
                            cuda_graph=cuda_graph,
                        )
                        if reference_tokens is None:
                            reference_tokens = run["token_ids"]
                        token_exact &= run["token_ids"] == reference_tokens
                        runs_by_backend[backend].append(run)

                torch_summary = summarize(runs_by_backend["torch"])
                cuda_summary = summarize(runs_by_backend["cuda"])
                result = {
                    "execution": "graph" if cuda_graph else "eager",
                    "concurrency": concurrency,
                    "output_tokens_per_request": output_tokens,
                    "execution_order": execution_order,
                    "tokens_exact": token_exact,
                    "torch": torch_summary,
                    "cuda": cuda_summary,
                    "throughput_change_percent": (
                        cuda_summary["throughput_tok_s_mean"]
                        / torch_summary["throughput_tok_s_mean"]
                        - 1
                    )
                    * 100,
                    "tpot_change_percent": (
                        cuda_summary["mean_tpot_ms"]
                        / torch_summary["mean_tpot_ms"]
                        - 1
                    )
                    * 100,
                }
                if not token_exact:
                    result["token_mismatches"] = token_mismatches(
                        reference_tokens,
                        runs_by_backend,
                    )
                    result["torch_tokens_stable"] = all(
                        run["token_ids"] == runs_by_backend["torch"][0]["token_ids"]
                        for run in runs_by_backend["torch"]
                    )
                    result["cuda_tokens_stable"] = all(
                        run["token_ids"] == runs_by_backend["cuda"][0]["token_ids"]
                        for run in runs_by_backend["cuda"]
                    )
                    result["reference_token_ids"] = reference_tokens
                    result["first_cuda_token_ids"] = runs_by_backend["cuda"][0][
                        "token_ids"
                    ]
                results.append(result)
                if not token_exact:
                    print(json.dumps(results, indent=2))
                    raise SystemExit("Torch/CUDA token mismatch")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
