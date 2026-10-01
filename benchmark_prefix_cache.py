"""Measure cold and warm request latency for repeated prompt prefixes."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from benchmark_serving import make_prompt, summarize
from nanovllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/prefix-cache.json"),
    )
    parser.add_argument("--prompt-length", type=int, default=768)
    parser.add_argument("--output-length", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--max-num-kvcache-blocks", type=int)
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument("--disable-prefix-cache", action="store_true")
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.repetitions < 2:
        raise ValueError("--repetitions must be at least 2")
    if args.prompt_length <= 0 or args.output_length <= 0:
        raise ValueError("prompt and output lengths must be positive")
    if args.prompt_length + args.output_length > args.max_model_len:
        raise ValueError("request exceeds --max-model-len")

def run_request(llm: LLM, prompt: list[int], sampling: SamplingParams) -> dict:
    submitted_at = perf_counter()
    request_id = llm.add_request(prompt, sampling)
    sequence = llm.scheduler.waiting[-1]
    first_token_at = None

    while not sequence.is_finished:
        llm.step()
        torch.cuda.synchronize()
        if sequence.num_completion_tokens and first_token_at is None:
            first_token_at = perf_counter()

    finished_at = perf_counter()
    if first_token_at is None:
        raise RuntimeError("request finished without producing a token")
    return {
        "request_id": request_id,
        "ttft_ms": (first_token_at - submitted_at) * 1000,
        "e2e_ms": (finished_at - submitted_at) * 1000,
        "output_tokens": sequence.num_completion_tokens,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    llm = LLM(
        str(args.model),
        enforce_eager=not args.use_cuda_graph,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_kvcache_blocks=args.max_num_kvcache_blocks,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=1,
        scheduling_policy="prefill_first",
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

    warmup_prompt = make_prompt(
        base_token_id,
        (base_token_id + 1001) % vocab_size,
        args.prompt_length,
    )
    llm.generate(
        [warmup_prompt],
        SamplingParams(temperature=0.1, max_tokens=2, ignore_eos=True),
        use_tqdm=False,
    )
    torch.cuda.synchronize()

    prompt = make_prompt(
        base_token_id,
        (base_token_id + 2001) % vocab_size,
        args.prompt_length,
    )
    llm.reset_runtime_metrics()
    requests = [
        run_request(llm, prompt, sampling) for _ in range(args.repetitions)
    ]
    cold = requests[0]
    warm = requests[1:]
    result = {
        "config": vars(args) | {
            "model": str(args.model),
            "output": str(args.output),
            "prefix_cache_enabled": not args.disable_prefix_cache,
            "chunked_prefill_enabled": not args.disable_chunked_prefill,
            "cuda_graph_enabled": args.use_cuda_graph,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        },
        "summary": {
            "cold_ttft_ms": cold["ttft_ms"],
            "warm_ttft_ms": summarize([request["ttft_ms"] for request in warm]),
            "warm_e2e_ms": summarize([request["e2e_ms"] for request in warm]),
            "warm_ttft_reduction_percent": (
                (cold["ttft_ms"] - sum(request["ttft_ms"] for request in warm) / len(warm))
                / cold["ttft_ms"]
                * 100
            ),
        },
        "runtime_metrics": llm.get_runtime_metrics(),
        "requests": requests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(json.dumps(result["runtime_metrics"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
