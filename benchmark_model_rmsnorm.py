import argparse
import json
import os
from statistics import median
from time import perf_counter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.path.expanduser("~/huggingface/Qwen3-0.6B"),
    )
    parser.add_argument("--custom", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()

    os.environ["NANOVLLM_USE_CUSTOM_RMSNORM"] = "1" if args.custom else "0"

    import torch
    from nanovllm import LLM, SamplingParams

    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=1,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=max(args.batch_size, 8),
        gpu_memory_utilization=0.75,
    )

    try:
        warmup_params = SamplingParams(
            temperature=1e-5,
            max_tokens=4,
            ignore_eos=True,
        )
        torch.manual_seed(100)
        llm.generate(["CUDA warmup request"], warmup_params, use_tqdm=False)

        durations_ms = []
        token_counts = []
        sampling_params = SamplingParams(
            temperature=1e-5,
            max_tokens=args.max_tokens,
            ignore_eos=True,
        )
        for trial in range(args.trials):
            prompts = [
                f"Trial {trial}, request {request}: explain one GPU concept."
                for request in range(args.batch_size)
            ]
            torch.manual_seed(1000 + trial)
            start = perf_counter()
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            durations_ms.append((perf_counter() - start) * 1000.0)
            token_counts.append(sum(len(item["token_ids"]) for item in outputs))

        metrics = llm.get_runtime_metrics()
        result = {
            "custom_rmsnorm": args.custom,
            "cuda_graph": not args.enforce_eager,
            "trials": args.trials,
            "batch_size": args.batch_size,
            "max_tokens": args.max_tokens,
            "durations_ms": durations_ms,
            "median_ms": median(durations_ms),
            "generated_tokens": token_counts,
            "model_runner": metrics["model_runner"],
        }
        print(json.dumps(result, indent=2))
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
