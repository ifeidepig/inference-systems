"""Final batch/concurrency sweep for compact ReplaySSM.

Compares the default minimal full-state snapshot path with the default-off
compact replay path.  Each point uses fresh engines and alternates execution
order by repetition.  The script prints JSON; callers decide where to archive
the raw result.
"""

import argparse
import hashlib
import json
import statistics
from types import SimpleNamespace

from benchmark_qwen35_mtp import run_once


def prompts_for_batch(batch_size: int) -> list[list[int]]:
    return [
        [9707, 11, 1879, (17 + index * 13) % 256]
        for index in range(batch_size)
    ]


def digest_tokens(token_ids: list[list[int]]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    return {
        "median": statistics.median(ordered),
        "q1": ordered[len(ordered) // 4],
        "q3": ordered[(3 * len(ordered)) // 4],
        "min": ordered[0],
        "max": ordered[-1],
    }


def compact_result(run: dict) -> dict:
    return {
        "throughput_tok_s": run["throughput_tok_s"],
        "mean_tpot_ms": run["mean_tpot_ms"],
        "mean_ttft_ms": run["mean_ttft_ms"],
        "peak_allocated_gib": run["peak_allocated_gib"],
        "elapsed_s": run["elapsed_s"],
        "token_digest": digest_tokens(run["token_ids"]),
    }


def summarize(runs: list[dict]) -> dict:
    return {
        "throughput_tok_s": distribution(
            [run["throughput_tok_s"] for run in runs]
        ),
        "mean_tpot_ms": distribution(
            [run["mean_tpot_ms"] for run in runs]
        ),
        "mean_ttft_ms": distribution(
            [run["mean_ttft_ms"] for run in runs]
        ),
        "peak_allocated_gib": distribution(
            [run["peak_allocated_gib"] for run in runs]
        ),
        "tokens_stable": len({run["token_digest"] for run in runs}) == 1,
    }


def percent_change(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=(1, 2, 4, 8)
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--num-speculative-tokens", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--max-num-kvcache-blocks", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument(
        "--gdn-decode-backend",
        choices=("torch", "cuda", "auto"),
        default="cuda",
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--omit-raw", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    if any(size not in (1, 2, 4, 8) for size in args.batch_sizes):
        parser.error("CUDA Graph sweep supports batch sizes 1/2/4/8")

    common = dict(
        model=args.model,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        max_num_kvcache_blocks=args.max_num_kvcache_blocks,
        gdn_decode_backend=args.gdn_decode_backend,
        parallel_verify=True,
        phase_profile=False,
        warmup=True,
        output_tokens=args.output_tokens,
    )
    points = []
    raw = []
    for batch_size in args.batch_sizes:
        prompts = prompts_for_batch(batch_size)
        grouped = {False: [], True: []}
        paired_exact = True
        for repetition in range(args.repetitions):
            order = (False, True) if repetition % 2 == 0 else (True, False)
            pair = {}
            for replay_ssm in order:
                run_args = SimpleNamespace(
                    **common,
                    replay_ssm=replay_ssm,
                )
                result = compact_result(
                    run_once(
                        run_args,
                        args.num_speculative_tokens,
                        prompts=prompts,
                    )
                )
                grouped[replay_ssm].append(result)
                pair[replay_ssm] = result
                raw.append(
                    {
                        "batch_size": batch_size,
                        "repetition": repetition,
                        "replay_ssm": replay_ssm,
                        **result,
                    }
                )
            paired_exact &= (
                pair[False]["token_digest"] == pair[True]["token_digest"]
            )

        minimal = summarize(grouped[False])
        replay = summarize(grouped[True])
        points.append(
            {
                "batch_size": batch_size,
                "minimal_snapshot": minimal,
                "compact_replay": replay,
                "paired_tokens_exact": paired_exact,
                "change_percent": {
                    "throughput": percent_change(
                        replay["throughput_tok_s"]["median"],
                        minimal["throughput_tok_s"]["median"],
                    ),
                    "tpot": percent_change(
                        replay["mean_tpot_ms"]["median"],
                        minimal["mean_tpot_ms"]["median"],
                    ),
                    "ttft": percent_change(
                        replay["mean_ttft_ms"]["median"],
                        minimal["mean_ttft_ms"]["median"],
                    ),
                    "peak_allocated": percent_change(
                        replay["peak_allocated_gib"]["median"],
                        minimal["peak_allocated_gib"]["median"],
                    ),
                },
            }
        )

    print(
        json.dumps(
            {
                "config": {
                    **vars(args),
                    "cuda_graph": not args.enforce_eager,
                    "order": "alternating",
                },
                "points": points,
                "raw": None if args.omit_raw else raw,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
