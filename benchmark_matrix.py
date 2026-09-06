"""Run a repeatable scheduling-policy comparison matrix."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


POLICIES = ("prefill_first", "decode_first")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark_results/matrix")
    )
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[256, 512, 768])
    parser.add_argument("--token-budgets", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--repetitions", type=int, default=5)
    return parser.parse_args()


def build_command(
    model: Path,
    output: Path,
    policy: str,
    prompt_length: int,
    token_budget: int,
    repetitions: int,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("benchmark_interference.py")),
        "--model",
        str(model),
        "--output",
        str(output),
        "--scheduling-policy",
        policy,
        "--prefill-prompt-length",
        str(prompt_length),
        "--max-num-batched-tokens",
        str(token_budget),
        "--repetitions",
        str(repetitions),
    ]


def run_case(
    args: argparse.Namespace,
    policy: str,
    prompt_length: int,
    token_budget: int,
) -> dict:
    output = args.output_dir / (
        f"prompt-{prompt_length}_budget-{token_budget}_{policy}.json"
    )
    command = build_command(
        args.model,
        output,
        policy,
        prompt_length,
        token_budget,
        args.repetitions,
    )
    print(
        f"running policy={policy} prompt={prompt_length} budget={token_budget}",
        flush=True,
    )
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            f"benchmark failed: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for prompt_length in args.prompt_lengths:
        for token_budget in args.token_budgets:
            cases = {
                policy: run_case(args, policy, prompt_length, token_budget)
                for policy in POLICIES
            }
            prefill = cases["prefill_first"]["aggregate"]
            decode = cases["decode_first"]["aggregate"]
            prefill_gap = prefill["interference_gap_ms"]["mean"]
            decode_gap = decode["interference_gap_ms"]["mean"]
            prefill_duration = prefill["duration_ms"]["mean"]
            decode_duration = decode["duration_ms"]["mean"]
            rows.append(
                {
                    "prompt_length": prompt_length,
                    "token_budget": token_budget,
                    "repetitions": args.repetitions,
                    "prefill_first_gap_ms": prefill_gap,
                    "decode_first_gap_ms": decode_gap,
                    "gap_reduction_percent": (1 - decode_gap / prefill_gap) * 100,
                    "prefill_first_duration_ms": prefill_duration,
                    "decode_first_duration_ms": decode_duration,
                    "duration_change_percent": (
                        decode_duration / prefill_duration - 1
                    ) * 100,
                }
            )

    json_output = args.output_dir / "summary.json"
    csv_output = args.output_dir / "summary.csv"
    json_output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with csv_output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(rows, indent=2))
    print(f"wrote {json_output}")
    print(f"wrote {csv_output}")


if __name__ == "__main__":
    main()
