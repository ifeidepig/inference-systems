"""Plot the summary produced by benchmark_matrix.py."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, default=Path("matrix.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = json.loads(args.summary.read_text(encoding="utf-8"))
    prompt_lengths = sorted({row["prompt_length"] for row in rows})

    figure, (gap_axis, duration_axis) = plt.subplots(2, 1, figsize=(8, 8))
    for prompt_length in prompt_lengths:
        selected = sorted(
            (row for row in rows if row["prompt_length"] == prompt_length),
            key=lambda row: row["token_budget"],
        )
        budgets = [row["token_budget"] for row in selected]
        label = f"prompt={prompt_length}"
        gap_axis.plot(
            budgets,
            [row["gap_reduction_percent"] for row in selected],
            marker="o",
            label=label,
        )
        duration_axis.plot(
            budgets,
            [row["duration_change_percent"] for row in selected],
            marker="o",
            label=label,
        )

    gap_axis.axhline(0, color="black", linewidth=0.8)
    gap_axis.set_ylabel("Decode gap reduction (%)")
    gap_axis.set_title("Decode-first scheduling benefit")
    gap_axis.grid(alpha=0.3)
    gap_axis.legend()

    duration_axis.axhline(0, color="black", linewidth=0.8)
    duration_axis.set_xlabel("Max batched-token budget")
    duration_axis.set_ylabel("Total duration change (%)")
    duration_axis.set_title("End-to-end cost")
    duration_axis.grid(alpha=0.3)
    duration_axis.legend()

    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
