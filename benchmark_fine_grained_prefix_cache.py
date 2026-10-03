"""Reproducible Qwen3.5 Hybrid Prefix Cache policy matrix."""

import argparse
import json
from types import SimpleNamespace

from benchmark_hybrid_prefix_cache import run_case


def _case(base, **overrides):
    values = vars(base).copy()
    values.update(overrides)
    return SimpleNamespace(**values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--shared-prefix-length", type=int, default=496)
    parser.add_argument("--unique-suffix-length", type=int, default=1)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--checkpoint-memory-mib", type=int, default=640)
    parser.add_argument(
        "--checkpoint-dtype",
        choices=("fp32", "bf16", "int8"),
        default="fp32",
    )
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-num-kvcache-blocks", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument(
        "--gdn-decode-backend",
        choices=("torch", "cuda", "auto"),
        default="torch",
    )
    parser.add_argument("--no-share", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.shared_prefix_length % 16:
        parser.error("shared prefix length must align to 16-token fine matching")

    base = SimpleNamespace(
        **vars(args),
        interval_blocks=8,
        interval_tokens=None,
        prefix_match_unit=256,
        retention_policy="periodic",
        eviction_policy="lru",
        internal_checkpoints=False,
        compare_internal=False,
    )
    variants = [
        ("no_cache", False, _case(base)),
        (
            "block_aligned",
            True,
            _case(
                base,
                prefix_match_unit=256,
                interval_tokens=256,
            ),
        ),
        (
            "fine_dense",
            True,
            _case(
                base,
                prefix_match_unit=16,
                interval_tokens=16,
            ),
        ),
        (
            "fine_adaptive",
            True,
            _case(
                base,
                prefix_match_unit=16,
                interval_tokens=4096,
                retention_policy="adaptive",
                eviction_policy="cost_aware",
            ),
        ),
        (
            "fine_internal",
            True,
            _case(
                base,
                prefix_match_unit=16,
                interval_tokens=4096,
                retention_policy="adaptive",
                eviction_policy="cost_aware",
                internal_checkpoints=True,
            ),
        ),
    ]

    results = []
    baseline = None
    for name, enabled, case_args in variants:
        result = run_case(
            case_args,
            enabled,
            internal_checkpoints=case_args.internal_checkpoints,
        )
        result["variant"] = name
        if baseline is None:
            baseline = result
        result["tokens_match_baseline"] = (
            result["token_ids"] == baseline["token_ids"]
        )
        result["ttft_change_percent"] = (
            result["ttft_ms"] / baseline["ttft_ms"] - 1
        ) * 100
        producer_ttft = result["producer_request"]["ttft_ms"]
        baseline_producer_ttft = baseline["producer_request"]["ttft_ms"]
        result["producer_ttft_change_percent"] = (
            producer_ttft / baseline_producer_ttft - 1
        ) * 100
        results.append(result)

    if args.summary_only:
        summary = [
            {
                "variant": result["variant"],
                "gdn_decode_backend": result["gdn_decode_backend"],
                "checkpoint_dtype": result["checkpoint_dtype"],
                "tokens_match_baseline": result["tokens_match_baseline"],
                "ttft_ms": result["ttft_ms"],
                "tpot_ms": result["tpot_ms"],
                "throughput_tok_s": result["throughput_tok_s"],
                "ttft_change_percent": result["ttft_change_percent"],
                "producer_ttft_ms": result["producer_request"]["ttft_ms"],
                "producer_ttft_change_percent": result[
                    "producer_ttft_change_percent"
                ],
                "committed_hit_tokens": result["scheduler"].get(
                    "hybrid_prefix_committed_hit_tokens", 0
                ),
                "checkpoint_bytes": result["model_runner"].get(
                    "hybrid_prefix_checkpoint_bytes", 0
                ),
                "cow_bytes": result["model_runner"].get(
                    "hybrid_prefix_cow_bytes", 0
                ),
                "cow_ms": result["model_runner"].get(
                    "hybrid_prefix_cow_ms", 0.0
                ),
                "producer_prefill_runs": result["producer_model_runner"].get(
                    "prefill_model_runs", 0
                ),
                "producer_schedule_count": result["producer_request"][
                    "schedule_count"
                ],
            }
            for result in results
        ]
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(results, indent=2))
    if any(not result["tokens_match_baseline"] for result in results):
        raise SystemExit("one or more variants differ from the cold baseline")


if __name__ == "__main__":
    main()
