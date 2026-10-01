"""Generate and validate official Qwen3.5 Hugging Face golden fixtures.

This script is intended for both the local 0.8B stepping-stone checkpoint and
the final cloud 9B gate.  HF and nano-vLLM run sequentially so their weights do
not coexist on GPU.

Examples:
    python tests/validate_qwen35_hf_golden.py \
        --model /models/Qwen3.5-9B --fixture /tmp/qwen35-9b-golden.pt \
        --mode hf

    PYTHONSAFEPATH=1 PYTHONPATH=. python \
        tests/validate_qwen35_hf_golden.py \
        --model /models/Qwen3.5-9B --fixture /tmp/qwen35-9b-golden.pt \
        --mode nano --num-speculative-tokens 2 --cuda-graph
"""

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, Qwen3_5ForCausalLM


def generate_hf_fixture(
    model_path: str,
    fixture_path: Path,
    prompt_ids: list[int],
    decode_steps: int,
) -> dict:
    full_config = AutoConfig.from_pretrained(model_path)
    model = Qwen3_5ForCausalLM.from_pretrained(
        model_path,
        config=full_config.text_config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).cuda().eval()
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
    try:
        with torch.inference_mode():
            output = model(
                input_ids=prompt,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            cache = output.past_key_values
            prompt_logits = output.logits[:, -1].float().cpu()
            prompt_hidden = output.hidden_states[-1][:, -1].float().cpu()
            greedy_tokens = []
            decode_logits = None
            decode_hidden = None
            for step in range(decode_steps):
                token = output.logits[:, -1].float().argmax(dim=-1)
                greedy_tokens.append(int(token))
                output = model(
                    input_ids=token[:, None],
                    past_key_values=cache,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                cache = output.past_key_values
                if step == 0:
                    decode_logits = output.logits[:, -1].float().cpu()
                    decode_hidden = output.hidden_states[-1][:, -1].float().cpu()
        fixture = {
            "model_type": full_config.model_type,
            "prompt_ids": torch.tensor([prompt_ids], dtype=torch.long),
            "prompt_last_hidden": prompt_hidden,
            "prompt_last_logits": prompt_logits,
            "decode1_hidden": decode_hidden,
            "decode1_logits": decode_logits,
            "greedy_tokens": torch.tensor(greedy_tokens, dtype=torch.long),
        }
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(fixture, fixture_path)
        return {
            "fixture": str(fixture_path),
            "greedy_tokens": greedy_tokens,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _compare_logits(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual = actual.float().cpu()
    expected = expected.float().cpu()
    delta = (actual - expected).abs()
    return {
        "actual_argmax": int(actual.argmax()),
        "expected_argmax": int(expected.argmax()),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "cosine": float(F.cosine_similarity(actual, expected, dim=0)),
    }


def _make_config(args, *, num_speculative_tokens: int):
    from nanovllm.config import Config

    return Config(
        model=args.model,
        max_num_batched_tokens=max(len(args.prompt_ids), 8),
        max_num_seqs=1,
        max_model_len=max(len(args.prompt_ids) + args.decode_steps + 4, 32),
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=not args.cuda_graph,
        max_num_kvcache_blocks=args.max_num_kvcache_blocks,
        max_num_state_slots=1,
        num_speculative_tokens=num_speculative_tokens,
        speculative_parallel_verify=not args.sequential_verify,
    )


def validate_nano(args, fixture_path: Path) -> dict:
    from nanovllm import SamplingParams
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.engine.sequence import Sequence
    from nanovllm.utils.context import reset_context

    fixture = torch.load(fixture_path, weights_only=True)
    prompt_ids = fixture["prompt_ids"][0].tolist()
    expected_tokens = fixture["greedy_tokens"].tolist()

    target_runner = None
    try:
        target_runner = ModelRunner(_make_config(args, num_speculative_tokens=0), 0, [])
        sequence = Sequence(
            prompt_ids,
            SamplingParams(temperature=0.0, max_tokens=args.decode_steps, ignore_eos=True),
        )
        sequence.block_table = [0]
        sequence.num_scheduled_tokens = len(sequence)
        target_runner.allocate_state_slot(sequence)
        input_ids, positions = target_runner.prepare_prefill([sequence])
        prompt_logits = target_runner.run_model(input_ids, positions, True)[-1]
        reset_context()

        sequence.num_cached_tokens = len(sequence)
        sequence.num_scheduled_tokens = 0
        sequence.append_token(expected_tokens[0])
        sequence.num_scheduled_tokens = 1
        input_ids, positions = target_runner.prepare_decode([sequence])
        decode_logits = target_runner.run_model(input_ids, positions, False)[-1]
        reset_context()

        generated = [int(prompt_logits.argmax()), int(decode_logits.argmax())]
        sequence.num_cached_tokens += 1
        sequence.num_scheduled_tokens = 0
        sequence.append_token(generated[-1])
        while len(generated) < len(expected_tokens):
            sequence.num_scheduled_tokens = 1
            input_ids, positions = target_runner.prepare_decode([sequence])
            logits = target_runner.run_model(input_ids, positions, False)[-1]
            reset_context()
            token = int(logits.argmax())
            generated.append(token)
            sequence.num_cached_tokens += 1
            sequence.num_scheduled_tokens = 0
            sequence.append_token(token)

        report = {
            "prompt_logits": _compare_logits(
                prompt_logits, fixture["prompt_last_logits"][0]
            ),
            "decode1_logits": _compare_logits(
                decode_logits, fixture["decode1_logits"][0]
            ),
            "target_tokens": generated,
            "target_tokens_match": generated == expected_tokens,
            "target_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
    finally:
        if target_runner is not None:
            target_runner.exit()
            target_runner = None
        gc.collect()
        torch.cuda.empty_cache()

    if args.num_speculative_tokens:
        spec_runner = None
        try:
            spec_runner = ModelRunner(
                _make_config(
                    args,
                    num_speculative_tokens=args.num_speculative_tokens,
                ),
                0,
                [],
            )
            sequence = Sequence(
                prompt_ids,
                SamplingParams(
                    temperature=0.0,
                    max_tokens=args.decode_steps,
                    ignore_eos=True,
                ),
            )
            sequence.block_table = [0]
            sequence.num_scheduled_tokens = len(sequence)
            spec_runner.allocate_state_slot(sequence)
            [first] = spec_runner.run([sequence], True)
            speculative_tokens = [first]
            sequence.num_cached_tokens = len(sequence)
            sequence.num_scheduled_tokens = 0
            sequence.append_token(first)
            while len(speculative_tokens) < len(expected_tokens):
                sequence.num_scheduled_tokens = 1
                tokens = spec_runner.run_speculative([sequence])[0]
                speculative_tokens.extend(tokens)
                sequence.num_cached_tokens += len(tokens)
                sequence.num_scheduled_tokens = 0
                for token in tokens:
                    sequence.append_token(token)
                sequence.num_cached_tokens = len(sequence) - 1
            speculative_tokens = speculative_tokens[: len(expected_tokens)]
            report.update(
                {
                    "speculative_tokens": speculative_tokens,
                    "speculative_tokens_match": speculative_tokens
                    == expected_tokens,
                    "speculative_metrics": spec_runner.get_metrics(),
                    "speculative_peak_allocated_gib": (
                        torch.cuda.max_memory_allocated() / 2**30
                    ),
                }
            )
        finally:
            if spec_runner is not None:
                spec_runner.exit()
                spec_runner = None
            gc.collect()
            torch.cuda.empty_cache()

    if report["prompt_logits"]["cosine"] < args.min_cosine:
        raise AssertionError("prefill logits cosine is below the required threshold")
    if report["decode1_logits"]["cosine"] < args.min_cosine:
        raise AssertionError("decode logits cosine is below the required threshold")
    if not report["target_tokens_match"]:
        raise AssertionError("nano target greedy tokens differ from HF")
    if args.num_speculative_tokens and not report["speculative_tokens_match"]:
        raise AssertionError("nano MTP greedy tokens differ from HF target")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--mode", choices=("hf", "nano", "both"), default="both")
    parser.add_argument(
        "--prompt-ids",
        type=int,
        nargs="+",
        default=[9707, 11, 1879, 0],
    )
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--num-speculative-tokens", type=int, default=2)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--sequential-verify", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-kvcache-blocks", type=int, default=2)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    args = parser.parse_args()

    results = {}
    if args.mode in ("hf", "both"):
        results["hf"] = generate_hf_fixture(
            args.model,
            args.fixture,
            args.prompt_ids,
            args.decode_steps,
        )
    if args.mode in ("nano", "both"):
        results["nano"] = validate_nano(args, args.fixture)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
