"""Trace how a newly arrived prefill request affects an active decode request."""

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
        "--output", type=Path, default=Path("benchmark_results/interference.json")
    )
    parser.add_argument("--decode-prompt-length", type=int, default=64)
    parser.add_argument("--decode-output-length", type=int, default=32)
    parser.add_argument("--inject-after-tokens", type=int, default=4)
    parser.add_argument("--prefill-prompt-length", type=int, default=768)
    parser.add_argument("--prefill-output-length", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument(
        "--scheduling-policy",
        choices=("prefill_first", "decode_first"),
        default="prefill_first",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.inject_after_tokens < args.decode_output_length:
        raise ValueError(
            "--inject-after-tokens must be positive and less than "
            "--decode-output-length"
        )
    for prompt_length, output_length in (
        (args.decode_prompt_length, args.decode_output_length),
        (args.prefill_prompt_length, args.prefill_output_length),
    ):
        if prompt_length <= 0 or output_length <= 0:
            raise ValueError("prompt and output lengths must be positive")
        if prompt_length + output_length > args.max_model_len:
            raise ValueError("a request exceeds --max-model-len")


def add_request(llm: LLM, prompt: list[int], output_length: int):
    llm.add_request(
        prompt,
        SamplingParams(temperature=0.1, max_tokens=output_length, ignore_eos=True),
    )
    return llm.scheduler.waiting[-1]


def token_gaps_ms(token_times: list[float]) -> list[float]:
    return [
        (current - previous) * 1000
        for previous, current in zip(token_times, token_times[1:])
    ]


def main() -> None:
    args = parse_args()
    validate_args(args)

    llm = LLM(
        str(args.model),
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=8,
        scheduling_policy=args.scheduling_policy,
    )
    token_ids = llm.tokenizer.encode(" benchmark", add_special_tokens=False)
    if not token_ids:
        raise RuntimeError("tokenizer did not produce a benchmark token")
    base_token_id = token_ids[0]
    vocab_size = llm.tokenizer.vocab_size

    # Warm both prompt shapes with a separate token range to exclude lazy setup.
    warmup_decode = make_prompt(
        base_token_id, (base_token_id + 1001) % vocab_size, args.decode_prompt_length
    )
    warmup_prefill = make_prompt(
        base_token_id, (base_token_id + 1002) % vocab_size, args.prefill_prompt_length
    )
    warmup_decode_pair = [
        make_prompt(
            base_token_id,
            (base_token_id + unique_offset) % vocab_size,
            args.decode_prompt_length,
        )
        for unique_offset in (1003, 1004)
    ]
    warmup_sampling = SamplingParams(
        temperature=0.1, max_tokens=2, ignore_eos=True
    )
    # Run separately: a joint warmup would only initialize batch-size=2 decode.
    llm.generate([warmup_decode], warmup_sampling, use_tqdm=False)
    llm.generate(warmup_decode_pair, warmup_sampling, use_tqdm=False)
    llm.generate([warmup_prefill], warmup_sampling, use_tqdm=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    decode_prompt = make_prompt(
        base_token_id, (base_token_id + 1) % vocab_size, args.decode_prompt_length
    )
    prefill_prompt = make_prompt(
        base_token_id, (base_token_id + 2) % vocab_size, args.prefill_prompt_length
    )

    started_at = perf_counter()
    decode_sequence = add_request(llm, decode_prompt, args.decode_output_length)
    sequences = {decode_sequence.seq_id: decode_sequence}
    token_times: dict[int, list[float]] = {decode_sequence.seq_id: []}
    observed_tokens = {decode_sequence.seq_id: 0}
    trace = []
    injected_at = None
    injection_trace_index = None
    injected_sequence = None
    step_index = 0

    while not llm.is_finished():
        waiting_before = [sequence.seq_id for sequence in llm.scheduler.waiting]
        running_before = [sequence.seq_id for sequence in llm.scheduler.running]
        step_started = perf_counter()
        _, stats = llm.step()
        torch.cuda.synchronize()
        step_finished = perf_counter()

        generated = []
        for request_id, sequence in sequences.items():
            new_tokens = sequence.num_completion_tokens - observed_tokens[request_id]
            if new_tokens:
                token_times[request_id].extend([step_finished] * new_tokens)
                observed_tokens[request_id] = sequence.num_completion_tokens
                generated.append({"request_id": request_id, "tokens": new_tokens})

        if stats.prefill_tokens and stats.decode_tokens:
            phase = "mixed"
        elif stats.prefill_tokens:
            phase = "prefill"
        else:
            phase = "decode"
        trace.append(
            {
                "step": step_index,
                "phase": phase,
                "prefill_tokens": stats.prefill_tokens,
                "decode_tokens": stats.decode_tokens,
                "duration_ms": (step_finished - step_started) * 1000,
                "waiting_before": waiting_before,
                "running_before": running_before,
                "generated": generated,
            }
        )
        step_index += 1

        should_inject = (
            injected_sequence is None
            and decode_sequence.num_completion_tokens >= args.inject_after_tokens
        )
        if should_inject:
            injected_at = perf_counter()
            injection_trace_index = len(trace)
            injected_sequence = add_request(
                llm, prefill_prompt, args.prefill_output_length
            )
            sequences[injected_sequence.seq_id] = injected_sequence
            token_times[injected_sequence.seq_id] = []
            observed_tokens[injected_sequence.seq_id] = 0

    finished_at = perf_counter()
    expected_lengths = {
        decode_sequence.seq_id: args.decode_output_length,
        injected_sequence.seq_id: args.prefill_output_length,
    }
    actual_lengths = {
        request_id: sequence.num_completion_tokens
        for request_id, sequence in sequences.items()
    }
    if actual_lengths != expected_lengths:
        raise RuntimeError(
            f"unexpected completion lengths: {actual_lengths} != {expected_lengths}"
        )
    active_kv_blocks = len(llm.scheduler.block_manager.used_block_ids)
    if active_kv_blocks:
        raise RuntimeError(f"KV block leak: {active_kv_blocks} blocks remain active")

    decode_token_times = token_times[decode_sequence.seq_id]
    decode_gaps = token_gaps_ms(decode_token_times)
    last_token_before_injection = max(
        timestamp for timestamp in decode_token_times if timestamp <= injected_at
    )
    first_token_after_injection = min(
        timestamp for timestamp in decode_token_times if timestamp > injected_at
    )
    interference_gap_ms = (
        first_token_after_injection - last_token_before_injection
    ) * 1000
    result = {
        "config": vars(args) | {"model": str(args.model), "output": str(args.output)},
        "summary": {
            "duration_s": finished_at - started_at,
            "decode_request_id": decode_sequence.seq_id,
            "injected_request_id": injected_sequence.seq_id,
            "injected_at_ms": (injected_at - started_at) * 1000,
            "decode_token_gaps_ms": summarize(decode_gaps),
            "decode_max_token_gap_ms": max(decode_gaps),
            "interference_gap_ms": interference_gap_ms,
            "prefill_steps_after_injection": sum(
                event["phase"] in ("prefill", "mixed")
                for event in trace[injection_trace_index:]
            ),
            "completion_lengths": actual_lengths,
            "active_kv_blocks_after_run": active_kv_blocks,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        },
        "trace": trace,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
