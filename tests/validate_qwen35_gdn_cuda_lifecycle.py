"""Official-model validation for fused GDN state lifecycle interactions.

This is intentionally an executable validation rather than a default unit
test: it requires the official checkpoint and a CUDA GPU with enough memory.
"""

from __future__ import annotations

import argparse
import json

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import SequenceStatus


def make_prompts(vocab_size: int, shared_prefix_length: int):
    safe_vocab = min(vocab_size, 200_000)
    shared = [
        (index * 37 + 11) % safe_vocab
        for index in range(shared_prefix_length)
    ]
    source = shared + [17]
    target = shared + [29]
    return source, target


def make_engine(args, *, backend: str, hybrid_prefix: bool) -> LLM:
    return LLM(
        args.model,
        max_num_batched_tokens=512,
        max_num_seqs=1,
        max_model_len=args.shared_prefix_length + args.output_tokens + 16,
        gpu_memory_utilization=0.9,
        enforce_eager=False,
        max_num_kvcache_blocks=16,
        max_num_state_slots=1,
        gdn_decode_backend=backend,
        prefix_match_unit=16,
        enable_prefix_cache=hybrid_prefix,
        enable_hybrid_prefix_cache=hybrid_prefix,
        hybrid_prefix_checkpoint_interval_blocks=8,
        hybrid_prefix_checkpoint_interval_tokens=4096,
        hybrid_prefix_checkpoint_memory_bytes=(
            args.checkpoint_memory_mib * 1024 * 1024
            if hybrid_prefix
            else 0
        ),
        hybrid_prefix_retention_policy="adaptive",
        hybrid_prefix_eviction_policy="cost_aware",
        enable_hybrid_internal_checkpoints=hybrid_prefix,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--shared-prefix-length", type=int, default=240)
    parser.add_argument("--output-tokens", type=int, default=4)
    parser.add_argument("--checkpoint-memory-mib", type=int, default=128)
    args = parser.parse_args()
    if args.shared_prefix_length % 16:
        parser.error("shared prefix must align to the 16-token match unit")
    if args.shared_prefix_length >= 256:
        parser.error("this validation requires a partial physical page")

    probe = make_engine(args, backend="torch", hybrid_prefix=False)
    try:
        source, target = make_prompts(
            probe.config.hf_config.vocab_size,
            args.shared_prefix_length,
        )
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=args.output_tokens,
            ignore_eos=True,
        )
        expected = probe.generate([target], sampling, use_tqdm=False)[0][
            "token_ids"
        ]
    finally:
        probe.exit()

    engine = make_engine(args, backend="cuda", hybrid_prefix=True)
    try:
        # Producer publishes the adaptive prompt-tail checkpoint and KV page.
        engine.generate(
            [source],
            SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
            use_tqdm=False,
        )
        engine.reset_runtime_metrics()

        request_id = engine.add_request(target, sampling)
        outputs, _ = engine.step()
        if outputs:
            raise AssertionError("request unexpectedly finished during first prefill")
        sequence = next(
            sequence
            for sequence in engine.scheduler.running
            if sequence.seq_id == request_id
        )
        if sequence.num_completion_tokens != 1:
            raise AssertionError("final prefill must emit exactly one token")
        first_slot = sequence.state_slot
        if first_slot is None:
            raise AssertionError("request has no recurrent state slot")

        # Force the same lifecycle used by memory-pressure preemption. Mutable
        # request KV/state is released, while the immutable prefix snapshot
        # remains available for the next admission.
        engine.scheduler.running.remove(sequence)
        engine.scheduler.preempt(sequence)
        if sequence.status is not SequenceStatus.WAITING:
            raise AssertionError("preempted request did not return to waiting")
        if sequence.state_slot is not None or sequence.block_table:
            raise AssertionError("preemption did not release mutable state")
        if first_slot not in engine.model_runner.state_manager.free_slot_ids:
            raise AssertionError("preempted recurrent slot was not returned")

        while not sequence.is_finished:
            engine.step()

        actual = sequence.completion_token_ids
        metrics = engine.get_runtime_metrics()
        scheduler = metrics["scheduler"]
        runner = metrics["model_runner"]
        if actual != expected:
            raise AssertionError(
                f"preempt-rehit output mismatch: {actual} != {expected}"
            )
        if sequence.preemption_count != 1:
            raise AssertionError("expected exactly one forced preemption")
        if scheduler["hybrid_prefix_restore_count"] < 2:
            raise AssertionError("initial admission and rehit must both restore state")
        if runner["hybrid_prefix_cow_count"] < 2:
            raise AssertionError("initial admission and rehit must both execute COW")
        if runner["decode_cudagraph_replays"] <= 0:
            raise AssertionError("continuation did not replay a CUDA graph")

        print(
            json.dumps(
                {
                    "tokens_match_torch_cold": True,
                    "token_ids": actual,
                    "preemption_count": sequence.preemption_count,
                    "committed_hit_tokens": scheduler[
                        "hybrid_prefix_committed_hit_tokens"
                    ],
                    "state_restores": scheduler[
                        "hybrid_prefix_restore_count"
                    ],
                    "cow_count": runner["hybrid_prefix_cow_count"],
                    "cow_bytes": runner["hybrid_prefix_cow_bytes"],
                    "decode_cudagraph_replays": runner[
                        "decode_cudagraph_replays"
                    ],
                },
                indent=2,
            )
        )
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
