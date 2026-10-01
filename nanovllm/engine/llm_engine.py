import atexit
import gc
from dataclasses import dataclass, fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
import torch

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


@dataclass(frozen=True)
class StepStats:
    prefill_tokens: int = 0
    decode_tokens: int = 0


class LLMEngine:

    def __init__(self, model, **kwargs):
        self._exited = False
        config_fields = {field.name for field in fields(Config) if field.init}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        self.ack_events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            ack_event = ctx.Event()
            # Permit the first command; subsequent commands wait until the
            # worker has copied the previous shared-memory payload.
            ack_event.set()
            process = ctx.Process(
                target=ModelRunner,
                args=(config, i, event, ack_event),
            )
            process.start()
            self.ps.append(process)
            self.events.append(event)
            self.ack_events.append(ack_event)
        self.model_runner = ModelRunner(
            config,
            0,
            self.events,
            self.ack_events,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(
            config,
            state_manager=self.model_runner.state_manager,
            state_controller=self.model_runner,
            prefix_checkpoint_pool=self.model_runner.prefix_checkpoint_pool,
        )
        atexit.register(self.exit)

    def exit(self):
        if self._exited:
            return
        self._exited = True
        atexit.unregister(self.exit)
        # Scheduler callbacks otherwise retain the ModelRunner (and therefore
        # all model/KV tensors) even after deleting ``self.model_runner``.
        if hasattr(self, "scheduler"):
            self.scheduler.state_controller = None
            self.scheduler.state_manager = None
        model_runner = self.model_runner
        model_runner.call("exit")
        del self.model_runner
        del model_runner
        for p in self.ps:
            p.join()
        # torch.compile/module cycles can defer tensor destruction even after
        # the runner itself is unreachable. Explicit engine shutdown is a
        # resource boundary, so collect those cycles and release cached blocks.
        gc.collect()
        torch.cuda.empty_cache()

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        arrival_time_ns: int | None = None,
    ):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if self.config.num_speculative_tokens and sampling_params.temperature > 1e-10:
            raise ValueError("native MTP speculative decoding currently supports greedy sampling only")
        if len(prompt) > self.config.max_model_len:
            raise ValueError("prompt exceeds max_model_len")
        seq = Sequence(prompt, sampling_params, arrival_time_ns=arrival_time_ns)
        self.scheduler.add(seq)
        return seq.seq_id

    def abort_request(self, request_id: int) -> bool:
        return self.scheduler.abort(request_id)

    def reset_runtime_metrics(self) -> None:
        self.scheduler.reset_metrics()
        self.model_runner.call("reset_metrics")

    def get_runtime_metrics(self) -> dict:
        return {
            "scheduler": self.scheduler.get_metrics(),
            "model_runner": self.model_runner.call("get_metrics"),
        }

    def get_request_metrics(self) -> list[dict]:
        return self.scheduler.get_request_metrics()

    def step(self):
        batches = self.scheduler.schedule()
        outputs = []
        prefill_tokens = 0
        decode_tokens = 0
        for batch in batches:
            batch_tokens = batch.num_tokens
            batch_started = perf_counter()
            use_speculative = (
                self.config.num_speculative_tokens > 0
                and not batch.is_prefill
            )
            if use_speculative:
                token_ids = self.model_runner.call(
                    "run_speculative", batch.seqs
                )
            else:
                internal_boundaries = self.scheduler.internal_prefix_boundaries(
                    batch.seqs,
                    batch.is_prefill,
                )
                token_ids = self.model_runner.call(
                    "run",
                    batch.seqs,
                    batch.is_prefill,
                    internal_boundaries,
                )
            self.scheduler.observe_batch(
                batch.is_prefill,
                batch_tokens,
                (perf_counter() - batch_started) * 1000,
            )
            pending_captures, blocks_hashed = (
                self.scheduler.prepare_prefix_captures(
                    batch.seqs, batch.is_prefill
                )
            )
            try:
                for pending in pending_captures:
                    if not self.scheduler.reserve_prefix_capture(pending):
                        continue
                    checkpoint_slot = None
                    try:
                        checkpoint_slot = self.model_runner.call(
                            "capture_prefix_checkpoint",
                            pending.state_slot,
                            (
                                pending.boundary_tokens
                                if pending.internal_state
                                else None
                            ),
                        )
                        self.scheduler.publish_prefix_capture(
                            pending, checkpoint_slot
                        )
                    except Exception:
                        if checkpoint_slot is not None:
                            self.model_runner.call(
                                "evict_prefix_checkpoint", checkpoint_slot
                            )
                        raise
            finally:
                if batch.is_prefill:
                    for seq in batch.seqs:
                        if seq.state_slot is not None:
                            self.model_runner.call(
                                "discard_internal_prefix_states",
                                seq.state_slot,
                            )
            if use_speculative:
                self.scheduler.postprocess_speculative(batch.seqs, token_ids)
            else:
                self.scheduler.postprocess(
                    batch.seqs,
                    token_ids,
                    batch.is_prefill,
                    blocks_already_hashed=blocks_hashed,
                )
            outputs.extend(
                (seq.seq_id, seq.completion_token_ids)
                for seq in batch.seqs
                if seq.is_finished
            )
            if batch.is_prefill:
                prefill_tokens += batch_tokens
            else:
                decode_tokens += (
                    sum(len(tokens) for tokens in token_ids)
                    if use_speculative
                    else batch_tokens
                )
        return outputs, StepStats(prefill_tokens, decode_tokens)

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, stats = self.step()
            elapsed = perf_counter() - t
            if stats.prefill_tokens:
                prefill_throughput = stats.prefill_tokens / elapsed
            if stats.decode_tokens:
                decode_throughput = stats.decode_tokens / elapsed
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
