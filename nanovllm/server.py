"""OpenAI-compatible HTTP entrypoint for nano-vLLM."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask

from nanovllm.async_llm import AsyncLLMEngine, RequestStream, StreamOutput
from nanovllm.sampling_params import SamplingParams


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=64, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    stream: bool = False
    ignore_eos: bool = False


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=64, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    stream: bool = False
    ignore_eos: bool = False


def _sampling_params(request: CompletionRequest | ChatCompletionRequest) -> SamplingParams:
    return SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        ignore_eos=request.ignore_eos,
    )


def _sse(data: dict | str) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n"


def _text_delta(previous: str, current: str) -> str:
    return current[len(previous):] if current.startswith(previous) else current


async def _collect(stream: RequestStream) -> StreamOutput:
    last_output = None
    try:
        async for output in stream:
            last_output = output
    finally:
        await stream.aclose()
    if last_output is None:
        raise RuntimeError("inference completed without producing a token")
    return last_output


def create_app(engine: AsyncLLMEngine, close_engine: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        if close_engine:
            await engine.close()

    app = FastAPI(title="nano-vLLM Server", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.model_name,
                    "object": "model",
                    "owned_by": "nanovllm",
                }
            ],
        }

    @app.get("/metrics")
    async def metrics():
        return engine.engine.get_runtime_metrics()

    @app.get("/metrics/requests")
    async def request_metrics():
        return {"data": engine.engine.get_request_metrics()}

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        request_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        model = request.model or engine.model_name
        try:
            stream = await engine.submit(request.prompt, _sampling_params(request))
        except (AssertionError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if request.stream:
            async def events():
                previous_text = ""
                try:
                    async for output in stream:
                        delta = _text_delta(previous_text, output.text)
                        previous_text = output.text
                        yield _sse({
                            "id": request_id,
                            "object": "text_completion",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "text": delta,
                                "finish_reason": output.finish_reason,
                            }],
                        })
                    yield _sse("[DONE]")
                finally:
                    await stream.aclose()

            return StreamingResponse(
                events(),
                media_type="text/event-stream",
                background=BackgroundTask(stream.aclose),
            )

        try:
            output = await _collect(stream)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        prompt_tokens = len(engine.engine.tokenizer.encode(request.prompt))
        completion_tokens = len(output.token_ids)
        return {
            "id": request_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "text": output.text,
                "finish_reason": output.finish_reason,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        model = request.model or engine.model_name
        messages = [message.model_dump() for message in request.messages]
        try:
            prompt = engine.engine.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            stream = await engine.submit(prompt, _sampling_params(request))
        except (AssertionError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if request.stream:
            async def events():
                previous_text = ""
                first_chunk = True
                try:
                    async for output in stream:
                        delta = _text_delta(previous_text, output.text)
                        previous_text = output.text
                        message_delta = {"content": delta}
                        if first_chunk:
                            message_delta["role"] = "assistant"
                            first_chunk = False
                        yield _sse({
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": message_delta,
                                "finish_reason": output.finish_reason,
                            }],
                        })
                    yield _sse("[DONE]")
                finally:
                    await stream.aclose()

            return StreamingResponse(
                events(),
                media_type="text/event-stream",
                background=BackgroundTask(stream.aclose),
            )

        try:
            output = await _collect(stream)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        prompt_tokens = len(engine.engine.tokenizer.encode(prompt))
        completion_tokens = len(output.token_ids)
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": output.text},
                "finish_reason": output.finish_reason,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve nano-vLLM over an OpenAI-compatible API")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument(
        "--scheduling-policy",
        choices=("prefill_first", "decode_first", "slo_aware"),
        default="decode_first",
    )
    parser.add_argument(
        "--waiting-admission-policy",
        choices=("fcfs", "hybrid_state_aware"),
        default="fcfs",
    )
    parser.add_argument(
        "--preemption-policy",
        choices=("lifo", "recompute_aware"),
        default="lifo",
    )
    parser.add_argument(
        "--hybrid-scheduler-candidate-window", type=int, default=8
    )
    parser.add_argument(
        "--hybrid-scheduler-aging-tokens-per-ms", type=float, default=0.5
    )
    parser.add_argument(
        "--hybrid-scheduler-max-wait-ms", type=float, default=200.0
    )
    parser.add_argument(
        "--hybrid-scheduler-min-saved-tokens", type=int, default=16
    )
    parser.add_argument(
        "--hybrid-scheduler-preemption-penalty", type=float, default=128.0
    )
    parser.add_argument(
        "--hybrid-scheduler-score-source",
        choices=("joint", "kv_only"),
        default="joint",
    )
    parser.add_argument("--disable-hybrid-scheduler-aging", action="store_true")
    parser.add_argument(
        "--disable-hybrid-scheduler-hysteresis", action="store_true"
    )
    parser.add_argument(
        "--disable-hybrid-scheduler-sticky-recovery", action="store_true"
    )
    parser.add_argument("--enable-scheduler-profiling", action="store_true")
    parser.add_argument("--scheduler-decision-history-size", type=int, default=4096)
    parser.add_argument("--target-ttft-ms", type=float, default=200.0)
    parser.add_argument("--target-tpot-ms", type=float, default=50.0)
    parser.add_argument("--slo-min-prefill-tokens", type=int, default=64)
    parser.add_argument("--slo-kv-pressure-threshold", type=float, default=0.9)
    parser.add_argument("--slo-queue-pressure-threshold", type=int, default=3)
    parser.add_argument("--slo-latency-safety-margin-ms", type=float, default=5.0)
    parser.add_argument("--stream-queue-size", type=int, default=16)
    parser.add_argument("--num-speculative-tokens", type=int, default=0)
    parser.add_argument("--enable-mtp-phase-profiling", action="store_true")
    parser.add_argument(
        "--gdn-decode-backend",
        choices=("torch", "cuda", "auto"),
        default="torch",
        help="Qwen3.5 one-token GDN decode backend; CUDA remains opt-in.",
    )
    parser.add_argument("--enable-hybrid-prefix-cache", action="store_true")
    parser.add_argument(
        "--prefix-match-unit",
        type=int,
        default=None,
        help=(
            "Prefix hash granularity in tokens. M1 requires it to equal "
            "the physical KV block size."
        ),
    )
    parser.add_argument(
        "--hybrid-prefix-checkpoint-interval-blocks",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--hybrid-prefix-checkpoint-interval-tokens",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--hybrid-prefix-checkpoint-memory-mib",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--hybrid-prefix-checkpoint-dtype",
        choices=("fp32", "bf16", "int8"),
        default="fp32",
    )
    parser.add_argument(
        "--hybrid-prefix-promotion-min-sightings",
        type=int,
        default=2,
        help=(
            "Total prefix sightings required before a KV-only shared "
            "junction is promoted; must be at least 2."
        ),
    )
    parser.add_argument(
        "--hybrid-prefix-retention-policy",
        choices=("periodic", "adaptive"),
        default="periodic",
    )
    parser.add_argument(
        "--hybrid-prefix-eviction-policy",
        choices=("lru", "cost_aware"),
        default="lru",
    )
    parser.add_argument(
        "--enable-hybrid-internal-checkpoints",
        action="store_true",
    )
    parser.add_argument(
        "--sequential-speculative-verify",
        action="store_true",
        help="Use the slow stepwise verifier as a correctness oracle.",
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-prefix-cache", action="store_true")
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    engine = AsyncLLMEngine(
        args.model,
        stream_queue_size=args.stream_queue_size,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        scheduling_policy=args.scheduling_policy,
        waiting_admission_policy=args.waiting_admission_policy,
        preemption_policy=args.preemption_policy,
        hybrid_scheduler_candidate_window=(
            args.hybrid_scheduler_candidate_window
        ),
        hybrid_scheduler_aging_tokens_per_ms=(
            args.hybrid_scheduler_aging_tokens_per_ms
        ),
        hybrid_scheduler_max_wait_ms=(
            args.hybrid_scheduler_max_wait_ms
        ),
        hybrid_scheduler_min_saved_tokens=(
            args.hybrid_scheduler_min_saved_tokens
        ),
        hybrid_scheduler_preemption_penalty=(
            args.hybrid_scheduler_preemption_penalty
        ),
        hybrid_scheduler_score_source=args.hybrid_scheduler_score_source,
        hybrid_scheduler_enable_aging=(
            not args.disable_hybrid_scheduler_aging
        ),
        hybrid_scheduler_enable_hysteresis=(
            not args.disable_hybrid_scheduler_hysteresis
        ),
        hybrid_scheduler_enable_sticky_recovery=(
            not args.disable_hybrid_scheduler_sticky_recovery
        ),
        enable_scheduler_profiling=args.enable_scheduler_profiling,
        scheduler_decision_history_size=args.scheduler_decision_history_size,
        scheduler_target_ttft_ms=args.target_ttft_ms,
        scheduler_target_tpot_ms=args.target_tpot_ms,
        slo_min_prefill_tokens=args.slo_min_prefill_tokens,
        slo_kv_pressure_threshold=args.slo_kv_pressure_threshold,
        slo_queue_pressure_threshold=args.slo_queue_pressure_threshold,
        slo_latency_safety_margin_ms=args.slo_latency_safety_margin_ms,
        enforce_eager=args.enforce_eager,
        enable_prefix_cache=not args.disable_prefix_cache,
        enable_chunked_prefill=not args.disable_chunked_prefill,
        num_speculative_tokens=args.num_speculative_tokens,
        enable_mtp_phase_profiling=args.enable_mtp_phase_profiling,
        gdn_decode_backend=args.gdn_decode_backend,
        speculative_parallel_verify=not args.sequential_speculative_verify,
        enable_hybrid_prefix_cache=args.enable_hybrid_prefix_cache,
        prefix_match_unit=args.prefix_match_unit,
        hybrid_prefix_checkpoint_interval_blocks=(
            args.hybrid_prefix_checkpoint_interval_blocks
        ),
        hybrid_prefix_checkpoint_interval_tokens=(
            args.hybrid_prefix_checkpoint_interval_tokens
        ),
        hybrid_prefix_checkpoint_memory_bytes=(
            args.hybrid_prefix_checkpoint_memory_mib * 1024 * 1024
        ),
        hybrid_prefix_checkpoint_dtype=(
            args.hybrid_prefix_checkpoint_dtype
        ),
        hybrid_prefix_promotion_min_sightings=(
            args.hybrid_prefix_promotion_min_sightings
        ),
        hybrid_prefix_retention_policy=(
            args.hybrid_prefix_retention_policy
        ),
        hybrid_prefix_eviction_policy=args.hybrid_prefix_eviction_policy,
        enable_hybrid_internal_checkpoints=(
            args.enable_hybrid_internal_checkpoints
        ),
    )
    uvicorn.run(create_app(engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
