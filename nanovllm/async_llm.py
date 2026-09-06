"""Asynchronous request submission and token streaming for nano-vLLM."""

import asyncio
from dataclasses import dataclass

from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams


@dataclass(frozen=True)
class StreamOutput:
    request_id: int
    token_id: int
    token_ids: tuple[int, ...]
    text: str
    finished: bool


class RequestStream:
    """Async iterator over tokens generated for one request."""

    def __init__(self, request_id: int, queue: asyncio.Queue):
        self.request_id = request_id
        self._queue = queue

    def __aiter__(self):
        return self

    async def __anext__(self) -> StreamOutput:
        output = await self._queue.get()
        if output is None:
            raise StopAsyncIteration
        return output


@dataclass
class _RequestState:
    sequence: object
    queue: asyncio.Queue
    observed_tokens: int = 0


class AsyncLLMEngine:
    """Run the synchronous engine step-by-step inside an asyncio task.

    Requests may be submitted while other requests are decoding. The engine
    yields control to the event loop between scheduler steps, which is the
    admission boundary for newly arrived requests.
    """

    def __init__(self, model: str, **kwargs):
        self.engine = LLM(model, **kwargs)
        self._requests: dict[int, _RequestState] = {}
        self._wakeup = asyncio.Event()
        self._engine_task: asyncio.Task | None = None
        self._closed = False

    async def submit(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
    ) -> RequestStream:
        if self._closed:
            raise RuntimeError("the async engine is closed")
        request_id = self.engine.add_request(prompt, sampling_params)
        sequence = self.engine.scheduler.waiting[-1]
        queue = asyncio.Queue()
        self._requests[request_id] = _RequestState(sequence, queue)
        if self._engine_task is None:
            self._engine_task = asyncio.create_task(self._run_engine())
        self._wakeup.set()
        return RequestStream(request_id, queue)

    async def close(self) -> None:
        """Reject new work and wait for already submitted requests to finish."""
        self._closed = True
        self._wakeup.set()
        if self._engine_task is not None:
            await self._engine_task

    async def _run_engine(self) -> None:
        while not self._closed or not self.engine.is_finished():
            if self.engine.is_finished():
                self._wakeup.clear()
                if self.engine.is_finished() and not self._closed:
                    await self._wakeup.wait()
                continue

            self.engine.step()
            self._publish_new_tokens()
            await asyncio.sleep(0)

    def _publish_new_tokens(self) -> None:
        finished_requests = []
        for request_id, state in self._requests.items():
            token_ids = state.sequence.completion_token_ids
            new_token_ids = token_ids[state.observed_tokens:]
            for index, token_id in enumerate(new_token_ids):
                is_last = state.sequence.is_finished and index == len(new_token_ids) - 1
                state.queue.put_nowait(
                    StreamOutput(
                        request_id=request_id,
                        token_id=token_id,
                        token_ids=tuple(token_ids[: state.observed_tokens + index + 1]),
                        text=self.engine.tokenizer.decode(
                            token_ids[: state.observed_tokens + index + 1]
                        ),
                        finished=is_last,
                    )
                )
            state.observed_tokens = len(token_ids)
            if state.sequence.is_finished:
                state.queue.put_nowait(None)
                finished_requests.append(request_id)

        for request_id in finished_requests:
            del self._requests[request_id]
