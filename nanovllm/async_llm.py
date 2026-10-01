"""Asynchronous request submission and token streaming for nano-vLLM."""

import asyncio
from dataclasses import dataclass
from collections.abc import Awaitable, Callable

from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams


@dataclass(frozen=True)
class StreamOutput:
    request_id: int
    token_id: int
    token_ids: tuple[int, ...]
    text: str
    finished: bool
    finish_reason: str | None = None


@dataclass(frozen=True)
class _StreamError:
    error: Exception


class RequestStream:
    """Async iterator over tokens generated for one request."""

    def __init__(
        self,
        request_id: int,
        queue: asyncio.Queue,
        cancel_callback: Callable[[int], Awaitable[bool]] | None = None,
    ):
        self.request_id = request_id
        self._queue = queue
        self._cancel_callback = cancel_callback
        self._closed = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> StreamOutput:
        if self._closed:
            raise StopAsyncIteration
        output = await self._queue.get()
        if output is None:
            self._closed = True
            raise StopAsyncIteration
        if isinstance(output, _StreamError):
            self._closed = True
            raise output.error
        if output.finished:
            self._closed = True
        return output

    async def cancel(self) -> bool:
        if self._closed:
            return False
        self._closed = True
        if self._cancel_callback is None:
            return False
        return await self._cancel_callback(self.request_id)

    async def aclose(self) -> None:
        await self.cancel()


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

    def __init__(self, model: str, stream_queue_size: int = 16, **kwargs):
        if stream_queue_size < 1:
            raise ValueError("stream_queue_size must be positive")
        self.engine = LLM(model, **kwargs)
        self.model_name = model
        self._stream_queue_size = stream_queue_size
        self._requests: dict[int, _RequestState] = {}
        self._wakeup = asyncio.Event()
        self._engine_task: asyncio.Task | None = None
        self._closed = False
        self._failure: Exception | None = None

    async def submit(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
    ) -> RequestStream:
        if self._closed:
            raise RuntimeError("the async engine is closed")
        if self._failure is not None:
            raise RuntimeError("the async engine has failed") from self._failure
        request_id = self.engine.add_request(prompt, sampling_params)
        sequence = self.engine.scheduler.waiting[-1]
        queue = asyncio.Queue(maxsize=self._stream_queue_size)
        self._requests[request_id] = _RequestState(sequence, queue)
        if self._engine_task is None:
            self._engine_task = asyncio.create_task(self._run_engine())
        self._wakeup.set()
        return RequestStream(request_id, queue, self.cancel)

    async def cancel(self, request_id: int) -> bool:
        state = self._requests.pop(request_id, None)
        if state is None:
            return False
        aborted = self.engine.abort_request(request_id)
        self._clear_queue(state.queue)
        state.queue.put_nowait(None)
        return aborted

    async def close(self) -> None:
        """Reject new work and wait for already submitted requests to finish."""
        self._closed = True
        self._wakeup.set()
        try:
            if self._engine_task is not None:
                await self._engine_task
        finally:
            self.engine.exit()
        if self._failure is not None:
            raise RuntimeError("the async engine stopped after an inference error") from self._failure

    async def _run_engine(self) -> None:
        try:
            while not self._closed or not self.engine.is_finished():
                if self.engine.is_finished():
                    self._wakeup.clear()
                    if self.engine.is_finished() and not self._closed:
                        await self._wakeup.wait()
                    continue

                self.engine.step()
                self._publish_new_tokens()
                await asyncio.sleep(0)
        except Exception as exc:
            self._failure = exc
            self._fail_requests(exc)

    def _publish_new_tokens(self) -> None:
        finished_requests = []
        for request_id, state in list(self._requests.items()):
            token_ids = state.sequence.completion_token_ids
            new_token_ids = token_ids[state.observed_tokens:]
            for index, token_id in enumerate(new_token_ids):
                is_last = state.sequence.is_finished and index == len(new_token_ids) - 1
                self._put_latest(
                    state.queue,
                    StreamOutput(
                        request_id=request_id,
                        token_id=token_id,
                        token_ids=tuple(token_ids[: state.observed_tokens + index + 1]),
                        text=self.engine.tokenizer.decode(
                            token_ids[: state.observed_tokens + index + 1]
                        ),
                        finished=is_last,
                        finish_reason=(
                            "length"
                            if is_last
                            and state.sequence.num_completion_tokens
                            >= state.sequence.max_tokens
                            else "stop" if is_last else None
                        ),
                    ),
                )
            state.observed_tokens = len(token_ids)
            if state.sequence.is_finished and self._requests.get(request_id) is state:
                finished_requests.append(request_id)

        for request_id in finished_requests:
            self._requests.pop(request_id, None)

    def _fail_requests(self, error: Exception) -> None:
        for state in self._requests.values():
            self._clear_queue(state.queue)
            state.queue.put_nowait(_StreamError(error))
        self._requests.clear()

    @staticmethod
    def _clear_queue(queue: asyncio.Queue) -> None:
        while not queue.empty():
            queue.get_nowait()

    @staticmethod
    def _put_latest(queue: asyncio.Queue, output: StreamOutput) -> None:
        if queue.full():
            queue.get_nowait()
        queue.put_nowait(output)
