import asyncio
import json
import unittest
from types import SimpleNamespace

try:
    from fastapi.testclient import TestClient
except ImportError:
    TestClient = None

from nanovllm.async_llm import (
    AsyncLLMEngine,
    RequestStream,
    StreamOutput,
    _RequestState,
)

if TestClient is not None:
    from nanovllm.server import create_app


class _FakeTokenizer:

    def encode(self, text):
        return text.split()

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)

    def decode(self, token_ids):
        return " ".join(map(str, token_ids))


class _FakeAsyncEngine:

    def __init__(self):
        self.model_name = "fake-model"
        self.engine = SimpleNamespace(
            tokenizer=_FakeTokenizer(),
            get_runtime_metrics=lambda: {"scheduler": {"kv_blocks_used": 0}},
            get_request_metrics=lambda: [{"request_id": 1, "ttft_ms": 2.0}],
        )
        self.closed = False

    async def submit(self, prompt, sampling_params):
        queue = asyncio.Queue()
        queue.put_nowait(StreamOutput(1, 11, (11,), "hello", False))
        queue.put_nowait(StreamOutput(1, 12, (11, 12), "hello world", True, "length"))
        queue.put_nowait(None)
        return RequestStream(1, queue)

    async def close(self):
        self.closed = True


@unittest.skipIf(TestClient is None, "serve dependencies are not installed")
class ServerTest(unittest.TestCase):

    def setUp(self):
        self.engine = _FakeAsyncEngine()
        self.client = TestClient(create_app(self.engine, close_engine=False))

    def test_non_streaming_completion(self):
        response = self.client.post(
            "/v1/completions",
            json={"prompt": "say hello", "max_tokens": 2, "temperature": 0.1},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["choices"][0]["text"], "hello world")
        self.assertEqual(body["choices"][0]["finish_reason"], "length")
        self.assertEqual(body["usage"]["prompt_tokens"], 2)
        self.assertEqual(body["usage"]["completion_tokens"], 2)

    def test_completion_accepts_greedy_temperature_zero(self):
        response = self.client.post(
            "/v1/completions",
            json={"prompt": "say hello", "max_tokens": 2, "temperature": 0.0},
        )

        self.assertEqual(response.status_code, 200)

    def test_streaming_completion_emits_text_deltas_and_done(self):
        with self.client.stream(
            "POST",
            "/v1/completions",
            json={"prompt": "say hello", "stream": True},
        ) as response:
            lines = [line for line in response.iter_lines() if line]

        self.assertEqual(response.status_code, 200)
        self.assertEqual(lines[-1], "data: [DONE]")
        chunks = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
        self.assertEqual(
            [chunk["choices"][0]["text"] for chunk in chunks],
            ["hello", " world"],
        )

    def test_chat_completion_formats_messages(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 2,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["choices"][0]["message"]["content"],
            "hello world",
        )

    def test_metrics_exposes_engine_snapshot(self):
        response = self.client.get("/metrics")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["scheduler"]["kv_blocks_used"], 0)

    def test_request_metrics_exposes_completed_requests(self):
        response = self.client.get("/metrics/requests")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["ttft_ms"], 2.0)


class SpeculativeStreamingBoundaryTest(unittest.TestCase):

    def test_queue_coalescing_keeps_complete_cumulative_output(self):
        engine = AsyncLLMEngine.__new__(AsyncLLMEngine)
        sequence = SimpleNamespace(
            completion_token_ids=[11, 12, 13],
            is_finished=True,
            num_completion_tokens=3,
            max_tokens=3,
        )
        queue = asyncio.Queue(maxsize=1)
        engine.engine = SimpleNamespace(tokenizer=_FakeTokenizer())
        engine._requests = {7: _RequestState(sequence, queue)}

        engine._publish_new_tokens()

        output = queue.get_nowait()
        self.assertEqual(output.token_id, 13)
        self.assertEqual(output.token_ids, (11, 12, 13))
        self.assertEqual(output.text, "11 12 13")
        self.assertTrue(output.finished)
        self.assertEqual(output.finish_reason, "length")
        self.assertNotIn(7, engine._requests)


if __name__ == "__main__":
    unittest.main()
