import unittest
import asyncio
from pathlib import Path

from benchmark_serving import (
    aggregate_trials,
    make_prompt,
    make_prompts,
    percentile,
    summarize,
)
from benchmark_interference import token_gaps_ms
from benchmark_matrix import build_command
from nanovllm.async_llm import RequestStream, StreamOutput, _StreamError


class PromptTest(unittest.TestCase):

    def test_first_token_makes_each_prompt_unique(self):
        self.assertEqual(make_prompt(7, 11, 4), [11, 7, 7, 7])

    def test_prompt_length_must_be_positive(self):
        with self.assertRaises(ValueError):
            make_prompt(7, 11, 0)

    def test_prompt_batch_has_distinct_first_tokens(self):
        prompts = make_prompts(7, length=3, count=3, vocab_size=100)
        self.assertEqual(prompts, [[8, 7, 7], [9, 7, 7], [10, 7, 7]])


class PercentileTest(unittest.TestCase):

    def test_single_value(self):
        self.assertEqual(percentile([7.0], 0.99), 7.0)

    def test_interpolates_between_values(self):
        self.assertEqual(percentile([0.0, 10.0], 0.50), 5.0)
        self.assertEqual(percentile([0.0, 10.0], 0.95), 9.5)

    def test_input_order_does_not_matter(self):
        self.assertEqual(percentile([3.0, 1.0, 2.0], 0.50), 2.0)

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            percentile([], 0.50)


class SummaryTest(unittest.TestCase):

    def test_reports_mean_and_tail_percentiles(self):
        result = summarize([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result["mean"], 2.5)
        self.assertEqual(result["p50"], 2.5)
        self.assertAlmostEqual(result["stddev"], 1.118033988749895)
        self.assertAlmostEqual(result["p95"], 3.85)
        self.assertAlmostEqual(result["p99"], 3.97)

    def test_aggregates_trial_and_request_metrics(self):
        trials = [
            {
                "summary": {
                    "duration_s": duration,
                    "request_throughput_rps": throughput,
                    "output_throughput_tokens_per_s": throughput * 10,
                    "peak_allocated_gib": peak,
                    "peak_reserved_gib": peak + 1,
                },
                "requests": [
                    {
                        "ttft_ms": latency,
                        "tpot_ms": latency + 1,
                        "e2e_ms": latency + 2,
                    }
                ],
            }
            for duration, throughput, peak, latency in (
                (1.0, 2.0, 3.0, 4.0),
                (3.0, 4.0, 5.0, 6.0),
            )
        ]

        result = aggregate_trials(trials)

        self.assertEqual(result["duration_s"]["mean"], 2.0)
        self.assertEqual(result["request_throughput_rps"]["mean"], 3.0)
        self.assertEqual(result["peak_allocated_gib"], 5.0)
        self.assertEqual(result["ttft_ms"]["mean"], 5.0)


class TokenGapTest(unittest.TestCase):

    def test_converts_consecutive_timestamps_to_milliseconds(self):
        gaps = token_gaps_ms([1.0, 1.01, 1.03])
        self.assertAlmostEqual(gaps[0], 10.0)
        self.assertAlmostEqual(gaps[1], 20.0)


class MatrixCommandTest(unittest.TestCase):

    def test_builds_one_isolated_benchmark_command(self):
        command = build_command(
            model=Path("model"),
            output=Path("result.json"),
            policy="decode_first",
            prompt_length=768,
            token_budget=256,
            repetitions=5,
        )
        self.assertIn("--scheduling-policy", command)
        self.assertIn("decode_first", command)
        self.assertIn("768", command)
        self.assertIn("256", command)
        self.assertIn("5", command)


class RequestStreamTest(unittest.IsolatedAsyncioTestCase):

    async def test_streams_outputs_until_finished_sentinel(self):
        queue = asyncio.Queue()
        stream = RequestStream(request_id=7, queue=queue)
        queue.put_nowait(StreamOutput(7, 11, (11,), "a", False))
        queue.put_nowait(StreamOutput(7, 12, (11, 12), "ab", True))
        queue.put_nowait(None)

        outputs = [output async for output in stream]

        self.assertEqual([output.token_id for output in outputs], [11, 12])
        self.assertFalse(outputs[0].finished)
        self.assertTrue(outputs[1].finished)

    async def test_propagates_background_engine_error(self):
        queue = asyncio.Queue()
        stream = RequestStream(request_id=7, queue=queue)
        queue.put_nowait(_StreamError(RuntimeError("model failed")))

        with self.assertRaisesRegex(RuntimeError, "model failed"):
            await anext(stream)

    async def test_cancel_invokes_engine_callback_once(self):
        cancelled = []

        async def cancel_callback(request_id):
            cancelled.append(request_id)
            return True

        stream = RequestStream(
            request_id=7,
            queue=asyncio.Queue(),
            cancel_callback=cancel_callback,
        )

        self.assertTrue(await stream.cancel())
        self.assertFalse(await stream.cancel())
        self.assertEqual(cancelled, [7])

    async def test_finished_output_ends_stream_without_sentinel(self):
        queue = asyncio.Queue()
        stream = RequestStream(request_id=7, queue=queue)
        queue.put_nowait(StreamOutput(7, 11, (11,), "done", True, "stop"))

        output = await anext(stream)

        self.assertTrue(output.finished)
        with self.assertRaises(StopAsyncIteration):
            await anext(stream)


if __name__ == "__main__":
    unittest.main()
