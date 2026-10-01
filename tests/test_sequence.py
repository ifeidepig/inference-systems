import unittest

from nanovllm import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus


class SequenceLifecycleTest(unittest.TestCase):

    def test_records_request_lifecycle_metrics(self):
        sequence = Sequence(
            [1, 2],
            SamplingParams(temperature=0.1, max_tokens=2, ignore_eos=True),
            arrival_time_ns=1_000_000,
            admitted_time_ns=1_500_000,
        )
        sequence.block_table = [3, 4]
        sequence.mark_scheduled(2_000_000)
        sequence.append_token(5, 3_000_000)
        sequence.mark_scheduled(4_000_000)
        sequence.append_token(6, 5_000_000)
        sequence.status = SequenceStatus.FINISHED
        sequence.mark_finished(6_000_000)

        metrics = sequence.lifecycle_metrics(now_ns=10_000_000)

        self.assertEqual(metrics["queue_ms"], 1.0)
        self.assertEqual(metrics["admission_delay_ms"], 0.5)
        self.assertEqual(metrics["ttft_ms"], 2.0)
        self.assertEqual(metrics["e2e_ms"], 5.0)
        self.assertEqual(metrics["tpot_ms"], 2.0)
        self.assertEqual(metrics["max_token_gap_ms"], 2.0)
        self.assertEqual(metrics["time_since_last_token_ms"], 1.0)
        self.assertEqual(metrics["schedule_count"], 2)
        self.assertEqual(metrics["peak_kv_blocks"], 2)

    def test_rejects_empty_prompt(self):
        with self.assertRaisesRegex(ValueError, "at least one token"):
            Sequence([])


if __name__ == "__main__":
    unittest.main()
