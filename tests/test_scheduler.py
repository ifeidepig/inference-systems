import unittest
from types import SimpleNamespace

from nanovllm import SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


def make_scheduler(policy: str) -> Scheduler:
    config = SimpleNamespace(
        max_num_seqs=4,
        max_num_batched_tokens=256,
        eos=0,
        kvcache_block_size=256,
        num_kvcache_blocks=16,
        scheduling_policy=policy,
    )
    return Scheduler(config)


def add_running_decode(scheduler: Scheduler) -> Sequence:
    sequence = Sequence(
        [1] * 64,
        SamplingParams(temperature=0.1, max_tokens=8, ignore_eos=True),
    )
    scheduler.block_manager.allocate(sequence, num_cached_blocks=0)
    sequence.status = SequenceStatus.RUNNING
    sequence.is_prefill = False
    scheduler.running.append(sequence)
    return sequence


def add_waiting_prefill(scheduler: Scheduler) -> Sequence:
    sequence = Sequence(
        [2] * 300,
        SamplingParams(temperature=0.1, max_tokens=8, ignore_eos=True),
    )
    scheduler.add(sequence)
    return sequence


class SchedulerPolicyTest(unittest.TestCase):

    def test_prefill_first_preserves_legacy_behavior(self):
        scheduler = make_scheduler("prefill_first")
        running = add_running_decode(scheduler)
        waiting = add_waiting_prefill(scheduler)

        batches = scheduler.schedule()

        self.assertEqual(len(batches), 1)
        self.assertTrue(batches[0].is_prefill)
        self.assertEqual(batches[0].seqs, [waiting])
        self.assertEqual(batches[0].num_tokens, 256)
        self.assertEqual(running.num_scheduled_tokens, 0)

    def test_decode_first_schedules_both_phases_within_budget(self):
        scheduler = make_scheduler("decode_first")
        running = add_running_decode(scheduler)
        waiting = add_waiting_prefill(scheduler)

        batches = scheduler.schedule()

        self.assertEqual(len(batches), 2)
        self.assertFalse(batches[0].is_prefill)
        self.assertEqual(batches[0].seqs, [running])
        self.assertTrue(batches[1].is_prefill)
        self.assertEqual(batches[1].seqs, [waiting])
        self.assertEqual(sum(batch.num_tokens for batch in batches), 256)
        self.assertEqual(running.num_scheduled_tokens, 1)
        self.assertEqual(waiting.num_scheduled_tokens, 255)


if __name__ == "__main__":
    unittest.main()
