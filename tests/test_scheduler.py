import unittest
from types import SimpleNamespace

from nanovllm import SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


def make_scheduler(
    policy: str,
    *,
    enable_prefix_cache: bool = True,
    enable_chunked_prefill: bool = True,
    target_ttft_ms: float = 200.0,
    target_tpot_ms: float = 50.0,
    kv_pressure_threshold: float = 0.9,
    queue_pressure_threshold: int = 3,
) -> Scheduler:
    config = SimpleNamespace(
        max_num_seqs=4,
        max_num_batched_tokens=256,
        eos=0,
        kvcache_block_size=256,
        num_kvcache_blocks=16,
        scheduling_policy=policy,
        enable_prefix_cache=enable_prefix_cache,
        enable_chunked_prefill=enable_chunked_prefill,
        scheduler_target_ttft_ms=target_ttft_ms,
        scheduler_target_tpot_ms=target_tpot_ms,
        slo_prefill_priority_threshold=0.8,
        slo_min_prefill_tokens=64,
        slo_kv_pressure_threshold=kv_pressure_threshold,
        slo_queue_pressure_threshold=queue_pressure_threshold,
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

    def test_abort_waiting_sequence_removes_it(self):
        scheduler = make_scheduler("decode_first")
        waiting = add_waiting_prefill(scheduler)

        aborted = scheduler.abort(waiting.seq_id)

        self.assertTrue(aborted)
        self.assertTrue(waiting.is_finished)
        self.assertNotIn(waiting, scheduler.waiting)
        self.assertTrue(scheduler.is_finished())
        self.assertEqual(scheduler.get_metrics()["completed_request_count"], 1)
        self.assertEqual(
            scheduler.get_request_metrics()[0]["status"], "aborted"
        )

    def test_abort_running_sequence_releases_kv_blocks(self):
        scheduler = make_scheduler("decode_first")
        running = add_running_decode(scheduler)
        free_blocks_before = len(scheduler.block_manager.free_block_ids)

        aborted = scheduler.abort(running.seq_id)

        self.assertTrue(aborted)
        self.assertTrue(running.is_finished)
        self.assertEqual(running.block_table, [])
        self.assertEqual(
            len(scheduler.block_manager.free_block_ids),
            free_blocks_before + 1,
        )
        self.assertTrue(scheduler.is_finished())

    def test_abort_unknown_sequence_is_a_noop(self):
        scheduler = make_scheduler("decode_first")

        self.assertFalse(scheduler.abort(999999))

    def test_chunked_prefill_can_be_disabled(self):
        scheduler = make_scheduler(
            "prefill_first", enable_chunked_prefill=False
        )
        waiting = add_waiting_prefill(scheduler)

        scheduled = scheduler._schedule_prefill(256, 4)

        self.assertEqual(scheduled, [waiting])
        self.assertEqual(waiting.num_scheduled_tokens, 300)
        self.assertEqual(len(waiting.block_table), 2)
        self.assertEqual(scheduler.get_metrics()["chunked_prefill_steps"], 0)

    def test_metrics_record_chunked_prefill(self):
        scheduler = make_scheduler("prefill_first")
        add_waiting_prefill(scheduler)

        scheduler.schedule()
        metrics = scheduler.get_metrics()

        self.assertEqual(metrics["chunked_prefill_steps"], 1)
        self.assertEqual(metrics["kv_blocks_used"], 2)
        self.assertEqual(metrics["kv_blocks_free"], 14)

    def test_prefix_cache_hit_is_observable(self):
        scheduler = make_scheduler("prefill_first")
        sampling = SamplingParams(
            temperature=0.1, max_tokens=1, ignore_eos=True
        )
        prompt = [3] * 300

        first = Sequence(prompt, sampling)
        scheduler.add(first)
        first_batch = scheduler.schedule()[0]
        scheduler.postprocess(first_batch.seqs, [9], is_prefill=True)
        second_batch = scheduler.schedule()[0]
        scheduler.postprocess(second_batch.seqs, [9], is_prefill=True)

        repeated = Sequence(prompt, sampling)
        scheduler.add(repeated)
        repeated_batch = scheduler.schedule()[0]
        metrics = scheduler.get_metrics()

        self.assertEqual(repeated.num_cached_tokens, 256)
        self.assertEqual(repeated_batch.num_tokens, 44)
        self.assertEqual(metrics["prefix_cache_hit_blocks"], 1)
        self.assertEqual(metrics["prefix_cache_eligible_blocks"], 2)
        self.assertEqual(metrics["prefix_cache_hit_rate"], 0.5)

    def test_disabling_prefix_cache_prevents_hashing_and_reuse(self):
        scheduler = make_scheduler(
            "prefill_first", enable_prefix_cache=False
        )
        waiting = add_waiting_prefill(scheduler)
        batch = scheduler.schedule()[0]

        scheduler.postprocess(batch.seqs, [9], is_prefill=True)
        metrics = scheduler.get_metrics()

        self.assertEqual(waiting.num_cached_tokens, 256)
        self.assertEqual(metrics["prefix_cache_queries"], 0)
        self.assertEqual(metrics["prefix_cache_hit_blocks"], 0)
        self.assertEqual(metrics["kv_blocks_cached"], 0)


class SloAwareSchedulerTest(unittest.TestCase):

    NOW_NS = 1_000_000_000

    def add_running(self, scheduler: Scheduler, token_gap_ms: float) -> Sequence:
        sequence = add_running_decode(scheduler)
        sequence.token_timestamps_ns = [
            self.NOW_NS - int(token_gap_ms * 1e6)
        ]
        return sequence

    def add_waiting(self, scheduler: Scheduler, wait_ms: float) -> Sequence:
        sequence = Sequence(
            [2] * 300,
            SamplingParams(temperature=0.1, max_tokens=8, ignore_eos=True),
            arrival_time_ns=self.NOW_NS - int(wait_ms * 1e6),
        )
        scheduler.add(sequence)
        return sequence

    def test_urgent_prefill_does_not_skip_active_decode(self):
        scheduler = make_scheduler("slo_aware")
        running = self.add_running(scheduler, token_gap_ms=10)
        waiting = self.add_waiting(scheduler, wait_ms=180)

        batches = scheduler._schedule_slo_aware(self.NOW_NS)

        self.assertEqual(len(batches), 2)
        self.assertFalse(batches[0].is_prefill)
        self.assertEqual(batches[0].seqs, [running])
        self.assertTrue(batches[1].is_prefill)
        self.assertEqual(batches[1].seqs, [waiting])
        self.assertEqual(
            scheduler.get_metrics()["last_slo_decision"]["decision"],
            "decode_first",
        )

    def test_decode_wins_and_limits_dynamic_prefill_budget(self):
        scheduler = make_scheduler("slo_aware")
        scheduler.prefill_ms_per_token_ewma = 0.2
        scheduler.prefill_base_ms = 20.0
        scheduler.decode_step_ms_ewma = 6.0
        running = self.add_running(scheduler, token_gap_ms=60)
        waiting = self.add_waiting(scheduler, wait_ms=100)

        batches = scheduler._schedule_slo_aware(self.NOW_NS)

        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0].seqs, [running])
        self.assertFalse(batches[0].is_prefill)
        self.assertEqual(batches[1].seqs, [waiting])
        self.assertTrue(batches[1].is_prefill)
        self.assertEqual(batches[1].num_tokens, 155)
        self.assertEqual(
            scheduler.get_metrics()["last_slo_decision"]["prefill_budget"],
            155,
        )

    def test_high_kv_pressure_defers_nonurgent_prefill(self):
        scheduler = make_scheduler(
            "slo_aware", kv_pressure_threshold=0.05
        )
        running = self.add_running(scheduler, token_gap_ms=60)
        self.add_waiting(scheduler, wait_ms=20)

        batches = scheduler._schedule_slo_aware(self.NOW_NS)

        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].seqs, [running])
        self.assertFalse(batches[0].is_prefill)
        self.assertEqual(
            scheduler.get_metrics()["slo_admission_deferred_steps"], 1
        )

    def test_infeasible_latency_budget_defers_prefill_until_urgent(self):
        scheduler = make_scheduler("slo_aware", target_tpot_ms=20.0)
        scheduler.prefill_ms_per_token_ewma = 0.2
        scheduler.prefill_base_ms = 20.0
        scheduler.decode_step_ms_ewma = 6.0

        deferred = scheduler._latency_bounded_prefill_budget(
            remaining_tokens=255, waiting_urgency=0.5
        )
        urgent = scheduler._latency_bounded_prefill_budget(
            remaining_tokens=255, waiting_urgency=0.9
        )

        self.assertEqual(deferred, 0)
        self.assertEqual(urgent, 255)
        self.assertEqual(scheduler.slo_infeasible_budget_steps, 2)

    def test_batch_observations_update_latency_estimates(self):
        scheduler = make_scheduler("slo_aware")

        scheduler.observe_batch(is_prefill=True, num_tokens=256, duration_ms=51.2)
        scheduler.observe_batch(is_prefill=True, num_tokens=32, duration_ms=20)
        scheduler.observe_batch(is_prefill=True, num_tokens=256, duration_ms=200)
        scheduler.observe_batch(is_prefill=False, num_tokens=4, duration_ms=8)

        self.assertEqual(scheduler.prefill_ms_per_token_ewma, 0.2)
        self.assertEqual(scheduler.prefill_base_ms, 20.0)
        self.assertEqual(scheduler.decode_step_ms_ewma, 8.0)
        self.assertEqual(scheduler.slo_rejected_latency_observations, 1)

    def test_queue_pressure_restores_full_prefill_budget(self):
        scheduler = make_scheduler("slo_aware", queue_pressure_threshold=3)
        scheduler.prefill_ms_per_token_ewma = 0.2
        scheduler.prefill_base_ms = 20.0
        scheduler.decode_step_ms_ewma = 6.0
        self.add_running(scheduler, token_gap_ms=10)
        for wait_ms in (20, 15, 10):
            self.add_waiting(scheduler, wait_ms=wait_ms)

        batches = scheduler._schedule_slo_aware(self.NOW_NS)

        self.assertEqual(batches[1].num_tokens, 255)
        self.assertEqual(scheduler.slo_throughput_priority_steps, 1)


if __name__ == "__main__":
    unittest.main()
