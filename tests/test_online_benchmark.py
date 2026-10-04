import unittest

from benchmark_online import (
    RequestSpec,
    build_prompt,
    generate_arrival_offsets,
    generate_scheduler_profile,
    generate_workload,
    output_digest,
    prompt_class,
    summarize_requests,
)


class ArrivalPatternTest(unittest.TestCase):

    def test_poisson_arrivals_are_sorted_and_reproducible(self):
        first = generate_arrival_offsets(5, 10.0, "poisson", seed=7, burst_size=2)
        second = generate_arrival_offsets(5, 10.0, "poisson", seed=7, burst_size=2)

        self.assertEqual(first, second)
        self.assertEqual(first[0], 0.0)
        self.assertEqual(first, sorted(first))

    def test_bursty_arrivals_group_requests(self):
        offsets = generate_arrival_offsets(
            7, 10.0, "bursty", seed=0, burst_size=3
        )

        self.assertEqual(offsets, [0.0, 0.0, 0.0, 300.0, 300.0, 300.0, 600.0])


class WorkloadTest(unittest.TestCase):

    def test_output_digest_is_order_sensitive_and_reproducible(self):
        self.assertEqual(output_digest([[1, 2], [3]]), output_digest([[1, 2], [3]]))
        self.assertNotEqual(output_digest([[1, 2], [3]]), output_digest([[3], [1, 2]]))

    def test_workload_generation_is_reproducible(self):
        kwargs = dict(
            num_requests=6,
            request_rate=4.0,
            arrival_pattern="poisson",
            prompt_lengths=[64, 768],
            output_lengths=[8, 32],
            shared_prefix_ratio=0.5,
            shared_prefix_groups=2,
            seed=11,
            burst_size=3,
        )

        self.assertEqual(generate_workload(**kwargs), generate_workload(**kwargs))

    def test_shared_group_builds_identical_prefix_and_unique_suffix(self):
        first = RequestSpec(0, 0.0, 8, 4, 1)
        second = RequestSpec(1, 0.0, 8, 4, 1)

        first_prompt = build_prompt(first, 10, 1000, shared_prefix_length=4)
        second_prompt = build_prompt(second, 10, 1000, shared_prefix_length=4)

        self.assertEqual(first_prompt[:4], second_prompt[:4])
        self.assertNotEqual(first_prompt[4:], second_prompt[4:])

    def test_prompt_classes(self):
        self.assertEqual(prompt_class(64), "short")
        self.assertEqual(prompt_class(256), "medium")
        self.assertEqual(prompt_class(768), "long")

    def test_summarizes_slo_violations(self):
        requests = [
            {
                "ttft_ms": 80.0,
                "tpot_ms": 20.0,
                "max_token_gap_ms": 30.0,
                "queue_ms": 10.0,
                "admission_delay_ms": 2.0,
                "e2e_ms": 150.0,
            },
            {
                "ttft_ms": 120.0,
                "tpot_ms": 40.0,
                "max_token_gap_ms": 60.0,
                "queue_ms": 30.0,
                "admission_delay_ms": 4.0,
                "e2e_ms": 250.0,
            },
        ]

        result = summarize_requests(requests, 100.0, 50.0)

        self.assertEqual(result["ttft_slo_violation_rate"], 0.5)
        self.assertEqual(result["tpot_slo_violation_rate"], 0.5)
        self.assertEqual(result["request_slo_violation_rate"], 0.5)
        self.assertEqual(result["request_latency_ms"]["p50"], 200.0)

    def test_scheduler_profiles_cover_required_workloads(self):
        for profile in (
            "shared_prefix",
            "multi_session",
            "unique_prompt",
            "kv_pressure",
            "kv_pressure_victim_choice",
            "multi_turn",
        ):
            workload = generate_scheduler_profile(profile)
            self.assertTrue(workload)
            self.assertEqual(
                [spec.request_index for spec in workload],
                list(range(len(workload))),
            )
        self.assertTrue(
            all(
                spec.shared_prefix_group is None
                for spec in generate_scheduler_profile("unique_prompt")
            )
        )
        victim_choice = generate_scheduler_profile(
            "kv_pressure_victim_choice"
        )
        self.assertEqual(victim_choice[0].shared_prefix_group, 0)
        self.assertIsNone(victim_choice[1].shared_prefix_group)


if __name__ == "__main__":
    unittest.main()
