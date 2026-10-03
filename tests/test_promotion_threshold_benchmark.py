from collections import Counter

from benchmark_promotion_threshold import build_prompt, build_trace
from benchmark_promotion_policy_simulation import run_trace


def test_frequency_workload_trace_counts_are_exact():
    expected = {
        "singleton": [1, 1, 1, 1],
        "pair": [2, 2, 2, 2],
        "triple": [3, 3, 3, 3],
        "hot": [8],
    }
    for workload, frequencies in expected.items():
        trace = build_trace(
            workload,
            num_prefixes=4,
            hot_frequency=8,
            zipf_requests=16,
            seed=7,
        )
        assert sorted(Counter(item[0] for item in trace).values()) == frequencies


def test_zipf_trace_is_deterministic_and_has_requested_length():
    first = build_trace(
        "zipf",
        num_prefixes=8,
        hot_frequency=8,
        zipf_requests=32,
        seed=11,
    )
    second = build_trace(
        "zipf",
        num_prefixes=8,
        hot_frequency=8,
        zipf_requests=32,
        seed=11,
    )
    assert first == second
    assert len(first) == 32


def test_prompt_builder_shares_prefix_and_changes_suffix():
    first = build_prompt(1000, 2, 1, 16, 8)
    second = build_prompt(1000, 2, 2, 16, 8)
    other_prefix = build_prompt(1000, 3, 1, 16, 8)

    assert first[:16] == second[:16]
    assert first[16:] != second[16:]
    assert first[:16] != other_prefix[:16]


def test_long_horizon_policy_simulation_distinguishes_pair_and_triple():
    pair = build_trace(
        "pair",
        num_prefixes=4,
        hot_frequency=8,
        zipf_requests=16,
        seed=7,
    )
    triple = build_trace(
        "triple",
        num_prefixes=4,
        hot_frequency=8,
        zipf_requests=16,
        seed=7,
    )

    pair_two = run_trace(
        pair,
        threshold=2,
        prefix_length=256,
        suffix_length=64,
        capacity=16,
    )
    pair_three = run_trace(
        pair,
        threshold=3,
        prefix_length=256,
        suffix_length=64,
        capacity=16,
    )
    triple_two = run_trace(
        triple,
        threshold=2,
        prefix_length=256,
        suffix_length=64,
        capacity=16,
    )
    triple_three = run_trace(
        triple,
        threshold=3,
        prefix_length=256,
        suffix_length=64,
        capacity=16,
    )

    assert pair_two["promotions_published"] == 4
    assert pair_two["useful_promotions"] == 0
    assert pair_three["promotions_published"] == 0
    assert triple_two["useful_promotions"] == 4
    assert triple_three["useful_promotions"] == 0
    assert (
        triple_two["alignment_replay_tokens"]
        < triple_three["alignment_replay_tokens"]
    )
