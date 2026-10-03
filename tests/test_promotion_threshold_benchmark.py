from collections import Counter

from benchmark_promotion_threshold import build_prompt, build_trace


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
