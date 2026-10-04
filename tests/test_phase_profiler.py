from nanovllm.engine.phase_profiler import PhaseProfiler


def test_disabled_phase_profiler_is_noop():
    profiler = PhaseProfiler(enabled=False)
    with profiler.phase("disabled", gpu=False):
        profiler.add_counter("ignored", 1)
    assert profiler.metrics() == {
        "enabled": False,
        "flush_count": 0,
        "flush_sync_ms": 0.0,
        "counters": {},
        "phases": {},
    }


def test_cpu_phase_profiler_summarizes_samples_and_counters():
    profiler = PhaseProfiler(enabled=True)
    with profiler.phase("cpu", gpu=False):
        pass
    profiler.record_cpu("cpu", 2.0)
    profiler.add_counter("bytes", 10)
    profiler.add_counter("bytes", 5)

    metrics = profiler.metrics()

    assert metrics["phases"]["cpu"]["cpu_ms"]["count"] == 2
    assert metrics["phases"]["cpu"]["gpu_ms"]["count"] == 0
    assert metrics["counters"]["bytes"] == 15.0
