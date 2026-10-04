"""Opt-in CPU/CUDA phase timing for inference-path diagnosis."""

from contextlib import contextmanager
from statistics import median
from time import perf_counter_ns

import torch
from torch.profiler import record_function


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(int(len(ordered) * fraction), len(ordered) - 1)
    return ordered[index]


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "total": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "q1": 0.0,
            "q3": 0.0,
        }
    ordered = sorted(values)
    return {
        "count": len(values),
        "total": sum(values),
        "mean": sum(values) / len(values),
        "median": median(values),
        "p95": _percentile(values, 0.95),
        "q1": ordered[len(ordered) // 4],
        "q3": ordered[(3 * len(ordered)) // 4],
    }


class PhaseProfiler:
    """Collect wall-clock enqueue time and deferred CUDA Event latency.

    Profiling is disabled by default. When enabled, phase events are resolved
    with one synchronization at the end of an iteration instead of forcing a
    synchronization at every phase boundary.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.reset()

    def reset(self) -> None:
        self.cpu_ms: dict[str, list[float]] = {}
        self.gpu_ms: dict[str, list[float]] = {}
        self.allocated_delta_bytes: dict[str, list[float]] = {}
        self.counters: dict[str, float] = {}
        self._pending_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self.flush_count = 0
        self.flush_sync_ms = 0.0

    @contextmanager
    def phase(self, name: str, *, gpu: bool = True):
        if not self.enabled:
            yield
            return
        use_cuda = gpu and torch.cuda.is_available()
        use_nvtx = torch.cuda.is_available()
        start_event = end_event = None
        allocated_before = 0
        if use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            allocated_before = torch.cuda.memory_allocated()
            start_event.record()
        if use_nvtx:
            torch.cuda.nvtx.range_push(name)
        torch_range = record_function(name)
        torch_range.__enter__()
        cpu_started = perf_counter_ns()
        try:
            yield
        finally:
            torch_range.__exit__(None, None, None)
            if use_nvtx:
                torch.cuda.nvtx.range_pop()
            cpu_elapsed = (perf_counter_ns() - cpu_started) / 1e6
            self.cpu_ms.setdefault(name, []).append(cpu_elapsed)
            if use_cuda:
                end_event.record()
                allocated_after = torch.cuda.memory_allocated()
                self.allocated_delta_bytes.setdefault(name, []).append(
                    float(allocated_after - allocated_before)
                )
                self._pending_events.append((name, start_event, end_event))

    def add_counter(self, name: str, value: int | float) -> None:
        if self.enabled:
            self.counters[name] = self.counters.get(name, 0.0) + float(value)

    def record_cpu(self, name: str, elapsed_ms: float) -> None:
        if self.enabled:
            self.cpu_ms.setdefault(name, []).append(float(elapsed_ms))

    def set_counter(self, name: str, value: int | float) -> None:
        if self.enabled:
            self.counters[name] = float(value)

    def flush(self) -> None:
        if not self.enabled or not self._pending_events:
            return
        started = perf_counter_ns()
        torch.cuda.synchronize()
        self.flush_sync_ms += (perf_counter_ns() - started) / 1e6
        self.flush_count += 1
        for name, start_event, end_event in self._pending_events:
            self.gpu_ms.setdefault(name, []).append(
                float(start_event.elapsed_time(end_event))
            )
        self._pending_events.clear()

    def metrics(self) -> dict[str, object]:
        phases = sorted(set(self.cpu_ms) | set(self.gpu_ms))
        return {
            "enabled": self.enabled,
            "flush_count": self.flush_count,
            "flush_sync_ms": self.flush_sync_ms,
            "counters": dict(sorted(self.counters.items())),
            "phases": {
                name: {
                    "cpu_ms": _summary(self.cpu_ms.get(name, [])),
                    "gpu_ms": _summary(self.gpu_ms.get(name, [])),
                    "allocated_delta_bytes": _summary(
                        self.allocated_delta_bytes.get(name, [])
                    ),
                }
                for name in phases
            },
        }
