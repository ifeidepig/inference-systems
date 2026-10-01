# SLO-Aware Scheduler Prototype

## Goal

The `slo_aware` policy controls how many prefill tokens may run after an active
decode batch. It targets the maximum inter-token gap rather than mean TPOT,
because one long prefill can be hidden by otherwise fast decode steps.

## Signals

- Waiting urgency: oldest request waiting time divided by the TTFT target.
- Decode urgency: largest active inter-token time divided by the TPOT target.
- KV pressure: used physical KV blocks divided by total blocks.
- Queue pressure: number of waiting requests.
- Online prefill and decode latency estimates.

## Decision Rules

When no decode request is active, the scheduler uses the full prefill token
budget. When decode requests are active, decode always runs first. The scheduler
then decides whether and how much prefill work can use the remaining token and
sequence capacity.

Non-urgent admission is deferred when KV pressure crosses its configured
threshold. A queue-pressure escape hatch restores the full remaining prefill
budget when three or more requests are waiting, preventing small chunks from
collapsing throughput during backlog growth.

For normal interleaving, the prefill budget is bounded by:

```text
available_ms = target_tpot_ms - decode_step_ms - safety_margin_ms
prefill_time = fixed_prefill_ms + incremental_ms_per_token * tokens
```

The fixed prefill cost is the minimum observed prefill duration. Incremental
cost is derived from an EWMA of near-full chunks. Small tail chunks do not update
the full-chunk estimate, and observations over twice the stable estimate are
treated as cold-compilation outliers for control purposes. Raw request latency
still includes those outliers.

If the fixed prefill cost itself exceeds the available latency budget, the SLO
is marked infeasible. Non-urgent prefill is deferred; once waiting urgency nears
the TTFT deadline, a larger chunk is allowed to make progress.

## RTX 3060 Prototype Results

Configuration:

- Qwen3-0.6B BF16, CUDA Graph enabled
- 24 requests, 3 requests/s
- Prompt lengths: 64, 256, and 768
- Output lengths: 8 and 32
- 50% shared-prefix requests
- TTFT target: 500 ms
- Maximum token-gap target: 42 ms
- Prefill token limit: 256
- `TORCHDYNAMO_DISABLE=1` for scheduler isolation

### Poisson Arrivals

| Metric | Decode first | SLO aware | Change |
| --- | ---: | ---: | ---: |
| Token SLO violation rate | 29.17% | 16.67% | -42.86% relative |
| Request SLO violation rate | 29.17% | 16.67% | -42.86% relative |
| Mean TTFT | 84.52 ms | 90.67 ms | +7.28% |
| Mean maximum token gap | 24.17 ms | 22.88 ms | -5.33% |
| Request throughput | 2.956 req/s | 2.941 req/s | -0.53% |

The policy reduces token-gap violations under low-queue Poisson traffic with a
small TTFT cost and nearly unchanged throughput.

### Bursty Arrivals

| Metric | Decode first | SLO aware | Change |
| --- | ---: | ---: | ---: |
| Token SLO violation rate | 37.50% | 54.17% | +44.44% relative |
| Request SLO violation rate | 37.50% | 75.00% | +100.00% relative |
| Mean TTFT | 168.03 ms | 253.47 ms | +50.84% |
| Mean maximum token gap | 28.14 ms | 30.55 ms | +8.56% |
| Request throughput | 3.321 req/s | 3.228 req/s | -2.80% |

The prototype is not robust to burst traffic. The queue-pressure escape hatch
reduces the regression compared with pure latency-bounded chunking, but it does
not restore baseline behavior quickly enough. This result is retained as a
design boundary, not presented as a universal improvement.

## Dynamic-Shape Compilation Finding

The first mixed online run exposed repeated `torch.compile` recompilation in the
current RMSNorm implementation. New dynamic token and rank shapes produced
second-scale cold outliers and polluted scheduler cost estimates. Scheduler A/B
tests therefore disable TorchDynamo while retaining CUDA Graph. This finding is
the concrete motivation for replacing compiled Add+RMSNorm with a shape-stable
Triton kernel.

## Next Iteration

The next scheduler version should estimate queued prefill work, not only request
count. It should enter throughput mode using predicted backlog drain time and
leave that mode with hysteresis. Poisson and bursty workloads must use separate
seeds and repeated runs before the policy is considered stable.
