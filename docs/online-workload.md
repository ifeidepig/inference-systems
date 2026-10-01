# Online Workload And Request Telemetry

## Purpose

`benchmark_online.py` replays requests over time instead of submitting one fixed
batch. It provides the workload and measurements needed to compare scheduling
policies under queueing, burst traffic, mixed request sizes, and shared prefixes.

## Arrival Patterns

- `poisson`: inter-arrival times are sampled from an exponential distribution.
- `bursty`: `burst_size` requests arrive together, with burst intervals selected
  to preserve the configured average request rate.

The random seed makes request shapes, prefix assignment, and Poisson arrivals
reproducible. A request keeps its planned arrival timestamp even if it becomes
due while a synchronous GPU step is running. This makes event-loop admission
delay visible instead of silently excluding it from TTFT.

## Request Shapes

Prompt and output lengths are sampled independently from the configured lists.
Requests selected for a shared-prefix group start with the same token sequence
and end with a request-specific suffix. This preserves full KV-cache blocks that
can be reused while preventing every prompt from being identical.

## Lifecycle Metrics

Each `Sequence` records:

- `admission_delay_ms`: engine admission time minus planned arrival time.
- `queue_ms`: first scheduler selection minus planned arrival time.
- `ttft_ms`: first generated token minus planned arrival time.
- `tpot_ms`: mean gap between generated tokens.
- `max_token_gap_ms`: largest observed gap between generated tokens.
- `e2e_ms`: request completion minus planned arrival time.
- `schedule_count`: number of prefill or decode scheduler selections.
- `preemption_count`: number of times KV blocks were released for recomputation.
- `prefix_cache_hit_blocks`: full prompt blocks reused by this request.
- `peak_kv_blocks`: maximum physical KV blocks referenced by this request.

Completed lifecycle snapshots are retained in a bounded in-memory history and
exposed by `/metrics/requests`. The history size is controlled by
`request_metrics_history_size` and defaults to 1024 entries.

## SLO Definitions

- A TTFT violation occurs when `ttft_ms > target_ttft_ms`.
- A token-latency violation occurs when `max_token_gap_ms > target_tpot_ms`.
- A request violation occurs when either condition is true.

The maximum token gap is used for the token SLO because a good mean TPOT can hide
one long decode stall caused by an interfering prefill.

## Example

```bash
python benchmark_online.py \
  --model /YOUR/MODEL/PATH \
  --num-requests 24 \
  --request-rate 8 \
  --arrival-pattern poisson \
  --prompt-lengths 64 256 768 \
  --output-lengths 8 32 \
  --shared-prefix-ratio 0.5 \
  --shared-prefix-length 512 \
  --scheduling-policy decode_first \
  --use-cuda-graph
```

The JSON output contains the generated workload, per-request lifecycle metrics,
short/medium/long prompt breakdowns, aggregate SLO violation rates, runtime
metrics, and a per-step queue/KV trace.
