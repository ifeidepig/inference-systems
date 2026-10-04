# Hybrid State-Aware Scheduler

## Scope

This is an opt-in control-plane policy for the Qwen3.5 hybrid runtime. It does
not replace the KV allocator, change Prefix Cache ownership, or add CPU KV
offload. It answers two narrower questions:

1. Which waiting request should be admitted next when several candidates are
   already present?
2. Which running request should be preempted when decode needs physical KV
   pages?

The defaults remain `fcfs` admission and `lifo` preemption. This preserves the
original nano-vLLM behavior unless both policies are explicitly enabled.

## Cache-aware admission

The scheduler examines at most the first `W` waiting requests. Every probe is
read-only: it may inspect the Prefix Cache and GDN checkpoint index, but it may
not allocate a page, update LRU order, count a hit, observe demand, or publish a
checkpoint promotion.

For request `i`:

```text
effective_reusable_i = longest jointly restorable KV/GDN boundary
uncached_i           = prompt_tokens_i - effective_reusable_i
score_i              = uncached_i - aging_tokens_per_ms * age_ms_i
```

The joint boundary is essential for Qwen3.5. A 496-token Full-Attention KV
candidate with only a 256-token recurrent/conv checkpoint is scored as 256
reusable tokens, not 496. A KV-only hit cannot skip the missing GDN state.

The minimum-score feasible request is selected, subject to four guards:

- bounded lookahead: candidates outside the first `W` positions cannot jump;
- hysteresis: a reordering must save at least `min_saved_tokens` when the FCFS
  head is feasible;
- hard aging deadline: an overdue request wins over cache affinity;
- sticky work: chunked-prefill continuations and preempted requests at the head
  are not bypassed.

These guards prevent a stream of hot prefixes from starving cold requests and
avoid reordering for negligible savings.

## Recompute-aware preemption

The old policy preempted the newest remaining running request. The opt-in policy
instead computes:

```text
recompute_tokens = committed_tokens - durable_reusable_boundary
reclaimable      = count(unique physical pages with ref_count == 1)
cost             = recompute_tokens / reclaimable
                   + repeated_preemption_penalty * preemption_count
```

`reclaimable` is deliberately not `len(block_table)`. A shared Prefix Cache page
with `ref_count > 1` remains resident after one request releases its reference,
so it cannot resolve the current allocation failure. Candidates that release no
physical pages receive infinite cost.

For a hybrid request, the durable boundary again means the KV/GDN intersection.
Mutable active GDN state is freed on preemption; only an independently published
checkpoint counts as reusable recovery state.

## Configuration

```text
--waiting-admission-policy hybrid_state_aware
--preemption-policy recompute_aware
--hybrid-scheduler-candidate-window 8
--hybrid-scheduler-aging-tokens-per-ms 0.5
--hybrid-scheduler-max-wait-ms 200
--hybrid-scheduler-min-saved-tokens 16
--hybrid-scheduler-preemption-penalty 128
--hybrid-scheduler-score-source joint
--enable-scheduler-profiling
```

Ablation-only switches disable aging, hysteresis, or sticky recovery without
duplicating scheduler implementations. Detailed decision histories and latency
samples are enabled only by `--enable-scheduler-profiling`; aggregate counters
remain available without profiling.

These flags are exposed by `nanovllm-serve` and `benchmark_online.py`.

## Metrics

Scheduler metrics include candidate probes, reorders, aging overrides, selected
reusable tokens, estimated saved prefill tokens, probe time, maximum observed
wait, selected reclaimable pages, estimated recompute tokens, avoided recompute
relative to LIFO, and victim-selection time. Per-request metrics include bypass
count, maximum consecutive bypasses, selected reusable tokens, preemption count,
and estimated recompute tokens lost.

The validation version additionally reports selected KV candidate, GDN
checkpoint and final joint boundaries separately; original candidate rank;
hysteresis/sticky/starvation counts; actual scheduled prefill/reused tokens;
victim computed/logical/reclaimable/recompute values; decision latency
distributions; and bounded per-decision traces. See
[serving-benchmark.md](serving-benchmark.md) for the formal matrix.

## Correctness and fairness coverage

The tests cover:

- side-effect-free Hybrid Prefix Cache probes;
- bounded Top-W selection and deterministic tie-breaking;
- hysteresis, hard aging, and sticky continuation/recovery;
- KV-only versus jointly restorable hybrid boundaries;
- shared physical pages excluded from reclaimable capacity;
- recompute-aware victim choice and legacy LIFO parity;
- shared-prefix, unique-prefix and repeated-hot fairness traces.

Run the focused suite:

```bash
PYTHONPATH=. pytest -q \
  tests/test_hybrid_state_aware_scheduler.py \
  tests/test_hybrid_state_scheduler_benchmark.py \
  tests/test_hybrid_prefix_cache.py \
  tests/test_scheduler.py \
  tests/test_qwen35_scheduler_state.py
```

## Control-plane experiment

`benchmark_hybrid_state_scheduler.py` executes the exact admission and
preemption selectors against deterministic cache snapshots. It excludes GPU
time by design and therefore must not be reported as an end-to-end speedup.

```bash
PYTHONPATH=. python benchmark_hybrid_state_scheduler.py \
  --output benchmark_results/hybrid_state_scheduler_control_plane.json
```

The 2026-10-04 local run used `W=8`, a 200 ms aging deadline, 0.5 aging
tokens/ms, 16-token hysteresis, a synthetic prefill rate of 4 tokens/ms and an
8 ms decode tail:

| Trace | FCFS queue P95 | Aware queue P95 | Reorders | Boundary demonstrated |
| --- | ---: | ---: | ---: | --- |
| Shared prefix | 344 ms | 220 ms | 3 | cache locality helps |
| Multi-session | 432 ms | 244 ms | 5 | competing hot sessions |
| Unique prompt | 1036 ms | 1036 ms | 0 | no-reuse negative/control case |
| KV-pressure-shaped | 856 ms | 408 ms | 3 | shorter effective work admitted first |
| Multi-turn-shaped | 840 ms | 600 ms | 3 | longer session history becomes reusable |

Total work and makespan are identical in this single-server cost model; the
policy changes ordering and queue latency, not model FLOPs already eliminated
by the cache. The synthetic preemption matrix selected 1,152 estimated replay
tokens versus 4,224 for LIFO across four scenarios. These numbers validate the
policy mechanics only.

### Local 0.8B smoke gate

A single greedy smoke A/B was also run on the RTX 3060 Laptop GPU with the
official Qwen3.5-0.8B-Base BF16 checkpoint, Torch GDN backend, eager execution,
six simultaneous 560-token prompts, one cold request followed by five requests
from a seeded 496-token shared-prefix group, `max_num_seqs=2`, 16 KV pages and a
64 MiB BF16 checkpoint budget. Both policies produced the same output-token
digest. The aware path reordered one request, recorded one real bypass, and
spent 3.21 ms probing across the run.

| Metric | FCFS/LIFO | State-aware | Change |
| --- | ---: | ---: | ---: |
| TTFT P50 | 7442 ms | 7249 ms | -2.6% |
| TTFT P95 | 8618 ms | 8428 ms | -2.2% |
| Queue P95 | 8062 ms | 7791 ms | -3.4% |
| Request throughput | 0.674 req/s | 0.693 req/s | +2.9% |

This remains a historical functional smoke gate, not the final result. The
five-run matrix below supersedes it.

## End-to-end experiment matrix

The formal GPU A/B uses `benchmark_scheduler_matrix.py`, which invokes the same
`benchmark_online.py` path for every configuration. It fixes request traces,
sampling and cache settings, randomizes fresh-engine configuration order, and
stores five raw runs before aggregation.

Common Qwen3.5 options:

```bash
COMMON="--model /path/to/Qwen3.5-0.8B-Base \
  --enable-hybrid-prefix-cache \
  --prefix-match-unit 16 \
  --hybrid-prefix-checkpoint-memory-mib 128 \
  --hybrid-prefix-checkpoint-dtype bf16 \
  --hybrid-prefix-checkpoint-interval-tokens 256 \
  --hybrid-prefix-retention-policy adaptive \
  --hybrid-prefix-eviction-policy cost_aware \
  --prefix-seed-requests 2 \
  --max-num-seqs 4"
```

`max_num_state_slots` is set automatically to `max_num_seqs` by this benchmark.
Suggested workloads:

| Workload | Main controls | Expected use |
| --- | --- | --- |
| Shared prefix | ratio 0.75-1.0, one group, 496-token prefix | positive case |
| Multi-session | ratio 0.75, 2-8 groups | locality/fairness trade-off |
| Unique prompt | ratio 0 | probe overhead and no-reorder control |
| KV pressure | low `max-num-kvcache-blocks`, longer outputs | preemption cost |
| Multi-turn proxy | one group, increasing prompt lengths in separate traces | session replay |

Primary metrics are TTFT P50/P95, maximum queue time, TPOT/max ITL, request and
token throughput, prefix hit/alignment loss, preemption count, replay estimate,
reorders, aging overrides, maximum bypass count, and scheduler probe time. A
policy is not accepted if token output diverges, cold-request starvation occurs,
or scheduler overhead erases the saved prefill time.

## Evidence boundary

The completed 0.8B matrix contains 35 main baseline/full comparisons, 60
ablation comparisons and five focused victim-choice comparisons with no greedy
digest mismatches. Results are workload-dependent: Low pressure is positive;
unique prompts are neutral; multi-session, multi-turn and High pressure expose
no-gain or negative trade-offs. The data does not generalize to 9B, TP/NCCL or
a fused GDN prefill backend. Full tables and causal analysis are in
[serving-benchmark.md](serving-benchmark.md).

## Related upstream designs

- SGLang's scheduler policy includes cache-aware prefix ordering and token-based
  aging: <https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_policy.py>
- vLLM cache-affinity proposal discusses maximum wait and sticky preempted work:
  <https://github.com/vllm-project/vllm/issues/42185>
- vLLM bounded-lookahead proposal emphasizes side-effect-free snapshots,
  remaining work, aging and hysteresis:
  <https://github.com/vllm-project/vllm/issues/53277>
- Hybrid checkpoint-aware scheduling motivation and preliminary evidence:
  <https://github.com/vllm-project/vllm/issues/57111>
