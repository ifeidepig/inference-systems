# Hybrid State-Aware Scheduler: Serving Benchmark

## 1. Problem

FCFS admission ignores work that is already reusable in the Prefix Cache, and
LIFO preemption ignores both recovery cost and whether releasing a request will
actually return physical KV pages to the free queue. This matters more for
Qwen3.5 than for a pure Transformer because a Full-Attention KV hit is usable
only up to a boundary that also has a GDN recurrent/conv checkpoint.

The benchmark asks four questions:

1. When does bounded cache-aware admission improve queueing and TTFT?
2. Which guard supplies fairness, and what does it cost?
3. Does recompute-aware preemption reduce replay per reclaimed physical page?
4. Is the scheduler's CPU cost small relative to serving wall time?

It does not evaluate new kernels, MTP variants, KV offload, or new scheduling
heuristics.

## 2. Baseline

The baseline is the same engine, model, cache, allocator, checkpoint policy and
online runner with:

```text
waiting_admission_policy = fcfs
preemption_policy        = lifo
```

The Full configuration changes only these scheduler controls:

```text
waiting_admission_policy = hybrid_state_aware
preemption_policy        = recompute_aware
score_source             = joint
aging/hysteresis/sticky  = enabled
```

Both configurations therefore share model execution, Prefix Cache population,
Paged KV allocation, GDN checkpoint capture/restore and output sampling.

## 3. Design under test

Admission probes at most the first W waiting requests and computes:

```text
remaining_prefill = prompt_tokens - scoring_reusable_tokens
score             = remaining_prefill - aging_factor * wait_ms
```

For the production policy, `scoring_reusable_tokens` is the longest boundary
present in both Full-Attention KV and GDN checkpoint state. Hysteresis rejects
small reorder gains, a hard wait deadline prevents repeated bypass, and sticky
recovery keeps preempted/chunk-continuation work at the queue head.

Preemption evaluates:

```text
cost = estimated_recompute_tokens / pages_with_ref_count_1
       + repeated_preemption_penalty
```

Shared pages with `ref_count > 1` are not reclaimable because they remain
resident after the victim releases its reference.

## 4. Experimental setup

The local validation target is the official Qwen3.5-0.8B-Base checkpoint on an
RTX 3060 Laptop GPU. The reproducible matrix uses greedy decoding, fixed
request-level prompt seeds, fresh engines, identical warmup and checkpoint
seeding, and randomized configuration order inside every repeat.

Defaults:

```text
dtype                         BF16 checkpoint/model configuration
GDN backend                   Torch correctness path
physical KV page              256 tokens
prefix match unit             16 tokens
GDN checkpoint storage        BF16
checkpoint budget             64 MiB
retention / eviction          adaptive / cost-aware
max batched tokens            256
runs                          5
seeds                         0,1,2,3,4
scheduler profiling           enabled
```

Raw results are stored as one JSON file per fresh-engine run under
`benchmark_results/.../raw`. Logs, command order and return codes are recorded
in `manifest.jsonl`; aggregate JSON/CSV are written under `summary`.
One ablation initialization encountered a transient CUDA OOM while the desktop
left only about 39 MiB free. The failure is retained in the manifest/log; a
resumable rerun skipped all completed JSON files and successfully filled the
missing run.

## 5. Workloads

| Workload | Purpose | Key setup |
| --- | --- | --- |
| shared-prefix | cache-aware positive case | one cold request followed by a seeded 496-token group |
| multi-session | locality among multiple sessions | two independently seeded prefix groups |
| unique-prompt | no-gain negative control | eight unrelated 560-token prompts |
| KV-pressure Low | no-preemption control | four active 256-token prompts, 8 KV pages |
| KV-pressure Medium | moderate shortage | same trace, 6 KV pages |
| KV-pressure High | repeated preemption | same trace, 5 KV pages |
| multi-turn | growing session-history proxy | 320/576/832-token shared-history turns mixed with cold requests |
| victim choice | isolate preemption selector | four concurrent requests; LIFO cold victim versus checkpointed hot victim |

The controlled metadata benchmark additionally forces `KV=5000` and
`GDN=4096`, then compares it against an aligned 4500-token candidate. This is
the clean joint-recovery experiment that cannot be guaranteed by a stochastic
online cache state.

## 6. Metrics

End-to-end metrics include TTFT, TPOT and request latency mean/P50/P95/P99;
input/output token throughput; request throughput; total makespan; peak GPU
allocation; and maximum active/waiting/KV usage.

Mechanism metrics include admission count, candidate count, original selected
rank, reorder count/rate, KV candidate/GDN checkpoint/joint boundary tokens,
remaining and actual prefill tokens, sticky/hysteresis/aging/starvation counts,
preemptions, victim computed/recompute tokens, logical versus reclaimable
blocks, and cost per reclaimed page.

With scheduler profiling enabled, bounded decision histories record every
admission and preemption decision. CPU latency summaries report mean,
P50/P95/P99/max for admission probes and victim selection, plus scheduler CPU
time as a fraction of serving wall time. Detailed histories are off by default
in serving.

## 7. Ablation

The runner exposes one shared implementation with the following configurations:

| Name | Admission | Score | Guards | Preemption |
| --- | --- | --- | --- | --- |
| baseline | FCFS | n/a | n/a | LIFO |
| topw_kv_only | Top-W | KV-only | off | LIFO |
| topw_joint | Top-W | KV/GDN joint | off | LIFO |
| joint_aging | Top-W | joint | aging only | LIFO |
| guarded_admission | Top-W | joint | aging+hysteresis+sticky | LIFO |
| preemption_only | FCFS | n/a | n/a | recompute-aware |
| full | Top-W | joint | all | recompute-aware |

`kv_only` changes ranking only. Actual model restoration still uses the safe
joint coordinator plan, so the ablation cannot corrupt model state.

## 8. End-to-end results

The stable five-run tables are generated from:

```text
benchmark_results/scheduler_formal_main_20261004/summary/summary.json
benchmark_results/scheduler_formal_ablation_20261004/summary/summary.json
benchmark_results/scheduler_victim_choice_formal_20261004/summary/summary.json
```

All 35 main-matrix baseline/full digest comparisons matched. Values below are
the median of five fresh-engine runs; deltas are Full relative to baseline.

| Workload | TTFT P50 | TTFT P95 | TTFT P99 | TPOT P95 | Req/s | Actual prefill tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Shared prefix | +0.9% | +0.8% | +0.4% | +26.5% | -0.4% | 1456 -> 1441 |
| Multi-session | +3.9% | +3.2% | +3.2% | -0.1% | -3.0% | 2112 -> 2162 |
| Unique prompt | -0.4% | +0.4% | +0.3% | -0.2% | -0.3% | 4743 -> 4743 |
| KV pressure Low | -67.2% | -31.8% | -30.4% | -1.6% | +41.6% | 1568 -> 1088 |
| KV pressure Medium | -62.7% | -18.5% | -14.0% | +39.5% | +13.8% | 1568 -> 1345 |
| KV pressure High | -48.6% | -1.4% | +2.6% | +75.8% | -2.1% | 1572 -> 1606 |
| Multi-turn | -13.3% | +9.0% | +9.9% | +50.7% | -9.2% | 3424 -> 3680 |

The policy is therefore not a universal throughput improvement. The strongest
positive case is Low pressure, where reordering raises actual reused tokens
from 480 to 960 and removes 480 prefill tokens. High pressure improves early
TTFT but harms inter-token latency and throughput. Unique prompts are the
expected no-gain control.

### Focused victim-choice result

A separate High-pressure trace admits four requests together and orders the
running deque so LIFO sees a cold victim while another request has a durable
joint checkpoint. Across five fresh-engine runs:

| Metric | LIFO baseline | Recompute-aware only | Delta |
| --- | ---: | ---: | ---: |
| Preemptions | 3 | 3 | 0 |
| Reclaimable physical pages | 3 | 3 | 0 |
| Estimated recompute tokens | 256 | 0 | -100% |
| TTFT P95 | 18605 ms | 18126 ms | -2.6% |
| TPOT P95 | 899 ms | 878 ms | -2.4% |
| Makespan | 19869 ms | 19276 ms | -3.0% |
| Request throughput | 0.403 req/s | 0.415 req/s | +3.1% |

The measured scheduled-prefill token total was unchanged, so the E2E gain
cannot be attributed solely to token-count reduction; the victim choice also
changes batching and restore timing. The safe claim is that the policy removed
the estimated recovery loss and improved this controlled trace, not that every
preemption workload becomes faster.

## 9. Mechanism analysis

The final analysis follows this chain for each workload:

```text
decision change
  -> reused/recomputed tokens and reclaimed pages
  -> observed prefill/decode work
  -> TTFT/TPOT/throughput
```

High-pressure Full exposes an important interaction: admission changes the
running set before memory pressure occurs. Baseline LIFO happens to select
requests with durable recovery, while guarded/full admission selects a set that
accumulates 512 estimated replay tokens. Both reclaim four pages, but Full runs
34 more prefill tokens and loses about 2-4% throughput. This is why
`preemption_only` is reported separately from Full.

The unguarded Top-W variants are worse: High-pressure prefill work increases
by about 52%, preemptions rise from four to five or six, starvation events reach
17-21, and throughput drops about 33%. Enabling sticky recovery and hysteresis
returns starvation to zero and bounds the additional work. The guards are not
cosmetic; they prevent reorder/preempt/re-admit churn.

For shared-prefix traffic, KV-only and joint scoring execute almost the same
work in this particular online trace, but KV-only performs one extra reorder.
The deterministic boundary ablation supplies the stronger semantic case:
KV=5000 with GDN=4096 is overestimated by 904 tokens and wins under KV-only;
joint scoring correctly selects the alternative 4500-token aligned request.

## 10. Scheduler overhead

Across the five-run main matrix, Full admission decision medians are usually
2-3 us because sticky decisions dominate. Admission P95 ranges from 98 us on
unique prompts to 547 us on multi-session; probe P95 reaches 650 us in that
deepest trace. Recompute-aware victim-selection P95 is 54-111 us when pressure
actually triggers it. Total scheduler CPU time is 1.2-5.2 ms per run and remains
between 0.005% and 0.038% of serving wall time. Baseline FCFS admission P95 is
about 2 us.

## 11. Negative and no-gain cases

- Unique prompts should produce no effective reorder; any latency change is
  scheduler overhead plus run-to-run GPU variance.
- A shallow queue gives Top-W no alternative candidate.
- A KV-only long hit with an earlier GDN checkpoint can rank the wrong request.
- Admission and preemption can interact negatively by changing which requests
  are resident when pressure arrives.
- A victim with many logical but shared pages may reclaim little memory.

These cases are retained rather than tuned away.

## 12. Limitations

- Local results cover one 0.8B model and one RTX 3060 Laptop GPU.
- Torch GDN prefill dominates wall time and is not representative of a fused
  production prefill backend.
- The multi-turn trace is a deterministic history-growth proxy, not a complete
  agent serving distribution.
- CPU timing uses `perf_counter_ns`; profiling is bounded but not free.
- TP/NCCL and the 9B model require the external compute platform.

## 13. Reproduction commands

Main paired matrix:

```bash
PYTHONPATH=. TORCHDYNAMO_DISABLE=1 python benchmark_scheduler_matrix.py \
  --model /path/to/Qwen3.5-0.8B-Base \
  --output-root benchmark_results/scheduler_formal_main \
  --runs 5 --seeds 0 1 2 3 4 \
  --configs baseline full \
  --workloads shared_prefix multi_session unique_prompt kv_pressure multi_turn \
  --pressure-levels low medium high
```

Focused ablation:

```bash
PYTHONPATH=. TORCHDYNAMO_DISABLE=1 python benchmark_scheduler_matrix.py \
  --model /path/to/Qwen3.5-0.8B-Base \
  --output-root benchmark_results/scheduler_formal_ablation \
  --runs 5 --seeds 0 1 2 3 4 \
  --configs all \
  --workloads shared_prefix kv_pressure \
  --pressure-levels high
```

Targeted victim choice:

```bash
PYTHONPATH=. TORCHDYNAMO_DISABLE=1 python benchmark_scheduler_matrix.py \
  --model /path/to/Qwen3.5-0.8B-Base \
  --output-root benchmark_results/scheduler_victim_choice \
  --runs 5 --seeds 0 1 2 3 4 \
  --configs baseline preemption_only \
  --workloads victim_choice --pressure-levels high
```

Control-plane joint-boundary and victim traces:

```bash
PYTHONPATH=. python benchmark_hybrid_state_scheduler.py \
  --output benchmark_results/hybrid_state_scheduler_control_plane.json
```

Re-aggregate existing raw files:

```bash
python aggregate_scheduler_results.py \
  --raw-dir benchmark_results/scheduler_formal_main/raw \
  --summary-dir benchmark_results/scheduler_formal_main/summary \
  --require-digest-match
```
