# Shared-Junction Promotion Threshold Study

## Semantics

The public setting is:

```text
hybrid_prefix_promotion_min_sightings = 2
```

It counts total sightings, not only KV-only misses:

```text
total_sightings = 1 producer + kv_only_observations
```

Threshold 2 preserves the Marconi-style behavior: A produces resident KV, B
observes `KV hit / GDN miss` and captures the junction, and C reuses it.
Threshold 3 waits through B, captures on C, and first benefits D.

Values below 2 are rejected. An eager first-sighting policy cannot discover an
arbitrary future shared junction without dense retention or external semantic
boundary information, so it is a separate control rather than the same demand
policy with an integer set to one.

## New metrics

The runtime now exposes:

- current and peak checkpoint occupancy, including per-reason occupancy;
- published promotions and promotions not yet reused;
- useful promotions (first subsequent hit) and useful-promotion ratio;
- shared-junction hit count and saved replay tokens;
- alignment/replay tokens caused by a missing checkpoint;
- shared-junction evictions and never-used promotion evictions.

## Workloads

`benchmark_promotion_threshold.py` generates deterministic traces with unique
suffixes and the following prefix-frequency distributions:

```text
singleton: each prefix appears once
pair:      each prefix appears twice
triple:    each prefix appears three times
hot:       a small set appears N times
zipf:      sampled 1/rank popularity
```

The benchmark rejects a trace when the prompt-tail checkpoint lands at or
before the shared boundary, because that workload would hit an existing GDN
checkpoint and would not exercise demand promotion.

## Local functional matrix

Official Qwen3.5-0.8B, BF16 model, INT8 checkpoint storage, 64 MiB checkpoint
budget, 64-token shared prefix, 32-token unique suffix, one warmup request, one
output token. These are single local runs for policy validation, not stable
serving performance claims.

| Workload | Threshold | Promotions | Useful | Useful ratio | Alignment replay | Saved replay | Unused resident |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| singleton x4 | 2 | 0 | 0 | 0% | 0 | 0 | 0 |
| singleton x4 | 3 | 0 | 0 | 0% | 0 | 0 | 0 |
| pair x4 | 2 | 4 | 0 | 0% | 256 | 0 | 4 |
| pair x4 | 3 | 0 | 0 | 0% | 256 | 0 | 0 |
| triple x4 | 2 | 4 | 4 | 100% | 256 | 256 | 0 |
| triple x4 | 3 | 4 | 0 | 0% | 512 | 0 | 4 |
| hot x8 | 2 | 1 | 1 | 100% | 64 | 384 | 0 |
| hot x8 | 3 | 1 | 1 | 100% | 128 | 320 | 0 |
| Zipf 12/3/1 | 2 | 2 | 2 | 100% | 128 | 704 | 0 |
| Zipf 12/3/1 | 3 | 2 | 1 | 50% | 256 | 576 | 1 |

Every threshold pair produced identical token digests.

For the hot prefix, threshold 2 reached the reusable regime on occurrence 3
(TTFT about 188 ms), while threshold 3 still replayed occurrence 3 (about
478 ms) and first reused on occurrence 4 (about 191 ms). Reversing threshold
execution order produced the same transition pattern, reducing warmup-order
confounding.

A separate 496-token shared-prefix gate verified threshold 3 end to end:

```text
request 1: producer
request 2: KV-only observer, no promotion
request 3: promoter, publishes CP@496
request 4: committed hit 496
```

## Interpretation

- Threshold 2 does not pollute singleton traffic because a resident matching
  KV prefix is required before promotion.
- Pair-only traffic is its negative case: the second and final request creates
  a checkpoint that never receives a later hit.
- Triple and hot traffic favor threshold 2 because it begins reuse one request
  earlier.
- In the measured Zipf trace, threshold 2 had higher promotion usefulness and
  lower replay than threshold 3.

The current evidence supports keeping 2 as the default. It does not prove that
2 is universal: a pair-heavy workload under tighter checkpoint pressure may
favor 3. Larger repeated traces, longer prefixes, and 9B/TP experiments remain
necessary before making a production-wide claim.

## Long-horizon control-plane sweep

The production `GDNCheckpointManager` and cost-aware eviction policy were then
replayed over 5 seeds for every combination of:

```text
Pair:   128 prefixes x 2 requests
Triple: 128 prefixes x 3 requests
Hot:    32 prefixes x 64 requests
Zipf:   64 prefixes / 1024 requests
Prefix: 256 / 496 / 1024 / 2048
Threshold: 2 / 3
```

This layer measures admission, occupancy, eviction, churn, and replay using the
real manager but assumes Full-Attention KV remains resident; it does not report
model TTFT. Two fixed-budget sensitivities were used: FP32-equivalent capacity
6 and INT8-equivalent capacity 24.

At prefix length 496:

| Traffic | Capacity | Threshold 2 | Threshold 3 |
| --- | ---: | --- | --- |
| Pair | 6 | 123 unused evictions; churn 980/1K | no promotion |
| Triple | 6 | useful 1.6%; churn 1299/1K | useful 0%; churn 654/1K |
| Hot | 6 | net replay saved -695K | net replay saved -699K |
| Zipf | 6 | net replay saved -93K | net replay saved -109K |
| Hot | 24 | net replay saved +408K | net replay saved +388K |
| Zipf | 24 | net replay saved +162K | net replay saved +158K |

With only six slots, unique prompt-tail checkpoints and shared junctions create
heavy churn and both thresholds can lose overall. Capacity 24 changes Hot/Zipf
to positive net replay savings. Compression therefore changes the operating
region of admission policy, but storage quality must be validated separately.

Across 256/496/1024/2048 prefixes the sign of each fixed-trace result stayed
stable while the absolute threshold-2 replay advantage grew approximately
linearly with prefix length.

## Five-seed real-model stability

The primary serving study fixes checkpoint storage to FP32 so admission is not
confounded by quantization. Every seed uses different prompt tokens, alternates
threshold execution order, performs a shape-matched warmup, and reports
median/IQR plus pooled request P50/P95.

For one hot prefix repeated eight times, occurrence 3 is the controlled policy
transition: threshold 2 already restores, while threshold 3 performs its final
cold replay.

| Shared prefix | Threshold 2 occurrence-3 TTFT | Threshold 3 occurrence-3 TTFT |
| ---: | ---: | ---: |
| 256 | 333 ms | 1,487 ms |
| 496 | 339 ms | 2,602 ms |
| 1024 | 354 ms | 5,113 ms |
| 2048 | 341 ms | 9,945 ms |

All 20 hot prefix-length/seed threshold pairs were token exact with FP32
checkpoints. Overall trace medians remained close because later hot-prefix hits
amortize one extra miss; occurrence-specific TTFT exposes the admission cost.

At prefix 496, five-seed workload cross-checks showed:

- Pair: threshold 2 published two checkpoints with zero later reuse; threshold
  3 published none. Pooled P50 was about 2.62 s for both.
- Triple: threshold 2 useful ratio was 100% and saved 992 replay tokens per
  trace; threshold 3 saved zero because promotion occurred on the final use.
- Zipf: seed-level trace TTFT median was about 342 ms (IQR 340-344) for
  threshold 2 versus 1,471 ms (IQR 1,467-1,474) for threshold 3. Alignment
  replay was 992 versus 1,984 tokens.

All Pair/Triple/Zipf FP32 threshold pairs were token exact.

## INT8 negative quality result

The first stability run used INT8 checkpoints. For prefix 256, 2 of 5 seeds
changed the occurrence-3 greedy token exactly when threshold 2 restored INT8
state and threshold 3 still used cold prefill. Repeating the same seed with
FP32 and BF16 checkpoints was token exact, isolating checkpoint quantization as
the cause rather than admission policy.

Therefore INT8 remains opt-in and cannot currently be presented as quality-safe
based on the earlier single 1024-token prompt. The policy performance evidence
uses FP32. This negative result also blocks making INT8 capacity numbers a
resume performance claim until broader quality work is completed.

## Reproduction

```bash
python benchmark_promotion_threshold.py \
  --model /path/to/Qwen3.5-0.8B-Base \
  --workload zipf \
  --thresholds 2 3 \
  --num-prefixes 4 \
  --zipf-requests 16 \
  --shared-prefix-length 64 \
  --unique-suffix-length 32 \
  --checkpoint-memory-mib 64 \
  --checkpoint-dtype int8

python benchmark_promotion_policy_simulation.py \
  --seeds 5 \
  --num-prefixes 128 \
  --hot-frequency 64 \
  --zipf-requests 1024 \
  --checkpoint-capacity 6

python benchmark_promotion_stability.py \
  --model /path/to/Qwen3.5-0.8B-Base \
  --workloads hot \
  --prefix-lengths 256 496 1024 2048 \
  --thresholds 2 3 \
  --seeds 5 \
  --checkpoint-dtype fp32
```
