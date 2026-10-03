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
```
