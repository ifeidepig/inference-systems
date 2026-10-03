# Hybrid Prefix Checkpoint Compression

## Scope

The active Qwen3.5 recurrent state remains FP32 and the convolution history
remains in the model dtype. Only immutable cached prefix checkpoints are
compressed. The pool supports:

```text
hybrid_prefix_checkpoint_dtype = fp32 | bf16 | int8
```

INT8 checkpoints use symmetric per-`(layer, head, key-channel)` scales. The
active layout is `[L, slot, H, K, V]`, so quantization reduces over `V` and
stores FP32 scales with shape `[slot, L, H, K, 1]`. Conv history remains in its
native dtype.

## Storage

For official Qwen3.5-0.8B BF16 / TP1:

| Format | Bytes/checkpoint | Capacity at 128 MiB | Allocated pool bytes |
| --- | ---: | ---: | ---: |
| FP32 | 19,537,920 | 6 | 117,227,520 |
| BF16 | 10,100,736 | 13 | 131,309,568 |
| INT8 + FP32 scale | 5,529,600 | 24 | 132,710,400 |

INT8 therefore provides 4x the checkpoint count of the current FP32 baseline
under the same 128 MiB budget. The whole-checkpoint byte reduction is 3.53x,
because FP32 scales and native BF16 conv state remain.

## Lifecycle and transactions

All formats implement the same pool contract:

- `capture` and sparse-internal `capture_tensors`;
- one-time dequantization into an FP32 request-owned active slot on `restore`;
- zero-on-`free` and zero-on-`rollback_capture`, including INT8 scale state;
- deterministic slot allocation and existing all-rank slot consensus;
- restore rollback to the pre-transaction private request state.

The checkpoint dtype does not change scheduler metadata, prefix hashes, COW,
CUDA Graph active-state addresses, or TP state sharding.

## Correctness evidence

The CPU suite covers FP32/BF16/INT8 capacity, storage dtype, round-trip error,
zero-state scale handling, sparse internal capture, free/rollback, and a
two-process Gloo INT8 transaction with injected rank failure.

A teacher-forced synthetic gated-delta recurrence compares FP32 with BF16 and
INT8 after 1, 32, 128, and 1024 continuation steps. The test requires finite
state, max state error below `1e-3`, and max recurrent-output error below
`3e-4` for its controlled stable-decay workload.

Official Qwen3.5-0.8B A/B/C validation used a 496-token shared prefix and
64-token unique suffix:

```text
A: prompt-tail producer
B: KV-only observation -> publish CP@496
C: restore CP@496 -> generate 1024 tokens
```

FP32, BF16, and INT8 generated the same 1024 greedy tokens for C. All formats
committed a 496-token hit. This is a deterministic regression gate, not a
general long-context quality result; LongBench/Needle remains pending.

The INT8 path also passed the CUDA Graph A/B/C route. Split-checkpoint FP32 and
INT8 runs matched each other for A, B, and C; comparisons are always made
within the same split/internal execution mode because those prefill modes have
different BF16 reduction boundaries.

## Performance evidence

Local RTX 3060, official 0.8B, BF16, eager, 20-iteration CUDA-event
microbenchmark after three warmups:

| Format | Capture median | Restore median | Peak temporary bytes |
| --- | ---: | ---: | ---: |
| FP32 | 0.152 ms | 0.132 ms | 0 |
| BF16 | 0.123 ms | 0.113 ms | 0 |
| INT8 | 0.720 ms | 0.351 ms | 38,043,648 |

In the single-run 1024-token A/B/C gate, C TTFT was approximately 347 ms
(FP32), 334 ms (BF16), and 333 ms (INT8); TPOT was approximately 37.3, 35.8,
and 36.5 ms. These single-run values establish that compression did not erase
the prefix-hit benefit, but they are not stable comparative performance claims.

The INT8 prototype currently uses separate PyTorch `amax/div/round/cast` and
dequantization operations. Its 38 MB transient allocation and extra launches
are the remaining optimization target. A fused CUDA quantize/dequantize path
should only be prioritized after repeated serving workloads show checkpoint
transfer on the critical path.

## Reproduction

```bash
python benchmark_checkpoint_compression.py \
  --memory-budget-mib 128 \
  --iterations 20 \
  --warmup 3

python benchmark_adaptive_prefix_promotion.py \
  --model /path/to/Qwen3.5-0.8B-Base \
  --shared-prefix-length 496 \
  --unique-suffix-length 64 \
  --setup-output-tokens 1 \
  --output-tokens 1024 \
  --checkpoint-memory-mib 128 \
  --checkpoint-dtype int8
```

## Evidence boundary

- No CUDA quantization kernel has been implemented yet.
- No LongBench/Needle score is claimed.
- Real 9B and NCCL multi-GPU latency remain external-platform work.
- The public default remains FP32; BF16/INT8 are explicit opt-in choices.
