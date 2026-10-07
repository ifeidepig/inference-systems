# ReplaySSM Research and Compact Replay Prototype

## Scope and naming

This repository now contains an opt-in **compact speculative replay** path for
Qwen3.5 GDN MTP verification. It implements the memory/correctness half of the
ReplaySSM idea:

```text
full recurrent snapshots per verify token
-> compact (normalized key, delta, log_decay) records
-> accepted-prefix fold into the active checkpoint
```

It does **not** claim to implement the full output-only, periodic-flush
ReplaySSM verify kernel. The full algorithm reconstructs GDN readouts directly
from a frozen checkpoint plus a circular ring, advances a cursor on accept, and
writes the full recurrent state only at a later flush boundary. That requires a
new chunked verify kernel, persistent ring cursors and flush/prefix-cache
coordination.

The runtime option is deliberately default-off:

```bash
python benchmark_qwen35_mtp.py \
  --model /path/to/Qwen3.5-0.8B-Base \
  --mode mtp --num-speculative-tokens 2 \
  --cuda-graph --gdn-decode-backend cuda \
  --replay-ssm
```

Server option:

```bash
nanovllm-serve ... --enable-mtp-replay-ssm
```

## Primary-source audit

### ReplaySSM author repository

The paper repository describes a state as one checkpoint plus a bounded input
buffer and lists separate standard/speculative implementations for Mamba2 and
GDN. It also makes clear that, at the current snapshot, only Mamba2 standard
decode is merged upstream; GDN and speculative paths remain in the research
fork or open PRs:

- <https://github.com/Johnny-Liou/ReplaySSM>
- GDN speculative kernel:
  <https://github.com/Johnny-Liou/ReplaySSM/blob/main/vllm/model_executor/layers/fla/ops/gdn_replayssm_spec_decode.py>

The full GDN kernel performs chunked delta-rule verification, writes compact
`d/k/g` records into a circular buffer, and periodically folds them into the
checkpoint. Its state layout, tensor-core reconstruction and vLLM cache/runtime
contract cannot be copied into nano-vLLM as a metadata-only change.

### vLLM RFC

The vLLM tracking RFC defines two modes, both default-off, and calls out the
runtime contract explicitly: accepted-token count, fixed CUDA-graph launch
sequence, ring size `B+T`, early flush, and per-block cursors:

- <https://github.com/vllm-project/vllm/issues/49232>

The published reference numbers show the central performance boundary: at
batch 1 Qwen3.5 GDN standard/speculative ReplaySSM is approximately neutral;
the gains grow at batch 32/256 on H100/B300-class hardware.

### SGLang RFC and implementation

SGLang separates three related optimizations:

1. buffered output-only standard GDN decode;
2. compact speculative cache replay, which removes full intermediate states;
3. a full chunked ReplaySSM verify kernel, which targets bandwidth and speed.

The RFC reports that compact replay reduced one speculative cache example from
2.32 GB to 0.06 GB but produced no throughput change at concurrency 32-64. It
therefore recommends measuring achieved HBM bandwidth before porting the full
kernel: high bandwidth utilization supports the rewrite; low utilization does
not.

- <https://github.com/sgl-project/sglang/issues/28511>
- Memory-pool layout:
  <https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/memory_pool.py>
- Commit/cursor integration:
  <https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/spec_utils.py>

## Current nano-vLLM design

### Verify records

For each GDN layer and verifier token, the reference recurrence already
computes:

```text
k_t          normalized key          [value_heads, key_dim]
delta_t      accepted state update   [value_heads, value_dim]
log_decay_t  log(alpha_t)            [value_heads]
```

The recurrent update is replayed exactly in the same FP32 order:

```text
S <- exp(log_decay_t) * S + outer(k_t, delta_t)
```

The active state pool is not modified during Verify. After greedy verification,
`commit_lengths = state_boundary + 1` acts as the logical cursor. One CUDA
kernel grid over `layers x requests x heads` reads each active recurrent matrix
once, folds only the accepted transitions, and writes one final state.

### Conv state

Replay records do not reconstruct the width-4 short-convolution window. Conv
history remains small, so Verify stores all conv boundaries and a second fused
CUDA kernel selects one accepted window per request across all layers.

### CUDA Graph

The captured Verify graph owns fixed buffers for:

```text
conv_history      [layers, batch*K, conv_dim, 3]
replay_key        [layers, batch*K, heads, key_dim]
replay_delta      [layers, batch*K, heads, value_dim]
replay_log_decay  [layers, batch*K, heads]
```

No full recurrent-history buffer is allocated. Accepted-prefix folding remains
outside the captured Verify graph, matching the previous select/scatter commit
boundary.

## Correctness gates

- compact records reconstruct every independent FP32 prefix boundary;
- fused recurrent fold matches the Torch oracle for batch 1/2/4 and K=1/2/4;
- fused conv commit selects variable boundaries exactly;
- eager and CUDA Graph MTP match target-only for K=1/2/4;
- immediate rejection and mixed accepted lengths are covered;
- official 0.8B, 128-token/request output streams match target-only;
- Compute Sanitizer memcheck reports zero errors.

Focused suite result: 27 passed.

## Memory result

Official Qwen3.5-0.8B BF16, batch 2, MTP-2:

| Verify speculative state | Bytes | MiB |
| --- | ---: | ---: |
| Minimal full-state snapshots | 39,075,840 | 37.27 |
| Conv history | 2,654,208 | 2.53 |
| Compact replay records | 1,184,256 | 1.13 |
| Compact total | 3,838,464 | 3.66 |

The fixed Verify buffer is reduced by 90.18%, or 10.18x. Five-run peak
allocated memory falls by about 34.9 MiB.

## Performance result

RTX 3060, official 0.8B BF16, batch 2, MTP-2, 32 output tokens/request,
CUDA Graph, fused CUDA GDN, five alternating fresh engines per arm:

| Metric | Minimal snapshots | Compact replay | Change |
| --- | ---: | ---: | ---: |
| Throughput median | 151.04 tok/s | 150.05 tok/s | -0.66% |
| Mean TPOT median | 10.55 ms | 10.66 ms | +1.05% |
| Verify GPU median | 14.06 ms | 13.20 ms | -6.12% |
| State commit GPU median | 0.184 ms | 0.406 ms | +0.222 ms |

The end-to-end result is effectively parity with a small negative tendency.
This is expected for a small model and batch 2: compact records eliminate
snapshot traffic, but fold-every-commit still reads and writes the full active
state once per round. The complete output-only ReplaySSM kernel is designed to
amortize that write across several rounds and is most valuable when Verify is
already HBM-bandwidth bound at high serving batch.

Raw summary:

`benchmark_results/mtp_compact_replay_summary_20261007.json`

## Decision

The compact path is retained as an opt-in memory/concurrency prototype. It is
not the default and is not presented as a throughput optimization.

Before implementing the full output-only circular kernel:

1. measure high-concurrency Verify bandwidth on the target cloud GPU;
2. require a workload where recurrent history/state traffic limits
   concurrency or Verify latency;
3. port the early-flush invariant (`buffer_len >= 2 * max_spec_len`);
4. make cursor updates and flush decisions graph-safe and per request slot;
5. coordinate forced flush with Hybrid Prefix checkpoint publication;
6. validate floating-point drift because chunked/tensor-core reconstruction is
   mathematically equivalent but not bit-exact.

