# Qwen3.5-0.8B Local Validation Report

Date: 2026-09-27

> Historical first integration gate. The correctness results remain useful,
> but the MTP performance section predates final-alignment LM-head removal,
> selective rollback, and minimal `K-1` rollback-history capture. Current MTP
> performance evidence is maintained in `docs/mtp-phase-profiling.md`.

This is a same-architecture stepping-stone validation for the Qwen3.5-9B
project. It is not a substitute for the final 9B cloud gate.

## Environment

```text
GPU: NVIDIA GeForce RTX 3060 Laptop GPU, 6 GiB
Python: 3.11.15
PyTorch: 2.8.0+cu128
Model: official Qwen3.5-0.8B-Base, BF16
Architecture: 24 layers = 18 GDN + 6 Full Attention
```

All authoritative commands use absolute script paths plus:

```bash
PYTHONSAFEPATH=1 PYTHONPATH=.
```

The environment's `sitecustomize` otherwise changes the working directory to
another editable nano-vLLM checkout.

## Checkpoint and Hugging Face golden

Official checkpoint audit:

```text
checkpoint keys: 488
target parameters: 297
explicit Vision/MTP skips for target: 168
target missing/unexpected: 0/0
MTP checkpoint keys: 15 -> 14 packed internal parameters
MTP missing/unexpected: 0/0
```

The HF fixture and nano validation are reproducible with:

```bash
python tests/validate_qwen35_hf_golden.py \
  --model /path/to/Qwen3.5-0.8B-Base \
  --fixture /tmp/qwen35-08b-validation.pt \
  --mode both \
  --decode-steps 8 \
  --num-speculative-tokens 2 \
  --cuda-graph
```

Results:

```text
HF greedy:   198, 262, 220, 16, 15, 15, 15, 15
nano target: 198, 262, 220, 16, 15, 15, 15, 15
nano MTP:    198, 262, 220, 16, 15, 15, 15, 15

prefill logits cosine: 0.9998893142
decode logits cosine:  0.9998866916
HF peak allocated:     1.4518 GiB
nano target peak:      1.5829 GiB
nano MTP graph peak:   1.6423 GiB (single-request golden run)
```

## Native MTP correctness gates

The suite covers:

- normal official MTP weights;
- deliberately zeroed MTP weights to force rejection;
- one-shot and chunked prefill;
- multiple consecutive speculative rounds;
- two requests with different accepted lengths in the same batch;
- direct per-token recurrent/conv state history versus every independently
  recomputed prefix;
- eager versus CUDA Graph target/MTP proposal parity;
- slow-consumer streaming with a queue capacity of one;
- finish reason, waiting cancellation, KV/state release, and engine cleanup.

In every case, the committed output tokens match target-only greedy decoding.

## Performance

Command for the main short benchmark:

```bash
python benchmark_qwen35_mtp.py \
  --model /path/to/Qwen3.5-0.8B-Base \
  --mode both \
  --output-tokens 8 \
  --cuda-graph \
  --repetitions 3
```

Each repetition creates a fresh engine and first warms the same two-request
shape that is measured.

| Metric | Target-only | Native MTP-2 |
|---|---:|---:|
| Throughput mean | 94.81 tok/s | 78.55 tok/s |
| Throughput population std | 3.68 | 0.92 |
| Mean TPOT | 10.86 ms | 14.91 ms |
| Maximum peak allocation | 1.65 GiB | 1.76 GiB |
| Output equality | reference | exact |
| Throughput delta | — | -17.2% |

The three target runs measured 89.60, 97.44, and 97.38 tok/s. The three MTP
runs measured 77.36, 79.59, and 78.69 tok/s.

A longer single run with 64 output tokens per request measured:

| Metric | Target-only | Native MTP-2 |
|---|---:|---:|
| Throughput | 164.67 tok/s | 159.26 tok/s |
| Mean TPOT | 10.79 ms | 11.21 ms |
| Throughput delta | — | -3.3% |

The long MTP run had a 58.3% draft acceptance rate, 21 parallel verifier
forwards, 21 target-decode graph replays, 64 MTP graph replays, and zero
accepted-prefix recomputes.

## Interpretation

This is a verified negative performance result, not a correctness failure.
The baseline one-token decode path is already very efficient under CUDA Graph.
The current MTP path captures target decode, recursive draft steps, and the
fixed-width parallel verifier, while writing FP32 per-token GDN state history
for exact variable-boundary commit. On short requests the remaining launch and
metadata costs are visible; on the longer run it approaches parity but does not
yet establish an acceleration.

Follow-up work has since profiled the path and removed the redundant final
history boundary; the remaining complete performance record is in
`docs/mtp-phase-profiling.md`. The final project claim still requires
the official 9B cloud golden, multi-GPU NCCL validation, and 9B benchmark.
