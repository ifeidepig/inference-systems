# nano-vLLM Qwen3.5 Fine-Grained Hybrid Prefix Cache 二次开发总结

日期：2026-10-01

## 1. 文档目的

本文集中说明本项目相对原始 nano-vLLM Prefix Cache 路径完成的二次开发，包括问题背景、架构变化、核心数据结构、事务边界、实验结果、代码入口和证据边界。

项目不是简单给原有 Prefix Cache 增加一个开关，而是为 Qwen3.5 的 Hybrid Attention 重新定义“可复用前缀”：

```text
可提交的 Hybrid Prefix Hit
= Full Attention KV 可达前缀
∩ 同一 token boundary 的 GDN recurrent/conv checkpoint
```

## 2. 原始实现与新增问题

### 2.1 原始 nano-vLLM Prefix Cache

原始路径面向普通 Full Attention 模型：

- 使用固定大小 physical KV blocks；
- 通过 chained block hash 判断完整 block 前缀是否相同；
- 使用 `block_table` 完成 logical block 到 physical block 的映射；
- 通过 `ref_count` 共享和回收缓存块；
- 命中粒度与 physical block size 绑定。

对普通 Transformer 而言，历史推理状态就是逐 token 的 K/V，因此完整 KV block 命中即可恢复计算。

### 2.2 Qwen3.5 Hybrid Attention 的额外状态

Qwen3.5 text backbone 混合使用：

```text
Full Attention -> 随 context 增长的 KV Cache
Gated DeltaNet -> 固定 shape 的 recurrent state + conv state
```

如果只复用 Full Attention KV，而把 GDN state 置零或恢复到错误边界，模型输出会静默错误。因此 KV-only Prefix Cache 对 Hybrid 模型不成立。

## 3. 二次开发总览

本次开发分为五个里程碑：

```text
M1  FullAttention/GDN managers + HybridPrefixCoordinator
M2  candidate boundary / retention / eviction 解耦
M3  fine-grained hash + partial-page KV COW + admission transaction
M4  sparse internal GDN checkpoints
M5  TP all-rank prepare / consensus / rollback
```

默认行为仍保持关闭：

```text
enable_hybrid_prefix_cache = false
```

未开启 Fine-Grained 参数时，系统仍使用原有 256-token block-aligned 行为。

## 4. M1：Hybrid Cache Coordinator

### 4.1 管理器拆分

新增三个控制面组件：

```text
FullAttentionPrefixManager
  -> KV prefix candidates、physical block admission、boundary metadata

GDNCheckpointManager
  -> recurrent/conv checkpoint index、retention metadata、eviction

HybridPrefixCoordinator
  -> 选择 KV/GDN 同时可恢复的最长 boundary
```

Scheduler 不再直接拼接两类缓存逻辑，而是先获得不可变的 `HybridPrefixAllocationPlan`，再修改请求资源所有权。

### 4.2 Boundary 不变量

全系统统一采用：

```text
checkpoint(N) = 处理完 tokens [0,N) 后的 GDN state
num_cached_tokens = N
resume input = position N
```

该约定避免 token 被重复应用或遗漏。

## 5. M2：Checkpoint Retention 解耦

将四个原本容易混淆的粒度拆开：

| 粒度 | 含义 |
|---|---|
| Physical KV block | 显存按多少 token 组织一页 |
| Prefix match unit | 最细在哪个 token boundary 判断前缀相同 |
| Candidate boundary | 哪些位置可以表达合法 GDN state |
| Retention / eviction | 哪些 state 真正保存、容量满后保留谁 |

当前 retention policy：

- `periodic`：保留固定周期边界；
- `adaptive`：组合 periodic、prompt-tail 与运行时发现的 shared-prefix junction。
  当请求发现 Full-Attention KV 可命中、但 GDN checkpoint 只能恢复到更早边界时，
  Coordinator 记录 alignment loss，并让该请求在本来必须执行的 Prefill 中提升最长
  KV-only boundary。第二个请求负责 capture，第三个及后续请求获得联合 KV/GDN hit。

当前 eviction policy：

- `lru`：淘汰最久未访问 entry；
- `cost_aware`：普通 checkpoint 沿用 boundary value；shared-junction 使用相对当前
  共同恢复点节省的 replay tokens，并计入真实 KV-only demand 与成功 hit。

Demand tracker 只保存有界 CPU metadata；固定预算 GPU checkpoint pool 不会因为观察到
大量一次性 prefix 而直接增长。Promotion intent 在 admission commit 后才进入 pending，
capture 失败、abort 或 preemption 会解除 pending，允许后续请求重试。

## 6. M3：Fine-Grained Prefix Match 与 Partial-page COW

### 6.1 Match unit 与 physical page 解耦

支持：

```text
prefix_match_unit < kvcache_block_size
kvcache_block_size % prefix_match_unit == 0
```

例如：

```text
physical page = 256 tokens
prefix_match_unit = 16 tokens
shared prefix = 500 tokens
longest hash hit = 496 tokens
```

BlockManager 新增独立 chained hash-unit index。每个 entry 记录：

```text
prefix_hash
unit_token_ids
tail_block_id
tail_valid_tokens
```

Physical page 被重新分配时，所有指向该页的 sub-block tail hashes 会失效，避免读取陈旧 KV。

### 6.2 Partial-page Copy-on-Write

命中 496 时，positions 256-495 与旧请求共享，但后续 suffix 不同。新请求不能继续写 cached source page，因此执行：

```text
pin source physical page
-> allocate private destination page
-> copy all Full Attention layers' K/V rows
-> restore GDN state@496
-> commit hit
-> release source pin
```

ModelRunner 对统一 KV tensor 执行：

```text
kv_cache[K/V, all_kv_layers, source_block, 0:valid_tokens]
->
kv_cache[K/V, all_kv_layers, destination_block, 0:valid_tokens]
```

### 6.3 Admission transaction

以下操作被视为同一事务：

```text
KV block reference/allocation
request state-slot allocation
partial-page COW
GDN checkpoint restore
hit-metadata commit
```

任一步失败都会归还：

- request `block_table`；
- KV ref-count；
- COW source pin；
- request state slot；
- 未发布 checkpoint slot。

Hit 指标只在 GPU COW 与 GDN restore 全部成功后提交。

## 7. M4：Sparse Internal GDN Checkpoint

旧 split-prefill：

```text
forward [0,N)
-> capture state@N
-> forward [N,end)
```

新增 internal checkpoint：

```text
one forward [0,end)
-> sparse CP@N
-> final RUN@end
```

GDN reference scan 接受排序后的 sparse token indices，只在这些位置 clone recurrent/conv state，不再物化完整 per-token state history。

ModelRunner 将所有 GDN layers 的同边界 state 组装成 checkpoint tensor，直接写入固定预算 checkpoint pool。

当前限制：internal checkpoint 实验路径要求 `max_num_seqs=1`；多请求 ragged checkpoint positions 尚未实现。

## 8. M5：Tensor Parallel 事务

Checkpoint capture/restore 与 KV COW 使用 all-rank consensus：

```text
all ranks local prepare/copy
-> all-reduce success
-> verify checkpoint slot IDs
-> rank 0 returns success
-> Scheduler publishes CPU metadata
```

失败语义：

- Capture 失败：成功 ranks 清零并归还本地 checkpoint slot；
- Slot ID 分叉：所有 ranks rollback，并重建确定性 free queue；
- Restore 失败：所有 ranks 恢复事务前 private request state；
- COW 失败：rank 0 触发 Scheduler admission rollback；
- 只有 rank 0 向 Scheduler 抛错，worker ranks 回滚后继续命令循环。

由于本机只有一张 GPU，分布式故障通过两进程 Gloo + CPU local shards 验证；真实 NCCL 与 Qwen3.5-9B 性能留待算力平台。

## 9. 显存管理

GDN checkpoint 使用固定 byte-budget GPU pool：

```text
capacity = memory_budget_bytes // bytes_per_checkpoint
```

不使用无界 Python tensor 字典，避免显存不可控增长与 allocator fragmentation。

Qwen3.5-0.8B 每份 checkpoint：

```text
19,537,920 bytes
```

五变体实验统一预分配：

```text
664,289,280 bytes
```

## 10. 正确性测试

已覆盖：

- cold/warm greedy token exactness；
- eager 与 CUDA Graph decode；
- `N-1/N/N+1` 边界舍入；
- chunked prefill；
- hit -> preempt -> re-hit -> abort；
- COW source/destination 隔离；
- physical-page reassignment 后 sub-block index 失效；
- COW 故障后的 KV/state/ref-count rollback；
- checkpoint capture/restore/free isolation；
- 两进程 Gloo capture/restore/COW 故障注入；
- worker failure 后继续下一成功事务；
- TP=2 GDN shard/forward equivalence。

## 11. 官方 0.8B 受控实验

Workload：

```text
model = Qwen3.5-0.8B-Base
shared prefix = 496 tokens
unique suffix = 1 token
output = 2 tokens
physical page = 256
fine match unit = 16
checkpoint budget = 640 MiB
```

### 11.1 Shared-prefix matrix

| Variant | Hit | TTFT | TPOT | Throughput | Producer forwards |
|---|---:|---:|---:|---:|---:|
| no cache | 0 | 2438.69 ms | 40.16 ms | 0.81 tok/s | 1 |
| block aligned | 256 | 1186.91 ms | 39.28 ms | 1.63 tok/s | 2 |
| fine dense | 496 | 38.23 ms | 38.44 ms | 25.99 tok/s | 32 |
| fine adaptive | 496 | 39.15 ms | 37.68 ms | 25.93 tok/s | 2 |
| fine internal | 496 | 38.83 ms | 38.93 ms | 25.63 tok/s | 1 |

所有变体输出 token 完全一致。

Fine hit 的 all-layer KV COW：

```text
copy bytes = 2,949,120
copy latency = 0.078-0.103 ms
```

### 11.2 No-share negative control

| Variant | Committed hit | Target TTFT delta |
|---|---:|---:|
| block aligned | 0 | +5.47% |
| fine dense | 0 | +61.14% |
| fine adaptive | 0 | +1.97% |
| fine internal | 0 | +1.35% |

结论：细粒度匹配不能单独使用。Dense checkpoint retention 会严重增加 producer forward 次数和 no-share 开销；adaptive retention 与 internal checkpoint 才能使 finer matching 具有实际工程价值。

上述结果均为本地单次受控实验，用于验证机制和性能方向，不代表生产置信区间。

## 12. 主要代码入口

| 文件 | 二次开发职责 |
|---|---|
| `nanovllm/engine/block_manager.py` | sub-block hash index、COW page plan、ref-count 与失效 |
| `nanovllm/engine/hybrid_prefix_cache.py` | managers、coordinator、retention、eviction |
| `nanovllm/engine/state_manager.py` | request state pool、checkpoint pool、TP transaction |
| `nanovllm/engine/scheduler.py` | admission transaction、boundary planning、capture publish |
| `nanovllm/engine/model_runner.py` | all-layer KV COW、sparse internal state、capture/restore |
| `nanovllm/models/qwen35_reference.py` | 稀疏 GDN state checkpoint reference |
| `benchmark_fine_grained_prefix_cache.py` | 五变体与 no-share workload |
| `benchmark_adaptive_prefix_promotion.py` | A/B/C shared-junction 提升与复用 |
| `tests/test_hybrid_prefix_cache.py` | fine hit、COW、lifecycle、failure rollback |
| `tests/test_hybrid_prefix_transaction.py` | all-rank transaction fault injection |
| `benchmark_checkpoint_compression.py` | FP32/BF16/INT8 容量与传输延迟 |
| `benchmark_promotion_threshold.py` | singleton/pair/triple/hot/Zipf admission policy |

### Promotion threshold 实验

`hybrid_prefix_promotion_min_sightings` 按总出现次数计数，默认 2。0.8B 本地小矩阵中，
threshold=2 在 pair-only 流量产生 4 个尚未复用 checkpoint，但在 triple/hot/Zipf
流量比 threshold=3 少一次 cold replay，并获得更高 useful-promotion ratio。默认继续
使用 2，同时将 pair-heavy 记录为负例。详见 `docs/promotion-threshold-study.md`。

### Checkpoint 压缩

Checkpoint pool 支持 `fp32/bf16/int8` 三种 recurrent storage，active request state
始终保持 FP32，conv checkpoint 保持 native dtype。INT8 按 `(layer, head,
key-channel)` 保存 FP32 scale，restore 时一次性反量化。官方 0.8B 的 128 MiB 预算
容量为 6/13/24 slots；1024-token A/B/C gate 中 BF16/INT8 均与 FP32 greedy tokens
一致。详细证据与限制见 `docs/checkpoint-compression.md`。

## 13. 复现命令

Shared-prefix：

```bash
PYTHONSAFEPATH=1 PYTHONPATH=. \
python benchmark_fine_grained_prefix_cache.py \
  --model /path/to/Qwen3.5-0.8B-Base \
  --shared-prefix-length 496 \
  --unique-suffix-length 1 \
  --output-tokens 2 \
  --checkpoint-memory-mib 640 \
  --max-num-batched-tokens 512 \
  --max-num-kvcache-blocks 32 \
  --summary-only
```

No-share：在上述命令增加 `--no-share`。

Adaptive v2 A/B/C：

```bash
python benchmark_adaptive_prefix_promotion.py \
  --model /path/to/Qwen3.5-0.8B-Base \
  --shared-prefix-length 496 \
  --unique-suffix-length 64
```

本地官方 0.8B 单次功能验收中，B 记录 496 alignment-lost tokens 并发布一次
shared-junction checkpoint，C committed hit 496 tokens；internal/split 两条路径的
C TTFT 分别约 352/367 ms。原始摘要见
`benchmark_results/adaptive_prefix_promotion_local_20261003.json`。这些数据只用于证明
promotion 生命周期可执行，不作为稳定性能结论。

## 14. 证据边界与后续工作

已经证明：

- 官方 0.8B checkpoint correctness；
- block-aligned 与 fine-grained hit；
- COW、internal checkpoint、CUDA Graph decode；
- Gloo all-rank failure rollback；
- KV-only alignment-loss 可观测性与 shared-junction second-sighting promotion；
- A 保存 unique-tail checkpoint、B 提升共享 junction、C 命中共同边界的 CPU 回归；
- split 与 sparse internal checkpoint 两条 promotion capture 路径；
- 单次 shared/no-share workload 方向。

仍待算力平台证明：

- Qwen3.5-9B 实际 TTFT/TPOT/吞吐；
- 多 GPU NCCL transaction latency；
- 重复多轮 median/P95/P99；
- 多请求 ragged internal checkpoints；
- Hybrid Prefix Cache 与 MTP 共存。
- Adaptive v2 在真实 GPU hot-prefix/unique-suffix 分布下的 TTFT 与显存收益。

## 15. 简历可用总结

> 面向 Qwen3.5 Hybrid Attention 为 nano-vLLM 设计并实现 Fine-Grained Hybrid Prefix Cache，将物理 KV page、prefix hash 粒度与 GDN checkpoint retention 解耦；通过 all-layer partial-page KV copy-on-write、adaptive/cost-aware checkpointing、sparse internal checkpoints 与 all-rank transaction rollback 保证状态一致性。在官方 0.8B 受控实验中，将 committed prefix hit 从 256 提升至 496 tokens，并将 dense checkpointing 的 no-share TTFT 开销从 +61.14% 降至 +1.35%；验证 cold/warm exactness、CUDA Graph decode、生命周期隔离与分布式故障回滚。
