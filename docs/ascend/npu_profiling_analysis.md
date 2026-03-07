# Ascend NPU 性能 Profiling 分析

## 概述

本文档记录了在 Ascend NPU 上运行 SAE 训练的性能 profiling 分析结果，基于 `msprof` 采集的数据，分析了整个训练 step 的时间分布、瓶颈定位和优化方向。

**Profiling 环境**:
- 模型: Qwen3-0.6B (28 层, hidden_size=1024)
- SAE 配置: expansion_factor=8, k=512, hookpoints=`layers.[0-30].self_attn.q_proj`
- 训练配置: 4 卡 DDP, batch_size=1, grad_acc_steps=8
- 数据来源: `prof_output/PROF_000001_20260306150848328_CIICBKNAGJNEHGFB`

---

## 1. 整体时间线结构

```
0 ────── 5.3s ──── 7.8s ────── 8.8s ────── 9.5s ────── 10.4s ──── 11.4s ──── 12.3s
   init    hcom     Step1       Step2       Step3       Step4     logging+sync
   idle    init   (warmup)
```

- **0 - 7.8s**: 初始化阶段 — Python/PyTorch 启动、模型加载、`hcomAicpuInit`(329ms)、rank 间同步等待
- **7.8s - 12.3s**: 实际训练 — 每个 step 约 **1097ms**
- **最后 ~2s**: logging 触发的 allreduce 阻塞（单次 allreduce 耗时 1065ms）

**NPU 利用率**: 在 14.7s 总时间线中，实际 NPU 计算仅 5.3s，其余为初始化、通信等待和 host 端开销。

---

## 2. 单步 (Step) 时间分解

以 Step 2 为例，wall time = **1097ms**。

### 2.1 高层分解

```
 ┌──────────────────────────────────────────────────────────────┐
 │                    单步时间分解 (1097ms)                       │
 ├──────────────────────────────────────────────────────────────┤
 │                                                              │
 │  ██████████████████████████████  SAE Forward     304ms (28%) │
 │  ████████████████████████████    SAE Backward    245ms (22%) │
 │  ████████████████                LLM Forward     165ms (15%) │
 │  ████████████████                数据搬运*       200ms (18%) │
 │  ████████                        Communication    96ms  (9%) │
 │  ████████                        NPU Idle         85ms  (8%) │
 │  ██                              Optimizer          2ms  (0%)│
 │                                                              │
 │  * Cast(bf16↔fp32)/Transpose/Elementwise 等辅助计算           │
 └──────────────────────────────────────────────────────────────┘
```

**核心结论: SAE 训练占总计算的 55-60%，LLM 推理仅占 15%。**

### 2.2 SAE Forward 详细 (304ms, 28%)

| 算子 | 核心类型 | 调用次数 | 耗时 | 说明 |
|------|---------|---------|------|------|
| EmbeddingBag | AI_VECTOR_CORE | 310 | 181.6ms | FusedDecoder forward decode |
| MatMul (encoder) | AI_CORE | 62 | 82ms | SAE 线性编码 `(N,d_in)@(d_in,d_sae)` |
| TopKV2 | MIX_AIV | 62 | 37ms | TopK 激活选择 |
| ReLU + LpNorm | AI_VECTOR_CORE | - | 3.5ms | ReLU 激活 + 解码器权重归一化 |

### 2.3 SAE Backward 详细 (245ms, 22%)

| 算子 | 核心类型 | 调用次数 | 耗时 | 说明 |
|------|---------|---------|------|------|
| IndexPutV2 | AI_VECTOR_CORE | 66 | 211.8ms | W_dec 梯度 (scatter_add/index_add) |
| ScatterElementsV2 | AI_VECTOR_CORE | 62 | 62ms | 梯度散射 |

### 2.4 LLM Forward 详细 (165ms, 15%)

| 算子 | 核心类型 | 调用次数 | 耗时 | 说明 |
|------|---------|---------|------|------|
| FlashAttentionScore | MIX_AIC | 152 | 57.7ms | 注意力计算 |
| MatMul (Q/K/V/O + MLP) | AI_CORE | ~600 | 76ms | 线性投影 |
| RMSNorm | AI_VECTOR_CORE | - | ~31ms | ReduceMean + Pows + Rsqrt |

### 2.5 数据搬运开销 (200ms, 18%)

| 算子 | 调用次数 | 耗时 | 说明 |
|------|---------|------|------|
| Cast (bf16↔fp32) | 2137 | 21.8ms | 精度转换 |
| Transpose | 994 | 59ms | 布局转换 |
| Mul/Add/Neg 等 | ~4000 | ~120ms | 各种 elementwise |

---

## 3. 执行模式分析

### 3.1 LLM 与 SAE 交织执行

训练并非"先跑完 LLM 再跑 SAE"，而是逐层交织：

```
Layer 0-5:   [FA][FA][FA][FA][FA][FA] → [TopK] → [EmbeddingBag] → [IndexPut/Scatter]
Layer 6-11:  [FA][FA][FA][FA][FA][FA] → [TopK] → [EmbeddingBag] → [IndexPut/Scatter]
Layer 12-17: [FA][FA][FA][FA][FA][FA] → [TopK] → [EmbeddingBag] → [IndexPut/Scatter]
...
Layer 24-30: [FA][FA][FA][FA][FA][FA] → [TopK] → [EmbeddingBag] → [IndexPut/Scatter]
→ Optimizer (Lerp 2ms) → AllReduce (96ms)
```

这种交织模式意味着 LLM 推理和 SAE 训练在时间上紧密耦合，hook 捕获每层的 q_proj 激活后立即进行 SAE 前向和反向传播。

### 3.2 NPU 空闲分析

NPU 空闲 85ms (8% of step):
- 主要来自 host 端 Python 开销和 kernel launch 间隙
- 每个 Cast 操作前有 ~2.7ms 的 host wait（kernel 编译/调度延迟）
- 不是主要瓶颈

---

## 4. 瓶颈排序

按对 step wall time 的影响排序:

| 排名 | 瓶颈 | 影响 | 根因 |
|------|------|------|------|
| 1 | **EmbeddingBag** (SAE decode forward) | 181ms (17%) | 跑在 AI_VECTOR_CORE 而非 AI_CORE Cube |
| 2 | **IndexPutV2** (SAE backward W_dec) | 212ms (19%) | scatter/index_add 跑在 AI_VECTOR_CORE |
| 3 | **Encoder MatMul** (SAE encode) | 82ms (7%) | 正常，已在 AI_CORE |
| 4 | **Communication** (DDP allreduce) | 96ms (9%) | 4 卡间梯度同步 |
| 5 | **NPU Idle** (host 开销) | 85ms (8%) | Python/kernel launch |
| 6 | **Cast** (精度转换) | 22ms (2%) | bf16↔fp32 频繁转换 |
| 7 | **CPU Fallback** (IndexPut) | ~12ms/step | 12 次 AI_CPU fallback |

---

## 5. 算子级统计 (op_statistic)

Top 15 算子按 NPU 耗时:

| 排名 | 算子 | 核心类型 | 调用次数 | 总耗时(us) | 占比 |
|------|------|---------|---------|-----------|------|
| 1 | EmbeddingBag | AI_VECTOR_CORE | 192 | 558,821 | 18.4% |
| 2 | MatMulV3 | AI_CORE | 1,752 | 380,918 | 12.5% |
| 3 | hcomAicpuInit | AI_CPU | 1 | 329,202 | 10.8% |
| 4 | IndexPutV2 | AI_VECTOR_CORE | 96 | 288,443 | 9.5% |
| 5 | Transpose | AI_VECTOR_CORE | 3,138 | 183,135 | 6.0% |
| 6 | Mul | AI_VECTOR_CORE | 7,053 | 153,848 | 5.1% |
| 7 | FlashAttentionScore | MIX_AIC | 484 | 146,057 | 4.8% |
| 8 | IndexPut (CPU!) | AI_CPU | 12 | 120,270 | 4.0% |
| 9 | TopKV2 | MIX_AIV | 96 | 113,887 | 3.7% |
| 10 | allreduce | AI_CPU | 959 | 112,113 | 3.7% |
| 11 | MatMulV2 | AI_CORE | 2,020 | 102,447 | 3.4% |
| 12 | Cast | AI_VECTOR_CORE | 7,601 | 72,002 | 2.4% |
| 13 | ScatterElementsV2 | AI_VECTOR_CORE | 192 | 70,763 | 2.3% |
| 14 | Add | AI_VECTOR_CORE | 4,609 | 57,010 | 1.9% |
| 15 | ReduceMean | MIX_AIV | 2,037 | 44,952 | 1.5% |

### MatMul 按 shape 分类

| Shape | 调用次数 | 耗时 | 归属 |
|-------|---------|------|------|
| `8192,4096;4096,1024` | 62 | 50.4ms | SAE encoder/decoder backward |
| `4096,1024;3072,1024` | 304 | 28.3ms | LLM MLP gate/up proj |
| `4096,1024;8192,1024` | 31 | 24.8ms | SAE encoder forward |
| `4096,1024;1024,1024` | 304 | 14.8ms | LLM Q/K/V/O projection |
| `4096,3072;1024,3072` | 152 | 14.4ms | LLM MLP down proj |
| `4096,2048;1024,2048` | 152 | 9.5ms | LLM K/V projection |

---

## 6. 通信分析

| 通信类型 | 调用次数 | 总耗时(us) | 占比 |
|---------|---------|-----------|------|
| allReduce | 962 | 2,264,855 | 99.4% |
| broadcast | 17 | 11,922 | 0.5% |
| allGather | 4 | 1,887 | 0.1% |

- 每步常规 allreduce: ~96ms（SAE 梯度同步）
- logging 触发的 allreduce: 最大单次 1065ms（所有 rank 同步）
- broadcast: 初始化阶段参数广播

---

## 7. 优化方向

### 7.1 优化 1: Forward decode — EmbeddingBag → scatter+matmul (预估 -15%)

**现状**: `F.embedding_bag` 跑在 AI_VECTOR_CORE，每次 2.9ms，共 181ms/step

**方案**: 构造稀疏系数矩阵后用 dense matmul (跑在 AI_CORE Cube):

```python
S = torch.zeros(N, M, dtype=top_acts.dtype, device=top_acts.device)
S.scatter_(1, top_indices.long(), top_acts)
out = S @ W_T  # AI_CORE Cube, 参考同 shape matmul ~0.8ms/call
```

**预计**: 181ms → ~50ms/step

### 7.2 优化 2: 消除 AI_CPU IndexPut fallback (预估 -4%)

12 次 `IndexPut` 跑在 AI_CPU，其中 1 次 117ms。需定位来源（可能是 `fused_encoder.py` backward 的 `grad_bias.index_add_()`）并替换为 NPU-native 实现。

### 7.3 优化 3: Backward 路径确认 (预估 -5~10%)

验证 FusedDecoder backward 的 `use_matmul` 分支（scatter_add + matmul vs index_add_）实际走了哪条路径。如果 M*N 超过 256MB 阈值走了慢路径，考虑调整阈值。

### 7.4 优化 4: 减少 Cast 操作 (预估 -2%)

7,601 次 Cast (bf16↔fp32) 共 72ms。考虑统一 SAE 计算精度，减少中间转换。

### 7.5 优化 5: 通信优化

- 考虑使用 gradient compression 或 allreduce bucketing
- 排查 logging 阶段的 1065ms allreduce 阻塞原因

---

## 8. 与 NVIDIA GPU 的对比

| 维度 | Ascend NPU | NVIDIA GPU |
|------|-----------|------------|
| SAE decode forward | `F.embedding_bag` (VECTOR_CORE) | Triton fused kernel |
| SAE decode backward | `index_add_` / `scatter_add_` | Triton autograd |
| 瓶颈 | EmbeddingBag 18% + IndexPut 10% | 通常 <5% |
| 优化空间 | 改用 matmul 路径 | 已高度优化 |

NPU 上的核心问题是 `EmbeddingBag` 和 `index_add_` 都跑在 AI_VECTOR_CORE（向量计算单元），无法利用 AI_CORE 的 Cube 矩阵加速单元。NVIDIA 上 Triton 可以直接生成高效的 fused kernel。

---

## 附录: Profiling 工具使用

```bash
# 采集 profiling 数据
msprof --application="/usr/local/python3.11.13/bin/torchrun" \
  --application-args="--nproc_per_node 4 -m sparsify ..." \
  --output ./prof_output

# 关键输出文件
prof_output/PROF_xxx/mindstudio_profiler_output/
├── op_statistic_xxx.csv      # 算子级统计（按类型聚合）
├── op_summary_xxx.csv        # 逐算子详细信息（含时间戳、shape、耗时）
├── task_time_xxx.csv          # kernel 级时间线
├── api_statistic_xxx.csv      # Host API 调用统计
└── communication_statistic_xxx.csv  # 通信统计
```
