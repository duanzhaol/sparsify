---
name: npu_profiling
description: Ascend NPU msprof 性能 Profiling 分析。分析 msprof 输出的算子统计、时间线、kernel 级利用率和通信开销，定位瓶颈并给出优化建议。在用户提供 profiling 数据目录或要求分析 NPU 性能时自动应用。
keywords:
  - profiling
  - msprof
  - 性能分析
  - npu
  - ascend
  - 瓶颈
  - performance
---

# NPU Profiling 分析

分析 Ascend NPU 上的 msprof 性能数据，从全局视角定位瓶颈。

---

## 何时使用本 Skill

- 用户提供 msprof profiling 数据目录（通常路径包含 `PROF_` 前缀或 `prof_output`）
- 用户要求分析 NPU 性能、训练瓶颈、算子耗时
- 用户要求对比优化前后的 profiling 结果

---

## 1. Profiling 数据结构

msprof 输出的目录结构：

```
prof_output/PROF_XXXXXX_YYYYMMDDHHMMSS_XXXXXXXX/
└── mindstudio_profiler_output/
    ├── op_statistic_*.csv           # 算子级聚合统计（按类型）
    ├── op_summary_*.csv             # 逐算子详细（含时间戳、shape、耗时）
    ├── task_time_*.csv              # kernel 级时间线
    ├── communication_statistic_*.csv # 通信统计
    ├── api_statistic_*.csv          # Host API 调用统计
    └── msprof_*.json                # 完整 trace（可导入 MindStudio）
```

### 各 CSV 关键列

**op_statistic**: `OP Type, Core Type, Count, Total Time(us), Ratio(%)`

**op_summary**: `Op Name, OP Type, Core Type, Task Start Time(us), Task Duration(us), Input Shapes`

**task_time**: `kernel_name, kernel_type, task_time(us), task_start(us)`

**communication_statistic**: `OP Type, Count, Total Time(us), Avg Time(us), Max Time(us)`

---

## 2. 分析脚本

本 skill 提供 4 个分析脚本，位于 `scripts/` 目录下。在分析前**先运行脚本获取结构化数据**，再基于输出进行深入分析。

### 2.1 一键全量分析

```bash
python .claude/skills/npu_profiling/scripts/full_analysis.py <PROF_DIR> [--output report.md]
```

运行所有 4 个子分析，可选生成 markdown 报告。

### 2.2 算子统计分析

```bash
python .claude/skills/npu_profiling/scripts/analyze_ops.py <PROF_DIR> [--top 20]
```

输出：
- Top-N 算子排名（按 NPU 耗时）
- 按核心类型汇总（AI_CORE vs AI_VECTOR_CORE vs AI_CPU）
- AI_CPU 回退检测
- VECTOR_CORE 热点算子标记

### 2.3 时间线分析

```bash
python .claude/skills/npu_profiling/scripts/analyze_timeline.py <PROF_DIR> [--skip-init-seconds 5]
```

输出：
- 自动 Step 边界检测（基于 FlashAttention 聚集模式）
- 每 Step 内算子按类别分解（LLM/SAE/通信/数据搬运）
- MatMul 按 Shape 分类（区分 LLM vs SAE 的矩阵乘法）
- LLM-SAE 交织执行模式检测
- NPU 空闲时间估算

### 2.4 Kernel 级时间

```bash
python .claude/skills/npu_profiling/scripts/analyze_kernel_time.py <PROF_DIR>
```

输出：
- 计算 vs 等待 vs 通信的高层分类
- NPU 计算利用率
- AI_CORE (Cube) vs AI_VECTOR_CORE 比例
- 各 kernel_type 详细占比

### 2.5 通信分析

```bash
python .claude/skills/npu_profiling/scripts/analyze_communication.py <PROF_DIR>
```

输出：
- 各通信原语统计（allReduce, broadcast, allGather）
- 长尾检测（最大值远大于平均值）

---

## 3. 分析方法论（Agent 必读）

### 3.1 全局视角优先

**不要只看算子耗时排名。** 必须回答以下问题：

1. **单步 wall time 是多少？** — 从 op_summary 的时间戳计算
2. **LLM 推理占多少？SAE 训练占多少？** — 通过 MatMul shape 分类 + 特征算子归类
3. **NPU 有多少时间在空闲？** — 从 task_time 的 NOTIFY_WAIT_SQE + EVENT_WAIT
4. **通信开销占多少？** — 从 communication_statistic
5. **有哪些 CPU 回退？** — 从 op_statistic 的 Core Type = AI_CPU

### 3.2 算子归属分类规则

SAE 训练中 LLM 和 SAE 交织执行，需要区分算子归属：

| 算子 | 归属 | 判断依据 |
|------|------|---------|
| FlashAttentionScore | LLM | 仅在 LLM 前向中出现 |
| EmbeddingBag | SAE Forward | FusedDecoder decode |
| TopKV2 | SAE Forward | TopK 激活选择 |
| IndexPutV2 / ScatterElementsV2 | SAE Backward | 梯度散射 |
| MatMul | **需按 shape 区分** | 见下方 |
| Cast / Transpose | 数据搬运 | 精度转换 / 布局变换 |
| Lerp / LerpV2 | Optimizer | SignSGD/schedulefree 更新 |
| allreduce | DDP 通信 | 梯度同步 |

### 3.3 MatMul Shape 分类

**关键**: MatMul 是 LLM 和 SAE 共享的算子，必须按 Input Shapes 区分：

对于 SAE（hidden=H, expansion=E, d_sae=H*E）：
- `(batch, d_sae; d_sae, H)` 或 `(d_sae, batch; batch, H)` → SAE encoder/decoder
- `(batch, H; d_sae, H)` → SAE encoder forward

对于 LLM：
- Shape 中出现 `3072` (intermediate_size) → MLP
- Shape 中出现 `1024;1024` (hidden_size) → Q/K/V/O projection
- Shape 中出现 `2048` (2 * head_dim * num_heads for K/V) → K/V projection

### 3.4 核心类型说明

| 核心类型 | 说明 | 适合的计算 |
|---------|------|-----------|
| AI_CORE | Cube 矩阵加速单元 | MatMul, Conv, 大规模并行 |
| AI_VECTOR_CORE | 向量计算单元 | Elementwise, scatter, embedding |
| MIX_AIC | 混合 Cube 为主 | FlashAttention |
| MIX_AIV | 混合 Vector 为主 | TopK, ReduceMean |
| AI_CPU | CPU 回退 | 不支持的算子，**严重性能问题** |

**重要**: 如果高耗时算子跑在 AI_VECTOR_CORE 而其计算模式可用 MatMul 表达，则应考虑重写为 AI_CORE Cube 路径。

### 3.5 典型瓶颈模式

1. **EmbeddingBag 在 VECTOR_CORE**: SAE decode 用 `F.embedding_bag` 实现稀疏解码，跑在向量单元。可改为 scatter + matmul 走 Cube。

2. **AI_CPU 回退**: `IndexPut`, `_embedding_bag_backward` 等算子在 NPU 上不支持，回退 CPU 导致跨设备搬运。用 `index_add_` 等 NPU 原生算子替代。

3. **大量 Cast**: bf16 ↔ fp32 频繁转换。统一计算精度可减少。

4. **通信长尾**: 单次 allreduce 耗时远超平均值，通常是 logging/checkpoint 触发的全局同步。

5. **NPU 空闲**: NOTIFY_WAIT_SQE 占比高说明 host 端 Python/kernel launch 开销大。

### 3.6 优化前后对比

对比两次 profiling 时，关注：
1. 目标算子是否消失或减少
2. 核心类型是否从 AI_VECTOR_CORE → AI_CORE
3. AI_CPU 回退是否消除
4. 单步 wall time 变化
5. NPU 利用率变化

---

## 4. 输出格式

分析完成后，输出应包含以下结构（可写入 docs/ 目录）：

```markdown
# Profiling 分析报告

## 概述
- 模型/配置信息
- Profiling 环境

## 1. 整体时间线
- 总时间、初始化、训练阶段
- NPU 利用率

## 2. 单步时间分解
- 各类别占比柱状图（文本形式）
- LLM vs SAE 比例

## 3. 执行模式
- 交织/顺序执行模式
- 每组的算子序列

## 4. 瓶颈排序
- 按影响排序的表格

## 5. 算子详细统计
- Top-N 表格
- MatMul shape 分类

## 6. 通信分析

## 7. 优化方向
- 每项优化的现状、方案、预估收益
```

---

## 5. 快速参考

### msprof 采集命令

```bash
# 基础采集
msprof --application="python" \
  --application-args="-m sparsify model_name" \
  --output ./prof_output

# torchrun 多卡采集
msprof --application="/usr/local/python3.11.13/bin/torchrun" \
  --application-args="--nproc_per_node 4 -m sparsify model_name --batch_size 1" \
  --output ./prof_output

# 只采集特定阶段（减少数据量）
# 在代码中使用 torch_npu.profiler 手动控制采集范围
```

### 常见 shape 映射（Qwen3-0.6B + SAE expansion=8）

| Shape 特征 | 含义 |
|-----------|------|
| 8192 维度 | d_sae = 1024 * 8 |
| 3072 维度 | Qwen3 intermediate_size |
| 1024 维度 | hidden_size |
| 2048 维度 | 2 * num_kv_heads * head_dim |
