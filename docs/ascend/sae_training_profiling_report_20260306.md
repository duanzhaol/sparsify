# SAE + LLM 联合训练 Profiling 报告（Ascend，2026-03-06）

本文档整理当前 SAE 训练在 Ascend NPU 上的 profiling 结果，覆盖：

- 任务背景与训练流程
- 本次 profiling 的执行过程
- LLM 与 SAE 在整条运行链路中的职责划分
- SAE 主要算子与对应代码位置
- 当前开销分布
- 剩余优化空间与优先级

本文重点使用如下 profiling 导出：

- 真实训练基线：`prof_output/PROF_000001_20260306150848328_CIICBKNAGJNEHGFB`
- 关键统计文件：
  - `prof_output/PROF_000001_20260306150848328_CIICBKNAGJNEHGFB/mindstudio_profiler_output/op_statistic_20260306151015.csv`
  - `prof_output/PROF_000001_20260306150848328_CIICBKNAGJNEHGFB/mindstudio_profiler_output/communication_statistic_20260306151015.csv`
  - `prof_output/PROF_000001_20260306150848328_CIICBKNAGJNEHGFB/mindstudio_profiler_output/api_statistic_20260306151015.csv`
  - `prof_output/PROF_000001_20260306150848328_CIICBKNAGJNEHGFB/mindstudio_profiler_output/msprof_20260306151010.json`

另参考了此前几次 profiling，用于判断哪些优化已生效、哪些尝试产生了回退：

- 旧真实训练 profile：`prof_output/PROF_000001_20260306143524869_MQMBFQNOHPFBLOBB`
- `scatter_add_` 回退 profile：`prof_output/PROF_000001_20260306145650098_HMIIFIDCRBBFCJCA`
- 较早的真实训练 profile：`prof_output/PROF_000001_20260306123933016_EHQNALDJPQONRCOB`

---

## 1. 任务背景

当前任务不是单独训练 LLM，也不是单独 benchmark SAE，而是：

1. 用 Qwen 主干执行前向。
2. 在多个 hookpoint 上截获激活。
3. 对每个 hookpoint 的激活执行 SAE 编码、解码、损失计算和局部反传。
4. 在 step 边界上做 SAE optimizer 更新、dead feature 统计和日志。

因此，真实运行中同时存在两条路径：

- `LLM path`：主干模型前向，负责把 token 跑到目标层。
- `SAE path`：hook 内执行的稀疏编码训练逻辑，包含前向、反向、统计、通信和 step 更新。

本项目里，真实瓶颈长期位于 SAE 一侧，而不是 LLM 主干。

---

## 2. 训练主流程

主循环位于 `sparsify/trainer.py:255` 之后，关键阶段如下：

### 2.1 step 内主线

1. 初始化 `did_fire` 掩码：`sparsify/trainer.py:255`
2. 为每个 hookpoint 注册 forward hook：`sparsify/trainer.py:459`
3. 执行主干模型前向：`sparsify/trainer.py:465`、`sparsify/trainer.py:470`
4. hook 中执行 SAE 逻辑：`sparsify/trainer.py:299`
5. hook 内局部 backward：`sparsify/trainer.py:427`
6. 到达 grad accumulation 边界后执行 optimizer step：`sparsify/trainer.py:491`
7. 更新 dead feature 统计并清空 `did_fire`：`sparsify/trainer.py:502`、`sparsify/trainer.py:535`

### 2.2 hook 内 SAE 逻辑

在 `sparsify/trainer.py:299` 定义的 hook 中，主要做以下工作：

1. 取模块输入并 flatten：`sparsify/trainer.py:302`、`sparsify/trainer.py:308`
2. 可选 Hadamard 旋转：`sparsify/trainer.py:310`
3. 调用对应 SAE：`sparsify/trainer.py:352`
4. 更新 `did_fire`：`sparsify/trainer.py:363`
5. 归约 `fvu` / `auxk_loss` / exceed metrics：`sparsify/trainer.py:368`、`sparsify/trainer.py:372`、`sparsify/trainer.py:405`
6. 对 SAE loss 执行 backward：`sparsify/trainer.py:427`

因此，从运行过程上看，SAE 并不是“几个独立算子”，而是嵌在 LLM forward 中的一整套小训练系统。

---

## 3. 当前实现中的关键优化

### 3.1 优化器 XOR swap 已移除

`trainer.py` 已从 `ScheduleFreeWrapper` 切到 `ScheduleFreeWrapperReference`：

- `sparsify/trainer.py:13`
- `sparsify/trainer.py:128`

checkpoint 保存时也做了对应处理：

- `sparsify/checkpoint.py:206`
- `sparsify/checkpoint.py:243`

这一步的作用是消除此前 `swap()` 中的 `BitwiseXor` AI_CPU 热点。此前这一路曾占真实训练总时间约 38%，现在已经不再是主瓶颈。

### 3.2 Encoder backward 已去掉大部分 Python for-k 风暴

当前 encoder 的关键逻辑在 `sparsify/fused_encoder.py:21`：

- 前向：`F.linear + relu + topk`，见 `sparsify/fused_encoder.py:23`
- `grad_input`：`EmbeddingBag`，见 `sparsify/fused_encoder.py:49`
- `grad_weight`：优先构造稠密系数矩阵 `S` 后 `S @ input`，见 `sparsify/fused_encoder.py:61`
- `grad_bias`：复用 `S.sum(1)`，避免额外 `index_add_`，见 `sparsify/fused_encoder.py:75`

### 3.3 Decoder backward 已避免旧版 gather+bmm 主路径

当前 decoder 的关键逻辑在 `sparsify/fused_decoder.py:22`：

- 前向重建：`EmbeddingBag`，见 `sparsify/fused_decoder.py:32`
- `grad_acts`：优先使用全量 `grad_output @ W_T.t()` 再 `gather` 标量，见 `sparsify/fused_decoder.py:47`
- `grad_W_T`：优先构造 `S` 后执行 `S @ grad_output`，见 `sparsify/fused_decoder.py:59`

这两处 fused backward 已经把旧版 `for-k` 反向中的数万次小 kernel 发射压缩掉了，这是当前 profile 能从“旧 for-k 风暴”转为“EmbeddingBag / IndexPutV2 / 张量整理”为主的核心原因。

---

## 4. LLM 与 SAE 的流程级占比

### 4.1 不能只按算子名机械相加

如果只看 `MatMulV3`、`FlashAttentionScore` 和 `EmbeddingBag` 这些大算子，会低估 SAE 的真实占比，因为 SAE 还带来了：

- hook 内 loss/backward
- `did_fire` 更新
- 指标归约
- DDP / all_reduce 通信
- optimizer step 和 dead feature bookkeeping
- 明显更多的 kernel launch 和 stream synchronize

因此，LLM 与 SAE 的比较必须结合：

- 设备算子时间
- 通信时间
- host API 发射与同步
- 训练代码的阶段划分

### 4.2 当前更合理的工程判断

基于 `op_statistic`、`communication_statistic`、`api_statistic` 以及 `trainer.py` 的执行路径，当前真实训练中更合理的判断是：

- `LLM` 占总 wall-time 约 `15%–25%`
- `SAE` 占总 wall-time 约 `70%–80%`
- 讨论时可以使用一个简化单值：`LLM ≈ 20%`，`SAE ≈ 75%`

如果只看“稳态设备计算”，则大致是：

- `LLM` 约 `24%`
- `SAE` 约 `68%`
- 其他约 `8%`

但加上通信、launch 和同步后，SAE 的真实占比会进一步上升。

---

## 5. 当前 SAE 算子开销

以下数据来自 `op_statistic_20260306151015.csv`，仅统计 SAE 相关算子。按本次窗口约 `12` 个训练 step 折算。

SAE 相关算子总时间约：`1,836,714 us = 1.837 s`。

| 算子 | Core | 调用次数 | 总耗时 | 每步耗时 | SAE 内占比 | 主要来源 |
|------|------|---------:|-------:|---------:|------------:|---------|
| `EmbeddingBag` | `AI_VECTOR_CORE` | 192 | 558.8 ms | 46.57 ms/step | 30.43% | decoder 前向重建 + encoder `grad_input` |
| `IndexPutV2` | `AI_VECTOR_CORE` | 96 | 288.4 ms | 24.04 ms/step | 15.70% | `did_fire[indices] = True` |
| `Transpose` | `AI_VECTOR_CORE` | 3138 | 183.1 ms | 15.26 ms/step | 9.97% | fused backward 中的布局整理 |
| `Mul` | `AI_VECTOR_CORE` | 7053 | 153.8 ms | 12.82 ms/step | 8.38% | 系数构造、逐元素乘法 |
| `IndexPut` | `AI_CPU` | 12 | 120.3 ms | 10.02 ms/step | 6.55% | `counts[did_fire[name]] = 0` |
| `TopKV2` | `MIX_AIV` | 96 | 113.9 ms | 9.49 ms/step | 6.20% | encoder `topk` |
| `Cast` | `AI_VECTOR_CORE` | 7601 | 72.0 ms | 6.00 ms/step | 3.92% | dtype 转换 |
| `ScatterElementsV2` | `AI_VECTOR_CORE` | 192 | 70.8 ms | 5.90 ms/step | 3.85% | `S.scatter_add_` 路径 |
| `Add` | `AI_VECTOR_CORE` | 4609 | 57.0 ms | 4.75 ms/step | 3.10% | 各类逐元素加法 |
| `ReduceMean` | `MIX_AIV` | 2037 | 45.0 ms | 3.75 ms/step | 2.45% | 指标/归一化 |
| `Pows` | `AI_VECTOR_CORE` | 2129 | 43.7 ms | 3.64 ms/step | 2.38% | loss / 标准化相关 |
| `ReduceSum` | `MIX_AIV` | 960 | 26.7 ms | 2.23 ms/step | 1.46% | fused backward 归约 |
| `AsStrided` | `AI_VECTOR_CORE` | 1936 | 26.4 ms | 2.20 ms/step | 1.44% | 张量视图操作 |
| `ZerosLike` | `AI_VECTOR_CORE` | 412 | 23.2 ms | 1.93 ms/step | 1.26% | `S` / 临时 buffer 初始化 |
| `TensorMove` | `AI_VECTOR_CORE` | 780 | 22.9 ms | 1.90 ms/step | 1.24% | 中间张量搬运 |
| `Neg` | `AI_VECTOR_CORE` | 1160 | 16.3 ms | 1.36 ms/step | 0.89% | loss 计算 |
| `BroadcastTo` | `AI_VECTOR_CORE` | 1160 | 14.4 ms | 1.20 ms/step | 0.78% | 广播扩展 |

前六项合计已占 SAE 时间约 `77%`，其中最值得继续优化的是：

1. `EmbeddingBag`
2. `IndexPutV2`
3. `IndexPut`

---

## 6. 算子与代码映射

下面从“算子 -> 代码路径 -> 语义”三个维度说明。

### 6.1 `EmbeddingBag`

代码位置：

- `sparsify/fused_decoder.py:32`
- `sparsify/fused_encoder.py:50`

语义：

- decoder 前向：根据 `top_indices` 和 `top_acts` 从 `W_dec` 中取出对应 latent 行并做加权求和，得到重建输出。
- encoder backward 的 `grad_input`：把反向信号沿着被选中的 latent 路径再聚合回输入空间。

为什么贵：

- 本质是稀疏随机 gather + sum。
- 从 `[num_latents, d_in]` 权重表中随机读取大量行，访存局部性差。
- 更偏 memory-bound，而不是 compute-bound。

### 6.2 `IndexPutV2`

代码位置：

- `sparsify/trainer.py:364`

语义：

- 把当前 batch 中激活过的 latent 标记进 `did_fire`。

为什么贵：

- 这是高频 hook 内操作。
- 索引数量大，且重复索引很多。
- 当前语义是“标记为 True”，但底层实现并不便宜。

### 6.3 `IndexPut`

代码位置：

- `sparsify/trainer.py:506`

语义：

- 在 step 边界，把本 step 激活过的特征对应的 `num_tokens_since_fired` 清零。

为什么贵：

- 当前是 bool mask 索引赋值。
- 该路径在 Ascend 上容易落到 AI_CPU 或较差实现。

### 6.4 `TopKV2`

代码位置：

- `sparsify/fused_encoder.py:33`

语义：

- 对 encoder pre-activation 做 top-k，保留最活跃的 latent。

为什么现在不是主瓶颈：

- 虽然 top-k 是 sparse encoder 的核心步骤，但在当前 profile 中它只占 SAE 时间约 `6.2%`。
- 当前更贵的是 top-k 之后的重建、反向和 bookkeeping。

### 6.5 `Transpose` / `Mul` / `Cast` / `ReduceSum` / `ScatterElementsV2`

代码位置：

- encoder `S.scatter_add_`：`sparsify/fused_encoder.py:64`
- decoder `S.scatter_add_`：`sparsify/fused_decoder.py:63`
- decoder `grad_acts`：`sparsify/fused_decoder.py:50`
- encoder fallback `index_add_`：`sparsify/fused_encoder.py:72`
- decoder fallback `index_add_`：`sparsify/fused_decoder.py:71`

语义：

- 这些算子多数不是独立业务逻辑，而是 fused backward 内部为构造系数矩阵 `S`、做矩阵乘法前的数据整理，以及生成各类临时张量所付出的开销。

为什么仍然显著：

- 虽然已经摆脱了旧版 Python for-k 的数万次小 kernel 风暴，但现在仍有不少张量布局调整与小算子链残留。

---

## 7. 已验证的优化结论

### 7.1 有效优化

#### A. 用 `ScheduleFreeWrapperReference` 替代 XOR swap 版本

结论：有效，且收益巨大。

原因：

- 旧版 `swap()` 里使用 `bitwise_xor_` 交换大 tensor 内容。
- `BitwiseXor` 在 Ascend 上会落到 AI_CPU。
- 对 `encoder.weight` / `W_dec` 这类大参数，每次 XOR 都很贵。

当前状态：

- 该瓶颈已经不再是主 hotspot。

#### B. fused encoder / decoder backward

结论：有效，已经消除了旧版最严重的 `for-k` launch 风暴。

原因：

- 把逐个 k 的 `index_add_ / gather / bmm / reduce` 改成 `S @ X` 这一类更大粒度的稠密计算。
- 大幅减少了 kernel 数量和发射开销。

### 7.2 失败或有回退风险的优化

#### A. `did_fire` 改成 `scatter_add_`

结论：不可取，已验证会变慢。

原因：

- `524,288` 个索引往长度 `8,192` 的目标做累加。
- 平均每个位置有大量写冲突。
- NPU 上 `ScatterElementsV2` 需要处理冲突序列化。

结果：

- 在回退 profile 中，`ScatterElementsV2` 明显膨胀，整体 step time 反而变差。

#### B. “把标记问题做成计数问题”通常不划算

`did_fire` 的真实语义是：

- 只关心某个 latent 是否在本 step 至少被激活过一次。

因此：

- `index_fill_` / “set” 语义通常比 `scatter_add_` / “count” 语义更合理。

---

## 8. 剩余优化空间

本节讨论的是“从当前版本再往下压，还有多少空间”。

### 8.1 P0：`did_fire` 路径

覆盖项：

- `IndexPutV2`：15.70%
- `IndexPut`：6.55%

合计约占 SAE 时间 `22%`。

特点：

- 这部分不是算法本质难，而是实现路径不理想。
- 语义其实很简单：标记一组 feature 被激活过；然后在 step 边界按 mask 清零计数器。

优化空间判断：

- 保守：还能省 `50%`
- 激进：有机会省 `70%–90%`

可尝试方向：

1. 把 hook 内 `did_fire` 写入改成更接近“set”语义的实现，避免加法型 scatter。
2. 把 step 边界 `counts[did_fire[name]] = 0` 改成更设备友好的路径，例如 `masked_fill_` 或等价张量表达。
3. 减少 hook 内同步次数，避免每个 micro-step 都做不必要的 `all_reduce`。

### 8.2 P1：`EmbeddingBag`

覆盖项：

- `EmbeddingBag` 占 SAE 时间 `30.43%`

特点：

- 这是当前 SAE 绝对值最大的算子。
- 但它是 memory-bound 问题，不像 `did_fire` 那样属于“明显的错误实现路径”。

优化空间判断：

- 仅换 kernel、但不改数据流：大约 `10%–25%`
- 如果允许对 `top_indices` 做排序/分桶/重排：大约 `25%–40%`
- 如果进一步允许改 `k`、latent 数、hookpoint 数：上限会更高，但已属于算法/配置层优化

可尝试方向：

1. 对 `top_indices` 排序或分桶，提升访存局部性。
2. 把 decoder/encoder 的 gather 模式调整成更友好的访问顺序。
3. 如果训练目标允许，适度减小 `k`。

### 8.3 P2：`Transpose / Mul / Cast / Reduce*` 链

覆盖项：

- `Transpose`、`Mul`、`Cast`、`Add`、`ReduceSum`、`AsStrided`、`TensorMove` 等

特点：

- 它们说明 fused backward 仍有可继续合并的小算子链。
- 单个算子不一定很重，但累计起来不小。

优化空间判断：

- 预计还能再挖出 SAE 时间 `5%–10%`

可尝试方向：

1. 减少 layout 来回转换。
2. 尽量让上下游消费同一种张量布局。
3. 把逐元素链尽量融合到更大的 kernel 中。

### 8.4 `TopKV2`

当前占 SAE 时间 `6.2%`，不建议作为下一优先级。

除非：

- 想直接改 `k`
- 或计划对 top-k 和后续 gather 做深度融合

否则单独优化 `TopKV2` 的收益有限。

---

## 9. 当前建议的优化优先级

### 第一阶段：先吃确定性收益

1. 优化 `did_fire` 路径
2. 避免 hook 内高频不必要同步
3. 确保 step 边界 mask 清零不走 AI_CPU

目标：

- 先把 `IndexPutV2 + IndexPut` 这条链打掉一大半

### 第二阶段：再啃大头

1. 重看 `EmbeddingBag` 的索引组织方式
2. 评估排序、分桶、按 latent 分组的可行性
3. 必要时做更专门的 fused kernel

### 第三阶段：继续磨 fused backward 尾部

1. 压 `Transpose / Cast / TensorMove`
2. 评估 `S` 构造链是否还可进一步融合
3. 减少中间张量与 launch 数量

---

## 10. 极限空间判断

在不改训练目标的大前提下，我对当前版本的极限优化空间做如下粗估：

- `did_fire` 路径：整体 SAE 还有约 `5%–15%` 可挖
- `EmbeddingBag`：整体 SAE 还有约 `8%–15%` 可挖
- fused backward 尾部：整体 SAE 还有约 `5%–10%` 可挖

综合来看：

- 现实可达的进一步收益：`20%–30%` SAE 时间
- 非常激进、接近当前实现极限的收益：`30%–40%`

如果再往上追求，则通常需要进入：

- 调小 `k`
- 调整 hookpoint 数量
- 调整 latent 规模
- 甚至改变 sparse 编码的数据流/训练结构

---

## 11. 后续建议

为了让下一轮 profiling 更容易直接读出 wall-time 分段，建议给训练流程加显式 range 标记，例如：

- `llm_forward`
- `sae_hook_forward`
- `sae_hook_backward`
- `sae_metrics`
- `optimizer_step`
- `dead_feature_update`

这样下次 profile 不必依赖算子归类推断，可以直接从 trace 上读出：

- LLM 前向 wall-time
- SAE hook wall-time
- optimizer / bookkeeping wall-time

这会让“LLM 占多少，SAE 占多少”的讨论从工程估算变成近似精确统计。
