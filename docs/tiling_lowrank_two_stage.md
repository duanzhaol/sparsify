# ANNS + 精算 的 SAE 编码器推理方案（面向 LUT，CPU）

## 1. 问题背景与现状

当前推理流程是：输入激活 `x` → 编码器产生 top-k latent 索引与激活 → 使用查找表（LUT）得到输出。  
由于 LUT 预先将每个 latent 的 decoder 结果乘上大模型权重，推理阶段几乎不再消耗 decoder 计算量；瓶颈完全在编码器：

- 传统编码器：`pre_acts = ReLU(W_enc x + b_enc)`，计算量约 `O(D * M)`。
- 在 CPU 上，`D * M` 的全量矩阵乘法通常是最慢的部分。

因此，核心目标是：**在 CPU 上显著降低编码器计算，同时尽量保持 top-k 的准确性**。

## 2. 目标与约束

目标：
- 极致降低 CPU 推理阶段的编码器开销。
- 保持 top-k 选择的准确性，避免召回损失导致重构精度下降。
- 与 LUT 推理流程兼容（decoder 计算不作为瓶颈）。

约束：
- 允许保留全秩 `W_enc` 用于精算。
- 允许离线构建索引（推理时只查询）。
- 需要显式处理 bias 与 ReLU。

## 3. 方案总览（ANNS + 精算）

采用 **Search-and-Refine**：

1) **Stage 1：ANNS 粗筛**  
使用 HNSW 等 ANNS 索引，在 CPU 上快速返回 `K_coarse` 个候选 latent IDs。

2) **Stage 2：精算重排**  
仅对候选 IDs 用全精度 `W_enc` 做点积 + `b_enc`，ReLU 后重排取最终 top-k。

3) **Stage 3：LUT 重构**  
用最终 top-k 索引与激活查表输出。

这套流程只需要保留全秩 `W_enc` 用于精算，不再依赖低秩或 tiling 才能获得速度。

## 4. 训练流程（简化）

### 4.1 训练全秩 SAE

目的：得到高质量的全秩 `W_enc`、`b_enc`、`W_dec`、`b_dec`。

步骤：
- 正常训练 SAE（FVU / AuxK / Multi-TopK）。
- 保存全秩 `W_enc` 与 `b_enc`。
- 预计算 LUT：`LUT[i] = W_dec[i] @ W_lm`。

备注：  
推理期使用 ANNS，与训练方式无耦合；无需额外蒸馏。

## 5. 索引构建（离线）

### 5.1 关键输入

- 索引向量：`W_enc` 的每一行（shape: `M x D`）。
- 查询向量：`x_centered = x - b_dec`（需与训练时编码一致）。

### 5.2 bias 处理策略

推荐：**Stage 1 忽略 bias，Stage 2 精算时加回**。  
原因：bias 会显著增加索引维度与复杂度，但 Stage 2 精算本来就会补齐。

可选：扩维处理 bias  
将 `x' = [x_centered, 1]`，`w' = [w, b_enc]`，直接在 ANNS 里做 inner product。  
优点是搜索更贴近真实分数，代价是索引维度和内存增加。

### 5.3 HNSW 参数建议（起点）

- `M`: 16–48（连接度，影响精度与内存）
- `efConstruction`: 200–400（构建质量）
- `efSearch`: 64–256（召回与速度权衡）
- 使用 **inner product** metric

## 6. 推理流程（详细）

每个输入 `x` 的推理：

**Stage 1：ANNS 粗筛**
- 计算 `x_centered = x - b_dec`
- 使用 HNSW 搜索得到 `K_coarse` 个候选 IDs

**Stage 2：全精度精算**
- 取候选行 `W_enc[ids]` 与 `b_enc[ids]`
- 计算 `scores = x_centered @ W_enc[ids].T + b_enc[ids]`
- `acts = ReLU(scores)`
- 取最终 top-k（可按全局 top-k）

**Stage 3：LUT 重构**
- 用最终 top-k 索引与激活累加 LUT 向量（再加 bias）

## 7. 复杂度直观估计

- Stage 1：`O(log M)` 或 `O(M^alpha)`（与 HNSW 参数相关）
- Stage 2：`O(K_coarse * D)`  
当 `K_coarse << M` 时，精算开销可控且远小于全量矩阵乘法。

## 8. 关键超参与建议

- `K_coarse`: 取 `8*k` 起步，精度不够时增大（如 `16*k`）
- `efSearch`: 提升召回率的关键参数，召回不足优先调它
- `M`: 增大可提升精度但占用更多内存

建议流程：
1) 固定 `k` 与 `K_coarse`。  
2) 扫描 `efSearch`，以 top-k 召回率为主指标。  
3) 以最小 `K_coarse` 满足召回阈值为目标，再平衡速度。

## 9. 可能风险与缓解

- **召回不足**：增大 `K_coarse` 与 `efSearch`。
- **bias 误差**：Stage 2 一定要加 `b_enc`。
- **负激活干扰**：Stage 2 必须 ReLU 后再排序。
- **索引过期**：训练后需重新构建索引。

## 10. 结论

在 CPU 推理场景中，**ANNS + 精算** 是最具性价比的方案：

- ANNS 负责快速召回候选；
- 精算保证 top-k 的准确性；
- 与 LUT 推理流程天然兼容。

只要把召回与精算做对，这一方案通常优于低秩或 tiling 的矩阵乘法路径。
