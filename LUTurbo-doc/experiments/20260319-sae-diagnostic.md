# Experiment: 20260319-sae-diagnostic

> 关联 idea：[ideas/sae-improvement.md](../ideas/sae-improvement.md)
> 决策树节点：B（基向量库构建）
> Worktree 分支：exp/sae-diagnostic
> 状态：planning

## 1. 目标

诊断当前 SAE 的瓶颈来源（字典质量 vs 选择质量 vs 训练配置 vs latent 结构），并通过低成本优化建立后续架构实验的基线。

## 2. 背景

当前 SAE 在 h=1024 时需要 K=128 才能保证精度。我们不知道瓶颈在哪里：
- (a) 字典本身不够好（基向量质量差）
- (b) 编码器选择不好（字典够好但选错了）
- (c) 训练配置有明显缺陷（如死特征未处理）
- (d) latent 是否有自然分组结构（决定 Group TopK 是否值得做）

本实验分三个子阶段：
- **Phase 0A**：字典与选择质量诊断（OMP / 死特征 / PCA 基线）
- **Phase 0B**：latent 分组 oracle（Group TopK 预诊断）
- **Phase 1**：低成本高信息量的训练实验（超参 + N/K + Tiling + Matryoshka）

## 3. 实现方案

### Phase 0A：字典与选择质量诊断（available now，不改 SAE）

#### 0A-1: OMP Oracle K-重构曲线（核心诊断）

新增脚本 `experiments/sae_diagnostic/oracle_k_curve.py`。

```python
def oracle_k_curve(sae, activations):
    """
    对每个样本 x，用三种方法在不同 K 下重构，对比 FVU。

    方法：
    (a) 编码器选择：TopK(Ex+b) 取前 K' 个，内积系数
    (b) OMP 选择：sklearn OMP 贪心选 K' 个基向量 + lstsq 系数
    (c) PCA 基线：前 K' 个主成分方向投影（全局低秩参考，非稀疏编码上界）

    K' = 16, 32, 48, 64, 96, 128, 192, 256
    """
    D = sae.W_dec.detach().float()  # [N, h]，float32 求解

    for x_batch in activations:
        x = x_batch.float()
        x_centered = x - sae.b_dec.float()

        # (a) 编码器 TopK — bf16 encode, float32 重构
        x_sae = x_batch.to(sae.dtype)
        preacts = x_sae @ sae.encoder.weight.T + sae.encoder.bias  # SAE dtype
        preacts_f32 = preacts.float()
        sorted_indices = preacts_f32.argsort(dim=-1, descending=True)
        for K_prime in K_values:
            idx = sorted_indices[:, :K_prime]
            coeffs = preacts_f32.gather(1, idx)
            # 全局累积 SSE 和方差，不做 batch 平均
            ...

        # (b) OMP — 逐样本，float32
        from sklearn.linear_model import OrthogonalMatchingPursuit
        D_np = D.T.cpu().numpy()  # [h, N]
        for i, x_i in enumerate(x_centered):
            for K_prime in K_values:
                omp = OrthogonalMatchingPursuit(n_nonzero_coefs=K_prime)
                omp.fit(D_np, x_i.cpu().numpy())
                ...

        # (c) PCA — 预先计算好 PCA 方向
        for K_prime in K_values:
            proj = x_centered @ pca_components[:K_prime].T
            recon = proj @ pca_components[:K_prime]
            ...
```

**关键实现细节**（参考 luturbo_experiment skill）：
- **使用 LUT 权重**，非训练 checkpoint：从 `/root/models/<Model>/lut/` 加载
- **bf16 编码，float32 求解**：编码用 SAE dtype 保证 TopK 选择一致，OMP/PCA/FVU 计算用 float32
- **全局累积指标**：累积 SSE 和方差总和，不做 batch 平均
- **hookpoint 映射**：SAE 训练在 INPUT 上，不是 output
- OMP 逐样本较慢，先用 2000 个样本，足够判断趋势

**PCA 注意**：PCA 是全局低秩可压缩性基线，不是稀疏编码的上界。OMP/SAE 使用 union-of-subspaces（每个样本自适应选基），表达力远强于 PCA 的固定子空间。PCA@32 解释方差低不意味着 OMP@32 也差。

#### 0A-2: 死特征审计

```python
def dead_feature_audit(sae, activations):
    """
    统计每个基向量在数据上的激活频率。
    同时确认训练配置（auxk_alpha, dead_feature_threshold, 训练总 token 数）。

    输出：
    - 死特征比例（激活频率 = 0 的特征数 / N）
    - 近死特征比例（激活频率 < 0.01%）
    - 激活频率直方图
    - 训练配置确认：auxk_alpha 实际值、dead_feature_threshold、训练总 token 数
    """
    feature_counts = torch.zeros(sae.num_latents, device=sae.device, dtype=torch.long)
    total_tokens = 0

    for x_batch in activations:
        x_sae = x_batch.to(sae.dtype)
        top_acts, top_indices, _ = sae.encode(x_sae)
        counts = torch.bincount(top_indices.flatten(), minlength=sae.num_latents)
        feature_counts += counts
        total_tokens += x_batch.shape[0]
```

**关键补充**：不仅看死特征比例，还要确认训练 token 数是否足以触发 `dead_feature_threshold`（默认 1000 万 token）。如果总训练 token < threshold，则 auxk 机制从未生效。

#### 0A-3: PCA 维度分析

与 0A-1 的 PCA 基线共用数据。额外输出：
- 累积解释方差比例曲线：前 K' = 16, 32, 64, 128, 256 个 PC 各解释多少方差
- 逐层逐算子分别报告

### Phase 0B：Latent 分组 Oracle（available now，不改 SAE）

新增脚本 `experiments/sae_diagnostic/latent_grouping_oracle.py`。

```python
def latent_grouping_oracle(sae, activations):
    """
    对现有 SAE 的 N 个 latent 做离线分组，测 group recall。

    分组方式：
    (1) Decoder 余弦相似度：对 D ∈ [N, h] 做 KMeans（余弦距离）
    (2) PMI 共激活图：用激活数据构建 PMI 矩阵，取 top-n 邻居做谱聚类

    对每个测试样本的真实 top-K：
    - 计算每个组的"组活跃度"= 组内被选中 latent 的激活值之和
    - 选 top-g 个最活跃的组
    - group_recall@g = |top-K ∩ 选中组内 latent 并集| / K

    参数扫描：G = {8, 16, 32, 64} × g = {2, 4, 8, 16}
    """
    D = sae.W_dec.detach().float()  # [N, h]

    # 方式 1: Decoder 相似度聚类
    D_normed = F.normalize(D, dim=1)
    for G in [8, 16, 32, 64]:
        labels = KMeans(G).fit(D_normed.cpu().numpy())

        for x_batch in activations:
            top_indices = sae.encode(x_batch.to(sae.dtype))[1]  # [batch, K]
            top_values = sae.encode(x_batch.to(sae.dtype))[0]

            for g in [2, 4, 8, 16]:
                # 计算每组活跃度，选 top-g 组
                group_mass = scatter_add(top_values, group_of[top_indices], G)
                top_groups = group_mass.topk(g).indices
                # 候选集 = 选中组的 latent 并集
                candidates = union(latents_in_group[top_groups])
                recall = |set(top_indices) & candidates| / K
```

**关键判定**：
- 对比 **随机分组基线**：同样 G、g 下随机分配 latent 到组的 group_recall（期望值 ≈ g/G）。真实分组 recall 相对随机基线的提升倍数是判断"是否有结构"的核心指标
- group_recall@g=4 显著高于随机基线且 > 85% → Group TopK 很有潜力
- group_recall@g=4 接近随机基线或 < 60% → 当前 latent 无自然分组，Group TopK 风险高
- 注意：oracle 差不完全排除 Group TopK（训练可能强制形成分组），但风险更高

### Phase 1：低成本高信息量训练实验

按代码可用性分类：

#### Available now（只改训练参数/config）

**1a: auxk_alpha × dead_feature_threshold 联合 sweep**

```bash
MODEL=/root/models/Qwen3-0.6B
DATASET=/root/fineweb-edu/sample/10BT-tokenized-qwen3-2048

# auxk_alpha sweep（同时注意 dead_feature_threshold）
for AUXK in 0 0.015625 0.03125 0.0625; do
    for DFT in 1000000 10000000; do
        python -m sparsify $MODEL $DATASET \
            --sae.k 128 --sae.expansion_factor 8 \
            --auxk_alpha $AUXK --dead_feature_threshold $DFT \
            --save_dir checkpoints/auxk_${AUXK}_dft_${DFT}
    done
done
```

**1b: Hadamard on/off**

```bash
python -m sparsify $MODEL $DATASET \
    --sae.k 128 --sae.expansion_factor 8 \
    --use_hadamard --save_dir checkpoints/hadamard_on
```

**1c: N/K sweep**

```bash
# expansion_factor × K 网格（第一轮用默认 auxk，不依赖 1a）
for EF in 4 8 16; do
    for K in 32 64 128; do
        python -m sparsify $MODEL $DATASET \
            --sae.k $K --sae.expansion_factor $EF \
            --save_dir checkpoints/nk_ef${EF}_k${K}
    done
done
# 1a 完成后，用最优 auxk 配置对关键 (N,K) 点精扫一轮
```

**1d: Tiling（已有 TiledSparseCoder 实现）**

```bash
for T in 2 4 8; do
    python -m sparsify $MODEL $DATASET \
        --sae.k 128 --sae.expansion_factor 8 \
        --num_tiles $T \
        --save_dir checkpoints/tiling_T${T}
done

# 变体：global_topk + input_mixing
python -m sparsify $MODEL $DATASET \
    --sae.k 128 --sae.expansion_factor 8 \
    --num_tiles 4 --global_topk \
    --save_dir checkpoints/tiling_T4_global

python -m sparsify $MODEL $DATASET \
    --sae.k 128 --sae.expansion_factor 8 \
    --num_tiles 4 --input_mixing \
    --save_dir checkpoints/tiling_T4_mixing
```

#### Requires minor infra change（只改损失函数）

**1e: Matryoshka 训练**

需要修改 `sparsify/trainer.py` 的损失计算部分。

```python
# 在 loss 计算中添加 Matryoshka 分支
if matryoshka:
    # 按激活值降序排列
    sorted_acts, sorted_idx = top_acts.sort(dim=-1, descending=True)
    total_loss = 0
    for K_prime, weight in [(32, 1.0), (64, 0.5), (K_max, 0.25)]:
        sub_acts = sorted_acts[:, :K_prime]
        sub_idx = sorted_idx[:, :K_prime]
        recon = sae.decode(sub_acts, sub_idx)
        total_loss += weight * fvu(x, recon)
    loss = total_loss / sum(weights)
```

**1f: 正交性正则**

需要修改 `sparsify/trainer.py` 添加正则项。

```python
# 正交性正则
active_indices = top_indices.unique()
D_active = sae.W_dec[active_indices].float()
gram = D_active @ D_active.T
orth_loss = ((gram - torch.eye(len(active_indices), device=gram.device)) ** 2).mean()
loss = fvu_loss + auxk_loss + lambda_orth * orth_loss
```

## 4. 运行方法

```bash
MODEL=/root/models/Qwen3-0.6B
LUT_DIR=/root/models/Qwen3-0.6B/lut
DATASET=/root/fineweb-edu/sample/10BT-tokenized-qwen3-2048
THRESHOLD_DIR=/root/sparsify-ascend/thresholds/Qwen3-0.6B

# Phase 0A：诊断（用 LUT 权重）
python -m experiments.sae_diagnostic.oracle_k_curve \
    --lut_dir $LUT_DIR \
    --model $MODEL \
    --dataset $DATASET \
    --num_samples 2000 \
    --output_dir experiments/sae_diagnostic/results/Qwen3-0.6B/

# Phase 0B：Latent 分组 oracle
python -m experiments.sae_diagnostic.latent_grouping_oracle \
    --lut_dir $LUT_DIR \
    --model $MODEL \
    --dataset $DATASET \
    --num_samples 4096 \
    --output_dir experiments/sae_diagnostic/results/Qwen3-0.6B/

# Phase 1：训练实验（见 §3 中的具体命令）
```

层选择：layers {0, 5, 10, 15, 20, 27} × {mlp, qkv, o_proj}

## 5. 观测指标

### Phase 0A

| 指标 | 含义 |
|------|------|
| FVU_encoder(K') | 编码器 TopK=K' 时的 FVU |
| FVU_omp(K') | OMP 选择 K' 个时的 FVU |
| FVU_pca(K') | PCA 前 K' 个 PC 的 FVU（全局低秩基线，非上界） |
| dead_ratio | 死特征比例 |
| near_dead_ratio | 激活频率 < 0.01% 的特征比例 |
| training_config | auxk_alpha / dead_feature_threshold / 训练总 token 数 |

### Phase 0B

| 指标 | 含义 |
|------|------|
| group_recall@(G,g) | top-g 组覆盖 top-K 的比例 |
| random_baseline@(G,g) | 随机分组下的 group_recall（期望 ≈ g/G） |
| recall_lift | group_recall / random_baseline（>1.5x 说明有结构） |
| group_recall_weighted@(G,g) | 按激活值加权的 group recall |
| group_balance | 各组大小分布的均匀度 |
| 最优 G/g 的候选集大小 | g×N/G（对比 C1h+C1i 的候选集） |

### Phase 1

| 指标 | 含义 |
|------|------|
| FVU@K=32, 64, 128 | 不同 K 下的重构质量 |
| p@τ=0.3, 0.5 | 需要在线补偿的维度比例 |
| dead_ratio | 训练后的死特征比例 |

**逐层逐算子报告**，每个实验都在同一评估框架下对比。

## 6. 预期结果与判定标准

### Phase 0A 判定

| 场景 | OMP(32) vs encoder(128) | 结论 | 后续 |
|------|:---:|------|------|
| 最优 | OMP(32) ≈ enc(128) | 字典够好，问题在选择 | 重点转 C1，架构实验优先级降低 |
| 中等 | OMP(32) 明显差 | 字典可改善 | 进入 Phase 1 + 实验 2 |
| 较差 | OMP(64) 也远差 | 需要根本性改变 | 实验 2 重点测残差 SAE 和 Gated |

注意：PCA 结果仅作参考。PCA@32 差不意味着 K=32 不可行（稀疏编码 >> 固定子空间）。

### Phase 0B 判定

| group_recall@g=4 | 结论 | 后续 |
|:---:|------|------|
| > 85% | latent 有自然分组 | Group TopK 高优先级，进入实验 3 |
| 60-85% | 有一定结构但不强 | Group TopK 中优先级，可尝试但风险较高 |
| < 60% | 无自然分组 | Group TopK 需通过训练强制形成，风险高，降低优先级 |

### Phase 1 判定

- auxk 调优后 FVU 下降 >10%？→ 低垂果实确认
- N/K sweep 中是否存在比 (N=8h, K=128) 更好的 Pareto 点？
- Tiling T=4 的 FVU@K=128 vs 标准 SAE？→ 分块对角可行性
- Matryoshka Top-32 FVU vs 标准 TopK=32 FVU？→ 重要性排序有效性
- Phase 1 的最优配置作为实验 2/3 的训练基线

## 7. 实际结果

（实验完成后填写）

## 8. 结论与影响

（实验完成后填写）
- Phase 0A 的 OMP 曲线直接决定实验 2 的优先级
- Phase 0B 的 group recall 决定实验 3 的优先级
- Phase 1 的最优配置作为实验 2/3 的训练基线
- 更新 decision-tree.md B 分支状态
