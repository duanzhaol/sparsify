# SAE Int8 I/O 量化训练设计

## 1. 背景

当前 `product_key_expert_jumprelu` SAE 已经完成了多轮训练后量化（PTQ）评估，结论比较明确：

- `expert-only W8A8` 误差几乎可以忽略；
- `full-encoder W8A8` 仍然非常稳定；
- `full-encoder + decoder W8` 和 `full-encoder + decoder W8A8` 也都处于可接受范围；
- 对当前 SAE 主干而言，8bit 已经不是主要精度瓶颈。

但这些结果仍然属于“训练后量化”视角。下一步需要回答的问题是：

1. 如果最终部署形态希望是 `int8 input -> SAE -> int8 output`，训练目标是否也应该改成更贴近这个部署形态；
2. 是否有必要从“输入输出量化感知训练”开始，逐步推进到更完整的 SAE QAT；
3. 怎样在尽量少扰动现有训练代码的前提下，先做一版最小可行验证。

## 2. 目标

本设计要解决两个层面的事情。

### 2.1 总体路线

给出 SAE 量化训练的整体阶段规划，明确每个阶段的目标、边界和风险。

### 2.2 第一阶段实现目标

设计一个最小可行版本，只让训练显式感知：

- 输入是 int8 友好的；
- 输出也是 int8 友好的；
- 但 SAE 参数、梯度、优化器状态仍然保持浮点。

这个阶段的核心不是“纯 int8 训练”，而是“输入输出量化感知训练（I/O QAT）”。

## 3. 非目标

本阶段不做以下事情：

- 不做纯 int8 训练；
- 不修改优化器状态精度；
- 不在第一版就把所有 SAE 内部线性层全部纳入 fake quant；
- 不在第一版就接入真实 CPU int8 kernel；
- 不在第一版就把在线补偿一起纳入量化训练。

## 4. 整体阶段规划

建议将 SAE 量化训练拆成五个阶段，从低风险到高风险逐步推进。

### Phase 0：统一评估口径

先不改训练，只统一训练后评估和训练中日志的口径，避免后续误判。

建议同时跟踪以下指标：

- `FVU_fp_teacher`：重构结果对原始浮点目标的误差；
- `FVU_deploy`：重构结果对部署态量化目标的误差；
- `exceed_alpha_0.5_fp_teacher`；
- `exceed_alpha_0.5_deploy`；
- `quant_floor = FVU(x_fp, DQ(Q(x_fp)))`。

这样可以区分：

- 是任务本身因为 int8 离散化而变简单了；
- 还是模型真的更适配部署目标了。

### Phase 1：I/O 量化感知训练

只对输入和输出插入 fake quant，训练主图仍然使用当前 SAE 架构和浮点参数。

这一阶段回答的问题是：

- 如果训练时显式模拟 `int8 input + int8 output`，是否能让模型更适应部署形态；
- 在不大改 SAE 架构的情况下，这件事是否已经值得做。

### Phase 2：主干 Partial QAT

在 Phase 1 稳定的基础上，再逐步把 SAE 主干中的关键线性层纳入 fake quant。

建议顺序：

1. `W_dec`
2. `expert_encoders`
3. `left_router` / `right_router`

这个顺序和已有 PTQ 结果一致，风险也最低。

### Phase 3：部署分布对齐训练

用量化 backbone 采集激活，再做 finetune 或重新训练。

这个阶段的目标是解决训练分布与部署分布之间的失配问题，尤其适用于最终部署本身就是 `W8A8 backbone` 的场景。

### Phase 4：在线补偿量化与补偿专用 QAT

如果主干量化后，在线补偿在 CPU 上成为新的瓶颈，再单独评估在线补偿是否值得量化，以及是否需要在线补偿专用 QAT。

### Phase 5：真实 CPU 部署验证

把“量化仿真 + QAT 可行”推进到“CPU 上真实 kernel 确实更快”。

这一阶段才重点关注：

- int8 kernel；
- 内存布局；
- pack/unpack 代价；
- 稀疏 gather 的真实性能。

## 5. 量化训练方法版图

从方法上，可以把后续路线分为四类。

### 5.1 PTQ

训练后量化，当前已经验证得非常成功。它的价值是快速摸清边界，缺点是训练目标和部署目标不完全一致。

### 5.2 I/O-QAT

只让训练显式看到输入输出量化噪声，不动 SAE 内部参数表示。

优点：

- 改动小；
- 风险低；
- 最贴近当前的阶段目标。

### 5.3 Partial QAT

只对部分模块做 fake quant，例如 decoder 或 encoder 主干。

优点是收益和风险比较均衡，适合在 I/O-QAT 跑通后继续推进。

### 5.4 Full QAT / Deploy-aligned QAT

把输入、输出、内部主干线性层乃至 backbone 激活分布都对齐部署形态。

优点是部署一致性最高，缺点是改动大、训练更不稳定。

## 6. 第一阶段设计原则

第一阶段遵循四条原则。

### 6.1 尽量只改训练外壳

优先修改：

- `sparsify/config.py`
- `sparsify/trainer.py`

尽量不在第一版就深入改动 `sparsify/sparse_coder.py` 中的所有 SAE 架构实现。

### 6.2 保持现有 SAE 结构稳定

第一阶段不改变：

- `product_key_expert_jumprelu` 的路由逻辑；
- JumpReLU threshold 逻辑；
- TopK / route selection 逻辑；
- 优化器和参数精度。

### 6.3 训练目标对齐部署，但不放弃浮点 teacher

训练应该感知 int8 I/O 的存在，但不应该把目标完全退化成“只拟合量化桶”。

### 6.4 与现有量化研究口径一致

第一阶段的 fake quant 规则应尽量复用当前 `quantization/` 目录中的假设：

- 对称 int8；
- activation 使用 per-token dynamic scale；
- 使用 `Q -> DQ` 的 fake quant；
- 用 STE 近似传递梯度。

## 7. 第一阶段前向链路

第一阶段训练链路定义如下：

```text
x_fp
 -> fake_quant_in
 -> x_qdq
 -> SAE
 -> xhat_fp
 -> fake_quant_out
 -> xhat_qdq
 -> loss
```

其中：

- `x_fp`：原始浮点激活；
- `x_qdq`：输入 fake quant 后的张量；
- `xhat_fp`：SAE 浮点输出；
- `xhat_qdq`：输出 fake quant 后的张量。

要点如下：

- SAE 前向吃的是 `x_qdq`，而不是原始 `x_fp`；
- loss 主要对 `xhat_qdq` 计算，因为目标是适应 `int8 output` 部署；
- 但 teacher 仍然保留原始 `x_fp` 的监督作用。

## 8. 第一阶段 loss 设计

第一阶段采用双目标 loss。

定义：

- `target_fp = x_fp`
- `target_deploy = fake_quant(x_fp)`
- `pred_deploy = fake_quant(xhat_fp)`

主损失定义为：

```text
L_main = L_fp + lambda_deploy * L_deploy
```

其中：

- `L_fp = FVU(pred_deploy, target_fp)`
- `L_deploy = FVU(pred_deploy, target_deploy)`

推荐初始值：

- `lambda_deploy = 0.25`

这样设计的原因是：

- `L_fp` 负责保住原始 BF16 teacher 语义；
- `L_deploy` 负责让模型适应 int8 部署目标；
- `pred_deploy` 而不是 `xhat_fp` 进入 loss，是因为当前关心的是“最终输出也会被量化”。

## 9. 第一阶段量化配置

建议在 `TrainConfig` 中新增以下配置项。

- `io_quant_mode`
  - `off`
  - `qat_io_int8`

- `io_quant_bits`
  - 第一版固定为 `8`

- `io_quant_granularity`
  - 第一版固定为 `per_token`

- `io_quant_clip_mode`
  - 第一版默认 `absmax`

- `io_loss_mode`
  - `fp_teacher`
  - `dual_target`
  - `deploy_target`

- `io_loss_deploy_weight`
  - 第一版默认 `0.25`

推荐默认配置：

```text
io_quant_mode = qat_io_int8
io_quant_bits = 8
io_quant_granularity = per_token
io_quant_clip_mode = absmax
io_loss_mode = dual_target
io_loss_deploy_weight = 0.25
```

## 10. 第一阶段代码改动边界

### 10.1 `sparsify/config.py`

新增 I/O 量化训练相关配置字段，并做基础校验。

### 10.2 `sparsify/trainer.py`

这是第一阶段的核心改动位置。

在 hook 到原始激活 `acts` 后，训练流程改成：

1. 保留 `acts_fp`
2. 如果开启 I/O 量化训练，则构造 `acts_in = fake_quant(acts_fp)`；否则 `acts_in = acts_fp`
3. SAE 前向使用 `acts_in`
4. 得到 `out.sae_out` 后，构造：
   - `recon_fp = out.sae_out`
   - `recon_deploy = fake_quant(recon_fp)`
   - `target_deploy = fake_quant(acts_fp)`
5. 用 trainer 重新计算：
   - `fvu_fp_teacher`
   - `fvu_deploy`
   - `quant_floor`
6. 用新的 `L_main` 替代单一 `out.fvu` 作为主损失
7. 保留：
   - `auxk_loss`
   - router regularization
   - 现有 dead-feature 逻辑

### 10.3 `sparse_coder.py`

第一版尽量不改或只做极小改动。

理由是：

- 当前所有 SAE 架构都已经依赖其各自的 `forward` 行为；
- 如果直接在架构内部大量加入 fake quant，改动面会过大；
- 第一阶段真正想验证的是“训练目标是否应该对齐 int8 I/O”，而不是“立刻把所有内部 matmul 训练都量化化”。

## 11. 第一阶段监控指标

建议训练和验证同时记录以下指标：

- `fvu_fp_teacher`
- `fvu_deploy`
- `quant_floor`
- `exceed_alpha_0.5_fp_teacher`
- `exceed_alpha_0.5_deploy`
- `input_clip_rate`
- `output_clip_rate`
- `input_scale_mean`
- `output_scale_mean`

其中：

- `quant_floor` 用来说明 int8 本身造成的误差下界；
- `clip_rate` 用来监控是否出现过于激进的裁剪。

## 12. 第一阶段成功标准

如果满足以下条件，就认为第一阶段值得继续推进：

1. `fvu_deploy` 相比原始 BF16 训练有改善，或者至少更加稳定；
2. `fvu_fp_teacher` 不出现明显恶化；
3. `exceed_alpha_0.5_deploy` 不恶化，最好有所改善；
4. `input_clip_rate` 和 `output_clip_rate` 不出现异常升高；
5. 训练过程稳定，没有出现明显 collapse 或梯度异常。

## 13. 风险与应对

### 风险 1：看起来指标变好，但只是目标变简单了

应对：同时报告 `fvu_fp_teacher`、`fvu_deploy` 和 `quant_floor`，避免只看单一口径。

### 风险 2：输出量化后 teacher 约束被削弱

应对：第一阶段默认使用双目标 loss，而不是只拟合量化目标。

### 风险 3：I/O 量化已经足够难，内部结构再量化会导致训练不稳

应对：第一阶段先不动内部主干线性层，把结构性量化推迟到 Phase 2。

### 风险 4：训练时量化收益有限

应对：第一阶段就是为了用最小改动验证这件事；如果收益不明显，可以及时止损，继续以 PTQ 为主。

## 14. 结论

建议正式启动 SAE 量化训练，但第一步不要做“纯 int8 训练”，也不要直接全链路 QAT。

当前最合理的路线是：

1. 先做 `Phase 1: I/O 量化感知训练`
2. 只改 `config + trainer + 少量 quant helper`
3. 通过双目标 loss 同时约束浮点 teacher 和部署态量化目标
4. 如果 Phase 1 成功，再推进到 SAE 主干 Partial QAT

这条路线最符合当前已有 PTQ 结果，也最符合“低风险验证 + 快速收敛结论”的要求。
