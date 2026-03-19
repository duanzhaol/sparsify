# 训练流程

本文说明当前代码主线中的训练流程，对应实现在 `sparsify/__main__.py` 和 `sparsify/trainer.py`。

关键结论：Sparsify 采用在线训练。系统不会先把激活值整体缓存到磁盘，而是在 Transformer 前向过程中通过 hookpoint 实时获取激活并更新 SAE。

## 1. CLI 入口与数据准备

命令 `python -m sparsify` 从 `sparsify/__main__.py` 进入，主流程如下：

1. 检查 `LOCAL_RANK`，按需初始化 DDP
2. 解析 `RunConfig`
3. 用 `AutoModel.from_pretrained()` 加载模型
4. 加载数据集（Hugging Face 或 memmap）
5. 数据未分词时做在线分词
6. 创建 `Trainer`，执行 `fit()`

补充细节：

- 模型加载使用 `AutoModel`，不是 `AutoModelForCausalLM`
- 若后端支持，模型优先使用 bf16 dtype
- DDP 模式下会先裁剪样本再分片，避免不同 rank 工作量不一致导致死锁

## 2. hookpoint 解析

在 `Trainer.__init__()` 中，hookpoint 的处理顺序是：

- 先展开范围语法（`expand_range_pattern()`）
- 再用 `model.base_model.named_modules()` 做模式匹配
- 若未显式传入 hookpoint，则按模型层列表自动生成
- 最后再应用 `layer_stride`

如果 `layers` 和 `hookpoints` 都未提供，默认覆盖全部 Transformer 层。

## 3. 宽度探测与 SAE 初始化

训练前会调用 `resolve_widths(..., hook_mode="input")` 探测输入维度。当前实现路径明确使用模块输入训练 SAE。

随后按 `hookpoint × seed` 组合创建 SAE：

- `num_tiles == 1` 时使用 `SparseCoder`
- `num_tiles > 1` 时使用 `TiledSparseCoder`

因此当 `init_seeds` 包含多个值时，一个 hookpoint 会对应多个 SAE 实例。

## 4. 优化器与训练步长

当前主线只保留一条优化器路径：

- `SignSGD`
- 外层包 `ScheduleFreeWrapperReference`

若未显式设置 `lr`，则按潜在维度规模应用默认缩放规则。

补充说明：

- 进度条按 batch 计数
- 真正的优化器 `step` 按 `grad_acc_steps` 触发

## 5. hook 回调内训练逻辑

每个 batch 都会临时注册 forward hook。回调中的核心步骤如下：

- 取模块输入并展平为 `[batch * seq, hidden]`
- 按配置决定是否执行 Hadamard 旋转
- 首步（且非 finetune）用激活均值初始化 `b_dec`
- 执行 SAE 前向，得到重建输出与稀疏索引
- 记录激活 latent 索引，用于后续死特征统计
- 累积 FVU / AuxK / exceed 指标
- 立即执行局部反向：`out.fvu + auxk_alpha * out.auxk_loss`

DDP 场景下，非同步边界会使用 `no_sync()` 降低不必要的梯度通信。

## 6. 部分前向优化

训练器会根据选定 hookpoint 计算最远目标层。若目标层不是最后一层，会调用 `partial_forward_to_layer()` 提前截断前向，减少无关层计算。

这个优化在只训练前层或中层 hookpoint 时效果最明显。

## 7. 优化器更新与死特征统计

在梯度累积边界，训练器会执行以下操作：

- 处理与解码器方向相关的梯度投影
- 执行 `optimizer.step()` 和 `zero_grad()`
- 更新 `num_tokens_since_fired`
- 把本步激活过的 latent 计数清零
- 使用 `MIN` all-reduce 在多卡间同步计数状态

当前实现采用“先收集索引、步末集中清零”，避免历史版本逐次 bool scatter 的高开销路径。

## 8. 保存与日志

训练过程中会周期性执行：

- 常规 checkpoint 保存（`CheckpointMixin`）
- 可选 best checkpoint 保存
- Weights & Biases 指标上报

保存内容包括 SAE 权重、优化器状态、`global_step`、`total_tokens`、死特征计数、最佳损失，以及可选的 Hadamard 状态。

## 9. 推荐阅读顺序

如需快速理解当前训练路径，建议按以下顺序阅读代码：

1. `sparsify/__main__.py`
2. `sparsify/config.py`
3. `sparsify/trainer.py`
4. `sparsify/sparse_coder.py`
5. `sparsify/tiled_sparse_coder.py`
6. `sparsify/checkpoint.py`
