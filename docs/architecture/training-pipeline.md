# 训练流水线

本页描述由 `sparsify/__main__.py` 和 `sparsify/trainer.py` 实现的当前端到端训练流程。

最重要的实践细节是 SAE 训练是钩子驱动和在线的：Sparsify 在训练前不会预计算和缓存激活值到磁盘。相反，Transformer 前向和 SAE 更新在每个训练步骤中交错进行。

## 1. CLI 入口和制品加载

`python -m sparsify` 通过 `sparsify/__main__.py` 进入。

高层步骤：

1. 如果存在 `LOCAL_RANK`，初始化分布式运行时
2. 解析 `RunConfig`
3. 使用 `AutoModel.from_pretrained()` 加载模型
4. 加载 Hugging Face 数据集或 memmap 数据集
5. 如需要，即时分词
6. 构建 `Trainer` 并启动 `fit()`

额外运行时细节：

- 模型使用 `AutoModel.from_pretrained()` 加载，而非 `AutoModelForCausalLM`
- 当 `is_bf16_supported()` 表明当前后端支持时，请求 bf16
- 在 DDP 模式下，样本在分片前被修剪，以便所有 rank 看到相等的工作量且不会死锁

## 2. 钩入点解析

在 `Trainer.__init__()` 中：

- 显式钩入点模式使用 `expand_range_pattern()` 扩展
- 模式与 `model.base_model.named_modules()` 匹配
- 如果未提供钩入点，训练器从模型层列表推断层模块
- 钩入点解析后应用 `layer_stride`

如果也未提供显式 `layers`，训练器默认为所有 Transformer 层。

## 3. 宽度解析和 SAE 构建

在创建 SAE 之前，训练器使用 `hook_mode="input"` 语义调用 `resolve_widths()`。这很重要：当前训练由模块输入驱动。

然后，对于每个已解析的钩入点和每个初始化种子：

- 构建 `SparseCoder`，或
- 当 `num_tiles > 1` 时构建 `TiledSparseCoder`

这意味着当 `init_seeds` 包含多个值时，一个钩入点可以映射到多个 SAE 实例。

## 4. 优化器设置

当前主线优化器路径是：

- `SignSGD`
- 由 `ScheduleFreeWrapperReference` 包装

如果省略 `lr`，训练器使用基于潜在变量计数的缩放规则。

活跃训练器中不再有多优化器分支。当前路径有意精简。

## 5. 批次处理

在 `fit()` 中，训练器：

- 冻结 Transformer 主干
- 创建 `DataLoader`
- 如果分布式训练激活，在首次使用时用 DDP 包装 SAE
- 跟踪令牌计数、时间和聚合指标

进度条计数批次，但优化器更新仅在每 `grad_acc_steps` 发生。

## 6. 前向钩子和 SAE 更新

对于每个批次，训练器在所有选定模块上注册前向钩子。

在钩子实现内部：

- 解包模块输入张量
- 展平批次和序列维度
- 可选应用 Hadamard 旋转
- 在非微调情况下，在第一步初始化解码器偏置
- 运行 SAE 前向传播
- 收集激活的潜在变量索引用于死特征跟踪
- 累积 FVU、AuxK 和可选的超出指标
- 在 `out.fvu + auxk_alpha * out.auxk_loss` 上执行局部反向传播

值得记住的细节：

- 激活从 `[batch, seq, hidden]` 展平为 `[batch * seq, hidden]`
- 解码器偏置初始化使用第一个观察到的批次的平均激活，可选跨 rank 全归约
- 启用时，每次前向前重新应用解码器归一化
- 启用 Hadamard 预处理时，超出指标在未旋转的张量上计算
- 当 DDP 激活时，钩子在 `no_sync()` 下运行，除同步边界外

这意味着 SAE 优化在 Transformer 前向期间内联发生，而非在单独的离线激活缓存阶段。

## 7. 部分前向优化

`Trainer` 计算所选钩入点所需的最大层索引。如果可能，它使用 `sparsify/utils.py` 中的 `partial_forward_to_layer()` 在所有所需钩子触发后提前停止 Transformer 前向。

这在早期或中层子集上训练 SAE 时特别有用。

## 8. 优化器步骤和死特征簿记

在梯度累积边界，训练器：

- 启用解码器归一化时，移除解码器平行梯度分量
- 运行 `optimizer.step()` 和 `optimizer.zero_grad()`
- 更新 `num_tokens_since_fired`
- 将步骤中激活的潜在变量计数器归零
- 使用 `MIN` 全归约跨 rank 同步这些计数器

簿记流程有意与旧版本不同：

- 训练器在钩子执行期间收集潜在变量索引
- 在步骤结束时，连接这些索引并直接归零计数器
- 它避免了在某些后端上昂贵的旧每前向 bool-scatter 路径

死特征簿记后，训练器：

- 更新 `total_tokens`
- 如果达到 `max_tokens`，提前停止
- 保存周期性检查点
- 将聚合指标记录到 Weights & Biases

当前实现使用收集的潜在变量索引，而非旧的每前向 bool-scatter 路径。

## 9. 保存和日志

训练器定期：

- 通过 `CheckpointMixin` 保存检查点
- 可选保存每个钩入点的最佳检查点
- 将指标记录到 Weights & Biases

检查点辅助函数位于 `sparsify/checkpoint.py`，也处理分块检查点加载。

保存的状态包括：

- SAE 权重
- 优化器状态
- `global_step`
- `total_tokens`
- 死特征计数器
- 最佳损失值
- 可选 Hadamard 旋转状态

## 10. 实用阅读地图

如果你想在代码中理解当前训练路径，按此顺序阅读：

1. `sparsify/__main__.py`
2. `sparsify/config.py`
3. `sparsify/trainer.py`
4. `sparsify/sparse_coder.py`
5. `sparsify/tiled_sparse_coder.py`
6. `sparsify/checkpoint.py`

## 11. 心智模型

如果你想要当前流水线的一个紧凑心智模型，它是：

1. 运行 Transformer 刚好足够触发选定的钩子
2. 在每个钩子内部，将模块输入视为 SAE 训练数据
3. 立即计算局部重建损失
4. 在几个批次上累积梯度
5. 步进一个共享的 SAE 优化器
6. 持久化检查点供后续阈值计算和 LUT 导出
