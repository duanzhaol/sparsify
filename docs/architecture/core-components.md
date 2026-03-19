# 核心组件

本文按模块说明 Phase 2 训练框架中的关键实现边界。重点不是把所有实验逻辑塞进训练器，而是让 SAE 架构、训练调度和实验产物各自承担清晰职责。

## `sparsify/sparse_coder.py`

`SparseCoder` 具有双重角色：

- 标准 TopK SAE 的具体实现
- 其他 SAE 变体共享行为的基类

主要职责：

- 初始化编码器、解码器权重和 `b_dec`
- 使用 top-k 进行稀疏编码
- 将稀疏 latent 解码回原始激活空间
- 计算 FVU 和可选 AuxK 损失
- 通过 `cfg.json` 与 `sae.safetensors` 保存和加载模型
- 暴露 `get_param_groups(base_lr)`、`decode()`、`forward()` 等统一接口供 `Trainer` 使用

实现要点：

- 初始化时，解码器权重从编码器权重复制
- `encode()` 会先做 `x - b_dec` 再进入融合编码器
- `decode()` 通过 `decoder_impl` 分发，不同后端自动选择实现
- 有死特征时，AuxK 会用死特征候选 latent 去拟合当前残差
- `load_any()` 会根据 `cfg.json` 中的 `architecture` 字段分发到正确的 SAE 子类

## `sparsify/gated_sparse_coder.py`

`GatedSparseCoder` 表示“选择”和“系数”解耦的 SAE：

- `W_gate` 负责给每个 latent 打路由分数
- `W_mag` 负责给被选中的 latent 提供连续系数
- 解码和损失计算仍复用基类逻辑

这一路径的重点不是推理期直接使用 gate 编码器，而是观察它训练出的解码器字典质量是否优于标准 TopK SAE。

## `sparsify/jumprelu_sparse_coder.py`

`JumpReLUSparseCoder` 采用 JumpReLU-fixedK 设计：

- 每个 latent 有可学习阈值
- 训练时用 STE 近似跨过硬阈值
- 输出端仍裁成固定 top-k，保持 `ForwardOutput` 和 `decode()` 接口稳定

这不是原生变长 JumpReLU SAE，而是为 Phase 2 小 K 对比实验设计的固定形状版本。

## `sparsify/group_topk_sparse_coder.py`

`GroupTopKSparseCoder` 是实验性结构化 SAE 路径。

- `group_router` 先选少量组
- 在选中组的 latent 并集中再做全局 top-k
- v0 使用 hard routing，主要用于验证训练框架、指标定义和 artifact 结构

注意：

- v0 hard routing 下，编码器能通过被选中组获得梯度，但路由器本身不直接从 FVU 获得可用梯度
- 因此 v0 的训练结果不应被当作结构化 SAE 可行性的正式判据
- 如果要正式比较效果，需要在后续版本中引入可导近似或辅助路由损失

## `sparsify/fused_encoder.py`

`FusedEncoder` 使用自定义 autograd 实现编码器前向与反向。

核心路径：

- 前向：`linear -> relu -> topk`
- 反向：优先使用 scatter + matmul
- 内存压力较大时回退到 gather / `bmm` / `index_add_`

前向输出是 `EncoderOutput`，包含：

- `top_acts`
- `top_indices`
- `pre_acts`

## `sparsify/fused_decoder.py`

`FusedDecoder` 是 `SparseCoder.decode()` 使用的稀疏解码后端。

核心路径：

- 将 top-k 激活视为稀疏系数矩阵
- 快速路径使用 scatter + matmul
- 系数矩阵过大时回退到 `embedding_bag` 风格实现

调用细节：

- 上层传入 `self.W_dec.mT`
- `fused_decode()` 内部会转换回解码器所需的行布局

## `sparsify/tiled_sparse_coder.py`

`TiledSparseCoder` 将宽激活切分为多个 tile，每个 tile 训练一个 SAE。

支持能力：

- 每个 tile 独立执行 top-k
- 跨 tile 的全局 top-k
- 可选 `input_mixing`（tile 维度可学习混合）

实现要点：

- `num_latents` 是所有 tile 的总和
- `b_dec` 以拼接视图形式对外暴露
- `set_b_dec_data()` 支持用一个全局均值向量初始化 tile 偏置
- `global_topk` 路径当前不启用 AuxK（全局 dead mask 逻辑尚未实现）

## `sparsify/trainer.py`

`Trainer` 负责整体训练调度。

主要职责：

- 解析 hookpoint
- 按 `hookpoint × seed` 初始化 SAE，并根据 `sae.architecture` 分发到正确实现
- 在 Transformer 上注册 forward hook
- 在 hook 内执行 SAE 前向、损失计算与反向传播
- 维护死特征统计
- 处理 checkpoint 保存、本地 artifact 写入和 W&B 指标上报

此外还负责：

- 运行名生成
- 可配置优化器构造
- 可选 Matryoshka / 正交正则 / 残差训练路径
- 多卡指标归约
- Hadamard 旋转对象按需延迟初始化

`Trainer` 与 SAE 的接口契约是：

- `sae.parameters()`
- `sae.num_latents`
- `sae.forward(...) -> ForwardOutput`
- `sae.decode(...)`
- `sae.get_param_groups(base_lr)`
- `sae.W_dec` 和 `sae.cfg`

其中 `W_dec` 目前仍是共享约定，用于正交正则等逻辑；如果未来变体继续增多，再考虑进一步抽象出更窄的接口。

## `sparsify/metrics_logger.py`

`MetricsLogger` 负责把一次训练运行固化为结构化 artifact。

核心输出：

- `manifest.json`：实验身份证，记录 run 名称、架构、模型、数据集、hookpoint、git 信息等
- `metrics.jsonl`：逐步训练指标
- `summary.json`：训练结束后的摘要指标

这一层的目标是让训练结果不只存在于 W&B，而是能被后续脚本稳定消费和聚合。

## `sparsify/device.py`

`device.py` 提供 CUDA / NPU / CPU 统一设备接口，避免上层逻辑直接依赖单一后端。

常用函数：

- `get_device_type()`
- `get_device_string()`
- `get_dist_backend()`
- `device_autocast()`
- `create_event()`、`synchronize()`

仓库仍保留 NPU 支持，但文档默认路径为 CUDA。

## `sparsify/utils.py`

`utils.py` 收纳训练与导出共用的通用函数。

关键函数：

- `resolve_widths()`：一次探测获取模块维度
- `partial_forward_to_layer()`：提前截断前向
- `decoder_impl`：按后端选择解码实现

其他常用工具：

- `get_layer_list()`：定位主 Transformer 层列表
- `get_max_layer_index()`：从 hookpoint 计算最远目标层
- `simple_parse_args_string()`：解析 `name=sample-10BT` 这类 CLI 参数
