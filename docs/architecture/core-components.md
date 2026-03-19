# 核心组件

本文按模块说明当前代码主线中的关键实现。

## `sparsify/sparse_coder.py`

`SparseCoder` 是标准 SAE 实现，负责完整的编码-解码训练闭环。

主要职责：

- 初始化编码器、解码器权重和 `b_dec`
- 使用 top-k 进行稀疏编码
- 将稀疏 latent 解码回原始激活空间
- 计算 FVU 和可选 AuxK 损失
- 通过 `cfg.json` 与 `sae.safetensors` 保存和加载模型

实现要点：

- 初始化时，解码器权重从编码器权重复制
- `encode()` 会先做 `x - b_dec` 再进入融合编码器
- `decode()` 通过 `decoder_impl` 分发，不同后端自动选择实现
- 有死特征时，AuxK 会用死特征候选 latent 去拟合当前残差

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
- 按 `hookpoint × seed` 初始化 SAE
- 在 Transformer 上注册 forward hook
- 在 hook 内执行 SAE 前向、损失计算与反向传播
- 维护死特征统计
- 处理 checkpoint 保存和 W&B 指标上报

此外还负责：

- 运行名生成
- 多卡指标归约
- Hadamard 旋转对象按需延迟初始化

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
