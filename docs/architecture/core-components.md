# 核心组件

本页总结当前代码库中的主要实现单元。

## `sparsify/sparse_coder.py`

`SparseCoder` 是标准 SAE 模块。

职责：

- 构建编码器、解码器和解码器偏置
- 使用 top-k 稀疏选择编码激活值
- 将 top-k 潜在变量解码回激活空间
- 计算 FVU 和可选的 AuxK 损失
- 通过 `cfg.json` 和 `sae.safetensors` 保存和加载检查点

关键设计点：

- 解码器行在初始化时镜像编码器权重
- 解码器归一化是可选的，但默认启用
- `forward()` 返回 `sae_out`、潜在变量激活、潜在变量索引、FVU 和 AuxK 损失
- `encode()` 在调用融合编码器前通过减去 `b_dec` 居中输入
- `decode()` 通过 `decoder_impl` 分发，后者在加速器后端选择融合解码器
- AuxK 重用死潜在变量候选来重建当前残差（当存在死特征时）

## `sparsify/fused_encoder.py`

`FusedEncoder` 使用自定义自动微分路径实现编码器前向和反向。

主要思想：

- 前向：`linear -> relu -> topk`
- 反向：当密集中间结果适合内存时使用 scatter-plus-matmul
- 回退：对于更大的情况使用 gather 和 `bmm` / `index_add_`

这保持了核心 top-k SAE 路径在加速器后端上的效率。

前向结果是包含以下内容的 `EncoderOutput`：

- `top_acts`
- `top_indices`
- `pre_acts`

## `sparsify/fused_decoder.py`

`FusedDecoder` 提供 `SparseCoder.decode()` 使用的稀疏解码路径。

主要思想：

- 将 top-k 潜在变量视为稀疏系数矩阵
- 对快速路径使用 scatter-plus-matmul
- 当密集系数矩阵太大时回退到 `embedding_bag` 风格逻辑

该实现也用于避免历史 NPU 流程中的弱后端行为。

在调用点，`SparseCoder.decode()` 传递 `self.W_dec.mT`，`fused_decode()` 将其转置回行优先解码器形式以供自定义函数使用。

## `sparsify/tiled_sparse_coder.py`

`TiledSparseCoder` 将激活维度拆分为分块，每个分块训练一个 SAE。

支持的模式：

- 每分块 top-k
- 在连接的分块预激活上进行全局 top-k
- 可选的分块空间输入混合

当你想要在不改变训练器接口的情况下对宽激活进行结构化分解时使用此功能。

重要的实现细节：

- `num_latents` 是跨分块的总和
- `b_dec` 作为跨分块偏置的连接视图暴露
- `set_b_dec_data()` 允许训练器从一个全局均值向量初始化分块解码器偏置
- `global_topk` 目前在分块前向中禁用 AuxK，因为那里未实现全局死掩码处理

## `sparsify/trainer.py`

`Trainer` 协调完整的 SAE 训练循环。

职责：

- 解析钩入点
- 每个钩入点和每个种子初始化 SAE
- 在 Transformer 上注册前向钩子
- 在钩子内部运行局部 SAE 损失和反向传播
- 维护死特征统计信息
- 保存检查点和可选的 W&B 指标

训练器还管理：

- 运行名称生成
- 死特征计数器
- 跨 rank 的批次指标归约
- 每个钩入点延迟创建的可选 Hadamard 旋转对象

## `sparsify/device.py`

此文件将 CUDA、NPU 和 CPU 行为抽象在一个接口后面。

值得注意的辅助函数：

- `get_device_type()`
- `get_device_string()`
- `get_dist_backend()`
- `device_autocast()`
- `create_event()` / `synchronize()`

仓库仍支持 CUDA 和 NPU 后端，但 CUDA 现在是主要文档化路径。

## `sparsify/utils.py`

训练和导出共享的实用函数。

亮点：

- 通过一次探测模型解析钩入点宽度
- 使用 `partial_forward_to_layer()` 提前停止模型前向
- 通过 `decoder_impl` 分发解码调用

`decoder_impl` 在加速器后端选择融合解码器，在 CPU 上回退到即时解码。

其他重要的辅助函数：

- `get_layer_list()` 查找模型的主 Transformer 层列表
- `get_max_layer_index()` 从钩入点名称提取最远需要的层
- `simple_parse_args_string()` 转换 CLI 数据集参数如 `name=sample-10BT`
