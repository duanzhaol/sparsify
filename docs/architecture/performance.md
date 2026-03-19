# 性能说明

本文仅讨论当前实现路径中仍然有效的性能要点，不覆盖历史实验分支。

## 主要优化点

### BF16 自动混合精度

`sparsify/device.py` 中的 `device_autocast()` 会在后端支持时启用 bf16 自动混合精度。这是当前训练加速的基础能力之一。

### 融合编码器与解码器

当前编码与解码都采用自定义 autograd：

- `sparsify/fused_encoder.py`
- `sparsify/fused_decoder.py`

两者的共同策略是：

- 内存允许时优先使用 scatter + matmul
- 超出阈值时回退到更省内存的实现

### 部分前向执行

如果目标 hookpoint 不在最后一层，`Trainer` 会通过 `partial_forward_to_layer()` 提前截断前向，减少无关层开销。

### `torch.compile`

`compile_model=True` 时会在 `Trainer.fit()` 中按层编译 Transformer。

当前行为：

- 主要用于降低 CUDA 小算子启动开销
- 在非 CUDA 后端会被 `TrainConfig.__post_init__()` 自动关闭

### 分块 SAE 的收益与代价

`TiledSparseCoder` 适用于宽激活的结构化拆分，但并非零成本优化。

常见收益：

- 单个 tile 上的编码/解码规模更小
- 可以更细粒度地控制激活结构

常见代价：

- 额外的分块与拼接开销
- latent 统计与调参更复杂
- `global_topk` 模式可能引入较大的块对角解码矩阵

### Hadamard 预处理

Hadamard 旋转可以改善激活分布，但会增加 hook 内计算量。它更接近“精度与速度的可调开关”，不是无成本加速项。

## CUDA 与 NPU 的定位

### CUDA

CUDA 是当前主线环境。默认建议：

- 优先在 CUDA 上运行主实验
- `compile_model` 仅在 CUDA 开启
- 性能分析与调优优先围绕 CUDA 路径展开

### NPU

NPU 兼容能力仍保留在代码中（`device.py` 和融合解码路径），但不再是文档主线。历史 NPU 分析材料已迁至 `docs/archive/ascend/`。

## 为什么要收敛到这套说明

旧文档包含大量历史分支与阶段性性能分析结论，容易干扰当前调优判断。本文只保留当前代码主线仍然生效的优化点，便于直接落地。
