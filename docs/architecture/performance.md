# 性能说明

本页总结 Sparsify 当前与性能相关的部分。它有意反映当前代码路径，而非历史实验。

## 主要性能杠杆

### BF16 自动类型转换

`sparsify/device.py` 中的 `device_autocast()` 在后端支持时使用 bf16 自动类型转换包装关键前向路径。这是现代 CUDA 和 NPU 加速器上的默认快速路径。

### 融合编码器和解码器

编码器和解码器都使用自定义自动微分路径：

- `sparsify/fused_encoder.py`
- `sparsify/fused_decoder.py`

两种实现都优先使用 scatter-plus-matmul，当密集中间结果仍在内存阈值内时；否则回退到更节省内存的逻辑。

### 部分 Transformer 前向

当所有选定的钩入点位于最终模型层之前时，`Trainer` 可以通过调用 `sparsify/utils.py` 中的 `partial_forward_to_layer()` 提前停止 Transformer 前向。

这避免了为不贡献训练激活值的后续层付出代价。

### `torch.compile`

`compile_model=True` 在 `Trainer.fit()` 中单独编译 Transformer 层。

当前状态：

- 旨在减少 CUDA 的内核启动开销
- 由 `TrainConfig.__post_init__()` 在非 CUDA 后端上自动禁用

### 分块 SAE 权衡

`TiledSparseCoder` 引入宽激活的结构化分解。

潜在收益：

- 更小的每分块编码器和解码器操作
- 对激活结构更多控制

成本：

- 额外的分块开销
- 更复杂的潜在变量统计
- 全局 top-k 模式可能构建大型块对角解码器

### Hadamard 旋转

当前主线支持 Hadamard 预处理，可以帮助处理激活结构和异常值行为。它在钩子路径中增加了额外工作，因此应被视为精度/性能权衡，而非免费优化。

## CUDA 与 NPU 定位

### CUDA

CUDA 是当前开发的主要运行时路径。

推荐默认值：

- 首先在 CUDA 上开始
- 仅在 CUDA 上启用 `compile_model`
- 将 CUDA 上的性能调试视为主线工作流

### NPU

NPU 兼容性仍通过 `sparsify/device.py` 和融合解码路径保留在代码库中，但它不再是主要文档目标。

历史 NPU 性能分析材料现存放于 `docs/archive/ascend/` 下。

## 有意移出的内容

旧文档强调了许多历史分支和性能分析快照。这些已被归档，以便本页保持聚焦于活跃实现接口。
