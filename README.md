# Sparsify

Sparsify 是 LUTurbo 的稀疏自编码器（SAE）训练与导出层。本仓库专注于在 Transformer 模块输入上训练稀疏自编码器、生成阈值统计信息，并导出面向 LUT 的制品，供下游 LUTurbo 推理流水线使用。

当前项目定位：

- NVIDIA/CUDA 是主要运行时路径
- Ascend/NPU 保留作为兼容性路径和历史参考
- 本仓库相比旧的 `sparsify-ascend` 分支有意精简了范围

## 本仓库的功能

- 通过钩子捕获 Transformer 激活值，训练稀疏自编码器
- 为选定的钩入点和层保存 SAE 检查点
- 计算 LUTurbo 补偿逻辑使用的肘部阈值统计信息
- 将训练好的 SAE 检查点导出为 LUT 友好的制品

## 本仓库不做什么

- 它不是完整的 LUTurbo 推理运行时
- 它不是包含大量遗留实验分支的通用可解释性工具包
- 它不再将 Ascend 作为主要目标平台进行文档化

## 安装

```bash
pip install -e .[dev]
```

从 PyPI 安装包仍然可用：

```bash
pip install eai-sparsify
```

## 快速开始

Qwen3 上的最小示例：

```bash
python -m sparsify Qwen/Qwen3-0.6B HuggingFaceFW/fineweb \
  --data_args "name=sample-10BT" \
  --text_column text \
  --hookpoints "layers.[7,14].self_attn.o_proj" \
  --batch_size 1 \
  --grad_acc_steps 8 \
  --ctx_len 2048 \
  --sae.expansion_factor 8 \
  --sae.k 128 \
  --save_dir checkpoints \
  --run_name qwen3-oproj-demo
```

当前 CLI 入口点实现在 `sparsify/__main__.py` 中。

如需更完整的端到端路径，请参阅：

- `docs/training/quickstart.md`
- `docs/export/sae-to-lut.md`

## 端到端工作流

实际上，本仓库围绕以下循环设计：

1. 使用 `python -m sparsify` 训练 SAE 检查点
2. 使用 `compute_elbow_thresholds.py` 计算肘部阈值
3. 使用 `convert_sae_to_lut.py` 导出 LUT 导向的制品

这是从 Transformer 激活值到 LUTurbo 可用资产的最短路径。

## 主要代码路径

- `sparsify/__main__.py`：CLI 入口、模型加载、数据集加载
- `sparsify/config.py`：活跃配置接口
- `sparsify/trainer.py`：钩子驱动的 SAE 训练循环
- `sparsify/sparse_coder.py`：标准 SAE 实现
- `sparsify/tiled_sparse_coder.py`：分块 SAE 变体
- `compute_elbow_thresholds.py`：阈值统计信息生成
- `convert_sae_to_lut.py`：LUT 导出流水线

## 文档

从这里开始：

- `docs/README.md`
- `docs/overview.md`
- `docs/training/quickstart.md`
- `docs/training/config-reference.md`
- `docs/training/qwen3-guide.md`
- `docs/architecture/training-pipeline.md`
- `docs/export/sae-to-lut.md`

历史材料（包括旧的代码走查和 Ascend 性能分析报告）现存放于 `docs/archive/` 下。

推荐阅读顺序：

1. `docs/overview.md`
2. `docs/training/quickstart.md`
3. `docs/training/config-reference.md`
4. `docs/training/qwen3-guide.md`
5. `docs/architecture/training-pipeline.md`
6. `docs/export/sae-to-lut.md`

## 训练模型

当前主线训练器围绕以下设计：

- 通过前向钩子进行模块输入 SAE 训练
- TopK 稀疏激活
- 通过 FVU 评估局部重建质量
- 可选的 AuxK 死特征恢复
- 可选的分块 SAE 和 Hadamard 预处理
- SignSGD 配合 `ScheduleFreeWrapperReference`

当文档与代码不一致时，以代码为准。

## 加载已保存的 SAE

程序化加载仍可通过 `SparseCoder.load_from_disk()` / `load_many()` / `load_from_hub()` 完成。

```python
from sparsify import Sae

sae = Sae.load_from_hub("EleutherAI/sae-llama-3-8b-32x", hookpoint="layers.10")
```

从磁盘加载也适用于训练生成的检查点目录：

```python
from sparsify import Sae

sae = Sae.load_from_disk("checkpoints/your_run/layers.7.self_attn.o_proj")
```

如需了解检查点布局的完整训练端说明，请参阅 `docs/training/config-reference.md`。

## 检查点与导出

本仓库生成的典型制品包括：

- `checkpoints/` 下的 SAE 检查点
- 可选的 `best/` 检查点快照
- `compute_elbow_thresholds.py` 生成的阈值 JSON 文件
- `convert_sae_to_lut.py` 生成的 LUT 导出输出

导出端期望检查点命名和投影族布局保持合理一致，因此建议将运行名称与投影族（如 `qproj`、`oproj` 和 `upproj`）对齐。

## 开发

- 运行 `pip install -e .[dev]`
- 使用 `python -m sparsify --help` 查看活跃的 CLI 接口
- 使用 `docs/` 下的文档获取当前指导，而非归档的遗留笔记
- 将 `docs/archive/` 视为历史背景，而非活跃使用指导
