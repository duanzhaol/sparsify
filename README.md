# Sparsify

Sparsify 是 LUTurbo 的稀疏自编码器（SAE）训练与导出模块，负责在 Transformer 模块输入上训练 SAE、生成阈值统计，并导出面向 LUT 的产物供下游 LUTurbo 推理使用。

项目定位：

- NVIDIA/CUDA 作为主运行平台
- Ascend/NPU 仅保留为兼容性路径和历史参考
- 本仓库相比旧的 `sparsify-ascend` 分支有意精简范围

## 本仓库的功能

- 通过前向 hook 捕获 Transformer 激活值并训练 SAE
- 为选定的 hookpoint 和层保存 SAE 检查点
- 计算 LUTurbo 补偿逻辑使用的肘部阈值统计信息
- 将训练好的 SAE 检查点导出为 LUT 友好的产物

## 本仓库不包含

- 不包含完整的 LUTurbo 推理运行时
- 不定位为包含大量历史实验分支的通用可解释性工具库
- 不再以 Ascend 作为文档主平台

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

- 更完整的端到端说明请参阅：

- `docs/training/quickstart.md`
- `docs/export/sae-to-lut.md`

## 端到端流程

典型流程如下：

1. 使用 `python -m sparsify` 训练 SAE 检查点
2. 使用 `compute_elbow_thresholds.py` 计算肘部阈值
3. 使用 `convert_sae_to_lut.py` 导出 LUT 产物

这是从 Transformer 激活值到 LUTurbo 可用产物的最短路径。

## 主要代码主线

- `sparsify/__main__.py`：CLI 入口、模型加载、数据集加载
- `sparsify/config.py`：当前生效的配置定义
- `sparsify/trainer.py`：hook 驱动的 SAE 训练流程
- `sparsify/sparse_coder.py`：标准 SAE 实现
- `sparsify/tiled_sparse_coder.py`：分块 SAE 变体
- `compute_elbow_thresholds.py`：阈值统计信息生成
- `convert_sae_to_lut.py`：LUT 导出流程

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

## 当前训练主线

当前训练主线包括：

- 通过前向 hook 在模块输入上训练 SAE
- TopK 稀疏激活
- 使用 FVU 评估局部重建质量
- 可选的 AuxK 死特征恢复
- 可选的分块 SAE 和 Hadamard 预处理
- SignSGD 配合 `ScheduleFreeWrapperReference`

当文档与代码不一致时，以代码为准。

## 加载已保存 SAE

程序化加载可通过 `SparseCoder.load_from_disk()` / `load_many()` / `load_from_hub()` 完成。

```python
from sparsify import Sae

sae = Sae.load_from_hub("EleutherAI/sae-llama-3-8b-32x", hookpoint="layers.10")
```

从磁盘加载也适用于训练生成的检查点目录：

```python
from sparsify import Sae

sae = Sae.load_from_disk("checkpoints/your_run/layers.7.self_attn.o_proj")
```

检查点布局的详细说明见 `docs/training/config-reference.md`。

## 检查点与导出

本仓库会产出以下内容：

- `checkpoints/` 下的 SAE 检查点
- 可选的 `best/` 检查点快照
- `compute_elbow_thresholds.py` 生成的阈值 JSON 文件
- `convert_sae_to_lut.py` 生成的 LUT 导出结果

导出脚本依赖相对稳定的命名和目录约定。建议让运行名称与投影类型（projection type）保持一致，例如 `qproj`、`oproj`、`upproj`。

## 开发

- 运行 `pip install -e .[dev]`
- 使用 `python -m sparsify --help` 获取当前 CLI 参数
- 文档以 `docs/` 下的当前主线为准，归档内容仅作历史参考
