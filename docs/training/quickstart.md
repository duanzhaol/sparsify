# 训练快速开始

本快速开始展示在 NVIDIA/CUDA 上使用 Sparsify 训练 SAE 的当前主线工作流。

这里的重点是实用性：运行一个训练任务，检查检查点输出，然后继续生成阈值和导出 LUT。

## 1. 安装

从仓库根目录：

```bash
pip install -e .[dev]
```

## 2. 选择最小训练运行

CLI 入口点是 `python -m sparsify`，实现在 `sparsify/__main__.py` 中。

示例：在选定的 Qwen3 注意力输出投影上训练。

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

注意：

- 钩入点可以使用 glob 模式或范围扩展语法，由 `sparsify/checkpoint.py` 中的 `expand_range_pattern()` 处理。
- 当前训练器使用模块输入作为 SAE 训练激活值。
- 如果数据集尚未分词，CLI 会即时进行分词。
- 上面的示例使用 `o_proj` 模块名，但 SAE 是在这些模块的输入上训练的。

## 3. 预期输出

一次运行会生成一个检查点目录，如：

```text
checkpoints/<run_name>_dp1_bs1_ga8_ef8_k128_<timestamp>/
```

典型内容：

- `config.json`：序列化的训练配置
- `state.pt`：训练器状态，如 `global_step` 和 `total_tokens`
- `optimizer_0.pt`：优化器状态
- `<hookpoint>/cfg.json`：单个钩入点的 SAE 配置
- `<hookpoint>/sae.safetensors`：SAE 权重

如果启用了 `save_best=True`，训练器还会创建一个 `best/` 子树，包含改进的每个钩入点快照。

## 4. 恢复或微调

恢复最新匹配的运行名称：

```bash
python -m sparsify Qwen/Qwen3-0.6B HuggingFaceFW/fineweb \
  --run_name qwen3-oproj-demo \
  --resume
```

从之前的检查点树进行微调：

```bash
python -m sparsify Qwen/Qwen3-0.6B HuggingFaceFW/fineweb \
  --hookpoints "layers.[7,14].self_attn.o_proj" \
  --finetune checkpoints/your_previous_run/best
```

当你想继续相同的运行状态（包括优化器和令牌计数器）时使用 `resume`。当你想从旧的 SAE 权重初始化但开始新的训练运行时使用 `finetune`。

## 5. 生成肘部阈值

一旦你有了投影族的 SAE 检查点，为相同的钩入点生成激活分布统计信息。

示例：

```bash
python compute_elbow_thresholds.py Qwen/Qwen3-0.6B \
  --dataset togethercomputer/RedPajama-Data-1T-Sample \
  --hookpoints "layers.[7,14].self_attn.o_proj" \
  --num_tokens 1000000 \
  --batch_size 8 \
  --ctx_len 2048 \
  --output thresholds_o.json
```

这做了什么：

- 加载模型进行推理
- 钩入模块输入
- 收集激活样本直到令牌预算上限
- 计算 Kneedle 风格的肘部值
- 写入包含 `elbow_p` 和 `elbow_value` 的 JSON 文件

如果你想要可视化检查，同时传递 `--plot_dir <dir>`。

## 6. 导出 SAE 检查点到 LUT 制品

训练和阈值生成后，运行导出器。

示例：

```bash
python convert_sae_to_lut.py Qwen/Qwen3-0.6B checkpoints \
  --output_dir lut_output \
  --proj_types oproj \
  --layers 7,14 \
  --threshold_dir thresholds
```

典型期望：

- 检查点基础目录包含导出器可以发现的投影族运行，其名称可被识别
- 选择的投影族与训练中使用的钩入点匹配
- 阈值文件通过 `--threshold_dir` 传递的目录中组织

## 7. 完整最小路径

对于单个投影族，实际流程是：

1. 使用 `python -m sparsify` 训练 SAE 检查点
2. 使用 `compute_elbow_thresholds.py` 计算肘部统计信息
3. 使用 `convert_sae_to_lut.py` 导出 LUT 制品

这是从原始模型激活值到 LUTurbo 可用资产的最短有效循环。

## 8. 常见后续步骤

- 使用 `compute_elbow_thresholds.py` 生成肘部阈值
- 使用 `convert_sae_to_lut.py` 转换选定的 SAE 检查点
- 在 `docs/architecture/training-pipeline.md` 中检查训练流水线

如果你不确定接下来阅读什么：

- 使用 `docs/training/config-reference.md` 了解参数详情
- 使用 `docs/training/qwen3-guide.md` 获取 Qwen3 特定建议
- 使用 `docs/export/sae-to-lut.md` 了解导出器行为和注意事项

## 9. 平台说明

- CUDA 是推荐的默认选项。
- `compile_model` 目前仅支持 CUDA；`TrainConfig.__post_init__` 在非 CUDA 后端上禁用它。
- NPU 支持仍通过 `sparsify/device.py` 存在，但文档不再将其作为默认工作流。
