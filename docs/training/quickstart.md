# 训练快速开始

> 说明：本文按 Phase 2 训练框架的使用方式编写，示例展示标准 TopK SAE 与架构变体共享的运行入口。

本文档介绍 Sparsify 在 NVIDIA/CUDA 平台训练 SAE 的主线流程，目标是先跑通训练，再完成阈值统计与 LUT 导出。

## 1. 安装

从仓库根目录：

```bash
pip install -e .[dev]
```

# 2. 运行最小训练示例

CLI 入口是 `python -m sparsify`（实现在 `sparsify/__main__.py` 中）。

示例：训练 Qwen3 的部分注意力投影输入。

```bash
python -m sparsify Qwen/Qwen3-0.6B HuggingFaceFW/fineweb \
  --data_args "name=sample-10BT" \
  --text_column text \
  --hookpoints "layers.[7,14].self_attn.o_proj" \
  --batch_size 1 \
  --grad_acc_steps 8 \
  --ctx_len 2048 \
  --optimizer signum \
  --sae.expansion_factor 8 \
  --sae.architecture topk \
  --sae.k 128 \
  --save_dir checkpoints \
  --run_name qwen3-oproj-demo
```

说明：

- `hookpoints` 支持 glob 模式和范围写法，由 `sparsify/checkpoint.py` 中的 `expand_range_pattern()` 处理。
- 当前实现使用模块输入作为 SAE 训练激活值。
- 如果数据集尚未分词，CLI 会在运行中调用分词逻辑。
- 示例中的 `o_proj` 名称用于匹配模块，实际训练的是该模块的输入。
- 若要切换架构变体，只需替换 `--sae.architecture` 及对应子参数。

例如，训练一个 Gated SAE：

```bash
python -m sparsify Qwen/Qwen3-0.6B HuggingFaceFW/fineweb \
  --data_args "name=sample-10BT" \
  --text_column text \
  --hookpoints "layers.[7,14].self_attn.o_proj" \
  --batch_size 1 \
  --grad_acc_steps 8 \
  --ctx_len 2048 \
  --optimizer signum \
  --sae.architecture gated \
  --sae.expansion_factor 8 \
  --sae.k 32 \
  --save_dir checkpoints \
  --run_name qwen3-oproj-gated
```

## 3. 预期输出

一次运行会在 `checkpoints/` 下生成一个检查点目录，如：

```text
checkpoints/<run_name>_dp1_bs1_ga8_ef8_k128_<timestamp>/
```

典型内容包括：

- `config.json`：序列化的训练配置
- `manifest.json`：运行身份证，记录架构、模型、数据集和 git 信息
- `metrics.jsonl`：逐步训练指标
- `summary.json`：训练结束摘要
- `state.pt`：训练器状态，如 `global_step` 和 `total_tokens`
- `optimizer_0.pt`：优化器状态
- `<hookpoint>/cfg.json`：单个 hookpoint 对应的 SAE 配置
- `<hookpoint>/sae.safetensors`：SAE 权重

如果启用了 `save_best=True`，训练器还会生成 `best/` 目录，保存当前更优的 SAE 快照。

## 4. 恢复训练或微调

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

`resume` 用于接着同一次训练继续跑（包括优化器状态和 token 计数）；`finetune` 用于加载旧权重后开启一轮新训练。

如果要训练残差 SAE（二级 SAE），可额外提供：

```bash
python -m sparsify Qwen/Qwen3-0.6B HuggingFaceFW/fineweb \
  --hookpoints "layers.[7,14].self_attn.o_proj" \
  --sae.architecture topk \
  --sae.k 32 \
  --residual_from checkpoints/level1_run
```

这里要求 `residual_from` 下的 hookpoint 命名与当前训练目标一致。

## 5. 生成肘部阈值

在完成某个投影类型的 SAE 检查点后，建议对相同 hookpoint 组运行激活分布统计。

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

该脚本会：

- 加载模型进行推理
- 在目标模块输入上注册 hook
- 收集激活样本直至令牌预算耗尽
- 计算 Kneedle 风格的肘部值
- 写入包含 `elbow_p` 与 `elbow_value` 的 JSON 文件

需要可视化曲线时，加上 `--plot_dir <dir>`。

## 6. 导出 SAE 检查点到 LUT 产物

训练和阈值生成后，运行导出器。

示例：

```bash
python convert_sae_to_lut.py Qwen/Qwen3-0.6B checkpoints \
  --output_dir lut_output \
  --proj_types oproj \
  --layers 7,14 \
  --threshold_dir thresholds
```

通常需要满足：

- 检查点目录中有导出脚本可识别的投影类型命名
- 你选择的投影类型与训练时的 hookpoint 对得上
- 阈值文件放在 `--threshold_dir` 指定的目录里

## 7. 完整最小路径

对单个投影类型，最短流程是：

1. 使用 `python -m sparsify` 训练 SAE 检查点
2. 使用 `compute_elbow_thresholds.py` 计算肘部统计信息
3. 使用 `convert_sae_to_lut.py` 导出 LUT 产物

这就是从模型激活到 LUTurbo 可用结果的最短闭环。

## 8. 常见后续动作

- 使用 `compute_elbow_thresholds.py` 生成肘部阈值
- 使用 `convert_sae_to_lut.py` 转换选定的 SAE 检查点
- 用脚本扫描 `metrics.jsonl` / `summary.json` 做多 run 横向比较
- 在 `docs/architecture/training-pipeline.md` 中查看训练流程

如果你不确定接下来阅读什么：

- 使用 `docs/training/config-reference.md` 了解参数详情
- 使用 `docs/training/qwen3-guide.md` 获取 Qwen3 特定建议
- 使用 `docs/export/sae-to-lut.md` 了解导出脚本行为和注意事项

## 9. 平台说明

- CUDA 是推荐的默认选项。
- `compile_model` 目前仅支持 CUDA；`TrainConfig.__post_init__` 在非 CUDA 后端上禁用它。
- NPU 支持仍通过 `sparsify/device.py` 保留，但文档不再将其作为默认流程。
