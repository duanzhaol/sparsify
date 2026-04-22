# Expert JumpReLU LUT 导出

本文档说明如何把 `expert_jumprelu` 检查点导出成运行时可直接加载的 LUT 目录。

## 脚本位置

导出脚本位于：

- `scripts/export/export_expert_jumprelu_lut.py`

它专门面向 `expert_jumprelu` 架构，使用单路 router 的专用导出格式。你只需要提供一个检查点根目录，脚本会自动扫描并合并对应层的来源。

## 输入约定

脚本接受两个位置参数：

1. `model_path`：基础模型路径或 Hugging Face 模型名
2. `checkpoint_dir`：包含 `qproj` / `upproj` 训练 run 的根目录

脚本会递归扫描 `checkpoint_dir`，寻找：

- `layers.*.self_attn.q_proj`
- `layers.*.mlp.up_proj`

每个候选层目录都需要包含：

- `cfg.json`
- `sae.safetensors`

如果同一层在多个 run 中都存在，脚本会优先选择运行名里时间戳更新的那个来源；同时会检查待导出层之间的关键配置是否一致，避免把不兼容的 checkpoint 混在一起。

## 输出格式

输出目录是一个“自包含 artifact”：

- `metadata.json`
- `layers.<layer_idx>.self_attn.qkv_proj.lut.safetensors`
- `layers.<layer_idx>.mlp.gate_up_proj.lut.safetensors`

其中：

- `metadata.json` 记录全局信息，包括架构、运行时目标、默认补偿模式、默认 ratio、模型元数据，以及每层导出的文件和维度信息
- 每个 `.lut.safetensors` 都是单层单算子的完整运行时包，运行时不需要再访问原始 SAE 权重

## 单文件内部张量

每个 `.lut.safetensors` 同时包含三类数据：

### Stage A：编码与路由

- `router_weight`
- `router_bias`
- `expert_encoder_weight`
- `expert_encoder_bias`
- `expert_threshold`
- `decoder_weight`
- `decoder_bias`

### Stage B：查表数据

- `precomputed_products`
- `bias_product`

### Stage C：补偿数据

- `compensation_weight_t`

## 常用命令

导出全部层并生成 merge 视图：

```bash
python scripts/export/export_expert_jumprelu_lut.py /root/models/Qwen3-4B checkpoints/qwen3-4B \
  --output-dir /tmp/expert_jumprelu_lut \
  --merge-output-dir /tmp/expert_jumprelu_merge \
  --layers 0-35 \
  --operators qkv gate_up \
  --compensation-ratio 0.25 \
  --dtype float16 \
  --device cpu \
  --batch-size 512
```

只导出部分层：

```bash
python scripts/export/export_expert_jumprelu_lut.py /root/models/Qwen3-4B checkpoints/qwen3-4B \
  --output-dir /tmp/expert_jumprelu_lut \
  --layers 14-27 \
  --operators qkv gate_up \
  --compensation-ratio 0.25
```

只导出注意力侧 `qkv`：

```bash
python scripts/export/export_expert_jumprelu_lut.py /root/models/Qwen3-4B checkpoints/qwen3-4B \
  --output-dir /tmp/expert_jumprelu_lut \
  --layers 0-35 \
  --operators qkv \
  --compensation-ratio 0.25
```

## 注意事项

- 当前脚本只支持 `expert_jumprelu`
- 当前自动发现逻辑只覆盖 `q_proj` 和 `up_proj` 训练入口
- 若层目录配置不一致，脚本会报错并拒绝 merge
- 若目标线性层没有 bias，导出时会按 0 bias 处理
