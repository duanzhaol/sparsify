# Product-Key JumpReLU LUT 导出

本文档说明如何把 `product_key_expert_jumprelu` 检查点导出成运行时可直接加载的 LUT 目录。

## 脚本位置

导出脚本位于：

- `scripts/export/export_product_key_expert_jumprelu_lut.py`

它专门面向 `product_key_expert_jumprelu` 架构，不再区分 `--qproj-best-dir` 与 `--upproj-best-dir`。你只需要提供一个检查点根目录，脚本会自动扫描并合并对应层的来源。

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

- `router_left_weight`
- `router_left_bias`
- `router_right_weight`
- `router_right_bias`
- `pair_left_index`
- `pair_right_index`
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

这样导出后的目录就是运行时完整输入，不需要再额外访问原始模型里的投影权重。

## 常用命令

导出全部层并生成 merge 视图：

```bash
python scripts/export/export_product_key_expert_jumprelu_lut.py /root/models/Qwen3-0.6B checkpoints \
  --output-dir /tmp/product_key_expert_jumprelu_lut \
  --merge-output-dir /tmp/product_key_expert_jumprelu_merge \
  --layers 0-27 \
  --operators qkv gate_up \
  --compensation-ratio 0.25 \
  --dtype float16 \
  --device cpu \
  --batch-size 512
```

只导出部分层：

```bash
python scripts/export/export_product_key_expert_jumprelu_lut.py /root/models/Qwen3-0.6B checkpoints \
  --output-dir /tmp/product_key_expert_jumprelu_lut \
  --layers 14-27 \
  --operators qkv gate_up
```

只导出注意力侧 `qkv`：

```bash
python scripts/export/export_product_key_expert_jumprelu_lut.py /root/models/Qwen3-0.6B checkpoints \
  --output-dir /tmp/product_key_expert_jumprelu_lut \
  --layers 0-27 \
  --operators qkv
```

## merge 功能

`--merge-output-dir` 会额外生成一个“合并视图”，方便检查每层到底选中了哪个 run：

- `qproj_best/`：每层选中的 `q_proj` 源目录符号链接
- `upproj_best/`：每层选中的 `up_proj` 源目录符号链接
- `merge_manifest.json`：记录每层对应的实际来源

这个目录主要用于审计和排查；真正给运行时使用的是 `--output-dir` 下的 LUT artifact。

## 适合的训练组织方式

如果你把层分成多段训练，比如 `0-13` 和 `14-27`：

- 可以直接把这些 run 放在同一个 `checkpoint_dir`
- 导出脚本会按层自动拼起来
- 只要关键配置一致，就不需要手工指定两个 best 目录

这正是脚本内置 merge 行为的主要用途。

## 注意事项

- 当前脚本只支持 `product_key_expert_jumprelu`
- 当前自动发现逻辑只覆盖 `q_proj` 和 `up_proj` 训练入口
- 若层目录配置不一致，脚本会报错并拒绝 merge
- 若目标线性层没有 bias，导出时会按 0 bias 处理
