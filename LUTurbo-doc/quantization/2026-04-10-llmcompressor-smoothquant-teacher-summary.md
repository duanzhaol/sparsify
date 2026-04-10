# Qwen3-0.6B 离线 SmoothQuant W8A8 Teacher 实现总结

## 背景

在已经完成的 `torchao dynamic W8A8 teacher` 路线里，我们确认了两件事：

- `Qwen3-0.6B` 的量化 teacher 可以接入现有 SAE 训练框架；
- 仅把 teacher 从 `BF16` 切到 `torchao W8A8`，`FVU / exceed_alpha_0.50` 不会明显恶化，但也没有明显变好。

因此，下一步更值得尝试的是一条更“离线校准型”的 teacher 路线：

- 先对 `Qwen3-0.6B` 做一次离线 SmoothQuant；
- 导出一个可重复加载的 `W8A8 teacher checkpoint`；
- 再把它接入现有 SAE 在线训练，用它来产生活激活。

## 本次实现目标

本轮先完成第一阶段：

- 使用 `LLM Compressor` 为 `Qwen3-0.6B` 导出一个离线 `SmoothQuant W8A8 teacher`；
- 让导出过程可复用、可记录、可测试；
- 验证导出后的 checkpoint 还能正常通过 Hugging Face 重新加载，并保留原有 hookpoint 命名。

本轮暂不包含：

- SmoothQuant teacher 的正式训练结果；
- SmoothQuant teacher 与 `SAE I/O Int8 QAT` 的组合实验；
- SmoothQuant teacher 的系统性 A/B 精度矩阵。

## 新增文件

### 1. `quantization/llmcompressor_smoothquant.py`

提供离线导出所需的公共能力：

- `SmoothQuantRecipeConfig`
  - 统一描述 recipe 参数；
- `prepare_tokenized_calibration_dataset(...)`
  - 从 `datasets.save_to_disk` 产生的本地 tokenized 数据集中抽取 calibration 样本；
  - 只保留 `input_ids`；
  - 截断到指定 `max_seq_length`；
  - 返回内存中的 `Dataset`；
- `import_llmcompressor_symbols()`
  - 统一处理 `LLM Compressor` 的实际 import 路径；
- `build_smoothquant_w8a8_recipe(...)`
  - 构造 `SmoothQuant + W8A8` recipe；
- `write_smoothquant_export_manifest(...)`
  - 记录导出所用模型、数据集、样本数与 recipe 参数。

### 2. `quantization/export_llmcompressor_smoothquant_w8a8_teacher.py`

这是一个独立的导出脚本，用于：

- 加载本地 `Qwen3-0.6B` 模型；
- 加载本地 tokenized calibration dataset；
- 用 `LLM Compressor` 执行一次离线 `SmoothQuant + W8A8` 导出；
- 将量化后的 teacher checkpoint 与 manifest 写到指定目录。

当前支持的主要参数包括：

- `--model-path`
- `--dataset-path`
- `--output-dir`
- `--num-calibration-samples`
- `--max-seq-length`
- `--shuffle-seed`
- `--smoothing-strength`
- `--device-map`
- `--torch-dtype`
- `--trust-remote-code`

### 3. `tests/test_llmcompressor_smoothquant.py`

覆盖了以下关键点：

- recipe 默认参数是否符合预期；
- calibration dataset 是否会被正确截断与限量；
- 缺少 `input_ids` 时是否会报错；
- manifest 是否正确写出；
- `llmcompressor` 的关键符号是否能被实际导入；
- 导出脚本能否作为顶层脚本执行 `--help`；
- calibration 准备流程不会调用 `Dataset.map/filter`。

## 当前实现中的关键工程决策

### 1. 采用 `LLM Compressor` 而不是继续扩展 `torchao`

这条路线的重点不是“让量化模型跑起来”，而是：

- 利用离线 calibration；
- 显式做 SmoothQuant 的激活/权重平滑；
- 导出一个独立 checkpoint，后续训练与评估可以重复使用。

因此这里选择：

- `SmoothQuantModifier`
- `GPTQModifier(..., scheme=\"W8A8\", targets=\"Linear\", ignore=[\"lm_head\"])`

作为第一版的离线导出 recipe。

### 2. calibration dataset 改为“只读原数据 + 内存重建”

一开始如果直接对 `load_from_disk(...)` 得到的数据集做 `map/filter`，
`datasets` 会尝试在原始数据目录附近创建临时文件。

在当前环境里，真实 calibration 数据位于工作区外：

- `/root/fineweb-edu/sample/10BT-tokenized-qwen3-2048`

这会触发只读文件系统错误。

因此现在的实现改为：

- 从原数据集中 `shuffle + select`；
- 手动提取 `input_ids`；
- 再用 `Dataset.from_dict(...)` 重建一份内存数据集。

这样做的好处是：

- 不污染原始数据目录；
- 不依赖原目录可写；
- 更适合后续在不同机器上复用。

### 3. 脚本支持直接顶层执行

为了支持如下使用方式：

```bash
python quantization/export_llmcompressor_smoothquant_w8a8_teacher.py --help
```

脚本里加入了 repo root 的 `sys.path` bootstrap，避免顶层运行时因为
`quantization` 不是已安装包而报 `ModuleNotFoundError`。

## 当前已验证结果

### 1. 单元测试

执行：

```bash
python -m pytest tests/test_llmcompressor_smoothquant.py -q
```

结果：

- `7 passed`

### 2. 语法检查

执行：

```bash
python -m py_compile \
  quantization/llmcompressor_smoothquant.py \
  quantization/export_llmcompressor_smoothquant_w8a8_teacher.py
```

结果：

- 通过。

### 3. 真实导出 smoke

实际成功执行过如下导出命令：

```bash
python quantization/export_llmcompressor_smoothquant_w8a8_teacher.py \
  --model-path /root/models/Qwen3-0.6B \
  --dataset-path /root/fineweb-edu/sample/10BT-tokenized-qwen3-2048 \
  --output-dir checkpoints/qwen3_smoothquant_w8a8_teacher_smoke \
  --num-calibration-samples 2 \
  --max-seq-length 128 \
  --device-map auto \
  --torch-dtype auto
```

导出成功后，目录中可见：

- `config.json`
- `model.safetensors`
- `recipe.yaml`
- `smoothquant_export_manifest.json`
- tokenizer 相关文件

### 4. 导出后重新加载验证

已验证以下两种加载方式都能成功：

- `AutoModelForCausalLM.from_pretrained(...)`
- `AutoModel.from_pretrained(...)`

并且对以下 hookpoint 做了存在性检查：

- `layers.0.self_attn.q_proj`
- `layers.13.self_attn.q_proj`

结果均为存在。

这说明：

- 导出的 checkpoint 至少在 Hugging Face 加载链路上是可用的；
- 现有 SAE 训练所依赖的模块命名仍然保留；
- 可以继续把它接进 `activation_source` 做在线训练。

### 5. 训练侧最小接入

当前已经新增一个最小训练接入：

- `activation_source=smoothquant_w8a8_backbone`

它的行为是：

- 使用 `activation_backbone_path` 指向离线导出的 SmoothQuant checkpoint；
- 继续复用现有 `AutoModel.from_pretrained(...) + register_forward_hook(...)` 路径；
- 不走 `torchao` 的运行时量化加载逻辑；
- 下游 SAE 训练流程不变。

配套还补了一个 smoke 脚本：

- `scripts/full/qwen3-0.6B/product_key_expert_jumprelu/train_smoothquant_teacher_smoke_q_0_13.sh`

这意味着目前已经具备：

- 导出离线 SmoothQuant teacher；
- 在训练配置里声明该 teacher 为 activation source；
- 直接发起一个最小 smoke 训练。

## 当前结论

截至当前，可以比较稳地给出以下判断：

1. `LLM Compressor` 已经可以在本仓库中对本地 `Qwen3-0.6B` 执行离线 SmoothQuant W8A8 导出；
2. 导出后的 checkpoint 可以重新加载，并保留我们关注的 `q_proj` hookpoint；
3. `SmoothQuant teacher` 已经完成最小训练侧接入，可以作为新的 `activation_source` 使用；
4. 下一阶段的重点不再是导出本身，而是实际跑出 `BF16 / torchao / SmoothQuant` 三路 A/B。

## 建议的下一步

按优先级，后续建议顺序是：

1. 给训练侧新增 `SmoothQuant W8A8 teacher` 的 `activation_source`；
2. 先跑一个 `layers.[0-13].self_attn.q_proj` 的 smoke 训练脚本；
3. 与当前 `BF16 teacher`、`torchao W8A8 teacher` 做最小三路对照；
4. 如果结果有正收益，再决定是否把它与 `SAE I/O Int8 QAT` 组合。
