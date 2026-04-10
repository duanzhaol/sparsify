# SAE I/O Int8 QAT Phase 1 总结

## 目的

本文档总结本轮 `SAE I/O Int8 QAT Phase 1` 的设计落地情况。

这一阶段的目标不是直接把整个 SAE 训练改成“纯 int8 训练”，而是先做一个最小、可控、可解释的版本：

- 输入显式感知 `int8`
- 输出显式感知 `int8`
- SAE 参数、优化器状态、梯度主路径仍然保持浮点

这样做的核心目的是先回答一个问题：

**如果训练时显式模拟 `int8 input + int8 output`，SAE 是否会更适应最终部署形态。**

## 背景

在这一步之前，我们已经完成了多轮 SAE 后训练量化（PTQ）仿真评估，结论比较稳定：

- `expert-only W8A8` 误差几乎可以忽略；
- `full-encoder W8A8` 误差仍然很小；
- `full-encoder + decoder W8 / W8A8` 的额外误差也比较温和。

这说明一个重要事实：

**8bit 本身已经不是当前 SAE 主干的主要精度风险点。**

因此，下一步更值得研究的问题，不再只是“训完以后能不能量化”，而是：

- 训练目标是否应该更接近真实部署目标；
- 输入输出是否应该在训练时就显式感知 `int8`；
- 后续是否值得进一步推进到更深入的 Partial QAT。

## 本阶段的总体思路

Phase 1 采用的是 **I/O 量化感知训练**，而不是全链路量化训练。

训练路径改成：

```text
acts_fp -> fake_quant_in -> SAE -> recon_fp -> fake_quant_out -> loss
```

其中：

- `acts_fp` 是原始浮点激活；
- `fake_quant_in` 用来模拟部署时的 `int8` 输入；
- `recon_fp` 是 SAE 浮点输出；
- `fake_quant_out` 用来模拟部署时最终输出也会落到 `int8`；
- loss 同时考虑“对原始浮点 teacher 的贴合”与“对部署态目标的贴合”。

这一步仍然保持：

- SAE 权重浮点训练；
- 优化器状态浮点；
- 不引入真实 int8 kernel；
- 不改 SAE 架构内部实现。

## 本次实现范围

本轮落地主要覆盖四部分。

### 1. 量化辅助模块

新增文件：

- `sparsify/train_quantization.py`

该模块目前提供：

- `IOQuantMetrics`
- `fake_quantize_activation_per_token`
- `compute_fvu_scalar`
- `compute_exceed_ratio`
- `summarize_io_quant_batch`
- `select_main_loss`

其中：

- fake quant 采用对称量化；
- 激活量化粒度为 `per_token`；
- 当前 Phase 1 固定为 `8bit`；
- helper 会返回部署侧重构、量化 floor、clip rate、scale mean 等指标。

### 2. 配置层

修改文件：

- `sparsify/config.py`

新增配置项：

- `io_quant_mode`
- `io_quant_bits`
- `io_quant_granularity`
- `io_quant_clip_mode`
- `io_loss_mode`
- `io_loss_deploy_weight`

当前 Phase 1 的校验规则是：

- `io_quant_mode` 仅支持 `off` / `qat_io_int8`
- `io_quant_bits` 固定为 `8`
- `io_quant_granularity` 固定为 `per_token`
- `io_quant_clip_mode` 固定为 `absmax`
- `io_loss_mode` 支持 `fp_teacher` / `dual_target` / `deploy_target`
- `io_loss_deploy_weight` 必须非负

### 3. Trainer 接入

修改文件：

- `sparsify/trainer.py`

Trainer 现在已经接入了 Phase 1 的主流程。

具体逻辑如下：

1. hook 到激活后先保留 `acts_fp`
2. 如果开启 `qat_io_int8`，构造 `acts_in = fake_quant(acts_fp)`；否则 `acts_in = acts_fp`
3. SAE 前向使用 `acts_in`
4. SAE 输出 `out.sae_out` 后，构造部署侧指标：
   - `recon_deploy = fake_quant(recon_fp)`
   - `target_deploy = fake_quant(acts_fp)`
   - `fvu_fp_teacher`
   - `fvu_deploy`
   - `quant_floor`
5. 根据 `io_loss_mode` 选择主损失：
   - `fp_teacher`
   - `deploy_target`
   - `dual_target`
6. 保留原有：
   - `auxk_loss`
   - router regularization
   - dead-feature 统计
   - matryoshka / ortho 等附加项

也就是说，这一步没有去改 SAE 内部架构，而是把 I/O 量化感知逻辑放在 `Trainer` 外层统一处理。

### 4. 启动脚本

修改/新增文件：

- `scripts/autoresearch_test.sh`
- `scripts/full/qwen3-0.6B/product_key_expert_jumprelu/train_io_int8.sh`

现在 `scripts/autoresearch_test.sh` 已经能够透传：

- `IO_QUANT_MODE`
- `IO_QUANT_BITS`
- `IO_QUANT_GRANULARITY`
- `IO_QUANT_CLIP_MODE`
- `IO_LOSS_MODE`
- `IO_LOSS_DEPLOY_WEIGHT`

因此已经具备从训练脚本直接发起 `Phase 1 I/O QAT` 的能力。

## 当前损失与监控口径

为了避免后续结果不好解释，这一阶段在训练侧显式区分了几类指标。

### 1. 主损失相关

- `fvu_fp_teacher`
  - 部署态重构对原始浮点目标的误差
- `fvu_deploy`
  - 部署态重构对部署态量化目标的误差
- `quant_floor`
  - 单纯输入输出量化本身带来的误差下界

### 2. exceed 指标

在开启 I/O QAT 时，exceed 指标拆成两套：

- `exceed_alpha_xx_fp_teacher`
- `exceed_alpha_xx_deploy`

这样可以区分：

- 模型是在更贴近原始 teacher；
- 还是更贴近部署世界里的量化目标。

### 3. 量化观测指标

还会额外记录：

- `input_clip_rate`
- `output_clip_rate`
- `input_scale_mean`
- `output_scale_mean`

这些指标主要用于观察量化动态是否稳定。

## 本轮实现中的一个关键修正

在 helper 模块落地之后，我们又修正了一次 `compute_fvu_scalar()` 的边界行为。

原先版本会在以下情况下直接报错：

- zero-variance target
- singleton-token batch

后来我们把它改成了更稳健的实现：

- 对分母使用 `clamp_min(1e-12)`；
- 让这些边界 batch 下的 FVU 仍然保持数值可定义，而不是 hard fail。

这样做的原因是，`summarize_io_quant_batch()` 更适合作为训练/评估 helper，而不是一个对边界输入非常脆弱的函数。

## 已完成的代码提交

截至当前，这个 Phase 1 相关的主要提交包括：

- `5a7516f` `Stabilize SAE I/O FVU edge cases`
- `23812c0` `Add SAE I/O quantization config flags`
- `f114d22` `test: cover io quant config validation`
- `c07e40b` `Wire SAE trainer to int8 I/O fake quantization`
- `350886c` `Add Phase 1 SAE int8 I/O training launch script`

这些提交已经把 Phase 1 的最小闭环搭起来了。

## 已完成的本地验证

在进入 smoke 训练之前，本地已经完成以下验证：

### 1. 单元测试

```bash
pytest tests/test_train_quantization.py -q
```

结果：

- `25 passed`

### 2. 语法与导入检查

```bash
python -m py_compile sparsify/train_quantization.py sparsify/config.py sparsify/trainer.py
```

结果：

- 通过

### 3. Shell 脚本语法检查

```bash
bash -n scripts/autoresearch_test.sh
bash -n scripts/full/qwen3-0.6B/product_key_expert_jumprelu/train_io_int8.sh
```

结果：

- 通过

### 4. CLI 参数检查

```bash
python -m sparsify --help | rg "io_quant|io_loss"
```

结果：

- 新增参数已经出现在 CLI 帮助中

## 当前状态

截至本文档撰写时：

- `Phase 1 I/O QAT` 的代码链路已经接通；
- 配置、trainer、helper、脚本和测试都已到位；
- 首次 smoke 训练已经开始；
- 真实训练指标结果还需要等本轮 smoke 输出后再补充分析。

因此，**当前这份文档主要总结的是“设计与实现完成情况”，而不是最终实验结论。**

## 一个推荐的 smoke 命令

当前推荐的 smoke 命令如下：

```bash
NPROC_PER_NODE=1 \
MASTER_PORT=29531 \
WANDB_PROJECT=qwen3-0.6B-product_key_expert_jumprelu-qproj-io-int8-smoke \
SAVE_DIR=checkpoints/product_key_expert_jumprelu_qproj_io_int8_smoke \
RUN_NAME=smoke_io_int8_qproj \
MAX_TOKENS=200000 \
SAVE_EVERY=10 \
WANDB_LOG_FREQUENCY=2 \
PRINT_COST_BREAKDOWN=0 \
COMPILE_MODEL=0 \
ARCHITECTURE=product_key_expert_jumprelu \
K=32 \
EXPANSION_FACTOR=1 \
NUM_EXPERTS=512 \
ACTIVE_EXPERTS=2 \
LATENTS_PER_EXPERT=56 \
OPTIMIZER=adam \
LR=8e-4 \
HOOKPOINTS='layers.[0-13].self_attn.q_proj' \
BATCH_SIZE=1 \
GRAD_ACC_STEPS=8 \
MICRO_ACC_STEPS=1 \
AUXK_ALPHA=0.03125 \
DEAD_FEATURE_THRESHOLD=10000000 \
USE_HADAMARD=0 \
IO_QUANT_MODE=qat_io_int8 \
IO_QUANT_BITS=8 \
IO_QUANT_GRANULARITY=per_token \
IO_QUANT_CLIP_MODE=absmax \
IO_LOSS_MODE=dual_target \
IO_LOSS_DEPLOY_WEIGHT=0.25 \
bash scripts/autoresearch_test.sh
```

这条命令的主要目的不是追求最终精度，而是先确认：

- 训练能够正常启动；
- 日志里能出现新的量化指标；
- `best_loss` / checkpoint 逻辑仍然正常工作。

## 这一步完成后意味着什么

本轮实现完成后，我们已经正式从“只做 PTQ 仿真评估”推进到了“训练时显式感知 int8 I/O”。

这意味着后续可以围绕以下问题继续做实验：

1. `I/O QAT` 是否真的能改善 `FVU_deploy`
2. `FVU_fp_teacher` 是否会被明显牺牲
3. `dual_target` / `fp_teacher` / `deploy_target` 三种 loss mode 哪种更合适
4. 如果 I/O QAT 收益明确，是否值得继续进入 `Partial QAT`

## 下一步建议

基于当前实现状态，建议下一步按下面顺序推进：

1. 先看 smoke 训练日志是否正常，确认指标字段完整。
2. 再做一个小规模对照实验：
   - `io_quant_mode=off`
   - `io_quant_mode=qat_io_int8`
3. 优先比较：
   - `fvu_fp_teacher`
   - `fvu_deploy`
   - `quant_floor`
   - `exceed_alpha_0.50_fp_teacher`
   - `exceed_alpha_0.50_deploy`
4. 如果结果正面，再继续考虑：
   - 更系统的 I/O QAT 实验
   - SAE 主干 Partial QAT
   - 在线补偿阶段是否也要进入 8bit

当前最重要的问题已经从：

**“训完以后能不能量化”**

转向了：

**“训练时显式模拟 int8 I/O，是否真的能让 SAE 更适应我们最终要部署的世界。”**
