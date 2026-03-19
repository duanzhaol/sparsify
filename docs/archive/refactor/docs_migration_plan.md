> Archived document: this file is kept for historical reference and may not match the current codebase.
> For current guidance, start from `docs/README.md` and the active docs under `docs/`.

# Sparsify Docs 迁移方案

## 1. 背景与目标

当前 `docs/` 目录中的很多文档来自 `sparsify-ascend` 时代，内容混杂了：

- 已删除或不再维护的实验性功能
- 已经过时的训练路径与参数体系
- 以 Ascend 为主的平台叙述
- 研究设计稿、代码走读、性能 profiling、用户指南等不同类型内容

而当前仓库已经过一次明显收敛的重构，代码主线更接近一个精简版的 SAE 训练基础设施，核心实现集中在：

- `sparsify/__main__.py`
- `sparsify/config.py`
- `sparsify/trainer.py`
- `sparsify/sparse_coder.py`
- `sparsify/tiled_sparse_coder.py`
- `sparsify/device.py`
- `sparsify/utils.py`

结合 `LUTurbo-doc/research-log.md`，本仓库在整个 LUTurbo 项目中的定位也已经比较清晰：

- 主要负责 SAE 训练
- 负责误差阈值统计等配套前处理
- 负责 LUT 导出前的中间产物准备
- 不负责 LUTurbo 的完整推理系统

因此，文档迁移的目标不是“修补旧文档”，而是重建一套以当前代码和当前项目定位为中心的文档体系。

## 2. 迁移原则

### 2.1 以当前代码为准

文档中的“功能说明”“配置项”“训练流程”“支持的平台”都应以当前代码事实为准，而不是沿用旧文档中的历史表述。

### 2.2 以 LUTurbo 项目语境重新定义仓库定位

文档需要明确写清：Sparsify 是 LUTurbo 项目中的 SAE 训练与导出基础设施，而不是完整的 LUTurbo 推理仓库。

### 2.3 默认平台切换为 NVIDIA/CUDA

当前仓库虽然历史上叫 `sparsify-ascend`，但实际主运行平台已经转向 NVIDIA。文档口径应改为：

- NVIDIA/CUDA 为默认主线
- Ascend/NPU 为额外兼容或历史实验支持
- Ascend 文档降级为归档材料，不再作为主入口

### 2.4 分离“稳定事实”和“历史探索”

主文档只保留当前真实使用路径与稳定能力。

以下内容不应继续混在主文档入口中：

- 已废弃功能设计稿
- 历史 profiling 报告
- 已结束的重构设计讨论
- 尚未进入当前主线实现的探索性方案

这些内容可以保留，但应统一迁移到 `archive/`。

## 3. 当前问题总结

### 3.1 主文档与当前代码不一致

旧文档中大量提到如下概念或功能：

- `transcode`
- `hook_mode`
- `loss_fn = ce / kl`
- `groupmax`
- `multi_topk`
- `skip_connection`
- `lowrank`
- `distill`
- `outlier clip`
- 多优化器分支（如 muon / adam）
- distribute_modules / 8bit 等旧分支

这些内容大多已经不是当前主线，部分甚至已经从当前代码设计中退出。

### 3.2 多篇文档之间高度重复

以下文档存在明显重叠：

- `docs/code_walkthrough.md`
- `docs/trainer_code_walkthrough.md`
- `docs/hook_function_analysis.md`

它们都在解释 trainer / hook / 流程，但基于的是旧版本结构，重复且容易互相矛盾。

### 3.3 Ascend 文档权重过高

`docs/ascend/` 里的 profiling 文档保留了大量历史分析，但它们不应再代表当前仓库的主叙事方向。

### 3.4 用户指南中混入了旧参数体系

例如 `docs/qwen3_sae_training_guide.md` 中仍包含许多当前不应继续宣传的配置项与命令风格，容易误导实际使用。

## 4. 新文档体系建议

建议将 `docs/` 重构为如下结构：

```text
docs/
├── README.md
├── overview.md
├── training/
│   ├── quickstart.md
│   ├── config-reference.md
│   └── qwen3-guide.md
├── architecture/
│   ├── core-components.md
│   ├── training-pipeline.md
│   └── performance.md
├── export/
│   └── sae-to-lut.md
└── archive/
    ├── ascend/
    ├── refactor/
    └── ideas/
```

这个结构对应三类内容：

- 主入口与项目定位
- 当前可用训练/实现文档
- 历史归档

## 5. 新文档职责定义

### 5.1 `docs/README.md`

职责：

- 作为文档首页
- 说明仓库定位
- 给出推荐阅读顺序
- 区分主文档与历史归档

必须明确写清：

- 当前主平台是 NVIDIA/CUDA
- Ascend 文档只是兼容/历史资料
- 本仓库服务于 LUTurbo 的 SAE 训练与导出链路

### 5.2 `docs/overview.md`

职责：

- 解释 Sparsify 在 LUTurbo 整体方案中的位置
- 描述本仓库的输入、输出与边界
- 将 SAE 训练、阈值统计、LUT 导出联系起来

建议覆盖内容：

- 训练数据与激活来源
- SAE checkpoint 产物
- `compute_elbow_thresholds.py` 的作用
- `convert_sae_to_lut.py` 的作用
- 与 LUTurbo 查表推理方案的衔接关系

### 5.3 `docs/training/quickstart.md`

职责：

- 给用户一个最短路径的上手流程
- 默认按 NVIDIA/CUDA 场景来写

建议覆盖内容：

- 安装方式
- 最小训练命令
- 常见输出目录
- checkpoint 结构
- 常见运行注意事项

### 5.4 `docs/training/config-reference.md`

职责：

- 提供当前真实可用的参数参考
- 以 `sparsify/config.py` 和 `sparsify/__main__.py` 为唯一事实来源

建议参数分组：

- 运行参数
- SAE 参数
- 训练参数
- hookpoint 选择
- tiling
- hadamard
- exceed metrics
- 保存与日志
- finetune / resume

### 5.5 `docs/training/qwen3-guide.md`

职责：

- 提供 Qwen3 场景下的当前训练建议
- 代替旧的 `docs/qwen3_sae_training_guide.md`

建议覆盖内容：

- 推荐 hookpoint
- 典型训练命令模板
- 不同模块输入的适用性
- 与 LUTurbo 目标有关的建议

注意：

- 不再出现旧参数体系
- 不再把已废弃功能当作可用选项

### 5.6 `docs/architecture/core-components.md`

职责：

- 讲清当前核心模块职责与关系

建议覆盖：

- `sparsify/sparse_coder.py`
- `sparsify/tiled_sparse_coder.py`
- `sparsify/fused_encoder.py`
- `sparsify/fused_decoder.py`
- `sparsify/device.py`
- `sparsify/utils.py`

### 5.7 `docs/architecture/training-pipeline.md`

职责：

- 成为当前代码结构理解的主入口
- 合并旧的 trainer / hook / walkthrough 文档

建议覆盖：

- `__main__.py` 的 CLI 与 artifact 加载
- `Trainer` 初始化过程
- hookpoint 解析
- SAE 初始化
- hook 中的激活捕获与训练
- optimizer step
- dead feature 统计
- checkpoint / logging

### 5.8 `docs/architecture/performance.md`

职责：

- 解释当前还成立的性能优化点
- 区分 CUDA 主线与 NPU 兼容

建议覆盖：

- bf16 autocast
- fused encoder / fused decoder
- partial forward
- `compile_model`
- DDP 下的统计与同步
- tiled / hadamard 的性能与建模关系

### 5.9 `docs/export/sae-to-lut.md`

职责：

- 解释训练好的 SAE 如何进入 LUTurbo 后续流程

建议覆盖：

- SAE checkpoint 到 LUT 的导出关系
- `convert_sae_to_lut.py` 的输入输出
- 阈值文件与 LUT 的关系
- 目前导出格式中哪些部分是稳定的，哪些仍属于“当前约定”

## 6. 旧文档迁移映射

### 6.1 保留并重写

| 旧文档 | 新去向 | 处理方式 |
|------|------|------|
| `docs/qwen3_sae_training_guide.md` | `docs/training/qwen3-guide.md` | 重写 |
| `docs/training_acceleration_guide.md` | `docs/architecture/performance.md` | 重写 |
| `docs/lut_format.md` | `docs/export/sae-to-lut.md` | 吸收并重写 |

### 6.2 合并后移除原入口

| 旧文档 | 新去向 | 处理方式 |
|------|------|------|
| `docs/code_walkthrough.md` | `docs/architecture/training-pipeline.md` | 合并 |
| `docs/trainer_code_walkthrough.md` | `docs/architecture/training-pipeline.md` | 合并 |
| `docs/hook_function_analysis.md` | `docs/architecture/training-pipeline.md` | 合并 |

### 6.3 迁移到归档

| 旧文档 | 新去向 | 处理方式 |
|------|------|------|
| `docs/refactoring_design.md` | `docs/archive/refactor/refactoring_design.md` | 归档 |
| `docs/hook_mode_design.md` | `docs/archive/refactor/hook_mode_design.md` | 归档 |
| `docs/moe_lowrank_encoder_design.md` | `docs/archive/ideas/moe_lowrank_encoder_design.md` | 归档 |
| `docs/tiling_lowrank_two_stage.md` | `docs/archive/ideas/tiling_lowrank_two_stage.md` | 归档 |
| `docs/ascend/ascend_profling.md` | `docs/archive/ascend/ascend_profling.md` | 归档 |
| `docs/ascend/npu_profiling_analysis.md` | `docs/archive/ascend/npu_profiling_analysis.md` | 归档 |
| `docs/ascend/sae_training_profiling_report_20260306.md` | `docs/archive/ascend/sae_training_profiling_report_20260306.md` | 归档 |

### 6.4 建议移出主索引或直接删除

| 旧文档 | 建议 |
|------|------|
| `docs/feature_analysis.md` | 不再作为主文档，建议归档或删除 |
| `docs/talking_with_claude.md` | 不应继续留在主 docs，建议删除或单独归档 |

## 7. 推荐的重写顺序

### 第一阶段：搭建新骨架

优先创建以下文件：

- `docs/README.md`
- `docs/overview.md`
- `docs/training/quickstart.md`
- `docs/architecture/training-pipeline.md`

目的：

- 先建立新的主叙事
- 让读者有一条正确的入口路径

### 第二阶段：补齐实用文档

重点重写：

- `docs/training/config-reference.md`
- `docs/training/qwen3-guide.md`
- `docs/architecture/performance.md`
- `docs/export/sae-to-lut.md`

目的：

- 让使用者能真正按当前代码使用仓库

### 第三阶段：整理归档

创建并迁移：

- `docs/archive/ascend/`
- `docs/archive/refactor/`
- `docs/archive/ideas/`

目的：

- 保留历史信息
- 同时避免其污染主入口

### 第四阶段：同步根 README

在 `docs/` 主体系稳定后，再更新仓库根 `README.md`，使其与新 docs 口径一致。

## 8. 推荐的文档口径

建议在新文档中统一使用如下定位表述：

> Sparsify 是 LUTurbo 项目中的 SAE 训练与导出基础设施，当前主要面向 NVIDIA/CUDA 环境；Ascend 支持仅作为兼容与历史实验保留。

以及：

> 本仓库聚焦 SAE 训练、误差阈值统计与 LUT 导出前处理，不覆盖 LUTurbo 的完整推理系统。

并补充一条原则：

> 文档内容以当前代码实现为准，历史实验与废弃方案统一归档。

## 9. 当前代码主线建议作为事实来源的文件

后续重写文档时，建议优先参考以下文件：

- `sparsify/__main__.py`
- `sparsify/config.py`
- `sparsify/trainer.py`
- `sparsify/sparse_coder.py`
- `sparsify/tiled_sparse_coder.py`
- `sparsify/fused_encoder.py`
- `sparsify/fused_decoder.py`
- `sparsify/device.py`
- `sparsify/utils.py`
- `compute_elbow_thresholds.py`
- `convert_sae_to_lut.py`
- `LUTurbo-doc/research-log.md`

## 10. 结论

本次 docs 迁移的核心不是增量修补，而是重新建立一套：

- 与当前代码一致
- 与 LUTurbo 项目关系清晰
- 以 NVIDIA 为默认主线
- 将历史探索与稳定使用路径明确分层

的文档体系。

推荐执行策略是：

1. 先建立新的主文档骨架
2. 再重写当前真实使用路径相关文档
3. 最后将旧设计稿和 Ascend profiling 全部迁入归档

这样可以在尽量少打扰当前代码的前提下，快速恢复文档的可信度与可维护性。
