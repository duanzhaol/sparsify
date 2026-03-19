# 配置参考

> 说明：本文同步 Phase 2 SAE 架构实验的配置设计，覆盖架构选择、训练增强和结构化 artifact 输出。

本文对应 `sparsify/config.py` 和 `sparsify/__main__.py` 的当前配置定义。

涉及两层配置：

- `SparseCoderConfig`：架构级 SAE 参数
- `TrainConfig` / `RunConfig`：训练循环、日志、数据集和运行时参数

`RunConfig` 继承 `TrainConfig`，所以 CLI 参数集合等于两者并集。

## 运行参数

`RunConfig` 在 `sparsify/__main__.py` 中定义。

| 参数 | 含义 |
| --- | --- |
| `model` | Hugging Face 模型名称或本地路径 |
| `dataset` | Hugging Face 数据集名称、本地数据集路径或 `.bin` memmap 数据集 |
| `split` | 要加载的数据集分割 |
| `ctx_len` | 分词或 memmap 读取时使用的上下文长度 |
| `hf_token` | 用于门控模型的 Hugging Face 令牌 |
| `revision` | 可选的模型版本 |
| `max_examples` | 可选的数据集上限 |
| `resume` | 从现有检查点树恢复 |
| `text_column` | 未分词数据集的文本列名 |
| `shuffle_seed` | 数据集随机种子 |
| `data_preprocessing_num_proc` | 分词工作进程数 |
| `data_args` | 编码为 `k=v,k=v` 的额外 `load_dataset()` 参数 |

说明：

- `dataset` 可以指向 Hugging Face 数据集、本地 `load_from_disk()` 数据集或由 `MemmapDataset` 处理的 memmap `.bin` 文件。
- 如果加载的数据集尚未包含 `input_ids`，CLI 会使用 `sparsify/data.py` 中的 `chunk_and_tokenize()` 进行分词。
- 在 DDP 模式下，rank 0 先加载模型和数据，其他 rank 在同步屏障（barrier）后再加载。数据集还会被裁剪和分片，保证每个 rank 的样本数一致。
- `resume=True` 尝试通过直接匹配 `save_dir/run_name` 或通过 glob 自动生成的运行名称后缀来恢复之前的运行。

## SAE 参数

`SparseCoderConfig` 定义了 SAE 相关架构参数。

| 参数 | 含义 |
| --- | --- |
| `sae.architecture` | 编码架构：`topk`、`gated`、`jumprelu`、`group_topk` |
| `sae.expansion_factor` | 当 `num_latents == 0` 时隐变量（latent）维度的扩展因子 |
| `sae.normalize_decoder` | 将解码器行归一化为单位范数 |
| `sae.num_latents` | 显式隐变量数；`0` 表示从扩展因子派生 |
| `sae.k` | 每个样本保留的隐变量数 |
| `sae.jumprelu_init_threshold` | JumpReLU-fixedK 的阈值初始值 |
| `sae.jumprelu_bandwidth` | JumpReLU STE 的平滑带宽 |
| `sae.num_groups` | Group TopK 的分组数 |
| `sae.active_groups` | Group TopK 中每个输入激活的组数 |

- `num_latents` 默认为 `d_in * expansion_factor`。
- `sae.k` 在标准 SAE 中表示每个样本可激活的特征总数。
- `sae.architecture` 决定 `Trainer` 在 `num_tiles == 1` 时实例化哪一种 SAE 实现。
- `jumprelu_*` 只在 `sae.architecture="jumprelu"` 时生效。
- `num_groups` 与 `active_groups` 只在 `sae.architecture="group_topk"` 时生效。
- `group_topk` 当前的 v0 设计是 hard routing 原型，主要用于验证训练框架与指标结构，不应作为结构化 SAE 可行性的正式判据。
- 分块模式下 `sae.k` 仍是全局预算，`TiledSparseCoder` 会均分成 `k_per_tile = k / num_tiles`。

## 训练参数

在 `TrainConfig` 中定义。

| 参数 | 含义 |
| --- | --- |
| `batch_size` | 序列批次大小 |
| `grad_acc_steps` | 外部训练步骤中的梯度累积 |
| `micro_acc_steps` | 微批次拆分因子 |
| `max_tokens` | 可选的总令牌预算 |
| `lr` | 基础学习率；`None` 使用 `Trainer.__init__()` 中的缩放规则 |
| `optimizer` | 优化器类型：`signum`、`adam`、`muon` |
| `auxk_alpha` | AuxK 死特征损失的权重 |
| `dead_feature_threshold` | 特征被判定为失活（dead）前，自上次激活以来的令牌数 |
| `matryoshka_ks` | 启用 Multi-K 重构损失时使用的 K 列表 |
| `matryoshka_weights` | 对应 `matryoshka_ks` 的损失权重 |
| `ortho_lambda` | 解码器正交性正则权重 |
| `residual_from` | Level 1 SAE checkpoint 路径；设置后训练残差 SAE |
| `save_metrics_jsonl` | 是否在运行目录中写入 `metrics.jsonl` |

说明：

- 训练器通过 `sae.get_param_groups(base_lr)` 收集参数组，因此不同 SAE 变体可以自行决定参数分组策略。
- `optimizer` 选择基础优化器，外层统一包 `ScheduleFreeWrapperReference`。
- 如果省略 `lr`，每个 SAE 参数组获得 `5e-3 / sqrt(num_latents / 2^14)`。
- `grad_acc_steps` 控制优化器步骤何时发生。
- `micro_acc_steps` 目前主要参与损失缩放和日志分母计算。当前实现不会在 hook 内显式拆分激活张量做微批训练。
- `max_tokens` 在累积令牌计数器超过预算后停止训练，然后在返回前保存检查点。
- `matryoshka_ks` 为空时禁用 Matryoshka；启用后会对 top-k 子集重构附加损失。
- `residual_from` 要求 Level 1 checkpoint 的 hookpoint 目录命名与当前训练目标一致，或由调用方自行保证映射关系。

## 超出指标

| 参数 | 含义 |
| --- | --- |
| `exceed_alphas` | 计算 exceed 比率时应用于肘部值的乘数 |
| `elbow_threshold_path` | 包含预计算肘部值的 JSON 文件 |

说明：

- exceed 指标只会在能够匹配到肘部值的 hookpoint 上计算。
- `CheckpointMixin._load_elbow_thresholds()` 先做精确匹配，再按层号/组件名做启发式匹配。
- 训练期间，超出比率基于绝对重建误差计算，并记录为 `exceed_alpha_<alpha>`。

## hookpoint 选择

| 参数 | 含义 |
| --- | --- |
| `hookpoints` | 模块名称或模式的显式列表 |
| `init_seeds` | 初始化每个 hookpoint SAE 时使用的随机种子 |
| `layers` | 未提供 hookpoints 时使用的层索引 |
| `layer_stride` | 对已解析 hookpoint 列表的下采样步幅 |

当前实现行为：

- hookpoint 基于 `model.base_model.named_modules()` 解析
- 如果省略 `hookpoints`，则从模型层列表推断层模块
- 训练在 `Trainer._hook_impl()` 中使用模块输入作为激活值
- 范围语法如 `layers.[7,14].self_attn.o_proj` 由 `sparsify/checkpoint.py` 中的 `expand_range_pattern()` 扩展
- 如果提供多个 `init_seeds`，训练器会按 `<hookpoint>/seed<seed>` 为每个 seed 创建一套 SAE

## 分块

| 参数 | 含义 |
| --- | --- |
| `num_tiles` | 将激活宽度拆分为独立的分块 |
| `global_topk` | 在所有分块预激活上竞争一个全局 top-k |
| `input_mixing` | 在分块级编码前学习一个分块空间混合矩阵 |

注意：

- `d_in` 必须能被 `num_tiles` 整除
- `sae.k` 也必须能被 `num_tiles` 整除
- 分块检查点通过 `sparsify/checkpoint.py` 保存和加载
- `global_topk` 在连接的分块预激活上构建一个联合 top-k
- `input_mixing` 插入一个可学习的分块空间混合矩阵，并在原始激活空间中重新计算 FVU

## Hadamard 旋转

| 参数 | 含义 |
| --- | --- |
| `use_hadamard` | 启用块对角 Hadamard 旋转 |
| `hadamard_block_size` | 块大小；必须是正的 2 的幂 |
| `hadamard_seed` | 置换的种子 |
| `hadamard_use_perm` | 是否在变换前应用置换 |

说明：

- Hadamard 旋转会在训练器首次看到实际激活维度后，为对应 hookpoint 延迟创建。
- 当 Hadamard 激活且启用超出指标时，训练器在测量超出比率前对目标值和重建值都进行反旋转。
- Hadamard 状态在检查点中保存为 `hadamard_rotations.pt`。

## 编译、保存和日志

| 参数 | 含义 |
| --- | --- |
| `compile_model` | 使用 `torch.compile` 编译 Transformer 层；仅在 CUDA 上激活 |
| `save_every` | 每 N 个优化器步骤保存 |
| `save_best` | 保存每个 hookpoint 的最佳检查点 |
| `save_dir` | 基础输出目录 |
| `log_to_wandb` | 启用 Weights & Biases 日志 |
| `run_name` | 可选的运行名称前缀 |
| `wandb_project` | 可选的 W&B 项目覆盖 |
| `wandb_log_frequency` | 日志频率 |
| `finetune` | 用于初始化 SAE 的检查点树路径 |

说明：

- `compile_model` 编译 Transformer 层，而非 SAE 模块本身。
- `save_every` 以优化器步骤边界衡量，而非原始前向钩子调用。
- `save_best` 在 `<run>/best/` 下保存各 hookpoint 的当前最优检查点。
- `finetune` 在 `fit()` 开始前加载现有 SAE 权重；`resume` 恢复 SAE 状态和训练器/优化器状态。
- `save_metrics_jsonl=True` 时，训练目录会额外写入 `manifest.json`、`metrics.jsonl` 和 `summary.json`。
- 如果 W&B 不可用或初始化失败，日志会自动禁用，且该标志会在各 rank 间同步。

## 值得记住的验证规则

`TrainConfig.__post_init__()` 目前强制执行：

- `layers` 和 `layer_stride` 不能同时指定
- `init_seeds` 不能为空
- `exceed_alphas` 必须为正
- `elbow_threshold_path` 在提供时必须存在
- `hadamard_block_size` 必须是正的 2 的幂
- `matryoshka_weights` 若提供，其长度应与 `matryoshka_ks` 一致
- `group_topk` 模式下应保证 `num_groups > 0`、`active_groups > 0`
- `compile_model` 在非 CUDA 后端上被静默禁用

## 检查点命名和布局

新运行构建如下名称：

```text
<run_name>_dp<world_size>_bs<batch_size>_ga<grad_acc_steps>_ef<expansion_factor>_k<k>_<timestamp>
```

保存运行中的重要文件：

- `config.json`：序列化的训练配置
- `manifest.json`：运行身份证，包含架构、模型、数据集、git 信息等元数据
- `metrics.jsonl`：逐步训练指标
- `summary.json`：训练结束后的摘要指标
- `state.pt`：`global_step` 和 `total_tokens`
- `optimizer_0.pt`：优化器状态
- `rank_0_state.pt`：死特征计数器和最佳损失元数据
- `<hookpoint>/cfg.json`：每个 SAE 的配置
- `<hookpoint>/sae.safetensors`：SAE 权重

分块检查点存储：

- 顶层 `cfg.json`，包含 `num_tiles`、`k_per_tile`、`global_topk` 和 `input_mixing`
- 每个分块的 `tile_<i>/sae.safetensors`
- 启用输入混合时的可选 `mixing.pt`
