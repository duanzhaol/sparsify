# Hyperparameter Sweep Scripts

自动化脚本，用于探索不同的 SAE 超参数组合（expansion_factor 和 k）。

## 两种脚本

### 1. Python脚本（推荐）

**优点**：
- 更灵活的配置
- 更好的错误处理
- 支持 dry-run 模式
- 自动生成详细报告

**使用方法**：

```bash
# 查看所有实验配置（不实际运行）
python scripts/hyperparam_sweep.py --dry-run

# 运行完整sweep
python scripts/hyperparam_sweep.py

# 快速测试（只训练少量样本）
python scripts/hyperparam_sweep.py --max-examples 1000

# 失败后继续（不停止整个sweep）
python scripts/hyperparam_sweep.py --continue-on-error

# 使用4张GPU而不是默认的8张
python scripts/hyperparam_sweep.py --gpus 4
```

### 2. Shell脚本（简单）

**优点**：
- 无需Python依赖
- 更直观，易于理解和修改

**使用方法**：

```bash
bash scripts/simple_sweep.sh
```

## 配置超参数

### Python脚本配置

编辑 `scripts/hyperparam_sweep.py` 中的配置部分：

```python
# 要扫描的超参数
SWEEP_PARAMS = {
    "expansion_factor": [4, 8, 16],      # 修改这里
    "k": [16, 24, 32, 40, 48, 64],       # 修改这里
}

# 每个实验训练的token数
BASE_CONFIG = {
    ...
    "max_tokens": 100_000_000,  # 100M tokens
    ...
}
```

### Shell脚本配置

编辑 `scripts/simple_sweep.sh` 的开头部分：

```bash
# Hyperparameter grids
EXPANSION_FACTORS=(4 8 16)              # 修改这里
K_VALUES=(16 24 32 40 48 64)            # 修改这里
MAX_TOKENS=100000000                    # 修改这里
```

## 实验设计建议

### 快速扫描（探索阶段）

```python
SWEEP_PARAMS = {
    "expansion_factor": [4, 8, 16],
    "k": [16, 32, 64],                   # 粗粒度
}
BASE_CONFIG["max_tokens"] = 10_000_000  # 10M tokens
```

预计时间：~9个实验

### 细粒度扫描（优化阶段）

假设发现 expansion_factor=8 最好，细化 k 的搜索：

```python
SWEEP_PARAMS = {
    "expansion_factor": [8],             # 固定最优值
    "k": [24, 28, 32, 36, 40],          # 细粒度扫描
}
BASE_CONFIG["max_tokens"] = 100_000_000  # 100M tokens
```

预计时间：~5个实验

### 完整训练（最终验证）

```python
SWEEP_PARAMS = {
    "expansion_factor": [8],
    "k": [32],                           # 已确定的最优值
}
BASE_CONFIG["max_tokens"] = 1_000_000_000  # 1B tokens
```

## 监控和分析

### 实时监控

训练过程中，可以通过 WandB 实时查看所有实验：

```bash
# 打开WandB项目
# 所有实验会以 "sweep_ef{N}_k{M}" 命名
```

### 结果对比

在 WandB 中：
1. 选择所有 `sweep_` 开头的runs
2. 点击 "Compare" 查看并排对比
3. 关键指标：
   - `fvu`: 重建损失（越低越好）
   - `dead_features_ratio`: 死特征比例（越低越好）
   - `l0`: 实际激活的特征数（应该≈k）

### 生成报告

```bash
# Python脚本会在结束时自动生成摘要
# Shell脚本会生成日志文件
ls sweep_*.log
```

## 常见使用场景

### 场景1：首次探索

不确定哪些超参数好，进行全面扫描：

```bash
# 编辑 hyperparam_sweep.py:
SWEEP_PARAMS = {
    "expansion_factor": [4, 8, 16, 32],
    "k": [16, 32, 64, 128],
}
BASE_CONFIG["max_tokens"] = 50_000_000  # 50M tokens

# 运行
python scripts/hyperparam_sweep.py
```

### 场景2：基于你的发现

你已经发现 k=32 比 k=64 好，想进一步细化：

```bash
# 编辑配置:
SWEEP_PARAMS = {
    "expansion_factor": [8],  # 保持现有配置
    "k": [20, 24, 28, 32, 36, 40],  # 在32附近细搜
}
BASE_CONFIG["max_tokens"] = 100_000_000  # 100M tokens

# 运行
python scripts/hyperparam_sweep.py
```

### 场景3：快速验证

只想快速测试脚本是否工作：

```bash
# 使用很少的数据
python scripts/hyperparam_sweep.py --max-examples 100 --dry-run  # 先预览
python scripts/hyperparam_sweep.py --max-examples 100  # 实际运行
```

## 中断和恢复

### 如果sweep中途中断：

1. **Python脚本**：
   - 使用 `--continue-on-error` 可以在单个实验失败后继续
   - 如果整个脚本中断，需要手动编辑 `SWEEP_PARAMS` 移除已完成的配置

2. **Shell脚本**：
   - 脚本会询问是否继续
   - 可以手动编辑脚本中的数组，移除已完成的配置

### 恢复策略：

```python
# 如果已经完成 ef=4,8 的所有实验，只想继续 ef=16:
SWEEP_PARAMS = {
    "expansion_factor": [16],  # 只保留未完成的
    "k": [16, 24, 32, 40, 48, 64],
}
```

## 注意事项

1. **端口冲突**：脚本会自动递增端口号避免冲突
2. **磁盘空间**：每个实验会保存checkpoints，注意磁盘空间
3. **时间估算**：
   - 10M tokens ≈ 10-20分钟（取决于硬件）
   - 100M tokens ≈ 1-2小时
   - 1B tokens ≈ 10-20小时

4. **GPU内存**：如果OOM，可以减少 `batch_size` 或增加 `grad_acc_steps`

## 示例输出

```
================================================================================
Hyperparameter Sweep Configuration
================================================================================
Total experiments: 18
Sweep parameters:
  - expansion_factor: [4, 8, 16]
  - k: [16, 24, 32, 40, 48, 64]
GPUs per experiment: 8
Tokens per experiment: 100,000,000
================================================================================

Start sweep? [y/N]: y

################################################################################
# Experiment 1/18
################################################################################

Experiment: sweep_ef4_k16_1219_2230
Parameters: expansion_factor=4, k=16
Command: torchrun --nproc_per_node=8 ...

[训练输出...]

✓ Reached target token count: 100,000,064 / 100,000,000
✓ Experiment completed successfully in 87.3 minutes

...

================================================================================
Sweep Summary
================================================================================
Completed: 18/18
Successful: 17
Failed: 1

Results:
  ✓ sweep_ef4_k16_1219_2230 (ef=4, k=16)
  ✓ sweep_ef4_k24_1219_2318 (ef=4, k=24)
  ...
  ✗ sweep_ef16_k64_1220_0342 (ef=16, k=64)

💡 Tip: Compare runs in WandB:
   1. Go to your WandB project
   2. Select all runs starting with 'sweep_'
   3. Click 'Compare' to see side-by-side metrics
================================================================================
```

## 故障排除

### 问题：CUDA OOM

**解决**：减少batch size或增加gradient accumulation

```python
BASE_CONFIG["batch_size"] = 1  # 已经是最小
BASE_CONFIG["grad_acc_steps"] = 16  # 从8增加到16
```

### 问题：端口冲突

**解决**：脚本会自动递增端口，如果仍有冲突，修改起始端口

```python
MASTER_PORT = 29600  # 使用不同的起始端口
```

### 问题：数据加载慢

**解决**：增加data preprocessing进程数

```python
BASE_CONFIG["data_preprocessing_num_proc"] = 16  # 从8增加
```
