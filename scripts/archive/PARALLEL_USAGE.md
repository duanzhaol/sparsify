# 并行运行两个Sweep脚本

两个脚本已经配置好，可以在8卡机器上并行运行。

## 脚本配置

### simple_sweep.sh
- Master port: 29500
- 待搜索参数：
  - expansion_factor: [4, 8, 12, 16, 20]
  - k: [8, 16]
- 总实验数: 5 × 2 = 10

### simple_sweep2.sh
- Master port: 29501
- 待搜索参数：
  - expansion_factor: [4, 8, 12, 16, 20]
  - k: [24, 32]
- 总实验数: 5 × 2 = 10

## 并行运行方法

### 方法1：两个Terminal（推荐）

**Terminal 1:**
```bash
cd /root/sparsify
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/simple_sweep.sh
```

**Terminal 2:**
```bash
cd /root/sparsify
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/simple_sweep2.sh
```

### 方法2：后台运行

```bash
cd /root/sparsify

# 启动第一个sweep（GPU 0-3，后台运行）
nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/simple_sweep.sh" > sweep1.log 2>&1 &

# 启动第二个sweep（GPU 4-7，后台运行）
nohup bash -c "CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/simple_sweep2.sh" > sweep2.log 2>&1 &

# 查看日志
tail -f sweep1.log
tail -f sweep2.log
```

### 方法3：使用 screen/tmux

```bash
# Screen方式
screen -S sweep1
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/simple_sweep.sh
# Ctrl+A D 断开

screen -S sweep2
CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/simple_sweep2.sh
# Ctrl+A D 断开

# 重新连接
screen -r sweep1
screen -r sweep2
```

## 监控GPU使用

```bash
watch -n 1 nvidia-smi

# 你应该看到：
# GPU 0-3: sweep1 的进程
# GPU 4-7: sweep2 的进程
```

## 修改搜索参数

编辑脚本中的这些行：

**simple_sweep.sh (line 14-15):**
```bash
EXPANSION_FACTORS=(4 8 12 16 20)
K_VALUES=(8 16)
```

**simple_sweep2.sh (line 14-15):**
```bash
EXPANSION_FACTORS=(4 8 12 16 20)
K_VALUES=(24 32)
```

## 推荐配置

根据你的发现（k=32 比 k=64 好），推荐以下配置：

### 配置A：细化k值搜索

**simple_sweep.sh:**
```bash
EXPANSION_FACTORS=(8)        # 固定你当前的expansion
K_VALUES=(20 24 28)          # 测试k=32以下的值
```

**simple_sweep2.sh:**
```bash
EXPANSION_FACTORS=(8)        # 固定你当前的expansion
K_VALUES=(32 36 40)          # 测试k=32及以上的值
```

### 配置B：测试不同expansion_factor

**simple_sweep.sh:**
```bash
EXPANSION_FACTORS=(4 8 12)   # 较小的expansion
K_VALUES=(32)                # 使用你发现的最优k
```

**simple_sweep2.sh:**
```bash
EXPANSION_FACTORS=(16 20 24) # 较大的expansion
K_VALUES=(32)                # 使用你发现的最优k
```

## 预计时间

假设每个实验100M tokens ≈ 1-2小时：

- **顺序运行**: 20个实验 × 1.5小时 = 30小时
- **并行运行**: 10个实验 × 1.5小时 = 15小时 ⚡

**节省50%时间！**

## 故障排除

### 端口冲突

如果遇到端口冲突，修改脚本中的 MASTER_PORT：
- simple_sweep.sh: `MASTER_PORT=29600`
- simple_sweep2.sh: `MASTER_PORT=29601`

### GPU内存不足

如果4卡内存不够，可以改为8卡配置：

```bash
# 修改脚本中的 NUM_GPUS=8

# 然后顺序运行（不并行）
bash scripts/simple_sweep.sh
bash scripts/simple_sweep2.sh
```

### 查看运行状态

```bash
# 查看正在运行的训练进程
ps aux | grep sparsify

# 查看日志文件
ls -lht sweep_*.log | head
```
