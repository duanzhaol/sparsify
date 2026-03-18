#!/bin/bash
# Oracle Baseline B: Hotset Selection (C1h)
# 热集选择 oracle 上界分析

set -e
cd "$(dirname "$0")/../../.."

# === 配置 ===
MODEL=/root/models/Qwen3-0.6B
LUT_DIR=/root/models/Qwen3-0.6B/lut
DATASET=/root/fineweb-edu/sample/10BT-tokenized-qwen3-2048/

# === 层选择（取消注释你要用的） ===
LAYERS="0 7 14 21 27"       # 5层均匀采样（默认）
# LAYERS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27"  # 全部28层
# LAYERS="7 14"             # 快速测试用

# === 算子选择（取消注释你要用的，可组合） ===
# OP_TYPES="mlp"              # 只测 MLP (gate_up_proj)，默认
OP_TYPES="mlp qkv o"      # 全部3种算子
# OP_TYPES="mlp qkv"        # MLP + attention QKV
# OP_TYPES="qkv"            # 只测 attention QKV (qkv_proj)
# OP_TYPES="o"              # 只测 attention output (o_proj)

# === 数据量 ===
NUM_SAMPLES=2560
SEQ_LEN=512
# NUM_SAMPLES=16; SEQ_LEN=128  # 快速测试

# === 输出 ===
OUTPUT_DIR=results/activation_patterns/hotset

python -u -m experiments.activation_patterns.hotset.run \
    --model $MODEL --lut_dir $LUT_DIR --dataset $DATASET \
    --num_samples $NUM_SAMPLES --seq_len $SEQ_LEN \
    --layers $LAYERS \
    --op_types $OP_TYPES \
    --output_dir $OUTPUT_DIR/ --device cuda

# === 汇总 ===
echo "Generating CSV summary..."
python -m experiments.activation_patterns.summarize \
    --results_dir results/activation_patterns/ \
    --output results/activation_patterns/summary.csv
