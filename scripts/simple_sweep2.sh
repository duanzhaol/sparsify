#!/bin/bash
# Simple hyperparameter sweep script
# Usage: bash scripts/simple_sweep.sh

set -e  # Exit on error

# Configuration
MODEL="~/models/Qwen3-8B/"
DATASET="~/fineweb-edu/sample/10BT"
NUM_GPUS=4
MAX_TOKENS=1000000  # 100M tokens per run

# Hyperparameter grids
EXPANSION_FACTORS=(4 8 12 16 20)
K_VALUES=(24 32)

# Base command template
BASE_CMD="torchrun --nproc_per_node $NUM_GPUS --master_port 29501 -m sparsify \
  $MODEL \
  $DATASET \
  --split train \
  --ctx_len 2048 \
  --hookpoints 'layers.0.self_attn.o_proj' \
  --hook_mode input \
  --batch_size 1 \
  --grad_acc_steps 8 \
  --micro_acc_steps 1 \
  --max_tokens $MAX_TOKENS \
  --activation topk \
  --normalize_decoder True \
  --num_latents 0 \
  --multi_topk False \
  --skip_connection False \
  --loss_fn fvu \
  --optimizer signum \
  --lr 5e-3 \
  --auxk_alpha 0.03125 \
  --dead_feature_threshold 10000000 \
  --save_every 100 \
  --save_best True \
  --save_dir checkpoints \
  --log_to_wandb True \
  --wandb_log_frequency 1 \
  --data_preprocessing_num_proc 8 \
  --shuffle_seed 42 \
  --text_column text \
  --init_seeds 0"

# Count total experiments
TOTAL=$((${#EXPANSION_FACTORS[@]} * ${#K_VALUES[@]}))
CURRENT=0
SUCCESSFUL=0
FAILED=0

echo "========================================"
echo "Hyperparameter Sweep"
echo "========================================"
echo "Total experiments: $TOTAL"
echo "Expansion factors: ${EXPANSION_FACTORS[*]}"
echo "K values: ${K_VALUES[*]}"
echo "Tokens per run: $MAX_TOKENS"
echo "========================================"
echo ""

# Confirm
read -p "Start sweep? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Log file
LOGFILE="sweep_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $LOGFILE"
echo ""

# Run all combinations
for EF in "${EXPANSION_FACTORS[@]}"; do
  for K in "${K_VALUES[@]}"; do
    CURRENT=$((CURRENT + 1))
    RUN_NAME="sweep_ef${EF}_k${K}_$(date +%m%d_%H%M)"

    echo "========================================"
    echo "Experiment $CURRENT/$TOTAL"
    echo "expansion_factor=$EF, k=$K"
    echo "run_name=$RUN_NAME"
    echo "========================================"

    # Build and run command
    CMD="$BASE_CMD --expansion_factor $EF -k $K --run_name $RUN_NAME"

    # Log command
    echo "[$(date)] Running: $RUN_NAME" >> "$LOGFILE"
    echo "$CMD" >> "$LOGFILE"

    # Execute
    if eval $CMD 2>&1 | tee -a "$LOGFILE"; then
      echo "✓ Success: $RUN_NAME"
      SUCCESSFUL=$((SUCCESSFUL + 1))
    else
      echo "✗ Failed: $RUN_NAME"
      FAILED=$((FAILED + 1))
      echo "[$(date)] FAILED: $RUN_NAME" >> "$LOGFILE"

      # Ask whether to continue
      read -p "Continue to next experiment? [Y/n]: " -n 1 -r
      echo
      if [[ $REPLY =~ ^[Nn]$ ]]; then
        echo "Stopping sweep."
        break 2
      fi
    fi

    echo ""
    sleep 5  # Wait between experiments
  done
done

# Summary
echo "========================================"
echo "Sweep Summary"
echo "========================================"
echo "Completed: $CURRENT/$TOTAL"
echo "Successful: $SUCCESSFUL"
echo "Failed: $FAILED"
echo "Log file: $LOGFILE"
echo "========================================"
