#!/bin/bash
# Parallel hyperparameter sweep - runs multiple experiments simultaneously
# Requires: 8 GPUs (or modify GPU allocation)
# Usage: bash scripts/parallel_sweep.sh

set -e

# Configuration
MODEL="~/models/Qwen3-8B/"
DATASET="~/fineweb-edu/sample/10BT"
MAX_TOKENS=100000000  # 100M tokens per run

# GPU allocation: each experiment uses 4 GPUs
# Assumes you have 8 GPUs total
GPU_GROUP_SIZE=4
GPU_GROUPS=("0,1,2,3" "4,5,6,7")

# Hyperparameter grids
EXPANSION_FACTORS=(4 8 16)
K_VALUES=(16 24 32 40 48 64)

# Base command (without GPU specification)
BASE_CMD="torchrun --nproc_per_node $GPU_GROUP_SIZE -m sparsify \
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

# Generate all experiments
EXPERIMENTS=()
for EF in "${EXPANSION_FACTORS[@]}"; do
  for K in "${K_VALUES[@]}"; do
    EXPERIMENTS+=("$EF:$K")
  done
done

TOTAL=${#EXPERIMENTS[@]}
echo "========================================"
echo "Parallel Hyperparameter Sweep"
echo "========================================"
echo "Total experiments: $TOTAL"
echo "GPU groups: ${#GPU_GROUPS[@]}"
echo "GPUs per experiment: $GPU_GROUP_SIZE"
echo "Max parallel experiments: ${#GPU_GROUPS[@]}"
echo "========================================"
echo ""

# Confirm
read -p "Start parallel sweep? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Track running jobs
declare -A RUNNING_JOBS  # job_id -> experiment_name
declare -A JOB_GPUS      # job_id -> gpu_group
COMPLETED=0
FAILED=0
CURRENT_EXP=0

# Function to start an experiment
start_experiment() {
    local ef=$1
    local k=$2
    local gpu_group=$3
    local port=$4

    local run_name="sweep_ef${ef}_k${k}_$(date +%m%d_%H%M)"
    local cmd="$BASE_CMD --expansion_factor $ef -k $k --run_name $run_name --master_port $port"
    local log_file="logs/${run_name}.log"

    mkdir -p logs

    echo "[$(date +%H:%M:%S)] Starting: $run_name on GPUs $gpu_group"

    # Run in background with specific GPUs
    (
        export CUDA_VISIBLE_DEVICES=$gpu_group
        eval $cmd > "$log_file" 2>&1
        echo $? > "logs/${run_name}.exit_code"
    ) &

    local job_id=$!
    RUNNING_JOBS[$job_id]=$run_name
    JOB_GPUS[$job_id]=$gpu_group

    echo "  Job ID: $job_id"
    echo "  Log: $log_file"
}

# Function to wait for any job to complete
wait_for_slot() {
    while true; do
        for job_id in "${!RUNNING_JOBS[@]}"; do
            if ! kill -0 $job_id 2>/dev/null; then
                # Job finished
                local run_name=${RUNNING_JOBS[$job_id]}
                local gpu_group=${JOB_GPUS[$job_id]}

                # Check exit code
                local exit_code_file="logs/${run_name}.exit_code"
                local exit_code=1
                if [[ -f "$exit_code_file" ]]; then
                    exit_code=$(cat "$exit_code_file")
                fi

                if [[ $exit_code -eq 0 ]]; then
                    echo "[$(date +%H:%M:%S)] ✓ Completed: $run_name"
                    COMPLETED=$((COMPLETED + 1))
                else
                    echo "[$(date +%H:%M:%S)] ✗ Failed: $run_name (exit code: $exit_code)"
                    FAILED=$((FAILED + 1))
                fi

                # Remove from tracking
                unset RUNNING_JOBS[$job_id]
                unset JOB_GPUS[$job_id]

                # Return the freed GPU group
                echo "$gpu_group"
                return
            fi
        done
        sleep 5
    done
}

# Main loop: schedule experiments
PORT=29500
for exp in "${EXPERIMENTS[@]}"; do
    CURRENT_EXP=$((CURRENT_EXP + 1))
    IFS=':' read -r ef k <<< "$exp"

    # Wait for an available GPU group
    if [[ ${#RUNNING_JOBS[@]} -ge ${#GPU_GROUPS[@]} ]]; then
        echo ""
        echo "All GPU groups busy, waiting for a slot..."
        AVAILABLE_GPU_GROUP=$(wait_for_slot)
    else
        # Find an available GPU group
        for gpu_group in "${GPU_GROUPS[@]}"; do
            local in_use=false
            for used_gpu in "${JOB_GPUS[@]}"; do
                if [[ "$used_gpu" == "$gpu_group" ]]; then
                    in_use=true
                    break
                fi
            done
            if [[ "$in_use" == false ]]; then
                AVAILABLE_GPU_GROUP=$gpu_group
                break
            fi
        done
    fi

    echo ""
    echo "========================================"
    echo "Experiment $CURRENT_EXP/$TOTAL"
    echo "ef=$ef, k=$k"
    echo "========================================"

    start_experiment $ef $k $AVAILABLE_GPU_GROUP $PORT
    PORT=$((PORT + 1))

    echo "Active experiments: ${#RUNNING_JOBS[@]}/${#GPU_GROUPS[@]}"
    echo ""

    sleep 2  # Small delay between launches
done

# Wait for all remaining jobs
echo ""
echo "Waiting for remaining experiments to complete..."
while [[ ${#RUNNING_JOBS[@]} -gt 0 ]]; do
    wait_for_slot > /dev/null
done

# Summary
echo ""
echo "========================================"
echo "Parallel Sweep Summary"
echo "========================================"
echo "Total experiments: $TOTAL"
echo "Completed: $COMPLETED"
echo "Failed: $FAILED"
echo "Logs directory: logs/"
echo "========================================"
