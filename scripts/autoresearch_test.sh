#!/usr/bin/env bash

set -euo pipefail

# 分布式启动参数。
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29501}"

# 输入输出路径配置。
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3-0.6B}"
DATASET_PATH="${DATASET_PATH:-$HOME/fineweb-edu/sample/10BT-tokenized-qwen3-2048}"
ELBOW_THRESHOLD_PATH="${ELBOW_THRESHOLD_PATH:-$HOME/sparsify/thresholds/Qwen3-0.6B/thresholds_o.json}"
WANDB_PROJECT="${WANDB_PROJECT:-qwen3-0.6B-auto-0323}"
SAVE_DIR="${SAVE_DIR:-checkpoints/auto}"
RUN_NAME="${RUN_NAME:-qwen3-0.6B}"

# 数据集预处理与采样参数。
MAX_EXAMPLES="${MAX_EXAMPLES:-1000000}"
SHUFFLE_SEED="${SHUFFLE_SEED:-1127}"
INIT_SEED="${INIT_SEED:-1127}"
DATA_PREPROCESSING_NUM_PROC="${DATA_PREPROCESSING_NUM_PROC:-120}"

# SAE 架构与目标 hook 配置。
ARCHITECTURE="${ARCHITECTURE:-topk}"
EXPANSION_FACTOR="${EXPANSION_FACTOR:-8}"
K="${K:-128}"
HOOKPOINTS="${HOOKPOINTS:-layers.[3].self_attn.o_proj}"
OPTIMIZER="${OPTIMIZER:-signum}"
USE_HADAMARD="${USE_HADAMARD:-0}"

# 优化器与有效 batch 配置。
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-8}"
MICRO_ACC_STEPS="${MICRO_ACC_STEPS:-1}"
LR="${LR:-8e-4}"
MAX_TOKENS="${MAX_TOKENS:-20000000}"
AUXK_ALPHA="${AUXK_ALPHA:-0.03125}"
DEAD_FEATURE_THRESHOLD="${DEAD_FEATURE_THRESHOLD:-10000000}"
SAVE_EVERY="${SAVE_EVERY:-1000}"

# 可选架构/正则附加参数。
NUM_GROUPS="${NUM_GROUPS:-}"
ACTIVE_GROUPS="${ACTIVE_GROUPS:-}"
JUMPRELU_INIT_THRESHOLD="${JUMPRELU_INIT_THRESHOLD:-}"
JUMPRELU_BANDWIDTH="${JUMPRELU_BANDWIDTH:-}"
ORTHO_LAMBDA="${ORTHO_LAMBDA:-}"
MATRYOSHKA_KS="${MATRYOSHKA_KS:-}"
MATRYOSHKA_WEIGHTS="${MATRYOSHKA_WEIGHTS:-}"
RESIDUAL_FROM="${RESIDUAL_FROM:-}"

# 可选运行时功能。
PROFILE="${PROFILE:-0}"
PROFILE_OUTPUT="${PROFILE_OUTPUT:-logs/nsys/qwen3-0.6B-sae}"
COMPILE_MODEL="${COMPILE_MODEL:-1}"

cmd=(
  torchrun
  --nproc_per_node "${NPROC_PER_NODE}"
  --master_port "${MASTER_PORT}"
  -m sparsify
  "${MODEL_PATH}"
  "${DATASET_PATH}"
  --split train
  --wandb_project "${WANDB_PROJECT}"
  --ctx_len 2048
  --max_examples "${MAX_EXAMPLES}"
  --text_column text
  --shuffle_seed "${SHUFFLE_SEED}"
  --data_preprocessing_num_proc "${DATA_PREPROCESSING_NUM_PROC}"
  --architecture "${ARCHITECTURE}"
  --expansion_factor "${EXPANSION_FACTOR}"
  --normalize_decoder
  --num_latents 0
  -k "${K}"
  --optimizer "${OPTIMIZER}"
  --hookpoints "${HOOKPOINTS}"
  --init_seeds "${INIT_SEED}"
  --batch_size "${BATCH_SIZE}"
  --grad_acc_steps "${GRAD_ACC_STEPS}"
  --micro_acc_steps "${MICRO_ACC_STEPS}"
  --lr "${LR}"
  --auxk_alpha "${AUXK_ALPHA}"
  --dead_feature_threshold "${DEAD_FEATURE_THRESHOLD}"
  --save_every "${SAVE_EVERY}"
  --save_best
  --save_dir "${SAVE_DIR}"
  --run_name "${RUN_NAME}"
  --save_metrics_jsonl
  --log_to_wandb
  --wandb_log_frequency 1
  --max_tokens "${MAX_TOKENS}"
  --elbow_threshold_path "${ELBOW_THRESHOLD_PATH}"
)

if [[ "${USE_HADAMARD}" == "1" ]]; then
  cmd+=(--use_hadamard)
else
  cmd+=(--nouse_hadamard)
fi

if [[ -n "${NUM_GROUPS}" ]]; then
  cmd+=(--num_groups "${NUM_GROUPS}")
fi

if [[ -n "${ACTIVE_GROUPS}" ]]; then
  cmd+=(--active_groups "${ACTIVE_GROUPS}")
fi

if [[ -n "${JUMPRELU_INIT_THRESHOLD}" ]]; then
  cmd+=(--jumprelu_init_threshold "${JUMPRELU_INIT_THRESHOLD}")
fi

if [[ -n "${JUMPRELU_BANDWIDTH}" ]]; then
  cmd+=(--jumprelu_bandwidth "${JUMPRELU_BANDWIDTH}")
fi

if [[ -n "${ORTHO_LAMBDA}" ]]; then
  cmd+=(--ortho_lambda "${ORTHO_LAMBDA}")
fi

if [[ -n "${RESIDUAL_FROM}" ]]; then
  cmd+=(--residual_from "${RESIDUAL_FROM}")
fi

if [[ -n "${MATRYOSHKA_KS}" ]]; then
  IFS=',' read -ra _mk_vals <<< "${MATRYOSHKA_KS}"
  for _mk in "${_mk_vals[@]}"; do
    cmd+=(--matryoshka_ks "${_mk}")
  done
fi

if [[ -n "${MATRYOSHKA_WEIGHTS}" ]]; then
  IFS=',' read -ra _mw_vals <<< "${MATRYOSHKA_WEIGHTS}"
  for _mw in "${_mw_vals[@]}"; do
    cmd+=(--matryoshka_weights "${_mw}")
  done
fi

if [[ "${COMPILE_MODEL}" == "1" ]]; then
  cmd+=(--compile_model)
fi

if [[ "${PROFILE}" == "1" ]]; then
  mkdir -p "$(dirname "${PROFILE_OUTPUT}")"
  exec nsys profile \
    --output "${PROFILE_OUTPUT}" \
    --force-overwrite true \
    --trace cuda,nvtx,osrt,cublas,cudnn \
    --sample none \
    --cpuctxsw none \
    --wait all \
    "${cmd[@]}"
fi

exec "${cmd[@]}"
