#!/usr/bin/env bash

set -euo pipefail

# 分布式启动参数。
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29501}"

# 输入输出路径配置。
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3-0.6B}"
ACTIVATION_SOURCE="${ACTIVATION_SOURCE:-hf_bf16}"
ACTIVATION_BACKBONE_PATH="${ACTIVATION_BACKBONE_PATH:-}"
DATASET_PATH="${DATASET_PATH:-$HOME/fineweb-edu/sample/10BT-tokenized-qwen3-2048}"
ELBOW_THRESHOLD_PATH="${ELBOW_THRESHOLD_PATH:-$HOME/sparsify/thresholds/Qwen3-0.6B/thresholds_q.json}"
WANDB_PROJECT="${WANDB_PROJECT:-qwen3-0.6B-auto-qproj}"
SAVE_DIR="${SAVE_DIR:-checkpoints/auto_qproj}"
RUN_NAME="${RUN_NAME:-qwen3-0.6B-qproj}"

# 数据集预处理与采样参数。
MAX_EXAMPLES="${MAX_EXAMPLES:-1000000}"
SHUFFLE_SEED="${SHUFFLE_SEED:-1127}"
INIT_SEED="${INIT_SEED:-1127}"
DATA_PREPROCESSING_NUM_PROC="${DATA_PREPROCESSING_NUM_PROC:-120}"

# SAE 架构与目标 hook 配置。
ARCHITECTURE="${ARCHITECTURE:-topk}"
EXPANSION_FACTOR="${EXPANSION_FACTOR:-1}"
K="${K:-128}"
HOOKPOINTS="${HOOKPOINTS:-layers.[3].self_attn.q_proj}"
OPTIMIZER="${OPTIMIZER:-signum}"
USE_HADAMARD="${USE_HADAMARD:-0}"
IO_QUANT_MODE="${IO_QUANT_MODE:-off}"
IO_QUANT_BITS="${IO_QUANT_BITS:-8}"
IO_QUANT_GRANULARITY="${IO_QUANT_GRANULARITY:-per_token}"
IO_QUANT_CLIP_MODE="${IO_QUANT_CLIP_MODE:-absmax}"
IO_LOSS_MODE="${IO_LOSS_MODE:-dual_target}"
IO_LOSS_DEPLOY_WEIGHT="${IO_LOSS_DEPLOY_WEIGHT:-0.25}"

# 优化器与有效 batch 配置。
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-8}"
MICRO_ACC_STEPS="${MICRO_ACC_STEPS:-1}"
LR="${LR:-8e-4}"
MAX_TOKENS="${MAX_TOKENS:-20000000}"
AUXK_ALPHA="${AUXK_ALPHA:-0.03125}"
DEAD_FEATURE_THRESHOLD="${DEAD_FEATURE_THRESHOLD:-10000000}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
WANDB_LOG_FREQUENCY="${WANDB_LOG_FREQUENCY:-10}"

# 可选架构/正则附加参数。
#
# 注意：如果 AutoResearch 新增了可调结构参数，必须同步在这里读取并透传到
# `python -m sparsify`。仅在 agent/env_overrides/runner 中注册参数是不够的；
# 若本脚本未追加对应 `--flag`，训练会静默回退到默认值，导致实验结果无效。
TRUNK_RANK="${TRUNK_RANK:-}"
NUM_CODES="${NUM_CODES:-}"
STAGE1_RATIO="${STAGE1_RATIO:-}"
FACTORIZED_HIDDEN_DIM="${FACTORIZED_HIDDEN_DIM:-}"
NUM_EXPERTS="${NUM_EXPERTS:-}"
ACTIVE_EXPERTS="${ACTIVE_EXPERTS:-}"
LATENTS_PER_EXPERT="${LATENTS_PER_EXPERT:-}"
NUM_GROUPS="${NUM_GROUPS:-}"
ACTIVE_GROUPS="${ACTIVE_GROUPS:-}"
JUMPRELU_INIT_THRESHOLD="${JUMPRELU_INIT_THRESHOLD:-}"
JUMPRELU_BANDWIDTH="${JUMPRELU_BANDWIDTH:-}"
ROUTER_LOAD_BALANCE_ALPHA="${ROUTER_LOAD_BALANCE_ALPHA:-}"
ROUTER_WARMUP_TOKENS="${ROUTER_WARMUP_TOKENS:-}"
ROUTER_WARMUP_INIT_TEMPERATURE="${ROUTER_WARMUP_INIT_TEMPERATURE:-}"
ORTHO_LAMBDA="${ORTHO_LAMBDA:-}"
MULTI_KS="${MULTI_KS:-}"
MATRYOSHKA_KS="${MATRYOSHKA_KS:-}"
MATRYOSHKA_WEIGHTS="${MATRYOSHKA_WEIGHTS:-}"
RESIDUAL_FROM="${RESIDUAL_FROM:-}"

# 可选运行时功能。
PROFILE="${PROFILE:-0}"
PROFILE_OUTPUT="${PROFILE_OUTPUT:-logs/nsys/qwen3-0.6B-sae}"
COMPILE_MODEL="${COMPILE_MODEL:-1}"
RESUME="${RESUME:-0}"
PRINT_COST_BREAKDOWN="${PRINT_COST_BREAKDOWN:-1}"

if [[ "${PRINT_COST_BREAKDOWN}" == "1" ]]; then
  ARCHITECTURE="${ARCHITECTURE}" \
  K="${K}" \
  EXPANSION_FACTOR="${EXPANSION_FACTOR}" \
  HOOKPOINTS="${HOOKPOINTS}" \
  TRUNK_RANK="${TRUNK_RANK}" \
  NUM_CODES="${NUM_CODES}" \
  STAGE1_RATIO="${STAGE1_RATIO}" \
  FACTORIZED_HIDDEN_DIM="${FACTORIZED_HIDDEN_DIM}" \
  NUM_EXPERTS="${NUM_EXPERTS}" \
  ACTIVE_EXPERTS="${ACTIVE_EXPERTS}" \
  LATENTS_PER_EXPERT="${LATENTS_PER_EXPERT}" \
  MULTI_KS="${MULTI_KS}" \
  python - <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

try:
    from research.AutoResearch.compatibility import extract_cost_extra_config
    from research.AutoResearch.target_profile import resolve_target_profile
    from sparsify.config import SparseCoderConfig
    from sparsify.sparse_coder import _get_sae_class

    cfg = {
        "HOOKPOINTS": os.environ.get("HOOKPOINTS", ""),
        "TRUNK_RANK": os.environ.get("TRUNK_RANK", ""),
        "NUM_CODES": os.environ.get("NUM_CODES", ""),
        "STAGE1_RATIO": os.environ.get("STAGE1_RATIO", ""),
        "FACTORIZED_HIDDEN_DIM": os.environ.get("FACTORIZED_HIDDEN_DIM", ""),
        "NUM_EXPERTS": os.environ.get("NUM_EXPERTS", ""),
        "ACTIVE_EXPERTS": os.environ.get("ACTIVE_EXPERTS", ""),
        "LATENTS_PER_EXPERT": os.environ.get("LATENTS_PER_EXPERT", ""),
    }
    profile = resolve_target_profile(cfg)
    all_ks = {int(os.environ["K"])}
    if os.environ.get("MULTI_KS"):
        all_ks.update(
            int(v) for v in os.environ["MULTI_KS"].split(",") if v.strip()
        )

    print(
        f"[cost] proxy: {profile.training_hookpoint} | "
        f"{profile.cost_model_label}"
    )
    for current_k in sorted(all_ks):
        cfg_kwargs = {
            "architecture": os.environ["ARCHITECTURE"],
            "k": current_k,
            "expansion_factor": int(os.environ["EXPANSION_FACTOR"]),
        }
        cfg_kwargs.update(extract_cost_extra_config(cfg) or {})
        sae_cfg = SparseCoderConfig(**cfg_kwargs)
        cls = _get_sae_class(os.environ["ARCHITECTURE"])
        model = cls(profile.d_in, sae_cfg, device="cpu")
        cost = model.selection_cost_estimate(profile.n_output)

        if cost.get("error"):
            print(
                f"[cost][k={current_k}] warning: failed to compute cost estimate: "
                f"{cost['error']}"
            )
            continue

        original = float(cost.get("original_matmul_accesses") or 0.0)
        budget = 1.5 * original if original > 0 else 0.0
        selection = float(cost.get("total_accesses") or 0.0)
        deployment = float(cost.get("deployment_accesses") or 0.0)
        combined = float(cost.get("combined_accesses") or (selection + deployment))

        def ratio(value: float) -> str:
            if original <= 0:
                return "n/a"
            return f"{value / original:.6f}x"

        def budget_ratio(value: float) -> str:
            if budget <= 0:
                return "n/a"
            return f"{value / budget:.6f}x"

        print(
            f"[cost][k={current_k}] original_matmul_accesses={int(original)} "
            f"budget_accesses={int(budget)}"
        )
        print(
            f"[cost][k={current_k}] selection_accesses={int(selection)} "
            f"selection_ratio={ratio(selection)} "
            f"selection_budget_ratio={budget_ratio(selection)}"
        )
        print(
            f"[cost][k={current_k}] deployment_accesses={int(deployment)} "
            f"deployment_ratio={ratio(deployment)}"
        )
        print(
            f"[cost][k={current_k}] total_accesses={int(combined)} "
            f"total_ratio={ratio(combined)} "
            f"total_budget_ratio={budget_ratio(combined)} "
            f"feasible={combined <= budget if budget > 0 else 'n/a'}"
        )
except Exception as exc:
    print(f"[cost] warning: failed to compute cost estimate: {exc}")
PY
fi

cmd=(
  torchrun
  --nproc_per_node "${NPROC_PER_NODE}"
  --master_port "${MASTER_PORT}"
  -m sparsify
  "${MODEL_PATH}"
  "${DATASET_PATH}"
  --split train
  --wandb_project "${WANDB_PROJECT}"
  --activation_source "${ACTIVATION_SOURCE}"
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
  --wandb_log_frequency "${WANDB_LOG_FREQUENCY}"
  --max_tokens "${MAX_TOKENS}"
  --elbow_threshold_path "${ELBOW_THRESHOLD_PATH}"
  --io_quant_mode "${IO_QUANT_MODE}"
  --io_quant_bits "${IO_QUANT_BITS}"
  --io_quant_granularity "${IO_QUANT_GRANULARITY}"
  --io_quant_clip_mode "${IO_QUANT_CLIP_MODE}"
  --io_loss_mode "${IO_LOSS_MODE}"
  --io_loss_deploy_weight "${IO_LOSS_DEPLOY_WEIGHT}"
)

if [[ -n "${ACTIVATION_BACKBONE_PATH}" ]]; then
  cmd+=(--activation_backbone_path "${ACTIVATION_BACKBONE_PATH}")
fi

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

if [[ -n "${TRUNK_RANK}" ]]; then
  cmd+=(--trunk_rank "${TRUNK_RANK}")
fi

if [[ -n "${NUM_CODES}" ]]; then
  cmd+=(--num_codes "${NUM_CODES}")
fi

if [[ -n "${STAGE1_RATIO}" ]]; then
  cmd+=(--stage1_ratio "${STAGE1_RATIO}")
fi

if [[ -n "${FACTORIZED_HIDDEN_DIM}" ]]; then
  cmd+=(--factorized_hidden_dim "${FACTORIZED_HIDDEN_DIM}")
fi

if [[ -n "${NUM_EXPERTS}" ]]; then
  cmd+=(--num_experts "${NUM_EXPERTS}")
fi

if [[ -n "${ACTIVE_EXPERTS}" ]]; then
  cmd+=(--active_experts "${ACTIVE_EXPERTS}")
fi

if [[ -n "${LATENTS_PER_EXPERT}" ]]; then
  cmd+=(--latents_per_expert "${LATENTS_PER_EXPERT}")
fi

if [[ -n "${JUMPRELU_INIT_THRESHOLD}" ]]; then
  cmd+=(--jumprelu_init_threshold "${JUMPRELU_INIT_THRESHOLD}")
fi

if [[ -n "${JUMPRELU_BANDWIDTH}" ]]; then
  cmd+=(--jumprelu_bandwidth "${JUMPRELU_BANDWIDTH}")
fi

if [[ -n "${ROUTER_LOAD_BALANCE_ALPHA}" ]]; then
  cmd+=(--router_load_balance_alpha "${ROUTER_LOAD_BALANCE_ALPHA}")
fi

if [[ -n "${ROUTER_WARMUP_TOKENS}" ]]; then
  cmd+=(--router_warmup_tokens "${ROUTER_WARMUP_TOKENS}")
fi

if [[ -n "${ROUTER_WARMUP_INIT_TEMPERATURE}" ]]; then
  cmd+=(--router_warmup_init_temperature "${ROUTER_WARMUP_INIT_TEMPERATURE}")
fi

if [[ -n "${ORTHO_LAMBDA}" ]]; then
  cmd+=(--ortho_lambda "${ORTHO_LAMBDA}")
fi

if [[ -n "${MULTI_KS}" ]]; then
  IFS=',' read -ra _independent_k_vals <<< "${MULTI_KS}"
  cmd+=(--multi_ks "${_independent_k_vals[@]}")
fi

if [[ -n "${RESIDUAL_FROM}" ]]; then
  cmd+=(--residual_from "${RESIDUAL_FROM}")
fi

if [[ -n "${MATRYOSHKA_KS}" ]]; then
  IFS=',' read -ra _mk_vals <<< "${MATRYOSHKA_KS}"
  cmd+=(--matryoshka_ks "${_mk_vals[@]}")
fi

if [[ -n "${MATRYOSHKA_WEIGHTS}" ]]; then
  IFS=',' read -ra _mw_vals <<< "${MATRYOSHKA_WEIGHTS}"
  cmd+=(--matryoshka_weights "${_mw_vals[@]}")
fi

if [[ "${COMPILE_MODEL}" == "1" ]]; then
  cmd+=(--compile_model)
fi

if [[ "${RESUME}" == "1" ]]; then
  cmd+=(--resume)
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
