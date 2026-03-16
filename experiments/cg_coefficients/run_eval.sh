#!/bin/bash
# CG Coefficients Phase 1 — 全层全算子实验
#
# 用法:
#   bash experiments/cg_coefficients/run_eval.sh
#
# 可选环境变量:
#   LAYERS="0 5 10 15 20 27"   # 自定义层范围 (默认全部 0-27)
#   NUM_SAMPLES=4096            # 样本数 (默认 4096)
#   CG_MAX_ITER=10              # CG 迭代数 (默认 10)

set -euo pipefail

MODEL=/root/models/Qwen3-0.6B
LUT_DIR=/root/models/Qwen3-0.6B/lut
THRESHOLD_DIR=/root/sparsify-ascend/thresholds/Qwen3-0.6B
DATASET=/root/fineweb-edu/sample/10BT-tokenized-qwen3-2048
RESULT_DIR=experiments/cg_coefficients/results/Qwen3-0.6B

NUM_SAMPLES="${NUM_SAMPLES:-4096}"
CG_MAX_ITER="${CG_MAX_ITER:-10}"
EVAL_BATCH_SIZE=256
DEVICE=cuda
TAU_VALUES="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"

# 默认跑全部 28 层；可通过环境变量 LAYERS 覆盖
if [ -z "${LAYERS:-}" ]; then
    LAYERS=$(seq 0 27)
fi

mkdir -p "${RESULT_DIR}"

run_one() {
    local lut_layer="$1"
    local hookpoint="$2"
    local threshold_file="$3"
    local elbow_key="$4"
    local output_name="$5"

    echo "=========================================="
    echo "Running: ${lut_layer}"
    echo "  hookpoint=${hookpoint}"
    echo "  elbow_key=${elbow_key}"
    echo "=========================================="

    python -m experiments.cg_coefficients.eval \
        --lut_dir "${LUT_DIR}" \
        --lut_layer "${lut_layer}" \
        --model "${MODEL}" \
        --hookpoint "${hookpoint}" \
        --dataset "${DATASET}" \
        --num_samples "${NUM_SAMPLES}" \
        --cg_max_iter "${CG_MAX_ITER}" \
        --eval_batch_size "${EVAL_BATCH_SIZE}" \
        --elbow_threshold_file "${threshold_file}" \
        --elbow_key "${elbow_key}" \
        --tau_values ${TAU_VALUES} \
        --output "${RESULT_DIR}/${output_name}.json" \
        --device "${DEVICE}"

    echo ""
}

for LAYER in ${LAYERS}; do
    # MLP gate_up_proj (SAE trained on up_proj input)
    run_one \
        "layers.${LAYER}.mlp.gate_up_proj" \
        "model.layers.${LAYER}.mlp.up_proj" \
        "${THRESHOLD_DIR}/thresholds_up.json" \
        "layer_${LAYER}/mlp_up_proj" \
        "layer${LAYER}_mlp_gate_up_proj"

    # Self-attention qkv_proj (SAE trained on q_proj input)
    run_one \
        "layers.${LAYER}.self_attn.qkv_proj" \
        "model.layers.${LAYER}.self_attn.q_proj" \
        "${THRESHOLD_DIR}/thresholds_q.json" \
        "layer_${LAYER}/self_attn_q_proj" \
        "layer${LAYER}_self_attn_qkv_proj"

    # Self-attention o_proj (SAE trained on o_proj input)
    run_one \
        "layers.${LAYER}.self_attn.o_proj" \
        "model.layers.${LAYER}.self_attn.o_proj" \
        "${THRESHOLD_DIR}/thresholds_o.json" \
        "layer_${LAYER}/self_attn_o_proj" \
        "layer${LAYER}_self_attn_o_proj"
done

# ── Summarize all results to CSV ────────────────────────────────────
echo "=========================================="
echo "Generating CSV summary..."
echo "=========================================="
python -m experiments.cg_coefficients.summarize \
    --results_dir "${RESULT_DIR}" \
    --output "${RESULT_DIR}/summary.csv"

echo "============================================================"
echo "All experiments completed."
echo "  JSON results: ${RESULT_DIR}/"
echo "  CSV summary:  ${RESULT_DIR}/summary.csv"
echo "============================================================"
