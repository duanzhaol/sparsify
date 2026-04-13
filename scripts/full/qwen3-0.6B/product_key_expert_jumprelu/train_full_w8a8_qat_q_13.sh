MODE=${1:-smoke}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
PER_DEVICE_BATCH_SIZE=${BATCH_SIZE:-2}

if [ "$MODE" = "formal" ]; then
		MAX_TOKENS=200000000
		RUN_SUFFIX=formal
else
		MAX_TOKENS=2000000
		RUN_SUFFIX=smoke
fi

WANDB_PROJECT=qwen3-0.6B-product_key_expert_jumprelu-qproj-full-w8a8-qat-${RUN_SUFFIX} \
		NPROC_PER_NODE=${NPROC_PER_NODE} \
		SAVE_DIR=checkpoints/product_key_expert_jumprelu_qproj_full_w8a8_qat_${RUN_SUFFIX} \
		RUN_NAME=product_key_expert_jumprelu_q_full_w8a8_qat_${RUN_SUFFIX} \
		MAX_TOKENS=${MAX_TOKENS} \
		ARCHITECTURE=product_key_expert_jumprelu \
		K=32 \
		EXPANSION_FACTOR=1 \
		NUM_EXPERTS=512 \
		ACTIVE_EXPERTS=2 \
		LATENTS_PER_EXPERT=56 \
		OPTIMIZER=adam \
		LR=8e-4 \
		HOOKPOINTS='layers.[13].self_attn.q_proj' \
		BATCH_SIZE=${PER_DEVICE_BATCH_SIZE} \
		GRAD_ACC_STEPS=8 \
		MICRO_ACC_STEPS=1 \
		AUXK_ALPHA=0.03125 \
		DEAD_FEATURE_THRESHOLD=10000000 \
		USE_HADAMARD=0 \
		COMPILE_MODEL=0 \
		IO_QUANT_MODE=qat_full_w8a8 \
		IO_QUANT_BITS=8 \
		IO_QUANT_GRANULARITY=per_token \
		IO_QUANT_CLIP_MODE=absmax \
		IO_LOSS_MODE=dual_target \
		IO_LOSS_DEPLOY_WEIGHT=0.25 \
		bash scripts/autoresearch_test.sh
