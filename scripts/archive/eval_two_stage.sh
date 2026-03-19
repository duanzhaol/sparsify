#normal
python scripts/eval_exceed.py \
      --checkpoint checkpoints/lowrank/qwen3-0.6B-lowrank_dp1_bs8_ga8_ef8_k128_20260118_220622 \
      --model /root/models/Qwen3-0.6B \
      --dataset /root/fineweb-edu/sample/10BT \
      --split train \
      --ctx_len 2048 \
      --batch_size 4 \
      --max_batches 50 \
      --out_json /tmp/qwen3-0.6b-two-stage/exceed_full.json
#slice
  python scripts/eval_exceed.py \
      --checkpoint checkpoints/lowrank/qwen3-0.6B-lowrank_dp1_bs8_ga8_ef8_k128_20260118_220622 \
      --model /root/models/Qwen3-0.6B \
      --dataset /root/fineweb-edu/sample/10BT \
      --split train \
      --ctx_len 2048 \
      --batch_size 4 \
      --max_batches 50 \
      --out_json /tmp/qwen3-0.6b-two-stage/exceed_two_stage_slice.json \
      --encoder_mode two_stage
#随机
  python scripts/eval_exceed.py \
      --checkpoint checkpoints/lowrank/qwen3-0.6B-lowrank_dp1_bs8_ga8_ef8_k128_20260118_220622 \
      --model /root/models/Qwen3-0.6B \
      --dataset /root/fineweb-edu/sample/10BT \
      --split train \
      --ctx_len 2048 \
      --batch_size 4 \
      --max_batches 50 \
      --out_json /tmp/qwen3-0.6b-two-stage/exceed_two_stage_random.json \
      --encoder_mode two_stage \
      --two_stage_proj random
# 训练pca
  python scripts/precompute/train_pca.py \
      --model /root/models/Qwen3-0.6B \
      --dataset /root/fineweb-edu/sample/10BT \
      --split train \
      --ctx_len 2048 \
      --batch_size 4 \
      --hookpoints "layers.[0,5,10,15].self_attn.q_proj" \
      --hook_mode input \
      --max_tokens 2000000 \
      --low_dim 128 \
      --out checkpoints/pca/two_stage_pca.pt
# Two-stage PCA projection
  python scripts/eval_exceed.py \
      --checkpoint checkpoints/lowrank/qwen3-0.6B-lowrank_dp1_bs8_ga8_ef8_k128_20260118_220622 \
      --model /root/models/Qwen3-0.6B \
      --dataset /root/fineweb-edu/sample/10BT \
      --split train \
      --ctx_len 2048 \
      --batch_size 4 \
      --max_batches 50 \
      --out_json /tmp/qwen3-0.6b-two-stage/exceed_two_stage_pca.json \
      --encoder_mode two_stage \
      --two_stage_proj pca \
      --two_stage_pca_path checkpoints/pca/two_stage_pca.pt