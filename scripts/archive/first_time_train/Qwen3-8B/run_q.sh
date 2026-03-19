set -e  # 任何命令失败就立刻退出

# Base command with common parameters
BASE_CMD="torchrun --nproc_per_node 4 --master_port 29501 -m sparsify \
      $HOME/models/Qwen3-8B/ \
      $HOME/fineweb-edu/sample/10BT \
      --split train \
      --wandb_project sparsify_sweep_1221 \
      --ctx_len 2048 \
      --max_examples 1000000 \
      --text_column text \
      --shuffle_seed 42 \
      --data_preprocessing_num_proc 8 \
      --activation topk \
      --expansion_factor 8 \
      --normalize_decoder True \
      --num_latents 0 \
      -k 512 \
      --multi_topk False \
      --skip_connection False \
      --hook_mode input \
      --init_seeds 0 \
      --batch_size 1 \
      --grad_acc_steps 8 \
      --micro_acc_steps 1 \
      --loss_fn fvu \
      --optimizer signum \
      --lr 8e-4 \
      --auxk_alpha 0.03125 \
      --dead_feature_threshold 10000000 \
      --save_every 100000 \
      --save_best True \
      --save_dir /data/checkpoints \
      --run_name qwen3-8b-layers.10.self_attn.o_proj \
      --log_to_wandb True \
      --wandb_log_frequency 1 \
      --elbow_threshold_path ./thresholds_q.json \
      --max_tokens 25000000 \
      --exceed_alphas 0.05 0.10 0.20 0.50 1.0 2.0"

# Run experiments with different hookpoints and ports
# $BASE_CMD --hookpoints "layers.[0-6].self_attn.q_proj"
# $BASE_CMD --hookpoints "layers.[7-13].self_attn.q_proj"
# $BASE_CMD --hookpoints "layers.[14-20].self_attn.q_proj"
$BASE_CMD  --hookpoints "layers.[21-27].self_attn.q_proj"
$BASE_CMD  --hookpoints "layers.[28-34].self_attn.q_proj"