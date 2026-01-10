python compute_elbow_thresholds.py \
    ~/models/Qwen3-0.6B/ \
    --dataset ~/fineweb-edu/sample/10BT \
    --hookpoints "layers.[0-35].self_attn.q_proj" \
    --num_tokens 100000 \
    --ctx_len 2048 \
    --output thresholds/Qwen3-0.6B/thresholds_q.json  \
    --max_percentile 0.95


torchrun --nproc_per_node 4 --master_port 29501  -m sparsify \
      ~/models/Qwen3-0.6B/ \
      ~/fineweb-edu/sample/10BT \
      --split "train" \
      --wandb_project 'qwen3-0.6B-0108' \
      --ctx_len 2048 \
      --max_examples 1000000 \
      --text_column "text" \
      --shuffle_seed 1127 \
      --data_preprocessing_num_proc 120 \
      --activation "topk" \
      --expansion_factor 8 \
      --normalize_decoder True \
      --num_latents 0 \
      -k 128 \
      --multi_topk False \
      --skip_connection False \
      --hookpoints "layers.[0,5,10,15].self_attn.q_proj" \
      --hook_mode input \
      --init_seeds 1127 \
      --batch_size 2 \
      --grad_acc_steps 8 \
      --micro_acc_steps 1 \
      --loss_fn "fvu" \
      --optimizer "signum" \
      --lr 8e-4 \
      --auxk_alpha 0.03125 \
      --dead_feature_threshold 10000000 \
      --save_every 200 \
      --save_best True \
      --save_dir "checkpoints" \
      --run_name "qwen3-0.6B" \
      --log_to_wandb True \
      --wandb_log_frequency 1 \
      --elbow_threshold_path ~/sparsify/thresholds/Qwen3-0.6B/thresholds_q.json \
      --max_tokens 200000000 \
      --exceed_alphas 0.10 0.20 0.3 0.4 0.5 0.6 0.7 1.0

torchrun --nproc_per_node 2 --master_port 29501  -m sparsify \
      /model-weights/Qwen3-0.6B/ \
      /mnt/data/fineweb-edu/sample/10BT \
      --split "train" \
      --wandb_project 'qwen3-0.6B-0108' \
      --ctx_len 2048 \
      --max_examples 1000000 \
      --text_column "text" \
      --shuffle_seed 1127 \
      --data_preprocessing_num_proc 120 \
      --activation "topk" \
      --expansion_factor 8 \
      --normalize_decoder True \
      --num_latents 0 \
      -k 128 \
      --multi_topk False \
      --skip_connection False \
      --hookpoints "layers.[0,5,10,15].self_attn.q_proj" \
      --hook_mode input \
      --init_seeds 1127 \
      --batch_size 2 \
      --grad_acc_steps 8 \
      --micro_acc_steps 1 \
      --loss_fn "fvu" \
      --optimizer "signum" \
      --lr 8e-4 \
      --auxk_alpha 0.03125 \
      --dead_feature_threshold 10000000 \
      --save_every 200 \
      --save_best True \
      --save_dir "checkpoints" \
      --run_name "qwen3-0.6B" \
      --log_to_wandb True \
      --wandb_log_frequency 1 \
      --elbow_threshold_path ~/workspace/sparsify/thresholds/Qwen3-0.6B/thresholds_q.json \
      --max_tokens 200000000 \
      --exceed_alphas 0.10 0.20 0.3 0.4 0.5 0.6 0.7 1.0


torchrun --nproc_per_node 8 --master_port 29501  -m sparsify \
      ~/models/Qwen3-0.6B/ \
      ~/fineweb-edu/sample/10BT \
      --split "train" \
      --wandb_project 'qwen3-0.6B-1224-o' \
      --ctx_len 2048 \
      --max_examples 1000000 \
      --text_column "text" \
      --shuffle_seed 42 \
      --data_preprocessing_num_proc 120 \
      --activation "topk" \
      --expansion_factor 8 \
      --normalize_decoder True \
      --num_latents 0 \
      -k 256 \
      --multi_topk False \
      --skip_connection False \
      --hookpoints "layers.[0-30].self_attn.o_proj" \
      --hook_mode input \
      --init_seeds 0 \
      --batch_size 1 \
      --grad_acc_steps 8 \
      --micro_acc_steps 1 \
      --loss_fn "fvu" \
      --optimizer "signum" \
      --lr 8e-4 \
      --auxk_alpha 0.03125 \
      --dead_feature_threshold 10000000 \
      --save_every 200 \
      --save_best True \
      --save_dir "checkpoints" \
      --run_name "qwen3-0.6B" \
      --log_to_wandb True \
      --wandb_log_frequency 1 \
      --elbow_threshold_path ~/sparsify/thresholds/Qwen3-0.6B/thresholds_o.json \
      --max_tokens 250000000 \
      --exceed_alphas 0.05 0.10 0.20 0.50 1.0 2.0


torchrun --nproc_per_node 8 --master_port 29501  -m sparsify \
      ~/models/Qwen3-0.6B/ \
      ~/fineweb-edu/sample/10BT \
      --split "train" \
      --wandb_project 'qwen3-0.6B-1224-up' \
      --ctx_len 2048 \
      --max_examples 1000000 \
      --text_column "text" \
      --shuffle_seed 42 \
      --data_preprocessing_num_proc 120 \
      --activation "topk" \
      --expansion_factor 8 \
      --normalize_decoder True \
      --num_latents 0 \
      -k 128 \
      --multi_topk False \
      --skip_connection False \
      --hookpoints "layers.[0-30].mlp.up_proj" \
      --hook_mode input \
      --init_seeds 0 \
      --batch_size 1 \
      --grad_acc_steps 8 \
      --micro_acc_steps 1 \
      --loss_fn "fvu" \
      --optimizer "signum" \
      --lr 8e-4 \
      --auxk_alpha 0.03125 \
      --dead_feature_threshold 10000000 \
      --save_every 200 \
      --save_best True \
      --save_dir "checkpoints" \
      --run_name "qwen3-0.6B" \
      --log_to_wandb True \
      --wandb_log_frequency 1 \
      --elbow_threshold_path ~/sparsify/thresholds/Qwen3-0.6B/thresholds_up.json \
      --max_tokens 250000000 \
      --exceed_alphas 0.05 0.10 0.20 0.50 1.0 2.0