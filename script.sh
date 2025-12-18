torchrun --nproc_per_node 1 -m sparsify \
      ~/models/Qwen3-8B/ \
      ~/fineweb-edu/sample/10BT \
      --split "train" \
      --ctx_len 2048 \
      --max_examples 1000000 \
      --text_column "text" \
      --shuffle_seed 42 \
      --data_preprocessing_num_proc 8 \
      --activation "topk" \
      --expansion_factor 8 \
      --normalize_decoder True \
      --num_latents 0 \
      -k 32 \
      --multi_topk False \
      --skip_connection False \
      --hookpoints "layers.0.self_attn.o_proj" \
      --hook_mode input \
      --init_seeds 0 \
      --batch_size 1 \
      --grad_acc_steps 8 \
      --micro_acc_steps 1 \
      --loss_fn "fvu" \
      --optimizer "signum" \
      --lr 5e-3 \
      --auxk_alpha 0.03125 \
      --dead_feature_threshold 10000000 \
      --save_every 100 \
      --save_best True \
      --save_dir "checkpoints" \
      --run_name "qwen3-8b-layer16-o-proj-input-sae" \
      --log_to_wandb True \
      --wandb_log_frequency 1 \
      --elbow_threshold_path ~/sparsify/thresholds.json \
      --exceed_alphas 0.05 0.10 0.20 0.50 1.0 2.0

CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node 2 --master_port 29502 -m sparsify \
      ~/models/Qwen3-8B/ \
      ~/fineweb-edu/sample/10BT \
      --split "train" \
      --ctx_len 2048 \
      --max_examples 1000000 \
      --text_column "text" \
      --shuffle_seed 42 \
      --data_preprocessing_num_proc 8 \
      --activation "topk" \
      --expansion_factor 8 \
      --normalize_decoder True \
      --num_latents 0 \
      -k 32 \
      --multi_topk False \
      --skip_connection False \
      --hookpoints "layers.0.self_attn.o_proj" \
      --hook_mode input \
      --init_seeds 0 \
      --batch_size 1 \
      --grad_acc_steps 8 \
      --micro_acc_steps 1 \
      --loss_fn "fvu" \
      --optimizer "signum" \
      --lr 5e-3 \
      --auxk_alpha 0.03125 \
      --dead_feature_threshold 10000000 \
      --save_every 100 \
      --save_best True \
      --save_dir "checkpoints" \
      --run_name "qwen3-8b-layer16-o-proj-input-sae" \
      --log_to_wandb True \
      --wandb_log_frequency 1 \
      --elbow_threshold_path ~/sparsify/thresholds.json \
      --exceed_alphas 0.05 0.10 0.20 0.50 1.0 2.0

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port 29501 -m sparsify \
      ~/models/Qwen3-8B/ \
      ~/fineweb-edu/sample/10BT \
      --split "train" \
      --ctx_len 2048 \
      --max_examples 1000000 \
      --text_column "text" \
      --shuffle_seed 42 \
      --data_preprocessing_num_proc 8 \
      --activation "topk" \
      --expansion_factor 8 \
      --normalize_decoder True \
      --num_latents 0 \
      -k 32 \
      --multi_topk False \
      --skip_connection False \
      --hookpoints "layers.0.self_attn.o_proj" \
      --hook_mode input \
      --init_seeds 0 \
      --batch_size 1 \
      --grad_acc_steps 8 \
      --micro_acc_steps 1 \
      --loss_fn "fvu" \
      --optimizer "signum" \
      --lr 5e-3 \
      --auxk_alpha 0.03125 \
      --dead_feature_threshold 10000000 \
      --save_every 100 \
      --save_best True \
      --save_dir "checkpoints" \
      --run_name "qwen3-8b-layer16-o-proj-input-sae" \
      --log_to_wandb True \
      --wandb_log_frequency 1 \
      --elbow_threshold_path ~/sparsify/thresholds.json \
      --exceed_alphas 0.05 0.10 0.20 0.50 1.0 2.0

torchrun --nproc_per_node 8  -m sparsify \
      ~/models/Qwen3-8B/ \
      ~/fineweb-edu/sample/10BT \
      --split "train" \
      --ctx_len 2048 \
      --max_examples 1000000 \
      --text_column "text" \
      --shuffle_seed 42 \
      --data_preprocessing_num_proc 8 \
      --activation "topk" \
      --expansion_factor 8 \
      --normalize_decoder True \
      --num_latents 0 \
      -k 32 \
      --multi_topk False \
      --skip_connection False \
      --hookpoints "layers.0.self_attn.o_proj" \
      --hook_mode input \
      --init_seeds 0 \
      --batch_size 1 \
      --grad_acc_steps 8 \
      --micro_acc_steps 1 \
      --loss_fn "fvu" \
      --optimizer "signum" \
      --lr 5e-3 \
      --auxk_alpha 0.03125 \
      --dead_feature_threshold 10000000 \
      --save_every 100 \
      --save_best True \
      --save_dir "checkpoints" \
      --run_name "qwen3-8b-layer16-o-proj-input-sae" \
      --log_to_wandb True \
      --wandb_log_frequency 1 \
      --elbow_threshold_path ~/sparsify/thresholds.json \
      --exceed_alphas 0.05 0.10 0.20 0.50 1.0 2.0