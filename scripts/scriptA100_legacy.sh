torchrun --nproc_per_node 2  --master_port 29501 -m sparsify \
        ~/models/Qwen3-0.6B/ \
        ~/fineweb-edu/sample/10BT-tokenized-qwen3-2048/ \
        --split "train" \
        --wandb_project 'qwen3-0.6B-0311' \
        --ctx_len 2048 \
        --max_examples 1000000 \
        --text_column "text" \
        --shuffle_seed 1127 \
        --data_preprocessing_num_proc 120 \
        --expansion_factor 8 \
        --normalize_decoder True \
        --num_latents 0 \
        -k 128 \
        --hookpoints "layers.[0,6,12,18].self_attn.q_proj" \
        --init_seeds 1127 \
        --batch_size 1 \
        --grad_acc_steps 8 \
        --micro_acc_steps 1 \
        --lr 8e-4 \
        --auxk_alpha 0.03125 \
        --dead_feature_threshold 10000000 \
        --save_every 100 \
        --save_best True \
        --save_dir "checkpoints/lowrank" \
        --run_name "qwen3-0.6B" \
        --log_to_wandb True \
        --wandb_log_frequency 1 \
        --max_tokens 200000000 --elbow_threshold_path ~/sparsify/thresholds/Qwen3-0.6B/thresholds_q.json


nsys profile     --output logs/nsys/qwen3-0.6B-sae     --force-overwrite true     --trace cuda,nvtx,osrt,cublas,cudnn     --sample none     --cpuctxsw none     --wait all \
torchrun --nproc_per_node 2  --master_port 29501 -m sparsify \
        ~/models/Qwen3-0.6B/ \
        ~/fineweb-edu/sample/10BT-tokenized-qwen3-2048/ \
        --split "train" \
        --wandb_project 'qwen3-0.6B-0311' \
        --ctx_len 2048 \
        --max_examples 1000000 \
        --text_column "text" \
        --shuffle_seed 1127 \
        --data_preprocessing_num_proc 120 \
        --expansion_factor 8 \
        --normalize_decoder True \
        --num_latents 0 \
        -k 128 \
        --hookpoints "layers.[0,6,12,18].self_attn.q_proj" \
        --init_seeds 1127 \
        --batch_size 1 \
        --grad_acc_steps 8 \
        --micro_acc_steps 1 \
        --lr 8e-4 \
        --auxk_alpha 0.03125 \
        --dead_feature_threshold 10000000 \
        --save_every 100 \
        --save_best True \
        --save_dir "checkpoints/lowrank" \
        --run_name "qwen3-0.6B" \
        --log_to_wandb True \
        --wandb_log_frequency 1 \
        --max_tokens 2000000 --elbow_threshold_path ~/sparsify/thresholds/Qwen3-0.6B/thresholds_q.json --compile_model
