python compute_elbow_thresholds.py \
    ~/models/Qwen3-0.6B/ \
    --dataset ~/fineweb-edu/sample/10BT \
    --hookpoints "layers.[0-35].self_attn.q_proj" \
    --num_tokens 100000 \
    --ctx_len 2048 \
    --output thresholds/Qwen3-0.6B/thresholds_q.json  \
    --max_percentile 0.95


torchrun --nproc_per_node 1 --master_port 29501  -m sparsify \
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
        --batch_size 8 \
        --grad_acc_steps 8 \
        --micro_acc_steps 1 \
        --loss_fn "fvu" \
        --optimizer "signum" \
        --lr 3e-3 \
        --auxk_alpha 0.03125 \
        --dead_feature_threshold 10000000 \
        --save_every 100 \
        --save_best True \
        --save_dir "checkpoints/lowrank" \
        --run_name "qwen3-0.6B-lowrank" \
        --log_to_wandb True \
        --wandb_log_frequency 1 \
        --elbow_threshold_path ~/workspace/sparsify/thresholds/Qwen3-0.6B/thresholds_q.json \
        --exceed_alphas 0.10 0.20 0.3 0.4 0.5 0.6 0.7 1.0 \
        --num_tiles 4