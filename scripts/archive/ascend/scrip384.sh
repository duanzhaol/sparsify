# 在ascend上测试一下能不能跑通，目前性能大概是a100的1/4，需要profile一下看看

torchrun --nproc_per_node 4 --master_port 29501 -m sparsify \
        /mnt/model/Qwen3-0.6B \
        /tmp/fineweb/sample/10BT \
        --split "train" \
        --wandb_project 'qwen3-0.6B-0304-ascend' \
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
        --batch_size 2 \
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
        --max_tokens 200000000 --elbow_threshold_path ./thresholds/Qwen3-0.6B/thresholds_q.json


#profiling

msprof --output=./prof_output --task-time=on --ai-core=on --ascendcl=on --application="xxxx"