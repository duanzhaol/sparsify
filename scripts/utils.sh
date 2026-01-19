#预处理数据集

python scripts/preprocess_dataset.py \
      --model ~/models/Qwen3-0.6B \
      --dataset ~/fineweb-edu/sample/10BT \
      --output ~/fineweb-edu/sample/10BT-tokenized-qwen3-2048 \
      --ctx_len 2048 \
      --num_proc 16


# 根据sae生成lut
python convert_sae_to_lut.py /root/models/Qwen3-0.6B /root/sparsify/checkpoints/ \
      --output_dir ./lut_output \
      --proj_types qkv gate_up oproj \
      --layers 0-27 \
      --threshold_dir /root/sparsify/thresholds/Qwen3-0.6B \
      --dtype bfloat16