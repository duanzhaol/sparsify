#预处理数据集

python scripts/preprocess_dataset.py \
      --model ~/models/Qwen3-0.6B \
      --dataset ~/fineweb-edu/sample/10BT \
      --output ~/fineweb-edu/sample/10BT-tokenized-qwen3-2048 \
      --ctx_len 2048 \
      --num_proc 16


# 根据 product_key_expert_jumprelu checkpoint 生成 LUT
python scripts/export/export_product_key_expert_jumprelu_lut.py /root/models/Qwen3-0.6B /root/sparsify/checkpoints/ \
      --output-dir ./lut_output \
      --merge-output-dir ./lut_merge \
      --operators qkv gate_up \
      --layers 0-27 \
      --compensation-ratio 0.25 \
      --dtype bfloat16 \
      --device cpu
