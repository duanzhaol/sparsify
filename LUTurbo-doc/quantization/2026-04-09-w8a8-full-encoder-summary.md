# W8A8 Full-Encoder 量化总结

## 目的

本文档记录本次针对 `product_key_expert_jumprelu` SAE checkpoint 的 full-encoder W8A8 后训练量化实验结果。

这次实验的目标是在上一轮 expert-only W8A8 几乎无损的基础上，继续验证一个更完整的问题：如果把 encoder 中的路由器和 expert 局部编码一起量化，整体精度是否仍然可接受。

## 实验配置

- 测试脚本：`quantization/eval_w8a8_full_encoder.py`
- 基座模型：`/root/models/Qwen3-0.6B`
- 数据集：`/root/fineweb-edu/sample/10BT-tokenized-qwen3-2048`
- Checkpoint 根目录：`checkpoints/product_key_expert_jumprelu_qproj/product_key_expert_jumprelu_q_dp2_bs1_ga8_ef1_k32_20260406_221636/best`
- 测试层范围：`layers.[0-13].self_attn.q_proj`
- 阈值文件：`thresholds/Qwen3-0.6B/thresholds_q.json`
- 样本数：`1024`
- Batch size：`1`
- 设备：`cuda:0`
- 结果目录：`quantization/results/20260409_170153`

## 本次量化范围

本次实验量化的是整个 encoder 主路径中的三段关键线性计算。

- 量化部分：
  - `left_router`
  - `right_router`
  - `expert_encoders`
- 保持浮点的部分：
  - `expert_encoder_bias`
  - `log_threshold` / JumpReLU threshold
  - `softmax`、`sigmoid`、`topk`
  - decoder 路径

需要注意，这里依然是量化效果仿真和指标评估，不是实际部署内核上的性能测试。

## 运行命令

```bash
python quantization/eval_w8a8_full_encoder.py \
  --checkpoint-root checkpoints/product_key_expert_jumprelu_qproj/product_key_expert_jumprelu_q_dp2_bs1_ga8_ef1_k32_20260406_221636/best \
  --hookpoints 'layers.[0-13].self_attn.q_proj' \
  --elbow-threshold-path thresholds/Qwen3-0.6B/thresholds_q.json \
  --num-samples 1024 \
  --batch-size 1
```

## 聚合结果

- 测试 hookpoint 数量：`14`
- 平均 `FVU` 增量：`+0.00020304`
- 平均 `exceed_alpha_0.50` 增量：`+0.00017823`
- `FVU` 最差层：`layers.0.self_attn.q_proj`
- `exceed_alpha_0.50` 最差层：`layers.0.self_attn.q_proj`

从整体结果看，full-encoder W8A8 的误差相比 expert-only 明显上升了一些，但绝对值仍然很小，整体上仍处于完全可接受的范围。

## 分层结果

```text
hookpoint                                                 fvu_base   fvu_w8a8      delta exceed_alpha_0.50_base exceed_alpha_0.50_w8a8      delta
-------------------------------------------------------------------------------------------------------------------------------------------------
layers.0.self_attn.q_proj                                 0.041858   0.042848   0.000990           0.042893           0.043818   0.000925
layers.1.self_attn.q_proj                                 0.114717   0.114834   0.000118           0.117915           0.118047   0.000132
layers.2.self_attn.q_proj                                 0.135097   0.135329   0.000232           0.155376           0.155619   0.000242
layers.3.self_attn.q_proj                                 0.188581   0.188721   0.000139           0.205522           0.205695   0.000174
layers.4.self_attn.q_proj                                 0.194873   0.194964   0.000090           0.240904           0.240989   0.000084
layers.5.self_attn.q_proj                                 0.187294   0.187400   0.000106           0.216245           0.216360   0.000115
layers.6.self_attn.q_proj                                 0.240321   0.240473   0.000152           0.296241           0.296381   0.000140
layers.7.self_attn.q_proj                                 0.255928   0.255999   0.000070           0.324274           0.324360   0.000086
layers.8.self_attn.q_proj                                 0.307073   0.307192   0.000118           0.361075           0.361180   0.000105
layers.9.self_attn.q_proj                                 0.299410   0.299487   0.000078           0.362374           0.362448   0.000074
layers.10.self_attn.q_proj                                0.300086   0.300195   0.000109           0.386498           0.386584   0.000086
layers.11.self_attn.q_proj                                0.297008   0.297381   0.000373           0.379995           0.380125   0.000130
layers.12.self_attn.q_proj                                0.338732   0.338806   0.000074           0.393126           0.393187   0.000061
layers.13.self_attn.q_proj                                0.324919   0.325112   0.000192           0.401269           0.401410   0.000142
```

## 结果解读

这 14 个 `q_proj` 层的结果说明，full-encoder W8A8 已经比 expert-only 更能反映真实量化边界：

- 大多数层的 `FVU` 漂移仍然处于 `1e-4` 量级；
- 大多数层的 `exceed_alpha_0.50` 漂移也仍然处于 `1e-4` 量级；
- 最敏感的层是 `layers.0.self_attn.q_proj`，`FVU` 和 `exceed_alpha_0.50` 的增量都接近 `1e-3`；
- `layers.11.self_attn.q_proj` 也表现出比其他中后层更明显的 `FVU` 漂移。

这说明新增误差的主要来源很可能来自 router 量化，而不是 expert matmul 本身。也就是说，`router` 确实比 `expert_encoders` 更敏感，但它目前仍然在可接受范围内。

## 与 expert-only 的对比

和上一轮 expert-only W8A8 相比：

- full-encoder W8A8 的平均 `FVU` 漂移大约扩大到了 `4.7x`
- full-encoder W8A8 的平均 `exceed_alpha_0.50` 漂移大约扩大到了 `4.2x`

这个增幅本身是合理的，因为这次新增量化的部分正是最容易影响离散路由行为的 `left_router/right_router`。

但从绝对值上看，这次误差依然足够小，因此总体结论依旧是正面的。

## 这次结果说明了什么

本次结果支持以下判断：

- `full encoder W8A8` 是可行的；
- `router` 量化会引入可见但不大的额外误差；
- 当前误差水平仍然完全可以接受；
- 如果继续推进更充分量化，下一阶段更应该优先覆盖 decoder，而不是先去动 threshold 或 bias 这些小参数。

但本次结果暂时还不能说明：

- decoder 量化后是否还能维持同样稳定；
- 进一步降到 4 bit 后是否仍然可接受；
- 真实部署内核上的速度收益与仿真结果是否一致。

## 建议的下一步

1. 将 `W_dec` 纳入量化评估，优先验证 decoder 大权重的量化边界。
2. 先做保守版 decoder weight quantization，再考虑 decoder W8A8。
3. 在 router 保持 8 bit 的前提下，逐步探索 `expert_encoders` / `W_dec` 的更低 bit 方案。
4. 在量化方案稳定后，再进入真实 int8 kernel 路线做速度和显存验证。
