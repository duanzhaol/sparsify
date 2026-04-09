# W8A8 Decoder 量化总结

## 目的

本文档记录本次针对 `product_key_expert_jumprelu` SAE checkpoint 的 decoder 侧量化实验结果。

这次实验是在上一轮 `full-encoder W8A8` 已经表现良好的基础上，继续回答两个问题：

- 如果把 decoder 大矩阵 `W_dec` 进一步做成 `int8 weight-only`，精度会不会明显恶化；
- 如果更激进一些，把 decoder 侧的稀疏激活 `top_acts` 也一起做成 `W8A8`，会不会再带来明显额外损失。

## 实验配置

- 基座模型：`/root/models/Qwen3-0.6B`
- 数据集：`/root/fineweb-edu/sample/10BT-tokenized-qwen3-2048`
- Checkpoint 根目录：`checkpoints/product_key_expert_jumprelu_qproj/product_key_expert_jumprelu_q_dp2_bs1_ga8_ef1_k32_20260406_221636/best`
- 测试层范围：`layers.[0-13].self_attn.q_proj`
- 阈值文件：`thresholds/Qwen3-0.6B/thresholds_q.json`
- 样本数：`1024`
- Batch size：`1`
- 设备：`cuda:0`

本轮主要对比三种设置：

1. `full-encoder W8A8`：量化 `left_router`、`right_router`、`expert_encoders`
2. `full-encoder W8A8 + decoder W8`：额外量化 `W_dec`
3. `full-encoder W8A8 + decoder W8A8`：额外量化 `W_dec` 和稀疏激活 `top_acts`

其中 `expert_encoder_bias`、`b_dec`、`log_threshold`、JumpReLU threshold、`topk` 等控制逻辑仍保持浮点。

需要注意，这里仍然是量化效果仿真和指标评估，不是实际 CPU int8 kernel 的真实性能测试。

## 运行命令

保守版 decoder `W8`：

```bash
python quantization/eval_w8a8_full_encoder_w8_decoder.py \
  --checkpoint-root checkpoints/product_key_expert_jumprelu_qproj/product_key_expert_jumprelu_q_dp2_bs1_ga8_ef1_k32_20260406_221636/best \
  --hookpoints 'layers.[0-13].self_attn.q_proj' \
  --elbow-threshold-path thresholds/Qwen3-0.6B/thresholds_q.json \
  --num-samples 1024 \
  --batch-size 1
```

激进版 decoder `W8A8`：

```bash
python quantization/eval_w8a8_full_encoder_w8a8_decoder.py \
  --checkpoint-root checkpoints/product_key_expert_jumprelu_qproj/product_key_expert_jumprelu_q_dp2_bs1_ga8_ef1_k32_20260406_221636/best \
  --hookpoints 'layers.[0-13].self_attn.q_proj' \
  --elbow-threshold-path thresholds/Qwen3-0.6B/thresholds_q.json \
  --num-samples 1024 \
  --batch-size 1
```

结果目录：

- `full-encoder W8A8`：`quantization/results/20260409_170153`
- `full-encoder W8A8 + decoder W8`：`quantization/results/20260409_193455`
- `full-encoder W8A8 + decoder W8A8`：`quantization/results/20260409_193815`

## 聚合结果

| 配置 | 平均 FVU 增量 | 平均 exceed_alpha_0.50 增量 | 最差层 |
|------|---------------|-----------------------------|--------|
| full-encoder W8A8 | `+0.00020304` | `+0.00017823` | `layers.0.self_attn.q_proj` |
| full-encoder W8A8 + decoder W8 | `+0.00037201` | `+0.00036838` | `layers.0.self_attn.q_proj` |
| full-encoder W8A8 + decoder W8A8 | `+0.00039353` | `+0.00038930` | `layers.0.self_attn.q_proj` |

从聚合结果可以直接看出：

- 把 `W_dec` 纳入 `int8 weight-only` 后，平均 `FVU` 增量从 `+2.03e-4` 上升到 `+3.72e-4`；
- 继续把 decoder 侧稀疏激活也做成 `W8A8` 后，平均 `FVU` 只额外增加了约 `+2.15e-5`；
- `exceed_alpha_0.50` 的变化趋势和 `FVU` 非常一致，decoder 激活量化带来的新增代价同样很小。

## 分层结果

### full-encoder W8A8 + decoder W8

```text
hookpoint                                                 fvu_base   fvu_w8a8      delta exceed_alpha_0.50_base exceed_alpha_0.50_w8a8      delta
-------------------------------------------------------------------------------------------------------------------------------------------------
layers.0.self_attn.q_proj                                 0.041858   0.042968   0.001110           0.042893           0.043850   0.000956
layers.1.self_attn.q_proj                                 0.114717   0.114954   0.000237           0.117915           0.118165   0.000251
layers.2.self_attn.q_proj                                 0.135097   0.135507   0.000409           0.155376           0.155867   0.000490
layers.3.self_attn.q_proj                                 0.188581   0.188869   0.000288           0.205522           0.205912   0.000390
layers.4.self_attn.q_proj                                 0.194873   0.195109   0.000236           0.240904           0.241215   0.000311
layers.5.self_attn.q_proj                                 0.187294   0.187555   0.000261           0.216245           0.216591   0.000346
layers.6.self_attn.q_proj                                 0.240321   0.240646   0.000325           0.296241           0.296624   0.000384
layers.7.self_attn.q_proj                                 0.255928   0.256139   0.000211           0.324274           0.324529   0.000255
layers.8.self_attn.q_proj                                 0.307073   0.307336   0.000263           0.361075           0.361327   0.000252
layers.9.self_attn.q_proj                                 0.299410   0.299648   0.000239           0.362374           0.362614   0.000240
layers.10.self_attn.q_proj                                0.300086   0.300356   0.000270           0.386498           0.386742   0.000244
layers.11.self_attn.q_proj                                0.297008   0.297623   0.000615           0.379995           0.380393   0.000398
layers.12.self_attn.q_proj                                0.338732   0.338997   0.000266           0.393126           0.393357   0.000231
layers.13.self_attn.q_proj                                0.324919   0.325397   0.000478           0.401269           0.401678   0.000409
```

### full-encoder W8A8 + decoder W8A8

```text
hookpoint                                                 fvu_base   fvu_w8a8      delta exceed_alpha_0.50_base exceed_alpha_0.50_w8a8      delta
-------------------------------------------------------------------------------------------------------------------------------------------------
layers.0.self_attn.q_proj                                 0.041858   0.042999   0.001141           0.042893           0.043860   0.000967
layers.1.self_attn.q_proj                                 0.114717   0.114984   0.000268           0.117915           0.118204   0.000289
layers.2.self_attn.q_proj                                 0.135097   0.135535   0.000437           0.155376           0.155906   0.000529
layers.3.self_attn.q_proj                                 0.188581   0.188894   0.000313           0.205522           0.205945   0.000423
layers.4.self_attn.q_proj                                 0.194873   0.195134   0.000260           0.240904           0.241251   0.000346
layers.5.self_attn.q_proj                                 0.187294   0.187578   0.000284           0.216245           0.216622   0.000376
layers.6.self_attn.q_proj                                 0.240321   0.240668   0.000348           0.296241           0.296645   0.000405
layers.7.self_attn.q_proj                                 0.255928   0.256159   0.000231           0.324274           0.324549   0.000275
layers.8.self_attn.q_proj                                 0.307073   0.307354   0.000281           0.361075           0.361340   0.000264
layers.9.self_attn.q_proj                                 0.299410   0.299665   0.000256           0.362374           0.362627   0.000253
layers.10.self_attn.q_proj                                0.300086   0.300371   0.000285           0.386498           0.386752   0.000254
layers.11.self_attn.q_proj                                0.297008   0.297641   0.000633           0.379995           0.380404   0.000409
layers.12.self_attn.q_proj                                0.338732   0.339013   0.000281           0.393126           0.393366   0.000241
layers.13.self_attn.q_proj                                0.324919   0.325412   0.000493           0.401269           0.401687   0.000419
```

## 结果解读

这轮结果最重要的结论有三个。

### 1. `W_dec` 本身非常适合继续量化

在 full-encoder 已经量化的基础上，把 decoder 大矩阵 `W_dec` 做成 `int8` 后，确实会进一步带来误差上升，但上升幅度仍然非常温和：

- `FVU` 平均只增加到 `3.7e-4` 量级；
- `exceed_alpha_0.50` 平均只增加到 `3.7e-4` 量级；
- 最差层仍然集中在 `layers.0.self_attn.q_proj`，整体模式和 full-encoder 阶段一致，没有出现新的异常层。

这说明 decoder 大权重矩阵并不是一个高风险量化点。

### 2. decoder 激活 `top_acts` 的 8bit 量化额外代价极小

从 `decoder W8` 到 `decoder W8A8` 的变化非常小：

- 平均 `FVU` 只额外增加约 `+2.15e-5`；
- 平均 `exceed_alpha_0.50` 只额外增加约 `+2.09e-5`。

这意味着在当前 `product_key_expert_jumprelu` SAE 上，decoder 稀疏激活本身也表现出了很强的 8bit 友好性。换句话说，真正的主要误差来源仍然不是 decoder 激活量化，而是前面 encoder/router 量化所带来的那部分漂移。

### 3. 当前 SAE 主路径已经接近“完整 8bit”研究边界

到这一轮为止，我们已经验证了以下主路径都可以进入 8bit 量化研究范围：

- `left_router`
- `right_router`
- `expert_encoders`
- `W_dec`
- `top_acts`

暂时还没有纳入 8bit 的，主要是：

- `expert_encoder_bias`
- `b_dec`
- `log_threshold`
- JumpReLU threshold / control-flow
- 在线补偿部分

因此当前最自然的下一步，不是再去碰一些很小的 bias/threshold，而是去分析在线补偿是否也值得量化。

## 当前结论

本次结果支持以下判断：

- `full-encoder + decoder W8` 是可行的；
- `full-encoder + decoder W8A8` 也是可行的；
- decoder 稀疏激活量化带来的额外代价极小；
- 对当前 SAE 主干而言，8bit 已经基本不是精度瓶颈。

从研究推进顺序上看，这一轮结果已经足够支持我们把注意力转向“在线补偿是否也要量化”这个问题。

## 建议的下一步

1. 评估在线补偿路径在主干已量化后的时间占比，确认它是否成为新的 CPU 推理瓶颈。
2. 设计 online compensation 的 `W8A8` 仿真评估，重点看 `FVU`、`exceed_alpha_0.50` 和补偿收益保留率。
3. 如果 online compensation 的 int8 仿真也很稳，再考虑是否有必要引入训练时量化或 QAT。
