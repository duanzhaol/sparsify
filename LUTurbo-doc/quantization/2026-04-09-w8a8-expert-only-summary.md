# W8A8 Expert-Only 量化总结

## 目的

本文档记录本次针对 `product_key_expert_jumprelu` SAE checkpoint 的 W8A8 后训练量化实验结果。

这次实验优先回答一个比较收敛的问题：如果我们只对 `expert_encoders` 的主矩阵乘路径做 W8A8 量化，而将路由、bias、阈值和解码路径继续保留为浮点，那么重建质量指标会不会出现明显退化？

## 实验配置

- 测试脚本：`quantization/eval_w8a8_expert_only.py`
- 基座模型：`/root/models/Qwen3-0.6B`
- 数据集：`/root/fineweb-edu/sample/10BT-tokenized-qwen3-2048`
- Checkpoint 根目录：`checkpoints/product_key_expert_jumprelu_qproj/product_key_expert_jumprelu_q_dp2_bs1_ga8_ef1_k32_20260406_221636/best`
- 测试层范围：`layers.[0-13].self_attn.q_proj`
- 阈值文件：`thresholds/Qwen3-0.6B/thresholds_q.json`
- 样本数：`1024`
- Batch size：`1`
- 设备：`cuda:0`
- 结果目录：`quantization/results/20260409_155713`

## 本次量化范围

本次实验只量化 expert matmul 路径。

- 量化部分：
  - `expert_encoders`
  - 权重采用对称 int8，按行静态 scale
  - 激活采用对称 int8，按 token 动态 scale
- 保持浮点的部分：
  - `left_router`
  - `right_router`
  - `expert_encoder_bias`
  - JumpReLU threshold 相关逻辑
  - decoder 路径

需要注意，这里做的是量化效果仿真和指标评估，不是实际部署内核上的性能测试。

## 运行命令

```bash
python quantization/eval_w8a8_expert_only.py \
  --checkpoint-root checkpoints/product_key_expert_jumprelu_qproj/product_key_expert_jumprelu_q_dp2_bs1_ga8_ef1_k32_20260406_221636/best \
  --hookpoints 'layers.[0-13].self_attn.q_proj' \
  --elbow-threshold-path thresholds/Qwen3-0.6B/thresholds_q.json \
  --num-samples 1024 \
  --batch-size 1
```

## 聚合结果

- 测试 hookpoint 数量：`14`
- 平均 `FVU` 增量：`+0.00004316`
- 平均 `exceed_alpha_0.50` 增量：`+0.00004222`
- `FVU` 最差层：`layers.13.self_attn.q_proj`
- `exceed_alpha_0.50` 最差层：`layers.13.self_attn.q_proj`

从整体结果看，这两个核心指标的退化都非常小。对于当前这组 checkpoint 和当前这套评估方式来说，expert-only W8A8 PTQ 的方向非常有前景。

## 分层结果

```text
hookpoint                                                 fvu_base   fvu_w8a8      delta exceed_alpha_0.50_base exceed_alpha_0.50_w8a8      delta
-------------------------------------------------------------------------------------------------------------------------------------------------
layers.0.self_attn.q_proj                                 0.041858   0.041955   0.000097           0.042893           0.042973   0.000079
layers.1.self_attn.q_proj                                 0.114717   0.114704  -0.000012           0.117915           0.117919   0.000004
layers.2.self_attn.q_proj                                 0.135097   0.135167   0.000069           0.155376           0.155424   0.000048
layers.3.self_attn.q_proj                                 0.188581   0.188619   0.000037           0.205522           0.205592   0.000070
layers.4.self_attn.q_proj                                 0.194873   0.194899   0.000025           0.240904           0.240919   0.000014
layers.5.self_attn.q_proj                                 0.187294   0.187348   0.000054           0.216245           0.216305   0.000059
layers.6.self_attn.q_proj                                 0.240321   0.240351   0.000030           0.296241           0.296274   0.000033
layers.7.self_attn.q_proj                                 0.255928   0.255935   0.000006           0.324274           0.324295   0.000021
layers.8.self_attn.q_proj                                 0.307073   0.307097   0.000023           0.361075           0.361110   0.000035
layers.9.self_attn.q_proj                                 0.299410   0.299422   0.000012           0.362374           0.362398   0.000024
layers.10.self_attn.q_proj                                0.300086   0.300121   0.000035           0.386498           0.386526   0.000028
layers.11.self_attn.q_proj                                0.297008   0.297082   0.000074           0.379995           0.380054   0.000059
layers.12.self_attn.q_proj                                0.338732   0.338767   0.000036           0.393126           0.393154   0.000029
layers.13.self_attn.q_proj                                0.324919   0.325036   0.000117           0.401269           0.401356   0.000087
```

## 结果解读

这 14 个 `q_proj` 层的结果整体非常稳定：

- `FVU` 的漂移基本都在 `1e-5` 到 `1e-4` 量级；
- `exceed_alpha_0.50` 的漂移也基本都在 `1e-5` 到 `1e-4` 量级；
- 没有任何一层出现明显异常抬升；
- `layers.1.self_attn.q_proj` 的 `FVU` 甚至出现了一个非常小的负增量，但这更像量化噪声导致的局部波动，不应被解读为系统性提升。

因此，本次实验的核心结论是：当前这版 expert-only W8A8 PTQ 方案，在不改动训练流程的前提下，已经可以较好地保持 SAE 重建质量。

## 这次结果说明了什么

本次结果支持以下判断：

- 可以继续沿着 expert-only PTQ 方向往下做；
- 可以把评估范围继续扩展到更多 checkpoint 和 `up_proj`；
- 在 PTQ 还没有明显失效之前，没有必要立刻跳到训练时量化。

但本次结果暂时还不能说明：

- 整个 encoder 全量 W8A8 后仍然一样稳定；
- 真实部署场景中的推理延迟一定会改善；
- 仿真结果与真实 int8 kernel 完全一致；
- router 或 threshold 路径也适合一起量化。

## 建议的下一步

1. 对 `layers.[14-27].mlp.up_proj` 做同样的评估。
2. 用更多训练好的 checkpoint 复现一次，确认这个结论是否具有稳定性。
3. 如果趋势依然稳定，再考虑把量化范围从 expert matmul 扩展到更多 encoder 组件。
4. 在指标验证完成后，再进入真实 kernel 路线，评估实际推理速度和显存收益。
