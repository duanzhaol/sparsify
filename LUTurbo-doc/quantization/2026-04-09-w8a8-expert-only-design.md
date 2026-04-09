# W8A8 Expert-Only 量化设计

## 目标

先用最小改动的方式验证一个核心问题：对于 `product_key_expert_jumprelu` SAE，如果只量化 encoder 中最重的 expert matmul 路径，是否能够在基本不损失重建质量的前提下，把后续真实 int8 部署路线跑通。

## 为什么先做这个

直接对整个 SAE encoder 做 W8A8，风险比较高，问题一旦出现也不容易定位。相比之下，先只量化 `expert_encoders` 有几个好处：

- 这是 encoder 里最核心、也最适合量化的矩阵乘部分；
- 可以先把量化误差和结构误差分开看；
- 如果这一部分已经出现明显退化，就没必要过早推进训练时量化；
- 如果这一部分足够稳定，就说明 PTQ 路线有继续扩展的价值。

## 本次设计范围

本次方案只做后训练量化（PTQ）仿真，不改训练流程，不引入真实 int8 kernel。

- 量化对象：
  - `expert_encoders`
- 保持浮点：
  - `left_router`
  - `right_router`
  - `expert_encoder_bias`
  - JumpReLU threshold 逻辑
  - decoder 路径
- 指标：
  - `FVU`
  - `exceed_alpha_0.5`

## 量化方法

- 权重量化：对称 int8，按行静态 scale
- 激活量化：对称 int8，按 token 动态 scale
- 计算过程：仿真 `int8 x int8 -> int32 accumulation -> dequant`

这个选择的目的不是追求最优量化算法，而是先快速验证“8 bit 是否基本可行”。

## 当前结果

在 `layers.[0-13].self_attn.q_proj` 上，使用 `1024` 个样本测试后：

- 平均 `FVU` 增量约为 `+0.00004316`
- 平均 `exceed_alpha_0.50` 增量约为 `+0.00004222`

从当前结果看，expert-only W8A8 PTQ 基本没有带来明显精度损失。

## 当前结论

这说明我们可以优先沿着下面这条路线推进：

1. 先把 expert-only PTQ 在更多层、更多 checkpoint 上验证清楚；
2. 如果趋势持续稳定，再逐步扩大到更多 encoder 路径；
3. 等 PTQ 边界明确之后，再决定是否值得投入训练时量化。

也就是说，当前没有必要立刻跳到 QAT。PTQ 这条路还远没有走到头。

## 下一步

1. 补测 `layers.[14-27].mlp.up_proj`
2. 复测更多训练 checkpoint
3. 评估是否可以扩大量化范围
4. 后续引入真实 int8 kernel 做延迟和显存验证
