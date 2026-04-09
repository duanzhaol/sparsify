# W8A8 Full-Encoder 量化设计

## 目标

在 expert-only W8A8 结果已经非常稳定的前提下，进一步验证一个更激进但仍然工程上合理的问题：如果将 `product_key_expert_jumprelu` SAE 的整个 encoder 主路径一起做 W8A8 量化，是否仍然能够保持可接受的重建精度。

这里的“full encoder”不是指把所有控制逻辑和小参数都量化，而是优先覆盖 encoder 中最关键、最有价值的几段线性计算。

## 为什么要做 full encoder

之前的 expert-only 方案只量化了 `expert_encoders`，它证明了 expert 局部 matmul 这块几乎可以无损压到 8 bit。

但如果想进一步接近真实部署场景，只量化 expert matmul 还不够，因为 encoder 中还有一部分重要计算没有覆盖：

- `left_router`
- `right_router`

这两部分虽然规模不如 expert 权重大，但它们直接决定 expert 路由，一旦量化误差影响到了路由排序，后续误差就不只是连续数值误差，而可能变成离散路径误差。因此，full encoder W8A8 是一个很关键的中间实验。

## 本次设计范围

本次方案仍然是后训练量化（PTQ）仿真，不改训练流程，也不引入真实 int8 kernel。

- 量化对象：
  - `left_router`
  - `right_router`
  - `expert_encoders`
- 保持浮点：
  - `expert_encoder_bias`
  - `log_threshold` / JumpReLU threshold
  - `softmax`、`sigmoid`、`topk`
  - decoder 路径
- 指标：
  - `FVU`
  - `exceed_alpha_0.5`

## 量化方法

- router 权重：对称 int8，按输出行静态 scale
- router 激活：对称 int8，按 token 动态 scale
- expert 权重：对称 int8，按行静态 scale
- expert 激活：对称 int8，按 token 动态 scale
- 计算过程：仿真 `int8 x int8 -> int32 accumulation -> dequant`

其中 router 和 expert 路径共用同一套量化假设，这样结果更容易横向比较。

## 与 expert-only 的关系

这次 full-encoder W8A8 设计，实际上是在上一版 expert-only 设计基础上的最小增量扩展：

- expert-only：只量化 `expert_encoders`
- full-encoder：量化 `left_router + right_router + expert_encoders`

这样做的好处是：

- 结果可以直接和上一版对照；
- 如果误差明显上升，可以比较清楚地归因到 router；
- 不需要同时引入 decoder 量化、threshold 量化等额外变量。

## 当前实验结果

在 `layers.[0-13].self_attn.q_proj`、`1024` 个样本上的 full-encoder W8A8 结果为：

- 平均 `FVU` 增量约为 `+0.00020304`
- 平均 `exceed_alpha_0.50` 增量约为 `+0.00017823`
- 最差层出现在 `layers.0.self_attn.q_proj`

和之前的 expert-only 相比，这次误差确实有所上升，但整体仍然处在很小的范围内，仍然完全可以接受。

## 当前结论

这次实验说明：

1. `expert_encoders` 几乎是“白送”的 W8A8 量化对象；
2. `router` 量化会带来更明显一些的误差，但目前仍然远未到不可接受的程度；
3. `full encoder W8A8` 已经具备继续推进的价值；
4. 如果后续要进一步做“更充分量化”，下一阶段应该优先考虑 decoder，而不是先去量化 threshold/bias 这些小参数。

## 下一步

1. 将 decoder 的大权重 `W_dec` 纳入量化研究；
2. 先做保守版 decoder weight quantization，再考虑 decoder W8A8；
3. 在保证 router 继续维持 8 bit 的前提下，探索 `expert_encoders` / `W_dec` 的更低 bit 方案；
4. 在量化边界确认后，再转向真实 int8 kernel 的速度和显存验证。
