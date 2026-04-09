# LUTurbo 方案简述

## 1. 这个方案要解决什么问题

LUTurbo要解决的是：**如何在 CPU 上把 LLM 推理里的大规模向量-矩阵乘法，变成更便宜的查表操作，同时尽量不损失模型精度。**

LLM 推理里最核心的计算之一是：

```text
y = xW
```

其中 `x` 是当前 token 的激活，`W` 是某个线性层的权重。  
在 CPU 上，这类计算往往是 memory-bound：真正慢的不是算术本身，而是反复读大矩阵权重。

所以整个方案的目标很直接：

- 尽量少读原始大权重 `W`
- 用一个更小、更稀疏、更适合查表的表示来近似 `x`
- 把大部分计算变成“查静态表 + 加权求和”
- 只对少量误差大的维度做精确补偿

## 2. 整体方案是什么

核心思路是先把输入激活 `x` 做**基向量分解**，也就是把它表示成少量 basis vector 的线性组合：

```text
x ≈ Σ_{i in S} α_i b_i
```

其中：

- `b_i` 是预先学出来的基向量
- `S` 是当前 token 实际激活的少量基向量索引
- `α_i` 是对应系数

有了这个表示以后，就可以把原始乘法改写成：

```text
y = xW ≈ Σ_{i in S} α_i (b_i W)
```

这里最关键的一步是：`b_i W` 可以离线预计算。

也就是说，部署时不再直接拿大向量 `x` 去乘大矩阵 `W`，而是：

1. 先找出当前 token 用到了哪些 basis vector
2. 查这些 basis 对应的预计算结果 `P_i = b_i W`
3. 按系数 `α_i` 做加权求和
4. 对少数误差大的输出维度再回退到精确计算

这本质上是用**存储换计算**：离线存更多表，在线减少大矩阵访存。

## 3. 方案的完整执行流程

### 3.1 离线阶段

离线阶段做两件事：

1. **训练稀疏自编码器（SAE）**
   - 输入是真实推理过程中采集到的层激活
   - 目标是让 SAE 学会用少量 latent/basis 重构这些激活
2. **预计算查找表**
   - 把学出来的 basis vector 与目标权重矩阵相乘
   - 得到静态表项 `P_i = b_i W`

训练完成后，SAE 的解码器字典就对应 LUTurbo 要用的 basis library。

### 3.2 在线阶段

对一个输入激活 `x`，在线流程是：

1. **选择 basis**
   - 找出当前 token 该激活哪些 basis vector
2. **得到系数**
   - 算出每个 basis 的组合权重
3. **查表并求和**
   - 读取对应的 `P_i`
   - 计算 `Σ α_i P_i`
4. **误差补偿**
   - 对重构误差超过阈值的少数输出维度，直接用原始权重精确算

因此，整个方法不是完全丢掉原始 matmul，而是把它变成：

- 大多数维度：走查表近似路径
- 少数难点维度：走精确补偿路径

## 4. 当前用来做向量分解的模型架构

当前这套向量分解模型，用的不是普通的 dense SAE，而是：

```text
product_key_expert_jumprelu
```

这也是 `scripts/full/qwen3-0.6B/product_key_expert_jumprelu/train.sh` 里实际训练的架构。

它的目标很明确：**不要再像普通 SAE 那样对整套大字典做一次稠密打分，而是先用一个很便宜的路由过程，把 token 分配到很少几个 expert，再只在这些 expert 里做稀疏选择。**

### 4.1 这个架构的结构

它可以理解成“**Product-Key 路由 + Expert 局部编码 + JumpReLU 稀疏门控 + 全局 TopK**”。

具体来说：

1. **全局字典被拆成很多 expert 子库**
   - 当前训练配置里：
     - `NUM_EXPERTS=512`
     - `LATENTS_PER_EXPERT=56`
   - 所以总 latent 容量是：

```text
512 x 56 = 28672
```

也就是说，这不是一个只有 32 个 latent 的小 SAE，而是一个总容量很大、但每次只激活很小一部分的结构化稀疏字典。

2. **先做 Product-Key expert 路由**
   - 模型里有两个小路由器：
     - `left_router`
     - `right_router`
   - 它们分别对输入 `x` 打分，然后把左右两边的分数组合成 expert 分数
   - 代码里 expert 的分数形式是：

```text
expert_logit(i, j) = left_logit(i) + right_logit(j)
```

也就是用一对 key 的组合来表示一个 expert，这就是 product-key 的来源。

3. **只激活少量 expert**
   - 当前训练配置里：

```text
ACTIVE_EXPERTS=2
```

   - 也就是每个 token 只会路由到 512 个 expert 里的 2 个

4. **只在被选中的 expert 内做局部编码**
   - 每个 expert 里有 56 个 latent
   - 所以每个 token 真正需要计算的候选 latent 数量只有：

```text
2 x 56 = 112
```

这比“对 28672 个 latent 全部做一次稠密打分”便宜得多。

5. **每个候选 latent 经过 JumpReLU 门控**
   - 对于选中的 expert `e` 和其中的第 `m` 个 latent，先算线性响应：

```text
a_pre(e, m) = <w_(e,m), x> + b_(e,m)
```

   - 然后做正半轴截断：

```text
a_pos(e, m) = ReLU(a_pre(e, m))
```

   - 再乘上一个可学习阈值的 JumpReLU 门控：

```text
gate(e, m) = sigmoid((a_pos(e, m) - t_(e,m)) / beta)
```

   - 最终激活值为：

```text
a(e, m) = a_pos(e, m) * gate(e, m) * p_e
```

其中：

- `t_(e,m)` 是每个 latent 自己的阈值
- `beta` 是 JumpReLU 的平滑带宽，默认配置里是 `0.1`
- `p_e` 是 expert 路由概率

6. **再做一次全局 TopK**
   - 当前训练配置里：

```text
K=32
```

   - 所以最终只保留这 112 个候选 latent 里最强的 32 个

7. **解码得到重构向量**
   - 用这 32 个激活去线性组合解码器字典，得到重构：

```text
x_hat = Σ z_i d_i
```

这里的 `d_i` 就是最终学到的 basis vector。  
对 LUTurbo 来说，后续真正要导出成查表系统的，正是这一套解码器 basis。

### 4.2 为什么这个架构适合 LUTurbo

因为 LUTurbo 的关键不是“能不能重构”，而是“**在线选择过程能不能足够便宜**”。

普通 dense SAE 的问题是：要先对整套大字典做一次完整打分，选择成本本身就很高。  
`product_key_expert_jumprelu` 通过结构化设计，把这个问题拆成了两层：

- **先用 product-key router 缩小搜索范围**
- **再在小 expert 子库里做 JumpReLU + TopK**

这样做的直接好处是：

- 总字典容量可以很大，保证表达能力
- 每个 token 的在线选择只访问很少一部分 latent
- 更接近 LUTurbo 最终需要的“低开销 basis 选择器”

## 5. 当前这份训练脚本对应的具体配置

`scripts/full/qwen3-0.6B/product_key_expert_jumprelu/train.sh` 里，当前主要配置是：

- 架构：`ARCHITECTURE=product_key_expert_jumprelu`
- 每个 token 最终激活数：`K=32`
- `EXPANSION_FACTOR=1`，但这个 family 在显式设置 expert 后，实际容量主要由 `NUM_EXPERTS x LATENTS_PER_EXPERT` 决定
- expert 数：`NUM_EXPERTS=512`
- 每个 token 激活 expert 数：`ACTIVE_EXPERTS=2`
- 每个 expert 的 latent 数：`LATENTS_PER_EXPERT=56`
- 优化器：`adam`
- 学习率：`8e-4`
- `AUXK_ALPHA=0.03125`
- 训练对象：
  - `layers.[0-13].self_attn.q_proj`
  - `layers.[14-27].self_attn.q_proj`
  - `layers.[0-13].mlp.up_proj`
  - `layers.[14-27].mlp.up_proj`

也就是说，当前不是训练一个统一的全模型 SAE，而是**按算子类型、按层段分别训练**这套结构化 SAE。

## 6. 一句话总结

这套 LUTurbo 方案，本质上是在做一件事：

**先用结构化稀疏 SAE 把 token 激活分解成少量 basis，再把原始 matmul 改写成“少量 basis 的查表加权和 + 少量精确补偿”。**

而当前具体承担“向量分解 / basis 选择”这一步的模型，就是 `product_key_expert_jumprelu`：  
它用 `Product-Key` 做低开销 expert 路由，用 `JumpReLU` 控制 expert 内部稀疏激活，再用全局 `TopK=32` 得到最终可部署的 basis 组合。
