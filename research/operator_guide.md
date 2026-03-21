# SAE Auto Research Idea Bank

> 目标：给自动研究 Agent 提供一组可发散的思路，用于探索 **如何在更小的 TopK 下训练出更好的 SAE**，并服务于 **CPU 上基于查表/LUT 的推理加速**。
>
> 这不是操作手册，也不是固定 recipe，而是一个 **idea bank / search space**。  
> Agent 可以从中抽取方向，组合成新的实验假设。

---

## Runtime Priorities

在参考下面的 idea bank 之前，先遵守这些当前阶段的运行时优先级。

1. 第一优先级不是扩展搜索空间，而是消除当前反复出现的 blocker：
   - `sanity_failed`
   - `first_step_timeout`
   - 变量隔离失效导致的混杂实验

2. 如果一轮失败在 `sanity`，优先修复对应架构本身的前向/反向兼容性问题。
   - 不要先改共享 trainer / runtime 路径，除非失败证据明确指向 trainer。
   - 如果 sanity 根本没过，不要把问题归因为训练阶段的 first-step bottleneck。

3. 一轮只允许一个主变量。
   - 不要同时修改 `architecture + lr`
   - 不要同时修改 `code + architecture + lr`
   - 只有在运行时明确允许的例外情况下，才允许耦合修改

4. 在系统稳定前，优先顺序应为：
   - 先拿到能稳定运行并能产生有效信号的实验
   - 再做架构比较
   - 最后才扩大到更复杂的训练目标或部署对齐目标

5. 当前不要轻易改动固定执行层的语义。
   - 不要改变 evaluation semantics
   - 不要改变 launcher 行为
   - 不要引入会让历史结果不可比的大改动

6. 如果最近的失败模式表明问题出在系统观测或运行时策略，不要把它误判成架构质量结论。

7. 下面的内容首先是 idea bank，不是强制 recipe。
   - 在当前 blocker 未解决前，应优先选择最小、最可解释、最容易归因的动作。
   - 只有当基础运行链路稳定后，才扩大到更发散的搜索。

---

## 0. 问题重述

当前问题不应只被表述为：

- “如何把 SAE 的 FVU 降低”
- “如何把 TopK 从 128 压到 32/64”

更准确的表述应该是：

- **如何在给定在线成本预算下，学到更适合查表推理的表示**
- **如何让少量激活原子覆盖对下游输出真正重要的部分**
- **如何让稀疏性变成可实现的加速，而不仅仅是数学上的稀疏**

因此，Agent 不应只搜索“更大的 SAE”或“更多训练步数”，而应搜索：

1. 输入空间是否适合做稀疏表示  
2. 稀疏机制是否容易优化  
3. 训练目标是否对齐最终部署目标  
4. 字典/码本结构是否适合 CPU 查表  
5. 稀疏结构是否具有实现友好性

---

## 1. 优先级最高的几个总方向

### 1.1 改输入空间，而不是只改 SAE 宽度

假设当前激活空间并不适合直接做 TopK 稀疏分解。  
那么更大的字典不一定解决问题，可能只是给了更多冗余原子。

可探索方向：

- 去均值、按维标准化
- PCA 旋转
- 部分 whitening
- 完整 whitening
- 仅做旋转，不做缩放
- 先降到一个较低维的主子空间，再做 SAE
- 不同层、不同模块、不同 token 类型分别估计变换

关注点：

- 坐标系是否让“少量原子组合”更自然
- 是否降低了 K 对重建质量的需求
- 是否减少了原子冗余

> 启发来源：whitening/rotation 常能改善表示空间；近期 SAE 工作也明确研究了 whitening 对 SAE 学习的影响。[[1]] [[4]]

---

### 1.2 不要只优化激活重建，要优化“下游输出保持”

对于 LUT 替代 matmul 的场景，真正重要的未必是：
\[
x \approx \hat{x}
\]
而更可能是：
\[
W x \approx W \hat{x}
\]

可探索方向：

- 激活重建 loss
- 下游线性层输出对齐 loss
- 两者的多目标组合
- 更关注 hardest case 的多目标聚合
- 分层设置不同的输出对齐权重
- 只对某些关键层引入 output-aware loss

值得让 Agent 思考的问题：

- 是否存在某些方向对 FVU 很重要，但对下游输出不重要
- 是否可以牺牲部分 FVU，换取更低 K 和更好的部署收益
- 是否可以按层、按模块决定“更重建激活”还是“更重建输出”

> 启发来源：OpenAI 的 SAE scaling/evaluation 强调除了 reconstruction，也应看更贴近应用的指标。[[1]]

---

### 1.3 不要只搜索固定 TopK，要搜索“稀疏机制本身”

当前固定逐样本 TopK 可能过于僵硬。  
Agent 应该把“如何定义稀疏”也视为搜索空间的一部分。

可探索方向：

- 固定 TopK
- BatchTopK
- ReLU + L1，再推理时截断为 TopK
- 先软稀疏，后硬稀疏
- JumpReLU
- straight-through 的 hard mask
- K curriculum
- 动态 K，但控制平均 K
- 分组 TopK / block-wise TopK

Agent 应考虑：

- 难样本是否需要更多 latent，简单样本是否可以更少
- 平均 K 固定是否优于每样本 K 固定
- 更平滑的训练是否能换来更好的最终 hard sparse 表示

> 启发来源：BatchTopK 放宽了逐样本固定 top-k 约束，在平均稀疏度不变时改善重建；JumpReLU 在给定稀疏水平下提升重建保真度。[[2]] [[3]]

---

## 2. 可发散的搜索簇

### 2.1 输入表示 / 预处理簇

可探索的假设：

- 原始激活不适合直接稀疏编码
- 激活存在强相关、重尾、尺度不均
- 激活空间中有“更好”的坐标系

可尝试的变化：

- mean centering
- per-channel normalization
- RMS normalization
- PCA rotation
- PCA whitening
- 部分 whitening（介于 rotation 与 whitening 之间）
- 先做低秩投影，再做 SAE
- 先做 learned linear transform，再做 SAE
- 按层/模块/token bucket 单独估计变换

研究问题：

- 哪些预处理最能降低所需 K
- 预处理是否改变了原子使用分布
- 不同层是否偏好不同的预处理强度
- 更“白”的空间是否更利于解释性，还是更利于部署 fidelity

---

### 2.2 目标函数 / 多目标优化簇

可探索的假设：

- 当前损失函数和目标错位
- recon / sparsity / output fidelity / utilization 之间存在冲突
- 简单线性加权可能导致某个目标淹没其他目标

可尝试的变化：

- recon loss
- output-aware loss
- 稀疏度 loss
- 平均 K 约束
- feature utilization / balance loss
- worst-case 更敏感的聚合方式
- 非线性 loss 聚合
- 梯度归一化思想
- 分阶段切换损失重心
- 前期更重 recon，后期更重 sparsity / output fidelity

研究问题：

- 哪种 loss 组合最容易在 K=32/64 下保持质量
- 是否存在某些层更适合 output-aware，而某些层更适合 pure recon
- 是否需要专门压制最差样本而不是优化平均误差

> 启发来源：科学空间关于多目标损失、广义平均、梯度归一化的讨论，适合迁移到 SAE 的多目标训练上。[[6]]

---

### 2.3 稀疏路由 / 原子分配簇

可探索的假设：

- 问题不在打分函数，而在分配方式
- 热门原子被过度使用，冷门原子难以启动
- 字典有效容量远小于名义容量

可尝试的变化：

- usage-aware TopK
- 给高频原子施加选择偏置
- 对低频原子给予探索奖励
- batch-level balancing
- 动态 bias 修正
- loss-free balancing 风格的选择修正
- 避免只靠 aux loss 惩罚不均衡
- 周期性 exploration mode / exploitation mode 切换

研究问题：

- 是否存在“少数 latent 占据大多数流量”的问题
- 使用分布更均匀时，FVU / output fidelity 是否改善
- balanced routing 是否能减少 dead atoms 和 duplicate atoms

> 启发来源：科学空间关于 Loss-Free MoE 的讨论强调，可通过修改分配方式而非一味添加辅助损失来改善路由。[[7]]

---

### 2.4 字典质量 / 原子健康度簇

可探索的假设：

- 字典不是太小，而是学坏了
- 原子之间高度相似
- 大量原子长期不被使用
- 宽度翻倍没收益是因为有效原子数没有增加

可尝试的变化：

- decoder 列归一化
- mutual coherence penalty
- pairwise decorrelation
- dead atom reset
- 用高残差样本重初始化
- 周期性替换低利用率原子
- 使用频率均衡
- 避免 encoder/decoder 尺度漂移

研究问题：

- 字典有效容量是多少，而不是名义容量是多少
- 原子是否集中在少数方向附近
- 是否需要把“字典健康度”作为一等指标纳入 agent 的奖励函数

---

### 2.5 宽度 / 稀疏度 / 深度的结构搜索簇

可探索的假设：

- “更宽”未必最重要
- 稀疏编码深度或层次结构可能比单层超宽更有效

可尝试的变化：

- 8192 / 16384 / 更宽字典
- 小字典 + 更好的预处理
- 两级字典
- coarse dictionary + residual dictionary
- block dictionary
- 分组字典
- mixture of dictionaries
- hierarchical sparse coding
- 分阶段 residual reconstruction

研究问题：

- 单层大字典 vs 两层残差字典，哪个更适合低 K
- 分块或层次结构是否更利于 CPU 实现
- 是否存在“先粗后细”的结构能减少平均 K

---

### 2.6 低秩 + 稀疏残差簇

可探索的假设：

- 激活中有很大一部分能量是“公共主干”
- 没必要让稀疏原子去同时承担主干 + 个性化残差
- SAE 应该只负责 residual，而不是一切

可尝试的变化：

- PCA/learned low-rank trunk + sparse residual
- 低秩部分固定，稀疏部分训练
- 低秩部分和 SAE 联合训练
- 先压缩到低维子空间，再做 sparse code
- coarse linear approximation + sparse correction

研究问题：

- 低秩主干是否能显著减少所需 K
- residual 是否比原始激活更容易稀疏表示
- CPU 上低秩主干 + LUT residual 是否比纯 LUT 更划算

> 启发来源：科学空间的低秩近似/骨架思路，以及从 ID/CUR 视角寻找“主干 + 补偿”的结构。[[5]]

---

### 2.7 离散码本 / 量化范式簇

可探索的假设：

- 当前任务本质更像“码本检索 + 累加”
- SAE 不是唯一形式，甚至可能不是最自然形式
- 离散表示可能更贴合 LUT 推理

可尝试的变化：

- VQ 风格码本
- residual VQ
- additive quantization
- product quantization
- FSQ 风格更简单离散化
- coarse VQ + sparse residual
- 多级码本近似
- block-wise codebook

研究问题：

- 离散码本是否比 sparse linear combination 更适合 CPU
- 同等精度下，VQ/AQ/PQ 是否拥有更规则的访存模式
- 是否可以把 SAE 看成“残差补偿器”，而不是主表示

> 启发来源：科学空间关于 VQ-VAE、FSQ、DiVeQ 的一系列讨论，都强调“训练与推理一致的硬离散表示”这一思路。[[8]] [[9]] [[10]]

---

### 2.8 分桶 / mixture / 条件化表示簇

可探索的假设：

- 一个统一字典很难覆盖所有 token / 上下文 / 位置 / phase
- 样本难度分布高度不均

可尝试的变化：

- 按 activation norm 分桶
- 按 token type 分桶
- 按 position bucket 分桶
- 按 layer phase 分桶
- prompt vs decode 分桶
- mixture of dictionaries
- routing 到不同子字典
- 条件化预处理 + 条件化 SAE

研究问题：

- 是否存在若干明显不同的数据子分布
- 局部字典是否比全局字典需要更小 K
- 分桶带来的路由开销是否值得

---

### 2.9 训练流程 / curriculum 簇

可探索的假设：

- 好字典需要先学会覆盖，再学会稀疏
- 一开始就强硬约束 K 可能导致坏局部最优
- 阶段化训练比一次成型更稳

可尝试的变化：

- soft sparse pretrain -> hard sparse finetune
- 大 K 预训练 -> 小 K 微调
- 逐步 anneal K
- 先重 recon 后重 deployment
- exploration-heavy 初期 -> exploitation-heavy 后期
- 周期性重置低效原子
- 周期性改变 loss 权重
- progressive dictionary growing / pruning

研究问题：

- 是否存在“先学好字典，再学好稀疏”这一更优路径
- 哪种 curriculum 最容易在低 K 下保持 fidelity
- 宽度、稀疏度、预处理是否需要分阶段共同调整

---

### 2.10 部署友好性 / CPU 约束簇

可探索的假设：

- 更低的数学误差不一定带来更快的真实系统速度
- 访存模式、cache 友好性、规则性和 K 同等重要

可尝试的变化：

- block-structured atoms
- 连续内存布局友好的字典
- 限制 atom 来自少数 block
- 限制每个样本访问的 block 数
- 用规则聚合替代完全自由聚合
- 固定模式的小 K，而不是波动很大的动态 K
- 把 cache miss / memory traffic proxy 纳入目标函数

研究问题：

- 什么样的稀疏结构最容易被 CPU 利用
- K=32 的乱访存是否可能比 K=64 的规则访存更差
- Agent 是否应直接优化“系统 proxy”，而不仅仅是 FVU

> 启发来源：科学空间对稀疏 attention 的讨论反复提醒：理论稀疏不自动等于提速，关键在实现是否真正利用了结构。[[11]]

---

## 3. Agent 可以主动维护的一组中间诊断指标

Agent 不应只看：

- FVU
- 平均 K

还应主动记录并分析：

### 3.1 字典健康度
- 每个 latent 的使用频率
- dead latent 比例
- 原子两两相似度
- 原子 norm 分布
- encoder score 分布

### 3.2 稀疏利用率
- 平均 K
- 中位数 K
- 不同样本难度下的 K 分布
- top-1 / top-8 / top-32 能量覆盖
- 高频原子是否垄断路由

### 3.3 任务相关质量
- activation recon
- output-aware error
- 部分层替换后的局部精度
- 全模型替换后的最终任务指标
- 不同 token bucket 的误差分布

### 3.4 系统 proxy
- 查表次数
- 聚合次数
- 访问 block 数
- 估算的 memory traffic
- 粗略 cache 友好性分数
- 在线编码成本

---

## 4. Agent 应该如何发散，而不是只做局部调参

### 4.1 不只做单变量 ablation
除了“只改一个超参”的局部搜索，也应允许组合搜索：

- 预处理 + 稀疏机制
- 稀疏机制 + 多目标 loss
- 低秩主干 + sparse residual
- usage-aware routing + dead-atom reset
- coarse VQ + residual SAE

很多有效方案未必来自某一个单独改动，而是来自两个思想的组合。

---

### 4.2 应允许“换问题表述”
如果若干轮后持续出现：

- K 降不下去
- 或 K 降下去但 FVU/输出误差崩
- 或训练很不稳定
- 或字典利用率长期极差

Agent 应考虑不是继续调参，而是切换问题表述，例如：

- 从 pure SAE 改成 low-rank + sparse residual
- 从 fixed TopK 改成 BatchTopK / JumpReLU
- 从 sparse coding 改成 residual VQ / additive quantization
- 从 global dictionary 改成 mixture / bucketed dictionaries

---

### 4.3 奖励函数不应只奖励最低 FVU
可考虑使用多目标奖励，例如：

- reconstruction quality
- output fidelity
- average K
- deployment friendliness
- dictionary utilization
- stability / reproducibility

并允许不同阶段有不同的奖励重点。

---

## 5. 可供 Agent 组合的“高层模板”

### 模板 A：空间先行
- 先找更好的输入空间
- 再在新空间里训练 SAE
- 再看 K 是否自然下降

### 模板 B：目标对齐
- 不追求最优激活重建
- 直接对齐下游输出
- 让 FVU 退居次要指标

### 模板 C：稀疏机制替换
- 认为问题主要来自 TopK 本身
- 搜索 BatchTopK / JumpReLU / soft-to-hard

### 模板 D：结构重写
- 从单层稀疏字典改成 coarse-to-fine
- 或 low-rank + sparse residual
- 或 VQ + residual correction

### 模板 E：路由与利用率优先
- 假设宽度够了，但利用率差
- 优先优化使用均衡、原子健康度和分配方式

### 模板 F：部署优先
- 不先追最低误差
- 先追规则访问、较小 block 数、可向量化聚合
- 在此基础上再修复误差

---

## 6. 一些应被明确允许的“大胆想法”

Agent 不应该被限制在“经典 SAE 调参”里。  
以下思路都应被允许探索：

- 用 PCA 主子空间替代部分 hidden 维度
- 用分层字典而不是单层字典
- 把 SAE 拆成 coarse branch + refinement branch
- 只对最关键的层使用更复杂 SAE，其余层用更简单近似
- 引入 routing bias，而非只改 loss
- 对不同 token bucket 使用不同 K
- 对 residual 而非原始激活做 SAE
- 让一部分激活走低秩支路，一部分走 sparse 支路
- 改为 block-sparse / group-sparse
- 改为 codebook 检索 + 小 residual 修正
- 用“最差样本误差”而不只是均值误差做筛选
- 明确把 CPU proxy 纳入优化目标

---

## 7. 建议 Agent 避免的几种局限

### 7.1 避免只围绕这几个旋钮打转
- 学习率
- batch size
- 训练步数
- 宽度翻倍
- 稀疏系数微调

这些当然可以调，但不应成为主要搜索方向。

### 7.2 避免只看单一平均指标
平均 FVU、平均 K 很容易掩盖问题。  
要看分布、分桶、最坏样本和部署 proxy。

### 7.3 避免默认“重建最好 = 部署最好”
对 LUT 推理场景，这个等价关系通常不成立。

### 7.4 避免默认“稀疏越强 = 越快”
若访存极不规则，可能更稀疏反而更慢。

---

## 8. 最后给 Agent 的高层提醒

1. **容量不足** 只是许多可能性中的一种。  
2. 如果“加宽无收益”，优先怀疑：  
   - 坐标系不好  
   - 路由不合理  
   - 字典冗余  
   - 目标错位  
3. 对这个任务，**更适合部署的表示** 可能比 **更标准的 SAE 表示** 更重要。  
4. 如果纯 TopK SAE 长期无法达到目标，不要犹豫切换到：
   - BatchTopK
   - JumpReLU
   - low-rank + sparse residual
   - residual VQ / additive quantization
5. 真正值得追求的是：
   - 小而规则的在线成本
   - 保住下游输出
   - 有健康利用率的字典/码本
   - 真实 CPU 上可实现的收益

---

## References / Inspiration

[1] OpenAI, *Scaling and Evaluating Sparse Autoencoders*.  
[2] Bussmann et al., *BatchTopK Sparse Autoencoders*.  
[3] Rajamanoharan et al., *Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders*.  
[4] Saraswatula & Klindt, *Data Whitening Improves Sparse Autoencoder Learning*.  
[5] 科学空间：低秩近似 / ID / CR / 骨架分解相关系列。  
[6] 科学空间：多目标损失、广义平均、梯度归一化相关讨论。  
[7] 科学空间：Loss-Free MoE / 换个思路来分配。  
[8] 科学空间：VQ-VAE 相关文章。  
[9] 科学空间：FSQ 相关文章。  
[10] 科学空间：DiVeQ 相关文章。  
[11] 科学空间：稀疏 Attention 与“理论稀疏不等于真实加速”的讨论。
