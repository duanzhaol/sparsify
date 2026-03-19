> Archived document: this file is kept for historical reference and may not match the current codebase.
> For current guidance, start from `docs/README.md` and the active docs under `docs/`.

# FusedDecoder：Ascend NPU 自定义解码器算子设计文档

## 1. 问题背景

### 1.1 SAE 解码器的工作原理

稀疏自编码器（SAE）的解码过程是一个**稀疏加权求和**：给定 top-k 个被激活的 latent 索引和对应的激活值，从权重矩阵中取出对应行并加权求和，重构出输出向量。

```
输入：
  top_indices: (N, k)     — 每个样本激活的 k 个 latent 索引
  top_acts:    (N, k)     — 对应的激活值
  W_dec:       (M, d_in)  — 解码器权重矩阵（M 个 latent，d_in 维输出）

输出：
  output:      (N, d_in)  — 重构向量

数学表达：
  output[n] = Σ_{i=0}^{k-1} top_acts[n,i] * W_dec[top_indices[n,i]]
```

这个操作在 PyTorch 中可以用 `F.embedding_bag(mode="sum", per_sample_weights=...)` 高效实现。

### 1.2 NPU 上的问题

在 Ascend NPU 上：

| 算子 | 前向 | 反向 |
|------|------|------|
| `F.embedding_bag` | NPU 原生支持 | **回退到 CPU** |

`aten::_embedding_bag_backward` 未被 Ascend CANN 算子库原生实现，因此在反向传播时数据会从 NPU 拷贝到 CPU 计算梯度，再拷回 NPU。这是训练流程中**唯一一个 CPU 回退**，会导致：

1. **NPU ↔ CPU 数据搬运开销**：每次反向传播都需要跨设备拷贝张量
2. **CPU 计算瓶颈**：CPU 上的 embedding_bag 反向远慢于加速器原生计算
3. **流水线阻塞**：同步搬运打断了 NPU 的异步执行流

训练时控制台会输出如下警告：
```
[W CpuFallbackWarn.cpp] Warning: aten::_embedding_bag_backward is falling back to CPU.
```

### 1.3 已有的参考：FusedEncoder

编码器侧（`fused_encoder.py`）已经用同样的思路解决了类似问题——通过自定义 `torch.autograd.Function`，前向用 PyTorch 原生算子，反向用 `index_add_` 等 NPU 原生算子替换不支持的反向操作。FusedDecoder 遵循完全相同的设计模式。

## 2. 设计思路

### 2.1 核心思想

**保留前向不变，只重写反向传播**。

通过 `torch.autograd.Function` 自定义前向和反向：
- 前向：直接调用 `F.embedding_bag`（NPU 原生支持）
- 反向：用 `index_add_` 和基础张量运算替代 `_embedding_bag_backward`（全部 NPU 原生支持）

### 2.2 反向传播推导

前向操作：`output[n] = Σ_i top_acts[n,i] * W_T[top_indices[n,i]]`

需要计算两个梯度：

**梯度 1：∂L/∂top_acts**

```
∂L/∂top_acts[n,i] = ∂L/∂output[n] · W_T[top_indices[n,i]]
                   = dot(grad_output[n], W_T[top_indices[n,i]])
```

实现：对每个 k 位置，用高级索引取出对应权重行，与 grad_output 逐元素相乘后求和。

**梯度 2：∂L/∂W_T**

```
∂L/∂W_T[m] = Σ_{(n,i): top_indices[n,i]=m} top_acts[n,i] * grad_output[n]
```

这是一个 scatter-add 操作：将 `top_acts[n,i] * grad_output[n]` 按 `top_indices[n,i]` 累加到对应权重行。

实现：使用 `index_add_` —— NPU 原生支持的就地索引累加算子。

### 2.3 参数传递约定

调用链中的转置约定需要特别注意：

```
SparseCoder 存储:  self.W_dec       形状 [M, d_in]
调用时传入:        self.W_dec.mT    形状 [d_in, M]     ← decoder_impl 接收的 W_dec 参数
eager_decode 内部: W_dec.mT         形状 [M, d_in]     ← 传给 F.embedding_bag
```

FusedDecoder 遵循同样的约定：
```python
def fused_decode(top_indices, top_acts, W_dec):
    # W_dec 来自调用点，形状 [d_in, M]
    # .mT 转回 [M, d_in] 给 F.embedding_bag
    return FusedDecoder.apply(top_indices, top_acts, W_dec.mT)
```

关键：`.mT` 放在 `FusedDecoder.apply()` 外面，这样 autograd 会自动处理转置操作的梯度传播，我们在自定义 Function 内部不需要关心转置的梯度。

## 3. 实现

### 3.1 代码结构

```
sparsify/
├── fused_encoder.py      # 编码器：已有的 NPU 优化
├── fused_decoder.py      # 解码器：新实现的 NPU 优化（本文档）
└── utils.py              # decoder_impl 分发：NPU → fused_decode
```

### 3.2 核心实现

```python
class FusedDecoder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, top_indices, top_acts, W_T):
        # W_T: (M, d_in)  — 已经是转置后的形状
        out = F.embedding_bag(
            top_indices, W_T, per_sample_weights=top_acts, mode="sum"
        )
        ctx.save_for_backward(top_indices, top_acts, W_T)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        top_indices, top_acts, W_T = ctx.saved_tensors
        k = top_indices.shape[-1]

        # ∂L/∂top_acts: 逐 k 位置计算内积
        if ctx.needs_input_grad[1]:
            grad_acts = torch.empty_like(top_acts)
            for i in range(k):
                selected = W_T[top_indices[:, i]]          # (N, d_in)
                grad_acts[:, i] = (grad_output * selected).sum(-1)

        # ∂L/∂W_T: 逐 k 位置用 index_add_ 累加
        if ctx.needs_input_grad[2]:
            grad_W_T = torch.zeros_like(W_T)
            for i in range(k):
                contrib = top_acts[:, i].unsqueeze(-1) * grad_output  # (N, d_in)
                grad_W_T.index_add_(0, top_indices[:, i], contrib.type_as(W_T))

        return None, grad_acts, grad_W_T
```

### 3.3 逐 k 循环的设计选择

反向传播中使用 `for i in range(k)` 循环而非一次性展开全部 k 个位置，这与 FusedEncoder 采用相同的策略：

- **内存节省**：避免分配 `(N, k, d_in)` 的中间张量。当 N=4096, k=128, d_in=2048 时，展开方式需要 ~4GB 中间内存，而循环方式每次只需 ~32MB
- **NPU 兼容性**：简单的逐步操作对 NPU 算子调度更友好
- **k 通常较小**：SAE 中 k 典型值为 32~128，循环开销可忽略

### 3.4 decoder_impl 分发逻辑

在 `utils.py` 中根据设备类型自动选择解码器实现：

```python
if get_device_type() == "npu":
    from .fused_decoder import fused_decode
    decoder_impl = fused_decode          # NPU: 自定义反向，无 CPU 回退
else:
    # CUDA: 优先 Triton kernel，回退到 eager
    decoder_impl = triton_decode or eager_decode
```

## 4. 与 FusedEncoder 的对比

| 特性 | FusedEncoder | FusedDecoder |
|------|-------------|-------------|
| 文件 | `fused_encoder.py` | `fused_decoder.py` |
| 前向操作 | `F.linear` + `topk` | `F.embedding_bag` |
| 反向核心算子 | `F.embedding_bag` (前向native) + `index_add_` | `index_add_` + 高级索引 |
| 替换的不支持算子 | 编码器侧的 scatter 操作 | `_embedding_bag_backward` |
| 循环维度 | `for i in range(k)` | `for i in range(k)` |
| 梯度目标 | input, weight, bias | top_acts, W_T |

两者构成对称的设计：编码器和解码器都使用自定义 autograd Function 绕开 NPU 上不支持的反向算子，用 `index_add_` 这个 NPU 原生支持的 scatter-add 操作来替代。

## 5. 测试

测试文件：`tests/ascend/test_fused_decoder.py`，共 22 个测试用例，覆盖 7 个维度：

| 类别 | 测试 | 说明 |
|------|------|------|
| 基础正确性 | `test_forward_shapes` | 输出形状验证 |
| | `test_forward_matches_naive` | 前向结果 vs naive 实现 |
| | `test_gradient_vs_naive` | 反向梯度 vs naive 实现 |
| | `test_gradient_vs_cpu` | NPU vs CPU 梯度一致性 |
| | `test_large_batch` | 大 batch (4096) 压力测试 |
| CPU 回退检测 | `test_no_cpu_fallback_warnings` | warnings 层面检测 |
| | `test_no_cpu_fallback_profiler` | profiler op trace 级检测 |
| 重复索引 | `test_duplicate_indices_all_same` | 极端重复：每行 k 个索引全相同 |
| | `test_duplicate_indices_high_collision` | 90% 重复，NPU vs CPU 交叉验证 |
| k 边界 | `test_k_boundary[1]` | k=1 退化情况 |
| | `test_k_boundary[2]` | k=2 最小非退化 |
| | `test_k_boundary[128]` | k=128 训练常用上限 |
| dtype | `test_dtype_forward_and_gradient[float32]` | fp32 标准精度 |
| | `test_dtype_forward_and_gradient[bfloat16]` | bf16 主训练路径 |
| | `test_dtype_forward_and_gradient[float16]` | fp16 兼容性 |
| 非连续输入 | `test_non_contiguous_W_dec` | 转置产生的非连续权重 |
| | `test_non_contiguous_acts` | 切片产生的非连续激活 |
| 大规模压力 | `test_large_scale_stress[baseline]` | N=4096, k=32 基准 |
| | `test_large_scale_stress[large_N_d]` | N=8192, d=256 |
| | `test_large_scale_stress[large_M_k]` | M=8192, k=64 |
| | `test_large_scale_stress[max_k]` | k=128 + 峰值显存验证 |
| 端到端回归 | `test_end_to_end_sae_training_step` | SparseCoder mini 训练循环 |

## 6. 效果

实现 FusedDecoder 后，SAE 训练在 Ascend NPU 上的前向和反向传播**全部使用 NPU 原生算子**，不再有任何 CPU 回退：

```
优化前：
  前向：全部 NPU 原生
  反向：_embedding_bag_backward 回退到 CPU  ← 唯一瓶颈

优化后：
  前向：全部 NPU 原生
  反向：全部 NPU 原生（index_add_ 替代 _embedding_bag_backward）
```

## 7. 性能定位

### 7.1 本算子的目标

**消除跨设备瓶颈 + 保证可用性**，而非追求极致吞吐上限。

相对于旧路径（`_embedding_bag_backward` CPU 回退），性能提升显著：
- 消除了 NPU ↔ CPU 数据搬运开销
- 消除了 CPU 计算瓶颈
- 消除了同步阻塞对异步执行流的打断

### 7.2 与 NVIDIA Triton/CUDA kernel 的差距

当前实现与 NVIDIA 上的深度融合 kernel（Triton `xformers_embedding_bag`）相比仍有差距：

| 维度 | NVIDIA Triton kernel | NPU FusedDecoder |
|------|---------------------|------------------|
| kernel launch | 单次 launch | O(k) 次 launch（for-k 循环） |
| 访存模式 | 融合读写，减少中间存储 | 每次循环产生中间读写 |
| 并行粒度 | warp-level 细粒度并行 | 算子级并行，依赖 runtime 调度 |
| 本质瓶颈 | memory-bound，受 HBM 带宽限制 | memory-bound + launch overhead |

关键原因：`for i in range(k)` 循环带来 O(k) 次 kernel launch 和更多中间张量读写。对于 k=32~128，这意味着 32~128 次 `index_add_` 调用，每次都有独立的 kernel 调度开销。

### 7.3 后续优化方向

如需进一步缩小与 NVIDIA 的性能差距：

1. **CANN 自定义算子**：用 AscendCL 或 TBE DSL 编写一个融合的 `sparse_decode_backward` kernel，将整个 for-k 循环融合为单次 kernel launch
2. **Ascend 社区跟进**：关注 torch_npu 后续版本是否原生支持 `_embedding_bag_backward`
3. **算法层面**：探索是否可以用矩阵运算替代 scatter-add（如稀疏矩阵乘法），减少 kernel launch 次数
