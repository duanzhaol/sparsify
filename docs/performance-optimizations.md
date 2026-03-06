# SAE Training Performance Optimizations

基于 nsys profiling 对 4-GPU DDP 模式下 SAE 训练流水线的深度分析，实施了以下性能优化。涉及三次提交（`4c902a0`, `94c0cad`, `065ee7a`），修改集中在 `trainer.py` 和 `config.py`。

## 性能结果

| 配置 | 吞吐量 | 提升 |
|------|--------|------|
| 优化前基线 | ~20 it/s | — |
| 全部优化 + `--compile_model` | ~29 it/s | **+45%** |

测试环境：4×GPU，Qwen3-0.6B，`grad_acc_steps=8`，4 个 hookpoints（layers 0/6/12/18 的 q_proj input）

---

## 优化 1：消除 FVU 模式下的冗余 tensor clone（`94c0cad`）

**文件：** `trainer.py:769-770`

**问题：** `all_outputs = outputs.detach().clone()` 在每次 hook 调用时执行，但 `all_outputs` 仅在 `ce`/`kl` loss 分支使用。在 FVU 模式（最常用模式）下，这是一次完整 activation tensor 的无效拷贝。

**改动：**
```python
# Before
all_outputs = outputs.detach().clone()

# After
if self.cfg.loss_fn in ("ce", "kl"):
    all_outputs = outputs.detach().clone()
```

**效果：** 对 hidden_dim=4096, batch=32, seq=2048 的模型，每步节省约 512MB 的 D2D 拷贝。nsys 验证：D2D copy 操作从 384 次/3222MB 降至 208 次/1612MB（减少 50%）。

---

## 优化 2：消除 outlier clipping 模式下的冗余 clone（`94c0cad`）

**文件：** `trainer.py:808-810`

**问题：** `original_outputs = outputs.detach().clone()` 用于 exceed 指标计算，但仅在配置了 elbow thresholds 和 exceed_alphas 时才需要。

**改动：**
```python
# Before
original_outputs = outputs.detach().clone()

# After
if name in self.elbow_thresholds and self.cfg.exceed_alphas:
    original_outputs = outputs.detach().clone()
```

**效果：** 未配置 exceed 指标时，消除一次完整 activation tensor 拷贝。

---

## 优化 3：消除 hook 内部的 GPU 同步阻塞（`4c902a0` + `94c0cad`）

**文件：** `trainer.py` 多处

**问题：** 训练循环中有多处 `torch.cuda.synchronize()` 调用，包括：
- 每步 forward/backward 计时前后的同步（所有步都执行）
- exceed 指标计算后的 `metrics_end.synchronize()`（在 hook 内部，每个 hookpoint 都同步一次）

这些同步强制 CPU 等待所有 GPU 操作完成，导致流水线序列化。

**改动（三部分）：**

**(a) 计时仅在需要记录日志的步执行：**
```python
# 预判下一步是否需要 log
should_time = (
    self.cfg.log_to_wandb
    and next_substep == 0
    and (next_step + 1) % self.cfg.wandb_log_frequency == 0
)
# 所有 CUDA event 操作都用 `if should_time` 守卫
```

**(b) 移除 forward/backward 计时中的即时同步：**
```python
# Before: 每步都同步
torch.cuda.synchronize()
forward_start.record()

# After: 仅在 log 步记录事件，延迟到最后一次性同步
if should_time and device.type == "cuda":
    forward_start.record()
```

**(c) 延迟 metrics 同步到 backward 完成后：**
```python
# Before: 每个 hookpoint 在 hook 内部同步
metrics_end.record()
metrics_end.synchronize()  # 阻塞 CPU！
total_metrics_time += metrics_start.elapsed_time(metrics_end) / 1000.0

# After: 收集事件对，统一在 backward 同步后解析
m_end = torch.cuda.Event(enable_timing=True)
m_end.record()
pending_metrics_events.append((m_start, m_end))

# ... 在外层循环中 ...
backward_end.synchronize()  # 仅一次同步
for m_start, m_end in pending_metrics_events:
    total_metrics_time += m_start.elapsed_time(m_end) / 1000.0
pending_metrics_events.clear()
```

**效果：** 消除非 log 步的所有 GPU 同步（约占总步数的 99%）。log 步也从 N_hookpoints 次同步减少到 1 次。nsys 验证：`cudaStreamSynchronize` 占 CUDA API 时间从 64.5% 大幅下降。

---

## 优化 4：DDP `no_sync()` 消除非同步步的梯度通信（`4c902a0`）

**文件：** `trainer.py:860-875`

**问题：** 默认情况下，DDP 在每次 `backward()` 后都触发 AllReduce 同步梯度。在使用 `grad_acc_steps > 1` 时，中间步的梯度同步是浪费的——只需在最后一个累积步同步。

**改动：**
```python
sync_gradients = (self.global_step + 1) % self.cfg.grad_acc_steps == 0

sync_context = (
    nullcontext()
    if sync_gradients or not isinstance(wrapped, DDP)
    else wrapped.no_sync()
)
with sync_context:
    out = wrapped(x=inputs, y=outputs, dead_mask=...)
    # ... loss.backward() ...
```

**效果：** 对 `grad_acc_steps=8`，每 8 步仅 1 次 AllReduce（而非 8 次）。DDP 通信量减少约 87.5%。

---

## 优化 5：批量 did_fire AllReduce（`94c0cad`）

**文件：** `trainer.py:1138-1149`

**问题：** 每个 hookpoint 的 `did_fire` mask 独立执行一次 NCCL AllReduce，N 个 hookpoints 就有 N 次小通信。

**改动：**
```python
# Before: N 次 AllReduce
for did_fire_mask in did_fire.values():
    dist.all_reduce(did_fire_mask, op=dist.ReduceOp.MAX)

# After: 合并为 1 次 AllReduce
did_fire_keys = list(did_fire.keys())
all_masks = torch.cat([did_fire[k] for k in did_fire_keys])
dist.all_reduce(all_masks, op=dist.ReduceOp.MAX)
offset = 0
for k in did_fire_keys:
    n = did_fire[k].shape[0]
    did_fire[k].copy_(all_masks[offset:offset + n])
    offset += n
```

**效果：** NCCL 调用次数从 N 降至 1，消除 per-call launch overhead。

---

## 优化 6：批量 AllReduce 聚合 metrics（`4c902a0`）

**文件：** `trainer.py:1130-1165, 1229-1255`

**问题：** FVU、auxk_loss、outlier_ratio、exceed 等指标在 hook 内部逐个调用 `self.maybe_all_reduce()`，导致大量小 NCCL 通信。

**改动：** 在 hook 内部仅累积本地值（不做 AllReduce），在 log 步统一通过两个批量 reduce 函数处理：

```python
# hook 内部：只累积本地值
avg_fvu[sae_key] += float(out.fvu.detach() / denom)  # 不再 all_reduce

# log 步：批量 reduce
reduced_avg_fvu = reduce_scalar_mapping(dict(avg_fvu))
reduced_avg_exceed = reduce_nested_scalar_mapping(dict(avg_exceed))
```

`reduce_scalar_mapping` 将所有标量打包成一个 tensor 做一次 AllReduce；`reduce_nested_scalar_mapping` 处理嵌套字典（exceed 指标）。

**效果：** 每步 hook 内的 NCCL 通信从 O(N_hookpoints × N_metrics) 次减少到 0 次，log 步集中处理。

---

## 优化 7：`torch.compile` 模型 transformer 层（`065ee7a`）

**文件：** `config.py:200-203`, `trainer.py:522-530`

**问题：** 模型前向每步产生约 9125 个小 kernel（elementwise 3930、dtype_convert 1297、layernorm 1729 等），kernel launch gap 累计约 38.7ms/step（23% GPU 空闲）。

**改动：**
```python
# config.py
compile_model: bool = False

# trainer.py
if self.cfg.compile_model:
    import torch._dynamo as _dynamo
    _dynamo.config.cache_size_limit = 128  # 允许 KV cache 状态变体编译
    _, layer_list = get_layer_list(self.model)
    for i in range(len(layer_list)):
        layer_list[i] = torch.compile(layer_list[i])
```

**设计要点：**
- **编译单个 transformer layer 而非整个模型**：SAE hook 注册在 layer 级别，编译单层确保每层内部（attention + FFN + layernorm + elementwise）被融合，而层间 hook 不受影响
- **`cache_size_limit=128`**：KV cache 的 `is_initialized` 状态在第一次前向时各层不同（layer 0 先初始化），dynamo 默认 limit=8 会触发回退。提高上限允许所有变体编译；第二次前向后所有 cache 已初始化，图即稳定
- **不关闭 KV cache**：`use_cache=False` 会改变模型执行路径（可能走不同 attention 实现），实测反而更慢（23→19 it/s）
- **默认关闭**：首次编译有预热开销，用户通过 `--compile_model` 显式开启

**效果：** 20 → 29 it/s（+45%），kernel 融合大幅减少 launch overhead。

**使用：**
```bash
python -m sparsify EleutherAI/pythia-160m --compile_model
torchrun --nproc_per_node gpu -m sparsify meta-llama/Meta-Llama-3-8B --compile_model
```

---

## 总结

| 优化 | 类型 | 核心机制 |
|------|------|----------|
| 冗余 clone 守卫 | 内存/带宽 | 条件化 tensor 拷贝 |
| GPU 同步消除 | 延迟 | 仅 log 步同步，延迟 event 解析 |
| DDP no_sync | 通信 | 梯度累积中间步跳过 AllReduce |
| 批量 AllReduce | 通信 | 合并多次小通信为一次 |
| metrics 批量 reduce | 通信 | hook 内零通信，log 步批量处理 |
| torch.compile | 计算 | kernel 融合，减少 launch overhead |
