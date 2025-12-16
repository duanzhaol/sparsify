# Sparsify Hook 函数逻辑分析

本文档详细分析 `trainer.py` 中 `hook` 函数的实现逻辑及其使用方式。

---

## 1. Hook 是什么

PyTorch 的 forward hook 是一种回调机制，允许在模块的 `forward()` 执行后拦截其输入和输出。

```python
handle = module.register_forward_hook(hook_fn)
# hook_fn(module, inputs, outputs) 会在 module.forward() 之后被调用
```

---

## 2. Hook 注册位置

**文件**: `trainer.py:482-508`

```python
# 构建 hookpoint 名称到模块的映射
name_to_module = {
    name: self.model.base_model.get_submodule(name)
    for name in self.cfg.hookpoints
}

# 在训练循环中注册 hook
handles = [
    mod.register_forward_hook(hook) for mod in name_to_module.values()
]
try:
    # 模型前向传播，触发所有 hook
    self.model(x)
finally:
    # 移除 hook
    for handle in handles:
        handle.remove()
```

**执行流程**:
1. 每个 batch 开始时，为所有目标模块注册 hook
2. 执行 `self.model(x)`，触发模型前向传播
3. 当执行到被 hook 的模块时，`hook()` 函数被调用
4. 前向传播结束后，移除所有 hook

---

## 3. Hook 函数完整逻辑

**文件**: `trainer.py:340-447`

### 3.1 函数签名

```python
def hook(module: nn.Module, inputs, outputs):
```

- `module`: 被 hook 的 PyTorch 模块
- `inputs`: 模块 `forward()` 的输入，通常是 tuple
- `outputs`: 模块 `forward()` 的输出

### 3.2 流程图

```
hook(module, inputs, outputs)
│
│ ┌─────────────────────────────────────────────────────────┐
│ │ 第一阶段：数据预处理                                      │
│ └─────────────────────────────────────────────────────────┘
│
├─► [1] 解包 inputs 和 outputs
│       if isinstance(inputs, tuple):
│           inputs = inputs[0]
│       if isinstance(outputs, tuple):
│           outputs, *aux_out = outputs
│
├─► [2] 获取模块名称
│       name = module_to_name[module]
│
├─► [3] 分布式数据收集（如果 distribute_modules=True）
│       dist.all_gather_into_tensor(world_outputs, outputs)
│       dist.all_gather_into_tensor(world_inputs, inputs)  # 仅 transcode
│
│ ┌─────────────────────────────────────────────────────────┐
│ │ 第二阶段：数据变换                                        │
│ └─────────────────────────────────────────────────────────┘
│
├─► [4] 展平维度 [batch, seq, hidden] → [batch*seq, hidden]
│       outputs = outputs.flatten(0, 1)
│       inputs = inputs.flatten(0, 1) if transcode else outputs  ← 【关键】
│       mask = mask.flatten(0, 1)
│
├─► [5] 应用 token mask（排除特定 token）
│       all_outputs = outputs.detach().clone()  # 保留用于 e2e
│       outputs = outputs[mask]
│       inputs = inputs[mask]
│
│ ┌─────────────────────────────────────────────────────────┐
│ │ 第三阶段：SAE 初始化（仅首次迭代）                         │
│ └─────────────────────────────────────────────────────────┘
│
├─► [6] 初始化 encoder bias（仅 transcode 模式）
│       if transcode:
│           mean = inputs.mean(0)
│           mean_image = -mean @ encoder.weight.T
│           encoder.bias = mean_image
│
├─► [7] 初始化 decoder bias
│       mean = outputs.mean(0)
│       b_dec = mean
│
├─► [8] 归一化 decoder 权重（仅 autoencoder 模式）
│       if normalize_decoder and not transcode:
│           sae.set_decoder_norm_to_unit_norm()
│
│ ┌─────────────────────────────────────────────────────────┐
│ │ 第四阶段：SAE 前向传播                                    │
│ └─────────────────────────────────────────────────────────┘
│
├─► [9] 调用 SAE
│       out = wrapped_sae(
│           x=inputs,       # SAE 输入
│           y=outputs,      # 重建目标
│           dead_mask=...   # 死特征 mask
│       )
│
├─► [10] 更新特征激活统计
│        did_fire[name][out.latent_indices.flatten()] = True
│
│ ┌─────────────────────────────────────────────────────────┐
│ │ 第五阶段：损失计算与反向传播                               │
│ └─────────────────────────────────────────────────────────┘
│
├─► [11] 根据 loss_fn 分支
│        │
│        ├─► loss_fn = "ce" 或 "kl"（端到端训练）
│        │       output[mask] = out.sae_out
│        │       return output  # 返回修改后的激活，继续前向传播
│        │
│        └─► loss_fn = "fvu"（局部训练）
│                loss = fvu + auxk_alpha * auxk_loss + multi_topk_fvu / 8
│                loss.backward()  # 局部反向传播
│
└─► 结束
```

### 3.3 关键代码详解

#### 3.3.1 inputs/outputs 选择逻辑

```python
# trainer.py:382-384
outputs = outputs.flatten(0, 1)
inputs = inputs.flatten(0, 1) if self.cfg.sae.transcode else outputs
```

| transcode | inputs 变量值 | outputs 变量值 | SAE 行为 |
|-----------|---------------|----------------|----------|
| False | outputs | outputs | autoencoder: 重建模块输出 |
| True | 原始 inputs | outputs | transcoder: 从输入预测输出 |

#### 3.3.2 SAE 调用

```python
# trainer.py:410-419
out = wrapped(
    x=inputs,      # SAE 编码的输入
    y=outputs,     # 重建的目标
    dead_mask=(
        self.num_tokens_since_fired[name] > self.cfg.dead_feature_threshold
        if self.cfg.auxk_alpha > 0
        else None
    ),
)
```

`wrapped` 是 SAE 模块（可能被 DDP 包装）：
- `x`: 输入给 encoder 的数据
- `y`: decoder 重建的目标
- `dead_mask`: 标记哪些特征已"死亡"，用于 AuxK 损失

#### 3.3.3 损失计算与反向传播

**局部训练 (fvu)**:
```python
# trainer.py:444-447
loss = out.fvu + self.cfg.auxk_alpha * out.auxk_loss + out.multi_topk_fvu / 8
loss.div(acc_steps).backward()
```

**端到端训练 (ce/kl)**:
```python
# trainer.py:425-430
output = all_outputs.clone()
output[mask] = out.sae_out.type_as(output)
output = output.reshape(out_shape)
return (output, *aux_out) if aux_out is not None else output
```

端到端模式下，hook 返回修改后的激活，替换原始输出继续前向传播。

---

## 4. 相关数据结构

### 4.1 maybe_wrapped

```python
# trainer.py:457-468
maybe_wrapped = (
    {
        name: DDP(sae, device_ids=[dist.get_rank()])
        for name, sae in self.saes.items()
    }
    if ddp
    else self.saes
)
```

- DDP 模式：`maybe_wrapped[name]` = `DDP(sae)`
- 非 DDP 模式：`maybe_wrapped[name]` = `sae`

### 4.2 did_fire

```python
# trainer.py:295-298
did_fire = {
    name: torch.zeros(sae.num_latents, device=device, dtype=torch.bool)
    for name, sae in self.saes.items()
}
```

布尔张量，记录每个 latent 在当前 step 是否被激活，用于死特征检测。

### 4.3 num_tokens_since_fired

```python
# trainer.py:168-171
self.num_tokens_since_fired = {
    name: torch.zeros(sae.num_latents, device=device, dtype=torch.long)
    for name, sae in self.saes.items()
}
```

计数器，记录每个 latent 距离上次激活已经过了多少 token。

---

## 5. 分布式训练支持

### 5.1 distribute_modules 模式

当 `distribute_modules=True` 时，不同 GPU 训练不同层的 SAE：

```python
# trainer.py:358-381
if self.cfg.distribute_modules:
    # 收集所有 GPU 的 outputs
    world_outputs = outputs.new_empty(
        outputs.shape[0] * dist.get_world_size(), *outputs.shape[1:]
    )
    dist.all_gather_into_tensor(world_outputs, outputs)
    outputs = world_outputs

    # transcode 模式还需要收集 inputs
    if self.cfg.sae.transcode:
        world_inputs = inputs.new_empty(...)
        dist.all_gather_into_tensor(world_inputs, inputs)
        inputs = world_inputs

    # 检查当前 rank 是否负责这个模块
    if name not in self.module_plan[dist.get_rank()]:
        return  # 跳过不属于当前 rank 的模块
```

### 5.2 DDP 模式

当使用 DDP（非 distribute_modules）时，每个 GPU 都有完整的 SAE 副本：

```python
# trainer.py:244-245
ddp = dist.is_initialized() and not self.cfg.distribute_modules
```

---

## 6. 完整调用链

```
Trainer.fit()
    │
    ├─► DataLoader 迭代
    │       │
    │       └─► 每个 batch:
    │               │
    │               ├─► 注册 hooks
    │               │       handles = [mod.register_forward_hook(hook) for ...]
    │               │
    │               ├─► 模型前向传播
    │               │       self.model(x)
    │               │           │
    │               │           └─► 触发各层 hook
    │               │                   hook(module, inputs, outputs)
    │               │                       │
    │               │                       ├─► SAE 前向传播
    │               │                       ├─► 计算损失
    │               │                       └─► 反向传播（fvu 模式）
    │               │
    │               ├─► 移除 hooks
    │               │       for handle in handles: handle.remove()
    │               │
    │               ├─► 优化器更新（每 grad_acc_steps 步）
    │               │       optimizer.step()
    │               │
    │               └─► 保存检查点（每 save_every 步）
    │
    └─► 训练结束，保存最终模型
```

---

## 7. 关键配置参数

| 参数 | 影响 hook 的行为 |
|------|-----------------|
| `sae.transcode` | 决定 inputs/outputs 的选择逻辑 |
| `sae.normalize_decoder` | 是否在每步归一化 decoder |
| `loss_fn` | 决定损失计算方式（局部 vs 端到端）|
| `auxk_alpha` | 是否计算 AuxK 损失 |
| `distribute_modules` | 是否跨 GPU 分布模块 |
| `exclude_tokens` | 影响 mask 的生成 |
