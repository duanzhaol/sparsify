# 查找表（Lookup Table）存储格式规范

## 文件组织结构

```
model_dir/
├── lut/
│   ├── metadata.json                                      # 全局元数据
│   └── model.layers.{layer_id}.{module_name}.lut.safetensors  # 各层查找表
```

### 文件命名规则

- **层查找表**: `{layer_path}.lut.safetensors`
  - 示例: `model.layers.0.self_attn.q_proj.lut.safetensors`
  - 示例: `model.layers.0.mlp.gate_proj.lut.safetensors`

---

## 文件内容规范

### 1. {layer_path}.lut.safetensors

每个文件存储单个线性层的完整查找表，包含该层专属的SAE编码器、解码器和预计算结果。

| 键名 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| `encoder_weight` | `[num_basis, input_dim]` | float16/bfloat16 | SAE编码器权重 |
| `encoder_bias` | `[num_basis]` | float16/bfloat16 | SAE编码器偏置 |
| `decoder_weight` | `[num_basis, input_dim]` | float16/bfloat16 | SAE解码器权重 W_dec |
| `decoder_bias` | `[input_dim]` | float16/bfloat16 | SAE解码器偏置 b_dec |
| `precomputed_products` | `[num_basis, output_dim]` | float16/bfloat16 | 预计算的基向量乘积 P_i = W_dec[i] @ W_target |
| `bias_product` | `[output_dim]` | float16/bfloat16 | 偏置乘积 b_dec @ W_target |

**示例**:
```python
{
    "encoder_weight": torch.Tensor([16384, 2048], dtype=torch.float16),
    "encoder_bias": torch.Tensor([16384], dtype=torch.float16),
    "decoder_weight": torch.Tensor([16384, 2048], dtype=torch.float16),
    "decoder_bias": torch.Tensor([2048], dtype=torch.float16),
    "precomputed_products": torch.Tensor([16384, 4096], dtype=torch.float16),
    "bias_product": torch.Tensor([4096], dtype=torch.float16)
}
```

---

### 2. metadata.json

全局元数据文件，描述查找表配置。

```json
{
    "version": "1.0",
    "sae_config": {
        "num_basis": 16384,
        "k_active": 20
    },
    "model_config": {
        "model_type": "llama",
        "num_layers": 32,
        "num_attention_heads": 32,
        "hidden_size": 4096
    },
    "layers": {
        "model.layers.0.self_attn.q_proj": {
            "input_dim": 2048,
            "output_dim": 4096,
            "file": "model.layers.0.self_attn.q_proj.lut.safetensors"
        },
        "model.layers.0.self_attn.k_proj": {
            "input_dim": 2048,
            "output_dim": 4096,
            "file": "model.layers.0.self_attn.k_proj.lut.safetensors"
        }
    },
    "creation_info": {
        "created_at": "2025-12-24T10:00:00Z",
        "source_model": "meta-llama/Llama-3.1-8B",
        "sae_model": "sae_2048_16384.safetensors"
    }
}
```

#### metadata.json 字段说明

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `version` | string | 是 | 格式版本号 |
| `sae_config.num_basis` | int | 是 | SAE基向量数量 |
| `sae_config.k_active` | int | 是 | 推理时激活的Top-K数量 |
| `model_config` | object | 否 | 原始模型配置信息 |
| `layers` | object | 是 | 各层查找表文件映射 |
| `layers.{key}.input_dim` | int | 是 | 该层输入维度 |
| `layers.{key}.output_dim` | int | 是 | 该层输出维度 |
| `layers.{key}.file` | string | 是 | 该层查找表文件名 |
| `creation_info` | object | 否 | 创建时间、来源模型等信息 |

---

## 数学定义

### 离线预计算

给定：
- SAE解码器: `W_dec` `[num_basis, input_dim]`, `b_dec` `[input_dim]`
- 目标线性层权重: `W_target` `[input_dim, output_dim]`

预计算：
```
precomputed_products[i, :] = W_dec[i, :] @ W_target  # [num_basis, output_dim]
bias_product = b_dec @ W_target                       # [output_dim]
```

### 在线推理

给定输入 `x` `[batch_size, input_dim]`:

**Step A - SAE编码**:
```
sparse_act = ReLU(x @ encoder_weight.T + encoder_bias)  # [batch_size, num_basis]
top_k_values, top_k_indices = TopK(sparse_act, k=k_active)  # 选择Top-K激活
```

**Step B - 查表组合**:
```
selected = precomputed_products[top_k_indices]  # [batch_size, k_active, output_dim]
output = sum(top_k_values * selected, dim=1) + bias_product  # [batch_size, output_dim]
```

**（可选）在线计算部分**:

存储decoder权重和偏置用于：

1. **重构质量评估**：计算重构误差以决定哪些维度需要精确计算
```
x_reconstructed = sparse_act @ decoder_weight + decoder_bias  # [batch_size, input_dim]
reconstruction_error = abs(x - x_reconstructed)               # [batch_size, input_dim]
important_dims = select_high_error_dimensions(reconstruction_error)
```

2. **动态在线补偿**：对重要维度进行精确的在线计算
```
# 对选中的重要维度直接计算，而非查表
online_output = x[:, important_dims] @ W_target[important_dims, :]
output[:, critical_output_dims] = online_output
```

最终：`output ≈ x @ W_target`

---

## 类型约定

- **浮点类型**: 优先使用 `float16` 或 `bfloat16` 以节省空间和带宽
- **整数类型**: 索引使用 `int32`
- **存储格式**: 使用 SafeTensors 格式，保证安全性和加载性能

---

## 版本历史

- **v1.0** (2025-12-24): 初始版本，支持SAE基向量分解的基础查找表
