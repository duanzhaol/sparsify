# Hook Mode 设计方案（简化版）

## 1. 目标

支持三种 hook 数据模式：
1. **output**: 在模块输出上训练 autoencoder（默认）
2. **input**: 在模块输入上训练 autoencoder（新增）
3. **transcode**: 从模块输入预测模块输出

## 2. 设计

### 2.1 新增参数

在 `TrainConfig` 中添加：

```python
hook_mode: Literal["output", "input", "transcode"] = "output"
"""
Activation hook mode:
- output: autoencoder on module outputs (default)
- input: autoencoder on module inputs
- transcode: predict module outputs from inputs
"""
```

### 2.2 删除旧参数

从 `SparseCoderConfig` 中删除：

```python
transcode: bool = False  # 删除
```

同时删除 `config.py` 中的 `TranscoderConfig` 别名。

## 3. 代码修改

### 3.1 config.py

```python
@dataclass
class SparseCoderConfig(Serializable):
    activation: Literal["groupmax", "topk"] = "topk"
    expansion_factor: int = 32
    normalize_decoder: bool = True
    num_latents: int = 0
    k: int = 32
    multi_topk: bool = False
    skip_connection: bool = False
    # 删除 transcode


SaeConfig = SparseCoderConfig
# 删除 TranscoderConfig


@dataclass
class TrainConfig(Serializable):
    sae: SparseCoderConfig

    # ... 其他现有参数 ...

    hook_mode: Literal["output", "input", "transcode"] = "output"
    """
    Activation hook mode:
    - output: autoencoder on module outputs (default)
    - input: autoencoder on module inputs
    - transcode: predict module outputs from inputs
    """
```

### 3.2 trainer.py 修改点

#### (1) 端到端 transcoder 处理 (行 329-331)

```python
# 原
if self.cfg.sae.transcode:

# 改
if self.cfg.hook_mode == "transcode":
```

#### (2) 分布式 all_gather (行 366-371)

```python
# 原
if self.cfg.sae.transcode:
    world_inputs = inputs.new_empty(...)
    dist.all_gather_into_tensor(world_inputs, inputs)
    inputs = world_inputs

# 改
if self.cfg.hook_mode in ("input", "transcode"):
    world_inputs = inputs.new_empty(
        inputs.shape[0] * dist.get_world_size(), *inputs.shape[1:]
    )
    dist.all_gather_into_tensor(world_inputs, inputs)
    inputs = world_inputs
```

#### (3) inputs/outputs 选择 (行 382-384) - 核心修改

```python
# 原
outputs = outputs.flatten(0, 1)
inputs = inputs.flatten(0, 1) if self.cfg.sae.transcode else outputs

# 改
outputs = outputs.flatten(0, 1)
inputs = inputs.flatten(0, 1)
match self.cfg.hook_mode:
    case "output":
        inputs = outputs
    case "input":
        outputs = inputs
    case "transcode":
        pass
```

#### (4) encoder bias 初始化 (行 398)

```python
# 原
if self.cfg.sae.transcode:

# 改
if self.cfg.hook_mode == "transcode":
```

#### (5) normalize_decoder - hook 内 (行 407)

```python
# 原
if raw.cfg.normalize_decoder and not self.cfg.sae.transcode:

# 改
if raw.cfg.normalize_decoder and self.cfg.hook_mode != "transcode":
```

#### (6) normalize_decoder - 优化器步后 (行 513)

```python
# 原
if self.cfg.sae.normalize_decoder and not self.cfg.sae.transcode:

# 改
if self.cfg.sae.normalize_decoder and self.cfg.hook_mode != "transcode":
```

## 4. 行为对照表

| hook_mode | SAE 输入 | SAE 目标 | normalize_decoder |
|-----------|----------|----------|-------------------|
| `output` | outputs | outputs | ✓ |
| `input` | inputs | inputs | ✓ |
| `transcode` | inputs | outputs | ✗ |

## 5. 使用示例

```bash
# 默认：autoencoder on outputs
python -m sparsify Qwen/Qwen3-8B --hookpoints "layers.16.mlp"

# 新增：autoencoder on inputs
python -m sparsify Qwen/Qwen3-8B --hookpoints "layers.16.self_attn.o_proj" --hook_mode input

# Transcoder
python -m sparsify Qwen/Qwen3-8B --hookpoints "layers.16.mlp" --hook_mode transcode
```

## 6. 修改清单

| 文件 | 修改 |
|------|------|
| `config.py` | 删除 `transcode`，删除 `TranscoderConfig`，添加 `hook_mode` |
| `trainer.py:329` | `self.cfg.sae.transcode` → `self.cfg.hook_mode == "transcode"` |
| `trainer.py:366` | `self.cfg.sae.transcode` → `self.cfg.hook_mode in ("input", "transcode")` |
| `trainer.py:382-384` | 重写为 match 语句 |
| `trainer.py:398` | `self.cfg.sae.transcode` → `self.cfg.hook_mode == "transcode"` |
| `trainer.py:407` | `not self.cfg.sae.transcode` → `self.cfg.hook_mode != "transcode"` |
| `trainer.py:513` | `not self.cfg.sae.transcode` → `self.cfg.hook_mode != "transcode"` |
