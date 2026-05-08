# Qwen3.5-4B SAE 训练流程

本文记录当前仓库对 Qwen3.5-4B 的支持状态、相关代码改动，以及从数据预处理到阈值生成、训练和导出的命令。

## 目标

Qwen3.5-4B 的三个 SAE 训练目标都是模块输入 hidden state，而不是 projection 输出。

| 目标 | 训练 hookpoint | 层 | SAE 输入 | 部署侧融合矩阵 |
| --- | --- | --- | --- | --- |
| GDN qkvz | `layers.X.linear_attn.in_proj_qkv` | `0-2,4-6,8-10,12-14,16-18,20-22,24-26,28-30` | layernorm 后进入 Gated DeltaNet projection 前的 `x ∈ R^2560` | `in_proj_qkv + in_proj_z`, `2560 -> 12288` |
| q_gatekv | `layers.X.self_attn.q_proj` | `3,7,11,15,19,23,27,31` | attention layernorm 后的 `x ∈ R^2560` | `q_proj + k_proj + v_proj`, 其中 `q_proj` 含 query/gate, `2560 -> 10240` |
| gate_up | `layers.X.mlp.up_proj` | `0-31` | MLP layernorm 后的 `x ∈ R^2560` | `gate_proj + up_proj`, `2560 -> 18432` |

Trainer 的 forward hook 使用模块输入，所以虽然 hookpoint 名字是 `in_proj_qkv`、`q_proj`、`up_proj`，实际训练目标都是进入这些线性层之前的 hidden state。

## 已修改内容

- `sparsify/__main__.py`
  - 对本地 Qwen3.5 模型路径使用 `AutoModelForCausalLM` 加载 text backbone，使 hookpoint 保持 `layers.X...` 风格。
  - 已分词数据集包含 `input_ids` 时跳过在线 tokenization。
- `sparsify/utils.py`
  - 支持从 `config.text_config.num_hidden_layers` 读取层数。
  - 支持嵌套 layer 名称的最大层号解析。
- `research/AutoResearch/target_profile.py`
  - 增加 Qwen3.5-4B 的 `gdn_qkvz`、`q_gatekv`、`gate_up` 三个 target profile。
- `compute_elbow_thresholds.py`
  - 保留原有 Kneedle elbow 算法。
  - 增加对 `Dataset.save_to_disk()` 输出目录的 `load_from_disk` 兼容。
- `scripts/export/export_expert_jumprelu_lut.py`
  - 增加 `gdn_qkvz`、`q_gatekv`、`gate_up` projection specs。
  - 支持 Qwen3.5 这种不同 operator 覆盖不同层集合的导出。
- `scripts/full/qwen3.5-4B/expert_jumprelu/`
  - 增加三类训练脚本：
    - `gdn_qkvz.sh`
    - `q_gatekv.sh`
    - `gate_up.sh`

## 数据预处理

建议在 `tmux` 里跑长任务。

```bash
tmux new -s qwen35-preprocess
```

预处理命令：

```bash
TOKENIZERS_PARALLELISM=false python scripts/archive/preprocess_dataset.py \
  --model /root/models/Qwen3.5-4B \
  --dataset /root/fineweb-edu/sample/10BT \
  --output /root/fineweb-edu/sample/10BT-tokenized-qwen3.5-2048 \
  --split train \
  --ctx_len 2048 \
  --text_column text \
  --num_proc 16
```

不要把 `--num_proc` 开到 120。当前 `chunk_and_tokenize()` 的 map batch size 是 2048，120 个 worker 会有很高的内存峰值，已经确认会触发 OOM kill。

预处理完成后检查：

```bash
python - <<'PY'
from datasets import load_from_disk

p = "/root/fineweb-edu/sample/10BT-tokenized-qwen3.5-2048"
ds = load_from_disk(p)
print("len", len(ds))
print("features", ds.features)
print("format", ds.format)
print("first_len", len(ds[0]["input_ids"]))
print("last_len", len(ds[len(ds) - 1]["input_ids"]))
PY
```

当前已生成数据集的检查结果：

```text
len 4882995
features {'input_ids': List(Value('int32')), 'overflow_to_sample_mapping': Value('int64')}
first_len 2048
last_len 2048
```

## 生成 thresholds

thresholds 使用 `compute_elbow_thresholds.py` 生成。它会在 hookpoint 模块输入上收集激活，对 `abs(x)` 的分位数曲线运行 Kneedle，输出每层的 `elbow_p` 和 `elbow_value`。

### GDN qkvz

```bash
python compute_elbow_thresholds.py /root/models/Qwen3.5-4B \
  --dataset /root/fineweb-edu/sample/10BT-tokenized-qwen3.5-2048 \
  --hookpoints "layers.[0-2,4-6,8-10,12-14,16-18,20-22,24-26,28-30].linear_attn.in_proj_qkv" \
  --num_tokens 100000 \
  --batch_size 1 \
  --ctx_len 2048 \
  --output thresholds/Qwen3.5-4B/thresholds_gdn_qkvz.json \
  --max_percentile 0.95
```

### q_gatekv

```bash
python compute_elbow_thresholds.py /root/models/Qwen3.5-4B \
  --dataset /root/fineweb-edu/sample/10BT-tokenized-qwen3.5-2048 \
  --hookpoints "layers.[3,7,11,15,19,23,27,31].self_attn.q_proj" \
  --num_tokens 100000 \
  --batch_size 1 \
  --ctx_len 2048 \
  --output thresholds/Qwen3.5-4B/thresholds_q_gatekv.json \
  --max_percentile 0.95
```

### gate_up

```bash
python compute_elbow_thresholds.py /root/models/Qwen3.5-4B \
  --dataset /root/fineweb-edu/sample/10BT-tokenized-qwen3.5-2048 \
  --hookpoints "layers.[0-31].mlp.up_proj" \
  --num_tokens 100000 \
  --batch_size 1 \
  --ctx_len 2048 \
  --output thresholds/Qwen3.5-4B/thresholds_gate_up.json \
  --max_percentile 0.95
```

### 检查 thresholds

```bash
python - <<'PY'
import json
from pathlib import Path

expected = {
    "thresholds/Qwen3.5-4B/thresholds_gdn_qkvz.json": {
        f"layer_{i}/linear_attn_in_proj_qkv"
        for group in [(0, 2), (4, 6), (8, 10), (12, 14), (16, 18), (20, 22), (24, 26), (28, 30)]
        for i in range(group[0], group[1] + 1)
    },
    "thresholds/Qwen3.5-4B/thresholds_q_gatekv.json": {
        f"layer_{i}/self_attn_q_proj" for i in [3, 7, 11, 15, 19, 23, 27, 31]
    },
    "thresholds/Qwen3.5-4B/thresholds_gate_up.json": {
        f"layer_{i}/mlp_up_proj" for i in range(32)
    },
}

for path, exp in expected.items():
    data = json.loads(Path(path).read_text())
    keys = set(data)
    bad = [
        k for k, v in data.items()
        if not (0 <= float(v["elbow_p"]) <= 0.95 and float(v["elbow_value"]) > 0)
    ]
    print(path)
    print("  expected", len(exp), "actual", len(keys), "missing", len(exp - keys), "extra", len(keys - exp), "bad", len(bad))
PY
```

当前生成结果：

```text
thresholds_gdn_qkvz.json: expected 24, actual 24, missing 0, extra 0, bad 0
thresholds_q_gatekv.json: expected 8, actual 8, missing 0, extra 0, bad 0
thresholds_gate_up.json: expected 32, actual 32, missing 0, extra 0, bad 0
```

当前 Qwen3.5-4B elbow value 统计：

```text
GDN qkvz input: n=24, min=0.4082, median=0.8125, mean=0.8860, max=1.1797
q_gatekv input: n=8, min=0.7227, median=0.8613, mean=0.8789, max=1.0781
gate_up input: n=32, min=0.2715, median=0.7324, mean=0.7983, max=1.1641
```

## 训练

三类训练脚本都使用：

- `MODEL_PATH=$HOME/models/Qwen3.5-4B`
- `DATASET_PATH=$HOME/fineweb-edu/sample/10BT-tokenized-qwen3.5-2048`
- `ARCHITECTURE=expert_jumprelu`
- `K=64`
- `NUM_EXPERTS=64`
- `ACTIVE_EXPERTS=2`
- `LATENTS_PER_EXPERT=128`
- `OPTIMIZER=adam`
- `LR=8e-4`
- `NPROC_PER_NODE=4`
- `MAX_TOKENS=100000000`

启动命令：

```bash
bash scripts/full/qwen3.5-4B/expert_jumprelu/gdn_qkvz.sh
```

```bash
bash scripts/full/qwen3.5-4B/expert_jumprelu/q_gatekv.sh
```

```bash
bash scripts/full/qwen3.5-4B/expert_jumprelu/gate_up.sh
```

输出目录：

```text
checkpoints/qwen3.5-4B/expert_jumprelu_gdn_qkvz
checkpoints/qwen3.5-4B/expert_jumprelu_q_gatekv
checkpoints/qwen3.5-4B/expert_jumprelu_gate_up
```

`gdn_qkvz.sh` 和 `gate_up.sh` 分两段层跑；`q_gatekv.sh` 覆盖 8 个 full attention 层。

## 导出

训练完成后，用 expert JumpReLU exporter。Qwen3.5 的三类 operator 层集合不同，不要对 `gdn_qkvz` 或 `q_gatekv` 使用 `--merge-output-dir`。

示例：

```bash
python scripts/export/export_expert_jumprelu_lut.py \
  /root/models/Qwen3.5-4B \
  checkpoints/qwen3.5-4B \
  --output-dir exports/qwen3.5-4B/expert_jumprelu_gdn_qkvz \
  --layers 0-2,4-6,8-10,12-14,16-18,20-22,24-26,28-30 \
  --operators gdn_qkvz \
  --compensation-ratio 0.25 \
  --dtype bfloat16 \
  --device cpu \
  --batch-size 1024
```

```bash
python scripts/export/export_expert_jumprelu_lut.py \
  /root/models/Qwen3.5-4B \
  checkpoints/qwen3.5-4B \
  --output-dir exports/qwen3.5-4B/expert_jumprelu_q_gatekv \
  --layers 3,7,11,15,19,23,27,31 \
  --operators q_gatekv \
  --compensation-ratio 0.25 \
  --dtype bfloat16 \
  --device cpu \
  --batch-size 1024
```

```bash
python scripts/export/export_expert_jumprelu_lut.py \
  /root/models/Qwen3.5-4B \
  checkpoints/qwen3.5-4B \
  --output-dir exports/qwen3.5-4B/expert_jumprelu_gate_up \
  --layers 0-31 \
  --operators gate_up \
  --compensation-ratio 0.25 \
  --dtype bfloat16 \
  --device cpu \
  --batch-size 1024
```

## 常见问题

### `preprocess_dataset.py` 报 required arguments

这个脚本使用命名参数，不接受三个 positional 参数。必须写：

```bash
--model ... --dataset ... --output ...
```

### preprocess worker abrupt died

如果日志是：

```text
RuntimeError: One of the subprocesses has abruptly died during map operation
```

先查内核日志：

```bash
journalctl -k --since "40 min ago" | rg -i "out of memory|oom|killed process|python"
```

之前确认过 `--num_proc 120` 会触发 OOM。用 `--num_proc 16` 更稳。

### SSH 断线

长任务用 `tmux`：

```bash
tmux new -s qwen35
tmux attach -t qwen35
```

## 已验证命令

```bash
pytest tests/test_compute_elbow_thresholds.py -q
python -m py_compile compute_elbow_thresholds.py
bash -n scripts/full/qwen3.5-4B/expert_jumprelu/gdn_qkvz.sh \
  scripts/full/qwen3.5-4B/expert_jumprelu/q_gatekv.sh \
  scripts/full/qwen3.5-4B/expert_jumprelu/gate_up.sh
```
