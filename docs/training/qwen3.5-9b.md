# Qwen3.5-9B SAE 训练流程

本文记录 Qwen3.5-9B 的结构、threshold 生成命令、实际训练配置及跨机器汇总方式。4B 的数据预处理和完整训练流程见 `docs/training/qwen3.5-4b.md`。

## 结构

本地模型路径：

```text
/root/models/Qwen3.5-9B
```

9B 和 4B 的 tokenizer 文件 hash 一致，可以复用当前 Qwen3.5 tokenized dataset：

```text
/root/fineweb-edu/sample/10BT-tokenized-qwen3.5-2048
```

模型权重、tokenized dataset、训练 checkpoint 和 LUT 体积较大，均不保存在 Git 中；复现训练需要在运行机器上提供这些路径。依赖版本范围见 `pyproject.toml`。

已确认结构：

| 项 | 值 |
| --- | --- |
| hidden size | `4096` |
| layers | `32` |
| vocab size | `248320` |
| GDN layers | `0-2,4-6,8-10,12-14,16-18,20-22,24-26,28-30` |
| full attention layers | `3,7,11,15,19,23,27,31` |
| MLP layers | `0-31` |

训练目标：

| 目标 | hookpoint | SAE 输入 | 部署侧融合矩阵 |
| --- | --- | --- | --- |
| GDN qkvz | `layers.X.linear_attn.in_proj_qkv` | GDN 输入 hidden state, `R^4096` | `in_proj_qkv + in_proj_z`, `4096 -> 12288` |
| q_gatekv | `layers.X.self_attn.q_proj` | full attention 输入 hidden state, `R^4096` | `q_proj + k_proj + v_proj`, `4096 -> 10240` |
| gate_up | `layers.X.mlp.up_proj` | MLP 输入 hidden state, `R^4096` | `gate_proj + up_proj`, `4096 -> 24576` |
| lm_head | `lm_head` | final hidden state, `R^4096` | `lm_head.weight`, `4096 -> 248320` |

## 生成 thresholds

```bash
mkdir -p thresholds/Qwen3.5-9B
```

### GDN qkvz

```bash
python compute_elbow_thresholds.py /root/models/Qwen3.5-9B \
  --dataset /root/fineweb-edu/sample/10BT-tokenized-qwen3.5-2048 \
  --hookpoints "layers.[0-2,4-6,8-10,12-14,16-18,20-22,24-26,28-30].linear_attn.in_proj_qkv" \
  --num_tokens 100000 \
  --batch_size 1 \
  --ctx_len 2048 \
  --output thresholds/Qwen3.5-9B/thresholds_gdn_qkvz.json \
  --max_percentile 0.95
```

### q_gatekv

```bash
python compute_elbow_thresholds.py /root/models/Qwen3.5-9B \
  --dataset /root/fineweb-edu/sample/10BT-tokenized-qwen3.5-2048 \
  --hookpoints "layers.[3,7,11,15,19,23,27,31].self_attn.q_proj" \
  --num_tokens 100000 \
  --batch_size 1 \
  --ctx_len 2048 \
  --output thresholds/Qwen3.5-9B/thresholds_q_gatekv.json \
  --max_percentile 0.95
```

### gate_up

```bash
python compute_elbow_thresholds.py /root/models/Qwen3.5-9B \
  --dataset /root/fineweb-edu/sample/10BT-tokenized-qwen3.5-2048 \
  --hookpoints "layers.[0-31].mlp.up_proj" \
  --num_tokens 100000 \
  --batch_size 1 \
  --ctx_len 2048 \
  --output thresholds/Qwen3.5-9B/thresholds_gate_up.json \
  --max_percentile 0.95
```

### lm_head

目前还没有生成 `thresholds_lm_head.json`；下面是需要训练 `lm_head` 时运行的命令。

```bash
python compute_elbow_thresholds.py /root/models/Qwen3.5-9B \
  --dataset /root/fineweb-edu/sample/10BT-tokenized-qwen3.5-2048 \
  --hookpoints lm_head \
  --num_tokens 100000 \
  --batch_size 1 \
  --ctx_len 2048 \
  --output thresholds/Qwen3.5-9B/thresholds_lm_head.json \
  --max_percentile 0.95
```

已保存的三份 threshold 文件分别覆盖 GDN 24 层、full attention 8 层、MLP 32 层。

## 训练

层内算子的三个脚本使用以下配置：

- `NPROC_PER_NODE=4`
- `BATCH_SIZE=1`
- `GRAD_ACC_STEPS=8`
- `COMPILE_MODEL=0`
- `K=128`, `NUM_EXPERTS=64`, `ACTIVE_EXPERTS=2`, `LATENTS_PER_EXPERT=192`
- `LR=8e-4`, `MAX_TOKENS=100000000`（每一段分别计数）

`gdn_qkvz.sh` 顺序训练三段：`0-2,4-6,8-9`，`10,12-14,16-18,20`，`21-22,24-26,28-30`。`gate_up.sh` 顺序训练 `0-11`、`12-21`、`22-31`。直接运行整个脚本会依次启动所有段；在多台机器上分工时，每台机器只运行分配给自己的命令段。`q_gatekv.sh` 一次覆盖全部 8 个 full attention 层。

先跑对应 threshold，再启动脚本：

```bash
bash scripts/full/qwen3.5-9B/expert_jumprelu/gdn_qkvz.sh
```

```bash
bash scripts/full/qwen3.5-9B/expert_jumprelu/q_gatekv.sh
```

```bash
bash scripts/full/qwen3.5-9B/expert_jumprelu/gate_up.sh
```

```bash
bash scripts/full/qwen3.5-9B/expert_jumprelu/lm_head.sh
```

`lm_head.sh` 单独使用 2 卡、`K=256` 和 `LATENTS_PER_EXPERT=256`，其 threshold 尚缺；目前没有本机的 9B `lm_head` checkpoint。

正式跑之前建议用 `MAX_TOKENS=200000` 做 smoke，确认 hookpoint、显存和日志都正常后再全量跑：

```bash
MAX_TOKENS=200000 bash scripts/full/qwen3.5-9B/expert_jumprelu/q_gatekv.sh
```

### 本机训练记录（2026-05）

| 算子 | 本机层 | 记录的训练进度 | 保存目录 |
| --- | --- | --- | --- |
| GDN qkvz | `0,1,2,4,5,6,8,9` | `summary.json` 记录 100,007,936 tokens，已达到 1 亿目标 | `checkpoints/qwen3.5-9B/expert_jumprelu_gdn_qkvz/` |
| gate_up | `0-11` | `metrics.jsonl` 最后一条为 70,713,344 tokens；有 `best/` 权重，但未见完成摘要 | `checkpoints/qwen3.5-9B/expert_jumprelu_gate_up/` |

实际两组运行均使用 `K=128`、64 experts、2 active experts、每 expert 192 latents。其他层/算子当时安排在其他机器上；本机尚无其 checkpoint。请在合并时核对各机器的配置，并决定是否采用未跑满的 `gate_up` 权重。

为了传输，这两组已另存为 `checkpoints/qwen3.5-9B-clean/`（约 7.6 GB）。只保留各层 `best/layers.*/cfg.json`、`sae.safetensors` 和少量运行配置/来源信息，不含 optimizer、DDP state、日志或重复的普通快照。该目录被 Git 忽略，必须单独备份。把其他机器同样精简的 run 目录汇总进同一个根目录后，exporter 可以递归发现各层的 `best/` 权重。

## 导出

`gdn_qkvz`、`q_gatekv`、`gate_up` 可以继续使用 expert JumpReLU exporter，注意输出目录换成 9B：

下面的完整 `gate_up` 命令需要先汇总全部 32 层；只导出本机已有层时，把 checkpoint 路径改成 `checkpoints/qwen3.5-9B-clean`、`--layers` 改为 `0-11`。GDN 本机层则使用 `--layers 0-2,4-6,8-9 --operators gdn_qkvz`。不要给 GDN 或 q_gatekv 使用 `--merge-output-dir`。

```bash
python scripts/export/export_expert_jumprelu_lut.py \
  /root/models/Qwen3.5-9B \
  checkpoints/qwen3.5-9B \
  --output-dir exports/qwen3.5-9B/expert_jumprelu_gate_up \
  --layers 0-31 \
  --operators gate_up \
  --compensation-ratio 0.25 \
  --dtype bfloat16 \
  --device cpu \
  --batch-size 1024
```

当前 dense `jumprelu` 的 `lm_head` 可以用 `export_dense_jumprelu_lm_head_lut.py` 导出；`expert_jumprelu` 的 `lm_head` 还没有专用 exporter。
