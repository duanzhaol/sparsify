#!/usr/bin/env python3
"""
Compute elbow thresholds for model activations.

Usage:
    python compute_elbow_thresholds.py MODEL_NAME \
        --hookpoints "layers.[0-10].self_attn.o_proj" \
        --num_tokens 10000000 \
        --output thresholds.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import matplotlib

from sparsify.data import MemmapDataset
from sparsify.trainer import expand_range_pattern
from sparsify.utils import get_layer_list
from fnmatch import fnmatchcase
from natsort import natsorted
import ml_dtypes

# 使用非交互式后端（避免显示窗口）
matplotlib.use('Agg')

def compute_kneedle_elbow(data: np.ndarray, max_percentile: float = 0.95) -> dict[str, float]:
    """
    计算Kneedle拐点（只保存elbow信息，不计算threshold）

    Args:
        data: 激活数据数组
        max_percentile: 最大分位数（过滤极端值）

    Returns:
        包含elbow_p, elbow_value的字典（不包含threshold，由用户指定alpha计算）
    """
    # 转换为绝对值并展平
    x_abs = np.abs(data).flatten()

    # 计算过滤后的分位数曲线（P0-max_percentile）
    num_points = 500
    percentiles = np.linspace(0, max_percentile, num_points)
    quantiles = np.quantile(x_abs, percentiles)

    # Kneedle算法：找分位数曲线偏离首尾连线最远的点
    p_start, p_end = percentiles[0], percentiles[-1]
    q_start, q_end = quantiles[0], quantiles[-1]

    # 计算每个点到直线的距离
    distances = np.zeros_like(percentiles)
    for i, (p, q) in enumerate(zip(percentiles, quantiles)):
        numerator = (q_end - q_start) * (p - p_start) - (p_end - p_start) * (q - q_start)
        denominator = np.sqrt((p_end - p_start)**2 + (q_end - q_start)**2)
        distances[i] = abs(numerator / denominator)

    # 找最大距离（排除前后5%的边界，避免数值误差）
    margin = 0.05 * (percentiles[-1] - percentiles[0])
    mask = (percentiles > percentiles[0] + margin) & (percentiles < percentiles[-1] - margin)
    valid_indices = np.where(mask)[0]

    if len(valid_indices) == 0:
        raise RuntimeError(
            f"❌ Kneedle阈值检测失败：没有有效的拐点候选！\n"
            f"   数据统计: min={x_abs.min():.4f}, max={x_abs.max():.4f}, "
            f"mean={x_abs.mean():.4f}, std={x_abs.std():.4f}"
        )

    max_idx = valid_indices[np.argmax(distances[valid_indices])]
    max_distance = distances[max_idx]

    # 检查距离是否太小（可能表示没有明显拐点）
    if max_distance < 1e-6:
        raise RuntimeError(
            f"❌ Kneedle阈值检测失败：未找到明显拐点！\n"
            f"   最大偏离距离: {max_distance:.2e}\n"
            f"   数据统计: min={x_abs.min():.4f}, max={x_abs.max():.4f}, "
            f"mean={x_abs.mean():.4f}, std={x_abs.std():.4f}"
        )

    elbow_p = percentiles[max_idx]
    elbow_value = quantiles[max_idx]

    return {
        'elbow_p': float(elbow_p),
        'elbow_value': float(elbow_value)
    }


def plot_elbow_curve(data: np.ndarray, result: dict, name: str, output_path: Path, max_percentile: float = 0.95):
    """
    绘制激活值分布和elbow point

    Args:
        data: 激活数据数组
        result: compute_kneedle_elbow的返回结果
        name: hookpoint名称
        output_path: 图片保存路径
        max_percentile: 最大分位数
    """
    # 计算分位数曲线（和compute_kneedle_elbow相同）
    x_abs = np.abs(data).flatten()
    num_points = 500
    percentiles = np.linspace(0, max_percentile, num_points)
    quantiles = np.quantile(x_abs, percentiles)

    # Kneedle算法的首尾连线
    p_start, p_end = percentiles[0], percentiles[-1]
    q_start, q_end = quantiles[0], quantiles[-1]

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制分位数曲线
    ax.plot(percentiles, quantiles, 'b-', linewidth=2, label='Quantile Curve')

    # 绘制首尾连线
    ax.plot([p_start, p_end], [q_start, q_end], 'r--', linewidth=1.5,
            alpha=0.7, label='Linear Baseline')

    # 标注elbow point
    elbow_p = result['elbow_p']
    elbow_value = result['elbow_value']
    ax.plot(elbow_p, elbow_value, 'ro', markersize=12, label='Elbow Point',
            zorder=5)

    # 添加elbow point的垂直线和文字注释
    ax.axvline(x=elbow_p, color='gray', linestyle=':', alpha=0.5)
    ax.annotate(f'Elbow Point\np={elbow_p:.3f}\nvalue={elbow_value:.6f}',
                xy=(elbow_p, elbow_value),
                xytext=(elbow_p + 0.1, elbow_value + (q_end - q_start) * 0.1),
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # 设置标签和标题
    ax.set_xlabel('Percentile', fontsize=12)
    ax.set_ylabel('Activation Value (Absolute)', fontsize=12)
    ax.set_title(f'Activation Distribution: {name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)

    # 添加统计信息文本框
    stats_text = (
        f'Data Statistics:\n'
        f'Min: {x_abs.min():.6f}\n'
        f'Max: {x_abs.max():.6f}\n'
        f'Mean: {x_abs.mean():.6f}\n'
        f'Std: {x_abs.std():.6f}\n'
        f'Samples: {len(x_abs):,}'
    )
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 保存图片
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def compute_elbow_for_layer(item, max_percentile, plot_dir=None):
    """
    并行计算单个layer的elbow（用于multiprocessing）

    Args:
        item: (name, acts) tuple
        max_percentile: 最大分位数
        plot_dir: 如果提供，则保存可视化图片（字符串路径）

    Returns:
        (name, result, error) tuple
    """
    name, acts = item
    try:
        result = compute_kneedle_elbow(acts, max_percentile)

        # 如果指定了plot_dir，绘制图片
        if plot_dir is not None:
            plot_dir_path = Path(plot_dir)  # 转换为Path对象
            plot_path = plot_dir_path / f"{name.replace('.', '_')}.png"
            print(f"      📊 Plotting {name} -> {plot_path}")  # 调试信息
            plot_elbow_curve(acts, result, name, plot_path, max_percentile)

        return (name, result, None)  # (name, result, error)
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return (name, None, error_msg)  # (name, result, error)


class ActivationCollector:
    """收集模型激活值"""

    def __init__(self, model, hookpoints: list[str], max_tokens: int, special_token_ids: set = None):
        self.model = model  # 完整的模型（用于前向传播）
        self.base_model = model.base_model if hasattr(model, 'base_model') else model
        self.hookpoints = hookpoints
        self.max_tokens = max_tokens
        self.special_token_ids = special_token_ids or set()  # 特殊token IDs (BOS, EOS, etc.)
        self.activations = defaultdict(list)
        self.attention_masks = []  # 保存attention masks
        self.input_ids_list = []  # 保存input_ids用于过滤特殊tokens
        self.tokens_collected = 0
        self.handles = []

    def _make_hook(self, name: str):
        """创建hook函数"""
        def hook(module, input, output):
            if self.tokens_collected >= self.max_tokens:
                return

            # 处理输出（可能是tuple）
            if isinstance(output, tuple):
                output = output[0]

            # 转换为float32再转numpy（numpy不支持bfloat16）
            if output.dtype == torch.bfloat16:
                output = output.float()

            # 转换为numpy并存储（使用CPU节省GPU内存）
            act = output.detach().cpu().numpy()
            self.activations[name].append(act)

        return hook

    def register_hooks(self):
        """注册所有hooks"""
        for name in self.hookpoints:
            module = self.base_model.get_submodule(name)  # 使用base_model
            handle = module.register_forward_hook(self._make_hook(name))
            self.handles.append(handle)

    def remove_hooks(self):
        """移除所有hooks"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def collect(self, dataloader, device):
        """收集激活值"""
        self.model.eval()
        self.register_hooks()

        try:
            with torch.no_grad():
                pbar = tqdm(dataloader, desc="Collecting activations")
                for batch in pbar:
                    if self.tokens_collected >= self.max_tokens:
                        break

                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch.get("attention_mask", None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.cpu().numpy()  # 转到CPU

                    batch_size, seq_len = input_ids.shape

                    # 运行前向传播（触发所有hooks）
                    self.model(input_ids)

                    # 保存attention mask和input_ids
                    if attention_mask is not None:
                        self.attention_masks.append(attention_mask)
                        # 只计数有效tokens（不包括padding）
                        valid_tokens = int(attention_mask.sum())
                        self.tokens_collected += valid_tokens
                    else:
                        # 没有mask，计数所有tokens
                        self.tokens_collected += batch_size * seq_len

                    # 保存input_ids用于过滤特殊tokens
                    self.input_ids_list.append(input_ids.cpu().numpy())

                    pbar.set_postfix({"tokens": f"{self.tokens_collected:,}/{self.max_tokens:,}"})

        finally:
            self.remove_hooks()

        # 合并所有批次的激活值和attention masks
        print("\nConcatenating activations...")

        # 合并attention masks（如果存在）
        if self.attention_masks:
            print(f"  Concatenating attention masks...")
            combined_mask = np.concatenate(self.attention_masks, axis=0)  # [total_samples, seq_len]
            print(f"  Attention mask shape: {combined_mask.shape}")
        else:
            combined_mask = None

        # 合并input_ids用于过滤特殊tokens
        if self.input_ids_list:
            print(f"  Concatenating input_ids...")
            combined_input_ids = np.concatenate(self.input_ids_list, axis=0)  # [total_samples, seq_len]
            print(f"  Input IDs shape: {combined_input_ids.shape}")
        else:
            combined_input_ids = None

        # 并行处理每个hookpoint的激活值
        print(f"  Processing {len(self.hookpoints)} hookpoints...")

        for name in self.hookpoints:
            if not self.activations[name]:
                print(f"  ⚠️  {name}: No activations collected!")
                continue

            # Concatenate激活值
            acts = np.concatenate(self.activations[name], axis=0)  # [total_samples, seq_len, d_in]

            # 创建过滤mask
            if combined_mask is not None or combined_input_ids is not None:
                # 展开activations: [total_samples, seq_len, d_in] -> [total_samples * seq_len, d_in]
                acts_flat = acts.reshape(-1, acts.shape[-1])
                total_tokens = acts_flat.shape[0]

                # 初始化过滤mask（全True表示保留所有）
                filter_mask = np.ones(total_tokens, dtype=bool)

                # 过滤padding tokens
                if combined_mask is not None:
                    mask_flat = combined_mask.reshape(-1)
                    filter_mask &= (mask_flat == 1)

                # 过滤特殊tokens (BOS, EOS, etc.)
                if combined_input_ids is not None and self.special_token_ids:
                    input_ids_flat = combined_input_ids.reshape(-1)
                    # 标记不是特殊token的位置
                    not_special = ~np.isin(input_ids_flat, list(self.special_token_ids))
                    filter_mask &= not_special

                # 应用过滤
                acts_filtered = acts_flat[filter_mask]

                # 统计
                valid_tokens = acts_filtered.shape[0]
                removed_tokens = total_tokens - valid_tokens
                padding_count = np.sum(~(combined_mask.reshape(-1) == 1)) if combined_mask is not None else 0
                special_count = removed_tokens - padding_count

                self.activations[name] = acts_filtered
                if special_count > 0:
                    print(f"  {name}: {acts_filtered.shape} (removed {padding_count:,} padding + {special_count:,} special tokens)")
                else:
                    print(f"  {name}: {acts_filtered.shape} (removed {padding_count:,} padding tokens)")
            else:
                self.activations[name] = acts
                print(f"  {name}: {acts.shape}")

        return dict(self.activations)


def main():
    parser = argparse.ArgumentParser(description="Compute elbow thresholds for model activations")
    parser.add_argument("model", type=str, help="Model name or path")
    parser.add_argument("--dataset", type=str, default="togethercomputer/RedPajama-Data-1T-Sample",
                        help="Dataset name or path")
    parser.add_argument("--hookpoints", nargs="+", required=True,
                        help='Hookpoints to collect (supports range syntax like "layers.[0-10].self_attn.o_proj")')
    parser.add_argument("--num_tokens", type=int, default=10_000_000,
                        help="Number of tokens to collect")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for data loading")
    parser.add_argument("--ctx_len", type=int, default=2048,
                        help="Context length / sequence length for memmap datasets")
    parser.add_argument("--output", type=str, default="thresholds.json",
                        help="Output JSON file")
    parser.add_argument("--max_percentile", type=float, default=0.95,
                        help="Max percentile for kneedle algorithm")
    parser.add_argument("--plot_dir", type=str, default=None,
                        help="Directory to save elbow curve plots (optional)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")

    args = parser.parse_args()

    print(f"🚀 Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map=args.device,
    )
    model.eval()

    # Load tokenizer to get special token IDs
    print(f"🔑 Loading tokenizer to identify special tokens...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Collect all special token IDs
    special_token_ids = set()
    if tokenizer.bos_token_id is not None:
        special_token_ids.add(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        special_token_ids.add(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        special_token_ids.add(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        special_token_ids.add(tokenizer.unk_token_id)
    # Add any additional special tokens
    if hasattr(tokenizer, 'additional_special_tokens_ids'):
        special_token_ids.update(tokenizer.additional_special_tokens_ids)

    print(f"   Found {len(special_token_ids)} special token IDs: {sorted(special_token_ids)}")

    print(f"📊 Loading dataset: {args.dataset}")

    # Try different dataset loading strategies
    dataset = None
    dataset_type = None

    # Strategy 1: Try memmap dataset (sparsify format)
    try:
        dataset = MemmapDataset(args.dataset, ctx_len=args.ctx_len, max_examples=None)
        dataset_type = "memmap"
        print(f"   ✅ Using memmap dataset: {len(dataset)} examples (ctx_len={args.ctx_len})")
    except (FileNotFoundError, ValueError, OSError, TypeError) as e:
        print(f"   ℹ️  Not a memmap dataset, trying HuggingFace...")

    # Strategy 2: Try HuggingFace dataset
    if dataset is None:
        from datasets import load_dataset

        try:
            # Estimate how many examples we need
            # Assuming avg sequence length ~= ctx_len
            estimated_examples = max(
                int(args.num_tokens / args.ctx_len * 2),  # 2x buffer for safety
                1000  # minimum 1k examples
            )
            print(f"   ℹ️  Estimated examples needed: {estimated_examples:,} (for {args.num_tokens:,} tokens)")

            # Load dataset with limit (use streaming if dataset is large)
            print(f"   🔄 Loading HuggingFace dataset (streaming mode)...")
            hf_dataset = load_dataset(args.dataset, split="train", streaming=True)

            # Take only what we need
            hf_dataset = hf_dataset.take(estimated_examples)
            print(f"   ✅ Will process ~{estimated_examples:,} examples")

            # Check if already tokenized (sample first element)
            first_example = next(iter(hf_dataset))
            if "input_ids" in first_example:
                print(f"   ✅ Dataset already tokenized")
                dataset = hf_dataset
                dataset_type = "hf_tokenized"
            else:
                # Need to tokenize (tokenizer already loaded above)
                print(f"   ℹ️  Dataset not tokenized, tokenizing...")

                # Detect text column
                text_column = None
                for col in ["text", "content", "document", "article"]:
                    if col in first_example:
                        text_column = col
                        break

                if text_column is None:
                    raise ValueError(
                        f"Could not find text column in dataset. "
                        f"Available columns: {list(first_example.keys())}"
                    )

                print(f"   ✅ Found text column: '{text_column}'")

                def tokenize_function(examples):
                    return tokenizer(
                        examples[text_column],
                        truncation=True,
                        max_length=args.ctx_len,
                        return_tensors=None,
                    )

                print(f"   🔄 Tokenizing ~{estimated_examples:,} examples...")
                dataset = hf_dataset.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=list(first_example.keys()),
                )
                dataset_type = "hf_raw"
                print(f"   ✅ Tokenization complete")

        except Exception as e:
            print(f"   ❌ Failed to load dataset: {e}")
            raise

    # Create dataloader
    from torch.utils.data import DataLoader

    def collate_fn(examples):
        """Collate function that handles different input formats"""
        input_ids_list = []
        for ex in examples:
            if isinstance(ex, dict):
                ids = ex.get("input_ids", ex.get("tokens", None))
            else:
                ids = ex

            if ids is None:
                raise ValueError(f"Could not find input_ids in example: {ex.keys() if isinstance(ex, dict) else type(ex)}")

            # Convert to tensor if needed
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids)

            input_ids_list.append(ids)

        # Pad all sequences to fixed ctx_len (not just batch max)
        target_len = args.ctx_len

        # Pad each sequence and create attention mask
        padded_ids = []
        attention_mask = []
        for ids in input_ids_list:
            seq_len = ids.shape[0]

            # Truncate if too long
            if seq_len > target_len:
                ids = ids[:target_len]
                seq_len = target_len

            # Create attention mask (1 for real tokens, 0 for padding)
            mask = torch.ones(seq_len, dtype=torch.long)

            if seq_len < target_len:
                # Pad with 0
                padding = torch.zeros(target_len - seq_len, dtype=ids.dtype)
                ids = torch.cat([ids, padding])
                # Extend mask with 0s for padding
                padding_mask = torch.zeros(target_len - seq_len, dtype=torch.long)
                mask = torch.cat([mask, padding_mask])

            padded_ids.append(ids)
            attention_mask.append(mask)

        # Stack into batch
        return {
            "input_ids": torch.stack(padded_ids),
            "attention_mask": torch.stack(attention_mask)
        }

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=0,  # Use 0 for better compatibility
    )

    # 扩展范围模式并匹配实际的module names
    print(f"\n🔍 Expanding hookpoint patterns...")
    expanded_patterns = []
    for pattern in args.hookpoints:
        expanded = expand_range_pattern(pattern)
        expanded_patterns.extend(expanded)
        if len(expanded) > 1:
            print(f"   {pattern} → {len(expanded)} patterns")

    # 匹配实际的module names
    matched_hookpoints = []
    for name, _ in model.base_model.named_modules():  # 使用 base_model，和trainer.py一致
        if any(fnmatchcase(name, pat) for pat in expanded_patterns):
            matched_hookpoints.append(name)

    matched_hookpoints = natsorted(matched_hookpoints)
    print(f"\n✅ Matched {len(matched_hookpoints)} hookpoints:")
    for hp in matched_hookpoints:
        print(f"   - {hp}")

    if not matched_hookpoints:
        print("❌ No hookpoints matched! Check your patterns.")
        return

    # 收集激活值
    print(f"\n📥 Collecting activations (target: {args.num_tokens:,} tokens)...")
    collector = ActivationCollector(model, matched_hookpoints, args.num_tokens, special_token_ids)
    activations = collector.collect(dataloader, args.device)

    # 计算elbow阈值（并行）
    print(f"\n🧮 Computing elbow thresholds (parallel)...")

    # 过滤掉空数据
    valid_activations = {name: acts for name, acts in activations.items() if len(acts) > 0}

    if not valid_activations:
        print("❌ No valid activation data collected!")
        return

    # 创建plot目录（如果指定）
    plot_dir_str = None
    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_dir_str = str(plot_dir)  # 转换为字符串用于multiprocessing
        print(f"   📊 Plots will be saved to: {plot_dir}")

    # 并行计算
    num_workers = min(len(valid_activations), cpu_count())
    print(f"   Using {num_workers} workers for {len(valid_activations)} hookpoints")

    thresholds = {}

    # 使用进程池并行计算
    with Pool(processes=num_workers) as pool:
        # 创建部分应用函数
        compute_func = partial(compute_elbow_for_layer,
                              max_percentile=args.max_percentile,
                              plot_dir=plot_dir_str)  # 传递字符串而不是Path对象

        # 并行计算并显示进度条
        results = list(tqdm(
            pool.imap(compute_func, valid_activations.items()),
            total=len(valid_activations),
            desc="Computing elbows"
        ))

    # 处理结果
    for name, result, error in results:
        if error is not None:
            print(f"   ❌ {name}: {error}")
            continue

        # 转换名称格式：layers.0.self_attn.o_proj → layer_0/self_attn_o_proj
        if name.startswith("layers."):
            parts = name.split('.', 2)  # ['layers', '0', 'self_attn.o_proj']
            layer_idx = parts[1]
            operator = parts[2].replace('.', '_')
            key = f"layer_{layer_idx}/{operator}"
        else:
            # 兜底：如果不是标准格式，直接转换
            key = name.replace(".", "_")

        thresholds[key] = result
        print(f"   ✅ {name}: elbow_p={result['elbow_p']:.4f}, elbow_value={result['elbow_value']:.6f}")

    # 保存结果
    output_path = Path(args.output)
    print(f"\n💾 Saving thresholds to {output_path}")
    with open(output_path, "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"\n✨ Done! Collected {collector.tokens_collected:,} tokens")
    print(f"   Results saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
