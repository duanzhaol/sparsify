"""预处理数据集并保存到本地"""
import sys
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path

# 添加 lowrank_encoder 到路径，修复 tiled_sparse_coder.py 的导入问题
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "lowrank_encoder"))

import datasets
from datasets import Dataset, load_dataset
from simple_parsing import field, parse
from transformers import AutoTokenizer

from sparsify.data import chunk_and_tokenize

# 启用 datasets 的进度条
datasets.utils.logging.set_verbosity_info()
datasets.enable_progress_bars()


@dataclass
class PreprocessConfig:
    model: str  # tokenizer 来源
    dataset: str  # 原始数据集路径
    output: str  # 输出路径

    split: str = "train"
    ctx_len: int = 2048
    text_column: str = "text"
    num_proc: int = field(default_factory=lambda: cpu_count() // 2)


def main():
    args = parse(PreprocessConfig)

    print(f"\n[1/4] Loading dataset from {args.dataset}")
    dataset = load_dataset(args.dataset, split=args.split)
    print(f"      Loaded {len(dataset)} examples")

    print(f"\n[2/4] Loading tokenizer from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"\n[3/4] Tokenizing with ctx_len={args.ctx_len}, num_proc={args.num_proc}")
    dataset = chunk_and_tokenize(
        dataset,
        tokenizer,
        max_seq_len=args.ctx_len,
        num_proc=args.num_proc,
        text_key=args.text_column,
    )
    print(f"      Tokenized into {len(dataset)} chunks")

    print(f"\n[4/4] Saving to {args.output}")
    dataset.save_to_disk(args.output, num_proc=args.num_proc)
    print(f"\nDone! Dataset saved with {len(dataset)} examples")


if __name__ == "__main__":
    main()
