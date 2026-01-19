import os
import sys
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
from datetime import timedelta
from glob import glob
from multiprocessing import cpu_count
from pathlib import Path

# Add project root to path so lowrank_encoder can be imported
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from simple_parsing import field, parse
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)

from .data import MemmapDataset, chunk_and_tokenize
from .trainer import TrainConfig, Trainer, load_sae_checkpoint
from .utils import simple_parse_args_string


@dataclass
class RunConfig(TrainConfig):
    model: str = field(
        default="HuggingFaceTB/SmolLM2-135M",
        positional=True,
    )
    """Name of the model to train."""

    dataset: str = field(
        default="EleutherAI/SmolLM2-135M-10B",
        positional=True,
    )
    """Path to the dataset to use for training."""

    split: str = "train"
    """Dataset split to use for training."""

    ctx_len: int = 2048
    """Context length to use for training."""

    # Use a dummy encoding function to prevent the token from being saved
    # to disk in plain text
    hf_token: str | None = field(default=None, encoding_fn=lambda _: None)
    """Huggingface API token for downloading models."""

    revision: str | None = None
    """Model revision to use for training."""

    load_in_8bit: bool = False
    """Load the model in 8-bit mode."""

    max_examples: int | None = None
    """Maximum number of examples to use for training."""

    resume: bool = False
    """Whether to try resuming from the checkpoint present at `checkpoints/run_name`."""

    text_column: str = "text"
    """Column name to use for text data."""

    shuffle_seed: int = 42
    """Random seed for shuffling the dataset."""

    data_preprocessing_num_proc: int = field(
        default_factory=lambda: cpu_count() // 2,
    )
    """Number of processes to use for preprocessing data"""

    data_args: str = field(
        default="",
    )
    """Arguments to pass to the HuggingFace dataset constructor in the
    format 'arg1=val1,arg2=val2'."""


def load_artifacts(
    args: RunConfig, rank: int
) -> tuple[PreTrainedModel, Dataset | MemmapDataset]:
    if args.load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = "auto"

    # End-to-end training requires a model with a causal LM head
    model_cls = AutoModel if args.loss_fn == "fvu" else AutoModelForCausalLM
    model = model_cls.from_pretrained(
        args.model,
        device_map={"": f"cuda:{rank}"},
        quantization_config=(
            BitsAndBytesConfig(load_in_8bit=args.load_in_8bit)
            if args.load_in_8bit
            else None
        ),
        revision=args.revision,
        torch_dtype=dtype,
        token=args.hf_token,
    )

    # For memmap-style datasets
    if args.dataset.endswith(".bin"):
        dataset = MemmapDataset(args.dataset, args.ctx_len, args.max_examples)
    else:
        # For Huggingface datasets
        try:
            kwargs = simple_parse_args_string(args.data_args)
            dataset = load_dataset(args.dataset, split=args.split, **kwargs)
        except ValueError as e:
            # Automatically use load_from_disk if appropriate
            if "load_from_disk" in str(e):
                dataset = Dataset.load_from_disk(args.dataset, keep_in_memory=False)
            else:
                raise e

        assert isinstance(dataset, Dataset)
        if "input_ids" not in dataset.column_names:
            tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)
            dataset = chunk_and_tokenize(
                dataset,
                tokenizer,
                max_seq_len=args.ctx_len,
                num_proc=args.data_preprocessing_num_proc,
                text_key=args.text_column,
            )
        else:
            print("Dataset already tokenized; skipping tokenization.")

        print(f"Shuffling dataset with seed {args.shuffle_seed}")
        dataset = dataset.shuffle(args.shuffle_seed)

        dataset = dataset.with_format("torch")
        if limit := args.max_examples:
            dataset = dataset.select(range(limit))

    return model, dataset


def run():
    local_rank = os.environ.get("LOCAL_RANK")
    ddp = local_rank is not None
    rank = int(local_rank) if ddp else 0

    if ddp:
        torch.cuda.set_device(int(local_rank))

        # Increase the default timeout in order to account for slow downloads
        # and data preprocessing on the main rank
        dist.init_process_group(
            "nccl", device_id=torch.device(rank), timeout=timedelta(weeks=1)
        )

        if rank == 0:
            print(f"Using DDP across {dist.get_world_size()} GPUs.")

    args = parse(RunConfig)

    # Prevent ranks other than 0 from printing
    with nullcontext() if rank == 0 else redirect_stdout(None):
        # Awkward hack to prevent other ranks from duplicating data preprocessing
        if not ddp or rank == 0:
            model, dataset = load_artifacts(args, rank)
        if ddp:
            dist.barrier()
            if rank != 0:
                model, dataset = load_artifacts(args, rank)

            # Drop examples that are indivisible across processes to prevent deadlock
            remainder_examples = len(dataset) % dist.get_world_size()
            dataset = dataset.select(range(len(dataset) - remainder_examples))

            dataset = dataset.shard(dist.get_world_size(), rank)

            # Drop examples that are indivisible across processes to prevent deadlock
            remainder_examples = len(dataset) % dist.get_world_size()
            dataset = dataset.select(range(len(dataset) - remainder_examples))

        print(f"Training on '{args.dataset}' (split '{args.split}')")
        print(f"Storing model weights in {model.dtype}")

        # Determine the checkpoint path for resume
        resume_path = None
        if args.resume:
            base_path = f"checkpoints/{args.run_name}" if args.run_name else "checkpoints/unnamed"

            # If exact path exists, use it
            if os.path.exists(base_path):
                resume_path = base_path
            else:
                # Try to find the latest checkpoint matching the pattern
                pattern = f"{base_path}_dp*_bs*_ga*_ef*_k*_*"
                matching_paths = sorted(glob(pattern))
                if matching_paths:
                    resume_path = matching_paths[-1]  # Use the latest (sorted by name, which includes timestamp)
                    print(f"Found checkpoint to resume from: {resume_path}")
                else:
                    raise FileNotFoundError(
                        f"No checkpoint found matching pattern: {pattern}\n"
                        f"If resuming, make sure the checkpoint exists or specify the full path."
                    )

        trainer = Trainer(args, dataset, model, resume_from=resume_path)
        if args.resume:
            trainer.load_state(resume_path)
        elif args.finetune:
            for name, sae in trainer.saes.items():
                load_sae_checkpoint(sae, f"{args.finetune}/{name}", device=str(model.device))

        trainer.fit()


if __name__ == "__main__":
    run()
