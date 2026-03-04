import logging
import os
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
from datetime import timedelta
from glob import glob
from multiprocessing import cpu_count

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from simple_parsing import field, parse
from transformers import AutoModel, AutoTokenizer, PreTrainedModel

from .checkpoint import load_sae_checkpoint
from .data import MemmapDataset, chunk_and_tokenize
from .device import (
    get_device,
    get_device_string,
    get_device_type,
    get_dist_backend,
    is_bf16_supported,
    set_device,
)
from .trainer import Trainer, TrainConfig
from .utils import simple_parse_args_string

logger = logging.getLogger(__name__)


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

    hf_token: str | None = field(default=None, encoding_fn=lambda _: None)
    """Huggingface API token for downloading models."""

    revision: str | None = None
    """Model revision to use for training."""

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
    dtype = torch.bfloat16 if is_bf16_supported() else "auto"

    model = AutoModel.from_pretrained(
        args.model,
        device_map={"": get_device_string(rank)},
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
            logger.info("Dataset already tokenized; skipping tokenization.")

        logger.info(f"Shuffling dataset with seed {args.shuffle_seed}")
        dataset = dataset.shuffle(args.shuffle_seed)

        dataset = dataset.with_format("torch")
        if limit := args.max_examples:
            dataset = dataset.select(range(limit))

    return model, dataset


def run():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    local_rank = os.environ.get("LOCAL_RANK")
    ddp = local_rank is not None
    rank = int(local_rank) if ddp else 0

    if ddp:
        set_device(int(local_rank))

        dist.init_process_group(
            get_dist_backend(),
            device_id=get_device(rank),
            timeout=timedelta(weeks=1),
        )

        if rank == 0:
            logger.info(f"Using DDP across {dist.get_world_size()} GPUs.")

    args = parse(RunConfig)

    # Prevent ranks other than 0 from printing
    with nullcontext() if rank == 0 else redirect_stdout(None):
        if not ddp or rank == 0:
            model, dataset = load_artifacts(args, rank)
        if ddp:
            dist.barrier()
            if rank != 0:
                model, dataset = load_artifacts(args, rank)

            # Drop examples indivisible across processes to prevent deadlock
            remainder_examples = len(dataset) % dist.get_world_size()
            dataset = dataset.select(range(len(dataset) - remainder_examples))

            dataset = dataset.shard(dist.get_world_size(), rank)

            remainder_examples = len(dataset) % dist.get_world_size()
            dataset = dataset.select(range(len(dataset) - remainder_examples))

        logger.info(f"Training on '{args.dataset}' (split '{args.split}')")
        logger.info(f"Storing model weights in {model.dtype}")

        # Determine the checkpoint path for resume
        resume_path = None
        if args.resume:
            base_path = (
                f"{args.save_dir}/{args.run_name}"
                if args.run_name
                else f"{args.save_dir}/unnamed"
            )

            if os.path.exists(base_path):
                resume_path = base_path
            else:
                pattern = f"{base_path}_dp*_bs*_ga*_ef*_k*_*"
                matching_paths = sorted(glob(pattern))
                if matching_paths:
                    resume_path = matching_paths[-1]
                    logger.info(f"Found checkpoint to resume from: {resume_path}")
                else:
                    raise FileNotFoundError(
                        f"No checkpoint found matching pattern: {pattern}\n"
                        f"If resuming, make sure the checkpoint exists or "
                        f"specify the full path."
                    )

        trainer = Trainer(args, dataset, model, resume_from=resume_path)
        if args.resume:
            trainer.load_state(resume_path)
        elif args.finetune:
            for name, sae in trainer.saes.items():
                load_sae_checkpoint(
                    sae, f"{args.finetune}/{name}", device=str(model.device)
                )

        trainer.fit()


if __name__ == "__main__":
    run()
