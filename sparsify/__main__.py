import logging
import os
import json
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
from datetime import timedelta
from glob import glob
from pathlib import Path
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


def _normalize_resume_signature_from_args(args: "RunConfig") -> dict:
    return {
        "sae": {
            "architecture": args.sae.architecture,
            "k": args.sae.k,
            "expansion_factor": args.sae.expansion_factor,
            "trunk_rank": args.sae.trunk_rank,
            "num_codes": args.sae.num_codes,
            "stage1_ratio": args.sae.stage1_ratio,
            "factorized_hidden_dim": args.sae.factorized_hidden_dim,
            "num_experts": args.sae.num_experts,
            "active_experts": args.sae.active_experts,
            "latents_per_expert": args.sae.latents_per_expert,
            "jumprelu_init_threshold": args.sae.jumprelu_init_threshold,
            "jumprelu_bandwidth": args.sae.jumprelu_bandwidth,
            "gated_temperature": args.sae.gated_temperature,
            "gated_init_logit": args.sae.gated_init_logit,
            "group_topk_size": args.sae.group_topk_size,
        },
        "batch_size": args.batch_size,
        "grad_acc_steps": args.grad_acc_steps,
        "micro_acc_steps": args.micro_acc_steps,
        "lr": args.lr,
        "auxk_alpha": args.auxk_alpha,
        "dead_feature_threshold": args.dead_feature_threshold,
        "hookpoints": list(args.hookpoints),
        "init_seeds": list(args.init_seeds),
        "num_tiles": args.num_tiles,
        "global_topk": args.global_topk,
        "input_mixing": args.input_mixing,
        "use_hadamard": args.use_hadamard,
        "compile_model": args.compile_model,
        "optimizer": args.optimizer,
        "matryoshka_ks": list(args.matryoshka_ks),
        "matryoshka_weights": list(args.matryoshka_weights),
        "ortho_lambda": args.ortho_lambda,
        "residual_from": args.residual_from,
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "ctx_len": args.ctx_len,
        "max_examples": args.max_examples,
        "shuffle_seed": args.shuffle_seed,
    }


def _normalize_resume_signature_from_checkpoint(path: str | Path) -> dict:
    with open(Path(path) / "config.json") as f:
        cfg = json.load(f)
    return {
        "sae": {
            "architecture": cfg.get("sae", {}).get("architecture"),
            "k": cfg.get("sae", {}).get("k"),
            "expansion_factor": cfg.get("sae", {}).get("expansion_factor"),
            "trunk_rank": cfg.get("sae", {}).get("trunk_rank"),
            "num_codes": cfg.get("sae", {}).get("num_codes"),
            "stage1_ratio": cfg.get("sae", {}).get("stage1_ratio"),
            "factorized_hidden_dim": cfg.get("sae", {}).get("factorized_hidden_dim"),
            "num_experts": cfg.get("sae", {}).get("num_experts"),
            "active_experts": cfg.get("sae", {}).get("active_experts"),
            "latents_per_expert": cfg.get("sae", {}).get("latents_per_expert"),
            "jumprelu_init_threshold": cfg.get("sae", {}).get("jumprelu_init_threshold"),
            "jumprelu_bandwidth": cfg.get("sae", {}).get("jumprelu_bandwidth"),
            "gated_temperature": cfg.get("sae", {}).get("gated_temperature"),
            "gated_init_logit": cfg.get("sae", {}).get("gated_init_logit"),
            "group_topk_size": cfg.get("sae", {}).get("group_topk_size"),
        },
        "batch_size": cfg.get("batch_size"),
        "grad_acc_steps": cfg.get("grad_acc_steps"),
        "micro_acc_steps": cfg.get("micro_acc_steps"),
        "lr": cfg.get("lr"),
        "auxk_alpha": cfg.get("auxk_alpha"),
        "dead_feature_threshold": cfg.get("dead_feature_threshold"),
        "hookpoints": list(cfg.get("hookpoints", [])),
        "init_seeds": list(cfg.get("init_seeds", [])),
        "num_tiles": cfg.get("num_tiles"),
        "global_topk": cfg.get("global_topk"),
        "input_mixing": cfg.get("input_mixing"),
        "use_hadamard": cfg.get("use_hadamard"),
        "compile_model": cfg.get("compile_model"),
        "optimizer": cfg.get("optimizer"),
        "matryoshka_ks": list(cfg.get("matryoshka_ks", [])),
        "matryoshka_weights": list(cfg.get("matryoshka_weights", [])),
        "ortho_lambda": cfg.get("ortho_lambda"),
        "residual_from": cfg.get("residual_from"),
        "model": cfg.get("model"),
        "dataset": cfg.get("dataset"),
        "split": cfg.get("split"),
        "ctx_len": cfg.get("ctx_len"),
        "max_examples": cfg.get("max_examples"),
        "shuffle_seed": cfg.get("shuffle_seed"),
    }


def _validate_resume_compatibility(args: "RunConfig", resume_path: str) -> None:
    current = _normalize_resume_signature_from_args(args)
    previous = _normalize_resume_signature_from_checkpoint(resume_path)
    if current == previous:
        return

    mismatches = []
    for key in sorted(set(previous) | set(current)):
        if key == "sae":
            for sae_key in sorted(set(previous["sae"]) | set(current["sae"])):
                if previous["sae"].get(sae_key) != current["sae"].get(sae_key):
                    mismatches.append(
                        f"sae.{sae_key}: checkpoint={previous['sae'].get(sae_key)!r} current={current['sae'].get(sae_key)!r}"
                    )
        elif previous.get(key) != current.get(key):
            mismatches.append(
                f"{key}: checkpoint={previous.get(key)!r} current={current.get(key)!r}"
            )
    raise ValueError(
        "Resume config mismatch; only token/logging fields may change between continuation stages:\n"
        + "\n".join(mismatches)
    )


def _is_local_artifact(path: str) -> bool:
    """Best-effort check for paths that can be loaded safely on all ranks in parallel."""
    return os.path.exists(path)


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
    parallel_local_load = _is_local_artifact(args.model) and _is_local_artifact(
        args.dataset
    )

    # Prevent ranks other than 0 from printing
    with nullcontext() if rank == 0 else redirect_stdout(None):
        if not ddp or parallel_local_load or rank == 0:
            model, dataset = load_artifacts(args, rank)
        if ddp:
            if not parallel_local_load:
                dist.barrier()
            if not parallel_local_load and rank != 0:
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
            _validate_resume_compatibility(args, resume_path)
            trainer.load_state(resume_path)
        elif args.finetune:
            for name, sae in trainer.saes.items():
                load_sae_checkpoint(
                    sae, f"{args.finetune}/{name}", device=str(model.device)
                )

        trainer.fit()


if __name__ == "__main__":
    run()
