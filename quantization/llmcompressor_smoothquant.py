from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk


@dataclass(frozen=True)
class SmoothQuantRecipeConfig:
    smoothing_strength: float = 0.8
    quant_scheme: str = "W8A8"
    quant_targets: str = "Linear"
    ignore: tuple[str, ...] = ("lm_head",)


def _select_calibration_split(dataset: Dataset | DatasetDict) -> Dataset:
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            return dataset["train"]
        return dataset[next(iter(dataset.keys()))]
    return dataset


def prepare_tokenized_calibration_dataset(
    dataset_path: str,
    *,
    num_samples: int,
    max_seq_length: int,
    shuffle_seed: int = 42,
) -> Dataset:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if max_seq_length <= 0:
        raise ValueError("max_seq_length must be positive")

    dataset = _select_calibration_split(load_from_disk(dataset_path))
    if "input_ids" not in dataset.column_names:
        raise ValueError(
            f"Calibration dataset at {dataset_path!r} must contain an 'input_ids' column"
        )

    sample_count = min(num_samples, len(dataset))
    dataset = dataset.shuffle(seed=shuffle_seed).select(range(sample_count))
    rows: list[list[int]] = []
    for example in dataset:
        trimmed = list(example["input_ids"][:max_seq_length])
        if trimmed:
            rows.append(trimmed)

    return Dataset.from_dict({"input_ids": rows})


def import_llmcompressor_symbols():
    try:
        from llmcompressor.entrypoints import oneshot
        from llmcompressor.modifiers.quantization import GPTQModifier
        from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
    except ImportError as exc:
        raise ImportError(
            "llmcompressor is required for SmoothQuant export. "
            "Install it with `pip install llmcompressor compressed-tensors`."
        ) from exc

    return oneshot, GPTQModifier, SmoothQuantModifier


def build_smoothquant_w8a8_recipe(
    cfg: SmoothQuantRecipeConfig,
    *,
    smoothquant_modifier_cls=None,
    gptq_modifier_cls=None,
) -> list[object]:
    if smoothquant_modifier_cls is None or gptq_modifier_cls is None:
        _, gptq_modifier_cls, smoothquant_modifier_cls = import_llmcompressor_symbols()

    return [
        smoothquant_modifier_cls(smoothing_strength=cfg.smoothing_strength),
        gptq_modifier_cls(
            targets=cfg.quant_targets,
            scheme=cfg.quant_scheme,
            ignore=list(cfg.ignore),
        ),
    ]


def write_smoothquant_export_manifest(
    output_dir: str | Path,
    *,
    model_path: str,
    dataset_path: str,
    calibration_size: int,
    max_seq_length: int,
    recipe_cfg: SmoothQuantRecipeConfig,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "smoothquant_export_manifest.json"
    manifest = {
        "model_path": model_path,
        "dataset_path": dataset_path,
        "calibration_size": calibration_size,
        "max_seq_length": max_seq_length,
        "recipe": asdict(recipe_cfg),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest_path
