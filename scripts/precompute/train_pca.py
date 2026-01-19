import json
import sys
from dataclasses import dataclass
from fnmatch import fnmatchcase
from multiprocessing import cpu_count
from pathlib import Path
from typing import Literal

# Ensure repo root is on sys.path when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from datasets import Dataset, load_dataset
from natsort import natsorted
from simple_parsing import field, list_field, parse
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from sparsify.data import MemmapDataset, chunk_and_tokenize
from sparsify.utils import (
    get_layer_list,
    get_max_layer_index,
    partial_forward_to_layer,
    resolve_widths,
    simple_parse_args_string,
)


@dataclass
class PcaConfig:
    model: str
    dataset: str
    hookpoints: list[str] = list_field()

    split: str = "train"
    ctx_len: int = 2048
    batch_size: int = 8
    max_examples: int | None = None
    max_tokens: int = 1_000_000
    max_batches: int | None = None

    hook_mode: Literal["output", "input", "transcode"] = "output"
    exclude_tokens: list[int] = list_field()

    low_dim: int = 128
    out: str = "two_stage_pca.pt"

    text_column: str = "text"
    shuffle_seed: int = 42
    data_preprocessing_num_proc: int = field(
        default_factory=lambda: max(cpu_count() // 2, 1),
    )
    data_args: str = field(default="")

    hf_token: str | None = field(default=None, encoding_fn=lambda _: None)
    revision: str | None = None
    load_in_8bit: bool = False
    device: str | None = None

    partial_forward: bool = True


def expand_range_pattern(pattern: str) -> list[str]:
    import re

    match = re.search(r"\[([0-9,\-]+)\]", pattern)
    if not match:
        return [pattern]

    range_spec = match.group(1)
    numbers = []

    for part in range_spec.split(","):
        if "-" in part:
            start, end = map(int, part.split("-"))
            numbers.extend(range(start, end + 1))
        else:
            numbers.append(int(part))

    numbers = sorted(set(numbers))
    return [pattern.replace(f"[{range_spec}]", str(num)) for num in numbers]


def resolve_hookpoints(model, patterns: list[str]) -> list[str]:
    expanded = []
    for pattern in patterns:
        expanded.extend(expand_range_pattern(pattern))

    raw = []
    for name, _ in model.base_model.named_modules():
        if any(fnmatchcase(name, pat) for pat in expanded):
            raw.append(name)

    return natsorted(raw)


def load_artifacts(args: PcaConfig):
    if args.load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = "auto"

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(
        args.model,
        device_map={"": device},
        quantization_config=(
            BitsAndBytesConfig(load_in_8bit=args.load_in_8bit) if args.load_in_8bit else None
        ),
        revision=args.revision,
        torch_dtype=dtype,
        token=args.hf_token,
    )

    if args.dataset.endswith(".bin"):
        dataset = MemmapDataset(args.dataset, args.ctx_len, args.max_examples)
    else:
        kwargs = simple_parse_args_string(args.data_args)
        dataset = load_dataset(args.dataset, split=args.split, **kwargs)
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
        if args.max_examples:
            dataset = dataset.select(range(args.max_examples))

    return model, dataset


def _safe_name(hookpoint: str) -> str:
    return hookpoint.replace("/", "_").replace(".", "_")


def _save_single(out_path: Path, hookpoint: str, matrix: torch.Tensor, mean: torch.Tensor, count: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "hookpoint": hookpoint,
        "pca_matrix": matrix.cpu(),
        "pca_mean": mean.cpu(),
        "num_tokens": count,
    }
    torch.save(payload, out_path)


def _save_multi(out_path: Path, payload: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)


def main() -> None:
    args = parse(PcaConfig)
    model, dataset = load_artifacts(args)
    device = model.device
    model.eval()

    if not args.hookpoints:
        raise ValueError("hookpoints must be provided")

    hookpoints = resolve_hookpoints(model, args.hookpoints)
    if not hookpoints:
        raise ValueError("No hookpoints matched the provided patterns")

    hook_mode_for_width = "input" if args.hook_mode == "input" else "output"
    widths = resolve_widths(model, hookpoints, hook_mode=hook_mode_for_width)
    for hook, width in widths.items():
        if args.low_dim > width:
            raise ValueError(f"low_dim ({args.low_dim}) > width ({width}) for {hook}")

    sums = {h: torch.zeros(widths[h], device=device, dtype=torch.float64) for h in hookpoints}
    sums_sq = {h: torch.zeros(widths[h], widths[h], device=device, dtype=torch.float64) for h in hookpoints}
    counts = {h: 0 for h in hookpoints}
    done = set()

    exclude_tokens = torch.tensor(args.exclude_tokens, device=device, dtype=torch.long)
    tokens_mask = None

    name_to_module = {name: model.base_model.get_submodule(name) for name in hookpoints}
    module_to_name = {v: k for k, v in name_to_module.items()}

    def hook(module, inputs, outputs):
        nonlocal tokens_mask
        if tokens_mask is None:
            return

        if isinstance(inputs, tuple):
            inputs = inputs[0]
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        hook_name = module_to_name[module]
        if hook_name in done:
            return

        outputs = outputs.flatten(0, 1)
        inputs = inputs.flatten(0, 1)
        match args.hook_mode:
            case "output":
                inputs = outputs
            case "input":
                outputs = inputs
            case "transcode":
                pass
            case _:
                raise ValueError(f"Unknown hook_mode: {args.hook_mode}")

        mask = tokens_mask.flatten(0, 1)
        if mask.sum() == 0:
            return

        x = inputs[mask]
        remaining = args.max_tokens - counts[hook_name]
        if remaining <= 0:
            done.add(hook_name)
            return
        if x.shape[0] > remaining:
            x = x[:remaining]

        x = x.to(torch.float64)
        sums[hook_name] += x.sum(0)
        sums_sq[hook_name] += x.T @ x
        counts[hook_name] += x.shape[0]
        if counts[hook_name] >= args.max_tokens:
            done.add(hook_name)

    handles = [mod.register_forward_hook(hook) for mod in name_to_module.values()]
    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    max_layer_idx = None
    if args.partial_forward:
        layers_name, _ = get_layer_list(model)
        max_layer_idx = get_max_layer_index(hookpoints, layers_name)

    processed_batches = 0
    batch_limit = args.max_batches if args.max_batches is not None else len(dl)
    report_every = 50

    with torch.inference_mode():
        for batch in dl:
            if len(done) == len(hookpoints):
                break
            x = batch["input_ids"].to(device)
            tokens_mask = torch.isin(x, exclude_tokens, invert=True)

            if max_layer_idx is not None:
                partial_forward_to_layer(model, x, max_layer_idx)
            else:
                model(x)

            processed_batches += 1
            if processed_batches % report_every == 0 or processed_batches == 1:
                progress = " ".join(
                    f"{hook}={counts[hook]}/{args.max_tokens}"
                    for hook in hookpoints
                )
                print(f"Progress: batches={processed_batches} {progress}")
            if processed_batches >= batch_limit:
                break

    for handle in handles:
        handle.remove()

    results = {}
    for hook in hookpoints:
        if counts[hook] == 0:
            raise ValueError(f"No tokens collected for hookpoint {hook}")
        mean = sums[hook] / counts[hook]
        cov = (sums_sq[hook] / counts[hook]) - torch.outer(mean, mean)
        evals, evecs = torch.linalg.eigh(cov)
        idx = torch.argsort(evals, descending=True)
        evals = evals[idx]
        evecs = evecs[:, idx][:, : args.low_dim]
        total_var = float(evals.sum())
        top_var = float(evals[: args.low_dim].sum())
        evr = top_var / total_var if total_var > 0 else 0.0

        results[hook] = {
            "pca_matrix": evecs.float().cpu(),
            "pca_mean": mean.float().cpu(),
            "num_tokens": counts[hook],
            "explained_variance_ratio": evr,
            "total_variance": total_var,
        }

    out_path = Path(args.out)
    if len(hookpoints) == 1:
        hook = hookpoints[0]
        if out_path.suffix:
            _save_single(out_path, hook, results[hook]["pca_matrix"], results[hook]["pca_mean"], results[hook]["num_tokens"])
            print(f"Saved PCA to {out_path}")
        else:
            out_path.mkdir(parents=True, exist_ok=True)
            file_path = out_path / f"{_safe_name(hook)}_pca.pt"
            _save_single(file_path, hook, results[hook]["pca_matrix"], results[hook]["pca_mean"], results[hook]["num_tokens"])
            print(f"Saved PCA to {file_path}")
    else:
        if out_path.suffix:
            _save_multi(out_path, results)
            print(f"Saved PCA dict to {out_path}")
        else:
            out_path.mkdir(parents=True, exist_ok=True)
            for hook, payload in results.items():
                file_path = out_path / f"{_safe_name(hook)}_pca.pt"
                _save_single(file_path, hook, payload["pca_matrix"], payload["pca_mean"], payload["num_tokens"])
                print(f"Saved PCA to {file_path}")

    print("PCA summary:")
    for hook, payload in results.items():
        print(
            f"  {hook} tokens={payload['num_tokens']} "
            f"evr@{args.low_dim}={payload['explained_variance_ratio']:.6f} "
            f"total_var={payload['total_variance']:.6f}"
        )


if __name__ == "__main__":
    main()
