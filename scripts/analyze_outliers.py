import json
import sys
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path
from typing import Literal

# Ensure repo root is on sys.path when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from simple_parsing import field, list_field, parse
from torch.utils.data import DataLoader

import eval_exceed
from sparsify.utils import get_layer_list, get_max_layer_index, partial_forward_to_layer

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


@dataclass
class OutlierConfig:
    checkpoint: str
    model: str
    dataset: str

    split: str = "train"
    ctx_len: int = 2048
    batch_size: int = 8
    max_batches: int | None = 100
    max_examples: int | None = None

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

    hookpoints: list[str] = list_field()
    hook_mode: Literal["output", "input", "transcode"] | None = None
    partial_forward: bool = True
    apply_hadamard: bool = False

    k: float = 3.0
    quantiles: list[float] = list_field(0.995)
    stats_use_abs: bool = True
    sample_per_batch: int = 512
    max_token_samples: int = 20000
    topk: int = 10
    token_outlier_hist_max: int = 128
    two_pass: bool = True
    out_json: str | None = None
    save_per_dim: bool = True
    plot_dir: str | None = None
    plot_bins: int = 50
    plot_format: str = "png"


class RunningDimStats:
    def __init__(
        self, sample_per_batch: int, max_token_samples: int, use_abs: bool
    ) -> None:
        self.sample_per_batch = sample_per_batch
        self.max_token_samples = max_token_samples
        self.use_abs = use_abs
        self.sum_sq = None
        self.max_val = None
        self.count = 0
        self.samples: list[torch.Tensor] = []

    def update(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return
        x = x.float()
        if self.use_abs:
            x = x.abs()
        if self.sum_sq is None:
            dim = x.shape[-1]
            self.sum_sq = torch.zeros(dim, dtype=torch.float64)
            self.max_val = torch.full((dim,), float("-inf"), dtype=torch.float32)
        self.sum_sq += (x.double() ** 2).sum(dim=0).cpu()
        self.max_val = torch.maximum(self.max_val, x.max(dim=0).values.cpu())
        self.count += x.shape[0]

        if self.sample_per_batch <= 0:
            return
        n = x.shape[0]
        k = min(self.sample_per_batch, n)
        idx = torch.randint(0, n, (k,), device=x.device)
        self.samples.append(x[idx].cpu())

    def finalize_quantiles(self, quantiles: list[float]) -> dict[str, torch.Tensor]:
        if not self.samples:
            return {str(q): torch.tensor([]) for q in quantiles}
        samples = torch.cat(self.samples, dim=0)
        if samples.shape[0] > self.max_token_samples:
            idx = torch.randperm(samples.shape[0])[: self.max_token_samples]
            samples = samples[idx]
        return {str(q): torch.quantile(samples, q, dim=0) for q in quantiles}

    def rms(self) -> torch.Tensor:
        if self.sum_sq is None or self.count == 0:
            return torch.tensor([])
        return torch.sqrt(self.sum_sq / self.count)


class OutlierMaskStats:
    def __init__(self, token_outlier_hist_max: int) -> None:
        self.outlier_count = 0
        self.total_count = 0
        self.token_count = 0
        self.max_outliers_per_token = 0
        self.per_dim_outlier_count = None
        self.token_outlier_hist_max = token_outlier_hist_max
        self.token_outlier_hist = (
            torch.zeros(token_outlier_hist_max + 1, dtype=torch.long)
            if token_outlier_hist_max > 0
            else None
        )

    def update(self, x: torch.Tensor, threshold: torch.Tensor) -> None:
        if x.numel() == 0:
            return
        mask = x.abs() > threshold
        if self.per_dim_outlier_count is None:
            self.per_dim_outlier_count = torch.zeros(mask.shape[-1], dtype=torch.long)
        self.outlier_count += int(mask.sum().item())
        self.total_count += mask.numel()
        self.token_count += mask.shape[0]
        self.per_dim_outlier_count += mask.sum(dim=0).cpu()
        max_per_token = int(mask.sum(dim=-1).max().item()) if mask.shape[0] > 0 else 0
        self.max_outliers_per_token = max(self.max_outliers_per_token, max_per_token)
        if self.token_outlier_hist is not None:
            per_token = mask.sum(dim=-1).cpu()
            per_token = torch.clamp(per_token, max=self.token_outlier_hist_max)
            hist = torch.bincount(
                per_token, minlength=self.token_outlier_hist_max + 1
            )
            self.token_outlier_hist += hist


def _run_pass(
    model: torch.nn.Module,
    dataset,
    hookpoints: list[str],
    hook_mode: str,
    exclude_tokens: list[int],
    partial_forward: bool,
    apply_hadamard: bool,
    hadamard_rotations: dict[str, eval_exceed.HadamardRotation],
    update_fn,
    batch_size: int,
    max_batches: int | None,
) -> None:
    name_to_module = {name: model.base_model.get_submodule(name) for name in hookpoints}
    module_to_name = {v: k for k, v in name_to_module.items()}
    tokens_mask = None

    def hook(module: torch.nn.Module, inputs, outputs):
        nonlocal tokens_mask
        if tokens_mask is None:
            return
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        name = module_to_name[module]
        outputs = outputs.flatten(0, 1)
        inputs = inputs.flatten(0, 1)
        match hook_mode:
            case "output":
                inputs = outputs
            case "input":
                outputs = inputs
            case "transcode":
                pass
            case _:
                raise ValueError(f"Unknown hook_mode: {hook_mode}")

        mask = tokens_mask.flatten(0, 1)
        if mask.sum() == 0:
            return

        outputs = outputs[mask]

        if apply_hadamard and name in hadamard_rotations:
            outputs = hadamard_rotations[name].rotate(outputs)

        update_fn(name, outputs)

    handles = [mod.register_forward_hook(hook) for mod in name_to_module.values()]

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    max_layer_idx = None
    if partial_forward:
        layers_name, _ = get_layer_list(model)
        max_layer_idx = get_max_layer_index(hookpoints, layers_name)

    batch_limit = max_batches if max_batches is not None else len(dl)
    processed_batches = 0

    with torch.inference_mode():
        for batch in dl:
            if processed_batches >= batch_limit:
                break
            x = batch["input_ids"].to(model.device)
            tokens_mask = torch.isin(
                x,
                torch.tensor(exclude_tokens, device=model.device, dtype=torch.long),
                invert=True,
            )
            if max_layer_idx is not None:
                partial_forward_to_layer(model, x, max_layer_idx)
            else:
                model(x)
            processed_batches += 1

    for h in handles:
        h.remove()


def _summarize_per_dim(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {"mean": 0.0, "max": 0.0, "p99": 0.0}
    return {
        "mean": float(values.mean().item()),
        "max": float(values.max().item()),
        "p99": float(torch.quantile(values, 0.99).item()),
    }

def _topk(values: torch.Tensor, k: int) -> list[dict[str, float]]:
    if values.numel() == 0 or k <= 0:
        return []
    k = min(k, values.numel())
    vals, idx = torch.topk(values, k)
    return [{"dim": int(i), "value": float(v)} for v, i in zip(vals.tolist(), idx.tolist())]


def _plot_hist(values: torch.Tensor, path: Path, title: str, bins: int) -> None:
    if values.numel() == 0:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(values.cpu().numpy(), bins=bins)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_scatter(x: torch.Tensor, y: torch.Tensor, path: Path, title: str) -> None:
    if x.numel() == 0 or y.numel() == 0:
        return
    plt.figure(figsize=(6, 4))
    plt.scatter(x.cpu().numpy(), y.cpu().numpy(), s=4, alpha=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    args = parse(OutlierConfig)
    checkpoint = Path(args.checkpoint)
    cfg = eval_exceed.load_train_config(checkpoint)

    hookpoints = args.hookpoints or (cfg.hookpoints if cfg else [])
    if not hookpoints:
        raise ValueError("hookpoints not provided and checkpoint has no config.json")

    hook_mode = args.hook_mode or (cfg.hook_mode if cfg else "output")
    exclude_tokens = cfg.exclude_tokens if cfg else []

    model, dataset = eval_exceed.load_artifacts(args, cfg.loss_fn if cfg else "fvu")
    model.eval()

    hadamard_rotations = {}
    if args.apply_hadamard:
        hadamard_rotations = eval_exceed.load_hadamard_rotations(
            checkpoint, hookpoints, model.device
        )

    dim_stats = {
        name: RunningDimStats(
            args.sample_per_batch, args.max_token_samples, args.stats_use_abs
        )
        for name in hookpoints
    }

    def update_dim_stats(name: str, outputs: torch.Tensor) -> None:
        dim_stats[name].update(outputs)

    _run_pass(
        model=model,
        dataset=dataset,
        hookpoints=hookpoints,
        hook_mode=hook_mode,
        exclude_tokens=exclude_tokens,
        partial_forward=args.partial_forward,
        apply_hadamard=args.apply_hadamard,
        hadamard_rotations=hadamard_rotations,
        update_fn=update_dim_stats,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )

    rms_by_hook = {name: stat.rms() for name, stat in dim_stats.items()}
    quantiles_by_hook = {
        name: stat.finalize_quantiles(args.quantiles) for name, stat in dim_stats.items()
    }

    mask_stats = {
        name: OutlierMaskStats(args.token_outlier_hist_max) for name in hookpoints
    }
    if args.two_pass:
        thresholds = {name: args.k * rms_by_hook[name] for name in hookpoints}

        def update_mask_stats(name: str, outputs: torch.Tensor) -> None:
            if thresholds[name].numel() == 0:
                return
            mask_stats[name].update(outputs, thresholds[name].to(outputs.device))

        _run_pass(
            model=model,
            dataset=dataset,
            hookpoints=hookpoints,
            hook_mode=hook_mode,
            exclude_tokens=exclude_tokens,
            partial_forward=args.partial_forward,
            apply_hadamard=args.apply_hadamard,
            hadamard_rotations=hadamard_rotations,
            update_fn=update_mask_stats,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
        )

    results = {}
    for name in hookpoints:
        stat = dim_stats[name]
        rms = rms_by_hook[name]
        max_val = stat.max_val if stat.max_val is not None else torch.tensor([])
        quantiles = quantiles_by_hook[name]
        entry = {
            "num_tokens": stat.count,
            "dim": int(rms.numel()),
            "stats_use_abs": args.stats_use_abs,
            "rms_summary": _summarize_per_dim(rms),
            "max_summary": _summarize_per_dim(max_val),
            "quantile_summary": {
                q: _summarize_per_dim(v) for q, v in quantiles.items()
            },
            "topk": {
                "rms": _topk(rms, args.topk),
                "max": _topk(max_val, args.topk),
                **{
                    f"q{q}": _topk(v, args.topk)
                    for q, v in quantiles.items()
                },
            },
        }

        if args.save_per_dim:
            entry["rms"] = rms.tolist()
            entry["max"] = max_val.tolist()
            entry["quantiles"] = {q: v.tolist() for q, v in quantiles.items()}

        if args.two_pass:
            mstat = mask_stats[name]
            per_dim_freq = (
                (mstat.per_dim_outlier_count.float() / max(mstat.token_count, 1))
                if mstat.per_dim_outlier_count is not None
                else torch.tensor([])
            )
            entry["outlier"] = {
                "k": args.k,
                "ratio": mstat.outlier_count / max(mstat.total_count, 1),
                "max_outliers_per_token": mstat.max_outliers_per_token,
                "per_dim_freq_summary": _summarize_per_dim(per_dim_freq),
            }
            if args.save_per_dim:
                entry["outlier"]["per_dim_freq"] = per_dim_freq.tolist()
            if mstat.token_outlier_hist is not None:
                entry["outlier"]["per_token_hist"] = {
                    "max_bucket": args.token_outlier_hist_max,
                    "counts": mstat.token_outlier_hist.tolist(),
                }
            entry["topk"]["outlier_freq"] = _topk(per_dim_freq, args.topk)

        results[name] = entry

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)

    if args.plot_dir:
        plot_path = Path(args.plot_dir)
        plot_path.mkdir(parents=True, exist_ok=True)
        if not HAS_MPL:
            print("WARNING: matplotlib not installed; skipping plots.")
        else:
            for name in hookpoints:
                entry = results[name]
                prefix = name.replace("/", "_").replace(".", "_")
                rms = torch.tensor(entry["rms"]) if args.save_per_dim else rms_by_hook[name]
                max_val = torch.tensor(entry["max"]) if args.save_per_dim else dim_stats[name].max_val
                _plot_hist(
                    rms,
                    plot_path / f"{prefix}_rms.{args.plot_format}",
                    f"{name} RMS",
                    args.plot_bins,
                )
                _plot_hist(
                    max_val,
                    plot_path / f"{prefix}_max.{args.plot_format}",
                    f"{name} MAX",
                    args.plot_bins,
                )
                _plot_scatter(
                    rms,
                    max_val,
                    plot_path / f"{prefix}_rms_vs_max.{args.plot_format}",
                    f"{name} RMS vs MAX",
                )
                for q_str, arr in entry["quantiles"].items() if args.save_per_dim else quantiles_by_hook[name].items():
                    q_vals = torch.tensor(arr) if args.save_per_dim else arr
                    _plot_hist(
                        q_vals,
                        plot_path / f"{prefix}_q{q_str}.{args.plot_format}",
                        f"{name} q{q_str}",
                        args.plot_bins,
                    )
                outlier = entry.get("outlier", {})
                if outlier.get("per_dim_freq") is not None:
                    per_dim = torch.tensor(outlier["per_dim_freq"])
                    _plot_hist(
                        per_dim,
                        plot_path / f"{prefix}_outlier_freq.{args.plot_format}",
                        f"{name} Outlier Freq",
                        args.plot_bins,
                    )
                if outlier.get("per_token_hist") is not None:
                    hist = outlier["per_token_hist"]["counts"]
                    plt.figure(figsize=(6, 4))
                    plt.bar(range(len(hist)), hist)
                    plt.title(f"{name} Per-token outlier count")
                    plt.tight_layout()
                    plt.savefig(
                        plot_path / f"{prefix}_outlier_per_token.{args.plot_format}"
                    )
                    plt.close()

    for name in hookpoints:
        entry = results[name]
        outlier = entry.get("outlier", {})
        print(f"[{name}] tokens={entry['num_tokens']} dim={entry['dim']}")
        print(f"  rms: mean={entry['rms_summary']['mean']:.6f} max={entry['rms_summary']['max']:.6f}")
        print(f"  max: mean={entry['max_summary']['mean']:.6f} max={entry['max_summary']['max']:.6f}")
        for q, stats in entry["quantile_summary"].items():
            print(f"  q{q}: mean={stats['mean']:.6f} max={stats['max']:.6f}")
        if outlier:
            print(
                "  outlier(k={k}): ratio={ratio:.6f} max_per_token={max_per_token}".format(
                    k=outlier.get("k", 0.0),
                    ratio=outlier.get("ratio", 0.0),
                    max_per_token=outlier.get("max_outliers_per_token", 0),
                )
            )


if __name__ == "__main__":
    main()
