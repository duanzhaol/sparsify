import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path
from typing import Literal

# Ensure repo root is on sys.path when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from datasets import Dataset, load_dataset
from simple_parsing import field, list_field, parse
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from sparsify.config import TrainConfig
from sparsify.data import MemmapDataset, chunk_and_tokenize
from sparsify.hadamard import HadamardRotation
from sparsify.outlier_clip import OutlierClipper
from sparsify.sparse_coder import SparseCoder
from sparsify.tiled_sparse_coder import TiledSparseCoder
from sparsify.eval.two_stage import TwoStageConfig
from sparsify.eval.encoders import build_encoder_strategies
from sparsify.eval.pca import load_pca_bundle
from sparsify.utils import (
    get_layer_list,
    get_max_layer_index,
    partial_forward_to_layer,
    simple_parse_args_string,
)
from lowrank_encoder import LowRankSparseCoder


@dataclass
class EvalConfig:
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
    exceed_alphas: list[float] = list_field()
    elbow_threshold_path: str | None = None

    out_json: str | None = None
    track_topk: bool = True
    track_latent_counts: bool = False
    partial_forward: bool = True
    encoder_mode: Literal["full", "two_stage"] = "full"
    two_stage_dim: int = 128
    two_stage_k: int = 1000
    two_stage_proj: Literal["slice", "random", "pca"] = "slice"
    two_stage_seed: int = 0
    two_stage_pca_path: str | None = None
    two_stage_pca_mean_path: str | None = None


class RunningStat:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0
        self.max_val = None

    def update(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return
        self.total += float(x.sum())
        self.count += x.numel()
        current_max = float(x.max())
        self.max_val = current_max if self.max_val is None else max(self.max_val, current_max)

    def to_dict(self) -> dict[str, float]:
        mean = self.total / self.count if self.count else 0.0
        max_val = self.max_val if self.max_val is not None else 0.0
        return {"mean": mean, "max": max_val, "count": float(self.count)}


def load_train_config(checkpoint: Path) -> TrainConfig | None:
    cfg_path = checkpoint / "config.json"
    if not cfg_path.exists():
        return None
    with open(cfg_path, "r") as f:
        data = json.load(f)
    return TrainConfig.from_dict(data, drop_extra_fields=True)


def load_elbow_thresholds(path: str, hookpoints: list[str]) -> dict[str, float]:
    with open(path, "r") as f:
        elbow_data = json.load(f)

    thresholds: dict[str, float] = {}
    for hookpoint in hookpoints:
        matched = False

        if hookpoint in elbow_data:
            thresholds[hookpoint] = elbow_data[hookpoint]["elbow_value"]
            matched = True
            continue

        parts = hookpoint.split(".")
        if len(parts) >= 2 and parts[0] in ("layers", "h", "model.layers"):
            layer_num = parts[1]
            component = ".".join(parts[2:]) if len(parts) > 2 else ""

            search_patterns = []
            if component:
                component_underscore = component.replace(".", "_")
                search_patterns.extend(
                    [
                        f"layer_{layer_num}/{component}",
                        f"layer_{layer_num}/{component_underscore}",
                    ]
                )
            search_patterns.append(f"layer_{layer_num}")

            for pattern in search_patterns:
                for json_key, value in elbow_data.items():
                    if pattern in json_key or json_key in hookpoint:
                        thresholds[hookpoint] = value["elbow_value"]
                        matched = True
                        break
                if matched:
                    break

        if not matched:
            print(f"WARNING: No elbow threshold found for hookpoint '{hookpoint}'")

    print(f"Loaded elbow thresholds for {len(thresholds)}/{len(hookpoints)} hookpoints")
    return thresholds


def load_artifacts(args: EvalConfig, loss_fn: str) -> tuple[torch.nn.Module, Dataset | MemmapDataset]:
    if args.load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = "auto"

    model_cls = AutoModel if loss_fn == "fvu" else AutoModelForCausalLM
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_cls.from_pretrained(
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
        try:
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
        if args.max_examples:
            dataset = dataset.select(range(args.max_examples))

    return model, dataset


def load_sae_from_disk(path: Path, device: str | torch.device):
    with open(path / "cfg.json", "r") as f:
        cfg = json.load(f)
    num_tiles = cfg.get("num_tiles", 1)
    encoder_rank = cfg.get("encoder_rank", 0)

    if num_tiles > 1:
        return TiledSparseCoder.load_from_disk(path, device=device)
    if encoder_rank > 0:
        return LowRankSparseCoder.load_from_disk(path, device=device)
    return SparseCoder.load_from_disk(path, device=device)


def load_hadamard_rotations(
    checkpoint: Path, hookpoints: list[str], device: str | torch.device
) -> dict[str, HadamardRotation]:
    """Load Hadamard rotation states from checkpoint if they exist."""
    hadamard_path = checkpoint / "hadamard_rotations.pt"
    if not hadamard_path.exists():
        return {}

    hadamard_states = torch.load(hadamard_path, map_location=device, weights_only=False)
    rotations = {}
    for name, state in hadamard_states.items():
        if name in hookpoints:
            rotations[name] = HadamardRotation.from_state_dict(state, device=device)
    print(f"Loaded Hadamard rotations for {len(rotations)} hookpoints")
    return rotations


def load_outlier_clippers(
    checkpoint: Path, hookpoints: list[str], device: str | torch.device
) -> dict[str, OutlierClipper]:
    """Load outlier clipper states from checkpoint if they exist."""
    clipper_path = checkpoint / "outlier_clippers.pt"
    if not clipper_path.exists():
        return {}

    clipper_states = torch.load(clipper_path, map_location=device, weights_only=False)
    clippers = {}
    for name, state in clipper_states.items():
        if name in hookpoints:
            clippers[name] = OutlierClipper.from_state_dict(state, device=device)
    print(f"Loaded outlier clippers for {len(clippers)} hookpoints")
    return clippers


from lowrank_encoder import LowRankSparseCoder


def main() -> None:
    args = parse(EvalConfig)
    checkpoint = Path(args.checkpoint)
    cfg = load_train_config(checkpoint)

    hookpoints = args.hookpoints or (cfg.hookpoints if cfg else [])
    if not hookpoints:
        raise ValueError("hookpoints not provided and checkpoint has no config.json")

    hook_mode = args.hook_mode or (cfg.hook_mode if cfg else "output")
    exceed_alphas = args.exceed_alphas or (cfg.exceed_alphas if cfg else [])
    elbow_path = args.elbow_threshold_path or (cfg.elbow_threshold_path if cfg else None)
    exclude_tokens = cfg.exclude_tokens if cfg else []
    init_seeds = cfg.init_seeds if cfg else [0]
    is_distill = bool(cfg and cfg.distill_from)

    model, dataset = load_artifacts(args, cfg.loss_fn if cfg else "fvu")
    device = model.device
    model.eval()

    if elbow_path:
        elbow_thresholds = load_elbow_thresholds(elbow_path, hookpoints)
    else:
        elbow_thresholds = {}

    if (checkpoint / "cfg.json").exists() and not (checkpoint / "config.json").exists():
        if len(hookpoints) != 1:
            raise ValueError("Single SAE checkpoint requires exactly one hookpoint")
        sae_keys = [hookpoints[0]]
        sae_paths = {sae_keys[0]: checkpoint}
        hook_to_sae_keys = {hookpoints[0]: sae_keys}
    else:
        sae_keys = []
        hook_to_sae_keys = defaultdict(list)
        for hook in hookpoints:
            if is_distill or len(init_seeds) <= 1:
                key = hook
                sae_keys.append(key)
                hook_to_sae_keys[hook].append(key)
            else:
                for seed in init_seeds:
                    key = f"{hook}/seed{seed}"
                    sae_keys.append(key)
                    hook_to_sae_keys[hook].append(key)

        sae_paths = {key: checkpoint / key for key in sae_keys}

    saes = {}
    for key, path in sae_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing SAE checkpoint: {path}")
        sae = load_sae_from_disk(path, device=device)
        sae.eval()
        sae.requires_grad_(False)
        saes[key] = sae

    # Load Hadamard rotations if they exist in checkpoint
    hadamard_rotations = load_hadamard_rotations(checkpoint, hookpoints, device)

    # Load outlier clippers if they exist in checkpoint
    outlier_clippers = load_outlier_clippers(checkpoint, hookpoints, device)
    # Get outlier loss mode from config (default to "weighted" for backwards compatibility)
    outlier_loss_mode = cfg.outlier_loss_mode if cfg and hasattr(cfg, 'outlier_loss_mode') else "weighted"

    pca_bundle = None
    if args.encoder_mode == "two_stage" and args.two_stage_proj == "pca":
        cache_path = checkpoint / "two_stage_pca.pt"
        pca_path = args.two_stage_pca_path
        if not pca_path:
            if cache_path.exists():
                pca_path = str(cache_path)
            else:
                raise ValueError(
                    "two_stage_pca_path is required for projection='pca' "
                    "when no cached PCA exists at checkpoint/two_stage_pca.pt"
                )
        pca_bundle = load_pca_bundle(pca_path)
        if Path(pca_path).resolve() != cache_path.resolve():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pca_bundle, cache_path)

    two_stage_cfg = TwoStageConfig(
        low_dim=args.two_stage_dim,
        k_coarse=args.two_stage_k,
        projection=args.two_stage_proj,
        seed=args.two_stage_seed,
    )
    encoder_strategies = build_encoder_strategies(
        saes,
        args.encoder_mode,
        two_stage_cfg=two_stage_cfg,
        pca_bundle=pca_bundle,
        pca_mean_path=args.two_stage_pca_mean_path,
    )

    name_to_module = {
        name: model.base_model.get_submodule(name)
        for name in hookpoints
    }
    module_to_name = {v: k for k, v in name_to_module.items()}

    exceed_counts: dict[str, dict[float, float]] = defaultdict(lambda: defaultdict(float))
    exceed_denoms: dict[str, dict[float, float]] = defaultdict(lambda: defaultdict(float))
    topk_stats: dict[str, RunningStat] = defaultdict(RunningStat)
    latent_counts: dict[str, torch.Tensor] = {}

    for key, sae in saes.items():
        if args.track_latent_counts:
            latent_counts[key] = torch.zeros(
                sae.num_latents, device=device, dtype=torch.long
            )

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
        hook_name = name.partition("/")[0]

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
        inputs = inputs[mask]

        # Apply Hadamard rotation if available for this hookpoint
        if name in hadamard_rotations:
            rotation = hadamard_rotations[name]
            outputs = rotation.rotate(outputs)
            inputs = rotation.rotate(inputs)

        # Apply outlier clipping if available for this hookpoint
        outlier_residual = None
        outlier_mask = None
        original_outputs = None
        if name in outlier_clippers:
            clipper = outlier_clippers[name]
            # Store original outputs for exceed calculation
            original_outputs = outputs.detach().clone()
            # Clip outputs (no stats update during inference)
            outputs, outlier_residual, outlier_mask = clipper.clip(outputs, update_stats=False)
            # Clip inputs with same threshold
            inputs, _, _ = clipper.clip(inputs, update_stats=False)

        # Pre-compute unrotated target for exceed metrics (avoid repeated unrotate in loop)
        original_target_for_exceed = None
        if name in hadamard_rotations and hook_name in elbow_thresholds:
            original_target_for_exceed = hadamard_rotations[name].unrotate(outputs)

        for sae_key in hook_to_sae_keys[hook_name]:
            out = encoder_strategies[sae_key].forward(inputs, outputs)

            if args.track_topk:
                topk_stats[sae_key].update(out.latent_acts)

            if args.track_latent_counts:
                counts = torch.bincount(
                    out.latent_indices.flatten(),
                    minlength=sae.num_latents,
                )
                latent_counts[sae_key] += counts

            if hook_name in elbow_thresholds and exceed_alphas:
                # CRITICAL: Compute exceed in ORIGINAL space
                if outlier_residual is not None:
                    # Full reconstruction strategy depends on loss mode:
                    # - weighted: SAE output on outlier dims is unconstrained, must zero out
                    # - inlier_only: SAE is encouraged to output ~0 on outlier dims, keep it
                    if outlier_loss_mode == "weighted":
                        original_recon = out.sae_out * (1 - outlier_mask) + outlier_residual
                    else:  # inlier_only
                        original_recon = out.sae_out + outlier_residual
                    # Original target = original_outputs (before clipping)
                    original_target = original_outputs
                elif name in hadamard_rotations:
                    # Use pre-computed unrotated target
                    original_target = original_target_for_exceed if original_target_for_exceed is not None else outputs
                    original_recon = hadamard_rotations[name].unrotate(out.sae_out)
                else:
                    original_target = outputs
                    original_recon = out.sae_out

                error_magnitude = torch.abs(original_target - original_recon)
                num_elements = float(error_magnitude.numel())
                if num_elements == 0:
                    continue
                for alpha in exceed_alphas:
                    threshold = alpha * elbow_thresholds[hook_name]
                    exceed_count = float((error_magnitude > threshold).sum())
                    exceed_counts[sae_key][alpha] += exceed_count
                    exceed_denoms[sae_key][alpha] += num_elements

    handles = [mod.register_forward_hook(hook) for mod in name_to_module.values()]

    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    max_layer_idx = None
    if args.partial_forward:
        layers_name, _ = get_layer_list(model)
        max_layer_idx = get_max_layer_index(hookpoints, layers_name)

    batch_limit = args.max_batches if args.max_batches is not None else len(dl)
    processed_batches = 0

    with torch.inference_mode():
        for batch in dl:
            x = batch["input_ids"].to(device)
            tokens_mask = torch.isin(
                x,
                torch.tensor(exclude_tokens, device=device, dtype=torch.long),
                invert=True,
            )

            if max_layer_idx is not None:
                partial_forward_to_layer(model, x, max_layer_idx)
            else:
                model(x)

            processed_batches += 1
            if processed_batches >= batch_limit:
                break

    for handle in handles:
        handle.remove()

    results = {
        "checkpoint": str(checkpoint),
        "model": args.model,
        "dataset": args.dataset,
        "num_batches": processed_batches,
        "encoder_mode": args.encoder_mode,
        "exceed": {},
        "topk": {},
    }
    if args.encoder_mode == "two_stage":
        results["two_stage"] = {
            "low_dim": args.two_stage_dim,
            "k_coarse": args.two_stage_k,
            "projection": args.two_stage_proj,
            "seed": args.two_stage_seed,
            "pca_path": args.two_stage_pca_path,
            "pca_mean_path": args.two_stage_pca_mean_path,
        }

    for sae_key in sae_keys:
        if exceed_alphas:
            results["exceed"][sae_key] = {
                f"{alpha:.4f}": (
                    exceed_counts[sae_key][alpha] / exceed_denoms[sae_key][alpha]
                    if exceed_denoms[sae_key][alpha] > 0
                    else 0.0
                )
                for alpha in exceed_alphas
            }

        if args.track_topk:
            results["topk"][sae_key] = topk_stats[sae_key].to_dict()

    if args.track_latent_counts:
        counts_path = checkpoint / "eval_latent_counts.pt"
        cpu_counts = {k: v.cpu() for k, v in latent_counts.items()}
        torch.save(cpu_counts, counts_path)
        results["latent_counts_path"] = str(counts_path)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print("Exceed metrics:")
    for sae_key, alpha_vals in results["exceed"].items():
        print(f"  {sae_key}")
        for alpha, val in alpha_vals.items():
            print(f"    alpha={alpha}: {val:.6f}")

    if args.track_topk:
        print("Top-k activation stats:")
        for sae_key, stats in results["topk"].items():
            print(
                f"  {sae_key} mean={stats['mean']:.6f} max={stats['max']:.6f} count={int(stats['count'])}"
            )


if __name__ == "__main__":
    main()
