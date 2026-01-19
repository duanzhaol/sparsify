from pathlib import Path

import torch


def _select_tensor(obj, keys: list[str], label: str) -> torch.Tensor:
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        for key in keys:
            if key in obj:
                return obj[key]
        if len(obj) == 1:
            return next(iter(obj.values()))
    raise ValueError(f"Could not extract {label} tensor from {type(obj)}")


def _maybe_select_tensor(obj, keys: list[str]) -> torch.Tensor | None:
    if isinstance(obj, dict):
        for key in keys:
            if key in obj:
                return obj[key]
    return None


def load_pca_bundle(path: str | Path):
    p = Path(path)
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(p))
    return torch.load(p, map_location="cpu")


def _maybe_get_bundle_entry(bundle, hookpoint: str):
    if not isinstance(bundle, dict):
        return None
    if hookpoint in bundle:
        return bundle[hookpoint]
    safe = hookpoint.replace("/", "_").replace(".", "_")
    if safe in bundle:
        return bundle[safe]
    return None


def select_pca_from_bundle(
    bundle,
    hookpoint: str,
    mean_path: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if torch.is_tensor(bundle):
        return bundle, None

    if isinstance(bundle, dict):
        matrix = _maybe_select_tensor(
            bundle, ["pca_matrix", "projection", "components", "P", "matrix"]
        )
        if matrix is not None:
            mean = _maybe_select_tensor(bundle, ["mean", "pca_mean", "mu"])
            if mean_path:
                mean_data = load_pca_bundle(mean_path)
                mean = _select_tensor(mean_data, ["mean", "pca_mean", "mu"], "pca_mean")
            return matrix, mean

        entry = _maybe_get_bundle_entry(bundle, hookpoint)
        if entry is None:
            raise ValueError(
                f"Could not find PCA entry for hookpoint '{hookpoint}' in bundle"
            )
        matrix = _select_tensor(
            entry, ["pca_matrix", "projection", "components", "P", "matrix"], "pca_matrix"
        )
        mean = _maybe_select_tensor(entry, ["mean", "pca_mean", "mu"])
        if mean_path:
            mean_data = load_pca_bundle(mean_path)
            mean = _select_tensor(mean_data, ["mean", "pca_mean", "mu"], "pca_mean")
        return matrix, mean

    raise ValueError(f"Unsupported PCA bundle type: {type(bundle)}")
