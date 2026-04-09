from __future__ import annotations

import json
from fnmatch import fnmatchcase
from pathlib import Path
from statistics import mean
from typing import Any

from natsort import natsorted
import torch

from sparsify.checkpoint import expand_range_pattern


def resolve_matching_hookpoints(
    hookpoint_patterns: list[str],
    available_module_names: list[str],
) -> list[str]:
    expanded_patterns: list[str] = []
    for pattern in hookpoint_patterns:
        expanded_patterns.extend(expand_range_pattern(pattern))

    matched = [
        name for name in available_module_names if any(fnmatchcase(name, pat) for pat in expanded_patterns)
    ]
    return natsorted(matched)


def resolve_checkpoint_paths(
    checkpoint_root: str | Path,
    hookpoints: list[str],
) -> dict[str, Path]:
    root = Path(checkpoint_root)
    resolved: dict[str, Path] = {}

    for hookpoint in hookpoints:
        candidate = root / hookpoint
        if (candidate / "cfg.json").exists():
            resolved[hookpoint] = candidate
            continue

        nested_cfgs = sorted(candidate.glob("*/cfg.json"))
        if len(nested_cfgs) == 1:
            resolved[hookpoint] = nested_cfgs[0].parent
            continue

        raise FileNotFoundError(
            f"Could not resolve checkpoint directory for hookpoint '{hookpoint}' under '{root}'"
        )

    return resolved


def load_elbow_thresholds_for_hookpoints(
    path: str | Path,
    hookpoints: list[str],
) -> dict[str, float]:
    with open(path, "r") as f:
        elbow_data = json.load(f)

    matched: dict[str, float] = {}
    for hookpoint in hookpoints:
        if hookpoint in elbow_data:
            matched[hookpoint] = float(elbow_data[hookpoint]["elbow_value"])
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

            found = False
            for pattern in search_patterns:
                for json_key, value in elbow_data.items():
                    if pattern in json_key or json_key in hookpoint:
                        matched[hookpoint] = float(value["elbow_value"])
                        found = True
                        break
                if found:
                    break

    return matched


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "aggregate": {
                "num_hookpoints": 0,
                "mean_fvu_delta": 0.0,
                "mean_exceed_alpha_0.50_delta": 0.0,
                "worst_fvu_hookpoint": None,
                "worst_exceed_alpha_0.50_hookpoint": None,
            },
            "per_hookpoint": [],
        }

    worst_fvu = max(records, key=lambda item: item["fvu_delta"])
    worst_exceed = max(records, key=lambda item: item["exceed_alpha_0.50_delta"])
    return {
        "aggregate": {
            "num_hookpoints": len(records),
            "mean_fvu_delta": mean(item["fvu_delta"] for item in records),
            "mean_exceed_alpha_0.50_delta": mean(
                item["exceed_alpha_0.50_delta"] for item in records
            ),
            "worst_fvu_hookpoint": worst_fvu["hookpoint"],
            "worst_exceed_alpha_0.50_hookpoint": worst_exceed["hookpoint"],
        },
        "per_hookpoint": records,
    }


def compute_reconstruction_metrics(
    target: Any,
    recon: Any,
    elbow_value: float,
    *,
    alpha: float = 0.50,
) -> dict[str, float]:
    target_t = target.to(torch.float32)
    recon_t = recon.to(torch.float32)
    error = target_t - recon_t
    total_variance = (target_t - target_t.mean(0)).pow(2).sum().clamp_min(1e-12)
    fvu = float(error.pow(2).sum() / total_variance)
    exceed = float((error.abs() > (alpha * elbow_value)).float().mean())
    return {
        "fvu": fvu,
        f"exceed_alpha_{alpha:.2f}": exceed,
    }
