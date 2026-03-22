"""Policy layer: pure validation functions for the autoresearch framework.

All functions are side-effect-free except ``behavioral_diff_test`` which
spawns a short-lived subprocess.
"""

from __future__ import annotations

import subprocess
from typing import Any

from .git_ops import REPO_ROOT
from .types import Action, BASE_ENV_DEFAULTS

MAX_INCUBATING_FAMILIES = 10
MAX_INCUBATING_PROXY_ROUNDS = 3
LANE_HIGH_K_THRESHOLD = 64


# ---------------------------------------------------------------------------
# Top-level validation
# ---------------------------------------------------------------------------


def validate_action(
    action: Action,
    state: Any,  # StateManager
    force_param_only: bool = False,
) -> tuple[Action | None, str | None]:
    """Run all policy checks. Returns (action, rejection_reason).

    If ``rejection_reason`` is not None the round should be aborted.
    The returned ``action`` may be modified (e.g. forced to param_only).
    """
    if force_param_only and action.is_code_edit:
        action = _force_param_only(action)

    # Variable isolation (skip when frontier is empty — first rounds need to set baseline)
    if state.frontier:
        baseline = _resolve_baseline_for_policy(action, state.frontier)
        changes = _classify_changes(action.env_overrides, baseline)
        ok, msg = check_variable_isolation(changes, action.change_type)
        if not ok:
            return action, f"Variable isolation violated: {msg}"

    # Incubation limits
    ok, msg = check_incubation_limits(
        state.families, action.family_name, action.family_stage,
    )
    if not ok:
        return action, f"Incubation limit: {msg}"

    return action, None


# ---------------------------------------------------------------------------
# Variable isolation
# ---------------------------------------------------------------------------


def check_variable_isolation(
    changes: dict[str, Any],
    change_type: str,
) -> tuple[bool, str]:
    """Single-variable principle. Returns (ok, violation)."""
    dims: list[str] = []
    if change_type in ("edit_sae_code", "edit_perf_code"):
        dims.append("code_edit")
    if changes.get("architecture"):
        dims.append("architecture")
    if changes.get("optimizer"):
        dims.append("optimizer")
    if changes.get("k"):
        dims.append("k")
    # ef coupled with architecture is acceptable
    if "architecture" in dims and changes.get("ef") and not changes.get("k"):
        pass
    elif changes.get("ef") and len(dims) > 0:
        dims.append("ef")
    # lr coupled with optimizer is acceptable
    if "optimizer" in dims and changes.get("lr") and not changes.get("architecture"):
        pass
    elif changes.get("lr") and len(dims) > 0:
        dims.append("lr")
    if len(dims) <= 1:
        return True, ""
    return False, f"Multiple primary variables: {', '.join(dims)}"


# ---------------------------------------------------------------------------
# Incubation management
# ---------------------------------------------------------------------------


def check_incubation_limits(
    families: dict[str, Any],
    family_name: str,
    family_stage: str,
) -> tuple[bool, str]:
    """Returns (ok, message)."""
    name = (family_name or "").lower()
    is_new = name not in families
    is_incubating_stage = family_stage not in ("mainline", "promote_to_mainline")

    # Count currently incubating
    active = sum(1 for f in families.values() if f.get("status") == "incubating")

    if is_new and is_incubating_stage and active >= MAX_INCUBATING_FAMILIES:
        stale = [
            n for n, f in families.items()
            if f.get("status") == "incubating"
            and sum(1 for c in f.get("tested_configs", []) if c.get("stage") != "mainline") >= MAX_INCUBATING_PROXY_ROUNDS
            and not any(c.get("decision") in ("keep", "promote") for c in f.get("tested_configs", []))
        ]
        return False, (
            f"Already {active} incubating families (limit {MAX_INCUBATING_FAMILIES}). "
            f"Stale: {stale}"
        )

    if not is_new and name in families:
        fam = families[name]
        if fam.get("status") == "incubating":
            configs = fam.get("tested_configs", [])
            proxy_rounds = sum(1 for c in configs if c.get("stage") != "mainline")
            has_positive = any(c.get("decision") in ("keep", "promote") for c in configs)
            if proxy_rounds >= MAX_INCUBATING_PROXY_ROUNDS and not has_positive:
                return False, f"Family '{name}' exhausted {proxy_rounds} rounds without positive result"

    return True, ""


def auto_archive_stale_families(families: dict[str, Any]) -> list[str]:
    """Archive stale incubating families. Mutates dict. Returns archived names."""
    archived: list[str] = []
    for name, fam in families.items():
        if fam.get("status") != "incubating":
            continue
        configs = fam.get("tested_configs", [])
        proxy_rounds = sum(1 for c in configs if c.get("stage") != "mainline")
        has_positive = any(c.get("decision") in ("keep", "promote") for c in configs)
        if proxy_rounds >= MAX_INCUBATING_PROXY_ROUNDS and not has_positive:
            fam["status"] = "archived"
            archived.append(name)
    return archived


# ---------------------------------------------------------------------------
# Stagnation detection
# ---------------------------------------------------------------------------


def detect_stagnation(
    consecutive_no_improve: int,
    consecutive_crashes: int,
) -> dict[str, Any]:
    """Returns {level, crash_streak, recommended_mode}."""
    crash_streak = consecutive_crashes >= 2
    if consecutive_no_improve >= 5:
        level, mode = "severe", "k_exploration"
    elif consecutive_no_improve >= 3:
        level, mode = "mild", "exploitation"
    else:
        level, mode = "none", "normal"
    if crash_streak:
        mode = "stabilize_after_crashes"
    return {
        "level": level,
        "crash_streak": crash_streak,
        "consecutive_no_improve": consecutive_no_improve,
        "consecutive_crashes": consecutive_crashes,
        "recommended_mode": mode,
    }


# ---------------------------------------------------------------------------
# Policy guidance (for prompt)
# ---------------------------------------------------------------------------


def build_policy_guidance(
    round_id: int,
    state: Any,  # StateManager
    stagnation: dict[str, Any],
) -> str:
    """Assemble policy guidance text for the agent prompt."""
    parts: list[str] = []

    # Crash recovery
    if stagnation["recommended_mode"] == "stabilize_after_crashes":
        parts.append(
            "CRASH RECOVERY MODE: Recent rounds crashed consecutively.\n"
            "You MUST use change_type='param_only' — do NOT edit code.\n"
            "Use the current best configuration with minor parameter adjustments."
        )
        return "\n\n".join(parts)

    # Stagnation
    no_improve = stagnation["consecutive_no_improve"]
    if stagnation["level"] == "mild":
        parts.append(
            f"STAGNATION WARNING: {no_improve} rounds without improvement.\n"
            "Switch to exploitation: sweep LR, auxk_alpha on current best.\n"
            "Prefer K reduction over EF reduction. Avoid new architectures."
        )
    elif stagnation["level"] == "severe":
        parts.append(
            f"SEVERE STAGNATION: {no_improve} rounds without improvement.\n"
            "1. If only K=128 tested, MUST try K=64 or K=32\n"
            "2. Otherwise try fundamentally different architecture\n"
            "3. Stop minor tweaks — they haven't worked"
        )

    # K exploration guidance
    k_guidance = _k_exploration_guidance(state.frontier, state.memory)
    if k_guidance and stagnation["level"] != "none":
        parts.append(k_guidance)

    # Sweep guidance in exploitation mode
    if stagnation["recommended_mode"] == "exploitation":
        sweep = _sweep_guidance(state)
        if sweep:
            parts.append(sweep)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Behavioral diff test
# ---------------------------------------------------------------------------


def behavioral_diff_test(architecture: str, k: int, ef: int) -> dict[str, Any]:
    """Compare candidate architecture encode() output vs baseline topk."""
    if architecture == "topk":
        return {"identical": False, "max_diff": 0.0, "architecture": architecture}

    code = f"""
import sys; sys.path.insert(0, '.')
import torch; torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    import torch_npu
    device = 'npu' if torch.npu.is_available() else device
except ImportError: pass
from sparsify import SparseCoder
from sparsify.config import SparseCoderConfig
x = torch.randn(4, 1024, device=device, dtype=torch.float32)
torch.manual_seed(0)
base = SparseCoder(1024, SparseCoderConfig(architecture='topk', k={k}, expansion_factor={ef}), device=device, dtype=torch.float32)
base_out = base.encode(x)
torch.manual_seed(0)
cand = SparseCoder(1024, SparseCoderConfig(architecture='{architecture}', k={k}, expansion_factor={ef}), device=device, dtype=torch.float32)
cand_out = cand.encode(x)
if hasattr(base_out, 'top_acts'):
    ba, bi = base_out.top_acts, base_out.top_indices
    ca, ci = cand_out.top_acts, cand_out.top_indices
elif isinstance(base_out, tuple) and len(base_out) >= 2:
    ba, bi = base_out[0], base_out[1]
    ca, ci = cand_out[0], cand_out[1]
else:
    print("DIFF:unknown_format"); sys.exit(0)
md = (ba - ca).abs().max().item() if ba.shape == ca.shape else float('inf')
print(f"DIFF:{{'identical' if torch.equal(ba, ca) and torch.equal(bi, ci) else 'different'}}|{{md}}")
"""
    try:
        result = subprocess.run(
            ["python", "-c", code], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=60,
        )
        for line in (result.stdout + result.stderr).splitlines():
            if line.startswith("DIFF:"):
                parts = line[5:].split("|")
                return {
                    "identical": parts[0] == "identical",
                    "max_diff": float(parts[1]) if len(parts) > 1 else 0.0,
                    "architecture": architecture,
                }
        return {"identical": False, "max_diff": -1.0, "architecture": architecture, "error": "parse_failed"}
    except subprocess.TimeoutExpired:
        return {"identical": False, "max_diff": -1.0, "architecture": architecture, "error": "timeout"}
    except Exception as exc:
        return {"identical": False, "max_diff": -1.0, "architecture": architecture, "error": str(exc)}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_baseline_for_policy(action: Action, frontier: dict[str, Any]) -> dict[str, str]:
    """Find the best frontier config matching the action's family to use as baseline.

    This ensures variable isolation is checked against the actual winning recipe,
    not against BASE_ENV_DEFAULTS. When agent sends ARCHITECTURE+K+EF+LR matching
    a frontier entry and only changes K, this correctly detects a single-variable change.
    """
    family = (action.family_name or "").lower()
    best_entry = None
    best_fvu = float("inf")

    for entry in frontier.values():
        if not isinstance(entry, dict):
            continue
        cfg = entry.get("config", {})
        arch = str(cfg.get("architecture", "")).lower()
        if arch == family:
            fvu = float(entry.get("fvu", float("inf")))
            if fvu < best_fvu:
                best_fvu = fvu
                best_entry = entry

    if best_entry is None:
        return dict(BASE_ENV_DEFAULTS)

    # Convert frontier config to env-var format
    cfg = best_entry["config"]
    base = dict(BASE_ENV_DEFAULTS)
    key_map = {
        "architecture": "ARCHITECTURE",
        "expansion_factor": "EXPANSION_FACTOR",
        "k": "K",
        "optimizer": "OPTIMIZER",
        "lr": "LR",
        "hookpoints": "HOOKPOINTS",
        "batch_size": "BATCH_SIZE",
        "grad_acc_steps": "GRAD_ACC_STEPS",
        "micro_acc_steps": "MICRO_ACC_STEPS",
        "auxk_alpha": "AUXK_ALPHA",
        "dead_feature_threshold": "DEAD_FEATURE_THRESHOLD",
        "use_hadamard": "USE_HADAMARD",
    }
    for json_key, env_key in key_map.items():
        val = cfg.get(json_key)
        if val is not None:
            if env_key == "USE_HADAMARD":
                base[env_key] = "1" if bool(val) else "0"
            else:
                base[env_key] = str(val)
    return base


def _force_param_only(action: Action) -> Action:
    """Return a copy with change_type forced to param_only."""
    d = action.to_dict()
    d["change_type"] = "param_only"
    d["needs_sanity"] = False
    return Action.from_dict(d)


def _classify_changes(
    env_overrides: list[dict[str, str]],
    defaults: dict[str, str],
) -> dict[str, Any]:
    """Classify which dimensions are changed vs defaults."""
    changed: dict[str, str] = {}
    for item in env_overrides:
        key = item.get("key", "")
        value = str(item.get("value", ""))
        if value != str(defaults.get(key, "")):
            changed[key] = value
    return {
        "architecture": "ARCHITECTURE" in changed,
        "optimizer": "OPTIMIZER" in changed,
        "lr": "LR" in changed,
        "k": "K" in changed,
        "ef": "EXPANSION_FACTOR" in changed,
        "other": [k for k in changed if k not in {"ARCHITECTURE", "OPTIMIZER", "LR", "K", "EXPANSION_FACTOR"}],
    }


def _k_exploration_guidance(frontier: dict[str, Any], memory: dict[str, Any]) -> str:
    """Check K value coverage and generate guidance."""
    tested_ks: set[int] = set()

    # From frontier
    for key, entry in frontier.items():
        k_val = entry.get("k") if isinstance(entry, dict) else None
        if k_val is None and str(key).isdigit():
            k_val = int(key)
        if k_val is not None:
            try:
                tested_ks.add(int(k_val))
            except (TypeError, ValueError):
                pass

    # From recent rounds
    for entry in memory.get("recent_rounds", []):
        k_val = entry.get("k")
        if k_val is not None:
            try:
                tested_ks.add(int(k_val))
            except (TypeError, ValueError):
                pass

    if not tested_ks:
        tested_ks.add(128)

    if tested_ks == {128}:
        return (
            "K EXPLORATION REQUIRED: Only K=128 has been tested.\n"
            "Test the best architecture with K=64 before opening new branches."
        )

    high_ks = {k for k in tested_ks if k >= LANE_HIGH_K_THRESHOLD}
    if 64 not in high_ks and high_ks:
        return f"K coverage: {sorted(tested_ks)}. Consider testing K=64."

    return f"K coverage: {sorted(tested_ks)}."


def _sweep_guidance(state: Any) -> str:
    """Generate simple sweep suggestions in exploitation mode."""
    m = state.memory
    # Find best active family
    best_family = None
    best_fvu = float("inf")
    for name, fam in m.get("architecture_families", {}).items():
        if fam.get("status") != "active":
            continue
        for key in ("best_fvu", "best_full_fvu", "best_proxy_fvu"):
            val = fam.get(key)
            if val is not None:
                try:
                    v = float(val)
                    if v < best_fvu:
                        best_fvu = v
                        best_family = name
                except (TypeError, ValueError):
                    pass

    # Collect tested params from recent rounds
    tested_lrs: set[str] = {BASE_ENV_DEFAULTS["LR"]}
    tested_auxk: set[str] = {BASE_ENV_DEFAULTS["AUXK_ALPHA"]}
    for entry in m.get("recent_rounds", []):
        # Not perfect but sufficient for guidance
        pass

    # Collect from round summaries
    for s in state.recent_round_summaries(limit=20):
        action = s.get("action", {})
        sf = str(action.get("family_name", "")).lower()
        if best_family and sf != best_family:
            continue
        overrides = action.get("env_overrides", [])
        if isinstance(overrides, list):
            for item in overrides:
                if item.get("key") == "LR":
                    tested_lrs.add(str(item.get("value", "")))
                elif item.get("key") == "AUXK_ALPHA":
                    tested_auxk.add(str(item.get("value", "")))

    all_lrs = {"2e-4", "4e-4", "8e-4", "1.6e-3", "3.2e-3"}
    all_auxk = {"0", "0.01", "0.03125", "0.0625"}
    untested_lrs = sorted(all_lrs - tested_lrs)
    untested_auxk = sorted(all_auxk - tested_auxk)

    parts: list[str] = [f"EXPLOITATION MODE for {best_family or 'topk'}:"]
    if untested_lrs:
        parts.append(f"  Untested LRs: {', '.join(untested_lrs)}")
    if untested_auxk:
        parts.append(f"  Untested auxk_alpha: {', '.join(untested_auxk)}")
    if not untested_lrs and not untested_auxk:
        parts.append("  All standard hyperparameters tested. Try different K or new architecture.")
    parts.append("  Change ONE parameter per round.")
    return "\n".join(parts)
