"""
Policy layer for the autoresearch loop.

Implements behavioral validation, variable isolation, stagnation detection,
K exploration guidance, incubation management, sweep guidance, and meta-analysis.
"""

from __future__ import annotations

import subprocess
import time
from typing import Any

from research.git_ops import REPO_ROOT
from research.state_io import BASE_ENV_DEFAULTS


# ---------------------------------------------------------------------------
# Phase 2: Behavioral Diff Test
# ---------------------------------------------------------------------------

def behavioral_diff_test(architecture: str, k: int, ef: int) -> dict[str, Any]:
    """Compare a candidate architecture's encode() output against baseline topk.

    Creates both SAEs with the same seed and random input, then compares
    encode() outputs. Returns {"identical": bool, "max_diff": float, "architecture": str}.
    """
    if architecture == "topk":
        return {"identical": False, "max_diff": 0.0, "architecture": architecture}

    test_code = f"""
import sys; sys.path.insert(0, '.')
import torch
torch.manual_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    import torch_npu
    device = 'npu' if torch.npu.is_available() else device
except ImportError:
    pass

from sparsify import SparseCoder
from sparsify.config import SparseCoderConfig

x = torch.randn(4, 1024, device=device, dtype=torch.float32)

# Baseline topk
cfg_base = SparseCoderConfig(architecture='topk', k={k}, expansion_factor={ef})
torch.manual_seed(0)
sae_base = SparseCoder(1024, cfg_base, device=device, dtype=torch.float32)
base_out = sae_base.encode(x)

# Candidate
cfg_cand = SparseCoderConfig(architecture='{architecture}', k={k}, expansion_factor={ef})
torch.manual_seed(0)
sae_cand = SparseCoder(1024, cfg_cand, device=device, dtype=torch.float32)
cand_out = sae_cand.encode(x)

# Compare
if hasattr(base_out, 'top_acts'):
    base_acts, cand_acts = base_out.top_acts, cand_out.top_acts
    base_idx, cand_idx = base_out.top_indices, cand_out.top_indices
elif isinstance(base_out, tuple) and len(base_out) >= 2:
    base_acts, base_idx = base_out[0], base_out[1]
    cand_acts, cand_idx = cand_out[0], cand_out[1]
else:
    print("DIFF:unknown_format")
    sys.exit(0)

acts_identical = torch.equal(base_acts, cand_acts)
idx_identical = torch.equal(base_idx, cand_idx)
max_diff = (base_acts - cand_acts).abs().max().item() if base_acts.shape == cand_acts.shape else float('inf')

if acts_identical and idx_identical:
    print(f"DIFF:identical|{{max_diff}}")
else:
    print(f"DIFF:different|{{max_diff}}")
"""
    try:
        result = subprocess.run(
            ["python", "-c", test_code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        for line in (result.stdout + result.stderr).splitlines():
            if line.startswith("DIFF:"):
                parts = line[5:].split("|")
                status = parts[0]
                max_diff = float(parts[1]) if len(parts) > 1 else 0.0
                return {
                    "identical": status == "identical",
                    "max_diff": max_diff,
                    "architecture": architecture,
                }
        # If we can't parse output, assume different (don't block) but log it
        stderr_snippet = result.stderr[:500] if result.stderr else ""
        print(f"behavioral_diff_test: could not parse output for {architecture}, passing through. stderr: {stderr_snippet}")
        return {"identical": False, "max_diff": -1.0, "architecture": architecture, "error": "parse_failed"}
    except subprocess.TimeoutExpired:
        print(f"behavioral_diff_test: subprocess timed out for {architecture}")
        return {"identical": False, "max_diff": -1.0, "architecture": architecture, "error": "timeout"}
    except Exception as exc:
        print(f"behavioral_diff_test: unexpected error for {architecture}: {exc}")
        return {"identical": False, "max_diff": -1.0, "architecture": architecture, "error": str(exc)}


# ---------------------------------------------------------------------------
# Phase 2: Variable Isolation
# ---------------------------------------------------------------------------

def classify_changes(env_overrides: list | dict, base_defaults: dict[str, str]) -> dict[str, Any]:
    """Classify which dimensions env_overrides change relative to base defaults.

    Returns {"architecture": bool, "optimizer": bool, "lr": bool, "k": bool,
             "ef": bool, "other": list[str]}
    """
    changed: dict[str, str] = {}
    if isinstance(env_overrides, dict):
        for key, value in env_overrides.items():
            if str(value) != str(base_defaults.get(key, "")):
                changed[key] = str(value)
    else:
        for item in env_overrides:
            key = item.get("key", "")
            value = str(item.get("value", ""))
            if value != str(base_defaults.get(key, "")):
                changed[key] = value

    return {
        "architecture": "ARCHITECTURE" in changed,
        "optimizer": "OPTIMIZER" in changed,
        "lr": "LR" in changed,
        "k": "K" in changed,
        "ef": "EXPANSION_FACTOR" in changed,
        "other": [k for k in changed if k not in {"ARCHITECTURE", "OPTIMIZER", "LR", "K", "EXPANSION_FACTOR"}],
    }


def check_variable_isolation(changes: dict[str, Any], change_type: str) -> tuple[bool, str]:
    """Check whether the action respects single-variable principle.

    Primary dimensions: architecture, optimizer, k.
    Code edit counts as one dimension.
    lr + optimizer may change together (coupled).
    Returns (ok, violation_description).
    """
    dimensions_changed: list[str] = []

    if change_type in ("edit_sae_code", "edit_perf_code"):
        dimensions_changed.append("code_edit")

    if changes.get("architecture"):
        dimensions_changed.append("architecture")
    if changes.get("optimizer"):
        dimensions_changed.append("optimizer")
    if changes.get("k"):
        dimensions_changed.append("k")

    # lr + optimizer is acceptable as coupled
    if "optimizer" in dimensions_changed and changes.get("lr") and not changes.get("architecture"):
        pass  # OK, lr is coupled with optimizer
    elif changes.get("lr") and len(dimensions_changed) > 0:
        dimensions_changed.append("lr")

    if len(dimensions_changed) <= 1:
        return True, ""

    return False, f"Multiple primary variables changed simultaneously: {', '.join(dimensions_changed)}"


# ---------------------------------------------------------------------------
# Phase 3: Dynamic Proxy Budget
# ---------------------------------------------------------------------------

def compute_proxy_budget(change_type: str, default: str) -> str:
    """Return token budget for proxy runs based on change type.

    edit_sae_code gets 40M tokens (~600 steps) to allow zero-initialized
    components to diverge from baseline.
    """
    if change_type == "edit_sae_code":
        return "40000000"
    return default


# ---------------------------------------------------------------------------
# Phase 3: Stagnation Detection
# ---------------------------------------------------------------------------

def detect_stagnation(agent_state: dict[str, Any]) -> dict[str, Any]:
    """Analyze agent state for stagnation patterns.

    Returns:
        level: "none" | "mild" (3-4 rounds) | "severe" (5+ rounds)
        crash_streak: bool (consecutive_crashes >= 2)
        recommended_mode: "normal" | "exploitation" | "k_exploration" | "revert_and_simplify"
    """
    no_improve = int(agent_state.get("consecutive_no_improve", 0))
    crashes = int(agent_state.get("consecutive_crashes", 0))

    crash_streak = crashes >= 2

    if no_improve >= 5:
        level = "severe"
        recommended_mode = "k_exploration"
    elif no_improve >= 3:
        level = "mild"
        recommended_mode = "exploitation"
    else:
        level = "none"
        recommended_mode = "normal"

    if crash_streak:
        recommended_mode = "revert_and_simplify"

    return {
        "level": level,
        "crash_streak": crash_streak,
        "consecutive_no_improve": no_improve,
        "consecutive_crashes": crashes,
        "recommended_mode": recommended_mode,
    }


def generate_stagnation_guidance(level: str, memory: dict[str, Any], state: dict[str, Any], recommended_mode: str = "normal") -> str:
    """Generate prompt guidance text based on stagnation level and recommended mode."""
    if recommended_mode == "revert_and_simplify":
        return (
            "CRASH RECOVERY MODE: Recent rounds have crashed consecutively.\n"
            "You MUST use change_type='param_only' this round — do NOT edit code.\n"
            "Use the current best working configuration with minor parameter adjustments only.\n"
            "Focus on stability before exploring new architectures."
        )

    if level == "none":
        return ""

    agent = state.get("agent", {})
    no_improve = int(agent.get("consecutive_no_improve", 0))

    if level == "mild":
        return (
            f"STAGNATION WARNING: {no_improve} consecutive rounds without improvement.\n"
            "Consider switching to exploitation mode:\n"
            "- Do a systematic hyperparameter sweep on the current best configuration\n"
            "- Try different learning rates, auxk_alpha values\n"
            "- If a promising incubating family exists, focus on stabilizing it\n"
            "- Avoid introducing new architecture families until current ones are exhausted"
        )

    if level == "severe":
        return (
            f"SEVERE STAGNATION: {no_improve} consecutive rounds without improvement.\n"
            "MANDATORY actions:\n"
            "1. If K=128 is the only tested K value, you MUST explore K=64 or K=32 on the best architecture\n"
            "2. If multiple K values have been tested, try a fundamentally different architecture approach\n"
            f"3. Do NOT continue minor parameter tweaks — they have not worked for {no_improve} rounds\n"
            "4. Consider whether the current search direction is fundamentally flawed"
        )

    return ""


def get_revert_target(memory: dict[str, Any]) -> str | None:
    """Find a safe git commit to revert sparsify/ code to after a crash streak.

    Strategy: find the most recent round with a healthy (non-crash) result
    from an active/promoted family, then look up that round's commit via
    git log. Returns the commit hash, or None if no safe target is found.
    """
    import subprocess
    from research.git_ops import REPO_ROOT

    # Find the most recent successful round
    best_round = -1
    families = memory.get("architecture_families", {})
    for name, family in families.items():
        if family.get("status") not in ("active", "promoted"):
            continue
        for config in family.get("tested_configs", []):
            if config.get("run_health") not in (None, "normal"):
                continue
            if config.get("decision") in ("keep", "promote", "incubate"):
                r = config.get("round")
                if r is not None and int(r) > best_round:
                    best_round = int(r)

    if best_round < 0:
        return None

    # Search git log for the commit that recorded that round
    pattern = f"experiment: round {best_round:04d}"
    try:
        result = subprocess.run(
            ["git", "log", "--all", "--grep", pattern, "--format=%H", "-1"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        commit = result.stdout.strip()
        return commit if commit else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Phase 4: K Exploration Guidance
# ---------------------------------------------------------------------------

def generate_k_exploration_guidance(frontier: dict[str, Any], memory: dict[str, Any] | None = None, results: list[dict[str, str]] | None = None) -> str:
    """Check all available data sources for K value coverage and generate guidance if needed."""
    tested_ks: set[int] = set()

    # 1. Check frontier
    if isinstance(frontier, dict):
        for key, entry in frontier.items():
            if isinstance(entry, dict):
                k_val = entry.get("k")
                if k_val is not None:
                    try:
                        tested_ks.add(int(k_val))
                    except (TypeError, ValueError):
                        pass

    # 2. Check memory.recent_rounds
    if memory:
        for entry in memory.get("recent_rounds", []):
            k_val = entry.get("k")
            if k_val is not None:
                try:
                    tested_ks.add(int(k_val))
                except (TypeError, ValueError):
                    pass

    # 3. Check results.tsv rows
    if results:
        for row in results:
            k_val = row.get("k")
            if k_val is not None and k_val != "":
                try:
                    tested_ks.add(int(k_val))
                except (TypeError, ValueError):
                    pass

    if not tested_ks:
        tested_ks.add(128)  # Assume default

    if tested_ks == {128}:
        return (
            "K EXPLORATION REQUIRED:\n"
            "Currently only K=128 has been tested. The project's core objective is to find\n"
            "configurations that work well with smaller K values.\n"
            "Please test the current best architecture with K=64, then K=32.\n"
            "This is the highest priority — use param_only with env_overrides to change K.\n"
            "Do NOT introduce new architectures until K exploration is done."
        )

    if 64 not in tested_ks and 32 not in tested_ks:
        return (
            f"K values tested so far: {sorted(tested_ks)}.\n"
            "Consider testing K=64 and K=32 to explore the K reduction frontier."
        )

    return ""


# ---------------------------------------------------------------------------
# Phase 4: Incubation Management
# ---------------------------------------------------------------------------

def audit_incubating_families(memory: dict[str, Any]) -> dict[str, Any]:
    """Scan architecture families and return incubation audit.

    Returns:
        stale: list of family names with >= 3 proxy rounds and no keep/promote
        active_count: number of currently incubating families
        over_limit: whether active_count > 2
    """
    families = memory.get("architecture_families", {})
    stale: list[str] = []
    active_count = 0

    for name, family in families.items():
        if family.get("status") != "incubating":
            continue
        active_count += 1
        configs = family.get("tested_configs", [])
        proxy_rounds = sum(1 for c in configs if c.get("stage") != "mainline")
        has_positive = any(c.get("decision") in ("keep", "promote") for c in configs)
        if proxy_rounds >= 3 and not has_positive:
            stale.append(name)

    return {
        "stale": stale,
        "active_count": active_count,
        "over_limit": active_count > 2,
    }


def enforce_incubation_limits(memory: dict[str, Any], action: dict[str, Any]) -> tuple[bool, str]:
    """Check if the action respects incubation limits.

    Returns (ok, message). ok=False if:
    - Creating a new incubating family but already >= 2 incubating
    - Continuing a family that used up its 3-round proxy quota
    """
    audit = audit_incubating_families(memory)
    family_name = str(action.get("family_name", "")).lower()
    family_stage = str(action.get("family_stage", ""))
    families = memory.get("architecture_families", {})

    # Check if this is a new family being incubated
    is_new = family_name not in families
    is_incubating_stage = family_stage not in ("mainline", "promote_to_mainline")

    if is_new and is_incubating_stage and audit["active_count"] >= 2:
        return False, (
            f"Cannot create new incubating family '{family_name}': "
            f"already {audit['active_count']} incubating families (limit: 2). "
            f"Stale families that could be archived: {audit['stale']}"
        )

    # Check if existing family exceeded its quota
    if not is_new and family_name in families:
        family = families[family_name]
        if family.get("status") == "incubating":
            configs = family.get("tested_configs", [])
            proxy_rounds = sum(1 for c in configs if c.get("stage") != "mainline")
            has_positive = any(c.get("decision") in ("keep", "promote") for c in configs)
            if proxy_rounds >= 3 and not has_positive:
                return False, (
                    f"Family '{family_name}' has used {proxy_rounds} proxy rounds "
                    f"without a keep/promote decision. Consider archiving it."
                )

    return True, ""


def auto_archive_stale_families(memory: dict[str, Any], max_proxy_rounds: int = 3) -> list[str]:
    """Archive families that exceeded their proxy quota without positive results.

    Modifies memory in-place. Returns list of archived family names.
    """
    families = memory.get("architecture_families", {})
    archived: list[str] = []

    for name, family in families.items():
        if family.get("status") != "incubating":
            continue
        configs = family.get("tested_configs", [])
        proxy_rounds = sum(1 for c in configs if c.get("stage") != "mainline")
        has_positive = any(c.get("decision") in ("keep", "promote") for c in configs)
        if proxy_rounds >= max_proxy_rounds and not has_positive:
            family["status"] = "archived"
            archived.append(name)

    return archived


# ---------------------------------------------------------------------------
# Phase 4: Sweep Guidance
# ---------------------------------------------------------------------------

def generate_sweep_guidance(
    frontier: dict[str, Any],
    memory: dict[str, Any],
    results: list[dict[str, str]] | None = None,
) -> str:
    """Generate systematic hyperparameter sweep suggestions in exploitation mode.

    Uses structured experiment history only: results.tsv for family coverage and
    round summaries for exact env_overrides. Avoids walking checkpoint artifacts.
    """
    import json
    from research.state_io import ROUND_SUMMARIES_DIR, BASE_ENV_DEFAULTS

    tested_lrs: set[str] = set()
    tested_auxk: set[str] = set()
    tested_efs: set[int] = set()

    # Determine the current best family to filter by.
    # Two-pass: first try families with full FVU (most trustworthy),
    # then fall back to proxy FVU if no family has full results.
    best_family: str | None = None
    best_fvu = float("inf")

    # Pass 1: families with best_full_fvu
    for fname, fdata in memory.get("architecture_families", {}).items():
        if fdata.get("status") not in ("active", "promoted"):
            continue
        full_fvu = fdata.get("best_full_fvu")
        if full_fvu is not None:
            try:
                fvu_val = float(full_fvu)
                if fvu_val < best_fvu:
                    best_fvu = fvu_val
                    best_family = fname
            except (TypeError, ValueError):
                pass

    # Pass 2: fall back to proxy FVU only if no full results found
    if best_family is None:
        for fname, fdata in memory.get("architecture_families", {}).items():
            if fdata.get("status") not in ("active", "promoted"):
                continue
            proxy_fvu = fdata.get("best_proxy_fvu")
            if proxy_fvu is not None:
                try:
                    fvu_val = float(proxy_fvu)
                    if fvu_val < best_fvu:
                        best_fvu = fvu_val
                        best_family = fname
                except (TypeError, ValueError):
                    pass

    relevant_experiment_ids = {
        row.get("experiment_id", "")
        for row in (results or [])
        if not best_family or str(row.get("architecture", "")).lower() == best_family
    }

    # Structured source: round summaries with action.env_overrides, filtered by family.
    for summary_path in sorted(ROUND_SUMMARIES_DIR.glob("round_*.json")):
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            action = summary.get("action", {})
            summary_family = str(action.get("family_name", "")).lower()
            if best_family and summary_family != best_family:
                continue
            proxy_result = summary.get("proxy_result") or {}
            full_result = summary.get("full_result") or {}
            experiment_ids = {
                str(proxy_result.get("experiment_id") or ""),
                str(full_result.get("experiment_id") or ""),
            }
            if relevant_experiment_ids and experiment_ids.isdisjoint(relevant_experiment_ids):
                continue
            overrides = action.get("env_overrides", [])
            if isinstance(overrides, dict):
                items = list(overrides.items())
            else:
                items = [(item.get("key"), item.get("value")) for item in overrides]
            for key, value in items:
                if key == "LR" and value:
                    tested_lrs.add(str(value))
                elif key == "AUXK_ALPHA" and value:
                    tested_auxk.add(str(value))
                elif key == "EXPANSION_FACTOR" and value:
                    try:
                        tested_efs.add(int(value))
                    except (TypeError, ValueError):
                        pass
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            continue

    # Always mark defaults as tested
    tested_lrs.add(BASE_ENV_DEFAULTS.get("LR", "8e-4"))
    tested_auxk.add(BASE_ENV_DEFAULTS.get("AUXK_ALPHA", "0.03125"))
    tested_efs.add(int(BASE_ENV_DEFAULTS.get("EXPANSION_FACTOR", "8")))

    all_lrs = ["2e-4", "4e-4", "8e-4", "1.6e-3", "3.2e-3"]
    all_auxk = ["0", "0.01", "0.03125", "0.0625"]
    all_efs = [4, 8, 16]

    untested_lrs = [v for v in all_lrs if v not in tested_lrs]
    untested_auxk = [v for v in all_auxk if v not in tested_auxk]
    untested_efs = [v for v in all_efs if v not in tested_efs]

    suggestions: list[str] = []
    if untested_lrs:
        suggestions.append(f"Untested learning rates: {', '.join(untested_lrs)}")
    else:
        suggestions.append("All candidate learning rates have been tested")
    if untested_auxk:
        suggestions.append(f"Untested auxk_alpha values: {', '.join(untested_auxk)}")
    else:
        suggestions.append("All candidate auxk_alpha values have been tested")
    if untested_efs:
        suggestions.append(f"Untested expansion_factor values: {', '.join(str(e) for e in untested_efs)}")
    else:
        suggestions.append("All candidate expansion_factor values have been tested")

    family_label = best_family or "default (topk)"
    tested_info = f"(family: {family_label}, tested LRs: {sorted(tested_lrs)}, tested auxk: {sorted(tested_auxk)}, tested EFs: {sorted(tested_efs)})"

    if not untested_lrs and not untested_auxk and not untested_efs:
        return (
            f"EXPLOITATION MODE: All standard hyperparameter candidates have been tested for {family_label} {tested_info}.\n"
            "Consider exploring K values (K=64, K=32) or trying a new architecture approach."
        )

    return (
        "EXPLOITATION MODE — Systematic Hyperparameter Sweep:\n"
        f"Target family: {family_label}. Already tested {tested_info}\n"
        "Focus on this architecture and sweep these parameters one at a time:\n"
        + "\n".join(f"- {s}" for s in suggestions)
        + "\n\nUse param_only with env_overrides. Change ONE parameter per round."
    )


def build_policy_guidance(
    round_id: int,
    state: dict[str, Any],
    memory: dict[str, Any],
    results: list[dict[str, str]],
    stagnation: dict[str, Any],
) -> tuple[str, bool]:
    """Assemble prompt guidance text from the strategy layer.

    Returns (guidance_text, meta_analysis_written).
    """
    parts: list[str] = []
    wrote_meta = False

    guidance = generate_stagnation_guidance(
        stagnation["level"],
        memory,
        state,
        recommended_mode=stagnation.get("recommended_mode", "normal"),
    )
    if guidance:
        parts.append(guidance)

    frontier = state.get("frontier", {})
    k_guidance = generate_k_exploration_guidance(frontier, memory=memory, results=results)
    if k_guidance and stagnation["level"] != "none":
        parts.append(k_guidance)

    if stagnation.get("recommended_mode") == "exploitation":
        sweep_guidance = generate_sweep_guidance(frontier, memory, results=results)
        if sweep_guidance:
            parts.append(sweep_guidance)

    if should_run_meta_analysis(round_id, state):
        meta_text = generate_meta_analysis(memory, results, state)
        if meta_text:
            parts.append(meta_text)
        wrote_meta = True

    return "\n\n".join(parts), wrote_meta


# ---------------------------------------------------------------------------
# Phase 5: Meta-Analysis
# ---------------------------------------------------------------------------

def should_run_meta_analysis(round_id: int, state: dict[str, Any]) -> bool:
    """Return True every 5 rounds."""
    last_meta = state.get("agent", {}).get("last_meta_round", 0)
    return round_id - int(last_meta) >= 5


def generate_meta_analysis(
    memory: dict[str, Any],
    results: list[dict[str, str]],
    state: dict[str, Any],
) -> str:
    """Generate structured meta-analysis text for prompt injection."""
    total_rounds = len(results)
    families = memory.get("architecture_families", {})

    # Count effective vs wasted rounds
    effective = sum(1 for r in results if r.get("decision") in ("keep", "promote", "incubate"))
    crashed = sum(1 for r in results if r.get("decision") == "crash")
    discarded = sum(1 for r in results if r.get("decision") == "discard")

    # K coverage
    tested_ks: set[str] = set()
    for r in results:
        k = r.get("k")
        if k:
            tested_ks.add(str(k))

    # Family summaries
    family_lines: list[str] = []
    for name, family in sorted(families.items()):
        status = family.get("status", "unknown")
        best_proxy = family.get("best_proxy_fvu")
        best_full = family.get("best_full_fvu")
        n_configs = len(family.get("tested_configs", []))
        best_str = ""
        if best_proxy is not None:
            best_str += f" best_proxy_fvu={best_proxy}"
        if best_full is not None:
            best_str += f" best_full_fvu={best_full}"
        family_lines.append(f"  - {name}: status={status}, rounds={n_configs}{best_str}")

    # Optimizer analysis
    optimizer_results: dict[str, list[str]] = {}
    for r in results:
        desc = r.get("description", "")
        fvu = r.get("val_fvu", "")
        # Basic grouping
        if fvu:
            for opt in ("adam", "signum", "muon"):
                if opt in desc.lower():
                    optimizer_results.setdefault(opt, []).append(fvu)

    sections = [
        "META-ANALYSIS (auto-generated every 5 rounds):",
        f"Total rounds: {total_rounds} | Effective: {effective} | Crashed: {crashed} | Discarded: {discarded}",
        f"K values tested: {sorted(tested_ks) if tested_ks else 'only K=128 (default)'}",
        "",
        "Architecture families:",
        "\n".join(family_lines) if family_lines else "  (none recorded)",
    ]

    if optimizer_results:
        sections.append("")
        sections.append("Optimizer observations:")
        for opt, fvus in sorted(optimizer_results.items()):
            sections.append(f"  - {opt}: {len(fvus)} runs, FVUs: {', '.join(fvus[:5])}")

    # Strategic recommendations
    recs: list[str] = []
    if not tested_ks or tested_ks == {"128"}:
        recs.append("CRITICAL: Explore K=64 and K=32 — this is the project's core objective")
    if crashed > total_rounds * 0.3:
        recs.append(f"High crash rate ({crashed}/{total_rounds}). Focus on stability before exploration.")
    incubating = [n for n, f in families.items() if f.get("status") == "incubating"]
    if len(incubating) > 2:
        recs.append(f"Too many incubating families ({len(incubating)}). Archive stale ones before starting new.")
    if effective == 0 and total_rounds > 3:
        recs.append("No effective rounds yet. Consider simplifying: use param_only on baseline topk.")

    if recs:
        sections.append("")
        sections.append("Strategic recommendations:")
        for rec in recs:
            sections.append(f"  ! {rec}")

    return "\n".join(sections)
