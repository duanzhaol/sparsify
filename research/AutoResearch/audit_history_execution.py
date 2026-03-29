"""Audit historical rounds for action/runtime/checkpoint config mismatches.

This script reconstructs the executed configuration for each round using:
1. `action.env_overrides` from round summaries
2. the saved `*.config.json` next to each training log (runner/runtime intent)
3. the saved checkpoint `config.json` (actual training config)

It emits a machine-readable report and can annotate problematic round summaries
with an `audit` block. For confirmed execution mismatches, it also writes
`result.invalid_reason` so downstream logic can filter them out.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
HISTORY_DIR = REPO_ROOT / "research" / "history"
ROUND_SUMMARIES_DIR = HISTORY_DIR / "round_summaries"
DEFAULT_REPORT_PATH = HISTORY_DIR / "execution_audit.json"
FRONTIER_PATH = HISTORY_DIR / "frontier.json"

INT_KEYS = {
    "K",
    "EXPANSION_FACTOR",
    "BATCH_SIZE",
    "GRAD_ACC_STEPS",
    "MICRO_ACC_STEPS",
    "DEAD_FEATURE_THRESHOLD",
    "TRUNK_RANK",
    "NUM_CODES",
    "FACTORIZED_HIDDEN_DIM",
    "NUM_EXPERTS",
    "ACTIVE_EXPERTS",
}
FLOAT_KEYS = {"LR", "AUXK_ALPHA", "STAGE1_RATIO"}
BOOL_KEYS = {"USE_HADAMARD", "COMPILE_MODEL"}

RUNTIME_CONFIG_MAP: dict[str, tuple[str, str]] = {
    "ARCHITECTURE": ("architecture", "top"),
    "EXPANSION_FACTOR": ("expansion_factor", "top"),
    "K": ("k", "top"),
    "OPTIMIZER": ("optimizer", "top"),
    "LR": ("lr", "top"),
    "BATCH_SIZE": ("batch_size", "top"),
    "GRAD_ACC_STEPS": ("grad_acc_steps", "top"),
    "MICRO_ACC_STEPS": ("micro_acc_steps", "top"),
    "AUXK_ALPHA": ("auxk_alpha", "top"),
    "DEAD_FEATURE_THRESHOLD": ("dead_feature_threshold", "top"),
    "USE_HADAMARD": ("use_hadamard", "top"),
}

CHECKPOINT_CONFIG_MAP: dict[str, tuple[str, str]] = {
    "ARCHITECTURE": ("architecture", "sae"),
    "EXPANSION_FACTOR": ("expansion_factor", "sae"),
    "K": ("k", "sae"),
    "OPTIMIZER": ("optimizer", "top"),
    "LR": ("lr", "top"),
    "BATCH_SIZE": ("batch_size", "top"),
    "GRAD_ACC_STEPS": ("grad_acc_steps", "top"),
    "MICRO_ACC_STEPS": ("micro_acc_steps", "top"),
    "AUXK_ALPHA": ("auxk_alpha", "top"),
    "DEAD_FEATURE_THRESHOLD": ("dead_feature_threshold", "top"),
    "USE_HADAMARD": ("use_hadamard", "top"),
    "COMPILE_MODEL": ("compile_model", "top"),
    "TRUNK_RANK": ("trunk_rank", "sae"),
    "NUM_CODES": ("num_codes", "sae"),
    "STAGE1_RATIO": ("stage1_ratio", "sae"),
    "FACTORIZED_HIDDEN_DIM": ("factorized_hidden_dim", "sae"),
    "NUM_EXPERTS": ("num_experts", "sae"),
    "ACTIVE_EXPERTS": ("active_experts", "sae"),
}

RESULT_FIELD_MAP: dict[str, str] = {
    "ARCHITECTURE": "architecture",
    "K": "k",
    "EXPANSION_FACTOR": "expansion_factor",
}


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _normalize_value(key: str, value: Any) -> Any:
    if value is None:
        return None
    if key in BOOL_KEYS:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if key in INT_KEYS:
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return str(value)
    if key in FLOAT_KEYS:
        try:
            return float(str(value))
        except (TypeError, ValueError):
            return str(value)
    if key == "ARCHITECTURE" or key == "OPTIMIZER":
        return str(value).strip().lower()
    return str(value)


def _values_match(key: str, expected: Any, actual: Any) -> bool:
    left = _normalize_value(key, expected)
    right = _normalize_value(key, actual)
    if key in FLOAT_KEYS and left is not None and right is not None:
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
    return left == right


def _action_env(summary: dict[str, Any]) -> dict[str, str]:
    action = summary.get("action", {}) if isinstance(summary.get("action"), dict) else {}
    env: dict[str, str] = {}
    for item in action.get("env_overrides", []) or []:
        key = item.get("key")
        value = item.get("value")
        if key is None or value is None:
            continue
        env[str(key)] = str(value)
    family_name = action.get("family_name")
    if family_name and "ARCHITECTURE" not in env:
        env["ARCHITECTURE"] = str(family_name)
    return env


def _runtime_config_path(summary: dict[str, Any]) -> Path | None:
    result = summary.get("result", {}) if isinstance(summary.get("result"), dict) else {}
    log_path = result.get("log_path")
    if not log_path:
        return None
    return Path(str(log_path).replace(".log", ".config.json"))


def _checkpoint_config_path(summary: dict[str, Any]) -> Path | None:
    result = summary.get("result", {}) if isinstance(summary.get("result"), dict) else {}
    checkpoint = result.get("checkpoint")
    if not checkpoint:
        return None
    return Path(str(checkpoint)) / "config.json"


def _config_value(raw: dict[str, Any], key: str, mapping: dict[str, tuple[str, str]]) -> Any:
    config_key, scope = mapping[key]
    if scope == "sae":
        return (raw.get("sae") or {}).get(config_key)
    return raw.get(config_key)


def _issue(
    kind: str,
    key: str,
    expected: Any,
    actual: Any,
    source: str,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "key": key,
        "expected": expected,
        "actual": actual,
        "source": source,
    }


def _build_invalid_reason(issues: list[dict[str, Any]]) -> str:
    checkpoint_issues = [i for i in issues if i["kind"] == "action_vs_checkpoint"]
    if checkpoint_issues:
        parts = []
        for item in checkpoint_issues:
            parts.append(
                f"{item['key']}={item['expected']} was requested, but checkpoint config saved {item['actual']!r}"
            )
        return (
            "Audit confirmed that requested runtime parameters did not reach the actual training config: "
            + "; ".join(parts)
            + ". This round should not be treated as a valid experimental record."
        )
    runtime_issues = [i for i in issues if i["kind"] == "action_vs_runtime"]
    if runtime_issues:
        parts = []
        for item in runtime_issues:
            parts.append(
                f"{item['key']}={item['expected']} was requested, but the saved runtime command config resolved to {item['actual']!r}"
            )
        return (
            "Audit confirmed that the executed command did not match the agent decision: "
            + "; ".join(parts)
            + ". This round should not be treated as a valid experimental record."
        )
    result_issues = [i for i in issues if i["kind"] == "action_vs_result"]
    if result_issues:
        parts = []
        for item in result_issues:
            parts.append(
                f"{item['key']} expected {item['expected']}, but parsed result recorded {item['actual']!r}"
            )
        return (
            "Audit found a mismatch between the agent decision and recorded result metadata: "
            + "; ".join(parts)
            + ". Treat this round as problematic until manually reviewed."
        )
    return "Audit found an execution/config mismatch in this round."


def audit_round(summary_path: Path) -> tuple[dict[str, Any], dict[str, Any], bool]:
    summary = _load_json(summary_path) or {}
    result = summary.get("result", {}) if isinstance(summary.get("result"), dict) else {}
    expected = _action_env(summary)
    runtime_cfg_path = _runtime_config_path(summary)
    checkpoint_cfg_path = _checkpoint_config_path(summary)
    runtime_cfg = _load_json(runtime_cfg_path) if runtime_cfg_path and runtime_cfg_path.exists() else None
    checkpoint_cfg = _load_json(checkpoint_cfg_path) if checkpoint_cfg_path and checkpoint_cfg_path.exists() else None

    issues: list[dict[str, Any]] = []
    verified_runtime_keys: list[str] = []
    verified_checkpoint_keys: list[str] = []
    missing_runtime_keys: list[str] = []
    missing_checkpoint_keys: list[str] = []

    for key, expected_value in expected.items():
        if key in RUNTIME_CONFIG_MAP:
            if runtime_cfg is None:
                missing_runtime_keys.append(key)
            else:
                verified_runtime_keys.append(key)
                actual = _config_value(runtime_cfg, key, RUNTIME_CONFIG_MAP)
                if not _values_match(key, expected_value, actual):
                    issues.append(_issue(
                        "action_vs_runtime",
                        key,
                        expected_value,
                        actual,
                        "runtime_config",
                    ))

        if key in CHECKPOINT_CONFIG_MAP:
            if checkpoint_cfg is None:
                missing_checkpoint_keys.append(key)
            else:
                verified_checkpoint_keys.append(key)
                actual = _config_value(checkpoint_cfg, key, CHECKPOINT_CONFIG_MAP)
                if not _values_match(key, expected_value, actual):
                    issues.append(_issue(
                        "action_vs_checkpoint",
                        key,
                        expected_value,
                        actual,
                        "checkpoint_config",
                    ))

        result_field = RESULT_FIELD_MAP.get(key)
        if result_field and result.get(result_field) is not None:
            actual = result.get(result_field)
            if not _values_match(key, expected_value, actual):
                issues.append(_issue(
                    "action_vs_result",
                    key,
                    expected_value,
                    actual,
                    "result_metadata",
                ))

    status = "ok"
    if issues:
        status = "problem"
    elif expected and (missing_runtime_keys or missing_checkpoint_keys):
        status = "ok_with_partial_evidence"
    elif not result.get("log_path") and not result.get("checkpoint"):
        status = "not_executed"

    audit = {
        "checked_at": int(time.time()),
        "status": status,
        "expected_env": expected,
        "verified_runtime_keys": sorted(set(verified_runtime_keys)),
        "verified_checkpoint_keys": sorted(set(verified_checkpoint_keys)),
        "missing_runtime_keys": sorted(set(missing_runtime_keys)),
        "missing_checkpoint_keys": sorted(set(missing_checkpoint_keys)),
        "runtime_config_path": str(runtime_cfg_path) if runtime_cfg_path else None,
        "checkpoint_config_path": str(checkpoint_cfg_path) if checkpoint_cfg_path else None,
        "issues": issues,
        "invalid_reason": _build_invalid_reason(issues) if issues else None,
    }
    return summary, audit, bool(issues)


def _rounds_on_frontier(problem_rounds: set[int]) -> list[int]:
    frontier = _load_json(FRONTIER_PATH) or {}
    hits: set[int] = set()
    for entry in frontier.values():
        if not isinstance(entry, dict):
            continue
        try:
            round_id = int(entry.get("round"))
        except (TypeError, ValueError):
            continue
        if round_id in problem_rounds:
            hits.add(round_id)
    return sorted(hits)


def _write_summary_annotation(summary_path: Path, summary: dict[str, Any], audit: dict[str, Any]) -> bool:
    changed = False
    if summary.get("audit") != audit:
        summary["audit"] = audit
        changed = True

    if audit["status"] == "problem":
        result = summary.setdefault("result", {})
        if result.get("invalid_reason") != audit["invalid_reason"]:
            # Preserve any existing invalid_reason that already matches the finding.
            if not result.get("invalid_reason"):
                result["invalid_reason"] = audit["invalid_reason"]
                changed = True
        if result.get("audit_problem") is not True:
            result["audit_problem"] = True
            changed = True

    if changed:
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit AutoResearch round execution history.")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Where to write the JSON audit report.",
    )
    parser.add_argument(
        "--annotate-summaries",
        action="store_true",
        help="Write audit annotations back into round summary JSON files.",
    )
    args = parser.parse_args()

    round_paths = sorted(ROUND_SUMMARIES_DIR.glob("round_*.json"))
    problem_rounds: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    annotated = 0

    for summary_path in round_paths:
        summary, audit, has_problem = audit_round(summary_path)
        status = str(audit["status"])
        status_counts[status] = status_counts.get(status, 0) + 1

        if has_problem:
            problem_rounds.append({
                "round": summary.get("round"),
                "family_name": summary.get("family_name") or summary.get("action", {}).get("family_name"),
                "decision": summary.get("result", {}).get("decision"),
                "issues": audit["issues"],
                "invalid_reason": audit["invalid_reason"],
                "runtime_config_path": audit["runtime_config_path"],
                "checkpoint_config_path": audit["checkpoint_config_path"],
            })

        if args.annotate_summaries and has_problem:
            if _write_summary_annotation(summary_path, summary, audit):
                annotated += 1

    problem_round_ids = {int(item["round"]) for item in problem_rounds if item.get("round") is not None}
    report = {
        "generated_at": int(time.time()),
        "rounds_total": len(round_paths),
        "status_counts": status_counts,
        "problem_rounds": problem_rounds,
        "problem_round_count": len(problem_rounds),
        "frontier_problem_rounds": _rounds_on_frontier(problem_round_ids),
        "annotated_summary_count": annotated,
    }
    args.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"rounds_total={report['rounds_total']}")
    print(f"problem_round_count={report['problem_round_count']}")
    print(f"frontier_problem_rounds={report['frontier_problem_rounds']}")
    print(f"annotated_summary_count={annotated}")
    for item in problem_rounds:
        print(f"r{item['round']}: {item['invalid_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
