"""Git operations and snapshot management for the autoresearch loop."""

from __future__ import annotations

import difflib
import hashlib
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = REPO_ROOT / "research"
HISTORY_DIR = RESEARCH_DIR / "history"

ALLOWED_EDIT_PREFIXES = ("sparsify/",)

SNAPSHOT_ROOTS = ("sparsify",)
SNAPSHOT_EXCLUDES = (
    "research/history/",
    "research/history_old/",
    "research/AutoResearch/",
    "sparsify/__pycache__/",
    "research/__pycache__/",
    "scripts/__pycache__/",
    "research/agent_loop.py",
    "research/git_ops.py",
    "research/state_io.py",
    "research/training.py",
    "research/prompts.py",
    "research/policy.py",
    "research/controller.py",
    "research/prepare.py",
    "research/program.md",
    "research/agent_action.schema.json",
    "scripts/autoresearch_test.sh",
)
SNAPSHOT_EXCLUDE_SUFFIXES = (".pyc", ".pyo")
TRACKED_HISTORY_PATHS: tuple[Path, ...] = ()  # Set by init_tracked_paths()


def init_tracked_paths(
    state_path: Path,
    results_path: Path,
    frontier_path: Path,
    memory_path: Path,
    timeline_path: Path,
    session_brief_path: Path,
    hints_path: Path,
) -> None:
    """Inject the tracked-history paths from the main module to avoid circular imports."""
    global TRACKED_HISTORY_PATHS
    TRACKED_HISTORY_PATHS = (
        state_path,
        results_path,
        frontier_path,
        memory_path,
        timeline_path,
        session_brief_path,
        hints_path,
    )


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd or REPO_ROOT, check=check, text=True, capture_output=True)


def git(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=check, text=True, capture_output=True)


def current_git_branch() -> str:
    return git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()


def current_git_commit() -> str:
    return git(["rev-parse", "HEAD"]).stdout.strip()


def worktree_dirty() -> bool:
    return bool(git(["status", "--porcelain"]).stdout.strip())


def branch_exists(branch: str) -> bool:
    return git(["rev-parse", "--verify", branch], check=False).returncode == 0


def ensure_clean_worktree_for_auto_commit() -> None:
    if worktree_dirty():
        raise RuntimeError(
            "Auto-commit mode requires a clean git worktree before starting the nightly loop."
        )


def stage_paths(paths: list[Path]) -> None:
    rel_paths = []
    for path in paths:
        if path.exists():
            rel_paths.append(path.relative_to(REPO_ROOT).as_posix())
    if rel_paths:
        git(["add", "--", *rel_paths])


def commit_message_for_round(round_id: int, action: dict[str, Any], result: dict[str, str], tier: str) -> str:
    family = str(action.get("family_name") or action.get("change_type") or "unknown")
    decision = str(result.get("decision") or "unknown")
    return f"experiment: round {round_id:04d} {tier} {family} {decision}"


def commit_round_state(
    round_id: int,
    action: dict[str, Any],
    result: dict[str, str],
    touched: list[str],
    round_summary_path: Path,
    tier: str,
) -> tuple[str | None, str]:
    paths = [REPO_ROOT / path for path in touched]
    paths.extend(TRACKED_HISTORY_PATHS)
    paths.append(round_summary_path)
    stage_paths(paths)
    if git(["diff", "--cached", "--quiet"], check=False).returncode == 0:
        return None, current_git_branch()
    message = commit_message_for_round(round_id, action, result, tier)
    git(["commit", "-m", message])
    return current_git_commit(), current_git_branch()


def snapshot_paths() -> dict[str, str]:
    snapshots: dict[str, str] = {}
    for root in SNAPSHOT_ROOTS:
        base = REPO_ROOT / root
        if not base.exists():
            continue
        count = 0
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            if any(rel.startswith(prefix) for prefix in SNAPSHOT_EXCLUDES):
                continue
            if rel.endswith(SNAPSHOT_EXCLUDE_SUFFIXES):
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            snapshots[rel] = digest
            count += 1
        print(f"  snapshot: {root}/ -> {count} files")
    print(f"  snapshot total: {len(snapshots)} files")
    return snapshots


def touched_files(before: dict[str, str], after: dict[str, str]) -> list[str]:
    paths = set(before) | set(after)
    return sorted(path for path in paths if before.get(path) != after.get(path))


def assert_allowed_changes(paths: list[str]) -> None:
    disallowed = [path for path in paths if not path.startswith(ALLOWED_EDIT_PREFIXES)]
    if disallowed:
        joined = ", ".join(disallowed)
        raise RuntimeError(f"Agent touched files outside allowed prefixes: {joined}")


def build_patch(before_snapshot: dict[str, str], after_paths: list[str], round_id: int) -> Path | None:
    """Build a unified diff patch and save it to a file. Returns the file path, or None if empty."""
    patch_parts: list[str] = []
    temp_root = HISTORY_DIR / ".snapshots"
    before_root = temp_root / f"round_{round_id:04d}_before"
    after_root = temp_root / f"round_{round_id:04d}_after"
    before_root.mkdir(parents=True, exist_ok=True)
    after_root.mkdir(parents=True, exist_ok=True)

    for rel in after_paths:
        if rel.endswith(SNAPSHOT_EXCLUDE_SUFFIXES) or "/__pycache__/" in rel:
            continue
        source = REPO_ROOT / rel
        before_path = before_root / rel
        after_path = after_root / rel
        before_path.parent.mkdir(parents=True, exist_ok=True)
        after_path.parent.mkdir(parents=True, exist_ok=True)
        before_text = ""
        # before_path was populated by capture_before_files() before agent edits
        if source.exists():
            try:
                after_text = source.read_text()
            except UnicodeDecodeError:
                continue
            after_path.write_text(after_text)
        else:
            after_text = ""
        if before_path.exists():
            try:
                before_text = before_path.read_text()
            except UnicodeDecodeError:
                continue
        before_lines = before_text.splitlines(keepends=True)
        after_lines = after_text.splitlines(keepends=True)
        patch = "".join(
            difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
            )
        )
        if patch:
            patch_parts.append(patch)

    if not patch_parts:
        return None
    patch_content = "\n".join(patch_parts)
    patch_path = HISTORY_DIR / "patches" / f"round_{round_id:04d}.patch"
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(patch_content)
    return patch_path


def cleanup_round_snapshots(round_id: int) -> None:
    temp_root = HISTORY_DIR / ".snapshots"
    for suffix in ("before", "after"):
        path = temp_root / f"round_{round_id:04d}_{suffix}"
        if path.exists():
            shutil.rmtree(path)


def cleanup_all_snapshots() -> None:
    temp_root = HISTORY_DIR / ".snapshots"
    if not temp_root.exists():
        return
    for path in temp_root.iterdir():
        if path.is_dir():
            shutil.rmtree(path)


def capture_before_files(paths: list[str], round_id: int) -> None:
    temp_root = HISTORY_DIR / ".snapshots"
    before_root = temp_root / f"round_{round_id:04d}_before"
    for rel in paths:
        src = REPO_ROOT / rel
        dst = before_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            dst.write_bytes(src.read_bytes())

