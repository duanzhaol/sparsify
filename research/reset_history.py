#!/usr/bin/env python3
"""Archive current research history and reset for a fresh experiment run."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

RESEARCH_DIR = Path(__file__).resolve().parent
HISTORY_DIR = RESEARCH_DIR / "history"
ARCHIVE_ROOT = RESEARCH_DIR / "history_archive"

# Files and directories to archive (relative to HISTORY_DIR)
ARCHIVE_FILES = [
    "state.json",
    "frontier.json",
    "memory.json",
    "timeline.jsonl",
    "results.tsv",
    "operator_hints.json",
    "session_brief.json",
    "current_status.json",
]
ARCHIVE_DIRS = [
    "logs",
    "round_summaries",
    ".snapshots",
]


def next_archive_number() -> int:
    """Scan existing archives and return the next sequence number."""
    if not ARCHIVE_ROOT.exists():
        return 1
    max_num = 0
    for entry in ARCHIVE_ROOT.iterdir():
        if entry.is_dir():
            m = re.match(r"^(\d+)_", entry.name)
            if m:
                max_num = max(max_num, int(m.group(1)))
    return max_num + 1


def has_anything_to_archive() -> bool:
    """Check if there are any history files or directories to archive."""
    if not HISTORY_DIR.exists():
        return False
    for f in ARCHIVE_FILES:
        if (HISTORY_DIR / f).exists():
            return True
    for d in ARCHIVE_DIRS:
        p = HISTORY_DIR / d
        if p.exists() and any(p.iterdir()):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive and reset research history")
    parser.add_argument(
        "--no-git", action="store_true",
        help="Skip git add/commit after clearing tracked history files",
    )
    args = parser.parse_args()

    if not has_anything_to_archive():
        print("Nothing to archive — history is already clean.")
        sys.exit(0)

    # Determine archive destination
    seq = next_archive_number()
    ts = time.strftime("%Y%m%d_%H%M%S")
    archive_name = f"{seq:02d}_{ts}"
    archive_dir = ARCHIVE_ROOT / archive_name
    archive_dir.mkdir(parents=True, exist_ok=True)

    moved = []

    # Move files
    for f in ARCHIVE_FILES:
        src = HISTORY_DIR / f
        if src.exists():
            shutil.move(str(src), str(archive_dir / f))
            moved.append(f)

    # Move directories
    for d in ARCHIVE_DIRS:
        src = HISTORY_DIR / d
        if src.exists() and any(src.iterdir()):
            shutil.move(str(src), str(archive_dir / d))
            moved.append(d + "/")
        elif src.exists():
            # Empty directory — just remove, no need to archive
            shutil.rmtree(src)

    print(f"Archived to {archive_dir.relative_to(RESEARCH_DIR.parent)}/")
    if moved:
        print(f"  {', '.join(moved)}")

    # Git commit the deletion of tracked files
    if not args.no_git:
        try:
            subprocess.run(
                ["git", "add", "research/history/"],
                cwd=str(RESEARCH_DIR.parent),
                check=True,
                capture_output=True,
            )
            result = subprocess.run(
                ["git", "diff", "--cached", "--quiet", "research/history/"],
                cwd=str(RESEARCH_DIR.parent),
                capture_output=True,
            )
            if result.returncode != 0:
                # There are staged changes to commit
                subprocess.run(
                    ["git", "commit", "-m", f"Reset history: archived to {archive_name}"],
                    cwd=str(RESEARCH_DIR.parent),
                    check=True,
                    capture_output=True,
                )
                print(f"Committed history reset to git.")
            else:
                print("No tracked history changes to commit.")
        except subprocess.CalledProcessError as e:
            print(f"Warning: git commit failed: {e}", file=sys.stderr)
            print("Run manually: git add research/history/ && git commit", file=sys.stderr)

    print("History cleared. Run agent_loop.py to start fresh.")


if __name__ == "__main__":
    main()
