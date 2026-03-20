"""Structured training artifact writer: manifest.json + metrics.jsonl + summary.json."""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


class MetricsLogger:
    """Writes per-step metrics to JSONL and a final summary JSON.

    Output files:
        manifest.json  — run identity (git info, model, dataset, architecture, etc.)
        metrics.jsonl  — per-step metrics, one JSON object per line
        summary.json   — final metrics written at training end
    """

    def __init__(self, log_dir: str | Path, config: dict, run_meta: dict):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._pending_lines = 0
        self._flush_every = 10

        self._write_manifest(run_meta)

        self.jsonl_path = self.log_dir / "metrics.jsonl"
        self._file = open(self.jsonl_path, "a")
        self._write_line({"type": "config", "data": config}, flush=True)

    def _write_manifest(self, meta: dict):
        """Write manifest.json with run identity information."""
        manifest = {
            "created_at": datetime.now().isoformat(),
            **meta,
        }
        # Auto-collect git info
        try:
            manifest["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
            manifest["git_branch"] = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            manifest["git_dirty"] = bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            )
        except Exception:
            pass

        with open(self.log_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    def log_step(self, step: int, total_tokens: int, metrics: dict):
        """Append one step's metrics to the JSONL file."""
        self._write_line(
            {
                "type": "step",
                "step": step,
                "total_tokens": total_tokens,
                "timestamp": time.time(),
                **metrics,
            }
        )

    def save_summary(self, summary: dict):
        """Write final summary JSON at training end."""
        with open(self.log_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    def _write_line(self, record: dict, *, flush: bool = False):
        self._file.write(json.dumps(record) + "\n")
        self._pending_lines += 1
        if flush or self._pending_lines >= self._flush_every:
            self._file.flush()
            self._pending_lines = 0

    def close(self):
        self._file.flush()
        self._file.close()
