"""Shared package paths."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = PACKAGE_DIR / "examples"
DESIGN_DIR = PACKAGE_DIR / "design"

DEFAULT_TASKPACK_FILENAME = "taskpack.json"
SCHEMA_PATH = EXAMPLES_DIR / "taskpack.schema.json"
TEMPLATE_PATH = EXAMPLES_DIR / "taskpack.template.json"
