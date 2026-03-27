"""Compatibility parsing and filtering for LUTurbo-oriented AutoResearch."""

from __future__ import annotations

import re
from typing import Iterable

DIRECT_COMPATIBLE = "direct_compatible"
EXTENDED_COMPATIBLE = "extended_compatible"
INCOMPATIBLE = "incompatible"
UNKNOWN_COMPATIBILITY = "unknown"

_LABEL_MAP = {
    "直接兼容": DIRECT_COMPATIBLE,
    "扩展兼容": EXTENDED_COMPATIBLE,
    "不兼容": INCOMPATIBLE,
    DIRECT_COMPATIBLE: DIRECT_COMPATIBLE,
    EXTENDED_COMPATIBLE: EXTENDED_COMPATIBLE,
    INCOMPATIBLE: INCOMPATIBLE,
}


def normalize_compatibility_label(label: str | None) -> str:
    if not label:
        return UNKNOWN_COMPATIBILITY
    cleaned = str(label).strip().strip("`")
    return _LABEL_MAP.get(cleaned, UNKNOWN_COMPATIBILITY)


def is_compatible_label(label: str | None) -> bool:
    normalized = normalize_compatibility_label(label)
    return normalized in {DIRECT_COMPATIBLE, EXTENDED_COMPATIBLE}


def parse_compatibility_registry(prior_text: str) -> dict[str, str]:
    """Parse family compatibility labels from markdown tables in prior history.

    Expected row shape:
    | `family` | ... | 直接兼容 | ... |
    """
    registry: dict[str, str] = {}
    if not prior_text:
        return registry

    for raw_line in prior_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        if "---" in line:
            continue

        cols = [part.strip() for part in line.split("|")[1:-1]]
        if len(cols) < 5:
            continue

        family = _normalize_family_name(cols[0])
        if not family:
            continue

        compat = normalize_compatibility_label(cols[4])
        if compat == UNKNOWN_COMPATIBILITY:
            continue

        registry[family] = compat

    return registry


def summarize_registry_counts(registry: dict[str, str]) -> str:
    direct = sum(1 for v in registry.values() if v == DIRECT_COMPATIBLE)
    extended = sum(1 for v in registry.values() if v == EXTENDED_COMPATIBLE)
    incompatible = sum(1 for v in registry.values() if v == INCOMPATIBLE)
    return (
        f"兼容性注册表：直接兼容 {direct} 个，"
        f"扩展兼容 {extended} 个，不兼容 {incompatible} 个。"
    )


def compatibility_hard_rules() -> str:
    """Short rules that must appear in every prompt."""
    return "\n".join([
        "LUTurbo 兼容性硬约束：",
        "1. 提案前必须先写出最终重构公式，并说明它如何导出到 LUTurbo/Lottable 推理链路。",
        "2. 每个 family 必须先判断兼容性：直接兼容 / 扩展兼容 / 不兼容。",
        "3. 明确标记为不兼容的 family，不得继续占据 frontier，也不得继续作为主线推进。",
        "4. 当前 frontier 只代表兼容 family 的最优点；不兼容 family 只能作为历史参考，不能驱动后续决策。",
    ])


def extract_full_prior_document(prior_text: str) -> str:
    """Return the full prior document for fresh-session prompts."""
    return prior_text.strip()


def extract_compatibility_digest(prior_text: str) -> str:
    """Extract the compact compatibility-focused sections for proposal prompts."""
    if not prior_text:
        return ""

    keep_headers = (
        "## 2. LUTurbo/Lottable 兼容性约束",
        "## 10.4 把“兼容性”写进 prompt，而不是靠事后人工筛选",
    )
    lines = prior_text.splitlines()
    extracted: list[str] = []
    capture = False
    for line in lines:
        if any(line.startswith(header) for header in keep_headers):
            capture = True
        elif capture and line.startswith("## "):
            capture = False

        if capture:
            extracted.append(line)

    return "\n".join(extracted).strip()


def filter_compatible_family_names(
    family_names: Iterable[str],
    registry: dict[str, str],
) -> list[str]:
    return [
        str(name).lower()
        for name in family_names
        if is_compatible_label(registry.get(str(name).lower()))
    ]


def _normalize_family_name(raw: str) -> str:
    text = raw.strip().strip("`")
    if not text:
        return ""
    if not re.fullmatch(r"[A-Za-z0-9_]+", text):
        return ""
    return text.lower()
