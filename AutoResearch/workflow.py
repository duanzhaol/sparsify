"""Workflow graph helpers."""

from __future__ import annotations

from collections import deque
from typing import Any


def build_adjacency(edges: list[dict[str, Any]]) -> dict[str, list[str]]:
    adjacency: dict[str, list[str]] = {}
    for edge in edges:
        src = str(edge.get("from", ""))
        dst = str(edge.get("to", ""))
        adjacency.setdefault(src, []).append(dst)
        adjacency.setdefault(dst, [])
    return adjacency


def reachable_nodes(
    entry_node: str,
    nodes: dict[str, Any],
    edges: list[dict[str, Any]],
) -> set[str]:
    if entry_node not in nodes:
        return set()
    adjacency = build_adjacency(edges)
    seen: set[str] = set()
    queue: deque[str] = deque([entry_node])
    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        for neighbor in adjacency.get(current, []):
            if neighbor not in seen:
                queue.append(neighbor)
    return seen


def unreachable_nodes(
    entry_node: str,
    nodes: dict[str, Any],
    edges: list[dict[str, Any]],
) -> list[str]:
    return sorted(set(nodes) - reachable_nodes(entry_node, nodes, edges))
