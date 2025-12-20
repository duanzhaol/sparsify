#!/usr/bin/env python3
"""Expand range patterns in hookpoints.

Supports syntax like:
- layers.[1-10].self_attn.o_proj  → expands to layers.1...layers.10
- layers.[0-5,10,15].mlp.act      → expands to 0,1,2,3,4,5,10,15
"""
import re
from typing import List


def expand_range_pattern(pattern: str) -> List[str]:
    """
    Expand a pattern with range syntax into multiple patterns.

    Supports:
    - layers.[N-M].xxx → layers.N.xxx, layers.(N+1).xxx, ..., layers.M.xxx
    - layers.[N,M,P].xxx → layers.N.xxx, layers.M.xxx, layers.P.xxx
    - layers.[N-M,P-Q].xxx → combined ranges

    Args:
        pattern: Pattern string potentially containing range syntax

    Returns:
        List of expanded patterns (may be length 1 if no range found)
    """
    # Match patterns like [1-10] or [0-5,10,15]
    match = re.search(r'\[([0-9,\-]+)\]', pattern)

    if not match:
        # No range syntax, return as-is
        return [pattern]

    range_spec = match.group(1)
    numbers = []

    # Split by comma for multiple ranges/numbers
    for part in range_spec.split(','):
        if '-' in part:
            # Range like "1-10"
            start, end = map(int, part.split('-'))
            numbers.extend(range(start, end + 1))
        else:
            # Single number like "5"
            numbers.append(int(part))

    # Remove duplicates and sort
    numbers = sorted(set(numbers))

    # Generate expanded patterns
    expanded = []
    for num in numbers:
        # Replace [range_spec] with the specific number
        expanded_pattern = pattern.replace(f'[{range_spec}]', str(num))
        expanded.append(expanded_pattern)

    return expanded


def expand_all_patterns(patterns: List[str]) -> List[str]:
    """Expand all patterns in a list."""
    result = []
    for pattern in patterns:
        result.extend(expand_range_pattern(pattern))
    return result


if __name__ == "__main__":
    # Test cases
    test_patterns = [
        "layers.[1-5].self_attn.o_proj",
        "layers.[0-2,10,15].mlp.act",
        "layers.*.self_attn.q_proj",  # No range, should pass through
        "layers.[1-10].self_attn.o_proj",
    ]

    for pattern in test_patterns:
        expanded = expand_range_pattern(pattern)
        print(f"\n{pattern}")
        print(f"  → {len(expanded)} patterns")
        if len(expanded) <= 5:
            for p in expanded:
                print(f"    - {p}")
        else:
            print(f"    - {expanded[0]}")
            print(f"    - ...")
            print(f"    - {expanded[-1]}")
