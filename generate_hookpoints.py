#!/usr/bin/env python3
"""Generate hookpoint patterns for a range of layers."""
import sys

def generate_hookpoints(start: int, end: int, pattern: str = "layers.{}.self_attn.o_proj"):
    """
    Generate hookpoint patterns for layers [start, end].

    Args:
        start: First layer (inclusive)
        end: Last layer (inclusive)
        pattern: Pattern template with {} for layer number

    Returns:
        List of hookpoint strings
    """
    return [pattern.format(i) for i in range(start, end + 1)]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_hookpoints.py START END [PATTERN]")
        print('Example: python generate_hookpoints.py 1 10')
        print('Example: python generate_hookpoints.py 0 5 "layers.{}.mlp.act"')
        sys.exit(1)

    start = int(sys.argv[1])
    end = int(sys.argv[2])
    pattern = sys.argv[3] if len(sys.argv) > 3 else "layers.{}.self_attn.o_proj"

    hookpoints = generate_hookpoints(start, end, pattern)

    # Output as space-separated strings (ready for bash)
    print(" ".join(f'"{hp}"' for hp in hookpoints))
