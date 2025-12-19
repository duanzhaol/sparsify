#!/usr/bin/env python3
"""
Quick start example: K-value sweep based on your findings

You discovered that k=32 performs better than k=64.
This script performs a fine-grained search around k=32.
"""

import sys
sys.path.insert(0, '/root/sparsify/scripts')

# Import the main sweep script and modify config
from hyperparam_sweep import BASE_CONFIG, SWEEP_PARAMS, main

# Override the sweep parameters for your use case
SWEEP_PARAMS = {
    "expansion_factor": [8],                    # Keep your current setting
    "k": [20, 24, 28, 32, 36, 40],             # Fine-grained search around 32
}

# Adjust training duration (100M tokens per run)
BASE_CONFIG["max_tokens"] = 100_000_000

# Quick test mode: uncomment this for fast testing
# BASE_CONFIG["max_tokens"] = 10_000_000  # Only 10M tokens for quick test

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                     Fine-Grained K-Value Sweep                             ║
╚════════════════════════════════════════════════════════════════════════════╝

Based on your findings:
  ✓ k=32 performs better than k=64
  ✓ k=32 has fewer dead features

This sweep will test k values around 32 to find the optimal setting:
  - k ∈ {20, 24, 28, 32, 36, 40}
  - expansion_factor = 8 (fixed)
  - 100M tokens per run

Total experiments: 6
Estimated time: ~6-12 hours (depending on hardware)

""")

if __name__ == "__main__":
    main()
