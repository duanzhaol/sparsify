# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sparsify is a library for training k-sparse autoencoders (SAEs) and transcoders on HuggingFace language model activations. It follows the recipe from "Scaling and evaluating sparse autoencoders" (Gao et al. 2024) and uses TopK activation functions instead of L1 penalties.

Key design choices:
- Computes activations on-the-fly (no disk caching) for scalability
- Uses TopK activation for direct sparsity enforcement
- Supports distributed training via DDP and module distribution across GPUs

## Common Commands

```bash
# Install for development
pip install -e .[dev]

# Basic training
python -m sparsify EleutherAI/pythia-160m

# Training with custom dataset
python -m sparsify EleutherAI/pythia-160m togethercomputer/RedPajama-Data-1T-Sample

# Training transcoders
python -m sparsify EleutherAI/pythia-160m --transcode

# Custom hookpoints (wildcards supported)
python -m sparsify gpt2 --hookpoints "h.*.attn" "h.*.mlp.act"

# Distributed training with torchrun
torchrun --nproc_per_node gpu -m sparsify meta-llama/Meta-Llama-3-8B --batch_size 1 --layers 16 24

# Memory-efficient multi-layer training (SAEs distributed across GPUs)
torchrun --nproc_per_node gpu -m sparsify meta-llama/Meta-Llama-3-8B --distribute_modules --layer_stride 2

# Finetuning from pretrained
python -m sparsify EleutherAI/pythia-160m --finetune EleutherAI/sae-pythia-160m-32x

# Run tests (requires CUDA)
pytest tests/
```

## Architecture

### Core Classes

- **SparseCoder** (`sparse_coder.py`): The SAE/transcoder model. Handles encoding (input → top-k latents) and decoding (latents → reconstruction). Aliased as `Sae`.

- **Trainer** (`trainer.py`): Training loop implementation. Manages:
  - Hook-based activation capture from the target model
  - Distributed training (DDP or module distribution)
  - Checkpoint saving/resuming
  - WandB logging

- **TrainConfig/SaeConfig** (`config.py`): Configuration dataclasses. `TrainConfig` contains training hyperparameters, `SaeConfig` (alias `SparseCoderConfig`) defines the SAE architecture.

### Training Flow

1. Model activations are captured via PyTorch forward hooks registered on specified hookpoints
2. Activations are encoded through the SAE's linear encoder + TopK selection
3. Loss is computed (FVU by default, or CE/KL for end-to-end training)
4. Gradients flow back through the SAE only (model is frozen)

### Hookpoint System

Hookpoints use Unix glob patterns (via `fnmatch`) to select model submodules:
- `layers.10` - specific layer
- `h.*.attn` - all attention modules
- `h.[012].mlp.act` - layers 0-2 MLP activations

### Distributed Training Modes

- **DDP (default)**: SAE weights replicated on all GPUs
- **`--distribute_modules`**: Each GPU trains SAEs for different layers (more memory efficient, requires even division of layers across GPUs)

## Key Configuration Options

- `expansion_factor`: Multiplier for latent dimension (default: 32)
- `k`: Number of active latents (default: 32)
- `loss_fn`: `fvu` (local reconstruction), `ce` or `kl` (end-to-end)
- `optimizer`: `signum` (default), `adam`, or `muon`
- `micro_acc_steps`: Split activations into microbatches to save memory
- `auxk_alpha`: Weight for auxiliary loss on dead features
