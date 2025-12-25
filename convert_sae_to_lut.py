#!/usr/bin/env python3
"""
Convert trained SAE checkpoints to LUT (Lookup Table) format for inference.

This script transforms sparse autoencoder (SAE) checkpoints into an optimized
lookup table format that enables efficient inference by precomputing matrix
products between SAE decoder weights and target model weights.

Usage:
    python convert_sae_to_lut.py /root/models/Qwen3-0.6B /root/sparsify/checkpoints \\
        --output_dir /root/sparsify/lut_tables \\
        --proj_types qproj oproj upproj \\
        --layers 0-27 \\
        --threshold_dir /root/sparsify/thresholds/Qwen3-0.6B
"""

import argparse
import json
import re
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM

# Module path mappings
MODULE_PATTERNS = {
    "qproj": "model.layers.{layer_idx}.self_attn.q_proj",
    "oproj": "model.layers.{layer_idx}.self_attn.o_proj",
    "upproj": "model.layers.{layer_idx}.mlp.up_proj",
    "kproj": "model.layers.{layer_idx}.self_attn.k_proj",
    "vproj": "model.layers.{layer_idx}.self_attn.v_proj",
}

CHECKPOINT_PATTERNS = {
    "qproj": "layers.{layer_idx}.self_attn.q_proj",
    "oproj": "layers.{layer_idx}.self_attn.o_proj",
    "upproj": "layers.{layer_idx}.mlp.up_proj",
    "kproj": "layers.{layer_idx}.self_attn.k_proj",
    "vproj": "layers.{layer_idx}.self_attn.v_proj",
}

PROJ_DIR_PATTERNS = {
    "qproj": "*-qproj",
    "oproj": "*-oproj",
    "upproj": "*-upproj",
    "kproj": "*-kproj",
    "vproj": "*-vproj",
}

THRESHOLD_FILES = {
    "qproj": "thresholds_q.json",
    "oproj": "thresholds_o.json",
    "upproj": "thresholds_up.json",
}


def parse_layer_range(range_str: str) -> list[int]:
    """
    Parse layer range string into list of layer indices.

    Supports syntax like:
    - "0-27" → [0, 1, 2, ..., 27]
    - "0-5,10,15" → [0, 1, 2, 3, 4, 5, 10, 15]

    Args:
        range_str: Range specification string

    Returns:
        Sorted list of unique layer indices
    """
    numbers = []
    for part in range_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            numbers.extend(range(start, end + 1))
        else:
            numbers.append(int(part))

    return sorted(set(numbers))


def find_checkpoint_path(
    checkpoint_base_dir: Path,
    proj_type: str,
    checkpoint_name: str
) -> Optional[Path]:
    """
    Find SAE checkpoint path for given projection type and checkpoint name.

    Handles both nested and flat directory structures:
    1. Nested: best/{checkpoint_name}/{checkpoint_name}/sae.safetensors
    2. Flat: best/{checkpoint_name}/sae.safetensors

    Args:
        checkpoint_base_dir: Base checkpoint directory
        proj_type: Projection type (qproj, oproj, upproj, etc.)
        checkpoint_name: Checkpoint name (e.g., "layers.0.self_attn.q_proj")

    Returns:
        Path to checkpoint directory, or None if not found
    """
    # Map proj_type to directory pattern
    pattern = PROJ_DIR_PATTERNS.get(proj_type)
    if pattern is None:
        return None

    # Find matching checkpoint directory
    search_pattern = str(checkpoint_base_dir / pattern)
    matching_dirs = glob(search_pattern)

    if not matching_dirs:
        return None

    checkpoint_dir = Path(matching_dirs[0])

    # Try nested structure first: best/{checkpoint_name}/{checkpoint_name}/
    nested_path = checkpoint_dir / "best" / checkpoint_name / checkpoint_name
    if nested_path.exists() and (nested_path / "sae.safetensors").exists():
        return nested_path

    # Try flat structure: best/{checkpoint_name}/
    flat_path = checkpoint_dir / "best" / checkpoint_name
    if flat_path.exists() and (flat_path / "sae.safetensors").exists():
        return flat_path

    return None


def load_sae_checkpoint(checkpoint_path: Path) -> dict:
    """
    Load SAE checkpoint from disk.

    Args:
        checkpoint_path: Path to checkpoint directory

    Returns:
        Dictionary containing:
        - encoder_weight: [num_latents, d_in]
        - encoder_bias: [num_latents]
        - decoder_weight: [num_latents, d_in]
        - decoder_bias: [d_in]
        - config: Configuration dict
    """
    # Load configuration
    with open(checkpoint_path / "cfg.json", "r") as f:
        config = json.load(f)

    # Load tensors from safetensors
    tensors = {}
    safetensors_path = str(checkpoint_path / "sae.safetensors")

    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        tensors['encoder_weight'] = f.get_tensor('encoder.weight')
        tensors['encoder_bias'] = f.get_tensor('encoder.bias')
        tensors['decoder_weight'] = f.get_tensor('W_dec')
        tensors['decoder_bias'] = f.get_tensor('b_dec')

    tensors['config'] = config
    return tensors


def get_model_weight(model, module_path: str) -> torch.Tensor:
    """
    Extract weight tensor from model module.

    Args:
        model: HuggingFace model
        module_path: Module path (e.g., "model.layers.0.self_attn.q_proj")

    Returns:
        Detached weight tensor

    Raises:
        ValueError: If module doesn't have weight attribute
    """
    module = model.get_submodule(module_path)

    if not hasattr(module, 'weight'):
        raise ValueError(f"Module {module_path} has no weight attribute")

    return module.weight.detach()


def compute_precomputed_products(
    W_dec: torch.Tensor,
    W_target: torch.Tensor,
    device: str,
    batch_size: Optional[int] = None
) -> torch.Tensor:
    """
    Compute precomputed products: W_dec @ W_target.T

    Args:
        W_dec: Decoder weight [num_latents, d_in]
        W_target: Target layer weight [d_out, d_in]
        device: Device for computation
        batch_size: If provided, compute in batches for memory efficiency

    Returns:
        Precomputed products [num_latents, d_out]
    """
    # Convert to float32 for computation (SAE weights are float32)
    W_dec = W_dec.float().to(device)
    W_target = W_target.float().to(device)

    if batch_size is None:
        # Compute all at once
        result = W_dec @ W_target.T
    else:
        # Batch computation for memory efficiency
        num_latents = W_dec.shape[0]
        d_out = W_target.shape[0]
        result = torch.zeros(num_latents, d_out, dtype=torch.float32, device=device)

        for i in range(0, num_latents, batch_size):
            end = min(i + batch_size, num_latents)
            result[i:end] = W_dec[i:end] @ W_target.T

    return result.cpu()


def compute_bias_product(
    b_dec: torch.Tensor,
    W_target: torch.Tensor,
    device: str
) -> torch.Tensor:
    """
    Compute bias product: b_dec @ W_target.T

    Args:
        b_dec: Decoder bias [d_in]
        W_target: Target layer weight [d_out, d_in]
        device: Device for computation

    Returns:
        Bias product [d_out]
    """
    # Convert to float32 for computation (SAE weights are float32)
    b_dec = b_dec.float().to(device)
    W_target = W_target.float().to(device)

    return (b_dec @ W_target.T).cpu()


def save_lut_file(
    output_path: Path,
    tensors_dict: dict,
    target_dtype: torch.dtype
):
    """
    Save LUT file in safetensors format.

    Args:
        output_path: Output file path
        tensors_dict: Dictionary of tensors to save
        target_dtype: Target dtype for conversion
    """
    # Convert all tensors to target dtype
    lut_tensors = {
        key: tensor.to(target_dtype)
        for key, tensor in tensors_dict.items()
    }

    # Create parent directories
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to safetensors
    save_file(lut_tensors, str(output_path))


def load_thresholds(
    threshold_dir: Optional[Path],
    proj_types: list[str]
) -> dict:
    """
    Load elbow thresholds from JSON files.

    Args:
        threshold_dir: Directory containing threshold files
        proj_types: List of projection types to load

    Returns:
        Merged dictionary of all thresholds
    """
    if threshold_dir is None:
        return {}

    thresholds = {}
    for proj_type in proj_types:
        filename = THRESHOLD_FILES.get(proj_type)
        if filename is None:
            continue

        file_path = threshold_dir / filename
        if file_path.exists():
            with open(file_path, "r") as f:
                thresholds.update(json.load(f))

    return thresholds


def generate_metadata(
    model_path: str,
    sae_configs: dict,
    layer_info: dict,
    thresholds: dict,
    model_config: dict
) -> dict:
    """
    Generate metadata.json content.

    Args:
        model_path: Path to source model
        sae_configs: SAE configurations for each layer
        layer_info: Layer information dictionary
        thresholds: Elbow thresholds
        model_config: Model configuration

    Returns:
        Metadata dictionary
    """
    # Get first SAE config for global parameters
    first_config = next(iter(sae_configs.values())) if sae_configs else {}

    # Compute num_basis from config
    d_in = first_config.get("d_in", 0)
    expansion_factor = first_config.get("expansion_factor", 0)
    num_basis = first_config.get("num_latents", 0)
    if num_basis == 0 and d_in > 0 and expansion_factor > 0:
        num_basis = d_in * expansion_factor

    return {
        "version": "1.0",
        "sae_config": {
            "num_basis": num_basis,
            "k_active": first_config.get("k", 0),
        },
        "model_config": {
            "model_type": model_config.get("model_type"),
            "num_layers": model_config.get("num_hidden_layers"),
            "num_attention_heads": model_config.get("num_attention_heads"),
            "hidden_size": model_config.get("hidden_size"),
        },
        "layers": layer_info,
        "elbow_thresholds": thresholds,
        "creation_info": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_model": model_path,
            "script_version": "1.0",
        }
    }


def process_layer(
    model,
    checkpoint_base_dir: Path,
    output_dir: Path,
    proj_type: str,
    layer_idx: int,
    target_dtype: torch.dtype,
    device: str,
    batch_compute: bool
) -> Optional[dict]:
    """
    Process a single layer: load SAE, compute precomputed products, save LUT.

    Args:
        model: HuggingFace model
        checkpoint_base_dir: Base checkpoint directory
        output_dir: Output directory
        proj_type: Projection type
        layer_idx: Layer index
        target_dtype: Target dtype for output
        device: Computation device
        batch_compute: Whether to use batch computation

    Returns:
        Layer info dict, or None on failure
    """
    # Get module and checkpoint paths
    module_path = MODULE_PATTERNS[proj_type].format(layer_idx=layer_idx)
    checkpoint_name = CHECKPOINT_PATTERNS[proj_type].format(layer_idx=layer_idx)

    # 1. Find checkpoint
    checkpoint_path = find_checkpoint_path(
        checkpoint_base_dir, proj_type, checkpoint_name
    )
    if checkpoint_path is None:
        print(f"  WARNING: Checkpoint not found for {checkpoint_name}")
        return None

    # 2. Load SAE
    try:
        sae_data = load_sae_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"  ERROR loading SAE checkpoint: {e}")
        return None

    # 3. Get model weight
    try:
        W_target = get_model_weight(model, module_path)
    except Exception as e:
        print(f"  ERROR getting model weight: {e}")
        return None

    # 4. Validate dimensions
    d_in = sae_data['decoder_weight'].shape[1]
    w_in = W_target.shape[1]
    if d_in != w_in:
        print(f"  ERROR: Dimension mismatch: SAE d_in={d_in}, W_target in_features={w_in}")
        return None

    # 5. Compute precomputed products
    try:
        batch_size = 1024 if batch_compute else None
        precomputed = compute_precomputed_products(
            sae_data['decoder_weight'],
            W_target,
            device,
            batch_size=batch_size
        )
    except Exception as e:
        print(f"  ERROR computing precomputed products: {e}")
        return None

    # 6. Compute bias product
    try:
        bias_prod = compute_bias_product(
            sae_data['decoder_bias'],
            W_target,
            device
        )
    except Exception as e:
        print(f"  ERROR computing bias product: {e}")
        return None

    # 7. Save LUT file
    lut_filename = f"{checkpoint_name}.lut.safetensors"
    lut_path = output_dir / lut_filename

    try:
        save_lut_file(
            lut_path,
            {
                "encoder_weight": sae_data['encoder_weight'],
                "encoder_bias": sae_data['encoder_bias'],
                "decoder_weight": sae_data['decoder_weight'],
                "decoder_bias": sae_data['decoder_bias'],
                "precomputed_products": precomputed,
                "bias_product": bias_prod,
            },
            target_dtype
        )
    except Exception as e:
        print(f"  ERROR saving LUT file: {e}")
        return None

    # 8. Return layer info
    return {
        "input_dim": int(d_in),
        "output_dim": int(W_target.shape[0]),
        "file": lut_filename,
        "module_path": checkpoint_name,
        "sae_config": sae_data['config']
    }


def auto_detect_layers(checkpoint_base_dir: Path, proj_type: str) -> list[int]:
    """
    Auto-detect available layers from checkpoint directory.

    Args:
        checkpoint_base_dir: Base checkpoint directory
        proj_type: Projection type to scan

    Returns:
        List of available layer indices
    """
    # Find checkpoint directory
    pattern = PROJ_DIR_PATTERNS.get(proj_type, "*-qproj")
    search_pattern = str(checkpoint_base_dir / pattern)
    matching_dirs = glob(search_pattern)

    if not matching_dirs:
        return []

    checkpoint_dir = Path(matching_dirs[0])
    best_dir = checkpoint_dir / "best"

    if not best_dir.exists():
        return []

    # Extract layer numbers from directory names
    layer_pattern = re.compile(r'layers\.(\d+)\.')
    layers = set()

    for subdir in best_dir.iterdir():
        if subdir.is_dir():
            match = layer_pattern.search(subdir.name)
            if match:
                layers.add(int(match.group(1)))

    return sorted(layers)


def main():
    parser = argparse.ArgumentParser(
        description="Convert SAE checkpoints to LUT format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python convert_sae_to_lut.py /root/models/Qwen3-0.6B /root/sparsify/checkpoints \\
      --output_dir /root/sparsify/lut_tables

  # With thresholds and specific layers
  python convert_sae_to_lut.py /root/models/Qwen3-0.6B /root/sparsify/checkpoints \\
      --output_dir ./lut_output \\
      --proj_types qproj oproj \\
      --layers 0-5,10 \\
      --threshold_dir /root/sparsify/thresholds/Qwen3-0.6B \\
      --dtype bfloat16
"""
    )

    # Required arguments
    parser.add_argument("model_path", type=str, help="Path to base model")
    parser.add_argument("checkpoint_base_dir", type=str, help="Base checkpoint directory")

    # Optional arguments
    parser.add_argument("--output_dir", type=str, default="./lut_output",
                       help="Output directory for LUT files (default: ./lut_output)")
    parser.add_argument("--proj_types", nargs="+",
                       default=["qproj", "oproj", "upproj"],
                       choices=["qproj", "oproj", "upproj", "kproj", "vproj"],
                       help="Projection types to process (default: qproj oproj upproj)")
    parser.add_argument("--layers", type=str,
                       help="Layer range (e.g., '0-27', '0-5,10') (default: auto-detect)")
    parser.add_argument("--threshold_dir", type=str,
                       help="Directory containing threshold files (optional)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       choices=["float16", "bfloat16", "float32"],
                       help="Output dtype (default: bfloat16)")
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Computation device (default: cuda if available)")
    parser.add_argument("--batch_compute", action="store_true",
                       help="Enable batch computation for memory efficiency")

    args = parser.parse_args()

    print("=" * 80)
    print("SAE to LUT Conversion Script")
    print("=" * 80)

    # 1. Load model
    print(f"\n🚀 Loading model from {args.model_path}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu"  # Keep on CPU, move tensors as needed
        )
        print(f"   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ ERROR: Failed to load model: {e}")
        return 1

    # 2. Parse or auto-detect layers
    checkpoint_base_dir = Path(args.checkpoint_base_dir)
    if args.layers:
        layer_indices = parse_layer_range(args.layers)
        print(f"\n📋 Using specified layers: {layer_indices}")
    else:
        # Auto-detect from first proj_type
        layer_indices = auto_detect_layers(checkpoint_base_dir, args.proj_types[0])
        if not layer_indices:
            print(f"\n✗ ERROR: No layers found in checkpoint directory")
            return 1
        print(f"\n📋 Auto-detected {len(layer_indices)} layers: {layer_indices}")

    # 3. Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir.absolute()}")

    # 4. Load thresholds
    threshold_dir = Path(args.threshold_dir) if args.threshold_dir else None
    thresholds = load_thresholds(threshold_dir, args.proj_types)
    if thresholds:
        print(f"   ✓ Loaded {len(thresholds)} elbow thresholds")

    # 5. Convert dtype string
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    target_dtype = dtype_map[args.dtype]
    print(f"   ✓ Output dtype: {args.dtype}")
    print(f"   ✓ Computation device: {args.device}")

    # 6. Process all layers
    print(f"\n🔄 Processing {len(args.proj_types)} projection types × {len(layer_indices)} layers...")

    all_layer_info = {}
    all_sae_configs = {}

    total = len(args.proj_types) * len(layer_indices)
    with tqdm(total=total, desc="Converting") as pbar:
        for proj_type in args.proj_types:
            for layer_idx in layer_indices:
                pbar.set_description(f"Processing {proj_type} layer {layer_idx}")

                result = process_layer(
                    model,
                    checkpoint_base_dir,
                    output_dir,
                    proj_type,
                    layer_idx,
                    target_dtype,
                    args.device,
                    args.batch_compute
                )

                if result:
                    module_path = result.pop('module_path')
                    sae_config = result.pop('sae_config')

                    # Store layer info using module_path as key
                    all_layer_info[module_path] = {
                        "input_dim": result["input_dim"],
                        "output_dim": result["output_dim"],
                        "file": result["file"]
                    }
                    all_sae_configs[module_path] = sae_config

                    pbar.write(
                        f"  ✓ {module_path}: [{result['input_dim']} → {result['output_dim']}]"
                    )
                else:
                    pbar.write(f"  ✗ Skipped {proj_type} layer {layer_idx}")

                pbar.update(1)

    # 7. Generate metadata
    print(f"\n📝 Generating metadata.json...")
    metadata = generate_metadata(
        args.model_path,
        all_sae_configs,
        all_layer_info,
        thresholds,
        model.config.to_dict()
    )

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✓ Saved to {metadata_path}")

    # 8. Summary
    print("\n" + "=" * 80)
    print("✨ Conversion complete!")
    print("=" * 80)
    print(f"  Files created: {len(all_layer_info)}")
    print(f"  Output directory: {output_dir.absolute()}")

    # Calculate total size
    total_size = sum(f.stat().st_size for f in output_dir.glob("*.safetensors"))
    print(f"  Total size: {total_size / (1024**3):.2f} GB")

    return 0


if __name__ == "__main__":
    exit(main())
