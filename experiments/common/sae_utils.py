"""SAE utilities shared across experiments."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from safetensors import safe_open

from sparsify.config import SparseCoderConfig
from sparsify.sparse_coder import SparseCoder


def load_sae_from_lut(
    lut_dir: str,
    layer_name: str,
    k: int | None = None,
    device: str | torch.device = "cpu",
) -> SparseCoder:
    """Load a SparseCoder from a LUT safetensors file.

    Args:
        lut_dir: Path to the LUT directory.
        layer_name: Layer name key in the LUT dir (e.g. 'layers.7.mlp.gate_up_proj').
        k: Number of top-K latents. Auto-detected from metadata.json if None.
        device: Target device.
    """
    lut_path = Path(lut_dir) / f"{layer_name}.lut.safetensors"
    if not lut_path.exists():
        raise FileNotFoundError(f"LUT file not found: {lut_path}")

    if k is None:
        metadata_path = Path(lut_dir) / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
            k = meta.get("sae_config", {}).get("k_active", 128)
        else:
            k = 128

    with safe_open(str(lut_path), framework="pt", device=str(device)) as f:
        encoder_weight = f.get_tensor("encoder_weight")
        encoder_bias = f.get_tensor("encoder_bias")
        decoder_weight = f.get_tensor("decoder_weight")
        decoder_bias = f.get_tensor("decoder_bias")

    num_latents, d_in = encoder_weight.shape
    cfg = SparseCoderConfig(num_latents=num_latents, k=k)
    sae = SparseCoder(d_in, cfg, device=device, dtype=encoder_weight.dtype)

    sae.encoder.weight.data.copy_(encoder_weight)
    sae.encoder.bias.data.copy_(encoder_bias)
    sae.W_dec.data.copy_(decoder_weight)
    sae.b_dec.data.copy_(decoder_bias)

    return sae


# Hookpoint name (model submodule) → LUT layer name mapping for Qwen3
HOOKPOINT_TO_LUT = {
    "mlp.up_proj": "mlp.gate_up_proj",
    "mlp.gate_proj": "mlp.gate_up_proj",
    "self_attn.q_proj": "self_attn.qkv_proj",
    "self_attn.k_proj": "self_attn.qkv_proj",
    "self_attn.v_proj": "self_attn.qkv_proj",
    "self_attn.o_proj": "self_attn.o_proj",
}


def get_layer_hookpoints(
    layer_idx: int,
    op_types: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Get (hookpoint_name, lut_layer_name) pairs for a given layer.

    Args:
        layer_idx: Transformer layer index.
        op_types: Which op types to include. Options: 'mlp', 'qkv', 'o'.
                  Defaults to ['mlp'] (gate_up_proj only).

    Returns:
        List of (hookpoint, lut_layer) tuples.
    """
    if op_types is None:
        op_types = ["mlp"]

    results = []
    for op_type in op_types:
        if op_type == "mlp":
            hookpoint = f"model.layers.{layer_idx}.mlp.up_proj"
            lut_layer = f"layers.{layer_idx}.mlp.gate_up_proj"
        elif op_type == "qkv":
            hookpoint = f"model.layers.{layer_idx}.self_attn.q_proj"
            lut_layer = f"layers.{layer_idx}.self_attn.qkv_proj"
        elif op_type == "o":
            hookpoint = f"model.layers.{layer_idx}.self_attn.o_proj"
            lut_layer = f"layers.{layer_idx}.self_attn.o_proj"
        else:
            raise ValueError(f"Unknown op_type: {op_type}")
        results.append((hookpoint, lut_layer))
    return results


@torch.no_grad()
def encode_topk(sae: SparseCoder, x: Tensor, top_k: int | None = None) -> tuple[Tensor, Tensor]:
    """Encode input through SAE and return top-k indices and values.

    Args:
        sae: The SparseCoder model.
        x: Input tensor of shape (batch, d_in).
        top_k: Number of top activations to return. Defaults to sae.cfg.k.

    Returns:
        top_indices: (batch, top_k) int32 tensor, sorted by activation value (descending).
        top_values: (batch, top_k) float32 tensor.
    """
    if top_k is None:
        top_k = sae.cfg.k
    x_centered = x.to(sae.dtype) - sae.b_dec
    pre_acts = F.relu(F.linear(x_centered, sae.encoder.weight, sae.encoder.bias))
    top_values, top_indices = pre_acts.topk(top_k, dim=-1, sorted=True)
    return top_indices.to(torch.int32), top_values.float()
