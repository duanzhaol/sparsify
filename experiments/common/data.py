"""Data loading and activation collection shared across experiments."""

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from sparsify.utils import partial_forward_to_layer, get_layer_list


def load_dataset_auto(dataset_path: str, tokenizer=None, max_seq_len: int = 2048):
    """Load dataset from local arrow files, local parquet, or HuggingFace hub."""
    from datasets import Dataset as HFDataset
    from datasets import concatenate_datasets, load_dataset

    from sparsify.data import chunk_and_tokenize

    path = Path(dataset_path)

    if path.is_dir():
        arrow_files = sorted(path.glob("data-*.arrow"))
        if arrow_files:
            shards = [HFDataset.from_file(str(f)) for f in arrow_files]
            ds = concatenate_datasets(shards)
            print(f"  Loaded pre-tokenized arrow dataset: {len(ds)} samples "
                  f"from {len(arrow_files)} shards")
            return ds.with_format("torch", columns=["input_ids"])

        parquet_files = sorted(path.glob("*.parquet"))
        if parquet_files:
            ds = load_dataset("parquet", data_files=[str(f) for f in parquet_files],
                              split="train")
            if tokenizer:
                ds = chunk_and_tokenize(ds, tokenizer, max_seq_len=max_seq_len)
            return ds

    ds = load_dataset(dataset_path, split="train")
    if tokenizer:
        ds = chunk_and_tokenize(ds, tokenizer, max_seq_len=max_seq_len)
    return ds


def collect_raw_activations(
    model,
    dataset,
    hookpoints: list[str],
    num_samples: int = 256,
    seq_len: int = 512,
    batch_size: int = 4,
    device: str | torch.device = "cuda",
) -> dict[str, dict[str, Any]]:
    """Collect raw activations from multiple hookpoints in a single forward pass.

    Follows the same pattern as sparsify.trainer: register hooks on ALL target
    modules simultaneously, run ONE forward pass per batch, collect all activations.

    Args:
        model: The pretrained causal LM.
        dataset: Tokenized dataset with 'input_ids' column.
        hookpoints: List of hookpoint names (e.g. ['model.layers.7.mlp.up_proj',
                    'model.layers.14.self_attn.o_proj']).
        num_samples: Number of sequences to process.
        seq_len: Max sequence length (truncate if longer).
        batch_size: Batch size for model forward.
        device: Device to run on.

    Returns:
        Dict mapping hookpoint name to:
            'activations': list[Tensor] — one (seq_len, d_in) tensor per sequence
            'seq_boundaries': list[int] — cumulative token counts [0, S, 2S, ...]
            'd_in': int — input dimension at this hookpoint
    """
    # Resolve max layer for partial forward optimization.
    layers_name, _ = get_layer_list(model)
    max_layer_idx = _infer_max_layer_idx(hookpoints, layers_name)

    # Register hooks on all target modules simultaneously (like trainer.py:538-541)
    name_to_module = {
        name: model.get_submodule(name)
        for name in hookpoints
    }
    module_to_name = {v: k for k, v in name_to_module.items()}

    # Per-hookpoint captured activations for current batch
    captured: dict[str, Tensor] = {}

    def hook_fn(module, inputs, output):
        inp = inputs[0] if isinstance(inputs, tuple) else inputs
        name = module_to_name[module]
        captured[name] = inp.detach()

    # Per-hookpoint accumulated results
    results: dict[str, dict[str, Any]] = {}
    for hp in hookpoints:
        results[hp] = {"activations": [], "seq_boundaries": [0], "d_in": None}

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    sequences_done = 0

    for batch in loader:
        if sequences_done >= num_samples:
            break

        input_ids = batch["input_ids"]
        if isinstance(input_ids, list):
            input_ids = torch.stack(input_ids)
        input_ids = input_ids[:, :seq_len].to(device)
        actual_batch_size = input_ids.shape[0]

        # Register hooks fresh each batch (like trainer.py:538-541)
        captured.clear()
        handles = [
            mod.register_forward_hook(hook_fn)
            for mod in name_to_module.values()
        ]
        try:
            with torch.no_grad():
                if max_layer_idx is not None:
                    partial_forward_to_layer(model, input_ids, max_layer_idx)
                else:
                    model(input_ids)
        finally:
            for h in handles:
                h.remove()

        # Distribute captured activations per sequence
        for hp in hookpoints:
            act = captured.get(hp)
            if act is None:
                continue
            # act shape: (batch, seq_len, d_in)
            B, S, D = act.shape
            if results[hp]["d_in"] is None:
                results[hp]["d_in"] = D

            total_tokens = results[hp]["seq_boundaries"][-1]
            for b in range(actual_batch_size):
                if sequences_done + b >= num_samples:
                    break
                results[hp]["activations"].append(act[b].cpu())  # (S, D)
                total_tokens += S
                results[hp]["seq_boundaries"].append(total_tokens)

        sequences_done += actual_batch_size
        if sequences_done % 32 == 0 or sequences_done >= num_samples:
            print(f"  Collected {min(sequences_done, num_samples)}/{num_samples} sequences")

    # Report summary
    for hp in hookpoints:
        n_seq = len(results[hp]["activations"])
        d_in = results[hp]["d_in"]
        total = results[hp]["seq_boundaries"][-1] if n_seq > 0 else 0
        print(f"  {hp}: {n_seq} seqs, {total} tokens, d_in={d_in}")

    return results


def _infer_max_layer_idx(hookpoints: list[str], layers_name: str) -> int | None:
    """Infer the maximum transformer layer index from hookpoint names.

    This handles hookpoints like ``model.layers.7.mlp.up_proj`` even when
    ``layers_name`` itself contains dots.
    """
    layer_parts = layers_name.split(".")
    max_idx = None

    for hookpoint in hookpoints:
        parts = hookpoint.split(".")
        for start in range(len(parts) - len(layer_parts)):
            if parts[start : start + len(layer_parts)] != layer_parts:
                continue

            idx_pos = start + len(layer_parts)
            if idx_pos >= len(parts):
                break

            try:
                layer_idx = int(parts[idx_pos])
            except ValueError:
                break

            max_idx = layer_idx if max_idx is None else max(max_idx, layer_idx)
            break

    return max_idx


def encode_activations(
    raw_data: dict[str, dict[str, Any]],
    lut_dir: str,
    hookpoint_to_lut: dict[str, str],
    device: str | torch.device = "cuda",
    top_mul: int = 2,
) -> dict[str, dict[str, Any]]:
    """Encode raw activations through SAE, returning top-K*top_mul indices/values.

    This is step 2: takes raw activations from collect_raw_activations() and
    encodes them through the corresponding SAE. Separated so that the encoding
    step can be swapped for other transforms in future experiments.

    Args:
        raw_data: Output of collect_raw_activations().
        lut_dir: Path to LUT directory.
        hookpoint_to_lut: Mapping from hookpoint name to LUT layer name
            (e.g. {'model.layers.7.mlp.up_proj': 'layers.7.mlp.gate_up_proj'}).
        device: Device for SAE computation.
        top_mul: Multiplier for K (default 2 → top-2K).

    Returns:
        Dict mapping lut_layer_name to:
            'top_indices': Tensor[total_tokens, top_mul*K] (int32)
            'top_values':  Tensor[total_tokens, top_mul*K] (float32)
            'seq_boundaries': list[int]
            'K': int
            'N': int
    """
    from .sae_utils import load_sae_from_lut, encode_topk

    results = {}

    for hookpoint, lut_layer in hookpoint_to_lut.items():
        hp_data = raw_data.get(hookpoint)
        if hp_data is None or not hp_data["activations"]:
            print(f"  WARNING: No activations for {hookpoint}, skipping")
            continue

        print(f"\n  Encoding {lut_layer} (from {hookpoint})...")
        sae = load_sae_from_lut(lut_dir, lut_layer, device=device)
        sae.eval()
        K = sae.cfg.k
        N = sae.num_latents
        top_k = top_mul * K
        print(f"    SAE: d_in={sae.d_in}, N={N}, K={K}, collecting top-{top_k}")

        all_indices = []
        all_values = []

        # Process each sequence's activations
        for seq_act in hp_data["activations"]:
            # seq_act: (seq_len, d_in), on CPU
            # Encode in chunks on device
            chunk_size = 2048
            seq_indices = []
            seq_values = []
            for i in range(0, seq_act.shape[0], chunk_size):
                chunk = seq_act[i:i+chunk_size].to(device)
                idx, val = encode_topk(sae, chunk, top_k=top_k)
                seq_indices.append(idx.cpu())
                seq_values.append(val.cpu())

            all_indices.append(torch.cat(seq_indices, dim=0))
            all_values.append(torch.cat(seq_values, dim=0))

        results[lut_layer] = {
            "top_indices": torch.cat(all_indices, dim=0),
            "top_values": torch.cat(all_values, dim=0),
            "seq_boundaries": hp_data["seq_boundaries"],
            "K": K,
            "N": N,
        }

        total = results[lut_layer]["top_indices"].shape[0]
        print(f"    Done: {total} tokens, shape {results[lut_layer]['top_indices'].shape}")

        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results
