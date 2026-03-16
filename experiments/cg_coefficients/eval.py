"""Evaluate CG coefficients vs inner-product coefficients on a pretrained SAE.

Usage (with SAE checkpoint):
    python -m experiments.cg_coefficients.eval \
        --checkpoint <path_to_sae_checkpoint> \
        --model <model_name> \
        --hookpoint <target_hookpoint> \
        --num_samples 1024 \
        --cg_max_iter 10

Usage (with LUT file):
    python -m experiments.cg_coefficients.eval \
        --lut_dir /root/models/Qwen3-0.6B/lut \
        --lut_layer layers.10.mlp.gate_up_proj \
        --model /root/models/Qwen3-0.6B \
        --hookpoint model.layers.10.mlp.up_proj \
        --num_samples 1024 \
        --cg_max_iter 10
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path so we can import sparsify
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from safetensors import safe_open

from sparsify.config import SparseCoderConfig
from sparsify.data import chunk_and_tokenize
from sparsify.sparse_coder import SparseCoder

from .cg_solver import cg_solve, exact_solve


def load_sae_from_lut(
    lut_dir: str, layer_name: str, k: int | None = None, device: str | torch.device = "cpu"
) -> SparseCoder:
    """Load a SparseCoder from a LUT safetensors file.

    LUT files contain: encoder_weight, encoder_bias, decoder_weight, decoder_bias.
    If k is None, reads from metadata.json in lut_dir (falls back to 128).
    """
    lut_path = Path(lut_dir) / f"{layer_name}.lut.safetensors"
    if not lut_path.exists():
        raise FileNotFoundError(f"LUT file not found: {lut_path}")

    # Auto-detect k from metadata.json if not specified
    if k is None:
        metadata_path = Path(lut_dir) / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
            k = meta.get("sae_config", {}).get("k_active", 128)
            print(f"  Auto-detected k={k} from metadata.json")
        else:
            k = 128

    with safe_open(str(lut_path), framework="pt", device=str(device)) as f:
        encoder_weight = f.get_tensor("encoder_weight")  # (N, d_in)
        encoder_bias = f.get_tensor("encoder_bias")  # (N,)
        decoder_weight = f.get_tensor("decoder_weight")  # (N, d_in)
        decoder_bias = f.get_tensor("decoder_bias")  # (d_in,)

    num_latents, d_in = encoder_weight.shape
    cfg = SparseCoderConfig(num_latents=num_latents, k=k)
    sae = SparseCoder(d_in, cfg, device=device, dtype=encoder_weight.dtype)

    sae.encoder.weight.data.copy_(encoder_weight)
    sae.encoder.bias.data.copy_(encoder_bias)
    sae.W_dec.data.copy_(decoder_weight)
    sae.b_dec.data.copy_(decoder_bias)

    return sae


# ── Global accumulators (fixes batch-average bias) ──────────────────────

@dataclass
class GlobalAccumulator:
    """Accumulates raw sums across all batches for correct global metrics."""

    # Sum of squared errors (numerators for MSE / FVU)
    sse_inner: float = 0.0
    sse_cg_enc: float = 0.0
    sse_cg_zero: float = 0.0
    sse_exact: float = 0.0

    # Total variance (FVU denominator) – accumulated per-batch with batch mean
    # We also accumulate sum(x) and sum(x^2) for a true global variance pass
    sum_x: Tensor | None = None
    sum_x_sq: Tensor | None = None

    # Total element count (for MSE denominator: num_samples * d_in)
    total_elements: int = 0
    # Total sample count
    total_samples: int = 0

    # Exceed: count / denom per tau
    exceed_count_inner: dict[float, int] = field(default_factory=dict)
    exceed_count_cg: dict[float, int] = field(default_factory=dict)
    exceed_count_exact: dict[float, int] = field(default_factory=dict)
    exceed_denom: dict[float, int] = field(default_factory=dict)

    # CG convergence (still averaged, fine for diagnostics)
    cg_iters_encoder_init: list[float] = field(default_factory=list)
    cg_iters_zero_init: list[float] = field(default_factory=list)
    residuals_encoder_init: list[list[float]] = field(default_factory=list)
    residuals_zero_init: list[list[float]] = field(default_factory=list)
    condition_numbers: list[float] = field(default_factory=list)


@torch.no_grad()
def evaluate_batch(
    sae: SparseCoder,
    x_orig: Tensor,
    cg_max_iter: int,
    tau_values: list[float],
    elbow_threshold: float,
    acc: GlobalAccumulator,
) -> dict:
    """Evaluate one batch, accumulate into GlobalAccumulator. Returns batch summary."""

    # ── Encode in SAE dtype (selection only) ────────────────────────────
    x_sae = x_orig.to(sae.dtype)
    top_acts, top_indices, _ = sae.encode(x_sae)

    # ── Everything else in float32 ──────────────────────────────────────
    x = x_orig.float()
    W_dec_f32 = sae.W_dec.float()
    b_dec_f32 = sae.b_dec.float()

    # Method A: inner-product coefficients (cast to f32 for fair comparison)
    top_acts_f32 = top_acts.float()
    B_S = W_dec_f32[top_indices]  # (batch, K, h)
    recon_inner = torch.bmm(top_acts_f32.unsqueeze(1), B_S).squeeze(1) + b_dec_f32

    target = x - b_dec_f32  # (batch, h)

    # Method B: CG with encoder init (float32 inside cg_solve)
    alpha_cg_enc, info_enc = cg_solve(
        B_S, target, alpha_init=top_acts_f32,
        max_iter=cg_max_iter, record_residuals=True,
    )
    recon_cg_enc = torch.bmm(alpha_cg_enc.unsqueeze(1), B_S).squeeze(1) + b_dec_f32

    # Method C: CG with zero init (float32)
    alpha_cg_zero, info_zero = cg_solve(
        B_S, target, alpha_init=None,
        max_iter=cg_max_iter, record_residuals=True,
    )
    recon_cg_zero = torch.bmm(alpha_cg_zero.unsqueeze(1), B_S).squeeze(1) + b_dec_f32

    # Method D: Exact least-squares baseline (float32)
    alpha_exact = exact_solve(B_S, target)
    recon_exact = torch.bmm(alpha_exact.unsqueeze(1), B_S).squeeze(1) + b_dec_f32

    # ── Accumulate global SSE ───────────────────────────────────────────
    batch_sse_inner = (x - recon_inner).pow(2).sum().item()
    batch_sse_cg_enc = (x - recon_cg_enc).pow(2).sum().item()
    batch_sse_cg_zero = (x - recon_cg_zero).pow(2).sum().item()
    batch_sse_exact = (x - recon_exact).pow(2).sum().item()

    acc.sse_inner += batch_sse_inner
    acc.sse_cg_enc += batch_sse_cg_enc
    acc.sse_cg_zero += batch_sse_cg_zero
    acc.sse_exact += batch_sse_exact

    n = x.shape[0]
    d = x.shape[1]
    acc.total_elements += n * d
    acc.total_samples += n

    # Accumulate for global variance
    if acc.sum_x is None:
        acc.sum_x = x.sum(dim=0).cpu()
        acc.sum_x_sq = x.pow(2).sum(dim=0).cpu()
    else:
        acc.sum_x += x.sum(dim=0).cpu()
        acc.sum_x_sq += x.pow(2).sum(dim=0).cpu()

    # ── Accumulate exceed counts ────────────────────────────────────────
    abs_err_inner = (x - recon_inner).abs()
    abs_err_cg = (x - recon_cg_enc).abs()
    abs_err_exact = (x - recon_exact).abs()
    num_elements = n * d

    for tau in tau_values:
        threshold = tau * elbow_threshold
        c_inner = int((abs_err_inner > threshold).sum().item())
        c_cg = int((abs_err_cg > threshold).sum().item())
        c_exact = int((abs_err_exact > threshold).sum().item())

        acc.exceed_count_inner[tau] = acc.exceed_count_inner.get(tau, 0) + c_inner
        acc.exceed_count_cg[tau] = acc.exceed_count_cg.get(tau, 0) + c_cg
        acc.exceed_count_exact[tau] = acc.exceed_count_exact.get(tau, 0) + c_exact
        acc.exceed_denom[tau] = acc.exceed_denom.get(tau, 0) + num_elements

    # ── Diagnostics (batch-level, fine for these) ───────────────────────
    acc.cg_iters_encoder_init.append(info_enc["iters_mean"])
    acc.cg_iters_zero_init.append(info_zero["iters_mean"])
    acc.residuals_encoder_init.append(info_enc["residuals"])
    acc.residuals_zero_init.append(info_zero["residuals"])

    # Condition number
    gram = torch.bmm(B_S, B_S.transpose(1, 2))
    svd_vals = torch.linalg.svdvals(gram.float())
    cond = (svd_vals[:, 0] / svd_vals[:, -1].clamp(min=1e-12))
    acc.condition_numbers.append(cond.median().item())

    # Return batch-level summary for progress display
    batch_mse_inner = batch_sse_inner / (n * d)
    batch_mse_cg = batch_sse_cg_enc / (n * d)
    batch_mse_exact = batch_sse_exact / (n * d)
    return {
        "mse_inner": batch_mse_inner,
        "mse_cg": batch_mse_cg,
        "mse_exact": batch_mse_exact,
    }


def finalize_metrics(acc: GlobalAccumulator, tau_values: list[float]) -> dict:
    """Compute final global metrics from accumulated data."""

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    N = acc.total_elements  # total scalar elements
    n = acc.total_samples

    # Global MSE
    mse_inner = acc.sse_inner / N
    mse_cg_enc = acc.sse_cg_enc / N
    mse_cg_zero = acc.sse_cg_zero / N
    mse_exact = acc.sse_exact / N

    # Global FVU: SSE / total_variance
    # total_variance = sum_i sum_j (x_ij - mean_j)^2
    # = sum_j [ sum_i x_ij^2 - n * mean_j^2 ]
    # = sum(x^2) - n * sum(mean^2)
    mean_x = acc.sum_x / n  # (d,)
    total_var = (acc.sum_x_sq.sum() - n * mean_x.pow(2).sum()).item()
    total_var = max(total_var, 1e-12)

    fvu_inner = acc.sse_inner / total_var
    fvu_cg_enc = acc.sse_cg_enc / total_var
    fvu_cg_zero = acc.sse_cg_zero / total_var
    fvu_exact = acc.sse_exact / total_var

    # Reductions (relative to inner-product baseline)
    def reduction(baseline, improved):
        return (baseline - improved) / baseline * 100 if baseline > 0 else 0.0

    result = {
        "MSE_inner": mse_inner,
        "MSE_cg_encoder_init": mse_cg_enc,
        "MSE_cg_zero_init": mse_cg_zero,
        "MSE_exact": mse_exact,
        "MSE_reduction_cg_encoder_init_%": reduction(mse_inner, mse_cg_enc),
        "MSE_reduction_cg_zero_init_%": reduction(mse_inner, mse_cg_zero),
        "MSE_reduction_exact_%": reduction(mse_inner, mse_exact),
        "FVU_inner": fvu_inner,
        "FVU_cg_encoder_init": fvu_cg_enc,
        "FVU_cg_zero_init": fvu_cg_zero,
        "FVU_exact": fvu_exact,
        "cg_iters_encoder_init": avg(acc.cg_iters_encoder_init),
        "cg_iters_zero_init": avg(acc.cg_iters_zero_init),
        "condition_number_median": avg(acc.condition_numbers),
        "total_samples": n,
        "total_elements": N,
    }

    # Exceed ratios
    for tau in sorted(tau_values):
        denom = acc.exceed_denom.get(tau, 1)
        p_inner = acc.exceed_count_inner.get(tau, 0) / denom
        p_cg = acc.exceed_count_cg.get(tau, 0) / denom
        p_exact = acc.exceed_count_exact.get(tau, 0) / denom
        result[f"exceed_ratio_inner_tau={tau}"] = p_inner
        result[f"exceed_ratio_cg_tau={tau}"] = p_cg
        result[f"exceed_ratio_exact_tau={tau}"] = p_exact
        result[f"p_reduction_cg_tau={tau}_%"] = reduction(p_inner, p_cg)
        result[f"p_reduction_exact_tau={tau}_%"] = reduction(p_inner, p_exact)

    # Residual histories (averaged across batches)
    for label, res_list in [
        ("residual_history_encoder_init", acc.residuals_encoder_init),
        ("residual_history_zero_init", acc.residuals_zero_init),
    ]:
        if res_list:
            max_len = max(len(r) for r in res_list)
            avg_res = []
            for i in range(max_len):
                vals = [r[i] for r in res_list if i < len(r)]
                avg_res.append(avg(vals))
            result[label] = avg_res

    return result


# ── Data loading ────────────────────────────────────────────────────────

def load_dataset_auto(dataset_path: str, tokenizer=None, max_seq_len: int = 2048):
    """Load dataset from local arrow files, local parquet, or HuggingFace hub."""
    from datasets import Dataset as HFDataset
    from datasets import concatenate_datasets, load_dataset

    path = Path(dataset_path)

    # Local directory with arrow files (pre-tokenized)
    if path.is_dir():
        arrow_files = sorted(path.glob("data-*.arrow"))
        if arrow_files:
            # Load ALL shards, not just the first one
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

    # HuggingFace hub dataset
    ds = load_dataset(dataset_path, split="train")
    if tokenizer:
        ds = chunk_and_tokenize(ds, tokenizer, max_seq_len=max_seq_len)
    return ds


def collect_activations(
    model,
    tokenizer,
    hookpoint: str,
    num_samples: int,
    batch_size: int = 8,
    dataset_path: str = "togethercomputer/RedPajama-Data-1T-Sample",
    max_seq_len: int = 2048,
) -> Tensor:
    """Collect model activations at the specified hookpoint.

    Captures the INPUT to the specified module (matching SAE training convention).
    """
    dataset = load_dataset_auto(dataset_path, tokenizer, max_seq_len=max_seq_len)

    activations = []
    collected = 0

    target_module = model.get_submodule(hookpoint)
    captured = []

    def hook_fn(module, inputs, output):
        inp = inputs[0] if isinstance(inputs, tuple) else inputs
        captured.append(inp.detach())

    handle = target_module.register_forward_hook(hook_fn)

    try:
        for i in range(0, min(len(dataset), num_samples * 2), batch_size):
            if collected >= num_samples:
                break

            batch = dataset[i : i + batch_size]
            input_ids = batch["input_ids"]
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids)
            input_ids = input_ids.to(model.device)

            captured.clear()
            with torch.no_grad():
                model(input_ids)

            if captured:
                act = captured[0].reshape(-1, captured[0].shape[-1])
                activations.append(act.cpu())
                collected += act.shape[0]
    finally:
        handle.remove()

    all_acts = torch.cat(activations, dim=0)[:num_samples]
    return all_acts


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CG coefficients vs inner-product coefficients"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to SAE checkpoint directory"
    )
    parser.add_argument(
        "--lut_dir", type=str, default=None,
        help="Path to LUT directory (alternative to --checkpoint)"
    )
    parser.add_argument(
        "--lut_layer", type=str, default=None,
        help="Layer name in LUT dir, e.g. 'layers.10.mlp.gate_up_proj'"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--hookpoint", type=str, required=True,
        help="Target hookpoint in the model (e.g. 'model.layers.10.mlp.up_proj')"
    )
    parser.add_argument(
        "--num_samples", type=int, default=1024, help="Number of activation samples"
    )
    parser.add_argument(
        "--cg_max_iter", type=int, default=10, help="Maximum CG iterations"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=128,
        help="Batch size for evaluation (SAE forward)",
    )
    parser.add_argument(
        "--dataset", type=str,
        default="togethercomputer/RedPajama-Data-1T-Sample",
        help="Dataset path (local dir with arrow/parquet files, or HF hub name)",
    )
    parser.add_argument(
        "--tau_values", type=float, nargs="+",
        default=[0.25, 0.3, 0.5],
        help="Tau values for exceed ratio",
    )
    parser.add_argument(
        "--elbow_threshold_file", type=str, default=None,
        help="Path to elbow threshold JSON file (e.g. thresholds_up.json)",
    )
    parser.add_argument(
        "--elbow_key", type=str, default=None,
        help="Key in elbow threshold file (e.g. 'layer_10/mlp_up_proj')",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON file for results"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto, cpu, cuda, npu)"
    )

    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load SAE
    if args.lut_dir:
        if not args.lut_layer:
            parser.error("--lut_layer is required when using --lut_dir")
        print(f"Loading SAE from LUT: {args.lut_dir}/{args.lut_layer}...")
        sae = load_sae_from_lut(args.lut_dir, args.lut_layer, device=device)
        checkpoint_label = f"{args.lut_dir}/{args.lut_layer}"
    elif args.checkpoint:
        print(f"Loading SAE from checkpoint: {args.checkpoint}...")
        sae = SparseCoder.load_from_disk(args.checkpoint, device=device)
        checkpoint_label = args.checkpoint
    else:
        parser.error("Either --checkpoint or --lut_dir must be specified")
    sae.eval()
    print(f"  d_in={sae.d_in}, num_latents={sae.num_latents}, k={sae.cfg.k}")

    # Load model and collect activations
    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, device_map={"": device}
    )
    model.eval()

    print(f"Collecting {args.num_samples} activations at '{args.hookpoint}'...")
    all_acts = collect_activations(
        model,
        tokenizer,
        args.hookpoint,
        args.num_samples,
        dataset_path=args.dataset,
    )
    print(f"  Collected {all_acts.shape[0]} activations, dim={all_acts.shape[1]}")

    # Free model memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load elbow threshold
    elbow_threshold = 1.0
    if args.elbow_threshold_file:
        with open(args.elbow_threshold_file) as f:
            elbow_data = json.load(f)
        if args.elbow_key and args.elbow_key in elbow_data:
            elbow_threshold = elbow_data[args.elbow_key]["elbow_value"]
        else:
            for json_key, value in elbow_data.items():
                if json_key in args.hookpoint or args.hookpoint.replace(".", "_") in json_key:
                    elbow_threshold = value["elbow_value"]
                    break
        print(f"  Elbow threshold: {elbow_threshold}")

    # Evaluate with global accumulator
    print(f"Evaluating (CG max_iter={args.cg_max_iter}, solve dtype=float32)...")
    acc = GlobalAccumulator()
    for i in range(0, all_acts.shape[0], args.eval_batch_size):
        batch = all_acts[i : i + args.eval_batch_size].to(device)
        batch_info = evaluate_batch(
            sae, batch, args.cg_max_iter, args.tau_values, elbow_threshold, acc,
        )
        batch_num = i // args.eval_batch_size + 1
        print(f"  Batch {batch_num}: "
              f"MSE_inner={batch_info['mse_inner']:.6f}, "
              f"MSE_cg={batch_info['mse_cg']:.6f}, "
              f"MSE_exact={batch_info['mse_exact']:.6f}")

    # Finalize global metrics
    results = finalize_metrics(acc, args.tau_values)

    # ── Print results ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS (all metrics are global, not batch-averaged)")
    print("=" * 70)
    print(f"  MSE inner-product:       {results['MSE_inner']:.6f}")
    print(f"  MSE CG (encoder init):   {results['MSE_cg_encoder_init']:.6f}")
    print(f"  MSE CG (zero init):      {results['MSE_cg_zero_init']:.6f}")
    print(f"  MSE exact (lstsq):       {results['MSE_exact']:.6f}")
    print(f"  MSE reduction CG (enc):  {results['MSE_reduction_cg_encoder_init_%']:.2f}%")
    print(f"  MSE reduction CG (zero): {results['MSE_reduction_cg_zero_init_%']:.2f}%")
    print(f"  MSE reduction exact:     {results['MSE_reduction_exact_%']:.2f}%")
    print()
    print(f"  FVU inner-product:       {results['FVU_inner']:.6f}")
    print(f"  FVU CG (encoder init):   {results['FVU_cg_encoder_init']:.6f}")
    print(f"  FVU CG (zero init):      {results['FVU_cg_zero_init']:.6f}")
    print(f"  FVU exact (lstsq):       {results['FVU_exact']:.6f}")
    print()
    print(f"  CG iters (encoder init): {results['cg_iters_encoder_init']:.2f}")
    print(f"  CG iters (zero init):    {results['cg_iters_zero_init']:.2f}")
    print(f"  Condition number (med):  {results['condition_number_median']:.2f}")
    print()

    for tau in sorted(args.tau_values):
        pi = results.get(f"exceed_ratio_inner_tau={tau}", 0)
        pc = results.get(f"exceed_ratio_cg_tau={tau}", 0)
        pe = results.get(f"exceed_ratio_exact_tau={tau}", 0)
        rc = results.get(f"p_reduction_cg_tau={tau}_%", 0)
        re = results.get(f"p_reduction_exact_tau={tau}_%", 0)
        print(f"  tau={tau}: p_inner={pi:.4f}, p_cg={pc:.4f}, p_exact={pe:.4f}, "
              f"p_red_cg={rc:.2f}%, p_red_exact={re:.2f}%")

    if "residual_history_encoder_init" in results:
        print()
        print("  Residual history (encoder init):")
        for i, r in enumerate(results["residual_history_encoder_init"]):
            print(f"    iter {i}: {r:.2e}")
        print("  Residual history (zero init):")
        for i, r in enumerate(results["residual_history_zero_init"]):
            print(f"    iter {i}: {r:.2e}")

    # ── Verdict ─────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    mse_red_cg = results["MSE_reduction_cg_encoder_init_%"]
    mse_red_exact = results["MSE_reduction_exact_%"]
    cond = results["condition_number_median"]
    p_red_cg_03 = results.get("p_reduction_cg_tau=0.3_%", 0.0)
    p_red_exact_03 = results.get("p_reduction_exact_tau=0.3_%", 0.0)

    # CG vs exact gap tells us if CG is converged
    cg_exact_gap = mse_red_exact - mse_red_cg
    print(f"  CG-to-exact gap: {cg_exact_gap:.2f}% "
          f"(CG={mse_red_cg:.2f}%, exact={mse_red_exact:.2f}%)")

    if cg_exact_gap > 2.0:
        print("  NOTE: CG has not fully converged to exact solution. "
              "Consider increasing --cg_max_iter.")

    if mse_red_exact > 10 or p_red_exact_03 > 15:
        print(f"VERDICT: SUCCESS (exact ceiling) - proceed to Phase 2")
        if mse_red_cg > 10 or p_red_cg_03 > 15:
            print(f"  CG with {args.cg_max_iter} iters already captures most of the gain.")
        else:
            print(f"  But CG({args.cg_max_iter} iters) doesn't fully realize it yet.")
    elif 5 <= mse_red_exact <= 10 and cond > 20:
        print("VERDICT: POTENTIAL - exact ceiling moderate, condition number high")
    elif 5 <= mse_red_exact <= 10 and cond < 10:
        print("VERDICT: LOW CEILING - basis vectors near-orthogonal, confirmed by exact solve")
    elif mse_red_exact < 5:
        print("VERDICT: INSUFFICIENT - even exact LS gives < 5% MSE reduction")
    else:
        print("VERDICT: MARGINAL - needs further investigation")

    print(f"  (MSE_red_cg={mse_red_cg:.2f}%, MSE_red_exact={mse_red_exact:.2f}%, "
          f"cond={cond:.1f}, p_red_cg(0.3)={p_red_cg_03:.2f}%, "
          f"p_red_exact(0.3)={p_red_exact_03:.2f}%)")
    print("=" * 70)

    # ── Save results ────────────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("experiments/cg_coefficients/results") / "cg_eval_results.json"

    serializable = {}
    for k, v in results.items():
        if isinstance(v, list):
            serializable[k] = [float(x) for x in v]
        elif isinstance(v, (int, float)):
            serializable[k] = v
        else:
            serializable[k] = float(v)

    serializable["config"] = {
        "checkpoint": checkpoint_label,
        "model": args.model,
        "hookpoint": args.hookpoint,
        "num_samples": args.num_samples,
        "cg_max_iter": args.cg_max_iter,
        "tau_values": args.tau_values,
        "elbow_threshold": elbow_threshold,
        "solve_dtype": "float32",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
