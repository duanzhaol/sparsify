#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoModelForCausalLM, AutoTokenizer

from quantization.llmcompressor_smoothquant import (
    SmoothQuantRecipeConfig,
    build_smoothquant_w8a8_recipe,
    import_llmcompressor_symbols,
    prepare_tokenized_calibration_dataset,
    write_smoothquant_export_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a Qwen3 W8A8 teacher with LLM Compressor SmoothQuant."
    )
    parser.add_argument("--model-path", required=True, help="Local Qwen3 model path")
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Calibration dataset path produced by datasets.save_to_disk",
    )
    parser.add_argument("--output-dir", required=True, help="Export directory")
    parser.add_argument("--num-calibration-samples", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--smoothing-strength", type=float, default=0.8)
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map for calibration/export",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="Transformers torch_dtype for calibration/export",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to HF loaders",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    oneshot, _, _ = import_llmcompressor_symbols()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )

    calibration_dataset = prepare_tokenized_calibration_dataset(
        args.dataset_path,
        num_samples=args.num_calibration_samples,
        max_seq_length=args.max_seq_length,
        shuffle_seed=args.shuffle_seed,
    )
    recipe_cfg = SmoothQuantRecipeConfig(
        smoothing_strength=args.smoothing_strength,
    )
    recipe = build_smoothquant_w8a8_recipe(recipe_cfg)

    oneshot(
        model=model,
        dataset=calibration_dataset,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=len(calibration_dataset),
    )

    model.save_pretrained(args.output_dir, save_compressed=True)
    tokenizer.save_pretrained(args.output_dir)
    manifest_path = write_smoothquant_export_manifest(
        args.output_dir,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        calibration_size=len(calibration_dataset),
        max_seq_length=args.max_seq_length,
        recipe_cfg=recipe_cfg,
    )
    print(f"Saved SmoothQuant W8A8 teacher to {args.output_dir}")
    print(f"Saved export manifest to {manifest_path}")


if __name__ == "__main__":
    main()
