"""Checkpoint save/load utilities for SAE training."""

import json
import logging
import os
import re
from glob import glob
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_model
from schedulefree import ScheduleFreeWrapper

from .hadamard import HadamardRotation
from .sparse_coder import SparseCoder
from .tiled_sparse_coder import TiledSparseCoder

logger = logging.getLogger(__name__)


def is_tiled_checkpoint(path: str | Path) -> bool:
    """Check if a checkpoint path contains a TiledSparseCoder checkpoint."""
    path = Path(path)
    cfg_path = path / "cfg.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
            return "num_tiles" in cfg and cfg["num_tiles"] > 1
    return False


def get_checkpoint_num_tiles(path: str | Path) -> int:
    """Get the num_tiles value from a checkpoint, returns 1 if not tiled."""
    path = Path(path)
    cfg_path = path / "cfg.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
            return cfg.get("num_tiles", 1)
    return 1


def load_sae_checkpoint(sae, path: str | Path, device: str) -> None:
    """Load SAE checkpoint, handling both regular and tiled formats."""
    path = Path(path)
    checkpoint_num_tiles = get_checkpoint_num_tiles(path)

    if isinstance(sae, TiledSparseCoder):
        if checkpoint_num_tiles == 1:
            raise TypeError(
                f"Checkpoint at {path} is regular SparseCoder (num_tiles=1) "
                f"but current config has num_tiles={sae.num_tiles}. "
                f"Cannot resume/finetune tiled SAE from non-tiled checkpoint."
            )
        if checkpoint_num_tiles != sae.num_tiles:
            raise ValueError(
                f"Checkpoint at {path} has num_tiles={checkpoint_num_tiles} "
                f"but current config has num_tiles={sae.num_tiles}. "
                f"num_tiles must match for resume/finetune."
            )
        for i, tile_sae in enumerate(sae.saes):
            tile_path = path / f"tile_{i}" / "sae.safetensors"
            load_model(tile_sae, str(tile_path), device=device)
    else:
        if checkpoint_num_tiles > 1:
            raise TypeError(
                f"Checkpoint at {path} is TiledSparseCoder (num_tiles={checkpoint_num_tiles}) "
                f"but current config has num_tiles=1. "
                f"Cannot resume/finetune regular SAE from tiled checkpoint."
            )
        load_model(sae, str(path / "sae.safetensors"), device=device)


def expand_range_pattern(pattern: str) -> list[str]:
    """Expand hookpoint patterns with range syntax.

    Supports syntax like:
    - layers.[1-10].self_attn.o_proj  -> layers.1...layers.10
    - layers.[0-5,10,15].mlp.act     -> layers.0...5, 10, 15
    - layers.*.xxx                    -> unchanged (normal glob)
    """
    match = re.search(r'\[([0-9,\-]+)\]', pattern)
    if not match:
        return [pattern]

    range_spec = match.group(1)
    numbers = []

    for part in range_spec.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            numbers.extend(range(start, end + 1))
        else:
            numbers.append(int(part))

    numbers = sorted(set(numbers))
    return [pattern.replace(f'[{range_spec}]', str(num)) for num in numbers]


class CheckpointMixin:
    """Mixin providing checkpoint save/load methods for Trainer."""

    def _load_elbow_thresholds(self, path: str):
        """Load elbow thresholds from JSON file and match to hookpoints."""
        with open(path, 'r') as f:
            elbow_data = json.load(f)

        for hookpoint in self.cfg.hookpoints:
            matched = False

            # Strategy 1: Direct match
            if hookpoint in elbow_data:
                self.elbow_thresholds[hookpoint] = elbow_data[hookpoint]["elbow_value"]
                matched = True
                continue

            # Strategy 2: Extract layer number and component
            parts = hookpoint.split('.')
            if len(parts) >= 2 and parts[0] in ('layers', 'h', 'model.layers'):
                layer_num = parts[1]
                component = '.'.join(parts[2:]) if len(parts) > 2 else ''

                search_patterns = []
                if component:
                    component_underscore = component.replace('.', '_')
                    search_patterns.extend([
                        f"layer_{layer_num}/{component}",
                        f"layer_{layer_num}/{component_underscore}",
                    ])
                search_patterns.append(f"layer_{layer_num}")

                for pattern in search_patterns:
                    for json_key, value in elbow_data.items():
                        if pattern in json_key or json_key in hookpoint:
                            self.elbow_thresholds[hookpoint] = value["elbow_value"]
                            matched = True
                            break
                    if matched:
                        break

            if not matched:
                logger.warning(f"No elbow threshold found for hookpoint '{hookpoint}'")

        logger.info(
            f"Loaded elbow thresholds for {len(self.elbow_thresholds)}/{len(self.cfg.hookpoints)} hookpoints"
        )

    def load_state(self, path: str):
        """Load the trainer state from disk."""
        device = self.model.device

        train_state = torch.load(
            f"{path}/state.pt", map_location=device, weights_only=True
        )
        self.global_step = train_state["global_step"]
        self.total_tokens = train_state.get("total_tokens", 0)

        for file in glob(f"{path}/rank_*_state.pt"):
            rank_state = torch.load(file, map_location=device, weights_only=True)

            for k in self.saes:
                if k in rank_state["num_tokens_since_fired"]:
                    self.num_tokens_since_fired[k] = rank_state[
                        "num_tokens_since_fired"
                    ][k]

                if not isinstance(rank_state["best_loss"], dict):
                    old_val = rank_state["best_loss"]
                    self.best_loss = {name: old_val for name in self.saes}
                elif k in rank_state["best_loss"]:
                    self.best_loss[k] = rank_state["best_loss"][k]

        logger.info(f"Resuming training at step {self.global_step} from '{path}'")

        for i, optimizer in enumerate(self.optimizers):
            opt_state = torch.load(
                f"{path}/optimizer_{i}.pt", map_location=device, weights_only=True
            )
            optimizer.load_state_dict(opt_state)

        for name, sae in self.saes.items():
            load_sae_checkpoint(sae, f"{path}/{name}", device=str(device))

        # Load Hadamard rotation states if they exist
        hadamard_path = f"{path}/hadamard_rotations.pt"
        if os.path.exists(hadamard_path):
            hadamard_states = torch.load(
                hadamard_path, map_location=device, weights_only=False
            )
            for name, state in hadamard_states.items():
                self.hadamard_rotations[name] = HadamardRotation.from_state_dict(
                    state, device=device
                )
            logger.info(
                f"Loaded Hadamard rotations for {len(self.hadamard_rotations)} hookpoints"
            )

    def _checkpoint(
        self, saes, path: str, rank_zero: bool, save_training_state: bool = True
    ):
        """Save SAEs and training state to disk."""
        logger.info("Saving checkpoint")

        for optimizer in self.optimizers:
            if isinstance(optimizer, ScheduleFreeWrapper):
                optimizer.eval()

        for name, sae in saes.items():
            sae.save_to_disk(f"{path}/{name}")

        if save_training_state and rank_zero:
            for i, optimizer in enumerate(self.optimizers):
                torch.save(optimizer.state_dict(), f"{path}/optimizer_{i}.pt")

            torch.save(
                {
                    "global_step": self.global_step,
                    "total_tokens": self.total_tokens,
                },
                f"{path}/state.pt",
            )

            self.cfg.save_json(f"{path}/config.json")

            # Save Hadamard rotation states if enabled
            if self.cfg.use_hadamard and self.hadamard_rotations:
                hadamard_states = {
                    name: rot.state_dict()
                    for name, rot in self.hadamard_rotations.items()
                }
                torch.save(hadamard_states, f"{path}/hadamard_rotations.pt")

            torch.save(
                {
                    "num_tokens_since_fired": self.num_tokens_since_fired,
                    "best_loss": self.best_loss,
                },
                f"{path}/rank_0_state.pt",
            )

        for optimizer in self.optimizers:
            if isinstance(optimizer, ScheduleFreeWrapper):
                optimizer.train()

    def save(self):
        """Save the SAEs and training state to disk."""
        path = f'{self.cfg.save_dir}/{self.full_run_name}'
        rank_zero = not dist.is_initialized() or dist.get_rank() == 0

        if rank_zero:
            self._checkpoint(self.saes, path, rank_zero)

        if dist.is_initialized():
            dist.barrier()

    def save_best(self, avg_loss: dict[str, float]):
        """Save individual sparse coders to disk if they have the lowest loss."""
        base_path = f'{self.cfg.save_dir}/{self.full_run_name}/best'
        rank_zero = not dist.is_initialized() or dist.get_rank() == 0

        any_improved = False
        for name in self.saes:
            if avg_loss.get(name, float("inf")) < self.best_loss[name]:
                self.best_loss[name] = avg_loss[name]
                any_improved = True

                if rank_zero:
                    self._checkpoint(
                        {name: self.saes[name]},
                        base_path,
                        rank_zero,
                        save_training_state=False,
                    )

        if any_improved and rank_zero:
            os.makedirs(base_path, exist_ok=True)

            for i, optimizer in enumerate(self.optimizers):
                torch.save(optimizer.state_dict(), f"{base_path}/optimizer_{i}.pt")

            torch.save(
                {
                    "global_step": self.global_step,
                    "total_tokens": self.total_tokens,
                },
                f"{base_path}/state.pt",
            )

            self.cfg.save_json(f"{base_path}/config.json")

            torch.save(
                {
                    "num_tokens_since_fired": self.num_tokens_since_fired,
                    "best_loss": self.best_loss,
                },
                f"{base_path}/rank_0_state.pt",
            )

        if dist.is_initialized():
            dist.barrier()
