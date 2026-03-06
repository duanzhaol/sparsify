import logging
import os
import time
from collections import defaultdict
from dataclasses import asdict
from fnmatch import fnmatchcase
from typing import Sized

import torch
import torch.distributed as dist
from datasets import Dataset as HfDataset
from natsort import natsorted
from schedulefree import ScheduleFreeWrapperReference
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from .checkpoint import CheckpointMixin, expand_range_pattern
from .config import TrainConfig
from .data import MemmapDataset
from .device import create_event, synchronize
from .hadamard import HadamardRotation
from .sign_sgd import SignSGD
from .sparse_coder import SparseCoder
from .tiled_sparse_coder import TiledSparseCoder
from .utils import (
    get_layer_list,
    get_max_layer_index,
    partial_forward_to_layer,
    resolve_widths,
)

logger = logging.getLogger(__name__)


class Trainer(CheckpointMixin):
    def __init__(
        self,
        cfg: TrainConfig,
        dataset: HfDataset | MemmapDataset,
        model: PreTrainedModel,
        resume_from: str | None = None,
    ):
        self.model = model

        if cfg.hookpoints:
            assert not cfg.layers, "Cannot specify both `hookpoints` and `layers`."

            # Expand range patterns like layers.[1-10].xxx
            expanded_patterns = []
            for pattern in cfg.hookpoints:
                expanded_patterns.extend(expand_range_pattern(pattern))

            # Replace wildcard patterns with actual module names
            raw_hookpoints = []
            for name, _ in model.base_model.named_modules():
                if any(fnmatchcase(name, pat) for pat in expanded_patterns):
                    raw_hookpoints.append(name)

            cfg.hookpoints = natsorted(raw_hookpoints)
        else:
            # If no layers are specified, train on all of them
            if not cfg.layers:
                N = model.config.num_hidden_layers
                cfg.layers = list(range(0, N))

            layers_name, _ = get_layer_list(model)
            cfg.hookpoints = [f"{layers_name}.{i}" for i in cfg.layers]

        cfg.hookpoints = cfg.hookpoints[:: cfg.layer_stride]

        self.cfg = cfg
        self.dataset = dataset
        self.resume_from = resume_from

        logger.info(f"Training on modules: {cfg.hookpoints}")

        # Detect maximum layer index for partial forward optimization
        layers_name, _ = get_layer_list(model)
        self._max_layer_for_fvu = get_max_layer_index(cfg.hookpoints, layers_name)

        device = model.device
        input_widths = resolve_widths(model, cfg.hookpoints, hook_mode="input")

        # Initialize SAEs
        logger.info(f"Initializing SAEs with random seed(s) {cfg.init_seeds}")
        if cfg.num_tiles > 1:
            logger.info(f"Tiled mode: splitting input into {cfg.num_tiles} tiles")

        self.saes = {}
        for hook in cfg.hookpoints:
            for seed in cfg.init_seeds:
                torch.manual_seed(seed)
                name = f"{hook}/seed{seed}" if len(cfg.init_seeds) > 1 else hook

                if cfg.num_tiles > 1:
                    self.saes[name] = TiledSparseCoder(
                        input_widths[hook],
                        cfg.sae,
                        cfg.num_tiles,
                        device,
                        dtype=torch.float32,
                        global_topk=cfg.global_topk,
                        input_mixing=cfg.input_mixing,
                    )
                else:
                    self.saes[name] = SparseCoder(
                        input_widths[hook],
                        cfg.sae,
                        device,
                        dtype=torch.float32,
                    )

        assert isinstance(dataset, Sized)

        # Optimizer: Signum with schedule-free wrapper
        pgs = [
            dict(
                params=sae.parameters(),
                lr=cfg.lr or 5e-3 / (sae.num_latents / (2**14)) ** 0.5,
            )
            for sae in self.saes.values()
        ]
        lrs = [f"{lr:.2e}" for lr in sorted(set(pg["lr"] for pg in pgs))]

        opt = ScheduleFreeWrapperReference(SignSGD(pgs), momentum=0.95)
        opt.train()
        self.optimizers = [opt]

        logger.info(
            f"Learning rate{'s' if len(lrs) > 1 else ''}: {', '.join(lrs)}"
        )

        self.global_step = 0
        self.total_tokens = 0

        self.num_tokens_since_fired = {
            name: torch.zeros(sae.num_latents, device=device, dtype=torch.long)
            for name, sae in self.saes.items()
        }

        # Load elbow thresholds if provided
        self.elbow_thresholds: dict[str, float] = {}
        if cfg.elbow_threshold_path:
            self._load_elbow_thresholds(cfg.elbow_threshold_path)

        self.best_loss: dict[str, float] = {
            name: float("inf") for name in self.saes.keys()
        }

        # Hadamard rotations (lazily initialized in hook when we know d_in)
        self.hadamard_rotations: dict[str, HadamardRotation] = {}
        if cfg.use_hadamard:
            logger.info(
                f"Hadamard rotation enabled: block_size={cfg.hadamard_block_size}, "
                f"seed={cfg.hadamard_seed}, use_perm={cfg.hadamard_use_perm}"
            )

    def fit(self):
        # Use Tensor Cores even for fp32 matmuls
        torch.set_float32_matmul_precision("high")

        self.model.requires_grad_(False)

        rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        ddp = dist.is_initialized()

        # Generate full run name
        if self.resume_from:
            self.full_run_name = self.resume_from.split("/")[-1]
        else:
            from datetime import datetime

            world_size = dist.get_world_size() if dist.is_initialized() else 1
            base_name = self.cfg.run_name or "unnamed"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.full_run_name = (
                f"{base_name}_dp{world_size}_bs{self.cfg.batch_size}"
                f"_ga{self.cfg.grad_acc_steps}_ef{self.cfg.sae.expansion_factor}"
                f"_k{self.cfg.sae.k}_{timestamp}"
            )

        wandb = None
        if self.cfg.log_to_wandb:
            try:
                import wandb as _wandb
            except ImportError:
                _wandb = None

            if _wandb is None:
                self.cfg.log_to_wandb = False
                if rank_zero:
                    logger.warning(
                        "Weights & Biases not installed, skipping logging."
                    )
            elif rank_zero:
                try:
                    project_name = (
                        self.cfg.wandb_project
                        or os.environ.get("WANDB_PROJECT", "sparsify")
                    )
                    _wandb.init(
                        entity=os.environ.get("WANDB_ENTITY", None),
                        name=self.full_run_name,
                        project=project_name,
                        config=asdict(self.cfg),
                        save_code=True,
                    )
                    wandb = _wandb
                except Exception:
                    logger.warning(
                        "wandb.init() failed, skipping logging."
                    )
                    self.cfg.log_to_wandb = False

            # Sync log_to_wandb across ranks so all_reduce calls stay consistent
            if ddp:
                flag = torch.tensor(
                    [self.cfg.log_to_wandb], dtype=torch.int32,
                    device=self.model.device,
                )
                dist.broadcast(flag, src=0)
                self.cfg.log_to_wandb = bool(flag.item())

        num_sae_params = sum(
            p.numel() for s in self.saes.values() for p in s.parameters()
        )
        num_model_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Number of SAE parameters: {num_sae_params:_}")
        logger.info(f"Number of model parameters: {num_model_params:_}")

        num_batches = len(self.dataset) // self.cfg.batch_size
        if self.global_step > 0:
            assert hasattr(self.dataset, "select"), "Dataset must implement `select`"
            n = self.global_step * self.cfg.batch_size
            ds = self.dataset.select(range(n, len(self.dataset)))  # type: ignore
        else:
            ds = self.dataset

        device = self.model.device
        dl = DataLoader(
            ds,  # type: ignore
            batch_size=self.cfg.batch_size,
            shuffle=False,
        )
        pbar = tqdm(
            desc="Training",
            disable=not rank_zero,
            initial=self.global_step,
            total=num_batches,
        )

        did_fire = {
            name: torch.zeros(sae.num_latents, device=device, dtype=torch.bool)
            for name, sae in self.saes.items()
        }

        acc_steps = self.cfg.grad_acc_steps * self.cfg.micro_acc_steps
        denom = acc_steps * self.cfg.wandb_log_frequency
        num_tokens_in_step = 0

        # Timing metrics
        total_forward_time = 0.0
        total_metrics_time = 0.0
        timing_step_count = 0

        # Logging accumulators
        avg_auxk_loss: dict[str, float] = defaultdict(float)
        avg_fvu: dict[str, float] = defaultdict(float)
        avg_exceed: dict[str, dict[float, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        avg_losses: dict[str, float] = {
            name: float("inf") for name in self.saes.keys()
        }

        # Device events for timing
        if device.type in ("cuda", "npu"):
            forward_start = create_event(enable_timing=True)
            forward_end = create_event(enable_timing=True)
        else:
            forward_start = forward_end = None

        name_to_module = {
            name: self.model.base_model.get_submodule(name)
            for name in self.cfg.hookpoints
        }
        maybe_wrapped: dict[str, DDP | SparseCoder | TiledSparseCoder] = {}
        module_to_name = {v: k for k, v in name_to_module.items()}

        # Build mapping from bare hook name to SAE keys (supports multi-seed)
        hook_to_sae_keys: dict[str, list[str]] = defaultdict(list)
        for sae_key in self.saes:
            hook = sae_key.partition("/")[0]
            hook_to_sae_keys[hook].append(sae_key)

        def hook(module: nn.Module, inputs, outputs):
            nonlocal total_forward_time, total_metrics_time, timing_step_count

            if isinstance(inputs, tuple):
                inputs = inputs[0]

            name = module_to_name[module]

            # Flatten batch and sequence dimensions, use input activations
            acts = inputs.flatten(0, 1)

            # Apply Hadamard rotation if enabled
            if self.cfg.use_hadamard:
                if name not in self.hadamard_rotations:
                    d_in = acts.shape[-1]
                    self.hadamard_rotations[name] = HadamardRotation(
                        d_in,
                        block_size=self.cfg.hadamard_block_size,
                        seed=self.cfg.hadamard_seed,
                        use_permutation=self.cfg.hadamard_use_perm,
                        device=acts.device,
                        dtype=acts.dtype,
                    )
                acts = self.hadamard_rotations[name].rotate(acts)

            # Pre-compute unrotated target for exceed metrics
            original_target_for_exceed = None
            if (
                self.cfg.use_hadamard
                and name in self.hadamard_rotations
                and name in self.elbow_thresholds
            ):
                original_target_for_exceed = self.hadamard_rotations[
                    name
                ].unrotate(acts)

            # Process each SAE for this hook (supports multi-seed)
            for sae_key in hook_to_sae_keys[name]:
                raw = self.saes[sae_key]

                # On the first iteration, initialize decoder bias from data mean
                if self.global_step == 0 and not self.cfg.finetune:
                    mean = self.maybe_all_reduce(acts.mean(0))

                    if hasattr(raw, 'set_b_dec_data'):
                        raw.set_b_dec_data(mean.to(raw.dtype))
                    else:
                        raw.b_dec.data = mean.to(raw.dtype)

                # Normalize decoder if applicable
                if raw.cfg.normalize_decoder and raw.W_dec.requires_grad:
                    raw.set_decoder_norm_to_unit_norm()

                wrapped = maybe_wrapped[sae_key]
                out = wrapped(
                    x=acts,
                    dead_mask=(
                        self.num_tokens_since_fired[sae_key]
                        > self.cfg.dead_feature_threshold
                        if self.cfg.auxk_alpha > 0
                        else None
                    ),
                )

                # Update the did_fire mask
                did_fire[sae_key][out.latent_indices.flatten()] = True
                self.maybe_all_reduce(did_fire[sae_key], "max")

                # Accumulate metrics
                avg_fvu[sae_key] += float(
                    self.maybe_all_reduce(out.fvu.detach()) / denom
                )
                if self.cfg.auxk_alpha > 0:
                    avg_auxk_loss[sae_key] += float(
                        self.maybe_all_reduce(out.auxk_loss.detach()) / denom
                    )

                # Compute exceed metrics if elbow thresholds available
                if name in self.elbow_thresholds and self.cfg.exceed_alphas:
                    if device.type in ("cuda", "npu"):
                        metrics_start_evt = create_event(enable_timing=True)
                        metrics_end_evt = create_event(enable_timing=True)
                        metrics_start_evt.record()
                    else:
                        metrics_time_start = time.perf_counter()

                    if (
                        self.cfg.use_hadamard
                        and name in self.hadamard_rotations
                    ):
                        original_target = (
                            original_target_for_exceed
                            if original_target_for_exceed is not None
                            else acts
                        )
                        original_recon = self.hadamard_rotations[
                            name
                        ].unrotate(out.sae_out)
                    else:
                        original_target = acts
                        original_recon = out.sae_out

                    error_magnitude = torch.abs(original_target - original_recon)
                    elbow_value = self.elbow_thresholds[name]
                    num_elements = error_magnitude.numel()

                    for alpha in self.cfg.exceed_alphas:
                        threshold = alpha * elbow_value
                        exceed_count = (
                            (error_magnitude > threshold).sum().float()
                        )
                        exceed_ratio = exceed_count / num_elements
                        avg_exceed[sae_key][alpha] += float(
                            self.maybe_all_reduce(exceed_ratio.detach()) / denom
                        )

                    if device.type in ("cuda", "npu"):
                        metrics_end_evt.record()
                        synchronize()
                        total_metrics_time += (
                            metrics_start_evt.elapsed_time(metrics_end_evt)
                            / 1000.0
                        )
                    else:
                        total_metrics_time += (
                            time.perf_counter() - metrics_time_start
                        )

                # Local backward pass
                loss = out.fvu + self.cfg.auxk_alpha * out.auxk_loss
                loss.div(acc_steps).backward()

        for batch in dl:
            x = batch["input_ids"].to(device)

            if not maybe_wrapped:
                # Wrap the SAEs with DDP. We have to do this after we set the
                # decoder bias, otherwise DDP will not register gradients
                # flowing to the bias after the first step.
                # Use LOCAL_RANK for device_ids (not global rank)
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                maybe_wrapped = (
                    {
                        name: DDP(sae, device_ids=[local_rank])
                        for name, sae in self.saes.items()
                    }
                    if ddp
                    else self.saes
                )

            # Bookkeeping for dead feature detection
            num_tokens_in_step += x.numel()

            # Start timing forward pass
            if device.type in ("cuda", "npu"):
                synchronize()
                forward_start.record()
            else:
                forward_time_start = time.perf_counter()

            # Forward pass on the model to capture activations via hooks
            handles = [
                mod.register_forward_hook(hook)
                for mod in name_to_module.values()
            ]
            try:
                if self._max_layer_for_fvu is not None:
                    partial_forward_to_layer(
                        self.model, x, self._max_layer_for_fvu
                    )
                else:
                    self.model(x)

                # End timing
                if device.type in ("cuda", "npu"):
                    forward_end.record()
                    synchronize()
                    total_forward_time += (
                        forward_start.elapsed_time(forward_end) / 1000.0
                    )
                else:
                    total_forward_time += (
                        time.perf_counter() - forward_time_start
                    )
                timing_step_count += 1

                avg_losses = dict(avg_fvu)

            finally:
                for handle in handles:
                    handle.remove()

            # Check if we need to actually do a training step
            step, substep = divmod(self.global_step + 1, self.cfg.grad_acc_steps)
            if substep == 0:
                if self.cfg.sae.normalize_decoder:
                    for sae in self.saes.values():
                        sae.remove_gradient_parallel_to_decoder_directions()

                for optimizer in self.optimizers:
                    optimizer.step()
                    optimizer.zero_grad()

                with torch.no_grad():
                    # Update the dead feature mask
                    for name, counts in self.num_tokens_since_fired.items():
                        counts += num_tokens_in_step
                        counts[did_fire[name]] = 0

                    # Accumulate total tokens
                    if dist.is_initialized():
                        num_tokens_tensor = torch.tensor(
                            num_tokens_in_step, device=device, dtype=torch.long
                        )
                        dist.all_reduce(
                            num_tokens_tensor, op=dist.ReduceOp.SUM
                        )
                        self.total_tokens += num_tokens_tensor.item()
                    else:
                        self.total_tokens += num_tokens_in_step

                    # Check if we've reached the target token count
                    if (
                        self.cfg.max_tokens
                        and self.total_tokens >= self.cfg.max_tokens
                    ):
                        if rank_zero:
                            logger.info(
                                f"Reached target token count: "
                                f"{self.total_tokens:,} / {self.cfg.max_tokens:,}"
                            )
                        self.save()
                        if dist.is_initialized():
                            dist.destroy_process_group()
                        return

                    # Reset stats for this step
                    num_tokens_in_step = 0
                    for mask in did_fire.values():
                        mask.zero_()

                if (step + 1) % self.cfg.save_every == 0:
                    self.save()
                    if self.cfg.save_best:
                        self.save_best(avg_losses)

                if (step + 1) % self.cfg.wandb_log_frequency == 0:
                    if self.cfg.log_to_wandb:
                        info = {}

                        # Timing metrics
                        if timing_step_count > 0:
                            avg_forward_time = (
                                total_forward_time / timing_step_count
                            )
                            forward_time_tensor = torch.tensor(
                                avg_forward_time, device=device
                            )
                            info["timing/forward_time"] = float(
                                self.maybe_all_reduce(forward_time_tensor)
                            )

                            if total_metrics_time > 0:
                                avg_metrics_time = (
                                    total_metrics_time / timing_step_count
                                )
                                metrics_time_tensor = torch.tensor(
                                    avg_metrics_time, device=device
                                )
                                info["timing/metrics_time"] = float(
                                    self.maybe_all_reduce(metrics_time_tensor)
                                )

                        for name in self.saes:
                            mask = (
                                self.num_tokens_since_fired[name]
                                > self.cfg.dead_feature_threshold
                            )
                            info[f"{name}/dead_pct"] = mask.float().mean().item()
                            info[f"{name}/fvu"] = avg_fvu[name]

                            if self.cfg.auxk_alpha > 0:
                                info[f"{name}/auxk"] = avg_auxk_loss[name]

                            # Exceed metrics
                            if name in avg_exceed:
                                for alpha, val in avg_exceed[name].items():
                                    info[f"{name}/exceed_alpha_{alpha:.2f}"] = val

                        if rank_zero:
                            info["_step"] = step
                            if wandb is not None:
                                wandb.log(info, step=self.total_tokens)

                    # Reset accumulators unconditionally
                    avg_auxk_loss.clear()
                    avg_fvu.clear()
                    avg_exceed.clear()
                    total_forward_time = 0.0
                    total_metrics_time = 0.0
                    timing_step_count = 0

            self.global_step += 1
            pbar.update()

        self.save()
        if self.cfg.save_best:
            self.save_best(avg_losses)

        pbar.close()

    def maybe_all_cat(self, x: Tensor) -> Tensor:
        """Concatenate a tensor across all processes."""
        if not dist.is_initialized():
            return x

        buffer = x.new_empty(
            [dist.get_world_size() * x.shape[0], *x.shape[1:]]
        )
        dist.all_gather_into_tensor(buffer, x)
        return buffer

    def maybe_all_reduce(self, x: Tensor, op: str = "mean") -> Tensor:
        if not dist.is_initialized():
            return x

        if op == "sum":
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        elif op == "mean":
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            x /= dist.get_world_size()
        elif op == "max":
            dist.all_reduce(x, op=dist.ReduceOp.MAX)
        else:
            raise ValueError(f"Unknown reduction op '{op}'")

        return x


# Support old name for compatibility
SaeTrainer = Trainer
