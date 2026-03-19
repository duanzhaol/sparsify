import logging
import os
import time
from collections import defaultdict
from contextlib import nullcontext
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
from .metrics_logger import MetricsLogger
from .sign_sgd import SignSGD
from .sparse_coder import SparseCoder, _get_sae_class
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
                    sae_cls = _get_sae_class(cfg.sae.architecture)
                    self.saes[name] = sae_cls(
                        input_widths[hook],
                        cfg.sae,
                        device,
                        dtype=torch.float32,
                    )

        assert isinstance(dataset, Sized)

        # Optimizer with configurable backend and get_param_groups interface
        pgs = []
        for sae in self.saes.values():
            base_lr = cfg.lr or 5e-3 / (sae.num_latents / (2**14)) ** 0.5
            pgs.extend(sae.get_param_groups(base_lr))
        lrs = [f"{lr:.2e}" for lr in sorted(set(pg["lr"] for pg in pgs))]

        if cfg.optimizer == "adam":
            base_opt = torch.optim.Adam(pgs)
        else:
            # signum (default)
            base_opt = SignSGD(pgs)
        opt = ScheduleFreeWrapperReference(base_opt, momentum=0.95)
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

        # Residual SAE: preload frozen Level 1 checkpoints
        self.residual_saes: dict[str, SparseCoder] = {}
        if cfg.residual_from:
            from pathlib import Path as _Path

            logger.info(f"Residual training from: {cfg.residual_from}")
            for hook in cfg.hookpoints:
                l1_path = _Path(cfg.residual_from) / hook
                self.residual_saes[hook] = (
                    SparseCoder.load_any(l1_path, device=device).eval()
                )
                self.residual_saes[hook].requires_grad_(False)

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

        # Initialize MetricsLogger for structured local result saving
        metrics_logger = None
        if self.cfg.save_metrics_jsonl and rank_zero:
            from pathlib import Path as _Path

            log_dir = _Path(self.cfg.save_dir) / self.full_run_name
            run_meta = {
                "run_name": self.full_run_name,
                "architecture": self.cfg.sae.architecture,
                "hookpoints": self.cfg.hookpoints,
                "init_seeds": self.cfg.init_seeds,
                "residual_from": self.cfg.residual_from,
            }
            metrics_logger = MetricsLogger(
                log_dir, asdict(self.cfg), run_meta
            )

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

        # Buffer to collect fired latent indices during each step.
        # Replaces the old did_fire bool mask + per-forward scatter_ with a
        # single cat+unique at step end, avoiding expensive AI_CPU fallback.
        fired_indices: dict[str, list[torch.Tensor]] = {
            name: [] for name in self.saes
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

        # Deferred metrics events (optimization: avoid per-hookpoint sync)
        pending_metrics_events: list[tuple] = []

        # Batched reduce helpers for metrics (optimization: one allreduce
        # per log step instead of per-hookpoint per-microbatch)
        def reduce_scalar_mapping(
            values: dict[str, float],
        ) -> dict[str, float]:
            if not values:
                return {}
            if not dist.is_initialized():
                return dict(values)
            keys = sorted(values)
            tensor = torch.tensor(
                [values[key] for key in keys], device=device
            )
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= dist.get_world_size()
            return dict(zip(keys, tensor.tolist()))

        def reduce_nested_scalar_mapping(
            values: dict[str, dict[float, float]],
        ) -> dict[str, dict[float, float]]:
            if not values:
                return {}
            if not dist.is_initialized():
                return {
                    outer_key: dict(inner)
                    for outer_key, inner in values.items()
                }
            flat_keys = [
                (outer_key, alpha)
                for outer_key in sorted(values)
                for alpha in sorted(values[outer_key])
            ]
            tensor = torch.tensor(
                [values[ok][ak] for ok, ak in flat_keys], device=device
            )
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= dist.get_world_size()
            result: dict[str, dict[float, float]] = defaultdict(dict)
            for (ok, ak), v in zip(flat_keys, tensor.tolist()):
                result[ok][ak] = v
            return dict(result)

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

        def _hook_impl(module: nn.Module, inputs, outputs):
            nonlocal total_forward_time, total_metrics_time, timing_step_count
            nonlocal should_time, sync_gradients

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

            # Residual SAE: subtract frozen Level 1 reconstruction.
            # Must be AFTER Hadamard so L1 operates in the same space it was trained in.
            if self.residual_saes and name in self.residual_saes:
                with torch.no_grad():
                    l1_out = self.residual_saes[name](acts).sae_out
                acts = acts - l1_out

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
                sync_context = (
                    nullcontext()
                    if sync_gradients or not isinstance(wrapped, DDP)
                    else wrapped.no_sync()
                )
                with sync_context:
                    out = wrapped(
                        x=acts,
                        dead_mask=(
                            self.num_tokens_since_fired[sae_key]
                            > self.cfg.dead_feature_threshold
                            if self.cfg.auxk_alpha > 0
                            else None
                        ),
                    )

                    # Collect fired latent indices (deferred update at optimizer step)
                    fired_indices[sae_key].append(out.latent_indices.flatten())

                    # Accumulate metrics locally (allreduce deferred to log step)
                    avg_fvu[sae_key] += float(out.fvu.detach() / denom)
                    if self.cfg.auxk_alpha > 0:
                        avg_auxk_loss[sae_key] += float(
                            out.auxk_loss.detach() / denom
                        )

                    # Compute exceed metrics if elbow thresholds available
                    if name in self.elbow_thresholds and self.cfg.exceed_alphas:
                        if should_time and device.type in ("cuda", "npu"):
                            metrics_start_evt = create_event(enable_timing=True)
                            metrics_start_evt.record()
                        elif device.type not in ("cuda", "npu"):
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
                                exceed_ratio.detach() / denom
                            )

                        if should_time and device.type in ("cuda", "npu"):
                            m_end = create_event(enable_timing=True)
                            m_end.record()
                            pending_metrics_events.append(
                                (metrics_start_evt, m_end)
                            )
                        elif device.type not in ("cuda", "npu"):
                            total_metrics_time += (
                                time.perf_counter() - metrics_time_start
                            )

                    # Local backward pass (inside sync_context for DDP no_sync)
                    loss = out.fvu + self.cfg.auxk_alpha * out.auxk_loss

                    # Matryoshka multi-K loss
                    if self.cfg.matryoshka_ks:
                        total_var = (acts - acts.mean(0)).pow(2).sum()
                        weights = (
                            self.cfg.matryoshka_weights
                            or [1.0] * len(self.cfg.matryoshka_ks)
                        )
                        # Sort by activation magnitude so [:mk] gets strongest
                        sorted_acts, sort_idx = out.latent_acts.sort(
                            dim=-1, descending=True
                        )
                        sorted_indices = out.latent_indices.gather(-1, sort_idx)
                        for mk, mw in zip(self.cfg.matryoshka_ks, weights):
                            sub_recon = raw.decode(
                                sorted_acts[:, :mk], sorted_indices[:, :mk]
                            )
                            sub_e = (acts - sub_recon).pow(2).sum()
                            loss = loss + mw * sub_e / total_var

                    # Orthogonality regularization on active decoder columns
                    if self.cfg.ortho_lambda > 0 and raw.W_dec is not None:
                        D_S = raw.W_dec[out.latent_indices]  # [batch, K, d_in]
                        gram = torch.bmm(
                            D_S, D_S.transpose(1, 2)
                        )  # [batch, K, K]
                        eye = torch.eye(
                            gram.shape[-1],
                            device=gram.device,
                            dtype=gram.dtype,
                        )
                        ortho_loss = (gram - eye).pow(2).sum() / gram.shape[0]
                        loss = loss + self.cfg.ortho_lambda * ortho_loss

                    loss.div(acc_steps).backward()

        # Prevent dynamo from tracing the hook body when torch.compile is used
        # on transformer layers. Without this, DDP's autograd hooks don't fire
        # properly inside dynamo resume functions.
        if self.cfg.compile_model:
            import torch._dynamo as _dynamo
            hook = _dynamo.disable(_hook_impl)
        else:
            hook = _hook_impl

        # Compile individual transformer layers to fuse elementwise kernels
        if self.cfg.compile_model:
            _dynamo.config.cache_size_limit = 128
            _, layer_list = get_layer_list(self.model)
            for i in range(len(layer_list)):
                layer_list[i] = torch.compile(layer_list[i])
            print(f"Compiled {len(layer_list)} transformer layers with torch.compile")

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

            # Determine if this step needs timing and gradient sync
            step_candidate, substep_candidate = divmod(
                self.global_step + 1, self.cfg.grad_acc_steps
            )
            sync_gradients = substep_candidate == 0
            should_time = (
                self.cfg.log_to_wandb
                and sync_gradients
                and (step_candidate + 1) % self.cfg.wandb_log_frequency == 0
            )

            # Start timing forward pass (only when logging)
            if should_time and device.type in ("cuda", "npu"):
                synchronize()
                forward_start.record()
            elif should_time:
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

                # End timing (only when logging, single sync resolves all)
                if should_time and device.type in ("cuda", "npu"):
                    forward_end.record()
                    synchronize()
                    total_forward_time += (
                        forward_start.elapsed_time(forward_end) / 1000.0
                    )
                    for m_start, m_end in pending_metrics_events:
                        total_metrics_time += (
                            m_start.elapsed_time(m_end) / 1000.0
                        )
                    pending_metrics_events.clear()
                elif should_time:
                    total_forward_time += (
                        time.perf_counter() - forward_time_start
                    )
                if should_time:
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
                    # Update dead feature counts: increment all, then zero
                    # locally-fired latents.  A subsequent allreduce with MIN
                    # propagates zeros from any GPU so the result is equivalent
                    # to the old did_fire bool mask approach, but avoids the
                    # expensive per-forward scatter_ (AI_CPU fallback on NPU).
                    for name, counts in self.num_tokens_since_fired.items():
                        counts += num_tokens_in_step
                        if fired_indices[name]:
                            idx = torch.cat(fired_indices[name])
                            # No unique() needed: writing 0 to duplicate
                            # indices is idempotent, and unique() falls back
                            # to AI_CPU on NPU (~181ms per step).
                            counts[idx] = 0

                    # Sync counts across GPUs: MIN ensures any GPU's zero
                    # (latent fired) propagates to all replicas.
                    if dist.is_initialized():
                        count_keys = list(self.num_tokens_since_fired.keys())
                        all_counts = torch.cat(
                            [self.num_tokens_since_fired[k] for k in count_keys]
                        )
                        dist.all_reduce(all_counts, op=dist.ReduceOp.MIN)
                        offset = 0
                        for k in count_keys:
                            n = self.num_tokens_since_fired[k].shape[0]
                            self.num_tokens_since_fired[k].copy_(
                                all_counts[offset : offset + n]
                            )
                            offset += n

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
                        if metrics_logger is not None:
                            metrics_logger.save_summary(
                                {
                                    "total_steps": step,
                                    "total_tokens": self.total_tokens,
                                    "final_fvu": {
                                        name: avg_fvu.get(name, 0.0)
                                        for name in self.saes
                                    },
                                    "best_fvu": dict(self.best_loss),
                                }
                            )
                            metrics_logger.close()
                        if dist.is_initialized():
                            dist.destroy_process_group()
                        return

                    # Reset fired indices buffer for next step
                    num_tokens_in_step = 0
                    for buf in fired_indices.values():
                        buf.clear()

                if (step + 1) % self.cfg.save_every == 0:
                    self.save()
                    if self.cfg.save_best:
                        self.save_best(avg_losses)

                if (step + 1) % self.cfg.wandb_log_frequency == 0:
                    should_log = (
                        self.cfg.log_to_wandb or metrics_logger is not None
                    )
                    if should_log:
                        # Batch reduce all metrics in one allreduce each
                        reduced_fvu = reduce_scalar_mapping(dict(avg_fvu))
                        reduced_auxk = reduce_scalar_mapping(
                            dict(avg_auxk_loss)
                        )
                        reduced_exceed = reduce_nested_scalar_mapping(
                            dict(avg_exceed)
                        )

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
                            info[f"{name}/fvu"] = reduced_fvu.get(name, 0.0)

                            if self.cfg.auxk_alpha > 0:
                                info[f"{name}/auxk"] = reduced_auxk.get(
                                    name, 0.0
                                )

                            # Exceed metrics
                            if name in reduced_exceed:
                                for alpha, val in reduced_exceed[name].items():
                                    info[f"{name}/exceed_alpha_{alpha:.2f}"] = val

                        if rank_zero:
                            info["_step"] = step
                            if wandb is not None:
                                wandb.log(info, step=self.total_tokens)
                            if metrics_logger is not None:
                                metrics_logger.log_step(
                                    step, self.total_tokens, info
                                )

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

        # Write final summary and close metrics logger
        if metrics_logger is not None:
            metrics_logger.save_summary(
                {
                    "total_steps": step if 'step' in dir() else self.global_step,
                    "total_tokens": self.total_tokens,
                    "final_fvu": {
                        name: avg_fvu.get(name, 0.0) for name in self.saes
                    },
                    "best_fvu": dict(self.best_loss),
                }
            )
            metrics_logger.close()

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
