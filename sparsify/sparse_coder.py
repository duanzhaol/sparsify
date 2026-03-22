import json
from fnmatch import fnmatch
from pathlib import Path
from typing import NamedTuple

import einops
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from natsort import natsorted
from safetensors import safe_open
from safetensors.torch import load_model, save_model
from torch import Tensor, nn

from .config import SparseCoderConfig
from .device import device_autocast
from .fused_encoder import EncoderOutput, fused_encoder
from .utils import decoder_impl


class ForwardOutput(NamedTuple):
    sae_out: Tensor

    latent_acts: Tensor
    """Activations of the top-k latents."""

    latent_indices: Tensor
    """Indices of the top-k features."""

    fvu: Tensor
    """Fraction of variance unexplained."""

    auxk_loss: Tensor
    """AuxK loss, if applicable."""


class SparseCoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = d_in
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor

        self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
        self.encoder.bias.data.zero_()

        if decoder:
            self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    @staticmethod
    def load_many(
        name: str,
        local: bool = False,
        layers: list[str] | None = None,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
        pattern: str | None = None,
    ) -> dict[str, "SparseCoder"]:
        """Load sparse coders for multiple hookpoints on a single model and dataset."""
        pattern = pattern + "/*" if pattern is not None else None
        if local:
            repo_path = Path(name)
        else:
            repo_path = Path(snapshot_download(name, allow_patterns=pattern))

        if layers is not None:
            return {
                layer: SparseCoder.load_any(
                    repo_path / layer, device=device, decoder=decoder
                )
                for layer in natsorted(layers)
            }
        files = [
            f
            for f in repo_path.iterdir()
            if f.is_dir() and (pattern is None or fnmatch(f.name, pattern))
        ]
        return {
            f.name: SparseCoder.load_any(f, device=device, decoder=decoder)
            for f in natsorted(files, key=lambda f: f.name)
        }

    @staticmethod
    def load_from_hub(
        name: str,
        hookpoint: str | None = None,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
    ) -> "SparseCoder":
        # Download from the HuggingFace Hub
        repo_path = Path(
            snapshot_download(
                name,
                allow_patterns=f"{hookpoint}/*" if hookpoint is not None else None,
            )
        )
        if hookpoint is not None:
            repo_path = repo_path / hookpoint

        # No layer specified, and there are multiple layers
        elif not repo_path.joinpath("cfg.json").exists():
            raise FileNotFoundError("No config file found; try specifying a layer.")

        return SparseCoder.load_any(repo_path, device=device, decoder=decoder)

    @classmethod
    def load_from_disk(
        cls,
        path: Path | str,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
    ) -> "SparseCoder":
        path = Path(path)

        with open(path / "cfg.json", "r") as f:
            cfg_dict = json.load(f)
            d_in = cfg_dict.pop("d_in")
            cfg = SparseCoderConfig.from_dict(cfg_dict, drop_extra_fields=True)

        safetensors_path = str(path / "sae.safetensors")

        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            first_key = next(iter(f.keys()))
            reference_dtype = f.get_tensor(first_key).dtype

        sae = cls(
            d_in, cfg, device=device, decoder=decoder, dtype=reference_dtype
        )

        load_model(
            model=sae,
            filename=safetensors_path,
            device=str(device),
            # TODO: Maybe be more fine-grained about this in the future?
            strict=decoder,
        )
        return sae

    @staticmethod
    def load_any(
        path: Path | str,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
    ) -> "SparseCoder":
        """Factory: load any SAE variant by reading architecture from cfg.json."""
        path = Path(path)

        with open(path / "cfg.json", "r") as f:
            cfg_dict = json.load(f)

        architecture = cfg_dict.get("architecture", "topk")
        cls = _get_sae_class(architecture)
        return cls.load_from_disk(path, device=device, decoder=decoder)

    def save_to_disk(self, path: Path | str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        save_model(self, str(path / "sae.safetensors"))
        with open(path / "cfg.json", "w") as f:
            json.dump(
                {
                    **self.cfg.to_dict(),
                    "d_in": self.d_in,
                },
                f,
            )

    @property
    def device(self):
        return self.b_dec.device

    @property
    def dtype(self):
        return self.b_dec.dtype

    def get_param_groups(self, base_lr: float) -> list[dict]:
        """Return optimizer parameter groups. Subclasses can override for per-component LR."""
        return [{"params": self.parameters(), "lr": base_lr}]

    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode the input and select the top-k latents."""
        x = x - self.b_dec
        return fused_encoder(x, self.encoder.weight, self.encoder.bias, self.cfg.k)

    def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."

        y = decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec.mT)
        return y + self.b_dec

    # Wrapping the forward in bf16 autocast improves performance by almost 2x
    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        top_acts, top_indices, pre_acts = self.encode(x)

        # If we aren't given a distinct target, we're autoencoding
        if y is None:
            y = x

        # Decode
        sae_out = self.decode(top_acts, top_indices)

        # Compute the residual
        e = y - sae_out

        # Used as a denominator for putting everything on a reasonable scale
        total_variance = (y - y.mean(0)).pow(2).sum()

        # Second decoder pass for AuxK loss
        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            # Heuristic from Appendix B.1 in the paper
            k_aux = y.shape[-1] // 2

            # Reduce the scale of the loss if there are a small number of dead latents
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            # Don't include living latents in this loss
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            # Encourage the top ~50% of dead latents to predict the residual of the
            # top k living latents
            e_hat = self.decode(auxk_acts, auxk_indices)
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
        )

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."

        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."
        if self.W_dec.grad is None:
            return

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )


def _get_sae_class(architecture: str) -> type:
    """Return the SparseCoder subclass for the given architecture string."""
    if architecture == "topk":
        return SparseCoder
    if architecture == "batch_topk":
        return BatchTopKSparseCoder
    if architecture == "adaptive_budget_topk":
        return AdaptiveBudgetTopKSparseCoder
    if architecture == "bucketed_topk":
        return BucketedTopKSparseCoder
    if architecture == "codebook_topk":
        return CodebookTopKSparseCoder
    if architecture == "residual_vq":
        return ResidualVQSparseCoder
    if architecture == "two_code_residual_vq":
        return TwoCodeResidualVQSparseCoder
    if architecture == "whitened_topk":
        return WhitenedTopKSparseCoder
    if architecture == "jumprelu":
        return JumpReLUSparseCoder
    if architecture == "gated":
        return GatedSparseCoder
    if architecture == "routed":
        return RoutedSparseCoder
    if architecture == "group_topk":
        return GroupTopKSparseCoder
    if architecture == "factorized_topk":
        return FactorizedTopKSparseCoder
    if architecture == "lowrank_residual":
        return LowRankResidualSparseCoder
    if architecture == "lowrank_gated_residual":
        return LowRankGatedResidualSparseCoder
    if architecture == "whitened_lowrank_gated_residual":
        return WhitenedLowRankGatedResidualSparseCoder
    if architecture == "lowrank_grouped_residual":
        return LowRankGroupedResidualSparseCoder
    if architecture == "two_stage_residual":
        return TwoStageResidualSparseCoder
    if architecture == "multi_branch_gated":
        return MultiBranchGatedSparseCoder
    raise ValueError(f"Unknown architecture: {architecture!r}")


# Allow for alternate naming conventions
Sae = SparseCoder


class WhitenedTopKSparseCoder(SparseCoder):
    """Top-k SAE with normalized low-rank input preconditioning."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device=device, dtype=dtype, decoder=decoder)
        hidden_dim = min(d_in, max(cfg.k * 2, d_in // 4))
        self.preconditioner_down = nn.Linear(
            d_in, hidden_dim, bias=False, device=device, dtype=dtype
        )
        self.preconditioner_up = nn.Linear(
            hidden_dim, d_in, bias=False, device=device, dtype=dtype
        )

        # Keep the family close to a stable normalized identity map while
        # still making step-0 behavior observably different from plain top-k.
        self.preconditioner_down.weight.data.zero_()
        self.preconditioner_up.weight.data.zero_()
        diagonal = min(d_in, hidden_dim)
        self.preconditioner_down.weight.data[:diagonal, :diagonal] = torch.eye(
            diagonal, device=device, dtype=dtype
        )
        self.preconditioner_up.weight.data[:diagonal, :diagonal] = 0.05 * torch.eye(
            diagonal, device=device, dtype=dtype
        )

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        centered = x - x.mean(dim=-1, keepdim=True)
        rms = centered.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
        normalized = centered * rms
        correction = self.preconditioner_up(self.preconditioner_down(normalized))
        mixed = normalized + 0.25 * torch.roll(normalized, shifts=1, dims=-1)
        whitened = mixed + correction
        return fused_encoder(
            whitened, self.encoder.weight, self.encoder.bias, self.cfg.k
        )


class AdaptiveBudgetTopKSparseCoder(SparseCoder):
    """Allocate a fixed batch-level feature budget across samples dynamically."""

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        acts = F.relu(F.linear(x, self.encoder.weight, self.encoder.bias))

        batch_size = acts.shape[0]
        if batch_size == 1:
            top_acts, top_indices = torch.topk(
                acts, self.cfg.k, dim=-1, sorted=False
            )
            return EncoderOutput(top_acts, top_indices, acts)

        difficulty = acts.detach().pow(2).mean(dim=-1)
        difficulty_sum = difficulty.sum().clamp_min(torch.finfo(acts.dtype).tiny)
        total_budget = batch_size * self.cfg.k

        raw_quota = difficulty / difficulty_sum * total_budget
        quotas = torch.floor(raw_quota).to(dtype=torch.long)
        quotas = quotas.clamp(min=1, max=self.cfg.k)

        remaining = total_budget - int(quotas.sum().item())
        if remaining > 0:
            order = torch.argsort(raw_quota - quotas.to(raw_quota.dtype), descending=True)
            for idx in order.tolist():
                if remaining == 0:
                    break
                if quotas[idx] < self.cfg.k:
                    quotas[idx] += 1
                    remaining -= 1
        elif remaining < 0:
            order = torch.argsort(raw_quota - quotas.to(raw_quota.dtype))
            for idx in order.tolist():
                if remaining == 0:
                    break
                if quotas[idx] > 1:
                    quotas[idx] -= 1
                    remaining += 1

        top_acts, top_indices = torch.topk(acts, self.cfg.k, dim=-1, sorted=False)
        rank = torch.arange(self.cfg.k, device=acts.device)
        active_mask = rank.unsqueeze(0) < quotas.unsqueeze(1)
        top_acts = top_acts * active_mask.to(top_acts.dtype)
        return EncoderOutput(top_acts, top_indices, acts)


class BatchTopKSparseCoder(SparseCoder):
    """Select top activations under a global batch-level budget."""

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        acts = F.relu(F.linear(x, self.encoder.weight, self.encoder.bias))

        batch_size = acts.shape[0]
        if batch_size == 1:
            top_acts, top_indices = torch.topk(
                acts, self.cfg.k, dim=-1, sorted=False
            )
            return EncoderOutput(top_acts, top_indices, acts)

        flat = acts.reshape(-1)
        total_budget = batch_size * self.cfg.k
        total_budget = min(total_budget, flat.numel())
        _, flat_indices = torch.topk(flat, total_budget, sorted=False)

        selected = acts.new_zeros(acts.shape)
        selected.view(-1).scatter_(0, flat_indices, flat.index_select(0, flat_indices))

        top_acts, top_indices = torch.topk(selected, self.cfg.k, dim=-1, sorted=False)
        return EncoderOutput(top_acts, top_indices, acts)


class BucketedTopKSparseCoder(SparseCoder):
    """Route samples between two dictionaries based on activation norm."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.d_in = d_in
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor

        self.low_encoder = nn.Linear(
            d_in, self.num_latents, device=device, dtype=dtype
        )
        self.high_encoder = nn.Linear(
            d_in, self.num_latents, device=device, dtype=dtype
        )
        self.low_encoder.bias.data.zero_()
        self.high_encoder.bias.data.zero_()
        self.high_encoder.weight.data.mul_(1.05)

        self.bucket_scale = nn.Parameter(
            torch.tensor(2.0, device=device, dtype=dtype)
        )
        self.bucket_bias = nn.Parameter(
            torch.tensor(0.0, device=device, dtype=dtype)
        )

        if decoder:
            decoder_init = 0.5 * (
                self.low_encoder.weight.data + self.high_encoder.weight.data
            )
            self.W_dec = nn.Parameter(decoder_init)
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        low_acts = F.relu(F.linear(x, self.low_encoder.weight, self.low_encoder.bias))
        high_acts = F.relu(
            F.linear(x, self.high_encoder.weight, self.high_encoder.bias)
        )

        norms = x.norm(dim=-1, keepdim=True)
        centered_norms = norms - norms.mean()
        gate = torch.sigmoid(self.bucket_scale * centered_norms + self.bucket_bias)
        acts = (1.0 - gate) * low_acts + gate * high_acts

        top_acts, top_indices = torch.topk(acts, self.cfg.k, dim=-1, sorted=False)
        return EncoderOutput(top_acts, top_indices, acts)


class CodebookTopKSparseCoder(SparseCoder):
    """Reconstruct a coarse codebook vector, then sparsify the residual."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.d_in = d_in
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor
        self.num_codes = min(256, max(32, cfg.k * 2))

        self.codebook = nn.Parameter(
            torch.randn(self.num_codes, d_in, device=device, dtype=dtype) * 0.02
        )
        self.code_router = nn.Linear(d_in, self.num_codes, device=device, dtype=dtype)
        self.code_router.bias.data.zero_()

        self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
        self.encoder.bias.data.zero_()

        if decoder:
            self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def _select_code(self, x: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.code_router(x)
        code_indices = logits.argmax(dim=-1)
        probs = logits.softmax(dim=-1)
        hard_assign = F.one_hot(code_indices, num_classes=self.num_codes).to(
            dtype=probs.dtype
        )
        # Keep the routed code hard in the forward pass while preserving a gradient
        # path through the router so DDP does not see unused parameters.
        routing = hard_assign + probs - probs.detach()
        coarse = routing @ self.codebook
        return coarse, logits

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        coarse, _ = self._select_code(x)
        residual = x - coarse
        return fused_encoder(
            residual, self.encoder.weight, self.encoder.bias, self.cfg.k
        )

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        coarse, _ = self._select_code(x_centered)
        residual = x_centered - coarse
        top_acts, top_indices, pre_acts = fused_encoder(
            residual, self.encoder.weight, self.encoder.bias, self.cfg.k
        )

        assert self.W_dec is not None, "Decoder weight was not initialized."
        sparse_residual = decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec.mT)
        sae_out = coarse + sparse_residual + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            aux_sparse = decoder_impl(auxk_indices, auxk_acts.to(self.dtype), self.W_dec.mT)
            e_hat = coarse + aux_sparse + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
        )


class ResidualVQSparseCoder(SparseCoder):
    """Hard codebook reconstruction with sparse residual correction."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.d_in = d_in
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor
        self.num_codes = min(512, max(64, cfg.k * 4))

        self.codebook = nn.Parameter(
            torch.randn(self.num_codes, d_in, device=device, dtype=dtype) * 0.02
        )
        self.code_router = nn.Linear(d_in, self.num_codes, device=device, dtype=dtype)
        self.code_router.bias.data.zero_()

        self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
        self.encoder.bias.data.zero_()

        if decoder:
            self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def _select_code(self, x: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.code_router(x)
        code_indices = logits.argmax(dim=-1)
        probs = logits.softmax(dim=-1)
        hard_assign = F.one_hot(code_indices, num_classes=self.num_codes).to(
            dtype=probs.dtype
        )
        routing = hard_assign + probs - probs.detach()
        coarse = routing @ self.codebook
        return coarse, logits

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        coarse, _ = self._select_code(x)
        residual = x - coarse
        return fused_encoder(
            residual, self.encoder.weight, self.encoder.bias, self.cfg.k
        )

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        coarse, _ = self._select_code(x_centered)
        residual = x_centered - coarse
        top_acts, top_indices, pre_acts = fused_encoder(
            residual, self.encoder.weight, self.encoder.bias, self.cfg.k
        )

        assert self.W_dec is not None, "Decoder weight was not initialized."
        sparse_residual = decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec.mT)
        sae_out = coarse + sparse_residual + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            aux_sparse = decoder_impl(auxk_indices, auxk_acts.to(self.dtype), self.W_dec.mT)
            e_hat = coarse + aux_sparse + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
        )


class TwoCodeResidualVQSparseCoder(ResidualVQSparseCoder):
    """Two hard codebook reconstructions followed by sparse residual correction."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device=device, dtype=dtype, decoder=decoder)
        self.second_codebook = nn.Parameter(
            torch.randn(self.num_codes, d_in, device=device, dtype=dtype) * 0.02
        )
        self.second_code_router = nn.Linear(
            d_in, self.num_codes, device=device, dtype=dtype
        )
        self.second_code_router.bias.data.zero_()

    def _select_code_from(
        self, x: Tensor, router: nn.Linear, codebook: Tensor
    ) -> tuple[Tensor, Tensor]:
        logits = router(x)
        code_indices = logits.argmax(dim=-1)
        probs = logits.softmax(dim=-1)
        hard_assign = F.one_hot(code_indices, num_classes=self.num_codes).to(
            dtype=probs.dtype
        )
        routing = hard_assign + probs - probs.detach()
        coarse = routing @ codebook
        return coarse, logits

    def _select_codes(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        first_coarse, first_logits = self._select_code_from(
            x, self.code_router, self.codebook
        )
        second_input = x - first_coarse
        second_coarse, second_logits = self._select_code_from(
            second_input, self.second_code_router, self.second_codebook
        )
        combined = first_coarse + second_coarse
        return combined, first_logits, second_logits

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        coarse, _, _ = self._select_codes(x)
        residual = x - coarse
        return fused_encoder(
            residual, self.encoder.weight, self.encoder.bias, self.cfg.k
        )

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        coarse, _, _ = self._select_codes(x_centered)
        residual = x_centered - coarse
        top_acts, top_indices, pre_acts = fused_encoder(
            residual, self.encoder.weight, self.encoder.bias, self.cfg.k
        )

        assert self.W_dec is not None, "Decoder weight was not initialized."
        sparse_residual = decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec.mT)
        sae_out = coarse + sparse_residual + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            aux_sparse = decoder_impl(auxk_indices, auxk_acts.to(self.dtype), self.W_dec.mT)
            e_hat = coarse + aux_sparse + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
        )


class JumpReLUSparseCoder(SparseCoder):
    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device=device, dtype=dtype, decoder=decoder)

        threshold = torch.full(
            (self.num_latents,),
            cfg.jumprelu_init_threshold,
            device=self.device,
            dtype=self.dtype,
        ).clamp_min(torch.finfo(self.dtype).tiny)
        init = torch.full(
            (self.num_latents,), 0.0, device=self.device, dtype=self.dtype
        )
        init.copy_(torch.log(torch.expm1(threshold)))
        self.log_threshold = nn.Parameter(init)

    @property
    def threshold(self) -> Tensor:
        return F.softplus(self.log_threshold)

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        pre_acts = F.linear(x, self.encoder.weight, self.encoder.bias)
        positive = F.relu(pre_acts)
        gate = torch.sigmoid((positive - self.threshold) / self.cfg.jumprelu_bandwidth)
        acts = positive * gate
        top_acts, top_indices = torch.topk(acts, self.cfg.k, dim=-1, sorted=False)
        return EncoderOutput(top_acts, top_indices, acts)


class GatedSparseCoder(SparseCoder):
    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device=device, dtype=dtype, decoder=decoder)
        self.gate_encoder = nn.Linear(
            d_in, self.num_latents, device=device, dtype=dtype
        )

        # Start close to plain ReLU top-k, then let the gate branch learn support.
        self.gate_encoder.weight.data.zero_()
        self.gate_encoder.bias.data.fill_(cfg.gated_init_logit)

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        pre_acts = F.linear(x, self.encoder.weight, self.encoder.bias)
        positive = F.relu(pre_acts)
        gate_logits = self.gate_encoder(x) / self.cfg.gated_temperature
        gate = torch.sigmoid(gate_logits)
        acts = positive * gate
        top_acts, top_indices = torch.topk(acts, self.cfg.k, dim=-1, sorted=False)
        return EncoderOutput(top_acts, top_indices, acts)


class RoutedSparseCoder(SparseCoder):
    """Gated SAE with a separate router used only for support selection."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device=device, dtype=dtype, decoder=decoder)
        self.gate_encoder = nn.Linear(
            d_in, self.num_latents, device=device, dtype=dtype
        )

        # Initialize the router independently so routed support is observably
        # different from plain top-k at step 0.
        nn.init.kaiming_uniform_(self.gate_encoder.weight, a=5**0.5)
        self.gate_encoder.bias.data.zero_()

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        pre_acts = F.linear(x, self.encoder.weight, self.encoder.bias)
        positive = F.relu(pre_acts)
        gate_logits = self.gate_encoder(x) / self.cfg.gated_temperature
        gate = torch.sigmoid(gate_logits)

        # Use the router to perturb support selection directly while decoding
        # with gated magnitudes, so routing is not a monotone rescaling of the
        # base activations.
        acts = positive * gate
        scores = acts + 0.1 * torch.tanh(gate_logits)
        _, top_indices = torch.topk(scores, self.cfg.k, dim=-1, sorted=False)
        top_acts = acts.gather(-1, top_indices)
        return EncoderOutput(top_acts, top_indices, acts)


class GroupTopKSparseCoder(SparseCoder):
    """Select one winner per local group before the global top-k competition."""

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        acts = F.relu(F.linear(x, self.encoder.weight, self.encoder.bias))

        group_size = self.cfg.group_topk_size
        if self.num_latents % group_size != 0:
            raise ValueError(
                "group_topk requires num_latents divisible by group_topk_size, "
                f"got num_latents={self.num_latents} and "
                f"group_topk_size={group_size}"
            )

        num_groups = self.num_latents // group_size
        if self.cfg.k > num_groups:
            raise ValueError(
                "group_topk requires k <= number of groups, "
                f"got k={self.cfg.k} and num_groups={num_groups}"
            )

        grouped = acts.view(*acts.shape[:-1], num_groups, group_size)
        winner_acts, winner_offsets = grouped.max(dim=-1)
        top_group_acts, top_group_indices = torch.topk(
            winner_acts, self.cfg.k, dim=-1, sorted=False
        )

        top_offsets = winner_offsets.gather(-1, top_group_indices)
        top_indices = top_group_indices * group_size + top_offsets
        return EncoderOutput(top_group_acts, top_indices, acts)


class FactorizedTopKSparseCoder(SparseCoder):
    """Top-k SAE with a low-rank ReLU mixing stage before latent scoring."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.d_in = d_in
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor

        hidden_dim = min(self.num_latents, max(d_in // 2, cfg.k * 4))
        self.factor_encoder = nn.Linear(
            d_in, hidden_dim, device=device, dtype=dtype
        )
        self.encoder = nn.Linear(
            hidden_dim, self.num_latents, device=device, dtype=dtype
        )
        self.factor_encoder.bias.data.zero_()
        self.encoder.bias.data.zero_()

        if decoder:
            effective_encoder = self.encoder.weight.data @ self.factor_encoder.weight.data
            self.W_dec = nn.Parameter(effective_encoder)
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        hidden = F.relu(
            F.linear(x, self.factor_encoder.weight, self.factor_encoder.bias)
        )
        acts = F.relu(F.linear(hidden, self.encoder.weight, self.encoder.bias))
        top_acts, top_indices = torch.topk(acts, self.cfg.k, dim=-1, sorted=False)
        return EncoderOutput(top_acts, top_indices, acts)


class LowRankResidualSparseCoder(SparseCoder):
    """Model a dense low-rank trunk, then sparsely code the remaining residual."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.d_in = d_in
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor

        # Keep a meaningfully expressive dense trunk before sparse residual
        # coding. Tying trunk rank too tightly to expansion factor made the
        # prototype collapse to a minimal rank under the strong EF=16 recipe.
        trunk_rank = min(d_in, max(cfg.k * 2, d_in // 4))
        self.trunk_encoder = nn.Linear(
            d_in, trunk_rank, bias=False, device=device, dtype=dtype
        )
        self.trunk_decoder = nn.Linear(
            trunk_rank, d_in, bias=False, device=device, dtype=dtype
        )
        self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
        self.encoder.bias.data.zero_()

        if decoder:
            self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x))
        residual = x - trunk
        return fused_encoder(
            residual, self.encoder.weight, self.encoder.bias, self.cfg.k
        )

    def decode_residual(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."
        return decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec.mT)

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x_centered))
        residual = x_centered - trunk
        top_acts, top_indices, pre_acts = fused_encoder(
            residual, self.encoder.weight, self.encoder.bias, self.cfg.k
        )

        sparse_residual = self.decode_residual(top_acts, top_indices)
        sae_out = trunk + sparse_residual + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            e_hat = trunk + self.decode_residual(auxk_acts, auxk_indices) + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
        )


class LowRankGatedResidualSparseCoder(LowRankResidualSparseCoder):
    """Low-rank trunk with a gated sparse residual selector."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device=device, dtype=dtype, decoder=decoder)
        self.gate_encoder = nn.Linear(
            d_in, self.num_latents, device=device, dtype=dtype
        )

        # Start near the winning low-rank residual recipe, then let the gate
        # branch learn which residual features deserve support at low K.
        self.gate_encoder.weight.data.zero_()
        self.gate_encoder.bias.data.fill_(cfg.gated_init_logit)

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x))
        residual = x - trunk
        pre_acts = F.linear(residual, self.encoder.weight, self.encoder.bias)
        positive = F.relu(pre_acts)
        gate_logits = self.gate_encoder(residual) / self.cfg.gated_temperature
        gate = torch.sigmoid(gate_logits)
        acts = positive * gate
        top_acts, top_indices = torch.topk(acts, self.cfg.k, dim=-1, sorted=False)
        return EncoderOutput(top_acts, top_indices, acts)

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x_centered))
        residual = x_centered - trunk

        pre_acts = F.linear(residual, self.encoder.weight, self.encoder.bias)
        positive = F.relu(pre_acts)
        gate_logits = self.gate_encoder(residual) / self.cfg.gated_temperature
        gate = torch.sigmoid(gate_logits)
        acts = positive * gate
        top_acts, top_indices = torch.topk(acts, self.cfg.k, dim=-1, sorted=False)

        sparse_residual = self.decode_residual(top_acts, top_indices)
        sae_out = trunk + sparse_residual + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            e_hat = trunk + self.decode_residual(auxk_acts, auxk_indices) + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
        )


class WhitenedLowRankGatedResidualSparseCoder(LowRankResidualSparseCoder):
    """Low-rank trunk with whitened residual support selection."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg, device=device, dtype=dtype, decoder=decoder)
        self.gate_encoder = nn.Linear(
            d_in, self.num_latents, device=device, dtype=dtype
        )

        mix_rank = min(d_in, max(cfg.k * 2, d_in // 8))
        self.preconditioner_down = nn.Linear(
            d_in, mix_rank, bias=False, device=device, dtype=dtype
        )
        self.preconditioner_up = nn.Linear(
            mix_rank, d_in, bias=False, device=device, dtype=dtype
        )

        # Keep the initial behavior close to a stable normalized identity while
        # ensuring the architecture is observably distinct from plain residual gating.
        self.gate_encoder.weight.data.zero_()
        self.gate_encoder.bias.data.fill_(cfg.gated_init_logit)
        self.preconditioner_down.weight.data.zero_()
        self.preconditioner_up.weight.data.zero_()
        diagonal = min(d_in, mix_rank)
        self.preconditioner_down.weight.data[:diagonal, :diagonal] = torch.eye(
            diagonal, device=device, dtype=dtype
        )
        self.preconditioner_up.weight.data[:diagonal, :diagonal] = 0.05 * torch.eye(
            diagonal, device=device, dtype=dtype
        )

    def _precondition_residual(self, residual: Tensor) -> Tensor:
        centered = residual - residual.mean(dim=-1, keepdim=True)
        rms = centered.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
        normalized = centered * rms
        mixed = normalized + 0.25 * torch.roll(normalized, shifts=1, dims=-1)
        correction = self.preconditioner_up(self.preconditioner_down(normalized))
        return mixed + correction

    def _compute_acts(self, residual: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        conditioned = self._precondition_residual(residual)
        pre_acts = F.linear(conditioned, self.encoder.weight, self.encoder.bias)
        positive = F.relu(pre_acts)
        gate_logits = self.gate_encoder(conditioned) / self.cfg.gated_temperature
        gate = torch.sigmoid(gate_logits)
        acts = positive * gate
        return acts, conditioned, pre_acts

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x))
        residual = x - trunk
        acts, _, _ = self._compute_acts(residual)
        top_acts, top_indices = torch.topk(acts, self.cfg.k, dim=-1, sorted=False)
        return EncoderOutput(top_acts, top_indices, acts)

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x_centered))
        residual = x_centered - trunk

        acts, _, _ = self._compute_acts(residual)
        top_acts, top_indices = torch.topk(acts, self.cfg.k, dim=-1, sorted=False)

        sparse_residual = self.decode_residual(top_acts, top_indices)
        sae_out = trunk + sparse_residual + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            e_hat = trunk + self.decode_residual(auxk_acts, auxk_indices) + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
        )


class LowRankGroupedResidualSparseCoder(LowRankResidualSparseCoder):
    """Low-rank trunk with group-local winner selection on the sparse residual."""

    def _group_topk(
        self, acts: Tensor
    ) -> tuple[Tensor, Tensor]:
        group_size = self.cfg.group_topk_size
        if self.num_latents % group_size != 0:
            raise ValueError(
                "lowrank_grouped_residual requires num_latents divisible by "
                f"group_topk_size, got num_latents={self.num_latents} and "
                f"group_topk_size={group_size}"
            )

        num_groups = self.num_latents // group_size
        if self.cfg.k > num_groups:
            raise ValueError(
                "lowrank_grouped_residual requires k <= number of groups, "
                f"got k={self.cfg.k} and num_groups={num_groups}"
            )

        grouped = acts.view(*acts.shape[:-1], num_groups, group_size)
        winner_acts, winner_offsets = grouped.max(dim=-1)
        top_group_acts, top_group_indices = torch.topk(
            winner_acts, self.cfg.k, dim=-1, sorted=False
        )
        top_offsets = winner_offsets.gather(-1, top_group_indices)
        top_indices = top_group_indices * group_size + top_offsets
        return top_group_acts, top_indices

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x))
        residual = x - trunk
        acts = F.relu(F.linear(residual, self.encoder.weight, self.encoder.bias))
        top_acts, top_indices = self._group_topk(acts)
        return EncoderOutput(top_acts, top_indices, acts)

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec
        trunk = self.trunk_decoder(self.trunk_encoder(x_centered))
        residual = x_centered - trunk
        acts = F.relu(F.linear(residual, self.encoder.weight, self.encoder.bias))
        top_acts, top_indices = self._group_topk(acts)

        sparse_residual = self.decode_residual(top_acts, top_indices)
        sae_out = trunk + sparse_residual + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)
            auxk_latents = torch.where(dead_mask[None], acts, -torch.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
            e_hat = trunk + self.decode_residual(auxk_acts, auxk_indices) + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
        )


class TwoStageResidualSparseCoder(SparseCoder):
    """Allocate a fixed total K across two sparse residual passes."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.d_in = d_in
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor

        self.stage1_k = max(1, cfg.k // 2)
        self.stage2_k = max(1, cfg.k - self.stage1_k)

        self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
        self.encoder.bias.data.zero_()
        self.residual_encoder = nn.Linear(
            d_in, self.num_latents, device=device, dtype=dtype
        )
        self.residual_encoder.bias.data.zero_()

        if decoder:
            decoder_init = 0.5 * (
                self.encoder.weight.data + self.residual_encoder.weight.data
            )
            self.W_dec = nn.Parameter(decoder_init)
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def _decode_sparse(self, acts: Tensor, indices: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."
        return decoder_impl(indices, acts.to(self.dtype), self.W_dec.mT)

    @device_autocast
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        x_centered = x - self.b_dec

        stage1_acts, stage1_indices, stage1_pre = fused_encoder(
            x_centered, self.encoder.weight, self.encoder.bias, self.stage1_k
        )
        stage1_out = self._decode_sparse(stage1_acts, stage1_indices)

        residual = x_centered - stage1_out
        stage2_acts, stage2_indices, stage2_pre = fused_encoder(
            residual,
            self.residual_encoder.weight,
            self.residual_encoder.bias,
            self.stage2_k,
        )
        stage2_out = self._decode_sparse(stage2_acts, stage2_indices)

        sae_out = stage1_out + stage2_out + self.b_dec

        if y is None:
            y = x

        e = y - sae_out
        total_variance = (y - y.mean(0)).pow(2).sum()

        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            k_aux = y.shape[-1] // 2
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            aux_stage1 = torch.where(dead_mask[None], stage1_pre, -torch.inf)
            aux_stage2 = torch.where(dead_mask[None], stage2_pre, -torch.inf)

            aux1_k = min(self.stage1_k, k_aux)
            aux2_k = min(self.stage2_k, max(1, k_aux - aux1_k))

            aux1_acts, aux1_indices = aux_stage1.topk(aux1_k, sorted=False)
            aux1_out = self._decode_sparse(aux1_acts, aux1_indices)

            aux2_acts, aux2_indices = aux_stage2.topk(aux2_k, sorted=False)
            aux2_out = self._decode_sparse(aux2_acts, aux2_indices)

            e_hat = aux1_out + aux2_out + self.b_dec
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        combined_acts = torch.cat((stage1_acts, stage2_acts), dim=-1)
        combined_indices = torch.cat((stage1_indices, stage2_indices), dim=-1)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            combined_acts,
            combined_indices,
            fvu,
            auxk_loss,
        )


class MultiBranchGatedSparseCoder(SparseCoder):
    """Mix several gated encoder branches before the final top-k selection."""

    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.d_in = d_in
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor
        self.num_branches = 3

        self.branch_encoders = nn.ModuleList(
            [
                nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
                for _ in range(self.num_branches)
            ]
        )
        self.branch_gates = nn.ModuleList(
            [
                nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
                for _ in range(self.num_branches)
            ]
        )
        self.branch_mix_logits = nn.Linear(
            d_in, self.num_branches, device=device, dtype=dtype
        )

        for encoder in self.branch_encoders:
            encoder.bias.data.zero_()
        for gate in self.branch_gates:
            gate.weight.data.zero_()
            gate.bias.data.fill_(cfg.gated_init_logit)

        self.branch_mix_logits.weight.data.zero_()
        self.branch_mix_logits.bias.data.zero_()

        if decoder:
            decoder_init = torch.stack(
                [branch.weight.data for branch in self.branch_encoders], dim=0
            ).mean(dim=0)
            self.W_dec = nn.Parameter(decoder_init)
            if self.cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    def encode(self, x: Tensor) -> EncoderOutput:
        x = x - self.b_dec
        branch_weights = torch.softmax(self.branch_mix_logits(x), dim=-1)

        branch_acts = []
        for branch_idx, (encoder, gate_encoder) in enumerate(
            zip(self.branch_encoders, self.branch_gates)
        ):
            pre_acts = F.linear(x, encoder.weight, encoder.bias)
            positive = F.relu(pre_acts)
            gate_logits = gate_encoder(x) / self.cfg.gated_temperature
            gate = torch.sigmoid(gate_logits)
            acts = positive * gate
            branch_weight = branch_weights[..., branch_idx].unsqueeze(-1)
            branch_acts.append(acts * branch_weight)

        acts = torch.stack(branch_acts, dim=0).sum(dim=0)
        top_acts, top_indices = torch.topk(acts, self.cfg.k, dim=-1, sorted=False)
        return EncoderOutput(top_acts, top_indices, acts)
