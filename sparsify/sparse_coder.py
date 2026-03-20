import json
from fnmatch import fnmatch
from pathlib import Path
from typing import NamedTuple

import einops
import torch
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
    if architecture == "gated":
        from .gated_sparse_coder import GatedSparseCoder
        return GatedSparseCoder
    if architecture == "jumprelu":
        from .jumprelu_sparse_coder import JumpReLUSparseCoder
        return JumpReLUSparseCoder
    if architecture == "group_topk":
        from .group_topk_sparse_coder import GroupTopKSparseCoder
        return GroupTopKSparseCoder
    if architecture == "routed_group_topk":
        from .routed_group_topk_sparse_coder import RoutedGroupTopKSparseCoder
        return RoutedGroupTopKSparseCoder
    if architecture == "residual_topk":
        from .residual_topk_sparse_coder import ResidualTopKSparseCoder
        return ResidualTopKSparseCoder
    if architecture == "factorized_topk":
        from .factorized_topk_sparse_coder import FactorizedTopKSparseCoder
        return FactorizedTopKSparseCoder
    if architecture == "mixture_topk":
        from .mixture_topk_sparse_coder import MixtureTopKSparseCoder
        return MixtureTopKSparseCoder
    raise ValueError(f"Unknown architecture: {architecture!r}")


# Allow for alternate naming conventions
Sae = SparseCoder
