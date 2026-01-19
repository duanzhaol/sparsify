"""Block-diagonal Hadamard transform for activation preprocessing.

This module implements the Hadamard rotation technique inspired by QuaRot,
which helps distribute outlier energy across dimensions, potentially improving
SAE reconstruction quality.
"""

import math

import torch
from torch import Tensor


def hadamard_matrix(n: int, dtype: torch.dtype = torch.float32) -> Tensor:
    """Generate normalized Hadamard matrix of size n (must be power of 2).

    The returned matrix H satisfies H @ H.T = I (orthonormal).

    Args:
        n: Size of the Hadamard matrix (must be power of 2)
        dtype: Data type of the output tensor

    Returns:
        Normalized Hadamard matrix of shape [n, n]
    """
    assert n > 0 and (n & (n - 1)) == 0, f"n must be a power of 2, got {n}"
    H = torch.tensor([[1.0]], dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)],
            dim=0,
        )
    return H / math.sqrt(n)  # Normalize so H @ H.T = I


def block_hadamard_transform(x: Tensor, block_size: int = 128) -> Tensor:
    """Apply block-diagonal Hadamard transform.

    Args:
        x: Input tensor of shape [..., d_in]
        block_size: Size of each Hadamard block (must be power of 2)

    Returns:
        Transformed tensor of same shape
    """
    *batch_dims, d_in = x.shape
    assert (
        d_in % block_size == 0
    ), f"d_in ({d_in}) must be divisible by block_size ({block_size})"

    # Reshape to [batch, num_blocks, block_size]
    num_blocks = d_in // block_size
    x_reshaped = x.reshape(-1, num_blocks, block_size)

    # Get Hadamard matrix
    H = hadamard_matrix(block_size, dtype=x.dtype).to(x.device)

    # Apply Hadamard to each block: x @ H.T
    # Since H is symmetric and orthonormal, H.T = H
    y = torch.einsum("nbi,ij->nbj", x_reshaped, H)

    # Reshape back
    return y.reshape(*batch_dims, d_in)


class HadamardRotation:
    """Manages block-diagonal Hadamard rotation with optional random permutation.

    This class implements the rotation technique from QuaRot paper, which applies:
    1. Random permutation P (optional)
    2. Block-diagonal Hadamard transform H

    The full transform is: y = H_block @ P @ x

    Since both P and H are orthogonal, the inverse is: x = P.T @ H_block @ y
    (Note: H is self-inverse for normalized Hadamard)
    """

    def __init__(
        self,
        d_in: int,
        block_size: int = 128,
        seed: int = 0,
        use_permutation: bool = True,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize Hadamard rotation.

        Args:
            d_in: Input dimension
            block_size: Size of each Hadamard block (must be power of 2)
            seed: Random seed for permutation
            use_permutation: Whether to apply random permutation before Hadamard
            device: Device for tensors
            dtype: Data type for Hadamard matrix
        """
        self.d_in = d_in
        self.block_size = block_size
        self.use_permutation = use_permutation
        self.seed = seed

        # Ensure dimensions are compatible
        if d_in % block_size != 0:
            raise ValueError(
                f"d_in ({d_in}) must be divisible by block_size ({block_size})"
            )

        # Generate random permutation
        if use_permutation:
            g = torch.Generator().manual_seed(seed)
            self.perm = torch.randperm(d_in, generator=g).to(device)
            self.inv_perm = torch.argsort(self.perm)
        else:
            self.perm = None
            self.inv_perm = None

        # Pre-compute Hadamard matrix
        self.H = hadamard_matrix(block_size, dtype=dtype).to(device)
        self._device = device
        self._dtype = dtype
        # Cache for different dtypes to avoid repeated conversions
        self._H_cache: dict[torch.dtype, Tensor] = {dtype: self.H}

    def to(self, device: str | torch.device) -> "HadamardRotation":
        """Move rotation to specified device."""
        self._device = device
        self.H = self.H.to(device)
        # Clear dtype cache since device changed
        self._H_cache = {self.H.dtype: self.H}
        if self.perm is not None:
            self.perm = self.perm.to(device)
            self.inv_perm = self.inv_perm.to(device)
        return self

    def _get_H(self, dtype: torch.dtype) -> Tensor:
        """Get H matrix for given dtype, with caching to avoid repeated conversions."""
        if dtype not in self._H_cache:
            self._H_cache[dtype] = self.H.to(dtype)
        return self._H_cache[dtype]

    def _apply_block_hadamard(self, x: Tensor) -> Tensor:
        """Apply block-diagonal Hadamard transform using pre-computed H matrix."""
        *batch_dims, d_in = x.shape
        assert d_in == self.d_in, f"Expected d_in={self.d_in}, got {d_in}"
        num_blocks = d_in // self.block_size
        x_reshaped = x.reshape(-1, num_blocks, self.block_size)

        # Use cached H matrix for this dtype
        H = self._get_H(x.dtype)
        y = torch.einsum("nbi,ij->nbj", x_reshaped, H)

        return y.reshape(*batch_dims, d_in)

    def rotate(self, x: Tensor) -> Tensor:
        """Apply Hadamard rotation: H_block @ P @ x

        Args:
            x: Input tensor of shape [..., d_in]

        Returns:
            Rotated tensor of same shape
        """
        assert x.shape[-1] == self.d_in, f"Expected d_in={self.d_in}, got {x.shape[-1]}"
        # Apply permutation first
        if self.perm is not None:
            x = x[..., self.perm]
        # Then Hadamard using pre-computed matrix
        return self._apply_block_hadamard(x)

    def unrotate(self, x: Tensor) -> Tensor:
        """Apply inverse rotation: P.T @ H_block @ x

        For normalized Hadamard, H.T = H and H @ H = I, so H^{-1} = H.

        Args:
            x: Rotated tensor of shape [..., d_in]

        Returns:
            Original-space tensor of same shape
        """
        # Apply Hadamard first (self-inverse) using pre-computed matrix
        x = self._apply_block_hadamard(x)
        # Then inverse permutation
        if self.inv_perm is not None:
            x = x[..., self.inv_perm]
        return x

    def state_dict(self) -> dict:
        """Get state dict for serialization."""
        return {
            "d_in": self.d_in,
            "block_size": self.block_size,
            "use_permutation": self.use_permutation,
            "seed": self.seed,
            "perm": self.perm,
            "inv_perm": self.inv_perm,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: dict,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "HadamardRotation":
        """Load from state dict."""
        instance = cls.__new__(cls)
        instance.d_in = state["d_in"]
        instance.block_size = state["block_size"]
        instance.use_permutation = state["use_permutation"]
        instance.seed = state.get("seed", 0)
        instance.perm = (
            state["perm"].to(device) if state["perm"] is not None else None
        )
        instance.inv_perm = (
            state["inv_perm"].to(device) if state["inv_perm"] is not None else None
        )
        instance.H = hadamard_matrix(instance.block_size, dtype=dtype).to(device)
        instance._device = device
        instance._dtype = dtype
        # Initialize dtype cache
        instance._H_cache = {dtype: instance.H}
        return instance


# Optional: Use fast-hadamard-transform library if available
try:
    from fast_hadamard_transform import hadamard_transform as _fast_hadamard

    HAS_FAST_HADAMARD = True
except ImportError:
    HAS_FAST_HADAMARD = False


def fast_hadamard_transform(x: Tensor, block_size: int = 128) -> Tensor:
    """Apply block-diagonal Hadamard transform with optional CUDA acceleration.

    Uses fast-hadamard-transform library if available, otherwise falls back
    to pure PyTorch implementation.

    Args:
        x: Input tensor of shape [..., d_in]
        block_size: Size of each Hadamard block

    Returns:
        Transformed tensor of same shape
    """
    if HAS_FAST_HADAMARD and x.is_cuda:
        # fast-hadamard-transform expects [batch, block_size]
        *batch_dims, d_in = x.shape
        num_blocks = d_in // block_size
        x_reshaped = x.reshape(-1, num_blocks, block_size)
        # Apply to each block
        y = _fast_hadamard(x_reshaped.reshape(-1, block_size))
        return y.reshape(*batch_dims, d_in)
    else:
        return block_hadamard_transform(x, block_size)
