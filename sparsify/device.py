"""Device abstraction layer for CUDA / Ascend NPU compatibility.

Provides unified APIs so the rest of the codebase never needs to call
``torch.cuda.*`` or ``torch.npu.*`` directly.
"""

from __future__ import annotations

import functools
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

_USE_NPU = False
try:
    import torch_npu  # noqa: F401

    _USE_NPU = torch.npu.is_available()
except ImportError:
    pass

_USE_CUDA = torch.cuda.is_available() and not _USE_NPU


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_device_type() -> str:
    """Return ``'npu'``, ``'cuda'``, or ``'cpu'``."""
    if _USE_NPU:
        return "npu"
    if _USE_CUDA:
        return "cuda"
    return "cpu"


def get_device(rank: int = 0) -> torch.device:
    """Return the appropriate :class:`torch.device` for *rank*."""
    return torch.device(f"{get_device_type()}:{rank}")


def get_device_string(rank: int = 0) -> str:
    """Return a device string like ``'cuda:0'`` or ``'npu:0'``."""
    return f"{get_device_type()}:{rank}"


def is_accelerator_available() -> bool:
    """Return ``True`` if either CUDA or NPU is available."""
    return _USE_CUDA or _USE_NPU


def is_bf16_supported() -> bool:
    """Check if bf16 is supported on the current accelerator."""
    if _USE_NPU:
        return True  # Ascend 910 series supports bf16
    if _USE_CUDA:
        return torch.cuda.is_bf16_supported()
    return False


def set_device(rank: int) -> None:
    """Set the current device for the given *rank*."""
    if _USE_NPU:
        torch.npu.set_device(rank)
    elif _USE_CUDA:
        torch.cuda.set_device(rank)


def synchronize() -> None:
    """Synchronize the current accelerator device."""
    if _USE_NPU:
        torch.npu.synchronize()
    elif _USE_CUDA:
        torch.cuda.synchronize()


def create_event(enable_timing: bool = True) -> Any:
    """Create an accelerator event for timing (or ``None`` on CPU)."""
    if _USE_NPU:
        return torch.npu.Event(enable_timing=enable_timing)
    if _USE_CUDA:
        return torch.cuda.Event(enable_timing=enable_timing)
    return None


def get_dist_backend() -> str:
    """Return the distributed backend name for the current platform."""
    if _USE_NPU:
        return "hccl"
    return "nccl"


def device_autocast(func):
    """Device-agnostic bf16 autocast decorator.

    Replaces ``@torch.autocast("cuda", dtype=torch.bfloat16, ...)`` with
    runtime device detection so the same code works on both CUDA and NPU.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with torch.autocast(
            get_device_type(),
            dtype=torch.bfloat16,
            enabled=is_bf16_supported(),
        ):
            return func(*args, **kwargs)

    return wrapper
