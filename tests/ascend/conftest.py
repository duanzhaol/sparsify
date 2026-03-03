"""Ascend NPU test configuration.

All tests under tests/ascend/ are automatically skipped when no NPU hardware
is available.  This is handled by the ``pytest_collection_modifyitems`` hook
below so that individual test files do not need to add their own skip markers.
"""

import pytest
import torch

# ---------------------------------------------------------------------------
# NPU detection
# ---------------------------------------------------------------------------

_HAS_NPU = False
try:
    import torch_npu  # noqa: F401

    _HAS_NPU = torch.npu.is_available()
except ImportError:
    pass

_requires_npu = pytest.mark.skipif(not _HAS_NPU, reason="Requires Ascend NPU")


# ---------------------------------------------------------------------------
# Auto-skip all tests in this directory when NPU is unavailable
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(items):
    for item in items:
        if "/ascend/" in str(item.fspath) or "\\ascend\\" in str(item.fspath):
            item.add_marker(_requires_npu)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def npu():
    """Return the device string ``'npu'``."""
    return "npu"


@pytest.fixture
def npu_device():
    """Return ``torch.device('npu:0')``."""
    return torch.device("npu:0")
