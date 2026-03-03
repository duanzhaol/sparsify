import pytest
import torch

from sparsify.device import get_device_type, is_accelerator_available

requires_accelerator = pytest.mark.skipif(
    not is_accelerator_available(), reason="CUDA or NPU required"
)


@pytest.fixture
def device():
    return get_device_type() if is_accelerator_available() else "cpu"
