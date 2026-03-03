"""Verify sparsify.device API returns correct results on real Ascend NPU."""

import torch

from sparsify.device import (
    create_event,
    get_device,
    get_device_string,
    get_device_type,
    get_dist_backend,
    is_accelerator_available,
    is_bf16_supported,
    set_device,
    synchronize,
)


def test_get_device_type_returns_npu():
    assert get_device_type() == "npu"


def test_get_device_returns_npu_device():
    d = get_device(0)
    assert d == torch.device("npu:0")


def test_get_device_string():
    assert get_device_string(0) == "npu:0"
    assert get_device_string(3) == "npu:3"


def test_is_accelerator_available():
    assert is_accelerator_available() is True


def test_is_bf16_supported():
    assert is_bf16_supported() is True


def test_set_device():
    set_device(0)  # should not raise


def test_synchronize():
    x = torch.randn(100, device="npu")
    _ = x + x  # ensure there is something to sync
    synchronize()  # should not raise


def test_create_event():
    event = create_event(enable_timing=True)
    assert event is not None


def test_event_record_and_elapsed():
    start = create_event(enable_timing=True)
    end = create_event(enable_timing=True)
    synchronize()
    start.record()
    x = torch.randn(512, 512, device="npu")
    _ = x @ x
    end.record()
    synchronize()
    elapsed = start.elapsed_time(end)
    assert elapsed >= 0


def test_get_dist_backend():
    assert get_dist_backend() == "hccl"
