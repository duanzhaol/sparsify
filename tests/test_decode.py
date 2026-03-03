import pytest
import torch

from sparsify.device import get_device_type, is_accelerator_available
from sparsify.utils import decoder_impl, eager_decode, triton_decode


@pytest.mark.skipif(not is_accelerator_available(), reason="CUDA or NPU required")
@pytest.mark.parametrize("d_in", [48, 64])  # Power of 2 and not
def test_decode(d_in: int):
    batch = 2
    d_sae = 128
    k = 10

    dev = get_device_type()
    # Fake data
    latents = torch.rand(batch, d_sae, device=dev)
    W_dec = torch.randn(d_sae, d_in, device=dev)

    top_vals, top_idx = latents.topk(k)
    eager_res = eager_decode(top_idx, top_vals, W_dec.mT)

    if decoder_impl is triton_decode:
        triton_res = triton_decode(top_idx, top_vals, W_dec.mT)
        torch.testing.assert_close(eager_res, triton_res)
    else:
        # Triton not available (NPU or no Triton installed); just verify
        # that the eager decode runs without error and has the right shape.
        assert eager_res.shape == (batch, d_in)
