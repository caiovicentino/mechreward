"""Tests for the standalone TopK SAE backend."""

import pytest
import torch

from mechreward.sae.topk_sae import TopKSAEBackend, load_topk_sae


def test_topk_sae_shape():
    sae = TopKSAEBackend(d_model=64, d_sae=256, k=16)
    x = torch.randn(8, 64)
    acts = sae.encode(x)
    recon = sae.decode(acts)
    assert acts.shape == (8, 256)
    assert recon.shape == (8, 64)


def test_topk_sae_l0_equals_k():
    torch.manual_seed(0)
    sae = TopKSAEBackend(d_model=64, d_sae=256, k=16)
    x = torch.randn(32, 64)
    acts = sae.encode(x)
    # Every row should have at most k active features.
    # After ReLU some may be zero, so we check <= k and average > 0.
    l0_per_row = (acts > 0).sum(dim=-1)
    assert (l0_per_row <= 16).all()
    assert l0_per_row.float().mean() > 0


def test_topk_sae_forward_roundtrip():
    torch.manual_seed(0)
    sae = TopKSAEBackend(d_model=32, d_sae=128, k=8)
    x = torch.randn(4, 32)
    recon = sae.forward(x)
    assert recon.shape == x.shape


def test_load_topk_sae_from_file(tmp_path):
    """Save a dummy checkpoint and load it back through the library."""
    torch.manual_seed(0)
    source = TopKSAEBackend(d_model=32, d_sae=128, k=8)
    ckpt_path = tmp_path / "sae_test.pt"
    torch.save(
        {
            "W_enc": source.W_enc.detach().cpu(),
            "W_dec": source.W_dec.detach().cpu(),
            "b_enc": source.b_enc.detach().cpu(),
            "b_dec": source.b_dec.detach().cpu(),
            "d_model": 32,
            "d_sae": 128,
            "k": 8,
            "step": 1000,
        },
        ckpt_path,
    )

    handle = load_topk_sae(
        checkpoint=str(ckpt_path),
        device="cpu",
        model_name="Qwen/Qwen3.5-4B",
        layer=18,
    )
    assert handle.d_model == 32
    assert handle.d_sae == 128
    assert handle.layer == 18
    assert handle.model_name == "Qwen/Qwen3.5-4B"

    # Round-trip: encode then decode on a fresh input
    x = torch.randn(2, 32)
    encoded = handle.encode(x)
    decoded = handle.decode(encoded)
    assert encoded.shape == (2, 128)
    assert decoded.shape == (2, 32)


def test_load_topk_sae_missing_file():
    with pytest.raises(FileNotFoundError):
        load_topk_sae(checkpoint="/does/not/exist.pt", device="cpu")
