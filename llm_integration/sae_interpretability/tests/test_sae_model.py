# Copyright 2025 Valeo.

"""Unit tests for SparseAutoencoder."""

import os
import tempfile

import pytest
import torch
import torch.nn.functional as F

from sae_interpretability.sae_model import SparseAutoencoder


@pytest.fixture
def sae():
    return SparseAutoencoder(input_dim=128, latent_dim=2048, l1_coeff=1e-3)


def test_output_shapes(sae):
    x = torch.randn(32, 128)
    x_hat, features, loss = sae(x)
    assert x_hat.shape == (32, 128)
    assert features.shape == (32, 2048)
    assert loss.ndim == 0  # scalar


def test_loss_is_finite(sae):
    x = torch.randn(64, 128)
    _, _, loss = sae(x)
    assert torch.isfinite(loss).item()


def test_decoder_rows_unit_norm_at_init(sae):
    """Decoder rows should have unit norm right after construction."""
    norms = sae.W_dec.data.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_normalize_decoder_restores_unit_norm(sae):
    with torch.no_grad():
        sae.W_dec.data *= 3.7  # corrupt
    sae.normalize_decoder()
    norms = sae.W_dec.data.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_encode_output_nonnegative(sae):
    x = torch.randn(16, 128)
    f = sae.encode(x)
    assert (f >= 0).all()


def test_encode_sparsity(sae):
    """After ReLU many features should be zero for small random input."""
    x = torch.randn(256, 128) * 0.1
    f = sae.encode(x)
    zero_frac = (f == 0).float().mean().item()
    # With random small weights + ReLU, expect at least 20% zeros
    assert zero_frac > 0.2, f"Expected sparsity > 0.2, got {zero_frac:.3f}"


def test_to_numpy_weights_shapes(sae):
    w = sae.to_numpy_weights()
    assert w['W_enc'].shape == (128, 2048)
    assert w['W_dec'].shape == (2048, 128)
    assert w['b_enc'].shape == (2048,)
    assert w['b_dec'].shape == (128,)
    assert w['pre_bias'].shape == (128,)


def test_checkpoint_roundtrip(sae):
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    try:
        import numpy as np
        torch.save({
            'model_state_dict': sae.state_dict(),
            'config': {'input_dim': 128, 'latent_dim': 2048, 'l1_coeff': 1e-3},
            'act_mean': np.zeros((1, 128), dtype=np.float32),
        }, path)
        loaded = SparseAutoencoder.from_checkpoint(path)
        x = torch.randn(8, 128)
        _, f_orig, _ = sae(x)
        _, f_load, _ = loaded(x)
        assert torch.allclose(f_orig, f_load, atol=1e-6)
    finally:
        os.unlink(path)


def test_l1_penalty_drives_sparsity():
    """Higher l1_coeff → sparser features after a few gradient steps."""
    torch.manual_seed(0)
    x = torch.randn(512, 128)

    def run_steps(l1_coeff, steps=20):
        m = SparseAutoencoder(128, 512, l1_coeff)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(steps):
            opt.zero_grad()
            _, _, loss = m(x)
            loss.backward()
            opt.step()
            m.normalize_decoder()
        with torch.no_grad():
            f = m.encode(x)
        return (f == 0).float().mean().item()

    sparse_high = run_steps(1e-1)
    sparse_low = run_steps(1e-5)
    assert sparse_high > sparse_low, (
        f"Higher l1 should produce more zeros: high={sparse_high:.3f}, low={sparse_low:.3f}"
    )
