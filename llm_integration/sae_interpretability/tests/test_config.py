# Copyright 2025 Valeo.

"""Unit tests for SAEConfig."""

import os
import tempfile

import pytest
import yaml

from sae_interpretability.config import SAEConfig


def test_default_values():
    cfg = SAEConfig()
    assert cfg.wayformer_hidden_dim == 128
    assert cfg.sae_expansion_factor == 16
    assert cfg.sae_l1_coeff == pytest.approx(1e-3)
    assert cfg.sae_learning_rate == pytest.approx(3e-4)
    assert cfg.ttc_critical_threshold == pytest.approx(1.5)
    assert cfg.hard_braking_g_threshold == pytest.approx(0.4)
    assert cfg.harvest_dt == pytest.approx(0.1)


def test_sae_latent_dim_property():
    cfg = SAEConfig(wayformer_hidden_dim=128, sae_expansion_factor=16)
    assert cfg.sae_latent_dim == 2048


def test_sae_latent_dim_scales():
    cfg = SAEConfig(wayformer_hidden_dim=64, sae_expansion_factor=32)
    assert cfg.sae_latent_dim == 2048


def test_from_yaml_roundtrip():
    data = {
        'wayformer_hidden_dim': 128,
        'sae_expansion_factor': 16,
        'sae_l1_coeff': 5e-4,
        'sae_learning_rate': 3e-4,
        'sae_batch_size': 4096,
        'sae_epochs': 20,
        'top_k_activations': 50,
        'harvest_max_timesteps': 80,
        'harvest_n_scenarios': 500,
        'harvest_dt': 0.1,
        'ttc_critical_threshold': 1.5,
        'hard_braking_g_threshold': 0.4,
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(data, f)
        path = f.name
    try:
        loaded = SAEConfig.from_yaml(path)
        assert loaded.sae_epochs == 20
        assert loaded.sae_l1_coeff == pytest.approx(5e-4)
    finally:
        os.unlink(path)


def test_bundled_yaml_matches_defaults():
    """The shipped sae_config.yaml should be consistent with SAEConfig defaults."""
    yaml_path = os.path.join(os.path.dirname(__file__), '..', 'sae_config.yaml')
    if not os.path.exists(yaml_path):
        pytest.skip("sae_config.yaml not found")
    loaded = SAEConfig.from_yaml(yaml_path)
    default = SAEConfig()
    assert loaded.wayformer_hidden_dim == default.wayformer_hidden_dim
    assert loaded.sae_expansion_factor == default.sae_expansion_factor
    assert loaded.sae_l1_coeff == pytest.approx(default.sae_l1_coeff)
    assert loaded.ttc_critical_threshold == pytest.approx(default.ttc_critical_threshold)
    assert loaded.hard_braking_g_threshold == pytest.approx(default.hard_braking_g_threshold)


def test_unknown_yaml_keys_ignored():
    data = {'wayformer_hidden_dim': 64, 'unknown_future_key': 'ignored'}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(data, f)
        path = f.name
    try:
        cfg = SAEConfig.from_yaml(path)
        assert cfg.wayformer_hidden_dim == 64
    finally:
        os.unlink(path)
