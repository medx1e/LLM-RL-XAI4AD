"""Pipeline-wide configuration for the SAE interpretability pipeline.

SAEConfig defaults are automatically loaded from the sibling sae_config.yaml
at import time.  Any field absent from the YAML falls back to its dataclass
default, so the class always constructs successfully even if the YAML is
missing or incomplete.

Typical usage
-------------
# Uses sae_config.yaml next to this file — no arguments needed:
cfg = SAEConfig.from_yaml()

# Override a specific YAML file:
cfg = SAEConfig.from_yaml("path/to/other_config.yaml")

# Override individual fields after loading:
cfg = SAEConfig.from_yaml()
cfg.sae_l1_coeff = 1e-5
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import yaml

# ---------------------------------------------------------------------------
# Path to the canonical YAML file (sibling of this module)
# ---------------------------------------------------------------------------
_DEFAULT_YAML = os.path.join(os.path.dirname(__file__), "sae_config.yaml")


def _load_yaml_defaults(path: str) -> dict:
    """Load YAML and return only keys that are valid SAEConfig fields."""
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    # Strip the 'tune' block and any other non-field keys at load time
    return data


@dataclass
class SAEConfig:
    # Wayformer residual stream dimension (dk * ff_mult from encoder config)
    wayformer_hidden_dim: int = 128

    # SAE architecture
    sae_expansion_factor: int = 16
    sae_l1_coeff: float = 1e-3
    sae_learning_rate: float = 3e-4
    sae_batch_size: int = 4096
    sae_epochs: int = 10

    # Feature annotation
    top_k_activations: int = 50

    # Causal steering
    steering_temperatures: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0])

    # Harvesting
    harvest_max_timesteps: int = 80
    harvest_n_scenarios: int = 500
    harvest_dt: float = 0.1  # Waymo dataset: 10 Hz

    # Telemetry thresholds
    ttc_critical_threshold: float = 1.5      # seconds
    hard_braking_g_threshold: float = 0.4   # g-force

    @property
    def sae_latent_dim(self) -> int:
        return self.wayformer_hidden_dim * self.sae_expansion_factor

    @classmethod
    def from_yaml(cls, path: str = _DEFAULT_YAML) -> SAEConfig:
        """Load config from a YAML file.

        Args:
            path: Path to the YAML file.  Defaults to the sibling
                  ``sae_config.yaml`` so callers can write simply::

                      cfg = SAEConfig.from_yaml()
        """
        raw = _load_yaml_defaults(path)
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in raw.items() if k in valid_fields}
        return cls(**filtered)
