"""Integrated Gradients baseline utilities.

The zero-vector baseline (default IG) is semantically wrong in V-MAX's
normalized observation space: zero ≠ "empty scene" — it means "all features
at the normalization center," which for agent features means "agent at ego
position with zero velocity."

This module provides a validity-zeroed mean baseline:
  1. Compute the mean observation over a dataset of any size
  2. Detect binary (validity/mask) features from the data itself
  3. Zero out those validity features in the mean

Result: "average road layout + average GPS path, but no agents and no
traffic lights" — a semantically clean reference for the IG path integral.

Completeness axiom is preserved:
  IG attributions sum to F(x) - F(baseline)
With the new baseline this means "importance relative to the typical empty
scene" instead of "importance relative to an all-zero nonsense input."

Usage (any dataset size):
    from posthoc_xai.utils.ig_baseline import compute_baseline

    # From platform_cache (240 obs):
    baseline = compute_baseline(raw_obs_array)           # (1655,)

    # From a larger dataset (streaming):
    accumulator = BaselineAccumulator()
    for batch in data_loader:
        accumulator.update(batch)
    baseline = accumulator.finalize()
"""

from __future__ import annotations
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Binary (validity) feature detection
# ---------------------------------------------------------------------------

def detect_binary_features(
    obs_array: np.ndarray,
    tol: float = 0.02,
) -> np.ndarray:
    """Return a boolean mask of features that appear binary in the dataset.

    A feature is considered binary if all observed values are within `tol`
    of either 0 or 1. This data-driven approach identifies validity/mask bits
    without hardcoding feature positions — robust to architecture changes.

    Args:
        obs_array: Shape ``(N, obs_dim)``. At least ~50 observations for
            reliable detection (more is better).
        tol: Tolerance for determining if a value is "close to 0 or 1."

    Returns:
        Boolean numpy array of shape ``(obs_dim,)``.
        True = feature is binary (validity/mask bit), False = continuous.
    """
    obs = np.array(obs_array)
    near_zero = np.abs(obs) <= tol
    near_one  = np.abs(obs - 1.0) <= tol
    is_binary = np.all(near_zero | near_one, axis=0)   # (obs_dim,)
    return is_binary


# ---------------------------------------------------------------------------
# Baseline computation
# ---------------------------------------------------------------------------

def compute_baseline(
    obs_array: np.ndarray,
    tol: float = 0.02,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Compute a validity-zeroed mean baseline from a set of observations.

    Steps:
      1. Compute the per-feature mean across all N observations.
      2. Detect binary features (validity/mask bits) via ``detect_binary_features``.
      3. Zero out those binary features in the mean.

    The result represents: "average road layout + average GPS path,
    but all agents and traffic lights absent (validity=0)."

    Args:
        obs_array: Shape ``(N, obs_dim)``. Can be any size — use
            ``BaselineAccumulator`` for very large datasets.
        tol: Tolerance for binary feature detection.
        dtype: Output dtype (default float32 for JAX compatibility).

    Returns:
        Baseline vector of shape ``(obs_dim,)``.
    """
    obs = np.array(obs_array, dtype=np.float64)   # high precision for mean

    mean_obs     = obs.mean(axis=0)                # (obs_dim,)
    binary_mask  = detect_binary_features(obs, tol=tol)

    baseline = mean_obs.copy()
    baseline[binary_mask] = 0.0                   # zero out all validity bits

    return baseline.astype(dtype)


def compute_baseline_stats(obs_array: np.ndarray, tol: float = 0.02) -> dict:
    """Return diagnostic stats about the baseline computation.

    Useful for verifying the binary detection is working as expected.
    """
    obs         = np.array(obs_array)
    binary_mask = detect_binary_features(obs, tol=tol)
    mean_obs    = obs.mean(axis=0)

    n_binary = binary_mask.sum()
    n_total  = len(binary_mask)

    # Which categories do the binary features fall in?
    CAT_RANGES = {
        "sdc_trajectory": (0,    40),
        "other_agents":   (40,   360),
        "roadgraph":      (360,  1360),
        "traffic_lights": (1360, 1635),
        "gps_path":       (1635, 1655),
    }
    binary_per_cat = {}
    for cat, (s, e) in CAT_RANGES.items():
        binary_per_cat[cat] = int(binary_mask[s:e].sum())

    return {
        "n_observations":    obs.shape[0],
        "obs_dim":           obs.shape[1],
        "n_binary_features": int(n_binary),
        "pct_binary":        float(n_binary / n_total * 100),
        "binary_per_category": binary_per_cat,
        "mean_obs_min":      float(mean_obs.min()),
        "mean_obs_max":      float(mean_obs.max()),
        "mean_obs_std":      float(mean_obs.std()),
    }


# ---------------------------------------------------------------------------
# Streaming accumulator for large datasets
# ---------------------------------------------------------------------------

class BaselineAccumulator:
    """Welford online algorithm for computing mean and binary mask incrementally.

    Use this when the full observation array doesn't fit in memory.

    Example::

        acc = BaselineAccumulator()
        for scenario in scenarios:
            obs_batch = load_observations(scenario)   # (T, obs_dim)
            acc.update(obs_batch)
        baseline = acc.finalize()
    """

    def __init__(self, tol: float = 0.02, dtype: np.dtype = np.float32):
        self.tol   = tol
        self.dtype = dtype
        self._n    = 0
        self._mean: Optional[np.ndarray] = None
        self._all_near_zero_or_one: Optional[np.ndarray] = None

    def update(self, obs_batch: np.ndarray):
        """Add a batch of observations to the accumulator.

        Args:
            obs_batch: Shape ``(batch, obs_dim)`` or ``(obs_dim,)`` for
                single observation.
        """
        batch = np.array(obs_batch, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch[np.newaxis, :]

        B = batch.shape[0]

        near_zero_or_one = (np.abs(batch) <= self.tol) | (np.abs(batch - 1.0) <= self.tol)

        if self._mean is None:
            self._mean = batch.mean(axis=0)
            self._all_near_zero_or_one = near_zero_or_one.all(axis=0)
            self._n = B
        else:
            # Welford update for mean
            new_n    = self._n + B
            delta    = batch.mean(axis=0) - self._mean
            self._mean += delta * (B / new_n)
            self._n = new_n
            # Binary mask: feature is binary only if ALL observed values match
            self._all_near_zero_or_one &= near_zero_or_one.all(axis=0)

    def finalize(self) -> np.ndarray:
        """Return the validity-zeroed mean baseline."""
        if self._mean is None:
            raise RuntimeError("No observations added yet.")
        baseline = self._mean.copy()
        baseline[self._all_near_zero_or_one] = 0.0
        return baseline.astype(self.dtype)

    @property
    def n_observations(self) -> int:
        return self._n
