"""Post-hoc XAI backend adapter.

Design target
-------------
Fully cached curated demo flow: the Streamlit tab loads cached artifacts
only; no live ``posthoc_xai.explain()`` calls during a demo.

On-disk cache layout (rooted at ``platform_cache/{model_slug}/``):

    scenario_{idx:04d}_artifact.pkl
        PlatformScenarioArtifact — ScenarioData + raw_observations +
        interesting_timesteps + notes + metadata.

    scenario_{idx:04d}_attr_{method}.pkl
        list[Attribution] of length == artifact.num_steps.  One
        pre-computed attribution per timestep for a single method.

    scenario_{idx:04d}_attention.pkl
        list[dict[str, np.ndarray]] of length == artifact.num_steps.
        Per-timestep attention weights (cross_attention, self_attention)
        captured from the Perceiver/LQ encoder.  Absent for non-attention
        models.

Runtime lookup (``get_explanation``/``get_attention``):
  1. CACHED SERIES — load the list, return the slice for ``step``.
  2. DEV FALLBACK  — only when ``explainable_model`` is passed AND raw
     observations are present.  Intended for development, NOT for demos.
  3. ``XAINotReadyError`` — surfaced to the tab as a user-visible message.

Curation API (``precompute_attribution_series`` etc.) populates tier 1.

Loading ExplainableModel
------------------------
``load_explainable_model(entry)`` is a pure Python function.  The Streamlit
tab should wrap it with ``@st.cache_resource`` only when the dev fallback
is intentionally enabled.  For curated demos, the model never needs to be
loaded at runtime — cached series alone are enough.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

import platform  # path bootstrap
from platform.shared.contracts import PlatformScenarioArtifact, XAINotReadyError
from platform.shared.model_catalog import PLATFORM_MODELS, ModelEntry

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PLATFORM_CACHE_ROOT = _PROJECT_ROOT / "platform_cache"

# Default methods used when the tab hasn't specified a preference.
DEFAULT_METHODS = ["vanilla_gradient", "integrated_gradients", "perturbation"]


# ---------------------------------------------------------------------------
# Internal path helpers
# ---------------------------------------------------------------------------

def _cache_dir(model_key: str) -> Path:
    entry = PLATFORM_MODELS[model_key]
    return _PLATFORM_CACHE_ROOT / entry.cache_slug


def _attr_path(model_key: str, scenario_idx: int, method: str) -> Path:
    return _cache_dir(model_key) / f"scenario_{scenario_idx:04d}_attr_{method}.pkl"


def _attention_path(model_key: str, scenario_idx: int) -> Path:
    return _cache_dir(model_key) / f"scenario_{scenario_idx:04d}_attention.pkl"


# ---------------------------------------------------------------------------
# Cache inspection (no JAX, safe at Streamlit import time)
# ---------------------------------------------------------------------------

def list_cached_methods(model_key: str, scenario_idx: int) -> list[str]:
    """Return the attribution method names that have a cached series on disk."""
    cache_dir = _cache_dir(model_key)
    if not cache_dir.exists():
        return []
    prefix = f"scenario_{scenario_idx:04d}_attr_"
    methods = []
    for f in cache_dir.glob(f"{prefix}*.pkl"):
        name = f.stem[len(prefix):]
        if name:
            methods.append(name)
    return sorted(methods)


def has_cached_attention(model_key: str, scenario_idx: int) -> bool:
    return _attention_path(model_key, scenario_idx).exists()


# ---------------------------------------------------------------------------
# Model loading (for dev fallback + precompute only)
# ---------------------------------------------------------------------------

def load_explainable_model(entry: ModelEntry):
    """Load and return an ExplainableModel for the given catalog entry.

    Imports posthoc_xai and triggers JAX/Flax initialization (expensive).
    Use only for:
      - the offline precompute scripts, or
      - a developer-mode tab that enables on-demand computation.

    Curated demo tabs should NOT call this.

    Returns
    -------
    posthoc_xai.ExplainableModel (PerceiverWrapper or GenericWrapper).
    """
    import posthoc_xai as xai
    return xai.load_model(entry.model_dir)


# ---------------------------------------------------------------------------
# Attribution access (cache-first with optional dev fallback)
# ---------------------------------------------------------------------------

def get_explanation(
    artifact: PlatformScenarioArtifact,
    step: int,
    method: str,
    explainable_model=None,
):
    """Return an Attribution for (artifact, step, method).

    Lookup order:
      1. Cached attribution series on disk — always preferred.
      2. Dev fallback: compute on-demand if ``explainable_model`` is
         provided AND ``artifact.raw_observations`` is present.
      3. Raise XAINotReadyError (no silent fallback).

    Parameters
    ----------
    artifact          : Loaded PlatformScenarioArtifact.
    step              : Zero-indexed timestep to explain.
    method            : Attribution method name (see posthoc_xai.METHOD_REGISTRY).
    explainable_model : OPTIONAL.  When None (default), cache-only mode.
                        When provided, the dev fallback is enabled.
                        Curated demo code should leave this as None.
    """
    # Tier 1 — cached series
    path = _attr_path(artifact.model_key, artifact.scenario_idx, method)
    if path.exists():
        with open(path, "rb") as fh:
            series = pickle.load(fh)
        if not isinstance(series, list):
            raise TypeError(
                f"Expected list[Attribution] at {path}, got {type(series).__name__}. "
                f"Delete and re-run the precompute script."
            )
        if step < 0 or step >= len(series):
            raise IndexError(
                f"Step {step} out of range for cached series of length {len(series)} "
                f"at {path}."
            )
        return series[step]

    # Tier 2 — dev fallback
    if explainable_model is not None and artifact.has_raw_observations:
        import posthoc_xai as xai
        if method not in xai.METHOD_REGISTRY:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Available: {list(xai.METHOD_REGISTRY.keys())}"
            )
        obs_t = artifact.raw_observations[step]
        return xai.METHOD_REGISTRY[method](explainable_model)(obs_t)

    # Tier 3 — not ready
    raise XAINotReadyError(
        f"No precomputed attribution for method='{method}' on model="
        f"'{artifact.model_key}', scenario={artifact.scenario_idx}.\n"
        f"Run the offline precompute script to generate the cache, or "
        f"enable dev mode by passing explainable_model=."
    )


def load_attribution_series(
    model_key: str,
    scenario_idx: int,
    method: str,
) -> Optional[list]:
    """Load entire cached attribution series (list[Attribution], length T).

    Returns None when no cache exists.  Intended for tab-level session_state
    priming so the tab can index any timestep without repeated disk reads.
    """
    path = _attr_path(model_key, scenario_idx, method)
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_attention_series(
    model_key: str,
    scenario_idx: int,
) -> Optional[list]:
    """Load entire cached attention series (list[dict], length T).

    Returns None when no cache exists.
    """
    path = _attention_path(model_key, scenario_idx)
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def get_explanations_all_methods(
    artifact: PlatformScenarioArtifact,
    step: int,
    methods: Optional[list[str]] = None,
    explainable_model=None,
) -> dict:
    """Return Attributions for every requested method at ``step``.

    When ``methods`` is None, returns all methods that have a cached
    series on disk for this (model, scenario).  Methods whose cache is
    missing and cannot be computed (no dev fallback) are silently
    omitted — the caller receives whichever subset is available.
    """
    if methods is None:
        methods = list_cached_methods(artifact.model_key, artifact.scenario_idx)

    results: dict = {}
    for m in methods:
        try:
            results[m] = get_explanation(artifact, step, m, explainable_model)
        except XAINotReadyError:
            pass
    return results


# ---------------------------------------------------------------------------
# Attention access (Perceiver-family only)
# ---------------------------------------------------------------------------

def get_attention(
    artifact: PlatformScenarioArtifact,
    step: int,
    explainable_model=None,
) -> Optional[dict]:
    """Return attention weights at ``step`` as a dict.

    Keys: 'cross_attention', 'self_attention' (arrays).

    Returns None when the model has no attention (non-Perceiver).
    Same cache/fallback/error semantics as get_explanation.
    """
    # Tier 1 — cached series
    path = _attention_path(artifact.model_key, artifact.scenario_idx)
    if path.exists():
        with open(path, "rb") as fh:
            series = pickle.load(fh)
        if step < 0 or step >= len(series):
            raise IndexError(
                f"Step {step} out of range for cached attention series "
                f"of length {len(series)} at {path}."
            )
        return series[step]

    # Tier 2 — dev fallback
    if explainable_model is not None:
        if not explainable_model.has_attention:
            return None
        if not artifact.has_raw_observations:
            raise XAINotReadyError(
                "Dev attention extraction requires raw_observations in "
                "the artifact."
            )
        obs_t = artifact.raw_observations[step]
        return explainable_model.get_attention(obs_t)

    # Tier 3 — not ready
    raise XAINotReadyError(
        f"No precomputed attention for model='{artifact.model_key}', "
        f"scenario={artifact.scenario_idx}.\n"
        f"Run the offline precompute script with attention capture, or "
        f"pass explainable_model= to enable dev mode."
    )


# ---------------------------------------------------------------------------
# Offline precompute API (called by future curation scripts)
# ---------------------------------------------------------------------------

def precompute_attribution_series(
    artifact: PlatformScenarioArtifact,
    method: str,
    explainable_model,
    steps: Optional[list[int]] = None,
    overwrite: bool = False,
) -> Path:
    """Compute and cache attributions for every timestep of a scenario.

    Writes a list[Attribution] of length ``artifact.num_steps`` to
    ``platform_cache/{slug}/scenario_{idx:04d}_attr_{method}.pkl``.

    Parameters
    ----------
    artifact          : PlatformScenarioArtifact — MUST have raw_observations.
    method            : posthoc_xai METHOD_REGISTRY key.
    explainable_model : A loaded ExplainableModel instance.
    steps             : Subset of timesteps to compute.  None → all steps.
                        When a subset is given, any pre-existing entries for
                        steps outside the subset are preserved (if the file
                        already exists and ``overwrite`` is False).
    overwrite         : When True, start from an empty list even if a cached
                        series already exists.

    Returns the Path where the series was written.
    """
    if not artifact.has_raw_observations:
        raise XAINotReadyError(
            "Cannot precompute attributions: artifact has no raw_observations."
        )

    import posthoc_xai as xai
    if method not in xai.METHOD_REGISTRY:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Available: {list(xai.METHOD_REGISTRY.keys())}"
        )

    T = artifact.num_steps
    path = _attr_path(artifact.model_key, artifact.scenario_idx, method)

    # Seed the series list: either from existing cache or fresh.
    if path.exists() and not overwrite:
        with open(path, "rb") as fh:
            series: list = pickle.load(fh)
        if len(series) != T:
            series = [None] * T
    else:
        series = [None] * T

    todo = list(range(T)) if steps is None else list(steps)
    method_fn = xai.METHOD_REGISTRY[method](explainable_model)
    for t in todo:
        obs_t = artifact.raw_observations[t]
        series[t] = method_fn(obs_t)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(series, fh)
    return path


def precompute_attention_series(
    artifact: PlatformScenarioArtifact,
    explainable_model,
    steps: Optional[list[int]] = None,
    overwrite: bool = False,
) -> Optional[Path]:
    """Compute and cache attention weights for every timestep.

    Returns None if the model does not support attention; otherwise returns
    the Path where the series was written.
    """
    if not explainable_model.has_attention:
        return None
    if not artifact.has_raw_observations:
        raise XAINotReadyError(
            "Cannot precompute attention: artifact has no raw_observations."
        )

    T = artifact.num_steps
    path = _attention_path(artifact.model_key, artifact.scenario_idx)

    if path.exists() and not overwrite:
        with open(path, "rb") as fh:
            series: list = pickle.load(fh)
        if len(series) != T:
            series = [None] * T
    else:
        series = [None] * T

    todo = list(range(T)) if steps is None else list(steps)
    for t in todo:
        obs_t = artifact.raw_observations[t]
        series[t] = explainable_model.get_attention(obs_t)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(series, fh)
    return path
