"""Scenario store — artifact discovery, loading, and saving.

Cache lookup order for a given (model_key, scenario_idx):

1. ``platform_cache/{model_slug}/scenario_{idx:04d}_artifact.pkl``
   Full PlatformScenarioArtifact (ScenarioData + raw_observations + metadata).
   Written by future curation scripts.

2. ``cbm/curated_scenarios/{legacy_cache_slug}/scenario_{idx:04d}_cache.pkl``
   Legacy ScenarioData pickle from the old bev_visualizer curation script.
   Wrapped into a PlatformScenarioArtifact with raw_observations=None.

3. Neither exists → returns None (caller decides how to handle).

All public functions are pure Python; no Streamlit imports here.
The scenario store deliberately does NOT import bev_visualizer at module
level to keep it light.  Imports happen inside load functions.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

from platform.shared.contracts import PlatformScenarioArtifact
from platform.shared.model_catalog import PLATFORM_MODELS, ModelEntry

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CBM_ROOT = _PROJECT_ROOT / "cbm"
_LEGACY_CACHE_ROOT = _CBM_ROOT / "curated_scenarios"
_PLATFORM_CACHE_ROOT = _PROJECT_ROOT / "platform_cache"


# ---------------------------------------------------------------------------
# Internal path helpers
# ---------------------------------------------------------------------------

def _artifact_path(model_key: str, scenario_idx: int) -> Path:
    entry = PLATFORM_MODELS[model_key]
    return _PLATFORM_CACHE_ROOT / entry.cache_slug / f"scenario_{scenario_idx:04d}_artifact.pkl"


def _legacy_path(model_key: str, scenario_idx: int) -> Optional[Path]:
    entry = PLATFORM_MODELS.get(model_key)
    if entry is None or entry.legacy_cache_slug is None:
        return None
    return _LEGACY_CACHE_ROOT / entry.legacy_cache_slug / f"scenario_{scenario_idx:04d}_cache.pkl"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def get_available_scenarios(model_key: str) -> list[int]:
    """Return sorted list of cached scenario indices for a given model key.

    Checks both the platform artifact cache and the legacy ScenarioData cache.
    """
    indices: set[int] = set()

    entry = PLATFORM_MODELS.get(model_key)
    if entry is None:
        return []

    # Platform artifact cache
    artifact_dir = _PLATFORM_CACHE_ROOT / entry.cache_slug
    if artifact_dir.exists():
        for f in artifact_dir.glob("scenario_*_artifact.pkl"):
            try:
                indices.add(int(f.stem.split("_")[1]))
            except (IndexError, ValueError):
                pass

    # Legacy cache
    if entry.legacy_cache_slug:
        legacy_dir = _LEGACY_CACHE_ROOT / entry.legacy_cache_slug
        if legacy_dir.exists():
            for f in legacy_dir.glob("scenario_*_cache.pkl"):
                try:
                    indices.add(int(f.stem.split("_")[1]))
                except (IndexError, ValueError):
                    pass

    return sorted(indices)


def list_cached_models() -> list[str]:
    """Return model keys that have at least one cached scenario available."""
    return [k for k in PLATFORM_MODELS if get_available_scenarios(k)]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_artifact(
    model_key: str,
    scenario_idx: int,
) -> Optional[PlatformScenarioArtifact]:
    """Load a PlatformScenarioArtifact from cache.

    Returns None if nothing is cached for this (model_key, scenario_idx).
    """
    # 1 — Try full platform artifact (has raw_observations)
    path = _artifact_path(model_key, scenario_idx)
    if path.exists():
        with open(path, "rb") as fh:
            artifact = pickle.load(fh)
        if not isinstance(artifact, PlatformScenarioArtifact):
            raise TypeError(
                f"Expected PlatformScenarioArtifact at {path}, "
                f"got {type(artifact).__name__}"
            )
        return artifact

    # 2 — Try legacy ScenarioData pickle
    legacy_path = _legacy_path(model_key, scenario_idx)
    if legacy_path is not None and legacy_path.exists():
        with open(legacy_path, "rb") as fh:
            scenario_data = pickle.load(fh)
        return PlatformScenarioArtifact(
            scenario_data=scenario_data,
            model_key=model_key,
            scenario_idx=scenario_idx,
            raw_observations=None,
            metadata={"source": "legacy_cache", "legacy_path": str(legacy_path)},
        )

    return None


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_artifact(artifact: PlatformScenarioArtifact) -> Path:
    """Persist a PlatformScenarioArtifact to the platform cache.

    Creates the parent directory if it does not exist.
    Returns the path it was saved to.
    """
    path = _artifact_path(artifact.model_key, artifact.scenario_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(artifact, fh)
    return path
