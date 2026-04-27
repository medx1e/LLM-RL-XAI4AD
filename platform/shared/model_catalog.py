"""Platform model catalog — working models only.

BROKEN models (sac_seed0/42/69) are intentionally excluded because they
require a 'speed_limit' roadgraph feature not present in Waymax (Issue 5
in GUIDE2LOAD_MODELS.md).

This catalog is the single source of truth for which models the platform
exposes.  Tabs should always query this catalog rather than constructing
model paths themselves.

Adding a new model: add one entry to PLATFORM_MODELS below.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CBM_ROOT = _PROJECT_ROOT / "cbm"
_RUNS_RLC = _CBM_ROOT / "runs_rlc"
_CBM_SCRATCH_DIR = _CBM_ROOT / "cbm_scratch_v2_lambda05"


def _p(rel: str) -> str:
    return str(_RUNS_RLC / rel)


def _slug(key: str) -> str:
    """Filesystem-safe slug from a model key (spaces/punctuation → underscores)."""
    return re.sub(r"[^\w]+", "_", key).strip("_")


# ---------------------------------------------------------------------------
# ModelEntry
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ModelEntry:
    """Metadata for one model in the platform catalog.

    Attributes
    ----------
    key
        Human-readable display name (matches the PLATFORM_MODELS dict key).
    model_dir
        Absolute path to the model's run directory inside ``runs_rlc/``.
    encoder_family
        Encoder architecture name: perceiver | mtr | wayformer | mgail | none.
    has_attention
        True when the encoder exposes extractable attention weights via
        ``PerceiverWrapper.get_attention()``.  Perceiver and MGAIL qualify.
    description
        One-line description for UI tooltips.
    is_primary
        True for models highlighted as the main demo targets.
    legacy_cache_slug
        Directory name under ``cbm/curated_scenarios/`` where legacy
        ScenarioData pickles were saved by the old curation script.
        ``None`` when no legacy cache exists for this model.
    """

    key: str
    model_dir: str
    encoder_family: str
    has_attention: bool
    description: str
    is_primary: bool = False
    legacy_cache_slug: Optional[str] = None

    @property
    def cache_slug(self) -> str:
        """Filesystem-safe slug used for the platform_cache directory."""
        return _slug(self.key)

    @property
    def exists_on_disk(self) -> bool:
        return Path(self.model_dir).is_dir()


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

PLATFORM_MODELS: dict[str, ModelEntry] = {
    # ── Perceiver / LQ — minimal reward (primary demo targets) ────────────
    "SAC Perceiver — WOMD seed 42": ModelEntry(
        key="SAC Perceiver — WOMD seed 42",
        model_dir=_p("womd_sac_road_perceiver_minimal_42"),
        encoder_family="perceiver",
        has_attention=True,
        description="Perceiver/LQ encoder · minimal reward · seed 42 · 97.47% accuracy",
        is_primary=True,
        legacy_cache_slug="SAC_Minimal_(WOMD_seed_42)",
    ),
    "SAC Perceiver — WOMD seed 69": ModelEntry(
        key="SAC Perceiver — WOMD seed 69",
        model_dir=_p("womd_sac_road_perceiver_minimal_69"),
        encoder_family="perceiver",
        has_attention=True,
        description="Perceiver/LQ encoder · minimal reward · seed 69 · 97.44% accuracy",
        is_primary=True,
    ),
    "SAC Perceiver — WOMD seed 99": ModelEntry(
        key="SAC Perceiver — WOMD seed 99",
        model_dir=_p("womd_sac_road_perceiver_minimal_99"),
        encoder_family="perceiver",
        has_attention=True,
        description="Perceiver/LQ encoder · minimal reward · seed 99 · 96.87% accuracy",
        is_primary=True,
    ),
    # ── Perceiver / LQ — basic reward ──────────────────────────────────────
    "SAC Perceiver Basic — WOMD seed 42": ModelEntry(
        key="SAC Perceiver Basic — WOMD seed 42",
        model_dir=_p("womd_sac_road_perceiver_basic_42"),
        encoder_family="perceiver",
        has_attention=True,
        description="Perceiver/LQ encoder · basic reward · seed 42 · 97.20% accuracy",
    ),
    "SAC Perceiver Basic — WOMD seed 69": ModelEntry(
        key="SAC Perceiver Basic — WOMD seed 69",
        model_dir=_p("womd_sac_road_perceiver_basic_69"),
        encoder_family="perceiver",
        has_attention=True,
        description="Perceiver/LQ encoder · basic reward · seed 69 · 97.06% accuracy",
    ),
    "SAC Perceiver Basic — WOMD seed 99": ModelEntry(
        key="SAC Perceiver Basic — WOMD seed 99",
        model_dir=_p("womd_sac_road_perceiver_basic_99"),
        encoder_family="perceiver",
        has_attention=True,
        description="Perceiver/LQ encoder · basic reward · seed 99",
    ),
    # ── Perceiver / LQ — complete reward ────────────────────────────────────
    "SAC Perceiver Complete — WOMD seed 42": ModelEntry(
        key="SAC Perceiver Complete — WOMD seed 42",
        model_dir=_p("womd_sac_road_perceiver_complete_42"),
        encoder_family="perceiver",
        has_attention=True,
        description="Perceiver/LQ encoder · complete reward · seed 42",
        is_primary=True,
        legacy_cache_slug="SAC_Complete_(WOMD_seed_42)",
    ),
    "SAC Perceiver Complete — WOMD seed 69": ModelEntry(
        key="SAC Perceiver Complete — WOMD seed 69",
        model_dir=_p("womd_sac_road_perceiver_complete_69"),
        encoder_family="perceiver",
        has_attention=True,
        description="Perceiver/LQ encoder · complete reward · seed 69",
    ),
    "SAC Perceiver Complete — WOMD seed 99": ModelEntry(
        key="SAC Perceiver Complete — WOMD seed 99",
        model_dir=_p("womd_sac_road_perceiver_complete_99"),
        encoder_family="perceiver",
        has_attention=True,
        description="Perceiver/LQ encoder · complete reward · seed 99",
    ),
    # ── MGAIL ───────────────────────────────────────────────────────────────
    "SAC MGAIL — WOMD seed 42": ModelEntry(
        key="SAC MGAIL — WOMD seed 42",
        model_dir=_p("womd_sac_road_mgail_minimal_42"),
        encoder_family="mgail",
        has_attention=True,
        description="MGAIL encoder · minimal reward · seed 42 · 96.66% accuracy",
    ),
    "SAC MGAIL — WOMD seed 69": ModelEntry(
        key="SAC MGAIL — WOMD seed 69",
        model_dir=_p("womd_sac_road_mgail_minimal_69"),
        encoder_family="mgail",
        has_attention=True,
        description="MGAIL encoder · minimal reward · seed 69",
    ),
    "SAC MGAIL — WOMD seed 99": ModelEntry(
        key="SAC MGAIL — WOMD seed 99",
        model_dir=_p("womd_sac_road_mgail_minimal_99"),
        encoder_family="mgail",
        has_attention=True,
        description="MGAIL encoder · minimal reward · seed 99",
    ),
    # ── MTR ─────────────────────────────────────────────────────────────────
    "SAC MTR — WOMD seed 42": ModelEntry(
        key="SAC MTR — WOMD seed 42",
        model_dir=_p("womd_sac_road_mtr_minimal_42"),
        encoder_family="mtr",
        has_attention=False,
        description="MTR encoder · minimal reward · seed 42 · ~96% accuracy",
    ),
    "SAC MTR — WOMD seed 69": ModelEntry(
        key="SAC MTR — WOMD seed 69",
        model_dir=_p("womd_sac_road_mtr_minimal_69"),
        encoder_family="mtr",
        has_attention=False,
        description="MTR encoder · minimal reward · seed 69",
    ),
    "SAC MTR — WOMD seed 99": ModelEntry(
        key="SAC MTR — WOMD seed 99",
        model_dir=_p("womd_sac_road_mtr_minimal_99"),
        encoder_family="mtr",
        has_attention=False,
        description="MTR encoder · minimal reward · seed 99",
    ),
    # ── CBM — Concept Bottleneck Models ─────────────────────────────────
    "CBM Scratch V2 — λ=0.5": ModelEntry(
        key="CBM Scratch V2 — λ=0.5",
        model_dir=str(_CBM_SCRATCH_DIR),
        encoder_family="perceiver",
        has_attention=False,
        description="CBM Scratch V2 · 15 concepts (phases 1+2+3) · λ=0.5 · 10GB val · 87.6% route progress",
        is_primary=True,
    ),
    # ── Wayformer ────────────────────────────────────────────────────────────
    "SAC Wayformer — WOMD seed 42": ModelEntry(
        key="SAC Wayformer — WOMD seed 42",
        model_dir=_p("womd_sac_road_wayformer_minimal_42"),
        encoder_family="wayformer",
        has_attention=False,
        description="Wayformer encoder · minimal reward · seed 42 · ~96% accuracy",
    ),
    "SAC Wayformer — WOMD seed 69": ModelEntry(
        key="SAC Wayformer — WOMD seed 69",
        model_dir=_p("womd_sac_road_wayformer_minimal_69"),
        encoder_family="wayformer",
        has_attention=False,
        description="Wayformer encoder · minimal reward · seed 69",
    ),
    "SAC Wayformer — WOMD seed 99": ModelEntry(
        key="SAC Wayformer — WOMD seed 99",
        model_dir=_p("womd_sac_road_wayformer_minimal_99"),
        encoder_family="wayformer",
        has_attention=False,
        description="Wayformer encoder · minimal reward · seed 99",
    ),
}

# Drop catalog entries whose run directory doesn't exist on disk.
PLATFORM_MODELS = {
    k: v for k, v in PLATFORM_MODELS.items() if v.exists_on_disk
}


# ---------------------------------------------------------------------------
# Convenience filters
# ---------------------------------------------------------------------------

def get_primary_models() -> dict[str, ModelEntry]:
    """Return the recommended demo models (Perceiver/LQ, minimal reward)."""
    return {k: v for k, v in PLATFORM_MODELS.items() if v.is_primary}


def get_attention_models() -> dict[str, ModelEntry]:
    """Return all models that support attention extraction."""
    return {k: v for k, v in PLATFORM_MODELS.items() if v.has_attention}


def get_models_by_encoder(encoder_family: str) -> dict[str, ModelEntry]:
    return {k: v for k, v in PLATFORM_MODELS.items() if v.encoder_family == encoder_family}
