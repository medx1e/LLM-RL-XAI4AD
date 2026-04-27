"""Reusable Streamlit BEV rendering component.

``render_bev_player(artifact, key_prefix)``
    Full episode player with timestep slider.
    Frame lookup order:
      1. st.session_state (within-session cache — instant)
      2. platform_cache/{slug}/scenario_*_frames.pkl (pre-rendered by precompute script)
      3. Render on demand with progress bar (fallback, ~40–60 s for 80 frames)
    Returns the currently selected step index.

``render_bev_frame(artifact, step, overlay_fn)``
    Single annotated frame — always re-renders so overlay_fn is applied fresh.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st

import platform  # triggers path setup
from platform.shared.contracts import OverlayFn, PlatformScenarioArtifact

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PLATFORM_CACHE_ROOT = _PROJECT_ROOT / "platform_cache"


# ---------------------------------------------------------------------------
# Session-state cache helpers
# ---------------------------------------------------------------------------

def _frame_cache_key(artifact: PlatformScenarioArtifact, prefix: str) -> str:
    return f"bev_frames__{prefix}__{artifact.model_key}__{artifact.scenario_idx}"


def _get_session_frames(artifact, prefix) -> Optional[list[np.ndarray]]:
    return st.session_state.get(_frame_cache_key(artifact, prefix))


def _set_session_frames(artifact, prefix, frames: list[np.ndarray]) -> None:
    st.session_state[_frame_cache_key(artifact, prefix)] = frames


# ---------------------------------------------------------------------------
# Pre-rendered frame loader (from platform_cache)
# ---------------------------------------------------------------------------

def _prerendered_frames_path(artifact: PlatformScenarioArtifact) -> Optional[Path]:
    from platform.shared.model_catalog import PLATFORM_MODELS
    entry = PLATFORM_MODELS.get(artifact.model_key)
    if entry is None:
        return None
    return (
        _PLATFORM_CACHE_ROOT
        / entry.cache_slug
        / f"scenario_{artifact.scenario_idx:04d}_frames.pkl"
    )


def _load_prerendered_frames(artifact: PlatformScenarioArtifact) -> Optional[list[np.ndarray]]:
    path = _prerendered_frames_path(artifact)
    if path is None or not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# On-demand renderer (fallback)
# ---------------------------------------------------------------------------

def _render_all_frames(
    artifact: PlatformScenarioArtifact,
    progress_bar=None,
) -> list[np.ndarray]:
    from bev_visualizer.bev_renderer import render_frame

    states = artifact.scenario_data.frame_states
    total = len(states)
    frames = []
    for step, state in enumerate(states):
        frames.append(render_frame(state, overlay_fn=None, step=step))
        if progress_bar is not None:
            progress_bar.progress(
                (step + 1) / total,
                text=f"Rendering frame {step + 1}/{total}…",
            )
    return frames


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_bev_player(
    artifact: PlatformScenarioArtifact,
    key_prefix: str = "bev",
) -> int:
    """Interactive BEV episode player with timestep slider.

    Returns the currently selected zero-indexed timestep.
    """
    num_steps = artifact.num_steps

    # 1 — Session-state cache
    frames = _get_session_frames(artifact, key_prefix)

    # 2 — Pre-rendered frames from disk
    if frames is None:
        frames = _load_prerendered_frames(artifact)
        if frames is not None:
            _set_session_frames(artifact, key_prefix, frames)

    # 3 — On-demand render with progress bar
    if frames is None:
        pb = st.progress(0, text="Rendering BEV frames (first time)…")
        frames = _render_all_frames(artifact, progress_bar=pb)
        pb.empty()
        _set_session_frames(artifact, key_prefix, frames)

    slider_key = f"{key_prefix}__slider__{artifact.model_key}__{artifact.scenario_idx}"
    playing_key = f"{key_prefix}__playing__{artifact.model_key}__{artifact.scenario_idx}"
    btn_key = f"{key_prefix}__playbtn__{artifact.model_key}__{artifact.scenario_idx}"

    is_playing = st.session_state.get(playing_key, False)

    # Advance slider BEFORE rendering it so Streamlit sees the updated value
    if is_playing:
        current = st.session_state.get(slider_key, 0)
        next_step = current + 1
        if next_step >= num_steps:
            st.session_state[playing_key] = False
        else:
            st.session_state[slider_key] = next_step

    col_slider, col_btn = st.columns([5, 1])
    with col_slider:
        step = st.slider(
            "Timestep",
            min_value=0,
            max_value=num_steps - 1,
            value=0,
            key=slider_key,
        )
    with col_btn:
        st.write("")  # vertical alignment spacer
        btn_label = "⏸" if is_playing else "▶"
        if st.button(btn_label, key=btn_key):
            st.session_state[playing_key] = not is_playing
            st.rerun()

    st.image(
        frames[step],
        caption=f"Step {step + 1} / {num_steps}",
        use_container_width=True,
    )

    if is_playing:
        time.sleep(0.12)
        st.rerun()

    return step


def render_bev_frame(
    artifact: PlatformScenarioArtifact,
    step: int,
    overlay_fn: Optional[OverlayFn] = None,
    caption: str = "",
) -> None:
    """Render a single BEV frame with optional overlay. Never cached."""
    from bev_visualizer.bev_renderer import render_frame

    state = artifact.scenario_data.frame_states[step]
    img = render_frame(state, overlay_fn=overlay_fn, step=step)
    st.image(img, caption=caption or f"Step {step + 1}", use_container_width=True)


def clear_bev_cache(artifact: PlatformScenarioArtifact, key_prefix: str = "bev") -> None:
    """Evict session-state frame cache for a given artifact."""
    key = _frame_cache_key(artifact, key_prefix)
    if key in st.session_state:
        del st.session_state[key]
