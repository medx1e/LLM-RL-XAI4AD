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

import base64
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image

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

def _frame_to_base64(frame: np.ndarray) -> str:
    buf = BytesIO()
    Image.fromarray(frame).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def render_bev_player(
    artifact: PlatformScenarioArtifact,
    key_prefix: str = "bev",
    height: Optional[int] = None,
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

    # 3 — On-demand render with progress bar (skeleton shown while loading)
    if frames is None:
        try:
            from platform.shared.html_components import skeleton
            st.markdown(skeleton("100%", "220px"), unsafe_allow_html=True)
        except ImportError:
            pass
        pb = st.progress(0, text="Rendering BEV frames (first time)…")
        frames = _render_all_frames(artifact, progress_bar=pb)
        pb.empty()
        _set_session_frames(artifact, key_prefix, frames)

    slider_key  = f"{key_prefix}__slider__{artifact.model_key}__{artifact.scenario_idx}"
    playing_key = f"{key_prefix}__playing__{artifact.model_key}__{artifact.scenario_idx}"
    btn_key     = f"{key_prefix}__playbtn__{artifact.model_key}__{artifact.scenario_idx}"

    # Initialise session state keys once so the slider never receives both
    # value= and a pre-existing session state value (causes a Streamlit warning).
    if slider_key not in st.session_state:
        st.session_state[slider_key] = 0
    if playing_key not in st.session_state:
        st.session_state[playing_key] = False

    is_playing = st.session_state[playing_key]

    # Advance slider BEFORE rendering it so Streamlit sees the updated value.
    if is_playing:
        next_step = st.session_state[slider_key] + 1
        if next_step >= num_steps:
            st.session_state[playing_key] = False
        else:
            st.session_state[slider_key] = next_step

    # ── Styled BEV player header ──────────────────────────────────────────────
    step_val = st.session_state.get(slider_key, 0)
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:8px;"
        f"padding:8px 12px;background:rgba(255,255,255,0.02);"
        f"border:1px solid #3b3f55;border-radius:10px 10px 0 0;"
        f"border-bottom:none;'>"
        f"<span style='display:inline-flex;align-items:center;gap:6px;"
        f"padding:2px 8px;font-size:11px;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:0.05em;border-radius:999px;border:1px solid #3b3f55;"
        f"background:#34384c;color:#f4f4f7;'>BEV</span>"
        f"<span style='font-family:\"JetBrains Mono\",monospace;font-size:11px;"
        f"color:#a7a8b3;'>t = {step_val}</span>"
        f"<span style='margin-left:auto;font-family:\"JetBrains Mono\",monospace;"
        f"font-size:10px;color:#3b3f55;'>{num_steps} steps</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Transport controls ────────────────────────────────────────────────────
    col_slider, col_btn = st.columns([5, 1])
    with col_slider:
        step = st.slider(
            "Timestep",
            min_value=0,
            max_value=num_steps - 1,
            key=slider_key,
            label_visibility="collapsed",
        )
    with col_btn:
        btn_label = "⏸" if is_playing else "▶"
        if st.button(btn_label, key=btn_key, width='stretch'):
            st.session_state[playing_key] = not is_playing
            st.rerun()

    # ── BEV frame ─────────────────────────────────────────────────────────────
    if height is not None:
        b64 = _frame_to_base64(frames[step])
        st.markdown(
            f'<div style="max-height:{height}px;overflow:hidden;line-height:0;">'
            f'<img src="data:image/png;base64,{b64}" '
            f'style="width:100%;max-height:{height}px;object-fit:contain;" />'
            f'</div>'
            f'<div style="font-size:11px;color:#6b7280;margin-top:2px;">'
            f'Step {step + 1} / {num_steps}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.image(
            frames[step],
            caption=f"Step {step + 1} / {num_steps}",
            width='stretch',
        )

    # Signal that a rerun is needed — callers must call schedule_bev_rerun()
    # at the END of their render() so all content (narration, charts…) is drawn
    # before Streamlit restarts the script.
    if is_playing:
        st.session_state["__bev_needs_rerun__"] = True
    else:
        st.session_state.pop("__bev_needs_rerun__", None)

    return step


def is_bev_playing(artifact: PlatformScenarioArtifact, key_prefix: str = "bev") -> bool:
    """True while the BEV player for this artifact is auto-advancing.

    Reads the same session-state flag ``render_bev_player`` writes, so callers
    can react to playback (e.g. only stream narrations while the episode plays).
    """
    playing_key = f"{key_prefix}__playing__{artifact.model_key}__{artifact.scenario_idx}"
    return bool(st.session_state.get(playing_key, False))


def schedule_bev_rerun(delay: float = 0.12) -> None:
    """Trigger a rerun for BEV playback.

    Call this at the very end of any tab that embeds render_bev_player().
    Placing it last ensures all page content (narration cards, charts, etc.)
    is fully rendered before Streamlit restarts the script for the next frame.
    """
    if st.session_state.pop("__bev_needs_rerun__", False):
        time.sleep(delay)
        st.rerun()


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
    st.image(img, caption=caption or f"Step {step + 1}", width='stretch')


def clear_bev_cache(artifact: PlatformScenarioArtifact, key_prefix: str = "bev") -> None:
    """Evict session-state frame cache for a given artifact."""
    key = _frame_cache_key(artifact, key_prefix)
    if key in st.session_state:
        del st.session_state[key]
