"""
Standalone Streamlit BEV Visualizer App.

Run with:
    streamlit run bev_visualizer/streamlit_app.py

Importable component
---------------------
To embed this visualizer in a larger platform, import the rendering
primitives directly:

    from bev_visualizer import MODEL_REGISTRY, run_rollout, render_frame, render_episode
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# ── Ensure project root + V-Max are on sys.path ──────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "V-Max") not in sys.path:
    sys.path.insert(0, str(_ROOT / "V-Max"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import time
import numpy as np
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="BEV Scenario Visualizer",
    page_icon="🚗",
    layout="wide",
)

from bev_visualizer.rollout_engine import ScenarioData
from bev_visualizer.bev_renderer import render_episode
import pickle

_ROOT = Path(__file__).resolve().parent.parent
CURATED_DIR = _ROOT / "curated_scenarios"

def get_available_models():
    """Scan curated_scenarios for model directories."""
    if not CURATED_DIR.exists(): return []
    return [d.name for d in CURATED_DIR.iterdir() if d.is_dir()]

def get_scenarios_for_model(model_name):
    """Scan a model directory for scenario pickle files."""
    p = CURATED_DIR / model_name
    if not p.exists(): return []
    files = [f.name for f in p.glob("scenario_*_cache.pkl")]
    return sorted(files)


# ── Session state helpers ─────────────────────────────────────────────────────

def _reset_rollout():
    """Clear any cached rollout so a fresh one is triggered."""
    for k in ["scenario_data", "frames"]:
        if k in st.session_state:
            del st.session_state[k]


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🚗 BEV Visualizer")
    st.markdown("---")

    available_models = get_available_models()
    if not available_models:
        st.warning("No pre-computed models found in `curated_scenarios/`.")
        model_name = None
    else:
        model_name = st.selectbox(
            "🤖 Cached Model",
            options=available_models,
            on_change=_reset_rollout,
        )

    available_scenarios = []
    if model_name:
        available_scenarios = get_scenarios_for_model(model_name)
    
    if not available_scenarios and model_name:
        st.warning("No cached scenarios found for this model.")
        scen_file = None
    else:
        scen_file = st.selectbox(
            "📂 Cached Scenario",
            options=available_scenarios,
            on_change=_reset_rollout,
        )

    st.markdown("---")
    animate = st.checkbox("▶ Animate episode", value=True)
    fps = st.slider("FPS", min_value=1, max_value=20, value=8)

    load_btn = st.button("🚀 Load & Render Scenario", type="primary", use_container_width=True, disabled=scen_file is None)


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("🗺 Bird's-Eye View — Closed-Loop Scenario Viewer")
st.markdown(
    "Select a model and scenario from the sidebar, then click **Run Rollout**."
)

# Trigger rollout
if load_btn and model_name and scen_file:
    _reset_rollout()
    file_path = CURATED_DIR / model_name / scen_file
    with st.spinner(f"Loading '{scen_file}' off disk..."):
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            st.session_state["scenario_data"] = data
        except Exception as exc:
            st.error(f"Failed to load cached scenario: {exc}")
            st.stop()

    with st.spinner("Rendering BEV frames from cached state..."):
        frames = render_episode(data.frame_states)
        st.session_state["frames"] = frames

    st.success(f"✅ Loaded {len(data.frame_states)} frames instantly.")

# Display results if we have them
if "scenario_data" in st.session_state:
    data = st.session_state["scenario_data"]
    frames = st.session_state["frames"]

    col_bev, col_stats = st.columns([3, 1])

    with col_bev:
        st.subheader("📺 BEV Scene")
        bev_placeholder = st.empty()

        if animate and len(frames) > 1:
            # Animate
            for i, img in enumerate(frames):
                bev_placeholder.image(img, caption=f"Step {i + 1} / {len(frames)}", use_container_width=True)
                time.sleep(1.0 / fps)
            # After animation, freeze on last frame
            bev_placeholder.image(frames[-1], caption=f"Final frame ({len(frames)} steps)", use_container_width=True)
        else:
            # Static: show last frame
            bev_placeholder.image(frames[-1], caption=f"Final frame ({len(frames)} steps)", use_container_width=True)

    with col_stats:
        st.subheader("📊 Episode Info")
        rewards_1d = data.rewards.flatten()   # (T,)
        dones_1d   = data.dones.flatten()     # (T,)

        total_reward = float(rewards_1d.sum())
        n_frames     = len(data.frame_states)

        # With jax.lax.scan, done[-1] is always True (natural end at step 80).
        # Early stop = done became True BEFORE the last step.
        first_done = int(dones_1d.argmax()) if dones_1d.max() > 0.5 else n_frames
        if first_done < n_frames - 1:
            status_str = f"❌ Early stop @ step {first_done + 1}"
        else:
            status_str = "✅ Completed"

        st.metric("Total Reward", f"{total_reward:.2f}")
        st.metric("Steps", n_frames)
        st.metric("Status", status_str)
        st.metric("Model", data.model_key.split("(")[0].strip())
        st.metric("Scenario #", data.scenario_idx)

        st.markdown("---")
        st.subheader("📈 Reward / step")
        st.line_chart(rewards_1d, height=150)

        st.subheader("📉 Cumulative Reward")
        st.line_chart(rewards_1d.cumsum(), height=150)
