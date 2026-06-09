"""CBM Explorer tab.

Layout:
    Sidebar  : archetype filter · scenario rank · concepts to show
    Left col : BEV episode player (same component as post-hoc tab)
               + episode stats (reward, outcome, route progress)
    Right col: Concept timeline (pred vs true, synchronized to BEV step)
               + Ego actions plot
               + Per-step concept value table
    Bottom   : Static analysis (Fig 1 concept quality, if figures exist)
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import streamlit as st

import platform  # path bootstrap  # noqa: F401
from platform.shared.bev_component import render_bev_player
from platform.shared.model_catalog import PLATFORM_MODELS
from platform.shared.scenario_store import get_available_scenarios, load_artifact
from platform.shared.html_components import context_strip, empty_state, chip
from platform.shared.charts import (
    concept_timeline_chart, concept_snapshot_chart,
)
from platform.shared.theme import CONCEPT_CATEGORIES, CONCEPT_CATEGORY_COLORS

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CBM_ROOT     = _PROJECT_ROOT / "cbm"
_FIGURES_DIR  = _CBM_ROOT / "figures"

CBM_MODEL_KEY = "CBM Scratch V2 — λ=0.5"

BINARY_CONCEPTS = {"traffic_light_red", "lead_vehicle_decelerating", "at_intersection"}

ARCHETYPE_META = {
    "red_light_stop": {
        "label":       "🔴 Red Light Stop",
        "description": "The CBM correctly identifies a red traffic light and brakes. "
                       "Watch `traffic_light_red` spike while ego decelerates.",
        "concepts":    ["traffic_light_red", "dist_to_traffic_light", "ego_speed"],
        "color":       "#E53935",
    },
    "ttc_success": {
        "label":       "⚠️ TTC Success",
        "description": "A lead vehicle slows. The CBM tracks TTC and brakes before impact.",
        "concepts":    ["ttc_lead_vehicle", "lead_vehicle_decelerating", "ego_speed"],
        "color":       "#FB8C00",
    },
    "curvature_nav": {
        "label":       "〰️ Curve Navigation",
        "description": "High path curvature detected via `path_curvature_max`. "
                       "The CBM adjusts steering accordingly.",
        "concepts":    ["path_curvature_max", "path_straightness", "heading_deviation"],
        "color":       "#43A047",
    },
    "concept_failure": {
        "label":       "❌ Concept Failure",
        "description": "The CBM misreads a key concept (pred ≠ true), "
                       "leading to a suboptimal or dangerous outcome.",
        "concepts":    ["traffic_light_red", "dist_nearest_object", "ttc_lead_vehicle"],
        "color":       "#8E24AA",
    },
}


# ── Data helpers ──────────────────────────────────────────────────────

@st.cache_data
def _load_index():
    p = _CBM_ROOT / "curated_scenarios.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_resource(show_spinner=False, max_entries=8)
def _load_artifact_cached(model_key: str, scenario_idx: int):
    """Cache the (large, ~120 MB) artifact across reruns.

    The CBM tab calls this on every playback rerun; without caching each frame
    would re-unpickle the full artifact (~3.7 s). cache_resource returns the
    same object with no per-call copy, so playback only pays the cost once per
    scenario. max_entries bounds memory across the curated scenarios.
    """
    return load_artifact(model_key, scenario_idx)



def _concept_plots(pred, true_, valid, cnames, selected, current_step, accent,
                   interesting_steps=None):
    """Return a matplotlib figure with one subplot per selected concept."""
    n = len(selected)
    if n == 0:
        return None

    fig, axes = plt.subplots(n, 1, figsize=(8, 2.0 * n), sharex=True)
    if n == 1:
        axes = [axes]

    steps = np.arange(80)

    for ax, cname in zip(axes, selected):
        if cname not in cnames:
            continue
        ci = cnames.index(cname)
        t_v = true_[:, ci]
        p_v = pred[:, ci]
        vm  = valid[:, ci]

        # Highlight interesting timesteps as shaded bands
        if interesting_steps:
            for t in interesting_steps:
                ax.axvspan(t - 0.5, t + 0.5, color=accent, alpha=0.18, zorder=0)

        # Grey invalid regions
        ax.fill_between(steps, 0, 1, where=~vm,
                        color="grey", alpha=0.13, zorder=0)
        ax.plot(steps, t_v, color="#1565C0", lw=1.8, label="Ground Truth")
        ax.plot(steps, p_v, color="#E65100", lw=1.4, ls="--", label="Predicted")
        ax.axvline(current_step, color=accent, lw=1.1, ls=":", alpha=0.9)

        # Metric annotation
        if vm.any():
            if cname in BINARY_CONCEPTS:
                acc = float(((p_v[vm] > 0.5) == (t_v[vm] > 0.5)).mean())
                ann = f"Acc={acc:.3f}"
            else:
                p_m, t_m = p_v[vm], t_v[vm]
                ss_res = np.sum((t_m - p_m) ** 2)
                ss_tot = np.sum((t_m - t_m.mean()) ** 2)
                r2  = (1 - ss_res / ss_tot) if ss_tot > 1e-10 else float("nan")
                ann = f"R²={r2:.3f}"
            ax.text(0.98, 0.92, ann, transform=ax.transAxes,
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none"))

        label = cname.replace("_", " ").title()
        if cname in BINARY_CONCEPTS:
            label += " (binary)"
        ax.set_title(label, fontsize=9, pad=2)
        ax.set_ylim(-0.05, 1.1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("[0, 1]", fontsize=7)

    axes[-1].set_xlabel("Timestep", fontsize=8)

    handles = [
        plt.Line2D([0], [0], color="#1565C0", lw=2, label="Ground Truth"),
        plt.Line2D([0], [0], color="#E65100", lw=2, ls="--", label="CBM Prediction"),
        plt.Rectangle((0, 0), 1, 1, fc="grey", alpha=0.3, label="Not valid"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=8,
               frameon=False, bbox_to_anchor=(1.0, 1.01))
    fig.tight_layout(rect=[0, 0, 0.88, 1], h_pad=2.5)
    return fig


def _action_plot(artifact, current_step, accent):
    """Acceleration + steering over episode."""
    # ego_actions stored in metadata by precompute script
    meta = artifact.metadata or {}
    raw_obs = artifact.raw_observations   # (T, obs_size) or None

    # Fall back to rewards as proxy if actions not available
    rewards = np.array(artifact.scenario_data.rewards)

    fig, axes = plt.subplots(1, 1, figsize=(8, 2.2))
    steps = np.arange(len(rewards))
    axes.plot(steps, rewards, color="#333", lw=1.2)
    axes.fill_between(steps, 0, rewards, where=rewards >= 0,
                      color="#43A047", alpha=0.4, label="Positive reward")
    axes.fill_between(steps, 0, rewards, where=rewards < 0,
                      color="#E53935", alpha=0.4, label="Negative reward")
    axes.axhline(0, color="grey", lw=0.6)
    axes.axvline(current_step, color=accent, lw=1.1, ls=":", alpha=0.9)
    axes.set_ylabel("Reward", fontsize=8)
    axes.set_xlabel("Timestep", fontsize=8)
    axes.set_title("Episode Reward", fontsize=9, pad=2)
    axes.legend(fontsize=7, frameon=False, loc="upper right")
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ── Playback fragment ─────────────────────────────────────────────────

def _render_cbm_player(
    artifact, local_idx, accent, selected_concepts,
    pred_c, true_c, valid_c, cnames, has_concepts, interesting_steps,
    auto_looping,
) -> None:
    """BEV player + step-synced charts, isolated in a fragment.

    While playing, the caller wraps this in ``st.fragment(run_every=…)`` so it
    reruns ONLY itself every ~60 ms — the rest of the page stays mounted and the
    keyed Plotly charts update in place instead of remounting (no blink).
    ``auto_looping`` is True when run_every is active.
    """
    col_bev, col_xai = st.columns([1, 1], gap="medium")

    with col_bev:
        st.markdown(
            "<h3 style='font-size:16px;font-weight:700;color:#f4f4f7;"
            "font-family:Inter,sans-serif;margin:0 0 10px;'>Episode Replay</h3>",
            unsafe_allow_html=True,
        )
        if artifact.notes:
            st.caption(artifact.notes)
        if interesting_steps:
            st.caption(
                f"★ Key event steps ({len(interesting_steps)}): "
                + ", ".join(str(t) for t in interesting_steps[:8])
                + ("…" if len(interesting_steps) > 8 else "")
            )
        current_step = render_bev_player(artifact, key_prefix="cbm")


    with col_xai:
        st.markdown(
            "<h3 style='font-size:16px;font-weight:700;color:#f4f4f7;"
            "font-family:Inter,sans-serif;margin:0 0 10px;'>Live Concept Snapshot</h3>",
            unsafe_allow_html=True,
        )
        if not has_concepts:
            st.markdown(
                empty_state("Concept data not in artifact",
                            command="python scripts/precompute_cbm_demo.py"),
                unsafe_allow_html=True,
            )
        else:
            fig_snap = concept_snapshot_chart(
                pred_c, valid_c, cnames, current_step,
                categories=CONCEPT_CATEGORIES,
                category_colors=CONCEPT_CATEGORY_COLORS,
                binary_concepts=BINARY_CONCEPTS,
            )
            # Stable key (NOT including current_step) so Streamlit reuses the
            # same chart component across playback reruns → in-place update,
            # no flash. Scenario idx in the key remounts on scenario switch.
            st.plotly_chart(
                fig_snap, width="stretch", config={"displayModeBar": False},
                key=f"cbm_snapshot_chart_{local_idx}",
            )

        if has_concepts:
            with st.expander("Concept timeline (full history — pred vs true)",
                             expanded=False):
                if not selected_concepts:
                    st.markdown(
                        empty_state("Select at least one concept on the left."),
                        unsafe_allow_html=True,
                    )
                else:
                    fig_c = concept_timeline_chart(
                        pred_c, true_c, valid_c, cnames,
                        selected_concepts, current_step, accent,
                        interesting_steps=interesting_steps,
                        binary_concepts=BINARY_CONCEPTS,
                    )
                    # Stable across playback steps; the selected-concept
                    # count is in the key so changing the selection (which
                    # changes the number of subplots) cleanly remounts.
                    st.plotly_chart(
                        fig_c, width="stretch",
                        config={"displayModeBar": False},
                        key=f"cbm_timeline_chart_{local_idx}_{len(selected_concepts)}",
                    )

            st.markdown(
                f"<div style='font-size:14px;font-weight:600;color:#f4f4f7;"
                f"font-family:Inter,sans-serif;margin:12px 0 4px;'>Selected concepts — pred vs true @ step {current_step}</div>",
                unsafe_allow_html=True,
            )
            rows = []
            for cn in (selected_concepts or cnames[:6]):
                if cn not in cnames:
                    continue
                ci = cnames.index(cn)
                vm = bool(valid_c[current_step, ci])
                rows.append({
                    "Concept": cn.replace("_", " "),
                    "True":    f"{true_c[current_step, ci]:.3f}" if vm else "—",
                    "Pred":    f"{pred_c[current_step, ci]:.3f}",
                    "Valid":   "✓" if vm else "✗",
                    "Error":   f"{abs(pred_c[current_step, ci] - true_c[current_step, ci]):.3f}" if vm else "—",
                })
            if rows:
                import pandas as pd
                st.dataframe(
                    pd.DataFrame(rows).set_index("Concept"),
                    width="stretch",
                    height=min(35 * len(rows) + 38, 280),
                )

    # ── Playback loop control ─────────────────────────────────────────
    # Advancing is driven by the fragment's run_every (set in render()), which
    # reruns ONLY this fragment. render_bev_player flips the playing flag off at
    # the last step; when that happens trigger ONE full rerun so render() can
    # disable run_every (otherwise the fragment would keep idle-ticking).
    playing_key = f"cbm__playing__{artifact.model_key}__{artifact.scenario_idx}"
    if auto_looping and not st.session_state.get(playing_key, False):
        st.rerun()


# ── Main render ───────────────────────────────────────────────────────

def render() -> None:
    st.markdown(context_strip("CBM Explorer"), unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:13px;color:#a7a8b3;margin-bottom:18px;'>"
        "<b style='color:#f4f4f7;'>CBM Scratch V2</b> · 15 concepts · "
        "400 WOMD validation scenarios. "
        "Select an archetype and scenario to explore what the CBM was 'thinking'.</p>",
        unsafe_allow_html=True,
    )

    # Check model is registered
    if CBM_MODEL_KEY not in PLATFORM_MODELS:
        st.error("CBM model not found in catalog.")
        return

    index = _load_index()
    if index is None:
        st.error(
            "curated_scenarios.json not found. "
            "Run `python cbm/find_demo_scenarios.py` first."
        )
        return

    available = get_available_scenarios(CBM_MODEL_KEY)
    all_cnames = index.get("concept_names_order", [
        "ego_speed","ego_acceleration","dist_nearest_object","num_objects_within_10m",
        "traffic_light_red","dist_to_traffic_light","heading_deviation",
        "progress_along_route","ttc_lead_vehicle","lead_vehicle_decelerating",
        "at_intersection","path_curvature_max","path_net_heading_change",
        "path_straightness","heading_to_path_end",
    ])

    # ── Layout: 260px control rail + flexible content area ──────────────────
    rail, content = st.columns([1, 4], gap="medium")

    with rail:
        st.markdown(
            "<div class='xai-rail' style='background:#25283a;border:1px solid #3b3f55;"
            "border-radius:12px;padding:14px;margin-top:8px;"
            "box-shadow:0 1px 0 rgba(255,255,255,0.02) inset,0 8px 24px rgba(0,0,0,0.25);'>"
            "<div style='font-size:11px;font-weight:700;text-transform:uppercase;"
            "letter-spacing:0.12em;color:#a7a8b3;margin-bottom:8px;'>Selection</div>",
            unsafe_allow_html=True,
        )
        arch_key = st.radio(
            "Archetype",
            options=list(ARCHETYPE_META.keys()),
            format_func=lambda k: ARCHETYPE_META[k]["label"],
            key="cbm__arch",
        )
        meta    = ARCHETYPE_META[arch_key]
        entries = index["archetypes"][arch_key]
        rank = st.slider(
            "Scenario rank",
            min_value=1, max_value=len(entries), value=1,
            key="cbm__rank",
        )
        entry     = entries[rank - 1]
        local_idx = entry["local_idx"]
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "<div style='font-size:11px;font-weight:700;text-transform:uppercase;"
            "letter-spacing:0.12em;color:#a7a8b3;margin:14px 0 8px;'>Concepts</div>",
            unsafe_allow_html=True,
        )
        selected_concepts = st.multiselect(
            "Concepts to show",
            options=all_cnames,
            default=[c for c in meta["concepts"] if c in all_cnames],
            key="cbm__concepts",
            label_visibility="collapsed",
        )

    # ── Content column ────────────────────────────────────────────────────────
    with content:
        accent = meta["color"]
        st.markdown(context_strip("CBM Explorer", model_label=CBM_MODEL_KEY,
                                   scenario_label=f"{meta['label']} · Rank {rank}"),
                    unsafe_allow_html=True)
        st.caption(meta["description"])

        # Top strip metric cards (scenario header)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("WOMD Scenario",      entry["scenario_idx"])
        c2.metric("Archetype Score",    f"{entry['score']:.4f}")
        c3.metric("Route Progress",     f"{entry['progress']:.3f}")
        c4.metric("At-Fault Collision", "✓ None" if entry["no_at_fault"] else "✗ Yes")

        # Precompute check
        if local_idx not in available:
            st.markdown(
                empty_state(
                    f"Scenario {local_idx} not yet precomputed for the CBM model.",
                    icon="◉",
                    hint="Static analysis below remains available.",
                    command="python scripts/precompute_cbm_demo.py --data cbm/data/training.tfrecord",
                ),
                unsafe_allow_html=True,
            )
            _render_static_fallback(entry, selected_concepts, all_cnames, accent)
            return

        artifact = _load_artifact_cached(CBM_MODEL_KEY, local_idx)
        if artifact is None:
            st.markdown(empty_state(f"Could not load artifact for scenario {local_idx}."),
                        unsafe_allow_html=True)
            return

        meta_data        = artifact.metadata or {}
        pred_c           = meta_data.get("pred_concepts")   # (80, 15)
        true_c           = meta_data.get("true_concepts")
        valid_c          = meta_data.get("valid_mask")
        cnames           = meta_data.get("concept_names", all_cnames)
        has_concepts     = pred_c is not None
        interesting_steps = artifact.interesting_timesteps or []

        # ── Main layout ────────────────────────────────────────────────
        # The player + step-synced charts live in a fragment. While playing,
        # run_every reruns ONLY this fragment every ~60 ms, so the rest of the
        # page stays mounted and the keyed charts update in place (no blink).
        # run_every is disabled when paused so tooltips/hover work normally.
        playing_key = f"cbm__playing__{artifact.model_key}__{artifact.scenario_idx}"
        is_playing  = bool(st.session_state.get(playing_key, False))
        player = st.fragment(run_every=0.06 if is_playing else None)(_render_cbm_player)
        player(
            artifact, local_idx, accent, selected_concepts,
            pred_c, true_c, valid_c, cnames, has_concepts, interesting_steps,
            is_playing,
        )

        # ── Static analysis section ───────────────────────────────────
        with st.expander("Static Analysis — Concept Quality (400 val scenarios)", expanded=False):
            _render_static_analysis()


def _render_static_fallback(entry, selected_concepts, all_cnames, accent):
    """Show static charts when artifact is not yet precomputed."""
    st.info("Showing static analysis from the full 400-scenario eval below.")
    _render_static_analysis()


def _render_static_analysis():
    """Show pre-generated thesis figures if available."""
    fig1 = _FIGURES_DIR / "fig1_concept_quality.png"
    fig4 = _FIGURES_DIR / "fig4_concept_temporal.png"

    if fig1.exists():
        st.image(str(fig1), caption="Concept quality — binary accuracy + continuous R²",
                 width='stretch')
    if fig4.exists():
        st.image(str(fig4), caption="Concept temporal evolution — mean pred vs true over 80 steps",
                 width='stretch')

    if not fig1.exists() and not fig4.exists():
        st.info(
            "Run `python cbm/generate_figures.py` to generate static analysis figures."
        )
