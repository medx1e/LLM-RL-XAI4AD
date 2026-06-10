"""Post-hoc XAI tab.

Layout
------
Sidebar  : model · scenario · methods (multiselect) · display options
Left col : BEV episode player  →  episode info panel
Right col: attribution plots (one block per selected method, stacked)
Bottom   : Perceiver attention section (semantic bar + optional BEV overlay)
"""

from __future__ import annotations

import math

import numpy as np
import streamlit as st

import platform  # path bootstrap
from platform.shared.bev_component import render_bev_player, render_bev_frame, schedule_bev_rerun
from platform.shared.contracts import XAINotReadyError
from platform.shared.model_catalog import PLATFORM_MODELS
from platform.shared.scenario_store import load_artifact
from platform.shared.html_components import (
    context_strip, heatmap_table, empty_state,
)
from platform.shared.charts import (
    category_importance_chart, entity_importance_chart,
    attribution_timeline_chart,
    attention_entity_chart, category_focus_chart,
    temporal_stability_chart,
)
from platform.shared.theme import AGENT_COLORS
from platform.posthoc.adapter import (
    has_cached_attention,
    list_cached_methods,
    list_posthoc_scenarios,
    load_attribution_series,
    load_attention_series,
)
from platform.posthoc.viz import (
    AGENT_ID_COLORS,
    aggregate_attention_by_entity,
    make_attention_overlay_fn,
    plot_attention_by_entity,
)


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _attr_key(model_key, scenario_idx, method):
    return f"posthoc__attr__{model_key}__{scenario_idx}__{method}"

def _attn_key(model_key, scenario_idx):
    return f"posthoc__attn__{model_key}__{scenario_idx}"

def _ensure_attr(model_key, scenario_idx, method):
    k = _attr_key(model_key, scenario_idx, method)
    if k not in st.session_state:
        st.session_state[k] = load_attribution_series(model_key, scenario_idx, method)
    return st.session_state[k]

def _ensure_attn(model_key, scenario_idx):
    k = _attn_key(model_key, scenario_idx)
    if k not in st.session_state:
        st.session_state[k] = load_attention_series(model_key, scenario_idx)
    return st.session_state[k]


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:
    primary_keys = [k for k, e in PLATFORM_MODELS.items() if e.is_primary]
    if not primary_keys:
        st.markdown(context_strip("Post-hoc XAI Explorer"), unsafe_allow_html=True)
        st.markdown(empty_state("No primary models found."), unsafe_allow_html=True)
        return

    # ── Layout: 260px control rail + flexible analysis area ──────────────────
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
        model_key = st.selectbox("Model", options=primary_keys, key="posthoc__model_key")
        # Only list scenarios that actually have Post-hoc data (attributions /
        # attention) cached. Scenarios carrying just an artifact + BEV frames —
        # e.g. those precomputed for the LLM Narration tab — share the same
        # platform_cache namespace but must not appear here.
        available_scenarios = list_posthoc_scenarios(model_key)
        if not available_scenarios:
            scenario_idx = None
        else:
            scenario_idx = st.selectbox(
                "Scenario",
                options=available_scenarios,
                format_func=lambda i: f"Scenario {i}",
                key="posthoc__scenario_idx",
            )
        st.markdown("</div>", unsafe_allow_html=True)

        cached_methods = list_cached_methods(model_key, scenario_idx) if scenario_idx is not None else []
        st.markdown(
            "<div style='font-size:11px;font-weight:700;text-transform:uppercase;"
            "letter-spacing:0.12em;color:#a7a8b3;margin:14px 0 8px;'>Methods</div>",
            unsafe_allow_html=True,
        )
        if not cached_methods:
            st.markdown(empty_state("No cached attributions."), unsafe_allow_html=True)
            selected_methods = []
        else:
            selected_methods = st.multiselect(
                "Attribution methods",
                options=cached_methods,
                default=cached_methods[:2] if len(cached_methods) >= 2 else cached_methods,
                key="posthoc__methods",
                label_visibility="collapsed",
            )

        st.markdown(
            "<div style='font-size:11px;font-weight:700;text-transform:uppercase;"
            "letter-spacing:0.12em;color:#a7a8b3;margin:14px 0 8px;'>Display</div>",
            unsafe_allow_html=True,
        )
        top_n = st.slider(
            "Top-N entities", min_value=5, max_value=30, value=10,
            key="posthoc__top_n",
        )
        normalize_entities = st.checkbox(
            "Normalize entities", value=False, key="posthoc__normalize",
        )
        normalize_per_token = st.checkbox(
            "Per-token category", value=False, key="posthoc__per_token",
        )

    # ── Content column ────────────────────────────────────────────────────────
    with content:
        st.markdown(context_strip("Post-hoc XAI Explorer"), unsafe_allow_html=True)

        if scenario_idx is None:
            st.markdown(
                empty_state(
                    f"No cached scenarios for {model_key}.",
                    icon="◫",
                    hint="Run the precompute script to generate cached scenarios.",
                    command="python scripts/precompute_posthoc_demo.py",
                ),
                unsafe_allow_html=True,
            )
            return

        artifact = load_artifact(model_key, scenario_idx)
        if artifact is None:
            st.markdown(
                empty_state(f"Could not load artifact for {model_key} / scenario {scenario_idx}."),
                unsafe_allow_html=True,
            )
            return

        entry = PLATFORM_MODELS[model_key]

        # Two-column analysis grid: BEV | methods
        col_bev, col_xai = st.columns([1, 1], gap="medium")

        with col_bev:
            st.markdown(
                "<h3 style='font-size:16px;font-weight:700;color:#f4f4f7;"
                "font-family:Inter,sans-serif;margin:0 0 10px;'>Episode Replay</h3>",
                unsafe_allow_html=True,
            )
            if artifact.notes:
                st.caption(artifact.notes)
            step = render_bev_player(artifact, key_prefix="posthoc")
            if artifact.interesting_timesteps:
                st.caption("Flagged: " + ", ".join(str(t) for t in artifact.interesting_timesteps))


        with col_xai:
            st.markdown(
                "<h3 style='font-size:16px;font-weight:700;color:#f4f4f7;"
                "font-family:Inter,sans-serif;margin:0 0 10px;'>Attribution Analysis</h3>",
                unsafe_allow_html=True,
            )
            if not selected_methods:
                st.markdown(
                    empty_state("Select at least one attribution method on the left."),
                    unsafe_allow_html=True,
                )
            else:
                # One tab per method keeps the column the height of a single
                # chart (aligned with the BEV) instead of stacking N charts.
                method_tabs = st.tabs(
                    [m.replace("_", " ").title() for m in selected_methods]
                )
                for tab, method in zip(method_tabs, selected_methods):
                    with tab:
                        _render_method_block(
                            artifact, step, method, top_n, normalize_entities,
                            normalize_per_token, model_key, scenario_idx,
                        )

        # Full-width attention panel
        has_attn = entry.has_attention and has_cached_attention(model_key, scenario_idx)
        if has_attn:
            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            _render_attention_section(artifact, model_key, scenario_idx, step)
        elif entry.has_attention:
            st.markdown(
                empty_state("Attention not cached yet.",
                            command="python scripts/precompute_posthoc_demo.py"),
                unsafe_allow_html=True,
            )

        # Collapsible evaluation report (last block)
        if selected_methods:
            with st.expander("Evaluation Report", expanded=False):
                _render_evaluation_report(
                    artifact, model_key, scenario_idx, step,
                    selected_methods, has_attn,
                )


# ---------------------------------------------------------------------------
# Attribution method block  (one per selected method)
# ---------------------------------------------------------------------------

def _render_method_block(
    artifact, step: int, method: str,
    top_n: int, normalize: bool, normalize_per_token: bool,
    model_key: str, scenario_idx: int,
) -> None:
    series = _ensure_attr(model_key, scenario_idx, method)
    if series is None:
        st.warning(f"Could not load series for '{method}'.")
        return
    if step >= len(series) or series[step] is None:
        st.warning(f"No attribution at step {step} for '{method}'.")
        return

    attribution = series[step]

    st.caption(
        f"**{attribution.method_name}** — step {step + 1}/{artifact.num_steps} "
        f"({attribution.computation_time_ms:.0f} ms)"
    )
    view = st.segmented_control(
        "chart_view",
        options=["Category", "Entities", "Timeline"],
        default="Category",
        key=f"sc_posthoc_{method}",
        label_visibility="collapsed",
    )
    if view == "Entities":
        st.plotly_chart(
            entity_importance_chart(attribution, top_n=top_n, normalize=normalize),
            width='stretch',
            config={"displayModeBar": False},
        )
    elif view == "Timeline":
        st.plotly_chart(
            attribution_timeline_chart(series, method_name=attribution.method_name, current_step=step),
            width='stretch',
            config={"displayModeBar": False},
        )
    else:
        st.plotly_chart(
            category_importance_chart(attribution, normalize_per_token=normalize_per_token),
            width='stretch',
            config={"displayModeBar": False},
        )


# ---------------------------------------------------------------------------
# Evaluation report  (inline, triggered by expander)
# ---------------------------------------------------------------------------

_METHOD_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
]


def _render_evaluation_report(
    artifact, model_key: str, scenario_idx: int, step: int,
    selected_methods: list[str], has_attn: bool,
) -> None:
    from platform.evaluation.metrics import (
        method_profile,
        pairwise_agreement,
        attention_attribution_alignment,
    )

    # Load series (may already be cached in session_state)
    series_map = {m: _ensure_attr(model_key, scenario_idx, m) for m in selected_methods}

    # ── 1. Method Profiles ────────────────────────────────────────────────────
    st.subheader("Method Profiles")
    st.caption(
        "**Sparsity** = fraction of features near-zero (high = focused). "
        "**Gini** = attribution inequality (high = concentrated on few features). "
        "**Top-10% coverage** = attribution mass in top 10% of features."
    )
    profile_cols = st.columns(len(selected_methods))
    profiles = {}
    for col, m in zip(profile_cols, selected_methods):
        p = method_profile(series_map[m], step)
        profiles[m] = p
        with col:
            st.markdown(f"**{m.replace('_', ' ').title()}**")
            sp, gn, tk = p["sparsity"], p["gini"], p["topk10"]
            ts = p["temporal_stability"]
            tc, tf = p["top_cat"], p["top_cat_frac"]
            st.metric("Sparsity", f"{sp:.0%}" if not math.isnan(sp) else "n/a")
            st.metric("Gini concentration", f"{gn:.2f}" if not math.isnan(gn) else "n/a")
            st.metric("Top-10% coverage", f"{tk:.0%}" if not math.isnan(tk) else "n/a")
            st.metric("Temporal stability", f"{ts:.2f}" if not math.isnan(ts) else "n/a")
            if tc != "n/a":
                frac_str = f" ({tf:.0%})" if not math.isnan(tf) else ""
                st.metric("Dominant category", f"{tc}{frac_str}")

    st.divider()

    # ── 2. Category focus comparison ──────────────────────────────────────────
    st.subheader("Category Focus")
    st.caption("Which input category does each method consider most important?")

    all_cats: set[str] = set()
    cat_data: dict[str, dict[str, float]] = {}
    for m in selected_methods:
        sm = series_map[m]
        if sm and step < len(sm) and sm[step] is not None:
            cat_imp = sm[step].category_importance
            cat_data[m] = {k: float(v) for k, v in cat_imp.items()}
            all_cats.update(cat_data[m].keys())
        else:
            cat_data[m] = {}

    cats = sorted(all_cats)
    if cats:
        st.plotly_chart(
            category_focus_chart(cat_data, selected_methods),
            width='stretch',
            config={"displayModeBar": False},
        )

        # Agreement callout
        dominant = {m: max(cat_data[m], key=cat_data[m].get, default="n/a")
                    if cat_data[m] else "n/a" for m in selected_methods}
        unique_dom = set(dominant.values()) - {"n/a"}
        if len(unique_dom) == 1:
            st.success(
                f"All methods agree: **{next(iter(unique_dom))}** is the most important category."
            )
        elif len(unique_dom) > 1:
            lines = ", ".join(f"**{m}** → {dominant[m]}" for m in selected_methods)
            st.warning(f"Methods disagree on the dominant category: {lines}.")

    st.divider()

    # ── 3. Method agreement heatmap (2+ methods) ──────────────────────────────
    if len(selected_methods) >= 2:
        st.subheader("Method Agreement")
        st.caption(
            "Spearman ρ between every pair of methods at this timestep. "
            "ρ near 1 = both methods rank features identically."
        )
        pairs = pairwise_agreement(series_map, step)
        n = len(selected_methods)
        mat = np.full((n, n), float("nan"))
        np.fill_diagonal(mat, 1.0)
        for (na, nb), rho in pairs.items():
            i, j = selected_methods.index(na), selected_methods.index(nb)
            mat[i, j] = mat[j, i] = rho

        short = [m.replace("_", " ") for m in selected_methods]
        matrix_list = [[float(mat[i, j]) for j in range(n)] for i in range(n)]
        st.markdown(
            heatmap_table(matrix_list, short, short, title="Pairwise attribution agreement (Spearman ρ)"),
            unsafe_allow_html=True,
        )

        valid_rhos = [v for v in pairs.values() if not math.isnan(v)]
        if valid_rhos:
            mr = float(np.mean(valid_rhos))
            level = "high" if mr > 0.7 else ("moderate" if mr > 0.4 else "low")
            st.caption(
                f"Mean agreement ρ = **{mr:.2f}** ({level}). "
                + ("Methods largely agree — findings are robust." if level == "high" else
                   "Partial disagreement — compare methods carefully." if level == "moderate" else
                   "Strong disagreement — treat individual explanations with caution.")
            )
        st.divider()

    # ── 4. Attention–attribution alignment ────────────────────────────────────
    if has_attn and selected_methods:
        st.subheader("Attention–Attribution Alignment")
        st.caption(
            "Does what the Perceiver *attends to* match what each attribution method "
            "considers *important*? Each cell is Spearman ρ between the attention weight "
            "and the attribution importance for that agent slot across all timesteps. "
            "**High ρ = method and attention tell the same story for this agent.**"
        )
        attn_series = _ensure_attn(model_key, scenario_idx)
        if attn_series is not None:
            with st.spinner("Computing alignment…"):
                alignment = attention_attribution_alignment(
                    attn_series, series_map, artifact,
                )

            n_agents = 8
            n_methods = len(selected_methods)
            mat = np.full((n_agents, n_methods), float("nan"))
            for j, m in enumerate(selected_methods):
                rhos = alignment.get(m, [])
                for i, rho in enumerate(rhos[:n_agents]):
                    mat[i, j] = rho

            method_labels = [m.replace("_", " ") for m in selected_methods]
            agent_labels  = [f"A{i}" for i in range(n_agents)]
            matrix_list   = [[float(mat[i, j]) for j in range(n_methods)] for i in range(n_agents)]
            st.markdown(
                heatmap_table(
                    matrix_list, agent_labels, method_labels,
                    title="Attention–attribution alignment (Spearman ρ per agent slot)",
                    row_colors=AGENT_ID_COLORS,
                ),
                unsafe_allow_html=True,
            )

            # Summary interpretation
            valid = mat[~np.isnan(mat)]
            if len(valid):
                high_agree = (valid > 0.6).mean()
                st.caption(
                    f"{high_agree:.0%} of (agent, method) pairs show high alignment (ρ > 0.6). "
                    + ("Attention and attribution are consistent — the Perceiver focuses on "
                       "what matters." if high_agree > 0.5 else
                       "Attention and attribution often disagree — the model may attend to "
                       "features for reasons not captured by gradient-based methods.")
                )


# ---------------------------------------------------------------------------
# Attention section
# ---------------------------------------------------------------------------

def _render_attention_section(artifact, model_key: str, scenario_idx: int, step: int) -> None:  # noqa: E501
    st.subheader("Perceiver Attention — Semantic View")
    st.caption(
        "Each agent slot (A0–A7) has a fixed identity colour shared between "
        "the BEV overlay and the bar chart. **Colour = which vehicle**, "
        "**bar length / border thickness = how much attention**."
    )

    attn_series = _ensure_attn(model_key, scenario_idx)
    if attn_series is None or step >= len(attn_series) or attn_series[step] is None:
        st.warning("Attention data unavailable for this timestep.")
        return

    attn_dict = attn_series[step]

    # Controls row
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 2])
    with ctrl_col1:
        available_keys = [k for k in attn_dict if "cross_attn" in k]
        if not available_keys:
            available_keys = list(attn_dict.keys())
        chosen_key = st.selectbox(
            "Attention layer",
            options=available_keys,
            index=available_keys.index("cross_attn_avg") if "cross_attn_avg" in available_keys else 0,
            key="posthoc__attn_key",
        )
    with ctrl_col2:
        show_overlay = st.checkbox(
            "Overlay on BEV",
            value=True,
            key="posthoc__attn_overlay",
            help="Paint each observation agent with its identity colour on the map",
        )
    with ctrl_col3:
        if show_overlay:
            agent_opts = [f"A{i}" for i in range(8)]
            sel_agents_labels = st.multiselect(
                "Agents to show",
                options=agent_opts,
                default=agent_opts,
                key="posthoc__overlay_agents",
            )
            selected_agents = {int(s[1:]) for s in sel_agents_labels}
        else:
            selected_agents = None

    # Side-by-side BEV + bar chart — critical for colour matching
    entity_weights = aggregate_attention_by_entity(attn_dict, key=chosen_key, artifact=artifact, step=step)

    if show_overlay:
        col_map, col_bars = st.columns([1, 1], gap="small")
        with col_map:
            # Synced transport controls — reads/writes the SAME session-state
            # keys as the main PostHoc BEV player so both stay in lock-step.
            playing_key = f"posthoc__playing__{artifact.model_key}__{artifact.scenario_idx}"
            attn_btn_key = f"posthoc__attn_playbtn__{artifact.model_key}__{artifact.scenario_idx}"
            is_playing = st.session_state.get(playing_key, False)
            ctrl_prog, ctrl_bt = st.columns([5, 1])
            with ctrl_prog:
                st.progress(
                    (step + 1) / artifact.num_steps,
                    text=f"Step {step + 1} / {artifact.num_steps}",
                )
            with ctrl_bt:
                btn_label = "⏸" if is_playing else "▶"
                if st.button(btn_label, key=attn_btn_key, width='stretch'):
                    st.session_state[playing_key] = not is_playing
                    st.rerun()
            overlay_fn = make_attention_overlay_fn(artifact, attn_series, selected_agents)
            render_bev_frame(
                artifact, step,
                overlay_fn=overlay_fn,
                caption=f"Identity colours — digit = slot index ({chosen_key})",
            )
        with col_bars:
            if entity_weights:
                st.plotly_chart(
                    attention_entity_chart(entity_weights),
                    width='stretch',
                    config={"displayModeBar": False},
                )
    else:
        if entity_weights:
            st.plotly_chart(
                attention_entity_chart(entity_weights),
                width='stretch',
                config={"displayModeBar": False},
            )

    # Trigger rerun for BEV playback after all content is rendered.
    schedule_bev_rerun()
