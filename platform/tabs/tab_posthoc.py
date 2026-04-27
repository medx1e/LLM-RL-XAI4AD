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

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import platform  # path bootstrap
from platform.shared.bev_component import render_bev_player, render_bev_frame
from platform.shared.contracts import XAINotReadyError
from platform.shared.model_catalog import PLATFORM_MODELS
from platform.shared.scenario_store import get_available_scenarios, load_artifact
from platform.posthoc.adapter import (
    has_cached_attention,
    list_cached_methods,
    load_attribution_series,
    load_attention_series,
)
from platform.posthoc.viz import (
    AGENT_ID_COLORS,
    aggregate_attention_by_entity,
    make_attention_overlay_fn,
    plot_attention_by_entity,
    plot_attribution_timeline,
    plot_category_importance,
    plot_entity_importance,
    plot_episode_info,
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
    st.title("Post-hoc XAI Explorer")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Controls")

        primary_keys = [k for k, e in PLATFORM_MODELS.items() if e.is_primary]
        if not primary_keys:
            st.error("No primary models found.")
            return

        model_key = st.selectbox("Model", options=primary_keys, key="posthoc__model_key")

        available_scenarios = get_available_scenarios(model_key)
        if not available_scenarios:
            st.warning(f"No cached scenarios for **{model_key}**.")
            return

        scenario_idx = st.selectbox(
            "Scenario",
            options=available_scenarios,
            format_func=lambda i: f"Scenario {i}",
            key="posthoc__scenario_idx",
        )

        st.divider()
        cached_methods = list_cached_methods(model_key, scenario_idx)
        if not cached_methods:
            st.warning("No cached attributions. Run precompute script.")
            selected_methods = []
        else:
            selected_methods = st.multiselect(
                "Attribution methods",
                options=cached_methods,
                default=cached_methods[:2] if len(cached_methods) >= 2 else cached_methods,
                help="Select one or more methods to compare side-by-side",
                key="posthoc__methods",
            )

        st.divider()
        st.subheader("Display options")
        top_n = st.slider(
            "Top entities shown",
            min_value=5, max_value=30, value=10,
            help="Number of highest-importance entities in the bar chart",
            key="posthoc__top_n",
        )
        normalize_entities = st.checkbox(
            "Normalize entities (relative share)",
            value=False,
            help="Rescale entity bars to sum=1, so bars show each entity's relative share of importance",
            key="posthoc__normalize",
        )
        normalize_per_token = st.checkbox(
            "Category: per-token importance",
            value=False,
            help="Divide category bar by its token count — fair cross-category comparison (roadgraph has 200 tokens vs 5 for SDC)",
            key="posthoc__per_token",
        )

    # ── Load artifact ────────────────────────────────────────────────────────
    artifact = load_artifact(model_key, scenario_idx)
    if artifact is None:
        st.error(f"Could not load artifact for {model_key} / scenario {scenario_idx}.")
        return

    entry = PLATFORM_MODELS[model_key]

    # ── Main two-column layout ────────────────────────────────────────────────
    col_bev, col_xai = st.columns([1, 1], gap="medium")

    with col_bev:
        st.subheader("Episode Replay")
        if artifact.notes:
            st.caption(artifact.notes)
        step = render_bev_player(artifact, key_prefix="posthoc")
        if artifact.interesting_timesteps:
            st.caption("Flagged: " + ", ".join(str(t) for t in artifact.interesting_timesteps))

        # Episode info panel
        st.subheader("Episode Statistics")
        rewards = np.array(artifact.scenario_data.rewards)
        dones   = np.array(artifact.scenario_data.dones).astype(bool)
        crash_steps = np.where(dones)[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Cumulative reward", f"{rewards.sum():.2f}")
        c2.metric("Reward @ step", f"{rewards[step]:.3f}")
        c3.metric("Episode outcome", "✕ Done" if dones.any() else "✓ Complete")

        fig_ep = plot_episode_info(artifact, current_step=step)
        st.pyplot(fig_ep, use_container_width=True)

    with col_xai:
        st.subheader("Attribution Analysis")
        if not selected_methods:
            st.info("Select at least one attribution method in the sidebar.")
        else:
            for method in selected_methods:
                _render_method_block(
                    artifact, step, method, top_n, normalize_entities,
                    normalize_per_token, model_key, scenario_idx,
                )

    # ── Attention section (Perceiver only) ────────────────────────────────────
    has_attn = entry.has_attention and has_cached_attention(model_key, scenario_idx)
    if has_attn:
        st.divider()
        _render_attention_section(artifact, model_key, scenario_idx, step)
    elif entry.has_attention:
        st.divider()
        st.info("Attention not cached yet. Re-run `precompute_posthoc_demo.py`.")

    # ── Evaluation report ─────────────────────────────────────────────────────
    if selected_methods:
        st.divider()
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

    with st.expander(
        f"**{attribution.method_name}** — step {step + 1}/{artifact.num_steps} "
        f"({attribution.computation_time_ms:.0f} ms)",
        expanded=True,
    ):
        tab_cat, tab_ent, tab_timeline = st.tabs(
            ["Category", "Entities", "Timeline"]
        )
        with tab_cat:
            st.pyplot(
                plot_category_importance(attribution, normalize_per_token=normalize_per_token),
                use_container_width=True,
            )
        with tab_ent:
            st.pyplot(
                plot_entity_importance(attribution, top_n=top_n, normalize=normalize),
                use_container_width=True,
            )
        with tab_timeline:
            st.pyplot(
                plot_attribution_timeline(series, method_name=attribution.method_name, current_step=step),
                use_container_width=True,
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
        fig, ax = plt.subplots(figsize=(9, max(2.5, len(cats) * 0.55)))
        x = np.arange(len(cats))
        bw = 0.8 / max(len(selected_methods), 1)
        for i, m in enumerate(selected_methods):
            vals = [cat_data[m].get(c, 0.0) for c in cats]
            offset = (i - len(selected_methods) / 2 + 0.5) * bw
            ax.barh(x + offset, vals, height=bw,
                    label=m.replace("_", " "),
                    color=_METHOD_PALETTE[i % len(_METHOD_PALETTE)], alpha=0.85)
        ax.set_yticks(x)
        ax.set_yticklabels(cats)
        ax.set_xlabel("Normalised importance")
        ax.legend(fontsize=8)
        ax.set_title("Attribution by category — all methods at current step")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

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

        fig, ax = plt.subplots(figsize=(max(3, n * 1.3), max(2.5, n * 1.1)))
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
        short = [m.replace("_", "\n") for m in selected_methods]
        ax.set_xticks(range(n)); ax.set_xticklabels(short, fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=8)
        for i in range(n):
            for j in range(n):
                v = mat[i, j]
                if not math.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                            color="black" if abs(v) < 0.7 else "white")
        plt.colorbar(im, ax=ax, label="Spearman ρ", fraction=0.046)
        ax.set_title("Pairwise attribution agreement")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

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

            fig, ax = plt.subplots(figsize=(max(4, n_methods * 1.5), max(3, n_agents * 0.7)))
            masked = np.ma.masked_invalid(mat)
            im = ax.imshow(masked, vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")

            ax.set_xticks(range(n_methods))
            ax.set_xticklabels([m.replace("_", "\n") for m in selected_methods], fontsize=8)
            ax.set_yticks(range(n_agents))
            ax.set_yticklabels([f"A{i}" for i in range(n_agents)], fontsize=8)

            # Colour the y-tick labels with agent identity colours
            for tick, color in zip(ax.get_yticklabels(), AGENT_ID_COLORS):
                tick.set_color(color)

            for i in range(n_agents):
                for j in range(n_methods):
                    v = mat[i, j]
                    if not math.isnan(v):
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                fontsize=8, color="black" if abs(v) < 0.65 else "white")

            plt.colorbar(im, ax=ax, label="Spearman ρ (attn vs. attr)", fraction=0.046)
            ax.set_title("Attention–attribution alignment per agent slot")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

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
        st.divider()

    # ── 5. Faithfulness placeholder ────────────────────────────────────────────
    st.subheader("Faithfulness — Deletion Curves")
    col_info, col_expect = st.columns([2, 1])
    with col_info:
        st.info(
            "Deletion curves require live model inference and have not been precomputed.\n\n"
            "Run:\n```\npython scripts/precompute_faithfulness.py "
            "--model womd_sac_road_perceiver_minimal_42 --scenario 0\n```"
        )
    with col_expect:
        st.markdown(
            "**What to expect**\n"
            "- Steep early drop = high faithfulness\n"
            "- AUC > 0.7 = method is trustworthy\n"
            "- Random baseline AUC ≈ 0.5"
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
    if show_overlay:
        col_map, col_bars = st.columns([1, 1], gap="small")
        with col_map:
            overlay_fn = make_attention_overlay_fn(artifact, attn_series, selected_agents)
            render_bev_frame(
                artifact, step,
                overlay_fn=overlay_fn,
                caption=f"Identity colours — digit = slot index ({chosen_key})",
            )
        with col_bars:
            fig = plot_attention_by_entity(
                attn_dict, key=chosen_key, artifact=artifact, step=step,
            )
            if fig:
                st.pyplot(fig, use_container_width=True)
    else:
        fig = plot_attention_by_entity(
            attn_dict, key=chosen_key, artifact=artifact, step=step,
        )
        if fig:
            st.pyplot(fig, use_container_width=True)
