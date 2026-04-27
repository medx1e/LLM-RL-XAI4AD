"""Evaluation & Report tab.

Story layout
------------
A  Selector sidebar (model / scenario / methods / step)
B  Method Profiles  — sparsity, concentration, top-category cards per method
C  Category Focus   — side-by-side horizontal bars comparing methods
D  Method Agreement — Spearman correlation heatmap (2+ methods only)
E  Temporal Stability — rank-Jaccard timeline per method
F  Faithfulness Placeholder — deletion-curve section, button to trigger precompute
G  AI Narrative Placeholder — "Generate Analysis" button for future LLM integration
"""

from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st

import platform  # path bootstrap
from platform.shared.model_catalog import PLATFORM_MODELS
from platform.shared.scenario_store import get_available_scenarios, load_artifact
from platform.posthoc.adapter import list_cached_methods, load_attribution_series
from platform.evaluation.metrics import method_profile, pairwise_agreement


# ---------------------------------------------------------------------------
# Palette helpers
# ---------------------------------------------------------------------------

_METHOD_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
]


def _method_color(i: int) -> str:
    return _METHOD_COLORS[i % len(_METHOD_COLORS)]


def _score_label(v: float, high_good: bool = True) -> str:
    if math.isnan(v):
        return "n/a"
    if high_good:
        return "High" if v >= 0.7 else ("Medium" if v >= 0.4 else "Low")
    return "Low" if v >= 0.7 else ("Medium" if v >= 0.4 else "High")


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_method_profiles(profiles: dict[str, dict], methods: list[str]) -> None:
    st.subheader("Method Profiles")
    st.caption(
        "Quick-read scorecards. **Sparsity** = how focused the attribution is "
        "(high = fewer features matter). **Concentration (Gini)** = how unequal "
        "the distribution is. **Top-10% coverage** = what fraction of total "
        "importance sits in the highest-scoring 10% of features."
    )
    cols = st.columns(len(methods))
    for col, name in zip(cols, methods):
        p = profiles[name]
        with col:
            st.markdown(f"**{name.replace('_', ' ').title()}**")
            sp, gn, tk = p["sparsity"], p["gini"], p["topk10"]
            ts = p["temporal_stability"]
            tc, tf = p["top_cat"], p["top_cat_frac"]

            st.metric("Sparsity", f"{sp:.0%}" if not math.isnan(sp) else "n/a",
                      help="Fraction of features with near-zero attribution")
            st.metric("Concentration (Gini)", f"{gn:.2f}" if not math.isnan(gn) else "n/a",
                      help="0=uniform, 1=single-feature; higher = more selective")
            st.metric("Top-10% coverage", f"{tk:.0%}" if not math.isnan(tk) else "n/a",
                      help="Attribution mass in the top 10% of features")
            st.metric("Temporal stability", f"{ts:.2f}" if not math.isnan(ts) else "n/a",
                      help="Mean Jaccard similarity of top-K feature sets across timesteps")
            if tc != "n/a":
                st.metric(
                    "Dominant category",
                    f"{tc} ({tf:.0%})" if not math.isnan(tf) else tc,
                    help="Category that absorbs the most attribution at this timestep",
                )

            # Interpretation line
            if not math.isnan(sp):
                quality = _score_label(gn, high_good=True)
                st.caption(
                    f"{quality} concentration — "
                    f"{'focused on few features' if gn > 0.6 else 'spread across many features'}. "
                    f"{'Stable across time.' if not math.isnan(ts) and ts > 0.6 else 'Variable across time.' if not math.isnan(ts) else ''}"
                )


def _render_category_focus(series_map: dict[str, list], step: int, methods: list[str]) -> None:
    st.subheader("Category Focus")
    st.caption(
        "Which input category does each method consider most important? "
        "Wide agreement across methods = robust finding."
    )

    all_cats: set[str] = set()
    cat_data: dict[str, dict[str, float]] = {}

    for name in methods:
        series = series_map[name]
        if series is None or step >= len(series) or series[step] is None:
            cat_data[name] = {}
            continue
        cat_imp = series[step].category_importance
        cat_data[name] = {k: float(v) for k, v in cat_imp.items()}
        all_cats.update(cat_data[name].keys())

    cats = sorted(all_cats)
    if not cats:
        st.info("No category data available.")
        return

    fig, ax = plt.subplots(figsize=(9, max(2.5, len(cats) * 0.6)))
    x = np.arange(len(cats))
    bar_w = 0.8 / max(len(methods), 1)

    for i, name in enumerate(methods):
        vals = [cat_data[name].get(c, 0.0) for c in cats]
        offset = (i - len(methods) / 2 + 0.5) * bar_w
        ax.barh(x + offset, vals, height=bar_w,
                label=name.replace("_", " "), color=_method_color(i), alpha=0.85)

    ax.set_yticks(x)
    ax.set_yticklabels(cats)
    ax.set_xlabel("Normalised importance")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Attribution by category — all methods")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Agreement sentence
    dominant = {n: max(cat_data[n], key=cat_data[n].get, default="n/a")
                if cat_data[n] else "n/a" for n in methods}
    unique_dom = set(dominant.values()) - {"n/a"}
    if len(unique_dom) == 1:
        st.success(f"All methods agree: **{next(iter(unique_dom))}** is the most important category.")
    elif len(unique_dom) > 1:
        listing = ", ".join(f"**{n}** → {dominant[n]}" for n in methods)
        st.warning(f"Methods disagree on the dominant category: {listing}.")


def _render_method_agreement(series_map: dict[str, list], step: int, methods: list[str]) -> None:
    if len(methods) < 2:
        return
    st.subheader("Method Agreement")
    st.caption(
        "Spearman rank correlation between each pair of methods at this timestep. "
        "Values near 1.0 = both methods rank features identically. "
        "Low agreement may indicate that methods capture different aspects of the policy."
    )

    pairs = pairwise_agreement(series_map, step)

    fig, ax = plt.subplots(figsize=(max(3, len(methods) * 1.2), max(2.5, len(methods) * 1.0)))
    mat = np.full((len(methods), len(methods)), float("nan"))
    np.fill_diagonal(mat, 1.0)
    for (na, nb), rho in pairs.items():
        i, j = methods.index(na), methods.index(nb)
        mat[i, j] = mat[j, i] = rho

    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(methods)))
    ax.set_yticks(range(len(methods)))
    short = [m.replace("_", "\n") for m in methods]
    ax.set_xticklabels(short, fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    for i in range(len(methods)):
        for j in range(len(methods)):
            val = mat[i, j]
            if not math.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color="black" if abs(val) < 0.7 else "white")
    plt.colorbar(im, ax=ax, label="Spearman ρ", fraction=0.046)
    ax.set_title("Pairwise attribution agreement")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Interpretation
    valid_rhos = [v for v in pairs.values() if not math.isnan(v)]
    if valid_rhos:
        mean_rho = np.mean(valid_rhos)
        level = "high" if mean_rho > 0.7 else ("moderate" if mean_rho > 0.4 else "low")
        st.caption(
            f"Mean agreement: **ρ = {mean_rho:.2f}** ({level}). "
            + ("Methods largely agree on feature importance." if level == "high" else
               "Methods partially disagree — worth comparing explanations carefully." if level == "moderate" else
               "Methods strongly disagree — treat individual explanations with caution.")
        )


def _render_temporal_stability(series_map: dict[str, list], methods: list[str]) -> None:
    st.subheader("Temporal Stability")
    st.caption(
        "How consistent is each method's top-20 feature set across consecutive timesteps? "
        "Each point is the Jaccard similarity between step t and t+1. "
        "High stability = the method gives reliable, predictable explanations over time."
    )

    fig, ax = plt.subplots(figsize=(9, 3))
    for i, name in enumerate(methods):
        series = series_map[name]
        if series is None:
            continue
        valid = [s for s in series if s is not None]
        if len(valid) < 2:
            continue

        def topk_set(attr, k=20):
            flat = np.abs(np.array(attr.raw).ravel())
            idx = np.argpartition(flat, -min(k, len(flat)))[-min(k, len(flat)):]
            return set(idx.tolist())

        jaccs = []
        xs = []
        for t, (a, b) in enumerate(zip(valid[:-1], valid[1:])):
            sa, sb = topk_set(a), topk_set(b)
            union = len(sa | sb)
            jaccs.append(len(sa & sb) / union if union else 0.0)
            xs.append(t + 1)

        if jaccs:
            ax.plot(xs, jaccs, label=name.replace("_", " "),
                    color=_method_color(i), linewidth=1.5, alpha=0.85)
            ax.fill_between(xs, jaccs, alpha=0.08, color=_method_color(i))

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Jaccard similarity (top-20 features)")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
    ax.legend(fontsize=8)
    ax.set_title("Top-feature set stability across timesteps")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _render_faithfulness_placeholder() -> None:
    st.subheader("Faithfulness — Deletion Curves")
    st.caption(
        "Faithfulness measures whether removing high-attribution features "
        "degrades the policy output more than removing random features. "
        "A faithful method's deletion curve should drop steeply at first "
        "(the most important features matter most)."
    )
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.info(
            "Deletion curves require live model inference and have not been "
            "precomputed for this scenario.\n\n"
            "To generate them, run:\n"
            "```\npython scripts/precompute_faithfulness.py "
            "--model womd_sac_road_perceiver_minimal_42 --scenario 0\n```"
        )
    with col_b:
        st.markdown("**What to expect**")
        st.markdown(
            "- A **steep** early drop = high faithfulness\n"
            "- **AUC** (area under curve) summarises faithfulness in one number\n"
            "- Random-baseline AUC ≈ 0.5; good methods score > 0.7"
        )


def _render_llm_placeholder() -> None:
    st.subheader("AI Narrative Analysis")
    st.caption(
        "An LLM will synthesise the above metrics into a plain-language story "
        "about what the attribution methods reveal about the driving policy."
    )
    col_btn, col_preview = st.columns([1, 2])
    with col_btn:
        if st.button("Generate Analysis", key="eval__llm_generate", disabled=True):
            pass
        st.caption("_Coming soon — LLM integration not yet wired._")
    with col_preview:
        st.markdown(
            "> **Example output:**\n"
            "> *'The Perceiver policy consistently attends to the roadgraph "
            "> (42% of attention) and nearby vehicles (35%), while GPS path "
            "> contributes only 8%. Vanilla Gradient and Integrated Gradients "
            "> agree strongly (ρ = 0.81), suggesting the roadgraph finding is "
            "> robust. However, the policy's attention is temporally unstable "
            "> after step 40 — coinciding with the lane-change manoeuvre — "
            "> which may warrant further investigation.'*"
        )


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:
    st.title("Evaluation & Report")
    st.markdown(
        "This page evaluates the **quality and consistency** of the post-hoc "
        "attribution methods applied to the selected scenario. Use it to judge "
        "which explanations are trustworthy and what they collectively reveal "
        "about the driving policy."
    )

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Evaluation Controls")

        primary_keys = [k for k, e in PLATFORM_MODELS.items() if e.is_primary]
        if not primary_keys:
            st.error("No primary models found.")
            return

        model_key = st.selectbox("Model", options=primary_keys, key="eval__model_key")

        available_scenarios = get_available_scenarios(model_key)
        if not available_scenarios:
            st.warning(f"No cached scenarios for **{model_key}**.")
            return

        scenario_idx = st.selectbox(
            "Scenario",
            options=available_scenarios,
            format_func=lambda i: f"Scenario {i}",
            key="eval__scenario_idx",
        )

        cached_methods = list_cached_methods(model_key, scenario_idx)
        if not cached_methods:
            st.warning("No cached attributions for this scenario.")
            return

        selected_methods = st.multiselect(
            "Methods to compare",
            options=cached_methods,
            default=cached_methods,
            key="eval__methods",
        )
        if not selected_methods:
            st.info("Select at least one method.")
            return

        st.divider()
        # Let user pick a reference timestep for per-step metrics
        artifact = load_artifact(model_key, scenario_idx)
        if artifact is None:
            st.error("Could not load artifact.")
            return

        step = st.slider(
            "Reference timestep",
            min_value=0,
            max_value=artifact.num_steps - 1,
            value=artifact.num_steps // 2,
            key="eval__step",
            help="Per-step metrics (profile, agreement) are computed at this timestep.",
        )

    # ── Load series ───────────────────────────────────────────────────────────
    @st.cache_data(show_spinner="Loading attribution series…")
    def _load_series(mk, si, methods_tuple):
        return {m: load_attribution_series(mk, si, m) for m in methods_tuple}

    series_map = _load_series(model_key, scenario_idx, tuple(selected_methods))

    # ── Per-method profiles ───────────────────────────────────────────────────
    from platform.evaluation.metrics import method_profile as _profile
    profiles = {m: _profile(series_map[m], step) for m in selected_methods}

    _render_method_profiles(profiles, selected_methods)
    st.divider()

    # ── Category focus ────────────────────────────────────────────────────────
    _render_category_focus(series_map, step, selected_methods)
    st.divider()

    # ── Method agreement ──────────────────────────────────────────────────────
    if len(selected_methods) >= 2:
        _render_method_agreement(series_map, step, selected_methods)
        st.divider()

    # ── Temporal stability ────────────────────────────────────────────────────
    _render_temporal_stability(series_map, selected_methods)
    st.divider()

    # ── Faithfulness placeholder ──────────────────────────────────────────────
    _render_faithfulness_placeholder()
    st.divider()

    # ── LLM narrative placeholder ─────────────────────────────────────────────
    _render_llm_placeholder()
