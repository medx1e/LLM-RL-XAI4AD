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
import streamlit as st

import platform  # path bootstrap
from platform.shared.model_catalog import PLATFORM_MODELS
from platform.shared.scenario_store import get_available_scenarios, load_artifact
from platform.shared.html_components import (
    context_strip, heatmap_table, empty_state, metric_card,
    section_badge, section_desc, method_profile_card, warning_banner,
)
from platform.shared.charts import (
    category_focus_chart, temporal_stability_chart,
)
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
    st.markdown(
        section_badge("A", "Method Profiles", "sparsity · Gini · top-10% · stability"),
        unsafe_allow_html=True,
    )
    st.markdown(
        section_desc(
            "Sparsity = how focused the attribution is. "
            "Concentration (Gini) = how unequal the feature weights are. "
            "Top-10% = fraction of total attribution mass in the top 10% of features."
        ),
        unsafe_allow_html=True,
    )
    cols = st.columns(len(methods))
    for col, name in zip(cols, methods):
        p = profiles[name]
        sp, gn, tk = p["sparsity"], p["gini"], p["topk10"]
        ts = p["temporal_stability"]
        tc, tf = p["top_cat"], p["top_cat_frac"]
        quality = _score_label(gn, high_good=True)
        color = _method_color(methods.index(name))
        dominant_str = "—"
        if tc != "n/a":
            dominant_str = tc + (f" ({tf:.0%})" if not math.isnan(tf) else "")
        with col:
            st.markdown(
                method_profile_card(
                    name=name,
                    color=color,
                    sparsity=f"{sp:.0%}" if not math.isnan(sp) else "n/a",
                    gini=f"{gn:.2f}" if not math.isnan(gn) else "n/a",
                    topk=f"{tk:.0%}" if not math.isnan(tk) else "n/a",
                    stability=f"{ts:.2f}" if not math.isnan(ts) else "n/a",
                    dominant=dominant_str,
                ),
                unsafe_allow_html=True,
            )
            if not math.isnan(sp):
                st.caption(
                    f"{quality} concentration — "
                    f"{'focused on few features' if gn > 0.6 else 'spread across many features'}. "
                    + (f"{'Stable' if ts > 0.6 else 'Variable'} across time." if not math.isnan(ts) else "")
                )


def _render_category_focus(series_map: dict[str, list], step: int, methods: list[str]) -> None:
    st.markdown(section_badge("B", "Category Focus"), unsafe_allow_html=True)
    st.markdown(
        section_desc("Which input category does each method consider most important? Wide agreement = robust finding."),
        unsafe_allow_html=True,
    )
    cat_data: dict[str, dict[str, float]] = {}
    all_cats: set[str] = set()
    for name in methods:
        series = series_map[name]
        if series is None or step >= len(series) or series[step] is None:
            cat_data[name] = {}
            continue
        cat_imp = series[step].category_importance
        cat_data[name] = {k: float(v) for k, v in cat_imp.items()}
        all_cats.update(cat_data[name].keys())

    if not all_cats:
        st.markdown(empty_state("No category data available."), unsafe_allow_html=True)
        return

    st.plotly_chart(
        category_focus_chart(cat_data, methods),
        width='stretch',
        config={"displayModeBar": False},
    )
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
    st.markdown(section_badge("C", "Method Agreement"), unsafe_allow_html=True)
    st.markdown(
        section_desc("Spearman ρ between each pair of methods at this timestep. Values near 1.0 = both methods rank features identically."),
        unsafe_allow_html=True,
    )
    pairs = pairwise_agreement(series_map, step)
    n = len(methods)
    mat = np.full((n, n), float("nan"))
    np.fill_diagonal(mat, 1.0)
    for (na, nb), rho in pairs.items():
        i, j = methods.index(na), methods.index(nb)
        mat[i, j] = mat[j, i] = rho

    short = [m.replace("_", " ") for m in methods]
    matrix_list = [[float(mat[i, j]) for j in range(n)] for i in range(n)]
    st.markdown(
        heatmap_table(matrix_list, short, short, title="Pairwise attribution agreement (Spearman ρ)"),
        unsafe_allow_html=True,
    )
    valid_rhos = [v for v in pairs.values() if not math.isnan(v)]
    if valid_rhos:
        mean_rho = float(np.mean(valid_rhos))
        level = "high" if mean_rho > 0.7 else ("moderate" if mean_rho > 0.4 else "low")
        st.caption(
            f"Mean agreement: **ρ = {mean_rho:.2f}** ({level}). "
            + ("Methods largely agree on feature importance." if level == "high" else
               "Methods partially disagree — worth comparing carefully." if level == "moderate" else
               "Methods strongly disagree — treat individual explanations with caution.")
        )


def _render_temporal_stability(series_map: dict[str, list], methods: list[str]) -> None:
    st.markdown(section_badge("D", "Temporal Stability"), unsafe_allow_html=True)
    st.markdown(
        section_desc("Jaccard similarity of the top-20 feature set between consecutive timesteps. High stability = reliable, predictable explanations over time."),
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        temporal_stability_chart(series_map, methods),
        width='stretch',
        config={"displayModeBar": False},
    )


def _render_faithfulness_placeholder() -> None:
    st.markdown(section_badge("E", "Faithfulness — Deletion Curves"), unsafe_allow_html=True)
    st.caption(
        "Faithfulness measures whether removing high-attribution features "
        "degrades the policy output more than removing random features. "
        "A faithful method's deletion curve should drop steeply at first "
        "(the most important features matter most)."
    )
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown(
            warning_banner(
                "Deletion curves require live model inference and have not been "
                "precomputed for this scenario. To generate them, run:",
                code="python scripts/precompute_faithfulness.py --model womd_sac_road_perceiver_minimal_42 --scenario 0",
            ),
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown("**What to expect**")
        st.markdown(
            "- A **steep** early drop = high faithfulness\n"
            "- **AUC** (area under curve) summarises faithfulness in one number\n"
            "- Random-baseline AUC ≈ 0.5; good methods score > 0.7"
        )


def _render_llm_placeholder() -> None:
    st.markdown(section_badge("F", "AI Narrative Analysis"), unsafe_allow_html=True)
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
    primary_keys = [k for k, e in PLATFORM_MODELS.items() if e.is_primary]
    if not primary_keys:
        st.markdown(context_strip("Evaluation"), unsafe_allow_html=True)
        st.markdown(empty_state("No primary models found."), unsafe_allow_html=True)
        return

    # ── Layout: 260px control rail + content ────────────────────────────────
    rail, content = st.columns([1, 4], gap="medium")

    with rail:
        st.markdown(
            "<div class='xai-rail panel xai-fade-in' style='padding:14px;margin-top:8px;'>"
            "<div style='font-size:10px;font-weight:700;text-transform:uppercase;"
            "letter-spacing:0.14em;color:var(--muted-fg,#a4a5b0);margin-bottom:10px;'>Selection</div>",
            unsafe_allow_html=True,
        )
        model_key = st.selectbox("Model", options=primary_keys, key="eval__model_key")
        available_scenarios = get_available_scenarios(model_key)
        if not available_scenarios:
            st.markdown("</div>", unsafe_allow_html=True)
            with content:
                st.markdown(context_strip("Evaluation"), unsafe_allow_html=True)
                st.markdown(empty_state(f"No cached scenarios for {model_key}."),
                            unsafe_allow_html=True)
            return
        scenario_idx = st.selectbox(
            "Scenario",
            options=available_scenarios,
            format_func=lambda i: f"Scenario {i}",
            key="eval__scenario_idx",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        cached_methods = list_cached_methods(model_key, scenario_idx)
        if not cached_methods:
            with content:
                st.markdown(context_strip("Evaluation"), unsafe_allow_html=True)
                st.markdown(empty_state("No cached attributions for this scenario."),
                            unsafe_allow_html=True)
            return

        st.markdown(
            "<div style='font-size:11px;font-weight:700;text-transform:uppercase;"
            "letter-spacing:0.14em;color:var(--muted-fg,#a4a5b0);margin:14px 0 8px;'>Methods</div>",
            unsafe_allow_html=True,
        )
        selected_methods = st.multiselect(
            "Methods to compare",
            options=cached_methods,
            default=cached_methods,
            key="eval__methods",
            label_visibility="collapsed",
        )

        artifact = load_artifact(model_key, scenario_idx)
        if artifact is None:
            with content:
                st.markdown(context_strip("Evaluation"), unsafe_allow_html=True)
                st.markdown(empty_state("Could not load artifact."), unsafe_allow_html=True)
            return

        st.markdown(
            "<div style='font-size:11px;font-weight:700;text-transform:uppercase;"
            "letter-spacing:0.14em;color:var(--muted-fg,#a4a5b0);margin:14px 0 8px;'>Reference step</div>",
            unsafe_allow_html=True,
        )
        step = st.slider(
            "Reference timestep",
            min_value=0,
            max_value=artifact.num_steps - 1,
            value=artifact.num_steps // 2,
            key="eval__step",
            label_visibility="collapsed",
        )

    # ── Content column ────────────────────────────────────────────────────────
    with content:
        st.markdown(
            context_strip("Evaluation",
                          model_label=model_key,
                          scenario_label=f"Scenario {scenario_idx}"),
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size:13px;color:var(--muted-fg,#a4a5b0);margin-bottom:18px;'>"
            "Quality and consistency of post-hoc attribution methods at "
            f"step <b style='color:#f4f4f7;'>{step}</b>.</p>",
            unsafe_allow_html=True,
        )

        if not selected_methods:
            st.markdown(
                empty_state("Select at least one method on the left."),
                unsafe_allow_html=True,
            )
            return

        @st.cache_data(show_spinner="Loading attribution series…")
        def _load_series(mk, si, methods_tuple):
            return {m: load_attribution_series(mk, si, m) for m in methods_tuple}

        series_map = _load_series(model_key, scenario_idx, tuple(selected_methods))

        from platform.evaluation.metrics import method_profile as _profile
        profiles = {m: _profile(series_map[m], step) for m in selected_methods}

        # ── Method Profiles (top metric strip) ────────────────────────────
        _render_method_profiles(profiles, selected_methods)
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        # ── 2-col grid: Category focus | Method agreement ─────────────────
        col_cat, col_agree = st.columns(2, gap="medium")
        with col_cat:
            _render_category_focus(series_map, step, selected_methods)
        with col_agree:
            if len(selected_methods) >= 2:
                _render_method_agreement(series_map, step, selected_methods)
            else:
                st.markdown(
                    "<h3 style='font-size:17px;font-weight:700;color:#f4f4f7;"
                    "font-family:Inter,sans-serif;margin:0 0 6px;'>Method Agreement</h3>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    empty_state("Select 2+ methods to compare agreement."),
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        # ── Temporal stability (full width) ───────────────────────────────
        _render_temporal_stability(series_map, selected_methods)

        # ── Faithfulness placeholder + LLM narrative (collapsibles) ───────
        with st.expander("Faithfulness — Deletion Curves", expanded=False):
            _render_faithfulness_placeholder()
        with st.expander("AI Narrative Analysis", expanded=False):
            _render_llm_placeholder()

        # ── Export bar ─────────────────────────────────────────────────────
        st.markdown(
            "<div style='display:flex;gap:10px;justify-content:flex-end;"
            "padding:14px 0 4px;border-top:1px solid #3b3f55;margin-top:14px;'>"
            "<button disabled style='background:transparent;border:1px solid #3b3f55;"
            "color:var(--muted-fg,#a4a5b0);border-radius:8px;padding:8px 14px;font-size:12px;"
            "font-weight:600;cursor:not-allowed;font-family:Inter,sans-serif;'>"
            "Download CSV</button>"
            "<button disabled style='background:transparent;border:1px solid #3b3f55;"
            "color:var(--muted-fg,#a4a5b0);border-radius:8px;padding:8px 14px;font-size:12px;"
            "font-weight:600;cursor:not-allowed;font-family:Inter,sans-serif;'>"
            "Download PDF report</button>"
            "</div>",
            unsafe_allow_html=True,
        )
