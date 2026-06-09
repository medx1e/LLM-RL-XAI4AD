"""Home tab — discovery surface (hero + feature cards + metric strip + scenarios + models)."""

from __future__ import annotations

import streamlit as st

import platform  # path bootstrap
from platform.shared.model_catalog import PLATFORM_MODELS
from platform.shared.scenario_store import get_available_scenarios
from platform.shared.html_components import (
    context_strip, empty_state, hero, feature_card,
    metric_card, section_badge, scenario_card,
)


_SVG_EYE = (
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/>'
    '<circle cx="12" cy="12" r="3"/></svg>'
)
_SVG_GRAPH = (
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="5" r="2"/><circle cx="5" cy="19" r="2"/><circle cx="19" cy="19" r="2"/>'
    '<line x1="12" y1="7" x2="5" y2="17"/><line x1="12" y1="7" x2="19" y2="17"/></svg>'
)
_SVG_CHAT = (
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>'
)
_SVG_CHART = (
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<line x1="18" y1="20" x2="18" y2="10"/>'
    '<line x1="12" y1="20" x2="12" y2="4"/>'
    '<line x1="6" y1="20" x2="6" y2="14"/></svg>'
)

# Accent palette harmonised with the indigo theme (was ColorBrewer Set1, which
# clashed against the dark UI). Indigo → teal → sky → amber.
_FEATURE_CARDS = [
    {
        "to":    "Post-hoc XAI",
        "icon":  _SVG_EYE,
        "color": "#6f6ae8",
        "title": "Post-hoc XAI",
        "desc":  "Attribution maps, Perceiver cross-attention, and entity-level inspection per timestep.",
        "tag":   "Open Post-hoc XAI →",
    },
    {
        "to":    "CBM Explorer",
        "icon":  _SVG_GRAPH,
        "color": "#2fbe87",
        "title": "CBM Explorer",
        "desc":  "Concept Bottleneck Model timelines across 15 interpretable driving concepts.",
        "tag":   "Open CBM Explorer →",
    },
    {
        "to":    "LLM Narration",
        "icon":  _SVG_CHAT,
        "color": "#4c9aed",
        "title": "LLM Narration",
        "desc":  "Natural-language narrations of policy decisions across LLM × toggle combinations.",
        "tag":   "Open LLM Narration →",
    },
    {
        "to":    "Evaluation",
        "icon":  _SVG_CHART,
        "color": "#f5a623",
        "title": "Evaluation",
        "desc":  "Method agreement, faithfulness, and attention–attribution alignment reports.",
        "tag":   "Open Evaluation →",
    },
]


def _navigate(tab: str) -> None:
    """Set the active tab and rerun."""
    st.session_state["nav__selected_tab"] = tab
    st.rerun()


def render() -> None:
    st.markdown(context_strip("XAI4AD"), unsafe_allow_html=True)

    primary_models  = {k: v for k, v in PLATFORM_MODELS.items() if v.is_primary}
    total_models    = len(PLATFORM_MODELS)
    total_scenarios = sum(len(get_available_scenarios(k)) for k in primary_models)
    cbm_concepts    = 15
    methods_repr    = "9+"

    # ── Hero (leads the page — the promise before the proof) ───────────────────
    st.markdown(
        hero(
            eyebrow="Research Platform · Waymo Open Motion",
            title="Understand <span style='color:var(--primary,#6f6ae8);'>why</span> your driving agent acted.",
            subtitle=(
                "Explore post-hoc attribution maps, concept bottleneck timelines, "
                "and LLM-generated narrations — all synchronized to a Bird's-Eye View "
                "episode replay of V-Max policies on WOMD."
            ),
            n_scenarios=total_scenarios if total_scenarios else None,
        ),
        unsafe_allow_html=True,
    )

    # ── CTA buttons (one primary action + one quiet secondary) ─────────────────
    _, cta_a, cta_b, _ = st.columns([2, 1, 1, 2])
    with cta_a:
        if st.button("Open Post-hoc XAI →", key="home_cta_posthoc", width="stretch"):
            _navigate("Post-hoc XAI")
    with cta_b:
        if st.button("Compare LLM narrations", key="home_cta_narr", width="stretch"):
            _navigate("LLM Narration")

    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)

    # ── Metric proof bar (evidence beneath the claim) ──────────────────────────
    m1, m2, m3, m4 = st.columns(4, gap="medium")
    with m1:
        st.markdown(metric_card("Models", str(total_models), "in catalog", large=True), unsafe_allow_html=True)
    with m2:
        st.markdown(metric_card("Cached scenarios", str(total_scenarios), "primary models", large=True), unsafe_allow_html=True)
    with m3:
        st.markdown(metric_card("CBM concepts", str(cbm_concepts), "phases 1+2+3", large=True), unsafe_allow_html=True)
    with m4:
        st.markdown(metric_card("Attribution methods", methods_repr, "cached per scenario", large=True), unsafe_allow_html=True)

    st.markdown("<div style='height:36px;'></div>", unsafe_allow_html=True)

    # ── Feature cards (canonical entry points — real navigation) ───────────────
    st.markdown(section_badge("A", "Platform Features", large=True), unsafe_allow_html=True)
    for col, card in zip(st.columns(4, gap="medium"), _FEATURE_CARDS):
        with col:
            st.markdown(
                feature_card(
                    color=card["color"],
                    title=card["title"],
                    desc=card["desc"],
                    icon_char=card["icon"],
                ),
                unsafe_allow_html=True,
            )
            if st.button(card["tag"], key=f"home_feat_{card['to']}", width="stretch"):
                _navigate(card["to"])

    st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)

    # ── Scenarios | Model discovery ───────────────────────────────────────────
    col_sc, col_md = st.columns([3, 2], gap="large")

    scenario_rows: list[tuple] = []
    for key, entry in primary_models.items():
        for idx in get_available_scenarios(key):
            scenario_rows.append((key, entry, idx))

    with col_sc:
        st.markdown(section_badge("B", "Curated Scenarios", "primary models", large=True), unsafe_allow_html=True)
        if not scenario_rows:
            st.markdown(
                empty_state(
                    "No pre-computed artifacts found",
                    icon="◫",
                    hint="Run the precompute script to generate cached scenarios.",
                    command="python scripts/precompute_posthoc_demo.py",
                ),
                unsafe_allow_html=True,
            )
        else:
            delay_classes = ["", "xai-delay-1", "xai-delay-2"]
            # Adapt the grid to the number of scenarios so a lone card doesn't
            # leave two-thirds of the row empty.
            cards_per_row = min(3, len(scenario_rows))
            for row_start in range(0, len(scenario_rows), cards_per_row):
                row_items = scenario_rows[row_start: row_start + cards_per_row]
                cols = st.columns(cards_per_row)
                for ci, (col, (key, entry, idx)) in enumerate(zip(cols, row_items)):
                    with col:
                        st.markdown(
                            scenario_card(
                                scenario_id=f"{idx:04d}",
                                model_name=key,
                                encoder_family=entry.encoder_family,
                                has_attention=entry.has_attention,
                                description=entry.description or "",
                                delay_class=delay_classes[ci % len(delay_classes)],
                            ),
                            unsafe_allow_html=True,
                        )
                        if st.button(
                            "Open in Post-hoc XAI →",
                            key=f"home_scen_{key}_{idx}",
                            width="stretch",
                        ):
                            # Pre-seed the Post-hoc tab's selection widgets.
                            st.session_state["posthoc__model_key"]    = key
                            st.session_state["posthoc__scenario_idx"] = idx
                            _navigate("Post-hoc XAI")

    with col_md:
        st.markdown(section_badge("C", "Model Discovery", large=True), unsafe_allow_html=True)
        table_rows = [
            {
                "Model":     key,
                "Family":    entry.encoder_family,
                "Attention": "✓" if entry.has_attention else "—",
                "Scenarios": len(get_available_scenarios(key)),
            }
            for key, entry in PLATFORM_MODELS.items()
        ]
        if table_rows:
            # Size to content (was a fixed 400px showing ~8 empty rows).
            table_height = min(len(table_rows) * 36 + 38, 420)
            st.dataframe(table_rows, width="stretch", hide_index=True, height=table_height)
        else:
            st.markdown(empty_state("No models registered."), unsafe_allow_html=True)
