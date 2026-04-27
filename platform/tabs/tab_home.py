"""Home tab — project overview and curated scenario table."""

from __future__ import annotations

import streamlit as st

import platform  # path bootstrap
from platform.shared.model_catalog import PLATFORM_MODELS
from platform.shared.scenario_store import get_available_scenarios


def render() -> None:
    st.title("Explainable RL for Autonomous Driving")
    st.markdown(
        """
        This platform demonstrates **post-hoc XAI** methods applied to
        V-Max policies trained on the Waymo Open Motion Dataset (WOMD).

        Use the sidebar to navigate between tabs:
        - **Post-hoc XAI** — gradient & perturbation attributions, attention heatmaps
        """
    )

    st.divider()
    st.subheader("Curated Demo Scenarios")
    st.markdown(
        "The table below shows all pre-computed scenarios available for exploration. "
        "Each scenario has a full attribution series cached for every listed method."
    )

    rows = []
    for key, entry in PLATFORM_MODELS.items():
        if not entry.is_primary:
            continue
        idxs = get_available_scenarios(key)
        for idx in idxs:
            rows.append(
                {
                    "Model": key,
                    "Encoder": entry.encoder_family,
                    "Attention": "✓" if entry.has_attention else "—",
                    "Scenario": idx,
                    "Notes": entry.description,
                }
            )

    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info(
            "No pre-computed artifacts found yet. "
            "Run `scripts/precompute_posthoc_demo.py` to generate them."
        )
