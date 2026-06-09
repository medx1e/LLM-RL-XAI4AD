"""Streamlit entry-point for the Explainable RL platform.

Run with:
    streamlit run app.py
"""

import sys
import importlib.util
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent

# Force our `platform` package to load even if stdlib platform is cached.
# Streamlit pre-imports stdlib `platform`, leaving it in sys.modules; we
# must explicitly override it with our package before any platform.* import.
def _bootstrap_platform_package() -> None:
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    pkg_init = _PROJECT_ROOT / "platform" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "platform", str(pkg_init),
        submodule_search_locations=[str(_PROJECT_ROOT / "platform")],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["platform"] = mod
    spec.loader.exec_module(mod)

_bootstrap_platform_package()

import streamlit as st

st.set_page_config(
    page_title="Explainable RL — Autonomous Driving",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

import platform  # path bootstrap  # noqa: E402 (must run after set_page_config)
from platform.tabs import tab_home, tab_posthoc, tab_cbm, tab_evaluation  # noqa: E402
from platform import llm_narration  # noqa: E402
from platform.shared.theme import inject_theme  # noqa: E402
from platform.shared.html_components import sidebar_brand  # noqa: E402

inject_theme()

_TABS = {
    "Home":          tab_home,
    "Post-hoc XAI":  tab_posthoc,
    "CBM Explorer":  tab_cbm,
    "LLM Narration": llm_narration,
    "Evaluation":    tab_evaluation,
}

_NAV_ICONS = {
    "Home":          "home",
    "Post-hoc XAI":  "visibility",
    "CBM Explorer":  "account_tree",
    "LLM Narration": "chat",
    "Evaluation":    "bar_chart",
}

# Initialise nav state on first load
if "nav__selected_tab" not in st.session_state:
    st.session_state["nav__selected_tab"] = "Home"

with st.sidebar:
    st.markdown(sidebar_brand(), unsafe_allow_html=True)
    st.markdown(
        "<div style='padding:4px 12px 6px;font-size:10px;font-weight:600;"
        "text-transform:uppercase;letter-spacing:0.12em;color:#a7a8b3;'>Workspace</div>",
        unsafe_allow_html=True,
    )
    for tab_name in _TABS:
        is_active = st.session_state["nav__selected_tab"] == tab_name
        if st.button(
            tab_name,
            key=f"nav_{tab_name}",
            icon=f":material/{_NAV_ICONS[tab_name]}:",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state["nav__selected_tab"] = tab_name
            st.rerun()
    st.markdown(
        "<div style='position:absolute;bottom:16px;left:12px;right:12px;"
        "font-family:\"JetBrains Mono\",monospace;font-size:10px;color:#3b3f55;'>",
        unsafe_allow_html=True,
    )

_TABS[st.session_state["nav__selected_tab"]].render()
