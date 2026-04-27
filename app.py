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
from platform.tabs import tab_home, tab_posthoc, tab_cbm  # noqa: E402


_TABS = {
    "Home": tab_home,
    "Post-hoc XAI": tab_posthoc,
    "CBM Explorer": tab_cbm,
}

with st.sidebar:
    st.title("Navigation")
    selected = st.radio(
        "Go to",
        options=list(_TABS.keys()),
        key="nav__selected_tab",
        label_visibility="collapsed",
    )

_TABS[selected].render()
