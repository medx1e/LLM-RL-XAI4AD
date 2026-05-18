"""LLM Narration tab — viewer for precomputed XRL narrations.

Single-file module (flat layout) that wires up:
  • Config constants (paths, driving-model map, LLM registry, toggle combos)
  • Lightweight dataclasses for the narration entries
  • Path resolution helpers
  • Loader functions that return Streamlit-ready `list[T]` (forward-filled)
  • The Streamlit `render()` entrypoint

The platform is a pure viewer — no LLM calls, no computation. All data is
produced offline by ``scripts/precompute_llm_narration.py`` and dropped into
``data/llm_narration/`` in the layout this module reads.

Synchronization follows the existing platform pattern: every list returned
by the loaders has length T (total scenario timesteps) and is indexed by the
same ``step`` integer the BEV slider produces.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Optional

import streamlit as st

import platform  # path bootstrap (idempotent)
from platform.shared.bev_component import render_bev_player
from platform.shared.model_catalog import PLATFORM_MODELS
from platform.shared.scenario_store import load_artifact


# =============================================================================
# CONFIG — single source of truth
# =============================================================================

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT: Path = _PROJECT_ROOT / "data" / "llm_narration"

# Narration events are emitted every 2 seconds. At 10 Hz simulator step rate
# that is 20 timesteps. Must match ``frequencies.report`` in xai_config.yaml.
NARRATION_INTERVAL_STEPS: int = 20

# Driving-model selector → PLATFORM_MODELS key. Used by both UI and precompute.
DRIVING_MODELS: dict[str, str] = {
    "LQ (Perceiver)": "SAC Perceiver — WOMD seed 42",
    "Wayformer":      "SAC Wayformer — WOMD seed 42",
}

# Short filesystem slug for each driving model. Keyed by PLATFORM_MODELS key
# (NOT the display label) so that ``driving_model_slug(model_key)`` resolves.
# Used in `data/llm_narration/` directory names so paths stay short.
DRIVING_MODEL_SLUGS: dict[str, str] = {
    "SAC Perceiver — WOMD seed 42": "lq",
    "SAC Wayformer — WOMD seed 42": "wayformer",
}

# LLM registry. Each entry mirrors the `llm:` sub-dict expected by
# llm_integration/xai/narration/llm_narrator.py (OpenRouter-style payload).
LLM_MODELS: dict[str, dict] = {
    "llama": {
        "display":      "Llama 3.1 70B (Groq)",
        "provider":     "groq",
        "model":        "llama-3.1-70b-versatile",
        "base_url":     "https://api.groq.com/openai/v1/chat/completions",
        "api_key_env":  "GROQ_API_KEY",
        "max_tokens":   400,
    },
    "gpt": {
        "display":      "GPT-4o (OpenAI)",
        "provider":     "openai",
        "model":        "gpt-4o",
        "base_url":     "https://api.openai.com/v1/chat/completions",
        "api_key_env":  "OPENAI_API_KEY",
        "max_tokens":   400,
    },
    "claude": {
        "display":      "Claude 3.5 Sonnet (OpenRouter)",
        "provider":     "openrouter",
        "model":        "anthropic/claude-3.5-sonnet",
        "base_url":     "https://openrouter.ai/api/v1/chat/completions",
        "api_key_env":  "OPENROUTER_API_KEY",
        "max_tokens":   400,
    },
}

# Toggle combinations. The precompute pipeline materializes one narration
# file per combo; the UI lets the user pick at viewing time.
TOGGLE_COMBOS: dict[str, dict] = {
    "full":              {"grounding": True,  "counterfactual": True},
    "no_grounding":      {"grounding": False, "counterfactual": True},
    "no_counterfactual": {"grounding": True,  "counterfactual": False},
    "minimal":           {"grounding": False, "counterfactual": False},
}


def toggle_key_from_flags(grounding: bool, counterfactual: bool) -> str:
    """Resolve the canonical toggle key for a (grounding, counterfactual) pair."""
    for key, flags in TOGGLE_COMBOS.items():
        if flags["grounding"] == grounding and flags["counterfactual"] == counterfactual:
            return key
    raise ValueError(f"No TOGGLE_COMBOS entry for grounding={grounding}, cf={counterfactual}")


# Curated scenario list. (scenario_idx, descriptive_label).
# The tab filters this list to scenarios that actually have precomputed data.
SCENARIO_CATALOG: list[tuple[int, str]] = [
    (0,  "Scenario 0 — Lane following"),
    (1,  "Scenario 1 — Intersection approach"),
    (2,  "Scenario 2 — Lead vehicle braking"),
    (3,  "Scenario 3 — Merge from on-ramp"),
    (4,  "Scenario 4 — Pedestrian crossing"),
    (5,  "Scenario 5 — Multi-agent intersection"),
    (6,  "Scenario 6 — Yield to oncoming"),
    (7,  "Scenario 7 — Cut-in maneuver"),
    (8,  "Scenario 8 — Stop-and-go traffic"),
    (9,  "Scenario 9 — Unprotected left turn"),
]

# Visual tone palette — Detailed (red), Caveat (yellow), Brief (green).
TONE_COLORS: dict[str, str] = {
    "detailed":         "#dc2626",
    "detailed_caveat":  "#f59e0b",
    "brief":            "#10b981",
    "caveat":           "#f59e0b",  # alias accepted in older artifacts
}

TONE_LABELS: dict[str, str] = {
    "detailed":         "DETAILED",
    "detailed_caveat":  "CAVEAT",
    "brief":            "BRIEF",
    "caveat":           "CAVEAT",
}


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclasses.dataclass
class NarrationEntry:
    """One LLM narration event aligned to a specific timestep."""
    timestep: int
    timestamp_sec: float
    tone: str
    text: str
    response_time_s: Optional[float] = None

    @classmethod
    def from_dict(cls, d: dict) -> "NarrationEntry":
        return cls(
            timestep=int(d["timestep"]),
            timestamp_sec=float(d.get("timestamp_sec", d["timestep"] * 0.1)),
            tone=str(d.get("tone", "brief")),
            text=str(d.get("text", "")),
            response_time_s=d.get("response_time_s"),
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# =============================================================================
# PATHS
# =============================================================================

def scenario_id(scenario_idx: int) -> str:
    """Canonical scenario directory name. Matches platform_cache convention."""
    return f"scenario_{scenario_idx:04d}"


def driving_model_slug(model_key: str) -> str:
    """Filesystem-safe slug for a driving model. Falls back to a generic slug."""
    if model_key in DRIVING_MODEL_SLUGS:
        return DRIVING_MODEL_SLUGS[model_key]
    return re.sub(r"[^\w]+", "_", model_key).strip("_").lower()


def get_scenario_dir(scenario_idx: int, model_key: str) -> Path:
    return DATA_ROOT / scenario_id(scenario_idx) / driving_model_slug(model_key)


def get_reports_dir(scenario_idx: int, model_key: str) -> Path:
    return get_scenario_dir(scenario_idx, model_key) / "reports"


def get_report_path(scenario_idx: int, model_key: str, step: int) -> Path:
    return get_reports_dir(scenario_idx, model_key) / f"t_{step:04d}.json"


def get_narrations_dir(scenario_idx: int, model_key: str) -> Path:
    return get_scenario_dir(scenario_idx, model_key) / "narrations"


def get_narrations_path(
    scenario_idx: int,
    model_key: str,
    llm_key: str,
    toggle_key: str,
) -> Path:
    return get_narrations_dir(scenario_idx, model_key) / f"{llm_key}__{toggle_key}.json"


# =============================================================================
# LOADERS — what Streamlit calls at runtime
# =============================================================================

def load_reports(scenario_idx: int, model_key: str) -> dict[int, dict]:
    """Load all per-event JSON reports for one (scenario, driving_model) pair.

    Returns
    -------
    dict mapping ``step -> report_dict``. Empty dict if nothing is precomputed.
    """
    reports_dir = get_reports_dir(scenario_idx, model_key)
    if not reports_dir.is_dir():
        return {}

    out: dict[int, dict] = {}
    for f in sorted(reports_dir.glob("t_*.json")):
        try:
            step = int(f.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        try:
            with open(f, "r") as fh:
                out[step] = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
    return out


def load_narration_entries(
    scenario_idx: int,
    model_key: str,
    llm_key: str,
    toggle_key: str,
) -> list[NarrationEntry]:
    """Load the raw (sparse) narration list for one combo. Empty if missing."""
    path = get_narrations_path(scenario_idx, model_key, llm_key, toggle_key)
    if not path.is_file():
        return []
    with open(path, "r") as fh:
        raw = json.load(fh)
    if not isinstance(raw, list):
        return []
    return [NarrationEntry.from_dict(d) for d in raw]


def _forward_fill_entries(
    entries: list[NarrationEntry],
    num_steps: int,
) -> list[Optional[NarrationEntry]]:
    """Expand sparse narration events to a length-T list (most-recent-holds).

    Steps before the first event are None.
    """
    by_step = {e.timestep: e for e in entries}
    out: list[Optional[NarrationEntry]] = [None] * num_steps
    current: Optional[NarrationEntry] = None
    for t in range(num_steps):
        if t in by_step:
            current = by_step[t]
        out[t] = current
    return out


def load_narrations_for_display(
    scenario_idx: int,
    model_key: str,
    llm_key: str,
    toggle_key: str,
    num_steps: int,
) -> list[Optional[NarrationEntry]]:
    """Streamlit-facing loader: returns ``list[T]``, forward-filled.

    ``narrations[step]`` is the narration that is currently "in effect" at
    that timestep — i.e. the most recent emitted event, or ``None`` if no
    narration has been emitted yet.
    """
    entries = load_narration_entries(scenario_idx, model_key, llm_key, toggle_key)
    return _forward_fill_entries(entries, num_steps)


def load_reports_for_display(
    scenario_idx: int,
    model_key: str,
    num_steps: int,
) -> list[Optional[dict]]:
    """Like ``load_narrations_for_display`` but for the structured reports.

    The user sees the most recent report at every in-between timestep.
    """
    reports = load_reports(scenario_idx, model_key)
    if not reports:
        return [None] * num_steps
    sorted_steps = sorted(reports.keys())
    out: list[Optional[dict]] = [None] * num_steps
    current: Optional[dict] = None
    next_idx = 0
    next_step = sorted_steps[0]
    for t in range(num_steps):
        while next_idx < len(sorted_steps) and sorted_steps[next_idx] <= t:
            current = reports[sorted_steps[next_idx]]
            next_idx += 1
            if next_idx < len(sorted_steps):
                next_step = sorted_steps[next_idx]
        out[t] = current
    return out


# =============================================================================
# SESSION-STATE CACHE
# =============================================================================

def _narr_key(scenario_idx, model_key, llm_key, toggle_key, num_steps):
    return f"llm_narr__narr__{model_key}__{scenario_idx}__{llm_key}__{toggle_key}__{num_steps}"


def _reports_key(scenario_idx, model_key, num_steps):
    return f"llm_narr__reports__{model_key}__{scenario_idx}__{num_steps}"


def _ensure_narrations(scenario_idx, model_key, llm_key, toggle_key, num_steps):
    k = _narr_key(scenario_idx, model_key, llm_key, toggle_key, num_steps)
    if k not in st.session_state:
        st.session_state[k] = load_narrations_for_display(
            scenario_idx, model_key, llm_key, toggle_key, num_steps,
        )
    return st.session_state[k]


def _ensure_reports(scenario_idx, model_key, num_steps):
    k = _reports_key(scenario_idx, model_key, num_steps)
    if k not in st.session_state:
        st.session_state[k] = load_reports_for_display(
            scenario_idx, model_key, num_steps,
        )
    return st.session_state[k]


# =============================================================================
# AVAILABILITY HELPERS — used by sidebar selectors
# =============================================================================

def has_precomputed_data(scenario_idx: int, model_key: str) -> bool:
    """True if any reports exist for this (scenario, driving_model) pair."""
    rd = get_reports_dir(scenario_idx, model_key)
    return rd.is_dir() and any(rd.glob("t_*.json"))


def available_narration_combos(
    scenario_idx: int, model_key: str,
) -> list[tuple[str, str]]:
    """All (llm_key, toggle_key) pairs that have a precomputed file on disk."""
    nd = get_narrations_dir(scenario_idx, model_key)
    if not nd.is_dir():
        return []
    out: list[tuple[str, str]] = []
    for f in nd.glob("*__*.json"):
        stem = f.stem
        if "__" not in stem:
            continue
        llm_key, toggle_key = stem.split("__", 1)
        if llm_key in LLM_MODELS and toggle_key in TOGGLE_COMBOS:
            out.append((llm_key, toggle_key))
    return sorted(out)


# =============================================================================
# UI HELPERS
# =============================================================================

def _render_narration_box(entry: Optional[NarrationEntry], header: str = "Narration") -> None:
    """Tone-coloured narration card."""
    if entry is None:
        st.info("No narration available yet at this timestep.")
        return

    color = TONE_COLORS.get(entry.tone, "#6b7280")
    label = TONE_LABELS.get(entry.tone, entry.tone.upper())

    badge = (
        f"<span style='background:{color};color:white;padding:2px 8px;"
        f"border-radius:4px;font-size:0.75rem;font-weight:600;"
        f"letter-spacing:0.05em;'>{label}</span>"
    )
    meta = f"<span style='color:#6b7280;font-size:0.8rem;'>t = {entry.timestep} · {entry.timestamp_sec:.1f}s</span>"

    container_style = (
        f"border-left:4px solid {color};"
        "padding:0.75rem 1rem;background:rgba(0,0,0,0.02);"
        "border-radius:0 4px 4px 0;margin-top:0.25rem;"
    )

    st.markdown(
        f"<div style='{container_style}'>"
        f"<div style='display:flex;justify-content:space-between;"
        f"align-items:center;margin-bottom:0.4rem;'>"
        f"<strong>{header}</strong>{badge}"
        f"</div>"
        f"<div style='color:#374151;line-height:1.5;'>{entry.text}</div>"
        f"<div style='margin-top:0.5rem;'>{meta}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _select_scenario(model_key: str) -> Optional[int]:
    """Sidebar scenario selector, filtered to scenarios with precomputed data."""
    available = [
        (idx, label) for idx, label in SCENARIO_CATALOG
        if has_precomputed_data(idx, model_key)
    ]
    if not available:
        st.sidebar.error(
            f"No precomputed narration data found for **{model_key}**.\n\n"
            f"Run `python scripts/precompute_llm_narration.py` first."
        )
        return None

    labels = [label for _, label in available]
    chosen_label = st.sidebar.selectbox(
        "Scenario", labels, key="llm_narr__scenario_label",
    )
    return next(idx for idx, label in available if label == chosen_label)


# =============================================================================
# MAIN RENDER
# =============================================================================

def render() -> None:
    st.title("LLM Narration — Attention-Grounded XRL")
    st.caption(
        "Pure viewer. All narrations are precomputed offline; the UI selects which "
        "file to load. Slider position drives the displayed narration via forward-fill."
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Controls")

        driving_label = st.radio(
            "Driving model",
            options=list(DRIVING_MODELS.keys()),
            key="llm_narr__driving_label",
        )
        model_key = DRIVING_MODELS[driving_label]
        if model_key not in PLATFORM_MODELS:
            st.error(f"Driving model '{model_key}' missing from PLATFORM_MODELS.")
            return

    scenario_idx = _select_scenario(model_key)
    if scenario_idx is None:
        return

    with st.sidebar:
        compare_mode = st.checkbox(
            "Compare two LLMs side-by-side",
            value=False,
            key="llm_narr__compare",
        )

        toggle_grounding = st.toggle(
            "Attention grounding", value=True, key="llm_narr__tg",
        )
        toggle_counterfactual = st.toggle(
            "Counterfactual alternatives", value=True, key="llm_narr__tc",
        )
        toggle_key = toggle_key_from_flags(toggle_grounding, toggle_counterfactual)

        combos_available = set(
            available_narration_combos(scenario_idx, model_key)
        )
        llm_options = [
            k for k in LLM_MODELS.keys() if (k, toggle_key) in combos_available
        ]
        if not llm_options:
            st.warning(
                f"No precomputed narrations for toggle combo **{toggle_key}**. "
                f"Try a different toggle or run the precompute script."
            )
            return

        llm_key_a = st.selectbox(
            "LLM model",
            options=llm_options,
            format_func=lambda k: LLM_MODELS[k]["display"],
            key="llm_narr__llm_a",
        )
        llm_key_b = None
        if compare_mode:
            others = [k for k in llm_options if k != llm_key_a] or llm_options
            llm_key_b = st.selectbox(
                "LLM model (B)",
                options=others,
                format_func=lambda k: LLM_MODELS[k]["display"],
                key="llm_narr__llm_b",
            )

    # ── Load scenario artifact (for num_steps + BEV) ──────────────────────────
    artifact = load_artifact(model_key, scenario_idx)
    if artifact is None:
        st.error(
            f"Missing scenario artifact for {model_key} / scenario {scenario_idx}. "
            f"Run the post-hoc precompute script first."
        )
        return
    num_steps = artifact.num_steps

    # ── BEV player drives the timestep ────────────────────────────────────────
    step = render_bev_player(artifact, key_prefix="llm_narr")

    # ── Load narration & report lists (cached in session) ─────────────────────
    narr_a = _ensure_narrations(scenario_idx, model_key, llm_key_a, toggle_key, num_steps)
    reports = _ensure_reports(scenario_idx, model_key, num_steps)

    # ── Narration display (single or side-by-side) ────────────────────────────
    if compare_mode and llm_key_b is not None:
        narr_b = _ensure_narrations(scenario_idx, model_key, llm_key_b, toggle_key, num_steps)
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption(LLM_MODELS[llm_key_a]["display"])
            _render_narration_box(narr_a[step], header="A")
        with col_b:
            st.caption(LLM_MODELS[llm_key_b]["display"])
            _render_narration_box(narr_b[step], header="B")
    else:
        _render_narration_box(narr_a[step], header=LLM_MODELS[llm_key_a]["display"])

    # ── Expandable structured-report viewer ───────────────────────────────────
    report = reports[step]
    is_at_event = (
        report is not None and int(report.get("step", -1)) == step
    )
    label = "📄 Structured JSON report"
    if report is None:
        label += " (none yet)"
    elif not is_at_event:
        label += f" (last event @ step {report.get('step')})"
    with st.expander(label, expanded=False):
        if report is None:
            st.info("No report has been emitted yet at this timestep.")
        else:
            cols = st.columns(4)
            cols[0].metric("Necessity",      f"{report.get('necessity_score', 0):.2f}")
            grounding = report.get("attention_grounding", {}) or {}
            gscore = grounding.get("grounding_score")
            cols[1].metric("Grounding",      f"{gscore:.2f}" if gscore is not None else "—")
            cols[2].metric("Decision class", str(report.get("decision_class", "—")))
            n_alts = len(report.get("alternatives") or [])
            cols[3].metric("Alternatives",   str(n_alts))
            st.json(report)
