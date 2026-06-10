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
import yaml

import platform  # path bootstrap (idempotent)
from platform.shared.bev_component import (
    is_bev_playing,
    render_bev_player,
    schedule_bev_rerun,
)
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

# Path to the LLM-narration YAML config (registry + active LLM + toggle combos).
# Mirrors the spirit of llm_integration/xai/config/xai_config.yaml: a single
# file the precompute script and the platform both read.
LLM_CONFIG_PATH: Path = _PROJECT_ROOT / "config" / "llm_narration.yaml"


def _load_llm_config(path: Path = LLM_CONFIG_PATH) -> dict:
    """Read the LLM narration YAML. Returns an empty config on missing file."""
    if not path.is_file():
        return {"active": None, "llms": {}, "toggle_combos": []}
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh) or {}
    cfg.setdefault("active", None)
    cfg.setdefault("llms", {})
    cfg.setdefault("toggle_combos", [])
    return cfg


_LLM_CONFIG: dict = _load_llm_config()

# LLM registry — loaded from config/llm_narration.yaml::llms. Each value
# mirrors the `llm:` sub-dict expected by llm_narrator.py (provider, model,
# base_url, api_key_env, max_tokens, enable_thinking), plus a UI `display`.
LLM_MODELS: dict[str, dict] = _LLM_CONFIG["llms"]

# Default LLM key for a new precompute run (overridable via --llm). Falls
# back to the first registry entry if `active:` is missing or wrong.
ACTIVE_LLM_KEY: Optional[str] = (
    _LLM_CONFIG["active"]
    if _LLM_CONFIG["active"] in LLM_MODELS
    else (next(iter(LLM_MODELS), None))
)

# Toggle combinations. Filesystem keys (`<key>__<toggle>.json`) MUST match
# the keys here; the dict is the source of truth for which combos exist.
TOGGLE_COMBOS: dict[str, dict] = {
    "full":              {"grounding": True,  "counterfactual": True},
    "no_grounding":      {"grounding": False, "counterfactual": True},
    "no_counterfactual": {"grounding": True,  "counterfactual": False},
    "minimal":           {"grounding": False, "counterfactual": False},
}

# Subset of TOGGLE_COMBOS the next precompute run materializes. Comes from
# the YAML's `toggle_combos:` list; falls back to every combo we know.
ACTIVE_TOGGLE_COMBOS: list[str] = [
    k for k in _LLM_CONFIG["toggle_combos"] if k in TOGGLE_COMBOS
] or list(TOGGLE_COMBOS.keys())

# Per-LLM evaluation metrics (hardcoded from DeepEval G-Eval benchmark).
# Keys must match the LLM registry keys in config/llm_narration.yaml.
# Order: Overall Cognition Score first, then the three sub-dimensions.
LLM_EVAL_METRICS: dict[str, dict[str, float]] = {
    "glm":   {"Overall Cognition Score": 0.892, "Situational Awareness": 0.931, "Reasoning": 0.858, "Communication": 0.882},
    "gemma": {"Overall Cognition Score": 0.827, "Situational Awareness": 0.906, "Reasoning": 0.739, "Communication": 0.844},
    "qwen":  {"Overall Cognition Score": 0.777, "Situational Awareness": 0.803, "Reasoning": 0.746, "Communication": 0.788},
}

_EVAL_METRIC_COLORS: dict[str, str] = {
    "Overall Cognition Score": "#10b981",
    "Situational Awareness": "#f59e0b",
    "Reasoning": "#818cf8",
    "Communication": "#38bdf8",
}


def _render_llm_eval_card(llm_key: str) -> None:
    """Render evaluation metric rows for a single LLM, labelled with model name."""
    metrics = LLM_EVAL_METRICS.get(llm_key)
    if not metrics:
        return
    display_name = LLM_MODELS.get(llm_key, {}).get("display", llm_key)
    rows = ""
    for name, value in metrics.items():
        color = _EVAL_METRIC_COLORS.get(name, "#a7a8b3")
        # Overall Cognition Score gets a slightly larger, highlighted style
        if name == "Overall Cognition Score":
            rows += (
                f"<div style='padding:7px 10px;background:#1a1d2e;border:1px solid {color};"
                f"border-radius:8px;display:flex;justify-content:space-between;align-items:center;'>"
                f"<span style='font-size:12px;font-weight:600;color:#e2e2e8;'>{name}</span>"
                f"<span style='font-size:14px;font-weight:800;color:{color};'>{value:.3f}</span></div>"
            )
        else:
            rows += (
                f"<div style='padding:5px 10px;background:#1a1d2e;border:1px solid #3b3f55;"
                f"border-radius:8px;display:flex;justify-content:space-between;align-items:center;'>"
                f"<span style='font-size:11px;color:#a7a8b3;'>{name}</span>"
                f"<span style='font-size:12px;font-weight:700;color:{color};'>{value:.3f}</span></div>"
            )
    st.markdown(
        f"<div style='margin-bottom:8px;'>"
        f"<div style='font-size:10px;font-weight:700;color:#6f6ae8;"
        f"letter-spacing:0.05em;margin-bottom:4px;'>{display_name}</div>"
        f"<div style='display:flex;flex-direction:column;gap:4px;'>{rows}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


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


def _streaming_variant(slot_key: str, timestep: Optional[int]) -> int:
    """Return a 0/1 flag that flips each time ``timestep`` changes for a slot.

    The streaming narration card swaps its animation-name on this flag to
    restart the token reveal on every new narration. The value is stable while
    a narration is forward-filled (``timestep`` unchanged), so the card HTML
    stays byte-identical across BEV reruns and the reveal does not re-trigger.
    """
    state_key = f"{slot_key}__seq"
    last_ts, count = st.session_state.get(state_key, (None, 0))
    if timestep != last_ts:
        count += 1
        st.session_state[state_key] = (timestep, count)
    return count % 2


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

def _render_narration_box(
    entry: Optional[NarrationEntry],
    header: str = "Narration",
    streaming: bool = False,
    variant: int = 0,
) -> None:
    """Tone-coloured narration card using the shared HTML component.

    When ``streaming`` is True the body reveals token-by-token (demo flourish).
    ``variant`` flips per new narration so the reveal restarts each time; see
    ``_streaming_variant`` and ``narration_card_streaming``.
    """
    from platform.shared.html_components import (
        narration_card,
        narration_card_streaming,
        empty_state,
    )
    if entry is None:
        st.markdown(
            empty_state("No narration available yet at this timestep."),
            unsafe_allow_html=True,
        )
        return
    rt_ms = (entry.response_time_s * 1000) if entry.response_time_s is not None else None
    if streaming:
        html = narration_card_streaming(
            model_name=header,
            subtitle=f"t = {entry.timestamp_sec:.1f}s · {entry.tone}",
            tone=entry.tone,
            text=entry.text,
            step=entry.timestep,
            response_time_ms=rt_ms,
            variant=variant,
        )
    else:
        html = narration_card(
            model_name=header,
            subtitle=f"t = {entry.timestamp_sec:.1f}s · {entry.tone}",
            tone=entry.tone,
            text=entry.text,
            step=entry.timestep,
            response_time_ms=rt_ms,
        )
    st.markdown(html, unsafe_allow_html=True)


def _select_scenario(model_key: str) -> Optional[int]:
    """In-page scenario selector, filtered to scenarios with precomputed data."""
    available = [
        (idx, label) for idx, label in SCENARIO_CATALOG
        if has_precomputed_data(idx, model_key)
    ]
    if not available:
        st.warning(
            f"No precomputed narration data found for {model_key}. "
            f"Run scripts/precompute_llm_narration.py first."
        )
        return None

    labels = [label for _, label in available]
    chosen_label = st.selectbox(
        "Scenario", labels, key="llm_narr__scenario_label",
    )
    return next(idx for idx, label in available if label == chosen_label)


# =============================================================================
# MAIN RENDER
# =============================================================================

def render() -> None:
    from platform.shared.html_components import context_strip, empty_state

    # ── Layout: 260px control rail + flexible content area ───────────────────
    rail, content = st.columns([1, 4], gap="medium")

    with rail:
        driving_label = "LQ (Perceiver)"
        model_key = DRIVING_MODELS[driving_label]

        st.markdown(
            "<div class='xai-rail' style='background:#25283a;border:1px solid #3b3f55;"
            "border-radius:12px;padding:14px;margin-top:8px;"
            "box-shadow:0 1px 0 rgba(255,255,255,0.02) inset,0 8px 24px rgba(0,0,0,0.25);'>"
            "<div style='font-size:11px;font-weight:700;text-transform:uppercase;"
            "letter-spacing:0.12em;color:#a7a8b3;margin-bottom:8px;'>Driving Model </div>"
            f"<span style='font-size:13px;font-weight:600;color:#f4f4f7;'>{driving_label}</span>",
            unsafe_allow_html=True,
        )
        # Fixed model — no selector
        
       

        if model_key not in PLATFORM_MODELS:
            with content:
                st.markdown(context_strip("LLM Narration"), unsafe_allow_html=True)
                st.markdown(empty_state(f"Driving model '{model_key}' missing from catalog."),
                            unsafe_allow_html=True)
            return

        st.markdown(
            "<div style='font-size:11px;font-weight:700;text-transform:uppercase;"
            "letter-spacing:0.12em;color:#a7a8b3;margin:14px 0 8px;'>Scenario</div>",
            unsafe_allow_html=True,
        )
        scenario_idx = _select_scenario(model_key)
        if scenario_idx is None:
            with content:
                st.markdown(context_strip("LLM Narration"), unsafe_allow_html=True)
                st.markdown(
                    empty_state(
                        f"No precomputed narration data for {model_key}.",
                        icon="✎",
                        command="python scripts/precompute_llm_narration.py",
                    ),
                    unsafe_allow_html=True,
                )
            return

        st.markdown(
            "<div style='font-size:11px;font-weight:700;text-transform:uppercase;"
            "letter-spacing:0.12em;color:#a7a8b3;margin:14px 0 8px;'>Mode</div>",
            unsafe_allow_html=True,
        )
        compare_mode = st.checkbox(
            "Compare two LLMs", value=False, key="llm_narr__compare",
        )
        toggle_grounding = st.toggle(
            "Attention grounding", value=True, key="llm_narr__tg",
        )
        toggle_counterfactual = st.toggle(
            "Counterfactual alternatives", value=True, key="llm_narr__tc",
        )
        toggle_key = toggle_key_from_flags(toggle_grounding, toggle_counterfactual)
        stream_effect = True

        combos_available = set(available_narration_combos(scenario_idx, model_key))
        llm_options = [k for k in LLM_MODELS if (k, toggle_key) in combos_available]
        if not llm_options:
            with content:
                st.markdown(context_strip("LLM Narration"), unsafe_allow_html=True)
                st.markdown(
                    empty_state(
                        f"No precomputed narrations for toggle combo '{toggle_key}'.",
                        icon="✎",
                        hint="Try a different toggle or run the precompute script.",
                    ),
                    unsafe_allow_html=True,
                )
            return

        st.markdown(
            "<div style='font-size:11px;font-weight:700;text-transform:uppercase;"
            "letter-spacing:0.12em;color:#a7a8b3;margin:14px 0 8px;'>LLM</div>",
            unsafe_allow_html=True,
        )
        llm_key_a = st.selectbox(
            "LLM model A",
            options=llm_options,
            format_func=lambda k: LLM_MODELS[k]["display"],
            key="llm_narr__llm_a",
            label_visibility="collapsed",
        )
        llm_key_b = None
        if compare_mode:
            others = [k for k in llm_options if k != llm_key_a] or llm_options
            llm_key_b = st.selectbox(
                "LLM model B",
                options=others,
                format_func=lambda k: LLM_MODELS[k]["display"],
                key="llm_narr__llm_b",
            )

        # Dynamic LLM evaluation metrics (updates with selected LLM)
        has_eval_a = llm_key_a in LLM_EVAL_METRICS
        has_eval_b = compare_mode and llm_key_b is not None and llm_key_b in LLM_EVAL_METRICS
        if has_eval_a or has_eval_b:
            st.markdown(
                "<div style='font-size:11px;font-weight:700;text-transform:uppercase;"
                "letter-spacing:0.12em;color:#a7a8b3;margin:14px 0 6px;'>LLM Evaluation</div>",
                unsafe_allow_html=True,
            )
            if has_eval_a:
                _render_llm_eval_card(llm_key_a)
            if has_eval_b:
                _render_llm_eval_card(llm_key_b)

    # ── Content column ────────────────────────────────────────────────────────
    with content:
        st.markdown(
            context_strip("LLM Narration",
                          model_label=LLM_MODELS[llm_key_a]["display"],
                          scenario_label=f"Scenario {scenario_idx}"),
            unsafe_allow_html=True,
        )

        artifact = load_artifact(model_key, scenario_idx)
        if artifact is None:
            st.markdown(
                empty_state(
                    f"Missing scenario artifact for {model_key} / scenario {scenario_idx}.",
                    command="python scripts/precompute_posthoc_demo.py",
                ),
                unsafe_allow_html=True,
            )
            return
        num_steps = artifact.num_steps

        # Top strip: BEV (60%) + quick stats (40%)
        col_bev, col_stats = st.columns([3, 2], gap="medium")
        with col_bev:
            step = render_bev_player(artifact, key_prefix="llm_narr", height=450)

        narr_a = _ensure_narrations(scenario_idx, model_key, llm_key_a, toggle_key, num_steps)
        narr_b = None
        if compare_mode and llm_key_b is not None:
            narr_b = _ensure_narrations(scenario_idx, model_key, llm_key_b, toggle_key, num_steps)
        reports = _ensure_reports(scenario_idx, model_key, num_steps)

        with col_stats:
            non_null_narr = len({n.timestep for n in narr_a if n is not None})
            non_null_rep  = len({r.get("step") for r in reports if r is not None})
            
            seen_rt_a = [
                n.response_time_s * 1000
                for n in narr_a
                if n is not None and n.response_time_s is not None and n.response_time_s > 0
            ]
            avg_rt_ms_a = sum(seen_rt_a) / len(seen_rt_a) if seen_rt_a else None

            st.markdown(
                "<div style='font-size:11px;font-weight:700;text-transform:uppercase;"
                "letter-spacing:0.12em;color:#a7a8b3;margin:6px 0 8px;'>Quick stats</div>",
                unsafe_allow_html=True,
            )

            if compare_mode and narr_b is not None:
                seen_rt_b = [
                    n.response_time_s * 1000
                    for n in narr_b
                    if n is not None and n.response_time_s is not None and n.response_time_s > 0
                ]
                avg_rt_ms_b = sum(seen_rt_b) / len(seen_rt_b) if seen_rt_b else None

                s1, s2, s3 = st.columns(3)
                s1.metric("Narrations", str(non_null_narr))
                s2.metric("Reports",    str(non_null_rep))
                s3.metric("Current t",   str(step))

                s4, s5 = st.columns(2)
                display_a = LLM_MODELS[llm_key_a]["display"]
                display_b = LLM_MODELS[llm_key_b]["display"]
                s4.metric(f"Avg latency ({display_a})", f"{avg_rt_ms_a:.0f} ms" if avg_rt_ms_a else "—")
                s5.metric(f"Avg latency ({display_b})", f"{avg_rt_ms_b:.0f} ms" if avg_rt_ms_b else "—")
            else:
                s1, s2 = st.columns(2)
                s1.metric("Narrations", str(non_null_narr))
                s2.metric("Reports",    str(non_null_rep))
                s3, s4 = st.columns(2)
                s3.metric("Avg latency", f"{avg_rt_ms_a:.0f} ms" if avg_rt_ms_a else "—")
                s4.metric("Current t",   str(step))

        # Narration cards (2-col on compare, 1-col otherwise)
        st.markdown(
            "<h3 style='font-size:16px;font-weight:700;color:#f4f4f7;"
            "font-family:Inter,sans-serif;margin:14px 0 10px;'>Narration</h3>",
            unsafe_allow_html=True,
        )
        # Stream the token reveal only while the episode is playing, so it does
        # not fire on initial page load / manual scrubbing.
        streaming = bool(stream_effect) and is_bev_playing(artifact, "llm_narr")

        entry_a = narr_a[step]
        slot_a = f"llm_narr__var__{model_key}__{scenario_idx}__{llm_key_a}"
        variant_a = _streaming_variant(slot_a, entry_a.timestep if entry_a else None)

        if compare_mode and llm_key_b is not None:
            entry_b = narr_b[step]
            slot_b = f"llm_narr__var__{model_key}__{scenario_idx}__{llm_key_b}"
            variant_b = _streaming_variant(slot_b, entry_b.timestep if entry_b else None)
            col_a, col_b = st.columns(2, gap="medium")
            with col_a:
                _render_narration_box(
                    entry_a, header=LLM_MODELS[llm_key_a]["display"],
                    streaming=streaming, variant=variant_a,
                )
            with col_b:
                _render_narration_box(
                    entry_b, header=LLM_MODELS[llm_key_b]["display"],
                    streaming=streaming, variant=variant_b,
                )
        else:
            _render_narration_box(
                entry_a, header=LLM_MODELS[llm_key_a]["display"],
                streaming=streaming, variant=variant_a,
            )

        # Structured report (expander)
        report = reports[step]
        is_at_event = (report is not None and int(report.get("step", -1)) == step)
        label = "Structured JSON report"
        if report is None:
            label += " (none yet)"
        elif not is_at_event:
            label += f" (last event @ step {report.get('step')})"
        with st.expander(label, expanded=False):
            if report is None:
                st.markdown(empty_state("No report emitted yet at this timestep."),
                            unsafe_allow_html=True)
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

    # Trigger rerun for BEV playback AFTER all content is rendered so the
    # narration cards and stats update every frame, not just when paused.
    schedule_bev_rerun()
