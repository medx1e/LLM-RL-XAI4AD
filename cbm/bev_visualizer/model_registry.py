"""
Model Registry for the BEV Visualizer.

This is the ONLY file you need to edit to register a new model.
Add an entry to MODEL_REGISTRY with the model name as the key.

Supported model types:
  - "sac" : A pretrained V-Max SAC baseline. Requires a 'run_dir'.
  - "cbm" : A CBM checkpoint. Requires 'checkpoint' + 'pretrained_dir'.

All paths are relative to the project root (the directory containing this
bev_visualizer/ folder). Use absolute paths if you prefer.
"""

from pathlib import Path

# ── Project root (one level up from this file) ──────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent


def _p(rel: str) -> str:
    """Resolve a path relative to the project root."""
    return str(_ROOT / rel)


# ── The Registry ─────────────────────────────────────────────────────────────
MODEL_REGISTRY: dict[str, dict] = {
    # ── SAC Baselines (Complete) ──────────────────────────────────────────
    "SAC Complete (WOMD seed 42)": {
        "type": "sac",
        "run_dir": _p("runs_rlc/womd_sac_road_perceiver_complete_42"),
        "description": "Full-capacity SAC baseline, seed 42.",
    },
    "SAC Complete (WOMD seed 69)": {
        "type": "sac",
        "run_dir": _p("runs_rlc/womd_sac_road_perceiver_complete_69"),
        "description": "Full-capacity SAC baseline, seed 69.",
    },
    "SAC Complete (WOMD seed 99)": {
        "type": "sac",
        "run_dir": _p("runs_rlc/womd_sac_road_perceiver_complete_99"),
        "description": "Full-capacity SAC baseline, seed 99.",
    },
    # ── SAC Baselines (Minimal) ───────────────────────────────────────────
    "SAC Minimal (WOMD seed 42)": {
        "type": "sac",
        "run_dir": _p("runs_rlc/womd_sac_road_perceiver_minimal_42"),
        "description": "Minimal SAC baseline, seed 42. (Faster load)",
    },
    "SAC Minimal (WOMD seed 69)": {
        "type": "sac",
        "run_dir": _p("runs_rlc/womd_sac_road_perceiver_minimal_69"),
        "description": "Minimal SAC baseline, seed 69.",
    },
    "SAC Minimal (WOMD seed 99)": {
        "type": "sac",
        "run_dir": _p("runs_rlc/womd_sac_road_perceiver_minimal_99"),
        "description": "Minimal SAC baseline, seed 99.",
    },
    # ── CBM Models ────────────────────────────────────────────────────────
    "CBM-V2 Frozen — 150GB (15 concepts)": {
        "type": "cbm",
        "checkpoint": _p("cbm_v2_frozen_womd_150gb/checkpoints/model_final.pkl"),
        "pretrained_dir": _p("runs_rlc/womd_sac_road_perceiver_minimal_42"),
        "num_concepts": 15,
        "concept_phases": [1, 2, 3],
        "description": (
            "CBM-V2 with 15 concepts (Phases 1-3), frozen encoder, "
            "trained on 150GB WOMD. 86% task accuracy, 96.3% binary concept accuracy."
        ),
    },
    # ── Add new models here ───────────────────────────────────────────────
    # Example:
    # "My New Model": {
    #     "type": "sac",
    #     "run_dir": _p("runs_rlc/my_new_run"),
    #     "description": "Brief description for the UI.",
    # },
}
