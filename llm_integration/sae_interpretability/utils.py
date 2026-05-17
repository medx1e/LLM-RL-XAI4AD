"""Utility helpers for SAE interpretability figures.

Provides a single ``save_figure`` helper that writes a matplotlib figure in
both PNG (high-resolution raster) and PDF (lossless vector) formats.

Usage::

    from sae_interpretability.utils import save_figure

    fig, ax = plt.subplots(...)
    # ... populate figure ...
    save_figure(fig, out_dir, "my_plot")
    # Writes: <out_dir>/my_plot.png  and  <out_dir>/my_plot.pdf
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# PhD-level figure style defaults
# ---------------------------------------------------------------------------

# Recommended: call apply_phd_style() once at the top of any plotting module.
def apply_phd_style() -> None:
    """Apply publication-quality matplotlib rcParams globally.

    Sets fonts, DPI, line widths, and tick formatting to levels appropriate
    for a PhD thesis or IEEE/NeurIPS paper.  Call once per process.
    """
    matplotlib.rcParams.update({
        # ── Typography ─────────────────────────────────────────────────────
        "font.family":        "serif",
        "font.serif":         ["DejaVu Serif", "Times New Roman", "Palatino"],
        "font.size":          11,
        "axes.titlesize":     13,
        "axes.labelsize":     11,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "legend.framealpha":  0.9,
        # ── Resolution ────────────────────────────────────────────────────
        "figure.dpi":         150,
        "savefig.dpi":        300,
        # ── Lines & patches ───────────────────────────────────────────────
        "lines.linewidth":    1.8,
        "patch.linewidth":    0.6,
        "axes.linewidth":     0.8,
        "grid.linewidth":     0.5,
        "grid.alpha":         0.4,
        # ── Layout ────────────────────────────────────────────────────────
        "figure.autolayout":  False,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        # ── Colour cycle (ColorBrewer qualitative palette) ─────────────────
        "axes.prop_cycle": matplotlib.cycler(color=[
            "#2166AC", "#D6604D", "#4DAC26", "#8073AC",
            "#F4A582", "#92C5DE", "#A6DBA0", "#E0E0E0",
        ]),
    })


# PhD colour constants (reusable across all plotting modules)
PHD_BLUE   = "#2166AC"
PHD_ORANGE = "#D6604D"
PHD_GREEN  = "#4DAC26"
PHD_PURPLE = "#8073AC"
PHD_TEAL   = "#01665E"
PHD_GRAY   = "#636363"

PHD_DIVERGING = "RdBu_r"   # for correlation heat-maps
PHD_SEQUENTIAL = "viridis"  # for magnitude-only maps
PHD_MAKO = "mako"           # for bar-chart palettes


# ---------------------------------------------------------------------------
# Core save helper
# ---------------------------------------------------------------------------

def save_figure(
    fig: plt.Figure,
    out_dir: Union[str, Path],
    stem: str,
    *,
    dpi: int = 300,
    bbox_inches: str = "tight",
    pad_inches: float = 0.05,
) -> None:
    """Save *fig* as both ``<stem>.png`` and ``<stem>.pdf`` inside *out_dir*.

    Args:
        fig:         The matplotlib Figure to save.
        out_dir:     Destination directory (created if it does not exist).
        stem:        Filename stem without extension, e.g. ``"best_per_concept"``.
        dpi:         Raster resolution for the PNG output (default 300 dpi).
        bbox_inches: Bounding-box clipping mode (default ``"tight"``).
        pad_inches:  Padding around the tight bounding box (default 0.05).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    common_kwargs = dict(bbox_inches=bbox_inches, pad_inches=pad_inches)

    pdf_path = out_dir / f"{stem}.pdf"

    fig.savefig(pdf_path, **common_kwargs)   # PDF is always vector

    print(f"[Figure] Saved → {pdf_path}")


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create *path* (and all parents) if it does not exist; return a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def _json_default(obj):
    """JSON serialiser for numpy types."""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_incremental_results(
    output_path: str,
    feature_name: str,
    feature_data: Dict[str, Any],
    all_features_meta: Dict[str, int],
    temperatures: List[float],
    n_scenarios: int,
    max_steps: int,
) -> None:
    """Incrementally update the JSON results file with a new feature."""
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                out_data = json.load(f)
        except json.JSONDecodeError:
            out_data = {}
    else:
        out_data = {}

    if 'features' not in out_data:
        out_data['features'] = all_features_meta
        out_data['temperatures'] = temperatures
        out_data['n_scenarios'] = n_scenarios
        out_data['max_steps'] = max_steps

    if 'results' not in out_data:
        out_data['results'] = {}

    out_data['results'][feature_name] = feature_data

    # Write to a temporary file and rename to avoid corruption if interrupted
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, 'w') as f:
        json.dump(out_data, f, indent=2, default=_json_default)
    os.replace(tmp_path, output_path)
