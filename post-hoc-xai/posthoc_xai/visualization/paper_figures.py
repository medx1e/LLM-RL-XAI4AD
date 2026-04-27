"""Publication-quality figure generation.

Reusable across all phases of the XAI study. All functions save PDF at 300 DPI
(vector-quality for thesis/paper inclusion) and optionally PNG for quick preview.

Usage:
    from posthoc_xai.visualization.paper_figures import (
        set_paper_style, plot_risk_stratified_correlation,
        plot_category_heatmap, plot_action_conditioned,
        plot_model_comparison, plot_correlation_distribution,
    )

    set_paper_style()  # call once at top of script
    fig = plot_risk_stratified_correlation(df, methods=["vg", "ig"], ...)
    save_figure(fig, "fig1_risk_corr", output_dir)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

# Colorblind-friendly palette (Wong 2011)
PALETTE = {
    "vg":            "#0072B2",   # blue
    "ig":            "#D55E00",   # vermillion
    "ig_mean":       "#E69F00",   # orange
    "gxi":           "#009E73",   # green
    "sarfa":         "#CC79A7",   # pink
    "attention":     "#56B4E9",   # sky blue
    "complete":      "#0072B2",
    "minimal":       "#D55E00",
    "basic":         "#009E73",
    # risk buckets
    "calm":          "#56B4E9",
    "moderate":      "#E69F00",
    "high":          "#D55E00",
    # categories
    "sdc_trajectory":"#4C72B0",
    "other_agents":  "#DD8452",
    "roadgraph":     "#55A868",
    "traffic_lights":"#C44E52",
    "gps_path":      "#8172B3",
}

CAT_LABELS = {
    "sdc_trajectory": "SDC",
    "other_agents":   "Agents",
    "roadgraph":      "Road",
    "traffic_lights": "TL",
    "gps_path":       "GPS",
}

METHOD_LABELS = {
    "vg":       "VG",
    "ig":       "IG (zero)",
    "ig_mean":  "IG (mean baseline)",
    "gxi":      "GxI",
    "sarfa":    "SARFA",
}

RISK_LABELS = {"calm": "Calm\n(risk<0.2)",
               "moderate": "Moderate\n(0.2–0.6)",
               "high": "High\n(risk>0.6)"}


def set_paper_style(font_size: int = 10):
    """Configure matplotlib for publication-quality output.

    Call once at the top of any script that generates paper figures.
    Uses a clean grid style with controlled font sizes.
    """
    sns.set_theme(style="whitegrid", font_scale=1.0)
    matplotlib.rcParams.update({
        "font.family":        "sans-serif",
        "font.size":          font_size,
        "axes.titlesize":     font_size + 1,
        "axes.labelsize":     font_size,
        "xtick.labelsize":    font_size - 1,
        "ytick.labelsize":    font_size - 1,
        "legend.fontsize":    font_size - 1,
        "figure.dpi":         150,      # screen preview
        "savefig.dpi":        300,      # saved files
        "savefig.bbox":       "tight",
        "pdf.fonttype":       42,       # embed fonts in PDF
        "ps.fonttype":        42,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
    })


def save_figure(
    fig: plt.Figure,
    name: str,
    output_dir: Path,
    formats: Sequence[str] = ("pdf", "png"),
):
    """Save figure in all requested formats.

    Args:
        fig:        Matplotlib figure to save.
        name:       Filename without extension (e.g. ``'fig1_risk_corr'``).
        output_dir: Directory to save into. Created if it doesn't exist.
        formats:    Iterable of format strings. Default: PDF + PNG.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 — Risk-stratified correlation
# ---------------------------------------------------------------------------

def plot_risk_stratified_correlation(
    data: pd.DataFrame,
    methods: Sequence[str],
    metric: str = "pearson_r",
    title: str = "Attention–Attribution Agreement by Risk Level",
    figsize: tuple = (7, 4),
) -> plt.Figure:
    """Grouped bar chart: mean ρ (or τ) per risk bucket per method.

    Args:
        data: DataFrame with columns: method, risk_bucket, and the metric column.
              Expected risk_bucket values: "calm", "moderate", "high".
        methods: Which methods to include, in display order.
        metric: Column name for the y-axis value.
        title: Figure title.
        figsize: Figure dimensions in inches.

    Returns:
        Matplotlib Figure (not yet saved — call save_figure).
    """
    risk_order  = ["calm", "moderate", "high"]
    n_risk      = len(risk_order)
    n_methods   = len(methods)
    x           = np.arange(n_risk)
    width       = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=figsize)

    for i, method in enumerate(methods):
        subset = data[data["method"] == method]
        means  = []
        sems   = []
        for rb in risk_order:
            vals = subset[subset["risk_bucket"] == rb][metric].dropna()
            means.append(vals.mean() if len(vals) else np.nan)
            sems.append(vals.sem()   if len(vals) > 1 else 0.0)

        offset = (i - n_methods / 2 + 0.5) * width
        color  = PALETTE.get(method, f"C{i}")
        label  = METHOD_LABELS.get(method, method)
        ax.bar(x + offset, means, width,
               yerr=sems, capsize=3,
               label=label, color=color, alpha=0.85, edgecolor="white")

    ax.axhline(0, color="gray", lw=0.7, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([RISK_LABELS[r] for r in risk_order])
    ax.set_ylabel("Mean Pearson ρ" if metric == "pearson_r" else f"Mean {metric}")
    ax.set_ylim(-0.5, 1.0)
    ax.set_title(title)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Per-category correlation heatmap
# ---------------------------------------------------------------------------

def plot_category_heatmap(
    data: pd.DataFrame,
    method: str,
    metric: str = "pearson_r",
    title: Optional[str] = None,
    figsize: tuple = (6, 3.5),
) -> plt.Figure:
    """Heatmap: categories (rows) × risk buckets (cols), color = ρ.

    Args:
        data: DataFrame with columns: category, risk_bucket, and metric.
        method: Method name (used in title if title is None).
        metric: Column for values.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    risk_order = ["calm", "moderate", "high"]
    cat_order  = list(CAT_LABELS.keys())
    subset     = data[data["method"] == method]

    # Build matrix
    matrix = np.full((len(cat_order), len(risk_order)), np.nan)
    for r, rb in enumerate(risk_order):
        for c, cat in enumerate(cat_order):
            vals = subset[(subset["risk_bucket"] == rb) &
                          (subset["category"] == cat)][metric].dropna()
            if len(vals):
                matrix[c, r] = vals.mean()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Pearson ρ", shrink=0.85)

    ax.set_xticks(range(len(risk_order)))
    ax.set_xticklabels([RISK_LABELS[r].replace("\n", " ") for r in risk_order])
    ax.set_yticks(range(len(cat_order)))
    ax.set_yticklabels([CAT_LABELS[c] for c in cat_order])

    # Annotate cells
    for r in range(len(risk_order)):
        for c in range(len(cat_order)):
            v = matrix[c, r]
            if not np.isnan(v):
                ax.text(r, c, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if abs(v) > 0.5 else "black")

    title = title or f"Category-level attention–{METHOD_LABELS.get(method, method)} ρ"
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Action-conditioned correlation
# ---------------------------------------------------------------------------

def plot_action_conditioned(
    data: pd.DataFrame,
    methods: Sequence[str],
    metric: str = "pearson_r",
    title: str = "Attention–Attribution Agreement by Action Type",
    figsize: tuple = (8, 4),
) -> plt.Figure:
    """Grouped bar chart: mean ρ per action type per method.

    Args:
        data: DataFrame with columns: method, action_type, metric, n.
    """
    action_order = ["braking", "accelerating", "steering", "neutral"]
    n_actions    = len(action_order)
    n_methods    = len(methods)
    x            = np.arange(n_actions)
    width        = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=figsize)

    for i, method in enumerate(methods):
        subset = data[data["method"] == method]
        means  = []
        sems   = []
        ns     = []
        for at in action_order:
            vals = subset[subset["action_type"] == at][metric].dropna()
            means.append(vals.mean() if len(vals) else np.nan)
            sems.append(vals.sem()   if len(vals) > 1 else 0.0)
            n_col = subset[subset["action_type"] == at]["n"]
            ns.append(int(n_col.sum()) if len(n_col) else 0)

        offset = (i - n_methods / 2 + 0.5) * width
        color  = PALETTE.get(method, f"C{i}")
        label  = METHOD_LABELS.get(method, method)
        bars   = ax.bar(x + offset, means, width, yerr=sems, capsize=3,
                        label=label, color=color, alpha=0.85, edgecolor="white")
        # Annotate n counts on first method only (to avoid clutter)
        if i == 0:
            for bar, n in zip(bars, ns):
                ax.text(bar.get_x() + bar.get_width() * n_methods / 2,
                        -0.06, f"n={n}", ha="center", va="top",
                        fontsize=7, color="gray")

    ax.axhline(0, color="gray", lw=0.7, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(action_order)
    ax.set_ylabel("Mean Pearson ρ")
    ax.set_ylim(-0.6, 1.0)
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Model comparison
# ---------------------------------------------------------------------------

def plot_model_comparison(
    data: pd.DataFrame,
    models: Sequence[str],
    method: str,
    metric: str = "pearson_r",
    stratify_by: str = "risk_bucket",
    title: Optional[str] = None,
    figsize: tuple = (7, 4),
) -> plt.Figure:
    """Grouped bar chart comparing models (complete vs minimal) for one method.

    Args:
        data: DataFrame with columns: model, stratify_by column, method, metric.
        models: Model names in display order.
        stratify_by: Column to stratify on (risk_bucket or action_type).
    """
    strat_values = (
        ["calm", "moderate", "high"]
        if stratify_by == "risk_bucket"
        else ["braking", "accelerating", "steering", "neutral"]
    )
    n_strat   = len(strat_values)
    n_models  = len(models)
    x         = np.arange(n_strat)
    width     = 0.8 / n_models

    fig, ax = plt.subplots(figsize=figsize)

    for i, model in enumerate(models):
        subset = data[(data["model"] == model) & (data["method"] == method)]
        means  = []
        sems   = []
        for sv in strat_values:
            vals = subset[subset[stratify_by] == sv][metric].dropna()
            means.append(vals.mean() if len(vals) else np.nan)
            sems.append(vals.sem()   if len(vals) > 1 else 0.0)

        offset = (i - n_models / 2 + 0.5) * width
        color  = PALETTE.get(model, f"C{i}")
        ax.bar(x + offset, means, width, yerr=sems, capsize=3,
               label=model, color=color, alpha=0.85, edgecolor="white")

    ax.axhline(0, color="gray", lw=0.7, linestyle="--")
    ax.set_xticks(x)
    strat_labels = (
        [RISK_LABELS[v].replace("\n", " ") for v in strat_values]
        if stratify_by == "risk_bucket" else strat_values
    )
    ax.set_xticklabels(strat_labels)
    ax.set_ylabel("Mean Pearson ρ")
    ax.set_ylim(-0.5, 1.0)
    title = title or f"Model comparison — {METHOD_LABELS.get(method, method)}"
    ax.set_title(title)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 5 — Per-scenario correlation distribution
# ---------------------------------------------------------------------------

def plot_correlation_distribution(
    data: pd.DataFrame,
    methods: Sequence[str],
    metric: str = "pearson_r",
    title: str = "Distribution of per-scenario attention–attribution ρ",
    figsize: tuple = (7, 4),
) -> plt.Figure:
    """Violin plot: distribution of per-scenario ρ for each method.

    Args:
        data: DataFrame with columns: method, scenario_id, metric.
    """
    fig, ax = plt.subplots(figsize=figsize)

    plot_data = []
    labels    = []
    for method in methods:
        vals = data[data["method"] == method][metric].dropna().values
        plot_data.append(vals)
        labels.append(METHOD_LABELS.get(method, method))

    parts = ax.violinplot(plot_data, positions=range(len(methods)),
                          showmedians=True, showextrema=True)

    for i, (pc, method) in enumerate(zip(parts["bodies"], methods)):
        pc.set_facecolor(PALETTE.get(method, f"C{i}"))
        pc.set_alpha(0.7)

    ax.axhline(0, color="gray", lw=0.7, linestyle="--", label="ρ=0")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Per-scenario Pearson ρ")
    ax.set_ylim(-1, 1)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 6 — Temporal attention-attribution series (qualitative)
# ---------------------------------------------------------------------------

def plot_temporal_series(
    timesteps: Sequence[int],
    attention_series: dict[str, list],
    method_series: dict[str, list],
    method_name: str,
    risk_series: Optional[list] = None,
    title: str = "",
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Dual-panel: top = attention vs attribution over time per category,
    bottom = collision risk.

    Args:
        timesteps: List of timestep indices.
        attention_series: {category: [values over T]}.
        method_series: Same structure for the attribution method.
        risk_series: Optional collision_risk over T.
    """
    n_panels = 2 if risk_series is not None else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize,
                             sharex=True, gridspec_kw={"height_ratios": [3, 1]} if n_panels == 2 else None)
    ax = axes[0] if n_panels == 2 else axes

    for cat in CAT_LABELS:
        col   = PALETTE.get(cat, "gray")
        label = CAT_LABELS[cat]
        a_vals = attention_series.get(cat, [])
        m_vals = method_series.get(cat, [])
        if a_vals:
            ax.plot(timesteps, a_vals, color=col, lw=1.5, label=f"{label} (attn)")
        if m_vals:
            ax.plot(timesteps, m_vals, color=col, lw=1.5, linestyle="--",
                    label=f"{label} ({METHOD_LABELS.get(method_name, method_name)})")

    ax.set_ylabel("Category importance")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.set_title(title or f"Attention vs {METHOD_LABELS.get(method_name, method_name)}")

    if n_panels == 2 and risk_series is not None:
        ax2 = axes[1]
        ax2.plot(timesteps, risk_series, color=PALETTE["high"], lw=1.5)
        ax2.fill_between(timesteps, 0, risk_series, alpha=0.2,
                         color=PALETTE["high"])
        ax2.set_ylabel("Collision risk")
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("Timestep")

    fig.tight_layout()
    return fig
