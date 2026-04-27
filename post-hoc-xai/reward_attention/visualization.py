"""Publication-ready figures for reward-conditioned attention analysis.

Figures:
  fig1_scatter_*.png     — scatter (risk vs attn) + regression + r/p annotation
  fig2_correlation_heatmap.png — full matrix heatmap (RdBu_r)
  fig3_temporal_*.png    — attention trajectories over event windows
  fig4_action_attention.png   — grouped bar chart: attention by action type
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats


# ---------------------------------------------------------------------------
# Paper style
# ---------------------------------------------------------------------------


def set_paper_style() -> None:
    """Apply publication-ready matplotlib rcParams."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


# ---------------------------------------------------------------------------
# Category display names and colors
# ---------------------------------------------------------------------------

ATTN_LABELS = {
    "attn_sdc":       "Ego (SDC)",
    "attn_agents":    "Other Agents",
    "attn_roadgraph": "Road Graph",
    "attn_lights":    "Traffic Lights",
    "attn_gps":       "GPS Path",
}

ATTN_COLORS = {
    "attn_sdc":       "#4C72B0",
    "attn_agents":    "#DD8452",
    "attn_roadgraph": "#55A868",
    "attn_lights":    "#C44E52",
    "attn_gps":       "#8172B2",
}

RISK_LABELS = {
    "collision_risk":  "Collision Risk",
    "safety_risk":     "Safety Risk",
    "navigation_risk": "Navigation Risk",
    "behavior_risk":   "Behavior Risk",
}


# ---------------------------------------------------------------------------
# Fig 1: Scatter plot (risk vs attention)
# ---------------------------------------------------------------------------


def plot_scatter(
    df: pd.DataFrame,
    risk_col: str,
    attn_col: str,
    title: str = "",
    save_path: str | Path | None = None,
    alpha: float = 0.15,
    max_points: int = 5000,
) -> plt.Figure:
    """Scatter plot: risk_col (x) vs attn_col (y) with regression line.

    Args:
        df: DataFrame with columns risk_col and attn_col.
        risk_col: X-axis variable (risk metric).
        attn_col: Y-axis variable (attention category).
        title: Figure title.
        save_path: If given, save the figure here.
        alpha: Point transparency.
        max_points: Subsample if more than this many points.

    Returns:
        matplotlib Figure.
    """
    set_paper_style()

    data = df[[risk_col, attn_col]].dropna()
    mask = np.isfinite(data[risk_col]) & np.isfinite(data[attn_col])
    data = data[mask]

    if len(data) == 0:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return fig

    if len(data) > max_points:
        data = data.sample(n=max_points, random_state=42)

    x = data[risk_col].values
    y = data[attn_col].values

    # Skip constant variables (correlation undefined)
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.text(0.5, 0.5, "Variable is constant\n(no variation in data)",
                transform=ax.transAxes, ha="center", va="center", color="gray")
        ax.set_title(title or f"{RISK_LABELS.get(risk_col, risk_col)} vs {ATTN_LABELS.get(attn_col, attn_col)}")
        if save_path:
            fig.savefig(save_path)
            plt.close(fig)
        return fig

    # Correlations
    with np.errstate(invalid="ignore"):
        rho, p_sp = stats.spearmanr(x, y)
        r, p_pe = stats.pearsonr(x, y)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    color = ATTN_COLORS.get(attn_col, "#555555")

    ax.scatter(x, y, alpha=alpha, s=6, color=color, rasterized=True)

    # Regression line
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, m * x_line + b, color="crimson", linewidth=1.5, zorder=5)

    # Annotation
    sig_str = "**" if p_sp < 0.01 else ("*" if p_sp < 0.05 else "")
    ax.text(
        0.97, 0.97,
        f"ρ={rho:+.3f}{sig_str}\np={p_sp:.3f}\nn={len(x):,}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
    )

    ax.set_xlabel(RISK_LABELS.get(risk_col, risk_col))
    ax.set_ylabel(ATTN_LABELS.get(attn_col, attn_col) + " attention")
    ax.set_title(title or f"{RISK_LABELS.get(risk_col, risk_col)} vs {ATTN_LABELS.get(attn_col, attn_col)}")

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Fig 2: Correlation heatmap
# ---------------------------------------------------------------------------


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    save_path: str | Path | None = None,
    title: str = "Spearman Correlation: Risk × Attention",
) -> plt.Figure:
    """Heatmap of full correlation matrix.

    Args:
        corr_matrix: DataFrame with risk_cols as index, attn_cols as columns.
        save_path: If given, save the figure here.
        title: Figure title.

    Returns:
        matplotlib Figure.
    """
    try:
        import seaborn as sns
    except ImportError:
        sns = None

    set_paper_style()

    # Rename for display
    row_labels = [RISK_LABELS.get(c, c) for c in corr_matrix.index]
    col_labels = [ATTN_LABELS.get(c, c) for c in corr_matrix.columns]

    matrix = corr_matrix.values.astype(float)

    fig, ax = plt.subplots(
        figsize=(max(6, len(col_labels) * 1.0), max(3, len(row_labels) * 0.8))
    )

    if sns is not None:
        display_df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
        sns.heatmap(
            display_df,
            ax=ax,
            cmap="RdBu_r",
            vmin=-1, vmax=1,
            center=0,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"label": "Spearman ρ", "shrink": 0.8},
        )
    else:
        # Fallback: plain matplotlib
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.8)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=30, ha="right")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)

    ax.set_title(title, pad=12)

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Fig 3: Temporal event plot
# ---------------------------------------------------------------------------


def plot_temporal_event(
    traj_df: pd.DataFrame,
    event_peak: int,
    save_path: str | Path | None = None,
    title: str = "Attention & Risk Over Event Window",
) -> plt.Figure:
    """Dual-axis time series: attention categories + safety risk.

    Args:
        traj_df: DataFrame with relative_t, attn_* columns, safety_risk.
        event_peak: Peak timestep (absolute) for vertical line annotation.
        save_path: If given, save the figure here.
        title: Figure title.

    Returns:
        matplotlib Figure.
    """
    set_paper_style()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    t = traj_df["relative_t"].values

    # Upper panel: attention categories
    for col, label in ATTN_LABELS.items():
        if col in traj_df.columns:
            ax1.plot(t, traj_df[col].values, label=label,
                     color=ATTN_COLORS[col], linewidth=1.5)

    ax1.axvline(0, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="Event peak")
    ax1.set_ylabel("Attention (fraction)")
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.8)
    ax1.set_title(title)

    # Lower panel: safety risk
    if "safety_risk" in traj_df.columns:
        ax2.fill_between(t, traj_df["safety_risk"].values, alpha=0.4, color="crimson")
        ax2.plot(t, traj_df["safety_risk"].values, color="crimson", linewidth=1.2)
    if "collision_risk" in traj_df.columns:
        ax2.plot(t, traj_df["collision_risk"].values, "--", color="darkorange",
                 linewidth=1.0, alpha=0.8, label="Collision risk")
    ax2.axvline(0, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.set_ylabel("Risk")
    ax2.set_xlabel("Timestep relative to event peak")
    ax2.set_ylim(-0.02, 1.12)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Fig 4: Action-conditioned attention bar chart
# ---------------------------------------------------------------------------


def plot_action_conditioned(
    action_attn_df: pd.DataFrame,
    save_path: str | Path | None = None,
    title: str = "Attention by Action Type",
) -> plt.Figure:
    """Grouped bar chart: attention per category, conditioned on action type.

    Args:
        action_attn_df: DataFrame with action_type index and attn_* columns.
            Produced by CorrelationAnalyzer.compute_action_conditioned_attention().
        save_path: If given, save the figure here.
        title: Figure title.

    Returns:
        matplotlib Figure.
    """
    set_paper_style()

    attn_cols = [c for c in ["attn_sdc", "attn_agents", "attn_roadgraph", "attn_lights", "attn_gps"]
                 if c in action_attn_df.columns]
    action_types = list(action_attn_df.index)

    n_groups = len(action_types)
    n_bars = len(attn_cols)
    width = 0.8 / n_bars
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for i, col in enumerate(attn_cols):
        vals = action_attn_df[col].values
        offset = (i - n_bars / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width=width * 0.9,
                      label=ATTN_LABELS.get(col, col),
                      color=ATTN_COLORS.get(col, "#888888"))

    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in action_types])
    ax.set_xlabel("Action type")
    ax.set_ylabel("Mean attention (fraction)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.2))

    # Add n= annotation
    if "n" in action_attn_df.columns:
        for xi, at in zip(x, action_types):
            n = int(action_attn_df.loc[at, "n"])
            ax.text(xi, ax.get_ylim()[1] * 0.98, f"n={n}", ha="center",
                    fontsize=8, color="gray")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)

    return fig
