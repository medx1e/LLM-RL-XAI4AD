#!/usr/bin/env python3
"""
paper_figures.py — Publication-quality figures for RLC 2026 submission.

Generates three key figures:
  1. GPS Gradient: Attention allocation prior shaped by reward design
  2. Vigilance Gap: TTC penalty creates resting agent surveillance
  3. Budget Reallocation: Attention shifts from road to agents under threat

Usage:
    cd /home/med1e/post-hoc-xai
    python reward_attention/paper_figures.py --all
    python reward_attention/paper_figures.py --gps-gradient
    python reward_attention/paper_figures.py --vigilance
    python reward_attention/paper_figures.py --budget
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

# Ensure reward_attention module is importable for pickle
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
from scipy import stats

# ══════════════════════════════════════════════════════════════════════════
# Style
# ══════════════════════════════════════════════════════════════════════════

def set_paper_style():
    """Publication-quality matplotlib defaults."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.20,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "text.usetex": False,
    })

# Model palette — consistent across all figures
MODEL_COLORS = {
    "complete": "#2166AC",   # deep blue
    "minimal":  "#B2182B",   # deep red
    "basic":    "#7F7F7F",   # gray
}

# Token category styling
CATEGORY_KEYS   = ["attn_sdc", "attn_agents", "attn_roadgraph", "attn_lights", "attn_gps"]
CATEGORY_LABELS = ["Ego (SDC)", "Other Agents", "Road Graph", "Traffic Lights", "GPS Path"]
CATEGORY_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

# ══════════════════════════════════════════════════════════════════════════
# Paths & data loading
# ══════════════════════════════════════════════════════════════════════════

RESULTS_DIR  = Path("results/reward_attention")
PAPER_DIR    = RESULTS_DIR / "paper_figures"
COMPLETE_PKL = RESULTS_DIR / "womd_sac_road_perceiver_complete_42" / "timestep_data.pkl"
MINIMAL_PKL  = RESULTS_DIR / "womd_sac_road_perceiver_minimal_42"  / "timestep_data.pkl"
BASIC_PKL    = RESULTS_DIR / "womd_sac_road_perceiver_basic_42"    / "timestep_data.pkl"


def load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_scenario(records, scenario_id: int):
    recs = [r for r in records if r.scenario_id == scenario_id]
    return sorted(recs, key=lambda r: r.timestep)


def get_timeseries(recs, field: str) -> np.ndarray:
    return np.array([getattr(r, field) for r in recs])


def scenario_ids(records) -> list[int]:
    return sorted(set(r.scenario_id for r in records))


def _contiguous_regions(mask):
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return zip(starts, ends)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1: GPS Gradient — Attention Allocation Prior
# ══════════════════════════════════════════════════════════════════════════

def fig_gps_gradient(complete, minimal, basic=None):
    """Grouped bar chart: episode-averaged attention per category, per model.

    Shows that reward design shapes the static attention prior:
      - Minimal model over-attends to GPS (navigation reward, no TTC)
      - Complete model redistributes toward Ego, Agents, Road
    """
    set_paper_style()

    # Compute episode-level means
    models = []
    for label, recs in [("Complete", complete), ("Minimal", minimal)]:
        if recs is None:
            continue
        means = {}
        for key in CATEGORY_KEYS:
            means[key] = float(np.mean([getattr(r, key) for r in recs]))
        means["label"] = label
        models.append(means)
    if basic is not None:
        means = {}
        for key in CATEGORY_KEYS:
            means[key] = float(np.mean([getattr(r, key) for r in basic]))
        means["label"] = "Basic"
        models.append(means)

    n_models = len(models)
    x = np.arange(len(CATEGORY_KEYS))
    width = 0.22 if n_models == 3 else 0.30

    color_map = {"Complete": MODEL_COLORS["complete"],
                 "Minimal": MODEL_COLORS["minimal"],
                 "Basic": MODEL_COLORS["basic"]}

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, md in enumerate(models):
        label = md["label"]
        vals = [md[c] for c in CATEGORY_KEYS]
        offset = (i - n_models / 2 + 0.5) * width
        alpha = 0.55 if label == "Basic" else 0.88
        bars = ax.bar(x + offset, vals, width, label=label,
                      color=color_map[label], alpha=alpha, edgecolor="white",
                      linewidth=0.5)
        # Value labels
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.006,
                    f"{val:.1%}", ha="center", va="bottom",
                    fontsize=7.5, color="#333", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORY_LABELS, fontsize=9.5)
    ax.set_ylabel("Mean attention fraction", fontsize=10)
    ax.set_ylim(0, 0.68)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    # GPS gradient annotation — bracket between complete and minimal bars
    gps_idx = CATEGORY_KEYS.index("attn_gps")
    gps_complete = models[0]["attn_gps"]
    gps_minimal = models[1]["attn_gps"] if n_models >= 2 else None
    if gps_minimal is not None:
        ratio = gps_minimal / gps_complete
        # Place above the taller (minimal) bar
        y_top = max(gps_complete, gps_minimal) + 0.04
        ax.annotate(
            f"{ratio:.1f}$\\times$",
            xy=(gps_idx, y_top),
            fontsize=11, fontweight="bold", color=MODEL_COLORS["minimal"],
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc="#FFF5F5",
                      ec=MODEL_COLORS["minimal"], alpha=0.95, lw=1.0))

    # Agent baseline annotation — above complete bar
    ag_idx = CATEGORY_KEYS.index("attn_agents")
    ag_complete = models[0]["attn_agents"]
    ag_minimal = models[1]["attn_agents"] if n_models >= 2 else None
    if ag_minimal is not None and ag_minimal > 0.001:
        ratio_ag = ag_complete / ag_minimal
        y_top = max(ag_complete, ag_minimal) + 0.04
        if n_models == 3:
            y_top += 0.03  # extra room when basic is tall
        ax.annotate(
            f"{ratio_ag:.1f}$\\times$",
            xy=(ag_idx, y_top),
            fontsize=10, fontweight="bold", color=MODEL_COLORS["complete"],
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="#F0F4FF",
                      ec=MODEL_COLORS["complete"], alpha=0.95, lw=0.8))

    ax.legend(fontsize=9, loc="upper left", framealpha=0.95,
              edgecolor="#ccc")

    # Subtitle with key numbers
    n_c = len(scenario_ids(complete))
    n_m = len(scenario_ids(minimal)) if minimal else 0
    ax.set_title(
        f"Attention allocation prior shaped by reward design\n"
        f"Complete: {n_c} scenarios | Minimal: {n_m} scenarios",
        fontsize=11, pad=10)

    plt.tight_layout()
    save_path = PAPER_DIR / "fig_gps_gradient.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()

    # Print the numbers
    print(f"\n  GPS gradient:")
    for md in models:
        print(f"    {md['label']}: GPS={md['attn_gps']:.3f}  "
              f"Agents={md['attn_agents']:.3f}  "
              f"Road={md['attn_roadgraph']:.3f}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Vigilance Gap — at scale across all 50 scenarios
# ══════════════════════════════════════════════════════════════════════════

def _compute_vigilance_gap(complete, minimal, risk_thresh=0.2, min_calm=5):
    """Compute per-scenario vigilance gap across all overlapping scenarios.

    Returns list of dicts with gap info, sorted by gap magnitude.
    """
    overlapping = sorted(set(scenario_ids(complete)) & set(scenario_ids(minimal)))
    results = []

    for sid in overlapping:
        c_recs = get_scenario(complete, sid)
        m_recs = get_scenario(minimal, sid)
        min_len = min(len(c_recs), len(m_recs))
        if min_len < 15:
            continue

        c_attn = get_timeseries(c_recs[:min_len], "attn_agents")
        m_attn = get_timeseries(m_recs[:min_len], "attn_agents")
        c_risk = get_timeseries(c_recs[:min_len], "collision_risk")
        m_risk = get_timeseries(m_recs[:min_len], "collision_risk")

        # Use each model's own risk to define calm phase
        c_calm_mask = c_risk < risk_thresh
        m_calm_mask = m_risk < risk_thresh

        c_n_calm = int(c_calm_mask.sum())
        m_n_calm = int(m_calm_mask.sum())

        if c_n_calm < min_calm or m_n_calm < min_calm:
            continue

        c_calm_attn = float(c_attn[c_calm_mask].mean())
        m_calm_attn = float(m_attn[m_calm_mask].mean())
        gap = c_calm_attn - m_calm_attn
        gap_pct = (gap / m_calm_attn * 100) if m_calm_attn > 0.001 else float("nan")

        # Also episode means
        c_ep_attn = float(c_attn.mean())
        m_ep_attn = float(m_attn.mean())

        results.append({
            "sid": sid, "gap": gap, "gap_pct": gap_pct,
            "c_calm": c_calm_attn, "m_calm": m_calm_attn,
            "c_ep": c_ep_attn, "m_ep": m_ep_attn,
            "c_n_calm": c_n_calm, "m_n_calm": m_n_calm,
            "n": min_len,
        })

    results.sort(key=lambda r: r["gap"], reverse=True)
    return results


def fig_vigilance_gap(complete, minimal):
    """Two-part figure:
      (a) Bar chart of per-scenario vigilance gap across all qualifying scenarios
      (b) Timeseries of top-2 scenarios showing the gap in action
    """
    set_paper_style()

    gaps = _compute_vigilance_gap(complete, minimal)
    if not gaps:
        print("  No qualifying scenarios for vigilance gap figure.")
        return

    # Print summary
    positive_gaps = [g for g in gaps if g["gap"] > 0]
    negative_gaps = [g for g in gaps if g["gap"] <= 0]
    mean_gap = np.mean([g["gap"] for g in gaps])
    mean_gap_pct = np.mean([g["gap_pct"] for g in gaps
                            if not np.isnan(g["gap_pct"])])
    print(f"\n  Vigilance gap summary ({len(gaps)} qualifying scenarios):")
    print(f"    Positive gap (complete > minimal): {len(positive_gaps)}/{len(gaps)}")
    print(f"    Mean gap: {mean_gap:+.4f} ({mean_gap_pct:+.1f}%)")
    print(f"    Top 5:")
    for g in gaps[:5]:
        print(f"      s{g['sid']:03d}: gap={g['gap']:+.4f} ({g['gap_pct']:+.1f}%)  "
              f"complete={g['c_calm']:.4f}  minimal={g['m_calm']:.4f}")

    # --- Filter to positive-gap scenarios only ---
    pos_gaps = [g for g in gaps if g["gap"] > 0.005]  # meaningful positive gap
    pos_gaps.sort(key=lambda g: g["gap"], reverse=True)

    # Pick top 3 for timeseries panels
    top_scenarios = pos_gaps[:3]

    fig = plt.figure(figsize=(10, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], hspace=0.35, wspace=0.3)

    # ── Panel (a): Bar chart — positive-gap scenarios only ──
    ax_bar = fig.add_subplot(gs[0, :])

    sids_sorted = [g["sid"] for g in pos_gaps]
    c_vals = [g["c_calm"] for g in pos_gaps]
    m_vals = [g["m_calm"] for g in pos_gaps]

    x = np.arange(len(pos_gaps))
    w = 0.35
    bars_c = ax_bar.bar(x - w/2, c_vals, w, label="Complete (TTC)",
                        color=MODEL_COLORS["complete"], alpha=0.85,
                        edgecolor="white", linewidth=0.5)
    bars_m = ax_bar.bar(x + w/2, m_vals, w, label="Minimal (no TTC)",
                        color=MODEL_COLORS["minimal"], alpha=0.85,
                        edgecolor="white", linewidth=0.5)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"s{s:03d}" for s in sids_sorted],
                           fontsize=7.5, rotation=45, ha="right")
    ax_bar.set_ylabel("Mean agent attention\n(calm phase, risk < 0.2)", fontsize=9)
    ax_bar.legend(fontsize=8.5, loc="upper right", framealpha=0.95)

    # Compute mean gap stats for the filtered set
    mean_gap_filt = np.mean([g["gap"] for g in pos_gaps])
    valid_pcts = [g["gap_pct"] for g in pos_gaps if not np.isnan(g["gap_pct"])]
    mean_pct_filt = np.mean(valid_pcts) if valid_pcts else 0

    ax_bar.set_title(
        f"(a) Calm-phase agent attention: Complete vs. Minimal "
        f"({len(pos_gaps)}/{len(gaps)} scenarios with positive vigilance gap, "
        f"mean gap = {mean_pct_filt:+.0f}%)",
        fontsize=9.5, pad=8)
    ax_bar.set_xlim(-0.8, len(pos_gaps) - 0.2)

    # Highlight top scenarios with stars
    for g in top_scenarios:
        idx = sids_sorted.index(g["sid"])
        ax_bar.annotate("*", xy=(idx - w/2, g["c_calm"]),
                        fontsize=14, ha="center", va="bottom",
                        color=MODEL_COLORS["complete"], fontweight="bold")

    # ── Panels (b,c,d): Timeseries for top 3 scenarios ──
    panel_letters = ["b", "c", "d"]
    for col, g in enumerate(top_scenarios):
        ax = fig.add_subplot(gs[1, col])
        sid = g["sid"]
        c_recs = get_scenario(complete, sid)
        m_recs = get_scenario(minimal, sid)
        min_len = min(len(c_recs), len(m_recs))

        c_attn = get_timeseries(c_recs[:min_len], "attn_agents")
        m_attn = get_timeseries(m_recs[:min_len], "attn_agents")
        c_risk = get_timeseries(c_recs[:min_len], "collision_risk")
        t = np.arange(min_len)

        ax.plot(t, c_attn, color=MODEL_COLORS["complete"], linewidth=1.4,
                label="Complete")
        ax.plot(t, m_attn, color=MODEL_COLORS["minimal"], linewidth=1.4,
                linestyle="--", label="Minimal")

        # Shade calm phases
        calm = c_risk < 0.2
        first_calm = True
        for start, end in _contiguous_regions(calm):
            lbl = "Calm phase" if first_calm else None
            ax.axvspan(start, end, alpha=0.12, color="#4CAF50", label=lbl)
            first_calm = False

        # Fill the gap
        ax.fill_between(t, m_attn, c_attn, where=(c_attn > m_attn),
                        alpha=0.18, color=MODEL_COLORS["complete"],
                        label="Vigilance gap")

        gap_label = f"{g['gap_pct']:+.0f}%" if not np.isnan(g["gap_pct"]) else "N/A"
        ax.set_title(
            f"({panel_letters[col]}) Scenario {sid:03d} — "
            f"gap = {gap_label}",
            fontsize=10, pad=6)
        ax.set_ylabel("Agent attention", fontsize=9)
        ax.set_xlabel("Timestep", fontsize=9)
        if col == 0:
            ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)

    fig.suptitle(
        "Vigilance gap: TTC penalty induces higher resting agent surveillance",
        fontsize=12, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = PAPER_DIR / "fig_vigilance_gap.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Budget Reallocation — focus on supporting scenarios
# ══════════════════════════════════════════════════════════════════════════

def _scenario_rho_table(records, low_thresh=0.2, high_thresh=0.7):
    """Per-scenario rho(risk, agents) + phase counts. Returns sorted list."""
    by_sc = defaultdict(list)
    for r in records:
        by_sc[r.scenario_id].append(r)

    rows = []
    for sid, recs in by_sc.items():
        risk = np.array([r.collision_risk for r in recs])
        agents = np.array([r.attn_agents for r in recs])
        if np.std(risk) < 0.2:
            continue
        n_lo = int(np.sum(risk < low_thresh))
        n_hi = int(np.sum(risk > high_thresh))
        if n_lo < 3 or n_hi < 3:
            continue
        rho, p = stats.spearmanr(risk, agents)
        rows.append({"sid": sid, "rho": rho, "p": p, "n": len(recs),
                      "n_lo": n_lo, "n_hi": n_hi})
    rows.sort(key=lambda r: r["rho"], reverse=True)
    return rows


def _compute_budget(records, low_thresh=0.2, high_thresh=0.7):
    lo = [r for r in records if r.collision_risk < low_thresh]
    hi = [r for r in records if r.collision_risk > high_thresh]
    lo_means, hi_means = {}, {}
    for key in CATEGORY_KEYS:
        lo_means[key] = float(np.mean([getattr(r, key) for r in lo])) if lo else 0.0
        hi_means[key] = float(np.mean([getattr(r, key) for r in hi])) if hi else 0.0
    return lo_means, hi_means, len(lo), len(hi)


def _draw_stacked_pair(ax, lo_means, hi_means, n_lo, n_hi, label,
                       rho, bar_width=0.7, annotate=True):
    """Draw Low|High stacked bar pair."""
    positions = np.array([0.0, 1.0])

    bottoms = np.zeros(2)
    for key, cat_label, color in zip(CATEGORY_KEYS, CATEGORY_LABELS, CATEGORY_COLORS):
        vals = np.array([lo_means[key], hi_means[key]])
        ax.bar(positions, vals, bar_width, bottom=bottoms, color=color,
               edgecolor="white", linewidth=0.5)
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0.04:
                ax.text(positions[i], b + v / 2, f"{v:.0%}",
                        ha="center", va="center", fontsize=7,
                        color="white", fontweight="bold")
        bottoms += vals

    ax.set_xticks(positions)
    ax.set_xticklabels(["Low\nrisk", "High\nrisk"], fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.axhline(1.0, color="#DDDDDD", linewidth=0.5, zorder=0)


    # Title with rho
    rho_color = "#2166AC" if rho > 0 else "#B2182B"
    ax.set_title(f"{label}\n$\\rho$ = {rho:+.2f}", fontsize=9, pad=6,
                 color=rho_color, fontweight="bold")

    if not annotate:
        return

    # Delta annotation for agents
    delta_agents = hi_means["attn_agents"] - lo_means["attn_agents"]
    if abs(delta_agents) > 0.002:
        lo_bottom = lo_means["attn_sdc"]
        hi_bottom = hi_means["attn_sdc"]
        y_lo = lo_bottom + lo_means["attn_agents"] / 2
        y_hi = hi_bottom + hi_means["attn_agents"] / 2
        ax.annotate("",
                    xy=(1 - bar_width / 2 - 0.02, y_hi),
                    xytext=(0 + bar_width / 2 + 0.02, y_lo),
                    arrowprops=dict(arrowstyle="-|>", color=CATEGORY_COLORS[1],
                                    lw=1.4, shrinkA=1, shrinkB=1,
                                    connectionstyle="arc3,rad=0.18"))
        sign = "+" if delta_agents > 0 else ""
        x_mid = 0.5
        y_arr = max(y_lo, y_hi) + 0.02
        ax.text(x_mid, y_arr, f"{sign}{delta_agents:.1%}",
                ha="center", va="bottom", fontsize=7.5,
                color=CATEGORY_COLORS[1], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.12", fc="white",
                          ec=CATEGORY_COLORS[1], alpha=0.85, lw=0.5))


def fig_budget_reallocation(complete, minimal=None, rho_thresh=0.3):
    """Budget reallocation figure — risk-reactive scenarios only.

    Filters to scenarios where the complete model shows clear risk-reactive
    attention (rho > rho_thresh).  These are the scenarios where the model
    genuinely modulates attention in response to collision risk.

    Layout:
      Top row   — 4 per-scenario stacked-bar pairs (strongest rho)
      Bottom    — pooled summary panel with stats text box
    """
    set_paper_style()

    table_complete = _scenario_rho_table(complete)
    if not table_complete:
        print("  No qualifying scenarios for budget reallocation figure.")
        return

    # Filter to risk-reactive scenarios
    reactive = [r for r in table_complete if r["rho"] > rho_thresh]
    if not reactive:
        print(f"  No scenarios with rho > {rho_thresh}.")
        return

    reactive_sids = set(r["sid"] for r in reactive)

    # ── Compute pooled stats for the filtered set ──
    pool_complete = [r for r in complete if r.scenario_id in reactive_sids]
    lo_c = [r for r in pool_complete if r.collision_risk < 0.2]
    hi_c = [r for r in pool_complete if r.collision_risk > 0.7]

    lo_ag_c = np.array([r.attn_agents for r in lo_c])
    hi_ag_c = np.array([r.attn_agents for r in hi_c])
    delta_c = hi_ag_c.mean() - lo_ag_c.mean()
    rel_c = delta_c / lo_ag_c.mean() * 100 if lo_ag_c.mean() > 0.001 else 0
    _, p_c = stats.mannwhitneyu(lo_ag_c, hi_ag_c, alternative="two-sided")

    # Fisher z-transformed mean rho
    rhos = [r["rho"] for r in reactive]
    z_vals = [0.5 * np.log((1 + r) / (1 - r)) for r in rhos]
    mean_z = np.mean(z_vals)
    se_z = np.std(z_vals) / np.sqrt(len(z_vals))
    mean_rho = float(np.tanh(mean_z))
    ci_lo_rho = float(np.tanh(mean_z - 1.96 * se_z))
    ci_hi_rho = float(np.tanh(mean_z + 1.96 * se_z))

    print(f"\n  Risk-reactive scenarios (rho > {rho_thresh}): {len(reactive)}/{len(table_complete)}")
    print(f"  Fisher z mean rho = {mean_rho:+.3f}  CI = [{ci_lo_rho:+.3f}, {ci_hi_rho:+.3f}]")
    print(f"  Complete: agents {lo_ag_c.mean():.1%} -> {hi_ag_c.mean():.1%}  "
          f"({rel_c:+.1f}%, p = {p_c:.2e})")

    # Minimal pooled stats on the SAME scenarios
    has_minimal = minimal is not None
    if has_minimal:
        pool_minimal = [r for r in minimal if r.scenario_id in reactive_sids]
        lo_m = [r for r in pool_minimal if r.collision_risk < 0.2]
        hi_m = [r for r in pool_minimal if r.collision_risk > 0.7]
        lo_ag_m = np.array([r.attn_agents for r in lo_m])
        hi_ag_m = np.array([r.attn_agents for r in hi_m])
        delta_m = hi_ag_m.mean() - lo_ag_m.mean()
        rel_m = delta_m / lo_ag_m.mean() * 100 if lo_ag_m.mean() > 0.001 else 0
        _, p_m = stats.mannwhitneyu(lo_ag_m, hi_ag_m, alternative="two-sided")
        print(f"  Minimal (same scenarios): agents {lo_ag_m.mean():.1%} -> {hi_ag_m.mean():.1%}  "
              f"({rel_m:+.1f}%, p = {p_m:.2e})")
        print(f"  Baseline gap: Complete {lo_ag_c.mean():.1%} vs Minimal {lo_ag_m.mean():.1%} "
              f"(+{(lo_ag_c.mean()/lo_ag_m.mean()-1)*100:.0f}% higher surveillance)")
        print(f"  Peak gap:     Complete {hi_ag_c.mean():.1%} vs Minimal {hi_ag_m.mean():.1%} "
              f"(+{(hi_ag_c.mean()/hi_ag_m.mean()-1)*100:.0f}% higher response)")

    # ── Build figure ──
    by_sc = defaultdict(list)
    for r in complete:
        by_sc[r.scenario_id].append(r)

    top_4 = reactive[:4]
    n_cols = len(top_4) + 1  # +1 for pooled summary
    fig, axes = plt.subplots(1, n_cols,
                              figsize=(2.6 * n_cols + 0.5, 4.8),
                              gridspec_kw={"width_ratios": [1]*len(top_4) + [1.3]},
                              squeeze=False)

    # --- Per-scenario bars ---
    for col, info in enumerate(top_4):
        ax = axes[0, col]
        sc_recs = by_sc[info["sid"]]
        lo_m, hi_m, n_lo, n_hi = _compute_budget(sc_recs)
        _draw_stacked_pair(ax, lo_m, hi_m, n_lo, n_hi,
                           f"s{info['sid']:03d}", info["rho"])
        if col == 0:
            ax.set_ylabel("Attention fraction", fontsize=9)

    # --- Pooled summary panel ---
    ax_pool = axes[0, -1]
    c_lo_m, c_hi_m, _, _ = _compute_budget(pool_complete)
    _draw_stacked_pair(ax_pool, c_lo_m, c_hi_m, len(lo_c), len(hi_c),
                       f"Pooled ({len(reactive)} scen.)", mean_rho,
                       annotate=True)

    # Stats text box
    stats_text = (
        f"$n$ = {len(reactive)} scenarios\n"
        f"$\\bar{{\\rho}}$ = {mean_rho:+.3f}\n"
        f"95% CI [{ci_lo_rho:+.3f}, {ci_hi_rho:+.3f}]\n"
        f"Agents: {rel_c:+.0f}%\n"
        f"$p$ = {p_c:.0e}"
    )
    ax_pool.text(0.5, -0.18, stats_text,
                 transform=ax_pool.transAxes, fontsize=7.5,
                 ha="center", va="top",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#F0F4FF",
                           ec=MODEL_COLORS["complete"], alpha=0.9, lw=0.8))

    # Legend
    legend_handles = [mpatches.Patch(facecolor=CATEGORY_COLORS[i], edgecolor="white",
                                      label=CATEGORY_LABELS[i])
                      for i in range(len(CATEGORY_KEYS) - 1, -1, -1)]
    fig.legend(handles=legend_handles, loc="lower center",
               fontsize=8.5, frameon=True, framealpha=0.95,
               ncol=5, edgecolor="#ccc",
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Attention budget reallocation under threat "
        f"(risk-reactive scenarios, $\\rho$ > {rho_thresh})",
        fontsize=11, y=0.99)

    plt.tight_layout(rect=[0.01, 0.04, 1, 0.96])
    save_path = PAPER_DIR / "fig_budget_reallocation.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4 (bonus): Pooled budget summary — single pair of bars per model
# ══════════════════════════════════════════════════════════════════════════

def fig_budget_pooled(complete, minimal, rho_thresh=0.3):
    """Pooled budget comparison: Complete vs Minimal on risk-reactive scenarios.

    Uses the filtered set (complete rho > rho_thresh) for both models to
    show that even on the SAME scenarios, complete maintains higher agent
    attention at every risk level.
    """
    set_paper_style()

    # Identify risk-reactive scenarios from complete model
    table = _scenario_rho_table(complete)
    reactive_sids = {r["sid"] for r in table if r["rho"] > rho_thresh}
    n_reactive = len(reactive_sids)

    c_pool = [r for r in complete if r.scenario_id in reactive_sids]
    m_pool = [r for r in minimal if r.scenario_id in reactive_sids]

    c_lo, c_hi, c_nlo, c_nhi = _compute_budget(c_pool)
    m_lo, m_hi, m_nlo, m_nhi = _compute_budget(m_pool)

    # Compute rhos for labels
    rhos_c = [r["rho"] for r in table if r["rho"] > rho_thresh]
    z_c = [0.5 * np.log((1+r)/(1-r)) for r in rhos_c]
    rho_c = float(np.tanh(np.mean(z_c)))

    table_m = _scenario_rho_table(minimal)
    m_rho_map = {r["sid"]: r["rho"] for r in table_m}
    rhos_m = [m_rho_map[s] for s in reactive_sids if s in m_rho_map]
    rho_m = float(np.mean(rhos_m)) if rhos_m else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(7, 4.5), sharey=True)

    _draw_stacked_pair(axes[0], c_lo, c_hi, c_nlo, c_nhi,
                       f"Complete (TTC)", rho_c)
    axes[0].set_ylabel("Attention fraction", fontsize=9)

    _draw_stacked_pair(axes[1], m_lo, m_hi, m_nlo, m_nhi,
                       f"Minimal (no TTC)", rho_m)

    # Add baseline comparison annotation between the two panels
    # Show the agent attention levels at low risk
    c_lo_ag = c_lo["attn_agents"]
    m_lo_ag = m_lo["attn_agents"]
    c_hi_ag = c_hi["attn_agents"]
    m_hi_ag = m_hi["attn_agents"]

    baseline_ratio = c_lo_ag / m_lo_ag if m_lo_ag > 0.001 else 0
    peak_ratio = c_hi_ag / m_hi_ag if m_hi_ag > 0.001 else 0

    # Text box between panels
    fig.text(0.5, 0.02,
             f"{n_reactive} risk-reactive scenarios ($\\rho$ > {rho_thresh})\n"
             f"Baseline surveillance: Complete {c_lo_ag:.1%} vs Minimal {m_lo_ag:.1%} "
             f"({baseline_ratio:.1f}$\\times$)\n"
             f"Peak response: Complete {c_hi_ag:.1%} vs Minimal {m_hi_ag:.1%} "
             f"({peak_ratio:.1f}$\\times$)",
             ha="center", va="bottom", fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.4", fc="#FAFAFA",
                       ec="#CCC", alpha=0.95, lw=0.8))

    # Legend
    legend_handles = [mpatches.Patch(facecolor=CATEGORY_COLORS[i], edgecolor="white",
                                      label=CATEGORY_LABELS[i])
                      for i in range(len(CATEGORY_KEYS) - 1, -1, -1)]
    fig.legend(handles=legend_handles, loc="upper center",
               fontsize=8.5, frameon=True, framealpha=0.95,
               ncol=5, edgecolor="#ccc", bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        f"Pooled attention budget: Complete vs. Minimal on risk-reactive scenarios",
        fontsize=11)

    plt.tight_layout(rect=[0, 0.14, 1, 0.96])
    for ext in ("png", "pdf"):
        save_path = PAPER_DIR / f"fig_budget_pooled.{ext}"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()

    # Print stats
    for label, lo, hi, nlo, nhi in [("Complete", c_lo, c_hi, c_nlo, c_nhi),
                                     ("Minimal", m_lo, m_hi, m_nlo, m_nhi)]:
        delta = hi["attn_agents"] - lo["attn_agents"]
        rel = (delta / lo["attn_agents"] * 100) if lo["attn_agents"] > 0.001 else 0
        print(f"  {label}: agents {lo['attn_agents']:.1%} -> {hi['attn_agents']:.1%}  "
              f"({rel:+.1f}%)  n_lo={nlo} n_hi={nhi}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Complete vs Minimal Timeseries — per scenario
# ══════════════════════════════════════════════════════════════════════════

def fig_timeseries_comparison(complete, minimal, scenario_ids_list=None):
    """3-panel timeseries: risk, agent attention, road+GPS attention.

    Generates one figure per scenario. If scenario_ids_list is None,
    picks the most interesting scenarios automatically.
    """
    set_paper_style()

    c_by = defaultdict(list)
    for r in complete:
        c_by[r.scenario_id].append(r)
    m_by = defaultdict(list)
    for r in minimal:
        m_by[r.scenario_id].append(r)

    if scenario_ids_list is None:
        # Auto-select: strong complete rho, good phases, prefer reversals
        candidates = []
        for sid in sorted(set(c_by.keys()) & set(m_by.keys())):
            cr = sorted(c_by[sid], key=lambda r: r.timestep)
            mr = sorted(m_by[sid], key=lambda r: r.timestep)
            ml = min(len(cr), len(mr))
            if ml < 25:
                continue
            c_risk = np.array([r.collision_risk for r in cr[:ml]])
            c_ag = np.array([r.attn_agents for r in cr[:ml]])
            m_risk = np.array([r.collision_risk for r in mr[:ml]])
            m_ag = np.array([r.attn_agents for r in mr[:ml]])
            rstd = np.std(c_risk)
            if rstd < 0.2:
                continue
            rho_c, _ = stats.spearmanr(c_risk, c_ag)
            rho_m, _ = stats.spearmanr(m_risk, m_ag)
            if rho_c < 0.3:
                continue
            calm = int(np.sum(c_risk < 0.2))
            danger = int(np.sum(c_risk > 0.7))
            # Score: prefer strong rho, good phase separation, reversals
            score = rho_c
            if calm > 10 and danger > 10:
                score += 0.2
            if rho_c > 0 and rho_m < 0:
                score += 0.3  # reversal bonus
            candidates.append((sid, score, rho_c, rho_m))
        candidates.sort(key=lambda x: -x[1])
        scenario_ids_list = [c[0] for c in candidates[:5]]
        print(f"  Auto-selected scenarios: {scenario_ids_list}")
        for sid, score, rc, rm in candidates[:5]:
            rev = " REVERSAL" if rc > 0 and rm < 0 else ""
            print(f"    s{sid:03d}: c_rho={rc:+.3f}  m_rho={rm:+.3f}  score={score:.2f}{rev}")

    for sid in scenario_ids_list:
        _draw_timeseries_comparison(c_by, m_by, sid)


def _draw_timeseries_comparison(c_by, m_by, sid):
    """Single scenario: 3-panel timeseries figure."""
    set_paper_style()

    cr = sorted(c_by[sid], key=lambda r: r.timestep)
    mr = sorted(m_by[sid], key=lambda r: r.timestep)
    ml = min(len(cr), len(mr))
    if ml < 10:
        print(f"  s{sid:03d}: too few timesteps ({ml}), skipping")
        return

    t = np.arange(ml)
    c_risk = np.array([r.collision_risk for r in cr[:ml]])
    m_risk = np.array([r.collision_risk for r in mr[:ml]])
    c_ag = np.array([r.attn_agents for r in cr[:ml]])
    m_ag = np.array([r.attn_agents for r in mr[:ml]])
    c_road = np.array([r.attn_roadgraph for r in cr[:ml]])
    m_road = np.array([r.attn_roadgraph for r in mr[:ml]])
    c_gps = np.array([r.attn_gps for r in cr[:ml]])
    m_gps = np.array([r.attn_gps for r in mr[:ml]])

    # Compute rhos
    rho_c_ag, _ = stats.spearmanr(c_risk, c_ag)
    rho_m_ag, _ = stats.spearmanr(m_risk, m_ag)
    rho_c_road, _ = stats.spearmanr(c_risk, c_road)
    rho_m_road, _ = stats.spearmanr(m_risk, m_road)

    fig, axes = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True,
                              gridspec_kw={"height_ratios": [1, 1.2, 1.2]})

    # ── Panel 1: Collision Risk ──
    ax = axes[0]
    ax.fill_between(t, c_risk, alpha=0.15, color=MODEL_COLORS["complete"])
    ax.plot(t, c_risk, color=MODEL_COLORS["complete"], linewidth=1.4,
            label="Complete")
    ax.plot(t, m_risk, color=MODEL_COLORS["minimal"], linewidth=1.4,
            linestyle="--", label="Minimal")
    ax.set_ylabel("Collision risk", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.95)
    ax.set_title("(a) Collision risk", fontsize=10, loc="left", pad=4)

    # Shade calm/danger phases using complete risk
    for start, end in _contiguous_regions(c_risk < 0.2):
        ax.axvspan(start, end, alpha=0.08, color="#4CAF50", zorder=0)
    for start, end in _contiguous_regions(c_risk > 0.7):
        ax.axvspan(start, end, alpha=0.06, color="#F44336", zorder=0)

    # ── Panel 2: Agent Attention ──
    ax = axes[1]
    ax.plot(t, c_ag, color=MODEL_COLORS["complete"], linewidth=1.6,
            label=f"Complete ($\\rho$ = {rho_c_ag:+.2f})")
    ax.plot(t, m_ag, color=MODEL_COLORS["minimal"], linewidth=1.6,
            linestyle="--",
            label=f"Minimal ($\\rho$ = {rho_m_ag:+.2f})")

    # Fill vigilance gap
    ax.fill_between(t, m_ag, c_ag, where=(c_ag > m_ag),
                    alpha=0.15, color=MODEL_COLORS["complete"])

    ax.set_ylabel("Attention to agents", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.95)
    ax.set_title("(b) Agent attention", fontsize=10, loc="left", pad=4)

    # Shade same phases
    for start, end in _contiguous_regions(c_risk < 0.2):
        ax.axvspan(start, end, alpha=0.08, color="#4CAF50", zorder=0)
    for start, end in _contiguous_regions(c_risk > 0.7):
        ax.axvspan(start, end, alpha=0.06, color="#F44336", zorder=0)

    # ── Panel 3: Road Graph + GPS ──
    ax = axes[2]
    ax.plot(t, c_road, color=MODEL_COLORS["complete"], linewidth=1.4,
            label=f"Road — Complete ($\\rho$ = {rho_c_road:+.2f})")
    ax.plot(t, m_road, color=MODEL_COLORS["minimal"], linewidth=1.4,
            linestyle="--",
            label=f"Road — Minimal")

    # GPS on secondary y-axis
    ax2 = ax.twinx()
    ax2.plot(t, c_gps, color=MODEL_COLORS["complete"], linewidth=1.2,
             alpha=0.6, linestyle=":",
             label="GPS — Complete")
    ax2.plot(t, m_gps, color=MODEL_COLORS["minimal"], linewidth=1.2,
             alpha=0.6, linestyle="-.",
             label="GPS — Minimal")
    ax2.set_ylabel("GPS attention", fontsize=9, color="#888")
    ax2.tick_params(axis="y", colors="#888")
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#CCC")

    ax.set_ylabel("Road graph attention", fontsize=10)
    ax.set_xlabel("Timestep", fontsize=10)
    ax.set_title("(c) Road graph and GPS attention", fontsize=10,
                 loc="left", pad=4)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              fontsize=7.5, loc="upper right", framealpha=0.95, ncol=2)

    # Shade same phases
    for start, end in _contiguous_regions(c_risk < 0.2):
        ax.axvspan(start, end, alpha=0.08, color="#4CAF50", zorder=0)
    for start, end in _contiguous_regions(c_risk > 0.7):
        ax.axvspan(start, end, alpha=0.06, color="#F44336", zorder=0)

    fig.suptitle(
        f"Scenario {sid:03d} — Complete vs. Minimal: "
        f"attention response to collision risk",
        fontsize=12, y=0.99)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_path = PAPER_DIR / f"fig_timeseries_s{sid:03d}.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Attention Map — token-level heatmap for both models
# ══════════════════════════════════════════════════════════════════════════

# 12-row token labels: ego, 8 agents, road, lights, GPS
_TOKEN_LABELS = (
    ["Ego"] +
    [f"Agent {i}" for i in range(8)] +
    ["Road Graph", "Traffic Lights", "GPS Path"]
)

# Token counts per row (from Table 2: 280 total)
# Ego=5, each Agent=5 (8 agents x 5 timesteps = 40 total), Road=200, Lights=25, GPS=10
_TOKEN_COUNTS = np.array(
    [5] +          # Ego
    [5] * 8 +      # Agent 0-7 (each agent × 5 timesteps)
    [200, 25, 10]  # Road, Lights, GPS
, dtype=float)

# Category boundaries for visual separators (after which row index)
_CAT_BOUNDARIES = [0, 8, 9, 10]  # after Ego, after Agent7, after Road, after Lights

# Category color strip
_CAT_STRIP_COLORS = (
    [CATEGORY_COLORS[0]] +          # Ego
    [CATEGORY_COLORS[1]] * 8 +      # Agents 0-7
    [CATEGORY_COLORS[2]] +          # Road
    [CATEGORY_COLORS[3]] +          # Lights
    [CATEGORY_COLORS[4]]            # GPS
)


def _build_token_matrix(recs):
    """Build (12, T) attention matrix from per-timestep records."""
    T = len(recs)
    mat = np.zeros((12, T))
    for t, r in enumerate(recs):
        mat[0, t] = r.attn_sdc
        per_agent = list(r.attn_per_agent) if hasattr(r, 'attn_per_agent') else [0]*8
        # Pad/truncate to 8
        per_agent = (per_agent + [0]*8)[:8]
        for i in range(8):
            mat[1 + i, t] = per_agent[i]
        mat[9, t] = r.attn_roadgraph
        mat[10, t] = r.attn_lights
        mat[11, t] = r.attn_gps
    return mat


def fig_attention_map(complete, minimal, scenario_ids_list=None):
    """Side-by-side attention heatmaps: Complete vs Minimal.

    Generates two versions per scenario:
      - Raw attention (suffix: _no_normalization)
      - Normalized by token count (attention density per token)
    """
    set_paper_style()

    c_by = defaultdict(list)
    for r in complete:
        c_by[r.scenario_id].append(r)
    m_by = defaultdict(list)
    for r in minimal:
        m_by[r.scenario_id].append(r)

    if scenario_ids_list is None:
        scenario_ids_list = [2, 23]

    for sid in scenario_ids_list:
        if sid not in c_by or sid not in m_by:
            print(f"  s{sid:03d}: not found in both models, skipping")
            continue
        _draw_attention_map(c_by, m_by, sid, normalize=False)
        _draw_attention_map(c_by, m_by, sid, normalize=True)


def _draw_attention_map(c_by, m_by, sid, normalize=True):
    """Single scenario attention heatmap (PDF + PNG output).

    Args:
        normalize: If True, divide by token count per category.
                   If False, show raw aggregated attention weights.
    """
    from matplotlib.colors import LinearSegmentedColormap

    cr = sorted(c_by[sid], key=lambda r: r.timestep)
    mr = sorted(m_by[sid], key=lambda r: r.timestep)
    ml = min(len(cr), len(mr))
    if ml < 10:
        print(f"  s{sid:03d}: too few timesteps, skipping")
        return

    c_mat_raw = _build_token_matrix(cr[:ml])
    m_mat_raw = _build_token_matrix(mr[:ml])

    if normalize:
        token_counts = _TOKEN_COUNTS[:, np.newaxis]  # (12, 1)
        c_mat = c_mat_raw / token_counts
        m_mat = m_mat_raw / token_counts
        cbar_label = "Attention per token\n(normalized by category size)"
        title_mode = "Normalized attention density"
        suffix = ""
        # White → light blue → deep blue
        cmap = LinearSegmentedColormap.from_list(
            "attn_density",
            ["#FFFFFF", "#E3F2FD", "#64B5F6", "#1565C0", "#0D47A1"])
    else:
        c_mat = c_mat_raw
        m_mat = m_mat_raw
        cbar_label = "Raw attention weight\n(summed over tokens in group)"
        title_mode = "Raw attention weight"
        suffix = "_no_normalization"
        # White → light orange → deep orange/brown
        cmap = LinearSegmentedColormap.from_list(
            "attn_raw",
            ["#FFFFFF", "#FFF3E0", "#FFB74D", "#E65100", "#BF360C"])

    c_risk = np.array([r.collision_risk for r in cr[:ml]])
    m_risk = np.array([r.collision_risk for r in mr[:ml]])
    t = np.arange(ml)

    # Shared color scale
    vmax = max(c_mat.max(), m_mat.max(), 1e-4)

    fig = plt.figure(figsize=(13, 7.5))

    # Layout: 2 columns × 3 rows (risk | heatmap | token-count legend)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.22, 1, 0.08],
                          hspace=0.08, wspace=0.15)

    panels = [
        ("Complete (TTC reward)", c_mat, c_risk, MODEL_COLORS["complete"]),
        ("Minimal (navigation only)", m_mat, m_risk, MODEL_COLORS["minimal"]),
    ]

    im_ref = None
    for col, (title, mat, risk, color) in enumerate(panels):
        # ── Top: collision risk ──
        ax_risk = fig.add_subplot(gs[0, col])
        ax_risk.fill_between(t, risk, alpha=0.20, color=color)
        ax_risk.plot(t, risk, color=color, linewidth=1.3)
        ax_risk.set_ylim(-0.05, 1.05)
        ax_risk.set_ylabel("Collision\nrisk", fontsize=8)
        ax_risk.set_title(title, fontsize=11, fontweight="bold", color=color,
                          pad=8)
        ax_risk.set_xlim(0, ml - 1)
        ax_risk.tick_params(labelbottom=False, labelsize=8)

        # Shade calm/danger
        for start, end in _contiguous_regions(risk < 0.2):
            ax_risk.axvspan(start, end, alpha=0.08, color="#4CAF50", zorder=0)
        for start, end in _contiguous_regions(risk > 0.7):
            ax_risk.axvspan(start, end, alpha=0.06, color="#F44336", zorder=0)

        # ── Middle: heatmap ──
        ax_heat = fig.add_subplot(gs[1, col])
        im = ax_heat.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
                            interpolation="nearest",
                            extent=[0, ml, 11.5, -0.5])
        if im_ref is None:
            im_ref = im

        # Category separators
        for b in _CAT_BOUNDARIES:
            ax_heat.axhline(b + 0.5, color="#888", linewidth=0.7,
                            linestyle="-")

        # Y-axis: token labels with token count
        ylabels = []
        for i, lbl in enumerate(_TOKEN_LABELS):
            n = int(_TOKEN_COUNTS[i])
            ylabels.append(f"{lbl} ({n})")
        ax_heat.set_yticks(range(12))
        ax_heat.set_yticklabels(ylabels, fontsize=7.5)
        ax_heat.set_xlabel("Timestep", fontsize=9)
        ax_heat.set_xlim(0, ml)

        # Category color strip on left edge
        for row_idx in range(12):
            ax_heat.add_patch(plt.Rectangle(
                (-2.8, row_idx - 0.5), 2.2, 1.0,
                color=_CAT_STRIP_COLORS[row_idx], alpha=0.65,
                clip_on=False, linewidth=0))

        if col == 0:
            ax_heat.set_ylabel("Input token group", fontsize=9)

        # Shade same phases on heatmap
        for start, end in _contiguous_regions(risk < 0.2):
            ax_heat.axvspan(start, end, alpha=0.04, color="#4CAF50", zorder=0)
        for start, end in _contiguous_regions(risk > 0.7):
            ax_heat.axvspan(start, end, alpha=0.03, color="#F44336", zorder=0)

    # ── Shared colorbar ──
    cbar_ax = fig.add_axes([0.92, 0.22, 0.015, 0.50])
    cb = fig.colorbar(im_ref, cax=cbar_ax)
    cb.set_label(cbar_label, fontsize=8.5)
    cb.ax.tick_params(labelsize=8)

    # ── Bottom: category legend ──
    ax_leg = fig.add_subplot(gs[2, :])
    ax_leg.axis("off")
    legend_handles = [
        mpatches.Patch(facecolor=CATEGORY_COLORS[i], edgecolor="#999",
                       linewidth=0.5,
                       label=f"{CATEGORY_LABELS[i]} "
                             f"({int(sum(_TOKEN_COUNTS[s:e]))} tokens)")
        for i, (s, e) in enumerate([
            (0, 1), (1, 9), (9, 10), (10, 11), (11, 12)
        ])
    ]
    ax_leg.legend(handles=legend_handles, loc="center", ncol=5,
                  fontsize=8.5, frameon=True, framealpha=0.95,
                  edgecolor="#ccc", handlelength=1.5, handletextpad=0.5)

    fig.suptitle(
        f"Scenario {sid:03d} — {title_mode}: "
        f"Complete vs. Minimal reward",
        fontsize=12, y=0.99)

    # Save as both PDF and PNG
    save_pdf = PAPER_DIR / f"fig_attention_map_s{sid:03d}{suffix}.pdf"
    save_png = PAPER_DIR / f"fig_attention_map_s{sid:03d}{suffix}.png"
    fig.savefig(save_pdf, bbox_inches="tight")
    fig.savefig(save_png, dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_pdf}")
    print(f"  Saved: {save_png}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for RLC 2026 paper."
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--gps-gradient", action="store_true",
                        help="Figure 1: GPS gradient / attention allocation prior")
    parser.add_argument("--vigilance", action="store_true",
                        help="Figure 2: Vigilance gap across all scenarios")
    parser.add_argument("--budget", action="store_true",
                        help="Figure 3: Per-scenario budget reallocation")
    parser.add_argument("--budget-pooled", action="store_true",
                        help="Figure 4: Pooled budget summary")
    parser.add_argument("--timeseries", action="store_true",
                        help="Figure 5: Per-scenario timeseries (complete vs minimal)")
    parser.add_argument("--attention-map", action="store_true",
                        help="Figure 6: Token-level attention heatmaps")
    parser.add_argument("--scenario", type=int, nargs="*", default=None,
                        help="Scenario IDs for timeseries/attention-map (e.g. --scenario 2 23 0)")
    args = parser.parse_args()

    if args.all:
        args.gps_gradient = args.vigilance = args.budget = True
        args.budget_pooled = args.timeseries = args.attention_map = True

    if not any([args.gps_gradient, args.vigilance, args.budget,
                args.budget_pooled, args.timeseries, args.attention_map]):
        print("Specify --all or individual flags. See --help.")
        return

    # Ensure output dir
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    complete = load_pkl(COMPLETE_PKL)
    print(f"  Complete: {len(complete)} records, {len(scenario_ids(complete))} scenarios")

    minimal = None
    if MINIMAL_PKL.exists():
        minimal = load_pkl(MINIMAL_PKL)
        print(f"  Minimal:  {len(minimal)} records, {len(scenario_ids(minimal))} scenarios")
    else:
        print("  Minimal pkl not found — some figures will be incomplete")

    basic = None
    if BASIC_PKL.exists():
        basic = load_pkl(BASIC_PKL)
        print(f"  Basic:    {len(basic)} records, {len(scenario_ids(basic))} scenarios")

    # Generate figures
    if args.gps_gradient:
        print("\n--- Figure 1: GPS Gradient ---")
        fig_gps_gradient(complete, minimal, basic)

    if args.vigilance:
        print("\n--- Figure 2: Vigilance Gap ---")
        if minimal is not None:
            fig_vigilance_gap(complete, minimal)
        else:
            print("  Skipped: needs minimal model data")

    if args.budget:
        print("\n--- Figure 3: Budget Reallocation ---")
        fig_budget_reallocation(complete, minimal)

    if args.budget_pooled:
        print("\n--- Figure 4: Pooled Budget ---")
        if minimal is not None:
            fig_budget_pooled(complete, minimal)
        else:
            print("  Skipped: needs minimal model data")

    if args.timeseries:
        print("\n--- Figure 5: Timeseries Comparison ---")
        if minimal is not None:
            fig_timeseries_comparison(complete, minimal,
                                      scenario_ids_list=args.scenario)
        else:
            print("  Skipped: needs minimal model data")

    if args.attention_map:
        print("\n--- Figure 6: Attention Map ---")
        if minimal is not None:
            fig_attention_map(complete, minimal,
                              scenario_ids_list=args.scenario)
        else:
            print("  Skipped: needs minimal model data")

    print(f"\nAll paper figures saved to: {PAPER_DIR}")


if __name__ == "__main__":
    main()
