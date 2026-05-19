"""Grouped analysis and visualisation for SAE feature annotations.

Given the pre-computed annotation JSON (from ``feature_annotator.annotate``)
and the raw data (features matrix + telemetry dict), this module generates
publication-quality figures saved in both PNG and PDF formats:

1. **Best Feature Per Concept** — horizontal bar chart of |ρ| per telemetry
   field, showing the single strongest feature for each concept.
2. **Label Distribution** — bar chart of how many features received each
   label, plus a pie chart of dead vs active features.
3. **Concept Group Heatmap** — top-3 features per label group, full ρ
   profile across all telemetry fields.
4. **Concept Correlation Summary** — bubble / lollipop chart summarising each
   unique concept's best |ρ| and feature count in a single compact figure.
   Replaces the old crowded clustermap.
5. **Selectivity Histogram** — distribution of activation counts across
   all features.

All figures are saved to ``data/sae_interpretability/figures/`` (or the
directory passed to each function) in both **.png** and **.pdf** formats.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend — safe for servers / SSH sessions
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

from sae_interpretability.utils import (
    apply_phd_style,
    save_figure,
    PHD_BLUE, PHD_ORANGE, PHD_GREEN, PHD_PURPLE, PHD_TEAL, PHD_GRAY,
    PHD_DIVERGING, PHD_SEQUENTIAL, PHD_MAKO,
)

# ---------------------------------------------------------------------------
# Style initialisation
# ---------------------------------------------------------------------------

apply_phd_style()

_FIG_DIR = os.path.join("data", "sae_interpretability", "figures")

sns.set_theme(
    style="whitegrid",
    context="paper",
    font_scale=1.1,
    rc={
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "grid.alpha":         0.35,
        "axes.titlesize":     13,
        "axes.labelsize":     11,
    },
)


def _ensure_fig_dir(fig_dir: str) -> None:
    os.makedirs(fig_dir, exist_ok=True)


def _strip_rho_suffix(label: str) -> str:
    """Remove the trailing (ρ=…) or (z=…) suffix for cleaner grouping.

    'high_ego_speed (ρ=+0.812)' → 'high_ego_speed'
    """
    return re.sub(r"\s*\(.*\)\s*$", "", label).strip()


# ===================================================================
# 1. Best Feature Per Concept
# ===================================================================

def best_feature_per_concept(
    annotations: List[Dict[str, Any]],
    tel_keys: List[str],
    fig_dir: str = _FIG_DIR,
) -> None:
    """For each telemetry field, find the single feature with highest |ρ|.

    Prints the table and saves a horizontal bar chart (PNG + PDF).
    """
    _ensure_fig_dir(fig_dir)
    active = [a for a in annotations if a.get("label", "") != "dead"]

    # Build best-per-concept lookup
    best: Dict[str, Tuple[int, float]] = {}  # concept → (feat_idx, rho)

    for ann in active:
        rho_scores = ann.get("all_rho_scores", {})
        for field, rho in rho_scores.items():
            abs_rho = abs(rho)
            if field not in best or abs_rho > abs(best[field][1]):
                best[field] = (ann["feature_idx"], rho)

    # Print table
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║          Best Feature Per Telemetry Concept                 ║")
    print("╠══════════════════════════════════╦══════════╦═══════════════╣")
    print("║ Concept                          ║ Feature  ║     |ρ|      ║")
    print("╠══════════════════════════════════╬══════════╬═══════════════╣")
    for concept in sorted(best.keys()):
        feat_idx, rho = best[concept]
        print(f"║ {concept:<32s} ║ {feat_idx:>8d} ║ {abs(rho):>12.4f} ║")
    print("╚══════════════════════════════════╩══════════╩═══════════════╝")

    if not best:
        return

    # Horizontal bar chart
    concepts = sorted(best.keys(), key=lambda c: abs(best[c][1]))
    rho_vals = [abs(best[c][1]) for c in concepts]
    n = len(concepts)
    palette = sns.color_palette(PHD_MAKO, n_colors=max(n, 2))

    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.5)))
    bars = ax.barh(
        concepts, rho_vals,
        color=palette, edgecolor="white", linewidth=0.6, height=0.65,
    )

    # Annotate feature indices on bars
    for bar, concept in zip(bars, concepts):
        feat_idx, rho = best[concept]
        x = bar.get_width()
        ax.text(
            x + 0.008, bar.get_y() + bar.get_height() / 2,
            f"f{feat_idx}  ρ={rho:+.3f}",
            va="center", fontsize=8.5, color="#333",
        )

    ax.set_xlabel("|Spearman ρ|", fontsize=11)
    ax.set_title("Best SAE Feature Per Telemetry Concept", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(rho_vals) * 1.30 if rho_vals else 1.0)
    ax.axvline(0.3, color="#bbb", linewidth=0.8, linestyle="--", zorder=0)
    ax.axvline(0.5, color="#aaa", linewidth=0.8, linestyle="--", zorder=0)

    fig.tight_layout()
    save_figure(fig, fig_dir, "best_per_concept")
    plt.close(fig)


# ===================================================================
# 2. Label Distribution
# ===================================================================

def label_distribution(
    annotations: List[Dict[str, Any]],
    fig_dir: str = _FIG_DIR,
) -> None:
    """Bar chart of label counts + pie chart of dead vs active (PNG + PDF)."""
    _ensure_fig_dir(fig_dir)

    counts: Dict[str, int] = defaultdict(int)
    n_dead = 0
    n_active = 0

    for ann in annotations:
        raw_label = ann.get("label", "unknown")
        if raw_label == "dead":
            n_dead += 1
        else:
            n_active += 1
            clean = _strip_rho_suffix(raw_label)
            counts[clean] += 1

    sorted_labels = sorted(counts.keys(), key=lambda k: counts[k], reverse=True)
    sorted_counts = [counts[k] for k in sorted_labels]
    n_labels = len(sorted_labels)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, max(5, n_labels * 0.38)),
        gridspec_kw={"width_ratios": [2.5, 1]},
    )

    # --- Bar chart ---
    palette = sns.color_palette(PHD_MAKO + "_r", n_colors=max(n_labels, 2))
    ax_bar = axes[0]
    ax_bar.barh(
        sorted_labels[::-1], sorted_counts[::-1],
        color=palette[::-1], edgecolor="white", linewidth=0.5, height=0.7,
    )
    ax_bar.set_xlabel("Number of Features", fontsize=11)
    ax_bar.set_title("Feature Label Distribution", fontsize=13, fontweight="bold")
    for i, (lbl, cnt) in enumerate(zip(sorted_labels[::-1], sorted_counts[::-1])):
        ax_bar.text(cnt + 0.25, i, str(cnt), va="center", fontsize=9, color="#333")

    # --- Pie chart ---
    ax_pie = axes[1]
    pie_data   = [n_active, n_dead]
    pie_labels = [f"Active\n({n_active})", f"Dead\n({n_dead})"]
    pie_colours = [PHD_TEAL, PHD_GRAY]
    wedges, texts, autotexts = ax_pie.pie(
        pie_data, labels=pie_labels, colors=pie_colours,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 10},
        wedgeprops={"edgecolor": "white", "linewidth": 1.8},
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_color("white")
        t.set_fontweight("bold")
    ax_pie.set_title("Dead vs Active Features", fontsize=12, fontweight="bold")

    fig.tight_layout()
    save_figure(fig, fig_dir, "label_distribution")
    plt.close(fig)

    # Print summary
    print("\n--- Label Distribution ---")
    for lbl, cnt in zip(sorted_labels, sorted_counts):
        print(f"  {lbl:<40s}  {cnt:>4d} features")
    print(f"  {'DEAD':<40s}  {n_dead:>4d} features")
    print(f"  Total: {n_active + n_dead} features ({n_active} active, {n_dead} dead)")


# ===================================================================
# 3. Top Features Per Concept Group — Heatmap
# ===================================================================

def concept_group_heatmap(
    annotations: List[Dict[str, Any]],
    tel_keys: List[str],
    top_n: int = 3,
    fig_dir: str = _FIG_DIR,
) -> None:
    """Group features by label, take top-N per group, plot ρ heatmap (PNG + PDF)."""
    _ensure_fig_dir(fig_dir)
    active = [a for a in annotations if a.get("label", "") != "dead"]

    # Group by cleaned label
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for ann in active:
        clean = _strip_rho_suffix(ann.get("label", ""))
        groups[clean].append(ann)

    # For each group, keep top_n by |ρ| (selectivity_score)
    rows_labels: List[str] = []
    rows_data: List[List[float]] = []

    for group_label in sorted(groups.keys()):
        members = sorted(
            groups[group_label],
            key=lambda a: a.get("selectivity_score", 0),
            reverse=True,
        )[:top_n]
        for ann in members:
            rho_scores = ann.get("all_rho_scores", {})
            row = [rho_scores.get(k, 0.0) for k in tel_keys]
            rows_data.append(row)
            rows_labels.append(f"f{ann['feature_idx']} — {group_label}")

    if not rows_data:
        print("[Analysis] No active features for concept group heatmap.")
        return

    mat = np.array(rows_data)

    fig_height = max(5, len(rows_labels) * 0.42)
    fig_width  = max(9, len(tel_keys) * 0.85)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        mat,
        xticklabels=[k.replace("_", "\n") for k in tel_keys],
        yticklabels=rows_labels,
        cmap=PHD_DIVERGING,
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.4,
        linecolor="#ddd",
        cbar_kws={"label": "Spearman ρ", "shrink": 0.70, "aspect": 20},
        annot=True, fmt=".2f", annot_kws={"fontsize": 7.5},
        ax=ax,
    )
    ax.set_title(
        f"Top-{top_n} Features Per Concept Group — Full ρ Profile",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Telemetry Field", fontsize=11)
    ax.set_ylabel("Feature (Group)", fontsize=11)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    save_figure(fig, fig_dir, "concept_group_heatmap")
    plt.close(fig)


# ===================================================================
# 4. Concept Correlation Summary  (NEW — replaces clustermap)
# ===================================================================

def concept_correlation_summary(
    annotations: List[Dict[str, Any]],
    fig_dir: str = _FIG_DIR,
) -> None:
    """Compact publication-quality summary of all unique driving concepts.

    For each unique concept label the figure shows:
    * A **lollipop** segment from 0 to the best |ρ| (colour-coded positive /
      negative).
    * A **bubble** whose area encodes the number of features that share the
      same concept label (i.e. how many SAE dimensions encode this concept).

    The figure is sorted by best |ρ| so the most selective concepts are at the
    top.  This replaces the crowded clustermap with a publication-quality,
    information-dense alternative suitable for a PhD thesis.

    Saved as PNG + PDF.
    """
    _ensure_fig_dir(fig_dir)
    active = [a for a in annotations if a.get("label", "") != "dead"]
    if not active:
        print("[Analysis] No active features — skipping concept_correlation_summary.")
        return

    # ── Aggregate per concept ──────────────────────────────────────────────
    # concept → {'best_rho': float, 'n_features': int, 'direction': str}
    concept_stats: Dict[str, Dict] = {}

    for ann in active:
        clean  = _strip_rho_suffix(ann.get("label", ""))
        rho_scores = ann.get("all_rho_scores", {})
        if not rho_scores:
            continue
        # Best |ρ| across all telemetry fields
        best_abs  = max(abs(v) for v in rho_scores.values())
        best_rho  = max(rho_scores.values(), key=abs)
        direction = "positive" if best_rho >= 0 else "negative"

        if clean not in concept_stats:
            concept_stats[clean] = {
                "best_abs_rho": best_abs,
                "best_rho":     best_rho,
                "direction":    direction,
                "n_features":   0,
            }
        # Update if this feature is more selective
        if best_abs > concept_stats[clean]["best_abs_rho"]:
            concept_stats[clean]["best_abs_rho"] = best_abs
            concept_stats[clean]["best_rho"]     = best_rho
            concept_stats[clean]["direction"]    = direction
        concept_stats[clean]["n_features"] += 1

    if not concept_stats:
        print("[Analysis] No concept stats — skipping concept_correlation_summary.")
        return

    # Sort by best |ρ| ascending (plot: highest → top)
    sorted_concepts = sorted(
        concept_stats.items(), key=lambda kv: kv[1]["best_abs_rho"]
    )
    labels    = [kv[0] for kv in sorted_concepts]
    rho_vals  = [kv[1]["best_abs_rho"]  for kv in sorted_concepts]
    n_feats   = [kv[1]["n_features"]    for kv in sorted_concepts]
    directions = [kv[1]["direction"]    for kv in sorted_concepts]

    n = len(labels)
    y = np.arange(n)

    # ── Figure layout ──────────────────────────────────────────────────────
    fig_h = max(5, n * 0.52)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    # Colour map: positive correlations → teal, negative → orange
    bar_colors = [PHD_TEAL if d == "positive" else PHD_ORANGE for d in directions]

    # ── Lollipop stems ─────────────────────────────────────────────────────
    ax.hlines(y, 0, rho_vals, colors=bar_colors, linewidth=2.0, alpha=0.65, zorder=2)

    # ── Bubble markers (area ∝ n_features) ────────────────────────────────
    min_area, max_area = 40, 400
    n_arr   = np.array(n_feats, dtype=float)
    areas   = min_area + (n_arr - n_arr.min()) / max(n_arr.max() - n_arr.min(), 1) * (max_area - min_area)

    scatter = ax.scatter(
        rho_vals, y,
        s=areas, c=bar_colors,
        edgecolors="white", linewidths=0.8,
        zorder=3, alpha=0.92,
    )

    # ── Inline annotations ─────────────────────────────────────────────────
    for xi, yi, n_f, rho in zip(rho_vals, y, n_feats, rho_vals):
        ax.text(xi + 0.012, yi, f"|ρ|={rho:.3f}  n={n_f}",
                va="center", fontsize=8, color="#333")

    # ── Reference lines ────────────────────────────────────────────────────
    for xref, ls in [(0.3, "--"), (0.5, "-.")]:
        ax.axvline(xref, color="#bbb", linewidth=0.9, linestyle=ls, zorder=1)
        ax.text(xref + 0.005, -0.6, f"ρ={xref}", fontsize=7.5, color="#aaa",
                rotation=90, va="bottom")

    # ── Axes dressing ─────────────────────────────────────────────────────
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("|Spearman ρ| (best across all telemetry fields)", fontsize=11)
    ax.set_title(
        "Concept Correlation Summary — Unique Driving Concepts",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlim(-0.02, min(max(rho_vals) * 1.35, 1.05))
    ax.set_ylim(-1, n)

    # ── Legend ─────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(facecolor=PHD_TEAL,   label="Positive correlation"),
        mpatches.Patch(facecolor=PHD_ORANGE, label="Negative correlation"),
    ]
    # Bubble size legend entries
    for n_val, label_str in [(1, "1 feature"), (5, "5 features")]:
        a = min_area + (n_val - n_arr.min()) / max(n_arr.max() - n_arr.min(), 1) * (max_area - min_area)
        legend_patches.append(
            plt.scatter([], [], s=a, c=PHD_GRAY, label=label_str,
                        edgecolors="white", linewidths=0.6)
        )
    ax.legend(
        handles=legend_patches,
        loc="lower right",
        fontsize=8.5,
        framealpha=0.9,
        title="Legend",
        title_fontsize=8,
    )

    ax.grid(axis="x", linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    save_figure(fig, fig_dir, "concept_correlation_summary")
    plt.close(fig)
    print(f"[Analysis] concept_correlation_summary → {fig_dir}/")


# ===================================================================
# 5. Selectivity Histogram
# ===================================================================

def selectivity_histogram(
    annotations: List[Dict[str, Any]],
    fig_dir: str = _FIG_DIR,
) -> None:
    """Histogram of activation counts across all features (PNG + PDF)."""
    _ensure_fig_dir(fig_dir)

    n_acts = [a.get("n_activations", 0) for a in annotations]

    fig, ax = plt.subplots(figsize=(10, 5))

    n_acts_arr = np.array(n_acts)
    nonzero    = n_acts_arr[n_acts_arr > 0]

    if len(nonzero) > 0:
        bins = np.logspace(
            np.log10(max(1, nonzero.min())),
            np.log10(nonzero.max() + 1),
            50,
        )
        ax.hist(
            nonzero, bins=bins,
            color=PHD_BLUE, edgecolor="white", linewidth=0.5, alpha=0.85,
        )
        ax.set_xscale("log")
    else:
        ax.hist(n_acts, bins=30, color=PHD_BLUE, edgecolor="white",
                linewidth=0.5, alpha=0.85)

    # Dead threshold
    vline = ax.axvline(
        x=10, color="#E53935", linestyle="--", linewidth=1.5, alpha=0.85,
        label="Dead threshold (n < 10)",
    )
    ax.legend(fontsize=9)

    ax.set_xlabel("Number of Timesteps Feature Activates (log scale)", fontsize=11)
    ax.set_ylabel("Count of Features", fontsize=11)
    ax.set_title(
        "Feature Selectivity — Activation Count Distribution",
        fontsize=13, fontweight="bold",
    )

    # Summary stats text box
    n_dead    = sum(1 for n in n_acts if n < 10)
    median_a  = int(np.median(n_acts)) if n_acts else 0
    mean_a    = np.mean(n_acts) if n_acts else 0
    textstr   = (
        f"Total features: {len(n_acts)}\n"
        f"Dead (n<10): {n_dead}\n"
        f"Median activations: {median_a}\n"
        f"Mean activations: {mean_a:.0f}"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor="white",
                 edgecolor="#ccc", alpha=0.92)
    ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=props)

    fig.tight_layout()
    save_figure(fig, fig_dir, "selectivity_histogram")
    plt.close(fig)


# ===================================================================
# 6. Export Feature Index Map
# ===================================================================

def export_feature_index_map(
    annotations: List[Dict[str, Any]],
    fig_dir: str = _FIG_DIR,
    output_filename: str = "feature_index_map.json",
) -> Dict[str, int]:
    """Build and save a ``{concept_name: best_feature_idx}`` mapping.

    Rules applied
    -------------
    * **Dead features are excluded** — any annotation whose label is ``'dead'``
      is skipped entirely.
    * **Zero-correlation features are excluded** — if every ρ value in
      ``all_rho_scores`` is exactly 0 (or the dict is absent), the feature is
      ignored.
    * **Per-concept deduplication** — multiple SAE features can receive the
      same concept label (e.g. ``'high_ego_speed'``).  For each concept the
      feature with the highest ``max |ρ|`` across all telemetry fields is kept.

    The resulting JSON is saved next to the other analysis figures.

    Args:
        annotations: List of per-feature annotation dicts.
        fig_dir: Directory where the JSON file is written.
        output_filename: Filename for the JSON output.

    Returns:
        The ``{concept_name: feature_idx}`` dict that was saved.
    """
    _ensure_fig_dir(fig_dir)

    # concept_name → (feature_idx, max_abs_rho)
    best_per_concept: Dict[str, tuple] = {}

    for ann in annotations:
        label = ann.get("label", "")

        if label == "dead":
            continue

        rho_scores = ann.get("all_rho_scores", {})
        if not rho_scores:
            continue

        abs_rhos = [abs(v) for v in rho_scores.values()]
        max_abs_rho = max(abs_rhos) if abs_rhos else 0.0

        if max_abs_rho == 0.0:
            continue

        concept  = _strip_rho_suffix(label)
        feat_idx = ann["feature_idx"]

        if concept not in best_per_concept or max_abs_rho > best_per_concept[concept][1]:
            best_per_concept[concept] = (feat_idx, max_abs_rho)

    # Flatten to {concept: idx}
    feature_map: Dict[str, int] = {
        concept: feat_idx
        for concept, (feat_idx, _) in sorted(best_per_concept.items())
    }

    out_path = os.path.join(fig_dir, output_filename)
    with open(out_path, "w") as f:
        json.dump(feature_map, f, indent=2)

    print(f"[Analysis] Feature index map ({len(feature_map)} concepts) → {out_path}")

    # Print table
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║        Feature Index Map (concept → best idx)        ║")
    print("╠══════════════════════════════════╦══════════╦════════╣")
    print("║ Concept                          ║ feat_idx ║ max|ρ| ║")
    print("╠══════════════════════════════════╬══════════╬════════╣")
    for concept, (feat_idx, max_rho) in sorted(
        best_per_concept.items(), key=lambda x: -x[1][1]
    ):
        print(f"║ {concept:<32s} ║ {feat_idx:>8d} ║ {max_rho:>6.4f} ║")
    print("╚══════════════════════════════════╩══════════╩════════╝")

    return feature_map


# ===================================================================
# Master run_analysis
# ===================================================================

def run_analysis(
    annotations: List[Dict[str, Any]],
    tel_keys: List[str],
    fig_dir: str = _FIG_DIR,
) -> Dict[str, int]:
    """Run all grouped analyses and generate all publication-quality figures.

    Figures are saved in both PNG (300 dpi) and PDF (vector) formats.

    Args:
        annotations: The list of per-feature annotation dicts (from the
            ``feature_annotator.annotate`` return value, or loaded from
            the annotations JSON under the ``'annotations'`` key).
        tel_keys: Ordered list of telemetry field names (column names for
            the correlation matrices).
        fig_dir: Output directory for figures.  Sub-directories are created
            automatically.

    Returns:
        The ``{concept_name: feature_idx}`` mapping saved to
        ``feature_index_map.json``.
    """
    print("\n" + "=" * 64)
    print("  SAE Feature Analysis — Grouped Visualisations")
    print("=" * 64)

    best_feature_per_concept(annotations, tel_keys, fig_dir)
    label_distribution(annotations, fig_dir)
    concept_group_heatmap(annotations, tel_keys, fig_dir=fig_dir)
    concept_correlation_summary(annotations, fig_dir=fig_dir)  # NEW (replaces clustermap)
    selectivity_histogram(annotations, fig_dir)
    feature_map = export_feature_index_map(annotations, fig_dir)

    print("\n" + "=" * 64)
    print(f"  All figures saved to {fig_dir}/  (PNG + PDF)")
    print("=" * 64 + "\n")

    return feature_map
