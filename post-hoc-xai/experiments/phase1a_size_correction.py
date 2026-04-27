"""Phase 1a — Size-corrected attribution normalization.

Applies per-feature size correction to all existing cached category importance
data and generates comparison plots: original vs size-corrected.

Datasets used (no model loading required):
  - event_xai_results/event_0{0,1,2}_*.json   (VG + IG time series)
  - scenario002_all_methods/summary_metrics.csv (7 methods, t=35)

Output: experiments/phase1a_results/
  - correction_factors.png       — how much each category is amplified/suppressed
  - event{00,01,02}_comparison.png — original vs corrected time series per event
  - scenario002_snapshot.png     — original vs corrected at t=35 for all 7 methods
  - findings.md                  — numerical summary
"""

import json
import csv
import sys
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_ROOT = _HERE.parent

# Load normalization module directly to avoid triggering posthoc_xai's JAX
# imports (the top-level __init__ imports jax which requires the vmax env).
_norm_path = _ROOT / "posthoc_xai" / "utils" / "normalization.py"
_norm_spec = importlib.util.spec_from_file_location("normalization", _norm_path)
_norm_mod = importlib.util.module_from_spec(_norm_spec)
_norm_spec.loader.exec_module(_norm_mod)

size_correct_attribution = _norm_mod.size_correct_attribution
correction_factors = _norm_mod.correction_factors
CATEGORY_FEATURE_COUNTS = _norm_mod.CATEGORY_FEATURE_COUNTS

EVENT_DIR = _HERE / "event_xai_results"
S002_CSV  = _HERE / "scenario002_all_methods" / "summary_metrics.csv"
OUT_DIR   = _HERE / "phase1a_results"
OUT_DIR.mkdir(exist_ok=True)

CATS  = ["sdc_trajectory", "other_agents", "roadgraph", "traffic_lights", "gps_path"]
LABELS = ["SDC", "Agents", "Road", "TL", "GPS"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

# ── helpers ──────────────────────────────────────────────────────────────────

def load_event(fname: str) -> dict:
    with open(EVENT_DIR / fname) as f:
        return json.load(f)


def cat_series_original(event_data: dict, method: str) -> dict[str, list]:
    return event_data["methods"][method]["category_series"]


def cat_series_corrected(event_data: dict, method: str) -> dict[str, list]:
    orig = event_data["methods"][method]["category_series"]
    timesteps = event_data["timesteps"]
    corrected = {c: [] for c in CATS}
    for i in range(len(timesteps)):
        snapshot = {c: orig[c][i] for c in CATS}
        sc = size_correct_attribution(snapshot)
        for c in CATS:
            corrected[c].append(sc[c])
    return corrected


def load_scenario002() -> list[dict]:
    """Returns list of dicts with method + category importances."""
    rows = []
    with open(S002_CSV) as f:
        for row in csv.DictReader(f):
            rows.append({
                "method": row["method"],
                **{c: float(row[f"cat_{c}"]) for c in CATS},
            })
    return rows

# ── plot 1: correction factors ────────────────────────────────────────────────

def plot_correction_factors():
    factors = correction_factors()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    vals = [factors[c] for c in CATS]
    bars = ax.bar(LABELS, vals, color=COLORS, alpha=0.85, edgecolor="white")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="no change")
    ax.set_ylabel("Correction multiplier  (× original)")
    ax.set_title("Per-feature size-correction factors\n"
                 "(relative to gps_path = 20 features, the smallest category)")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"×{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "correction_factors.png", dpi=150)
    plt.close(fig)
    print("  saved: correction_factors.png")


# ── plot 2: event time series ─────────────────────────────────────────────────

def plot_event_comparison(event_fname: str, out_name: str, title: str):
    data = load_event(event_fname)
    timesteps = data["timesteps"]
    methods = list(data["methods"].keys())  # VG, IG

    fig, axes = plt.subplots(len(methods), 2, figsize=(12, 4 * len(methods)),
                             sharey=False)
    if len(methods) == 1:
        axes = [axes]

    for row_i, method in enumerate(methods):
        orig = cat_series_original(data, method)
        corr = cat_series_corrected(data, method)

        for col_i, (series, label) in enumerate([(orig, "Original"), (corr, "Size-corrected")]):
            ax = axes[row_i][col_i]
            for c, lbl, col in zip(CATS, LABELS, COLORS):
                ax.plot(timesteps, series[c], label=lbl, color=col, linewidth=1.8)
            ax.set_title(f"{method}  —  {label}", fontsize=9)
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Category importance")
            ax.set_ylim(0, 1)
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / out_name, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_name}")


# ── plot 3: scenario002 snapshot (all 7 methods) ─────────────────────────────

def plot_scenario002_snapshot():
    rows = load_scenario002()
    methods = [r["method"] for r in rows]
    n = len(methods)
    x = np.arange(len(CATS))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    orig_matrix  = np.array([[r[c] for c in CATS] for r in rows])
    corr_matrix  = np.array([
        [size_correct_attribution({c: r[c] for c in CATS})[c] for c in CATS]
        for r in rows
    ])

    method_colors = plt.cm.tab10(np.linspace(0, 0.9, n))

    for ax, matrix, title in zip(
        axes, [orig_matrix, corr_matrix],
        ["Original normalization", "Size-corrected normalization"]
    ):
        for i, (method, color) in enumerate(zip(methods, method_colors)):
            offset = (i - n / 2) * (width / n)
            ax.bar(x + offset, matrix[i], width / n, label=method,
                   color=color, alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS)
        ax.set_ylabel("Category importance")
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Scenario 002, t=35  —  All 7 methods", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scenario002_snapshot.png", dpi=150)
    plt.close(fig)
    print("  saved: scenario002_snapshot.png")


# ── numerical summary ─────────────────────────────────────────────────────────

def compute_summary():
    """Print and return key numbers for the findings note."""
    print("\n=== NUMERICAL SUMMARY ===")
    print(f"\nFeature counts: {CATEGORY_FEATURE_COUNTS}")
    print(f"\nCorrection factors (× original importance):")
    for cat, factor in correction_factors().items():
        print(f"  {cat:20s}: ×{factor:.4f}")

    # Event 02 IG at peak (t=35, last timestep)
    data = load_event("event_02_s000.json")
    ig_series = data["methods"]["integrated_gradients"]["category_series"]
    peak_idx = len(data["timesteps"]) - 1
    ig_peak = {c: ig_series[c][peak_idx] for c in CATS}
    ig_peak_corr = size_correct_attribution(ig_peak)

    print("\nEvent_02 IG at t=35 (peak):")
    print(f"  {'Category':20s}  {'Original':>10}  {'Corrected':>10}  {'Change':>10}")
    for c, lbl in zip(CATS, LABELS):
        orig_v = ig_peak[c]
        corr_v = ig_peak_corr[c]
        delta = corr_v - orig_v
        print(f"  {lbl:20s}  {orig_v:10.3f}  {corr_v:10.3f}  {delta:+10.3f}")

    # Scenario 002 all methods
    rows = load_scenario002()
    print("\nScenario 002 t=35 — Roadgraph importance, original vs corrected:")
    print(f"  {'Method':25s}  {'Road (orig)':>12}  {'Road (corr)':>12}")
    for r in rows:
        orig_road = r["roadgraph"]
        corr_road = size_correct_attribution({c: r[c] for c in CATS})["roadgraph"]
        print(f"  {r['method']:25s}  {orig_road:12.3f}  {corr_road:12.3f}")

    print("\nScenario 002 t=35 — GPS importance, original vs corrected:")
    print(f"  {'Method':25s}  {'GPS (orig)':>12}  {'GPS (corr)':>12}")
    for r in rows:
        orig_gps = r["gps_path"]
        corr_gps = size_correct_attribution({c: r[c] for c in CATS})["gps_path"]
        print(f"  {r['method']:25s}  {orig_gps:12.3f}  {corr_gps:12.3f}")

    return ig_peak, ig_peak_corr, rows


# ── findings markdown ─────────────────────────────────────────────────────────

def write_findings(ig_peak, ig_peak_corr, s002_rows):
    lines = ["# Phase 1a Findings — Size-Corrected Attribution\n"]
    lines.append(
        "## What size correction does\n\n"
        "Current pipeline: `cat_imp[c] = sum(abs(raw[c])) / sum(abs(raw_all))` — "
        "sums over ALL features in the category. Larger categories accumulate more "
        "attribution purely from count.\n\n"
        "Correction: `corrected[c] = (cat_imp[c] / n_c) / Σ(cat_imp[c'] / n_c')` — "
        "divides by feature count before renormalizing. Answers 'which category has "
        "the most influence **per input dimension**?' instead of total.\n"
    )

    lines.append("\n## Correction factors (relative to GPS = 20 features)\n\n")
    lines.append("| Category | Features | Multiplier |\n|---|---|---|\n")
    for cat, lbl in zip(CATS, LABELS):
        n = CATEGORY_FEATURE_COUNTS[cat]
        f = correction_factors()[cat]
        lines.append(f"| {lbl} | {n} | ×{f:.3f} |\n")

    lines.append(
        "\nRoadgraph (1000 features) gets a ×0.020 multiplier — "
        "its importance is divided by 50× relative to GPS.\n"
    )

    lines.append("\n## Event_02 IG at peak (t=35)\n\n")
    lines.append("| Category | Original | Corrected | Change |\n|---|---|---|---|\n")
    for c, lbl in zip(CATS, LABELS):
        orig_v = ig_peak[c]
        corr_v = ig_peak_corr[c]
        lines.append(f"| {lbl} | {orig_v:.3f} | {corr_v:.3f} | {corr_v - orig_v:+.3f} |\n")

    lines.append("\n## Scenario 002 t=35 — Roadgraph original vs corrected\n\n")
    lines.append("| Method | Road (orig) | Road (corr) | GPS (orig) | GPS (corr) |\n|---|---|---|---|---|\n")
    for r in s002_rows:
        orig_road = r["roadgraph"]
        orig_gps = r["gps_path"]
        sc = size_correct_attribution({c: r[c] for c in CATS})
        corr_road = sc["roadgraph"]
        corr_gps = sc["gps_path"]
        lines.append(f"| {r['method']} | {orig_road:.3f} | {corr_road:.3f} | {orig_gps:.3f} | {corr_gps:.3f} |\n")

    lines.append(
        "\n## Interpretation\n\n"
        "Size correction dramatically changes the *ranking* of categories. Roadgraph "
        "drops from dominant (~50–70% original) to near-zero per-feature importance, "
        "while GPS rises sharply.\n\n"
        "**What this means for the thesis:**\n"
        "- The 'roadgraph dominates' finding is correct as a statement about *total* "
        "attribution — the model's output is most sensitive to the aggregate road "
        "geometry signal.\n"
        "- But each individual GPS waypoint carries far more information per feature "
        "than each roadgraph point — GPS is a high-density, compact signal.\n"
        "- Both perspectives should be reported. The thesis chapter should clarify "
        "which normalization is used and why total attribution is the appropriate "
        "primary metric (the model processes all 1000 roadgraph features together, "
        "not one at a time).\n"
        "- Size-corrected numbers should be included as a robustness note or table.\n"
    )

    findings_path = OUT_DIR / "findings.md"
    with open(findings_path, "w") as f:
        f.writelines(lines)
    print(f"  saved: findings.md")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Phase 1a — Size-corrected attribution normalization")
    print(f"Output: {OUT_DIR}\n")

    print("Plotting correction factors...")
    plot_correction_factors()

    print("Plotting event time series comparisons...")
    plot_event_comparison(
        "event_00_s001.json", "event00_comparison.png",
        "Event 00 — Evasive steering (s001, MEDIUM)"
    )
    plot_event_comparison(
        "event_01_s000.json", "event01_comparison.png",
        "Event 01 — Evasive steering (s000, LOW)"
    )
    plot_event_comparison(
        "event_02_s000.json", "event02_comparison.png",
        "Event 02 — Hazard onset (s000, CRITICAL)"
    )

    print("Plotting scenario002 all-methods snapshot...")
    plot_scenario002_snapshot()

    ig_peak, ig_peak_corr, s002_rows = compute_summary()

    print("\nWriting findings note...")
    write_findings(ig_peak, ig_peak_corr, s002_rows)

    print("\nDone. Review phase1a_results/ for all outputs.")


if __name__ == "__main__":
    main()
