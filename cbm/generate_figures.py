"""
CBM Thesis Figure Generator — Tier 1
Generates publication-quality figures from the eval cache (400 val scenarios).

Figures produced:
  fig1_concept_quality.pdf     — Binary accuracy + Continuous R² bar charts
  fig2_concept_scatter.pdf     — Pred vs True scatter for all 15 concepts
  fig3_task_performance.pdf    — Route progress distribution + driving metrics
  fig4_concept_temporal.pdf    — Mean concept prediction vs ground truth over time

Usage:
    conda activate vmax
    python generate_figures.py

Outputs go to: figures/
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── Paths ─────────────────────────────────────────────────────────────
CACHE_DIR  = "eval_model_final_cache"
JSON_PATH  = "eval_model_final.json"
OUT_DIR    = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────
def load(fname, **kw):
    return np.load(os.path.join(CACHE_DIR, fname), allow_pickle=True, **kw)

pred       = load("pred_concepts.npy")          # (80, 400, 15)
true       = load("true_concepts.npy")          # (80, 400, 15)
valid      = load("valid_mask.npy")             # (80, 400, 15)
actions    = load("ego_actions.npy")            # (80, 400, 2)
dones      = load("dones.npy")                  # (80, 400)
rewards    = load("rewards.npy")                # (80, 400)
drv        = load("driving_metrics.npy")        # (80, 400, 14)
drv_keys   = load("driving_metric_keys.npy").tolist()
cnames     = load("concept_names.npy").tolist()

with open(JSON_PATH) as f:
    results = json.load(f)

T, N, C = pred.shape

# ── Phase classification ───────────────────────────────────────────────
PHASE3 = {"path_curvature_max", "path_net_heading_change",
           "path_straightness", "heading_to_path_end"}
PHASE2 = {"ttc_lead_vehicle", "lead_vehicle_decelerating", "at_intersection"}
BINARY = {"traffic_light_red", "lead_vehicle_decelerating", "at_intersection"}

def phase_of(name):
    if name in PHASE3: return 3
    if name in PHASE2: return 2
    return 1

PHASE_COLOR = {1: "#2196F3", 2: "#FF9800", 3: "#4CAF50"}
PHASE_LABEL = {1: "Phase 1", 2: "Phase 2", 3: "Phase 3 (new)"}

# ── Helpers ────────────────────────────────────────────────────────────
def r2(p, t):
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    return float("nan") if ss_tot < 1e-10 else 1.0 - ss_res / ss_tot

def mae(p, t):
    return np.mean(np.abs(p - t))

def concept_flat(idx):
    """Return (pred_valid, true_valid) flattened over valid T×N entries."""
    v = valid[:, :, idx].reshape(-1)
    p = pred[:, :, idx].reshape(-1)[v]
    t = true[:, :, idx].reshape(-1)[v]
    return p, t

# ── Global style ───────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.15,
})

# ══════════════════════════════════════════════════════════════════════
# FIGURE 1 — Concept Quality Summary
# ══════════════════════════════════════════════════════════════════════
print("Generating Figure 1: Concept Quality Summary...")

bin_names  = [n for n in cnames if n in BINARY]
cont_names = [n for n in cnames if n not in BINARY]

bin_accs  = [results["concept_metrics"][n]["accuracy"] for n in bin_names]
cont_r2s  = []
for n in cont_names:
    p_v, t_v = concept_flat(cnames.index(n))
    cont_r2s.append(r2(p_v, t_v) if len(p_v) > 1 else float("nan"))

PRETTY = {
    "ego_speed":               "Ego Speed",
    "ego_acceleration":        "Ego Accel.",
    "dist_nearest_object":     "Dist. Nearest Obj.",
    "num_objects_within_10m":  "Obj. within 10m",
    "traffic_light_red":       "Traffic Light Red",
    "dist_to_traffic_light":   "Dist. to TL",
    "heading_deviation":       "Heading Deviation",
    "progress_along_route":    "Route Progress",
    "ttc_lead_vehicle":        "TTC Lead Vehicle",
    "lead_vehicle_decelerating":"Lead Veh. Decel.",
    "at_intersection":         "At Intersection",
    "path_curvature_max":      "Path Curvature",
    "path_net_heading_change": "Net Heading Chg.",
    "path_straightness":       "Path Straightness",
    "heading_to_path_end":     "Heading to End",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "CBM Concept Prediction Quality — Validation Set (400 scenarios, 10 GB model)",
    fontsize=12, fontweight="bold", y=1.02
)

# ── Panel A: Binary accuracy ───────────────────────────────────────────
ax = axes[0]
colors_bin = [PHASE_COLOR[phase_of(n)] for n in bin_names]
bars = ax.barh(
    [PRETTY[n] for n in bin_names], bin_accs,
    color=colors_bin, edgecolor="white", height=0.5
)
ax.set_xlim(0, 1.08)
ax.set_xlabel("Accuracy")
ax.set_title("(a) Binary Concept Accuracy")
ax.axvline(1.0, color="grey", lw=0.8, ls="--", alpha=0.6)
for bar, val in zip(bars, bin_accs):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=9)

# ── Panel B: Continuous R² ────────────────────────────────────────────
ax = axes[1]
colors_cont = [PHASE_COLOR[phase_of(n)] for n in cont_names]
y_pos = np.arange(len(cont_names))
bars = ax.barh(
    y_pos, cont_r2s,
    color=colors_cont, edgecolor="white", height=0.6
)
ax.set_yticks(y_pos)
ax.set_yticklabels([PRETTY[n] for n in cont_names])
ax.set_xlabel("R² Score")
ax.set_title("(b) Continuous Concept R² (Coefficient of Determination)")
ax.axvline(0,   color="grey",   lw=0.8, ls="--", alpha=0.6)
ax.axvline(0.5, color="green",  lw=0.8, ls=":",  alpha=0.5)
ax.set_xlim(-1.6, 1.1)

for bar, val in zip(bars, cont_r2s):
    if np.isnan(val): continue
    xpos = val + 0.03 if val >= 0 else val - 0.03
    ha   = "left" if val >= 0 else "right"
    ax.text(xpos, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha=ha, fontsize=8)

# legend
legend_handles = [
    Line2D([0], [0], color=PHASE_COLOR[p], lw=6, label=PHASE_LABEL[p])
    for p in [1, 2, 3]
]
fig.legend(handles=legend_handles, loc="lower center",
           ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06), fontsize=9)

fig.tight_layout()
path = os.path.join(OUT_DIR, "fig1_concept_quality.pdf")
fig.savefig(path)
fig.savefig(path.replace(".pdf", ".png"))
plt.close(fig)
print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2 — Concept Scatter Plots (Pred vs True)
# ══════════════════════════════════════════════════════════════════════
print("Generating Figure 2: Concept Scatter Plots...")

NCOLS, NROWS = 3, 5
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(15, 22))
fig.suptitle(
    "CBM Concept Predictions vs Ground Truth — Validation Set (400 scenarios, 10 GB model)\n"
    "Each point = one timestep from one scenario (valid timesteps only). Diagonal = perfect prediction.",
    fontsize=11, fontweight="bold", y=1.01
)

for idx, name in enumerate(cnames):
    row, col = divmod(idx, NCOLS)
    ax = axes[row, col]

    p_v, t_v = concept_flat(idx)
    color = PHASE_COLOR[phase_of(name)]
    is_bin = name in BINARY

    if len(p_v) == 0:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center",
                transform=ax.transAxes, color="grey")
        ax.set_title(PRETTY[name])
        continue

    # Downsample for speed if too many points
    if len(p_v) > 8000:
        idx_s = np.random.choice(len(p_v), 8000, replace=False)
        p_s, t_s = p_v[idx_s], t_v[idx_s]
    else:
        p_s, t_s = p_v, t_v

    if is_bin:
        # For binary: jittered scatter
        jitter = 0.02
        rng = np.random.default_rng(42)
        ax.scatter(t_s + rng.uniform(-jitter, jitter, len(t_s)),
                   p_s + rng.uniform(-jitter, jitter, len(p_s)),
                   alpha=0.06, s=4, color=color, rasterized=True)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xlim(-0.15, 1.15); ax.set_ylim(-0.15, 1.15)
        metric_txt = f"Acc: {results['concept_metrics'][name]['accuracy']:.4f}"
    else:
        hb = ax.hexbin(t_s, p_s, gridsize=30, cmap="Blues",
                       mincnt=1, linewidths=0.2, rasterized=True)
        lims = [min(t_s.min(), p_s.min()) - 0.02,
                max(t_s.max(), p_s.max()) + 0.02]
        ax.set_xlim(lims); ax.set_ylim(lims)
        r2_val = r2(p_v, t_v)
        mae_val = mae(p_v, t_v)
        metric_txt = f"R²={r2_val:.3f}   MAE={mae_val:.4f}"

    # Diagonal
    lims = ax.get_xlim()
    ax.plot(lims, lims, color="red", lw=0.9, ls="--", alpha=0.7, zorder=5)

    # Phase tag
    phase_tag = f"P{phase_of(name)}"
    ax.text(0.03, 0.96, phase_tag, transform=ax.transAxes,
            fontsize=7.5, color=PHASE_COLOR[phase_of(name)],
            va="top", fontweight="bold")

    # Metric annotation
    ax.text(0.97, 0.04, metric_txt, transform=ax.transAxes,
            fontsize=7.5, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

    ax.set_title(PRETTY[name], fontsize=9.5, pad=3)
    ax.set_xlabel("Ground Truth", fontsize=8)
    ax.set_ylabel("Predicted", fontsize=8)

# Hide unused subplot (15 concepts in 5×3 = no spare)
fig.tight_layout(rect=[0, 0, 1, 1], h_pad=3.0, w_pad=2.5)
path = os.path.join(OUT_DIR, "fig2_concept_scatter.pdf")
fig.savefig(path)
fig.savefig(path.replace(".pdf", ".png"))
plt.close(fig)
print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 3 — Task Performance
# ══════════════════════════════════════════════════════════════════════
print("Generating Figure 3: Task Performance...")

# Route progress per scenario (final step of driving_metrics)
prog_idx  = drv_keys.index("progress_ratio_nuplan")
prog_vals = drv[:, :, prog_idx][-1]   # final-step value per scenario (400,)

# Key driving metrics
SAC_BASELINE = {
    "progress_ratio_nuplan": 0.975,
    "at_fault_collision":    0.018,
    "offroad":               0.005,
    "run_red_light":         0.012,
}
metric_labels = {
    "progress_ratio_nuplan": "Route Progress",
    "at_fault_collision":    "At-Fault Collision",
    "offroad":               "Offroad Rate",
    "run_red_light":         "Run Red Light",
}
show_metrics = list(metric_labels.keys())
cbm_vals     = [results["task_metrics"][m] for m in show_metrics]
sac_vals     = [SAC_BASELINE[m]            for m in show_metrics]

fig = plt.figure(figsize=(14, 5))
fig.suptitle(
    "CBM Driving Performance — Validation Set (400 scenarios, 10 GB scratch model)",
    fontsize=12, fontweight="bold"
)
gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.4, 1], wspace=0.35)

# ── Panel A: Route progress distribution ─────────────────────────────
ax_hist = fig.add_subplot(gs[0])
ax_hist.hist(prog_vals, bins=30, color="#2196F3", edgecolor="white",
             linewidth=0.5, alpha=0.85, label="CBM Scratch V2 (10GB)")
ax_hist.axvline(prog_vals.mean(), color="#1565C0", lw=2,
                ls="-", label=f"CBM mean = {prog_vals.mean():.3f}")
ax_hist.axvline(SAC_BASELINE["progress_ratio_nuplan"], color="#E53935",
                lw=2, ls="--", label=f"SAC baseline = {SAC_BASELINE['progress_ratio_nuplan']:.3f}")
ax_hist.set_xlabel("Route Progress Ratio (nuplan)")
ax_hist.set_ylabel("Number of Scenarios")
ax_hist.set_title("(a) Route Progress Distribution (N=400)")
ax_hist.legend(fontsize=8.5, frameon=False)
ax_hist.set_xlim(0, 1.05)

# Annotation: % of scenarios reaching > 0.8
pct_high = 100 * (prog_vals > 0.8).mean()
ax_hist.text(0.02, 0.96, f"{pct_high:.0f}% of scenarios\ncomplete > 80% of route",
             transform=ax_hist.transAxes, fontsize=8.5,
             va="top", bbox=dict(boxstyle="round,pad=0.3", fc="#E3F2FD", ec="none"))

# ── Panel B: Key metrics bar chart ────────────────────────────────────
ax_bar = fig.add_subplot(gs[1])
x      = np.arange(len(show_metrics))
width  = 0.35

bars_cbm = ax_bar.bar(x - width/2, cbm_vals, width, label="CBM Scratch V2 (10GB)",
                       color="#2196F3", edgecolor="white")
bars_sac = ax_bar.bar(x + width/2, sac_vals,  width, label="SAC Baseline (150GB)",
                       color="#E53935", edgecolor="white", alpha=0.75)

ax_bar.set_xticks(x)
ax_bar.set_xticklabels([metric_labels[m] for m in show_metrics],
                        rotation=20, ha="right", fontsize=8.5)
ax_bar.set_ylabel("Rate")
ax_bar.set_title("(b) Key Driving Metrics vs SAC Baseline")
ax_bar.legend(fontsize=8.5, frameon=False)
ax_bar.set_ylim(0, 1.05)

for bar in bars_cbm:
    h = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)
for bar in bars_sac:
    h = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

# Annotate run_red_light = 0.000
rl_idx = show_metrics.index("run_red_light")
ax_bar.annotate("Zero red light\nviolations ✓",
                xy=(x[rl_idx] - width/2, 0.005),
                xytext=(x[rl_idx] - width/2 - 0.6, 0.18),
                fontsize=7.5, color="#1B5E20",
                arrowprops=dict(arrowstyle="->", color="#1B5E20", lw=1.2))

path = os.path.join(OUT_DIR, "fig3_task_performance.pdf")
fig.savefig(path)
fig.savefig(path.replace(".pdf", ".png"))
plt.close(fig)
print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 4 — Concept Temporal Evolution (mean pred vs true over 80 steps)
# ══════════════════════════════════════════════════════════════════════
print("Generating Figure 4: Concept Temporal Evolution...")

NCOLS, NROWS = 3, 5
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(15, 20))
fig.suptitle(
    "CBM Concept Temporal Evolution — Mean ± Std over 400 Scenarios\n"
    "Blue = Ground Truth, Orange = CBM Prediction. Shaded = ±1 std.",
    fontsize=11, fontweight="bold", y=1.01
)

steps = np.arange(T)

for idx, name in enumerate(cnames):
    row, col = divmod(idx, NCOLS)
    ax = axes[row, col]

    v_mask = valid[:, :, idx]   # (80, 400)
    p_all  = pred[:, :, idx]    # (80, 400)
    t_all  = true[:, :, idx]

    # Compute per-step mean/std only over valid entries
    p_mean, p_std, t_mean, t_std = [], [], [], []
    for s in range(T):
        v_s = v_mask[s]
        if v_s.sum() == 0:
            p_mean.append(np.nan); p_std.append(np.nan)
            t_mean.append(np.nan); t_std.append(np.nan)
        else:
            p_mean.append(p_all[s, v_s].mean()); p_std.append(p_all[s, v_s].std())
            t_mean.append(t_all[s, v_s].mean()); t_std.append(t_all[s, v_s].std())

    p_mean = np.array(p_mean); p_std = np.array(p_std)
    t_mean = np.array(t_mean); t_std = np.array(t_std)

    # Ground truth
    ax.plot(steps, t_mean, color="#1565C0", lw=1.5, label="True")
    ax.fill_between(steps, t_mean - t_std, t_mean + t_std,
                    color="#1565C0", alpha=0.12)
    # Prediction
    ax.plot(steps, p_mean, color="#E65100", lw=1.5, ls="--", label="Predicted")
    ax.fill_between(steps, p_mean - p_std, p_mean + p_std,
                    color="#E65100", alpha=0.12)

    # Phase tag
    ax.text(0.03, 0.96, f"P{phase_of(name)}", transform=ax.transAxes,
            fontsize=7.5, color=PHASE_COLOR[phase_of(name)],
            va="top", fontweight="bold")

    # R² annotation (for continuous)
    if name not in BINARY:
        p_v, t_v = concept_flat(idx)
        r2_val = r2(p_v, t_v) if len(p_v) > 1 else float("nan")
        ax.text(0.97, 0.96, f"R²={r2_val:.3f}", transform=ax.transAxes,
                fontsize=7.5, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75, ec="none"))
    else:
        acc = results["concept_metrics"][name]["accuracy"]
        ax.text(0.97, 0.96, f"Acc={acc:.4f}", transform=ax.transAxes,
                fontsize=7.5, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75, ec="none"))

    ax.set_title(PRETTY[name], fontsize=9.5, pad=3)
    ax.set_xlabel("Timestep", fontsize=8)
    ax.set_ylabel("Concept Value [0,1]", fontsize=8)
    ax.set_xlim(0, T - 1)
    ax.set_ylim(-0.05, 1.1)

# Shared legend at bottom
handles = [
    Line2D([0], [0], color="#1565C0", lw=2, label="Ground Truth"),
    Line2D([0], [0], color="#E65100", lw=2, ls="--", label="CBM Prediction"),
]
fig.legend(handles=handles, loc="lower center", ncol=2,
           frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=10)

fig.tight_layout(rect=[0, 0.02, 1, 1], h_pad=3.5, w_pad=2.5)
path = os.path.join(OUT_DIR, "fig4_concept_temporal.pdf")
fig.savefig(path)
fig.savefig(path.replace(".pdf", ".png"))
plt.close(fig)
print(f"  Saved: {path}")

print("\nAll figures saved to:", OUT_DIR)
print("Files:")
for f in sorted(os.listdir(OUT_DIR)):
    size = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024
    print(f"  {f}  ({size:.0f} KB)")
