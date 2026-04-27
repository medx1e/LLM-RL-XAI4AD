"""Temporal XAI experiment: all 7 methods on scenario 002.

Runs every XAI method across the full episode of scenario 002, then
generates:
  - Per-method temporal category-importance line plots
  - Multi-method comparison grid (one subplot per category)
  - Stacked area charts for each method
  - Per-agent importance for each method
  - Sparsity (Gini) over time for all methods
  - Single-timestep method-comparison bar chart at event peak
  - Deletion/insertion curves at peak timestep
  - Summary CSV with all metrics

Usage (from repo root, conda vmax env):
    python experiments/scenario002_all_methods.py
"""

import os
import sys
import json
import time

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")

VMAX_REPO = os.path.join(_ROOT, "V-Max")
if VMAX_REPO not in sys.path:
    sys.path.insert(0, VMAX_REPO)

# Make posthoc_xai and event_mining importable
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import jax.numpy as jnp

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_DIR    = "runs_rlc/womd_sac_road_perceiver_minimal_42"
DATA_PATH    = "data/training.tfrecord"
SCENARIO_IDX = 2          # 0-based → s002
TIMESTEP_STRIDE = 5       # analyse every 5th step
OUTPUT_DIR   = "experiments/scenario002_all_methods"
# Reduced n_steps/n_samples for speed while still being accurate after the fix
IG_STEPS     = 25         # IG uses trapezoidal rule → accurate with 25 intervals
SG_SAMPLES   = 25         # SmoothGrad samples

os.makedirs(OUTPUT_DIR, exist_ok=True)

CATEGORY_COLORS = {
    "sdc_trajectory": "#4488FF",
    "other_agents":   "#FF4444",
    "roadgraph":      "#888888",
    "traffic_lights": "#FFCC00",
    "gps_path":       "#00CC66",
}

METHOD_COLORS = {
    "vanilla_gradient":    "#1f77b4",
    "gradient_x_input":    "#ff7f0e",
    "integrated_gradients":"#2ca02c",
    "smooth_grad":         "#d62728",
    "perturbation":        "#9467bd",
    "feature_ablation":    "#8c564b",
    "sarfa":               "#e377c2",
}


# ── Step 1: Load model ───────────────────────────────────────────────────────
print("=" * 65)
print("STEP 1: Loading model")
print("=" * 65)

import posthoc_xai as xai

model = xai.load_model(MODEL_DIR, data_path=DATA_PATH)
print(f"Model      : {model.name}")
print(f"Obs structure: {model.observation_structure}")

# ── Step 2: Extract scenario 002 ─────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"STEP 2: Extracting scenario {SCENARIO_IDX:03d}")
print("=" * 65)

from event_mining import EventMiner
from event_mining.integration.vmax_adapter import VMaxAdapter
from vmax.simulator import make_data_generator

loaded = model._loaded
data_gen = make_data_generator(
    path=DATA_PATH,
    max_num_objects=loaded.config.get("max_num_objects", 64),
    include_sdc_paths=True,
    batch_dims=(1,),
    seed=42,
    repeat=1,
)

adapter = VMaxAdapter(store_raw_obs=True)
adapter.prepare(model)

data_iter = iter(data_gen)
scenario = None
for i in range(SCENARIO_IDX + 1):
    scenario = next(data_iter)

sid = f"s{SCENARIO_IDX:03d}"
sd = adapter.extract_scenario_data(model, scenario, sid, rng_seed=SCENARIO_IDX)

print(f"Scenario   : {sid}")
print(f"Total steps: {sd.total_steps}")
print(f"Collision  : {sd.has_collision}")
print(f"Off-road   : {sd.has_offroad}")
print(f"Route compl: {sd.route_completion:.1%}")

# ── Step 3: Mine events ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3: Mining events in scenario 002")
print("=" * 65)

miner = EventMiner()
events = miner.mine_scenario(sd)
print(f"Events found: {len(events)}")
for e in events:
    print(
        f"  {e.event_type.value:22s} sev={e.severity.value:8s} "
        f"t={e.onset:3d}–{e.offset:3d} peak={e.peak:3d} agent={e.causal_agent_id}"
    )

# Pick the most severe event for focused plots
events_sorted = sorted(events, key=lambda e: (e.severity_score, e.duration), reverse=True)
top_event = events_sorted[0] if events_sorted else None
if top_event:
    print(f"\nTop event for focused plots: {top_event.event_type.value} at t={top_event.peak}")

# ── Step 4: Initialise all XAI methods ───────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4: Initialising all 7 XAI methods")
print("=" * 65)

methods = {
    "vanilla_gradient":     xai.VanillaGradient(model),
    "gradient_x_input":     xai.GradientXInput(model),
    "integrated_gradients": xai.IntegratedGradients(model, n_steps=IG_STEPS, baseline="zero"),
    "smooth_grad":          xai.SmoothGrad(model, n_samples=SG_SAMPLES, noise_std=0.1),
    "perturbation":         xai.PerturbationAttribution(model, per_category=True),
    "feature_ablation":     xai.FeatureAblation(model, replacement="zero"),
    "sarfa":                xai.SARFA(model, per_category=True),
}
print(f"Methods: {list(methods.keys())}")

# ── Step 5: Temporal sweep ────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5: Running temporal XAI sweep")
print("=" * 65)

assert sd.raw_observations is not None, "raw_observations not stored — check VMaxAdapter(store_raw_obs=True)"

timesteps = list(range(0, sd.total_steps, TIMESTEP_STRIDE))
# Always include event key points
if top_event:
    for kt in [top_event.onset, top_event.peak, top_event.offset]:
        if 0 <= kt < sd.total_steps and kt not in timesteps:
            timesteps.append(kt)
    timesteps.sort()

print(f"Timesteps to analyse: {len(timesteps)} — {timesteps}")

# Results storage
# method_name -> {category: [val_t0, val_t1, ...]}
cat_series   = {m: {} for m in methods}
# method_name -> {entity: [val_t0, val_t1, ...]}
entity_series = {m: {} for m in methods}
# method_name -> {metric: [val_t0, val_t1, ...]}
sparsity_series = {m: {"gini": [], "entropy": []} for m in methods}
# Timings
timings = {m: [] for m in methods}

# JIT warm-up on first observation so times are comparable afterwards
print("\n  JIT warm-up on t=0 ...")
obs0 = jnp.array(sd.raw_observations[0])
for m_name, method in methods.items():
    t0 = time.time()
    _ = method(obs0)
    elapsed = (time.time() - t0) * 1000
    print(f"    {m_name:25s}  warm-up: {elapsed:.0f} ms")

print("\n  Temporal sweep:")
from posthoc_xai.metrics import sparsity as sparsity_metrics

for t in timesteps:
    obs = jnp.array(sd.raw_observations[t])
    print(f"  t={t:3d}", end="", flush=True)

    for m_name, method in methods.items():
        t0 = time.time()
        attr = method(obs)
        elapsed = (time.time() - t0) * 1000
        timings[m_name].append(elapsed)

        # Category importance
        for cat, imp in attr.category_importance.items():
            cat_series[m_name].setdefault(cat, []).append(float(imp))

        # Agent entity importance
        if "other_agents" in attr.entity_importance:
            for ent, imp in attr.entity_importance["other_agents"].items():
                entity_series[m_name].setdefault(ent, []).append(float(imp))

        # Sparsity
        sp = sparsity_metrics.compute_all(attr)
        sparsity_series[m_name]["gini"].append(sp["gini"])
        sparsity_series[m_name]["entropy"].append(sp["entropy"])

        print(".", end="", flush=True)

    print()  # newline after each timestep

print("\nAverage inference times (after JIT warm-up):")
for m_name in methods:
    mean_t = np.mean(timings[m_name][1:]) if len(timings[m_name]) > 1 else timings[m_name][0]
    print(f"  {m_name:25s}: {mean_t:.1f} ms/step")

# ── Step 6: Single-timestep analysis at peak ──────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6: Single-timestep analysis at peak / midpoint")
print("=" * 65)

peak_t = top_event.peak if top_event else sd.total_steps // 2
print(f"Using timestep t={peak_t}")

obs_peak = jnp.array(sd.raw_observations[peak_t])
peak_attributions = {}
for m_name, method in methods.items():
    attr = method(obs_peak)
    peak_attributions[m_name] = attr
    print(
        f"  {m_name:25s}: "
        + ", ".join(f"{k}={v:.3f}" for k, v in attr.category_importance.items())
    )

# ── Step 7: Deletion/insertion curves at peak ─────────────────────────────────
print("\n" + "=" * 65)
print("STEP 7: Deletion / insertion curves at t=" + str(peak_t))
print("=" * 65)

from posthoc_xai.metrics import faithfulness

del_data = []  # [(name, pcts, outputs)]
ins_data = []

for m_name, attr in peak_attributions.items():
    pcts_d, out_d = faithfulness.deletion_curve(model, obs_peak, attr, n_steps=15)
    pcts_i, out_i = faithfulness.insertion_curve(model, obs_peak, attr, n_steps=15)
    del_auc = faithfulness.area_under_deletion_curve(out_d)
    ins_auc = faithfulness.area_under_insertion_curve(out_i)
    del_data.append((m_name, pcts_d, out_d))
    ins_data.append((m_name, pcts_i, out_i))
    print(f"  {m_name:25s}: del_AUC={del_auc:.4f}  ins_AUC={ins_auc:.4f}")

# ── Step 8: Sparsity summary ──────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 8: Sparsity summary at t=" + str(peak_t))
print("=" * 65)

from posthoc_xai.metrics import sparsity as sparsity_metrics

for m_name, attr in peak_attributions.items():
    sp = sparsity_metrics.compute_all(attr)
    print(
        f"  {m_name:25s}: gini={sp['gini']:.3f}  entropy={sp['entropy']:.3f}  "
        f"top10={sp['top_10_concentration']:.3f}"
    )

# ── Step 9: Save summary CSV ──────────────────────────────────────────────────
rows = []
for m_name, attr in peak_attributions.items():
    sp = sparsity_metrics.compute_all(attr)
    pcts_d, out_d = del_data[[n for n,_,_ in del_data].index(m_name)][1], \
                    del_data[[n for n,_,_ in del_data].index(m_name)][2]
    del_auc = faithfulness.area_under_deletion_curve(out_d)
    pcts_i, out_i = ins_data[[n for n,_,_ in ins_data].index(m_name)][1], \
                    ins_data[[n for n,_,_ in ins_data].index(m_name)][2]
    ins_auc = faithfulness.area_under_insertion_curve(out_i)
    row = {
        "method": m_name,
        "peak_timestep": peak_t,
        "gini": sp["gini"],
        "entropy": sp["entropy"],
        "top10": sp["top_10_concentration"],
        "deletion_auc": del_auc,
        "insertion_auc": ins_auc,
        "avg_time_ms": np.mean(timings[m_name][1:]) if len(timings[m_name]) > 1 else timings[m_name][0],
    }
    row.update({f"cat_{k}": v for k, v in attr.category_importance.items()})
    rows.append(row)

df_summary = pd.DataFrame(rows)
df_summary.to_csv(os.path.join(OUTPUT_DIR, "summary_metrics.csv"), index=False)
print(f"\nSummary CSV saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 10: Generating plots")
print("=" * 65)

def _add_event_markers(ax, ev):
    if ev is None:
        return
    ax.axvspan(ev.onset, ev.offset, alpha=0.08, color="red", zorder=0)
    ax.axvline(ev.onset, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(ev.peak, color="red", linestyle="-", alpha=0.9, linewidth=1.5)
    if ev.offset != ev.onset:
        ax.axvline(ev.offset, color="orange", linestyle="--", alpha=0.5, linewidth=1)


# ── Plot A: Per-method temporal category importance (7 subplots, stacked) ─────
fig, axes = plt.subplots(len(methods), 1, figsize=(14, 3.5 * len(methods)), sharex=True)
for ax, (m_name, cs) in zip(axes, cat_series.items()):
    for cat, vals in cs.items():
        color = CATEGORY_COLORS.get(cat, "#AAAAAA")
        ax.plot(timesteps, vals, "-o", label=cat, color=color, markersize=3, linewidth=1.5)
    _add_event_markers(ax, top_event)
    ax.set_ylabel("Importance", fontsize=8)
    ax.set_title(m_name, fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", ncol=3)
    ax.grid(True, alpha=0.25)

axes[-1].set_xlabel("Timestep")
fig.suptitle(f"Category Importance Over Time — {sid} (all 7 methods)", fontsize=13, fontweight="bold")
fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "A_temporal_category_all_methods.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"  Saved: {path}")


# ── Plot B: Multi-method grid (one subplot per category) ─────────────────────
categories = list(next(iter(cat_series.values())).keys())
n_cats = len(categories)
cols = min(3, n_cats)
rows = (n_cats + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

for idx, cat in enumerate(categories):
    r, c = divmod(idx, cols)
    ax = axes[r][c]
    for m_name, cs in cat_series.items():
        vals = cs.get(cat, [0] * len(timesteps))
        color = METHOD_COLORS.get(m_name, "#888888")
        ax.plot(timesteps, vals, "-o", label=m_name, color=color, markersize=2, linewidth=1.2)
    _add_event_markers(ax, top_event)
    ax.set_title(cat.replace("_", " ").title(), fontsize=10)
    ax.set_xlabel("Timestep", fontsize=8)
    ax.set_ylabel("Importance", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=7)

# hide unused axes
for idx in range(n_cats, rows * cols):
    r, c = divmod(idx, cols)
    axes[r][c].set_visible(False)

handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8,
           bbox_to_anchor=(0.5, 1.01))
fig.suptitle(f"Method Comparison Per Category — {sid}", fontsize=13,
             fontweight="bold", y=1.04)
fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "B_temporal_per_category_grid.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path}")


# ── Plot C: Stacked area charts (one per method, 2 rows × 4 cols) ────────────
n_methods = len(methods)
cols_c = 4
rows_c = (n_methods + cols_c - 1) // cols_c
fig, axes = plt.subplots(rows_c, cols_c, figsize=(5 * cols_c, 4 * rows_c), squeeze=False)

for idx, (m_name, cs) in enumerate(cat_series.items()):
    r, c = divmod(idx, cols_c)
    ax = axes[r][c]
    cats_list = list(cs.keys())
    vals_arr = np.array([cs[cat] for cat in cats_list])
    colors = [CATEGORY_COLORS.get(cat, "#AAAAAA") for cat in cats_list]
    ax.stackplot(timesteps, vals_arr, labels=cats_list, colors=colors, alpha=0.85)
    _add_event_markers(ax, top_event)
    ax.set_ylim(0, 1)
    ax.set_title(m_name, fontsize=8, fontweight="bold")
    ax.set_xlabel("Timestep", fontsize=7)
    ax.tick_params(labelsize=7)

for idx in range(n_methods, rows_c * cols_c):
    r, c = divmod(idx, cols_c)
    axes[r][c].set_visible(False)

axes[0][0].legend(fontsize=6, loc="upper left")
fig.suptitle(f"Attribution Composition (Stacked) — {sid}", fontsize=12, fontweight="bold")
fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "C_stacked_all_methods.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"  Saved: {path}")


# ── Plot D: Agent importance over time (one subplot per method) ───────────────
fig, axes = plt.subplots(n_methods, 1, figsize=(14, 3.2 * n_methods), sharex=True)
if n_methods == 1:
    axes = [axes]

causal_id = top_event.causal_agent_id if top_event else None

for ax, (m_name, es) in zip(axes, entity_series.items()):
    if not es:
        ax.set_title(f"{m_name} — no entity data", fontsize=8)
        continue
    agent_peaks = {a: max(vals) for a, vals in es.items()}
    top_agents = sorted(agent_peaks, key=agent_peaks.get, reverse=True)[:6]
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_agents)))
    for agent, color in zip(top_agents, colors):
        is_causal = (agent == f"agent_{causal_id}")
        lw = 2.5 if is_causal else 1.0
        lbl = f"{agent} ★" if is_causal else agent
        ax.plot(timesteps, es[agent], "-o", label=lbl, color=color,
                linewidth=lw, markersize=2)
    _add_event_markers(ax, top_event)
    ax.set_ylabel("Importance", fontsize=8)
    ax.set_title(m_name, fontsize=9, fontweight="bold")
    ax.legend(fontsize=6, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.25)

axes[-1].set_xlabel("Timestep")
causal_str = f"causal=agent_{causal_id}" if causal_id is not None else "no event"
fig.suptitle(f"Agent Importance Over Time — {sid} ({causal_str})", fontsize=12,
             fontweight="bold")
fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "D_agent_importance_all_methods.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"  Saved: {path}")


# ── Plot E: Sparsity (Gini) over time ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, metric_key, ylabel in zip(axes,
                                   ["gini", "entropy"],
                                   ["Gini Coefficient", "Normalised Entropy"]):
    for m_name, sp in sparsity_series.items():
        color = METHOD_COLORS.get(m_name, "#888888")
        ax.plot(timesteps, sp[metric_key], "-o", label=m_name, color=color,
                markersize=3, linewidth=1.5)
    _add_event_markers(ax, top_event)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + " over Time")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

fig.suptitle(f"Sparsity Metrics — {sid}", fontsize=12, fontweight="bold")
fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "E_sparsity_over_time.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"  Saved: {path}")


# ── Plot F: Single-timestep bar comparison at peak ────────────────────────────
cats_list = list(next(iter(peak_attributions.values())).category_importance.keys())
n_c = len(cats_list)
n_m = len(peak_attributions)
x = np.arange(n_c)
width = 0.8 / n_m

fig, ax = plt.subplots(figsize=(12, 6))
for i, (m_name, attr) in enumerate(peak_attributions.items()):
    vals = [attr.category_importance[c] for c in cats_list]
    offset = (i - n_m / 2 + 0.5) * width
    color = METHOD_COLORS.get(m_name, "#888888")
    ax.bar(x + offset, vals, width, label=m_name, color=color, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels([c.replace("_", "\n") for c in cats_list])
ax.set_ylabel("Importance")
ax.set_title(f"All-Method Category Importance at t={peak_t} — {sid}", fontsize=12)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "F_peak_timestep_comparison.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"  Saved: {path}")


# ── Plot G: Deletion / insertion curves ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax_del, ax_ins = axes
for m_name, pcts, outs in del_data:
    color = METHOD_COLORS.get(m_name, "#888888")
    ax_del.plot(pcts, outs, label=m_name, color=color, linewidth=1.8)
for m_name, pcts, outs in ins_data:
    color = METHOD_COLORS.get(m_name, "#888888")
    ax_ins.plot(pcts, outs, label=m_name, color=color, linewidth=1.8)

ax_del.set_xlabel("Fraction of Features Removed")
ax_del.set_ylabel("Model Output (action sum)")
ax_del.set_title(f"Deletion Curve at t={peak_t}  (lower AUC = better)")
ax_del.legend(fontsize=7)
ax_del.grid(True, alpha=0.3)

ax_ins.set_xlabel("Fraction of Features Added")
ax_ins.set_ylabel("Model Output (action sum)")
ax_ins.set_title(f"Insertion Curve at t={peak_t}  (higher AUC = better)")
ax_ins.legend(fontsize=7)
ax_ins.grid(True, alpha=0.3)

fig.suptitle(f"Deletion / Insertion Faithfulness — {sid}", fontsize=12, fontweight="bold")
fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "G_deletion_insertion_curves.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"  Saved: {path}")


# ── Plot H: Heatmap — category × method at peak ──────────────────────────────
method_names = list(peak_attributions.keys())
heat_data = np.array([
    [peak_attributions[m].category_importance[c] for c in cats_list]
    for m in method_names
])  # (n_methods, n_cats)

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(heat_data, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(n_c))
ax.set_xticklabels([c.replace("_", "\n") for c in cats_list], fontsize=9)
ax.set_yticks(range(n_m))
ax.set_yticklabels(method_names, fontsize=9)
plt.colorbar(im, ax=ax, label="Normalised Importance")
# Annotate cells
for i in range(n_m):
    for j in range(n_c):
        ax.text(j, i, f"{heat_data[i, j]:.3f}", ha="center", va="center",
                fontsize=7, color="black" if heat_data[i, j] < heat_data.max() * 0.6 else "white")
ax.set_title(f"Category × Method Importance Heatmap at t={peak_t} — {sid}", fontsize=11)
fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "H_importance_heatmap.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"  Saved: {path}")


# ─── Final summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("EXPERIMENT COMPLETE")
print("=" * 65)
print(f"\nScenario : {sid}")
print(f"Steps    : {sd.total_steps}")
print(f"Events   : {len(events)}")
print(f"Methods  : {list(methods.keys())}")
print(f"Timesteps: {len(timesteps)} analysed (stride={TIMESTEP_STRIDE})")
print(f"\nOutput files in {OUTPUT_DIR}/:")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, fname)
    if os.path.isfile(fpath):
        size = os.path.getsize(fpath)
        print(f"  {fname:50s}  {size/1024:.1f} KB")

print(f"\nSummary metrics table:")
print(df_summary[["method","gini","entropy","deletion_auc","insertion_auc","avg_time_ms"]].to_string(index=False))
