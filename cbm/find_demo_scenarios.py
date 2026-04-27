"""
CBM Demo Scenario Finder

Filters the eval cache (400 val scenarios) to find the best candidates
for each demo archetype. Outputs:

  curated_scenarios.json          — ranked scenario indices + scores per archetype
  curated_scenarios_data.npz      — concept/action arrays for top scenarios only
                                    (ready for the visualization platform)

Archetypes:
  1. red_light_stop      — model sees red light, predicts it, brakes, survives
  2. ttc_success         — lead vehicle closes in, model brakes before collision
  3. curvature_nav       — sharp curve ahead, model steers correctly, completes
  4. concept_failure     — model misreads a concept AND outcome is bad

Usage:
    conda activate vmax
    python find_demo_scenarios.py
"""

import json
import os
import numpy as np

# ── Load cache ────────────────────────────────────────────────────────
BASE = "eval_model_final_cache"

def load(f):
    return np.load(os.path.join(BASE, f), allow_pickle=True)

pred      = load("pred_concepts.npy")          # (80, 400, 15)
true      = load("true_concepts.npy")
valid     = load("valid_mask.npy")             # (80, 400, 15)  bool
actions   = load("ego_actions.npy")            # (80, 400, 2)   [accel, steer]
dones     = load("dones.npy")                  # (80, 400)
rewards   = load("rewards.npy")
drv       = load("driving_metrics.npy")        # (80, 400, 14)
drv_keys  = load("driving_metric_keys.npy").tolist()
cnames    = load("concept_names.npy").tolist()
scen_idx  = load("scenario_indices.npy")       # (400,)

T, N, C = pred.shape

# ── Concept index shortcuts ───────────────────────────────────────────
CI = {name: i for i, name in enumerate(cnames)}

# ── Driving metric index shortcuts ───────────────────────────────────
DI = {name: i for i, name in enumerate(drv_keys)}

# ── Helpers ───────────────────────────────────────────────────────────
def no_at_fault_collision(n):
    return drv[:, n, DI["at_fault_collision"]].max() < 0.5

def episode_completes(n):
    return dones[:-1, n].max() < 0.5

def progress(n):
    return drv[-1, n, DI["progress_ratio_nuplan"]]

def concept_active(name, n, threshold=0.7):
    """True if ground-truth concept exceeds threshold at any step where valid."""
    v = valid[:, n, CI[name]]
    t = true[:, n, CI[name]]
    return (t[v] > threshold).any() if v.any() else False

def pred_tracks(name, n, threshold=0.7):
    """True if model prediction also exceeds threshold when concept is active."""
    v = valid[:, n, CI[name]]
    p = pred[:, n, CI[name]]
    return (p[v] > threshold).any() if v.any() else False

def max_braking(n):
    """Most negative acceleration over episode."""
    return actions[:, n, 0].min()

def concept_error_max(name, n):
    """Max |pred - true| over valid steps for a concept."""
    v = valid[:, n, CI[name]]
    if not v.any():
        return 0.0
    return float(np.abs(pred[:, n, CI[name]][v] - true[:, n, CI[name]][v]).max())

# ── Score functions (higher = better candidate) ───────────────────────

def score_red_light_stop(n):
    """
    Model sees red light, predicts it, brakes hard, survives.
    """
    tl_name = "traffic_light_red"
    v = valid[:, n, CI[tl_name]]
    t = true[:, n, CI[tl_name]]
    p = pred[:, n, CI[tl_name]]
    accel = actions[:, n, 0]

    if not v.any():
        return -1.0

    # Steps where TL is actually red
    red_steps = v & (t > 0.8)
    if not red_steps.any():
        return -1.0

    tl_pred_confidence = p[red_steps].mean()   # how well model predicts red
    braking_during_red = (-accel[red_steps]).clip(0).max()  # max braking when red
    survived = 1.0 if no_at_fault_collision(n) else 0.0
    prog = progress(n)

    return float(tl_pred_confidence * 0.35 +
                 braking_during_red * 0.35 +
                 survived * 0.20 +
                 prog * 0.10)

def score_ttc_success(n):
    """
    TTC drops low (danger), model brakes, no collision.
    """
    ttc_name = "ttc_lead_vehicle"
    v = valid[:, n, CI[ttc_name]]
    t = true[:, n, CI[ttc_name]]
    accel = actions[:, n, 0]

    if not v.any():
        return -1.0

    # Low TTC = danger (small normalized value)
    danger_steps = v & (t < 0.35)
    if not danger_steps.any():
        return -1.0

    danger_severity = (1.0 - t[danger_steps]).mean()   # how close was danger
    braking = (-accel[danger_steps]).clip(0).max()      # did it brake
    survived = 1.0 if no_at_fault_collision(n) else 0.0
    pred_ttc = pred[:, n, CI[ttc_name]][danger_steps].mean()
    pred_awareness = 1.0 - pred_ttc                     # model also saw danger

    return float(danger_severity * 0.30 +
                 braking * 0.30 +
                 survived * 0.25 +
                 pred_awareness * 0.15)

def score_curvature_nav(n):
    """
    High path curvature ahead, model navigates successfully.
    """
    curv_name = "path_curvature_max"
    t = true[:, n, CI[curv_name]]

    max_curv = t.max()
    if max_curv < 0.5:
        return -1.0

    completed = 1.0 if episode_completes(n) else 0.0
    prog = progress(n)
    pred_curv_awareness = pred[:, n, CI[curv_name]].max()  # model predicts curvature

    return float(max_curv * 0.35 +
                 completed * 0.35 +
                 prog * 0.15 +
                 pred_curv_awareness * 0.15)

def score_concept_failure(n):
    """
    Model badly misreads a key concept AND outcome is bad.
    Higher = clearer failure mode for XAI analysis.
    """
    # Check key concepts for large prediction error
    key_concepts = ["traffic_light_red", "dist_nearest_object", "ttc_lead_vehicle"]
    max_err = max(concept_error_max(c, n) for c in key_concepts)

    # Bad outcome: collision or poor progress
    collision = 1.0 if not no_at_fault_collision(n) else 0.0
    poor_prog = max(0.0, 0.6 - progress(n))

    if max_err < 0.3 and collision == 0.0:
        return -1.0

    return float(max_err * 0.50 +
                 collision * 0.35 +
                 poor_prog * 0.15)

# ── Score all scenarios ───────────────────────────────────────────────
print("Scoring all 400 scenarios per archetype...")

archetypes = {
    "red_light_stop":  score_red_light_stop,
    "ttc_success":     score_ttc_success,
    "curvature_nav":   score_curvature_nav,
    "concept_failure": score_concept_failure,
}

TOP_K = 10
results = {}

for arch_name, score_fn in archetypes.items():
    scores = np.array([score_fn(n) for n in range(N)])
    valid_mask = scores > 0
    ranked = np.argsort(scores)[::-1]
    top = [i for i in ranked if scores[i] > 0][:TOP_K]

    print(f"\n  [{arch_name}] — {valid_mask.sum()} eligible scenarios")
    print(f"  {'Rank':<5} {'Scenario':>10} {'Score':>8} {'Progress':>10} {'No Collision':>14}")
    print(f"  {'-'*52}")

    entries = []
    for rank, n in enumerate(top):
        prog = float(progress(n))
        safe = no_at_fault_collision(n)
        print(f"  {rank+1:<5} {int(scen_idx[n]):>10} {scores[n]:>8.4f} {prog:>10.4f} {'✓' if safe else '✗':>14}")
        entries.append({
            "rank":            rank + 1,
            "local_idx":       int(n),
            "scenario_idx":    int(scen_idx[n]),
            "score":           float(scores[n]),
            "progress":        prog,
            "no_at_fault":     bool(safe),
            "episode_done":    bool(not episode_completes(n)),
        })

    results[arch_name] = entries

# ── Save JSON ─────────────────────────────────────────────────────────
out_json = "curated_scenarios.json"
with open(out_json, "w") as f:
    json.dump({
        "model":      "cbm_scratch_v2_lambda05",
        "n_total":    N,
        "top_k":      TOP_K,
        "archetypes": results,
    }, f, indent=2)
print(f"\nSaved: {out_json}")

# ── Save data arrays for top scenarios ───────────────────────────────
# Collect unique top scenario local indices across all archetypes
top_locals = sorted(set(
    e["local_idx"]
    for arch in results.values()
    for e in arch
))
print(f"\nCollecting data for {len(top_locals)} unique top scenarios...")

sel = np.array(top_locals)

np.savez_compressed(
    "curated_scenarios_data.npz",
    # Arrays sliced to top scenarios only
    pred_concepts   = pred[:, sel, :].astype(np.float32),      # (80, M, 15)
    true_concepts   = true[:, sel, :].astype(np.float32),
    valid_mask      = valid[:, sel, :],
    ego_actions     = actions[:, sel, :].astype(np.float32),   # (80, M, 2)
    dones           = dones[:, sel].astype(np.float32),
    rewards         = rewards[:, sel].astype(np.float32),
    driving_metrics = drv[:, sel, :].astype(np.float32),
    # Metadata
    local_indices   = sel,
    scenario_indices= scen_idx[sel],
    concept_names   = np.array(cnames),
    driving_metric_keys = np.array(drv_keys),
)

size_mb = os.path.getsize("curated_scenarios_data.npz") / 1e6
print(f"Saved: curated_scenarios_data.npz  ({size_mb:.1f} MB)")
print(f"\nReady for platform. Load with:")
print("  data = np.load('curated_scenarios_data.npz', allow_pickle=True)")
print("  json.load(open('curated_scenarios.json'))")
