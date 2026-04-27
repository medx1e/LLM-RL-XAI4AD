#!/usr/bin/env python3
"""Empirical analysis of candidate path-based spatial concepts for CBM-V2.

Computes all candidate path concepts on real TFRecord data and reports:
  - Per-concept statistics: valid rate, min/max/mean/std
  - Degeneracy checks (constant/near-constant values)
  - Variance analysis (which concepts vary meaningfully across scenes)
  - Redundancy analysis (pairwise correlation between candidates)
  - Logged-data (t=0) vs closed-loop (rollout) considerations

Usage:
    /home/med1e/anaconda3/envs/vmax/bin/python scripts/analyze_path_spatial_concepts.py \
        --data data/training.tfrecord \
        --model_dir runs_rlc/womd_sac_road_perceiver_minimal_42 \
        --num_scenarios 100
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "V-Max"))

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from concepts.types import ObservationConfig
from concepts.adapters import observation_to_concept_input
from concepts.normalize import denorm_xy, denorm_vel, OBJ_VEL, OBJ_YAW
from concepts.geometry import l2_norm, wrap_angle, project_onto_path

ENCODER_REMAP = {"perceiver": "lq", "mgail": "lqh"}
OBS_TYPE_REMAP = {"road": "vec", "lane": "vec"}


# =====================================================================
# Candidate path-based spatial concept extractors
# =====================================================================
# These mirror the signature of concepts.extractors but are exploratory.
# Each returns (raw_value, valid_mask) with shape (...).

def path_curvature_mean(path_xy_m):
    """Mean unsigned curvature of the path polyline (1/m).

    Curvature at each interior point estimated via the Menger curvature:
      kappa = 2 * |cross(AB, AC)| / (|AB| * |BC| * |AC|)
    """
    A = path_xy_m[..., :-2, :]   # (…, P-2, 2)
    B = path_xy_m[..., 1:-1, :]
    C = path_xy_m[..., 2:, :]

    AB = B - A
    BC = C - B
    AC = C - A

    cross = jnp.abs(AB[..., 0] * AC[..., 1] - AB[..., 1] * AC[..., 0])
    len_AB = l2_norm(AB, axis=-1)
    len_BC = l2_norm(BC, axis=-1)
    len_AC = l2_norm(AC, axis=-1)

    denom = len_AB * len_BC * len_AC + 1e-8
    kappa = 2.0 * cross / denom  # (…, P-2)

    return jnp.mean(kappa, axis=-1)  # (…,)


def path_curvature_max(path_xy_m):
    """Max unsigned curvature along the path (1/m)."""
    A = path_xy_m[..., :-2, :]
    B = path_xy_m[..., 1:-1, :]
    C = path_xy_m[..., 2:, :]

    AB = B - A
    BC = C - B
    AC = C - A

    cross = jnp.abs(AB[..., 0] * AC[..., 1] - AB[..., 1] * AC[..., 0])
    len_AB = l2_norm(AB, axis=-1)
    len_BC = l2_norm(BC, axis=-1)
    len_AC = l2_norm(AC, axis=-1)

    denom = len_AB * len_BC * len_AC + 1e-8
    kappa = 2.0 * cross / denom
    return jnp.max(kappa, axis=-1)


def path_total_heading_change(path_xy_m):
    """Total absolute heading change along path (radians).

    Sum of |delta_heading| between consecutive segments.
    Distinguishes straight roads (~0) from winding roads (large).
    """
    seg = path_xy_m[..., 1:, :] - path_xy_m[..., :-1, :]  # (…, P-1, 2)
    headings = jnp.arctan2(seg[..., 1], seg[..., 0])       # (…, P-1)
    delta = headings[..., 1:] - headings[..., :-1]         # (…, P-2)
    delta_wrapped = wrap_angle(delta)
    return jnp.sum(jnp.abs(delta_wrapped), axis=-1)


def path_net_heading_change(path_xy_m):
    """Net (signed) heading change from first to last path segment (radians).

    Positive = left turn, negative = right turn. Near 0 = straight or S-curve.
    """
    seg = path_xy_m[..., 1:, :] - path_xy_m[..., :-1, :]
    headings = jnp.arctan2(seg[..., 1], seg[..., 0])
    return wrap_angle(headings[..., -1] - headings[..., 0])


def path_straightness(path_xy_m):
    """Straightness ratio: chord_length / arc_length.

    1.0 = perfectly straight, <1.0 = curved. Robust to normalization.
    """
    chord = l2_norm(path_xy_m[..., -1, :] - path_xy_m[..., 0, :], axis=-1)
    seg_lens = l2_norm(
        path_xy_m[..., 1:, :] - path_xy_m[..., :-1, :], axis=-1
    )  # (…, P-1)
    arc = jnp.sum(seg_lens, axis=-1)
    return chord / (arc + 1e-8)


def dist_to_path(path_xy_m):
    """Lateral distance (m) from SDC (origin) to the nearest path point.

    At t=0 in SDC frame the ego is at (0,0). This measures how far
    off the path centerline the SDC is.
    """
    origin = jnp.zeros(path_xy_m.shape[:-2] + (2,))
    lateral_dist, _ = project_onto_path(origin, path_xy_m)
    return lateral_dist


def dist_to_path_end(path_xy_m):
    """Distance (m) from SDC to the last path waypoint.

    Proxy for "how far ahead the route extends". At t=0 this is roughly
    the lookahead distance.
    """
    return l2_norm(path_xy_m[..., -1, :], axis=-1)


def path_arc_length(path_xy_m):
    """Total arc length of the path polyline (m)."""
    seg_lens = l2_norm(
        path_xy_m[..., 1:, :] - path_xy_m[..., :-1, :], axis=-1
    )
    return jnp.sum(seg_lens, axis=-1)


def heading_to_path_end(path_xy_m):
    """Heading angle (rad) from ego to the last path point.

    Near 0 = path end is straight ahead, large = path curves away.
    """
    end_xy = path_xy_m[..., -1, :]
    return jnp.arctan2(end_xy[..., 1], end_xy[..., 0])


def is_turning(path_xy_m, threshold_rad=0.3):
    """Binary: is the path executing a significant turn?

    Based on total heading change exceeding threshold (~17 degrees).
    """
    total = path_total_heading_change(path_xy_m)
    return (total > threshold_rad).astype(jnp.float32)


def turn_direction(path_xy_m):
    """Categorical-ish: net heading change sign.

    >0 = left turn, <0 = right turn, ~0 = straight.
    Returns the signed net heading change directly.
    """
    return path_net_heading_change(path_xy_m)


# All candidates with names
CANDIDATE_CONCEPTS = {
    "path_curvature_mean":        path_curvature_mean,
    "path_curvature_max":         path_curvature_max,
    "path_total_heading_change":  path_total_heading_change,
    "path_net_heading_change":    path_net_heading_change,
    "path_straightness":          path_straightness,
    "dist_to_path":               dist_to_path,
    "dist_to_path_end":           dist_to_path_end,
    "path_arc_length":            path_arc_length,
    "heading_to_path_end":        heading_to_path_end,
    "is_turning":                 is_turning,
    "turn_direction":             turn_direction,
}


# =====================================================================
# Data loading (reused from audit.py pattern)
# =====================================================================

def load_env_and_data(model_dir, data_path, max_num_objects=64):
    from waymax import dynamics
    from vmax.simulator import make_env_for_evaluation, make_data_generator

    with open(f"{model_dir}/.hydra/config.yaml") as f:
        config = yaml.safe_load(f)

    encoder_type = config["network"]["encoder"]["type"]
    if encoder_type in ENCODER_REMAP:
        config["network"]["encoder"]["type"] = ENCODER_REMAP[encoder_type]
    obs_type = OBS_TYPE_REMAP.get(config["observation_type"], config["observation_type"])

    env = make_env_for_evaluation(
        max_num_objects=max_num_objects,
        dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
        sdc_paths_from_data=True,
        observation_type=obs_type,
        observation_config=config["observation_config"],
        termination_keys=config["termination_keys"],
        noisy_init=False,
    )

    data_gen = make_data_generator(
        path=data_path,
        max_num_objects=max_num_objects,
        include_sdc_paths=True,
        batch_dims=(1,),
        seed=42,
        repeat=1,
    )

    unflatten_fn = env.get_wrapper_attr("features_extractor").unflatten_features

    obs_cfg = config["observation_config"]
    concept_config = ObservationConfig(
        obs_past_num_steps=obs_cfg.get("obs_past_num_steps", 5),
        num_closest_objects=obs_cfg.get("objects", {}).get("num_closest_objects", 8),
        roadgraph_top_k=obs_cfg.get("roadgraphs", {}).get("roadgraph_top_k", 200),
        num_closest_traffic_lights=obs_cfg.get("traffic_lights", {}).get("num_closest_traffic_lights", 5),
        num_target_path_points=obs_cfg.get("path_target", {}).get("num_points", 10),
        max_meters=obs_cfg.get("roadgraphs", {}).get("max_meters", 70),
    )

    return env, data_gen, unflatten_fn, concept_config


# =====================================================================
# Main analysis
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Path spatial concepts empirical analysis")
    parser.add_argument("--data", type=str, default="data/training.tfrecord")
    parser.add_argument("--model_dir", type=str, default="runs_rlc/womd_sac_road_perceiver_minimal_42")
    parser.add_argument("--num_scenarios", type=int, default=100)
    parser.add_argument("--max_num_objects", type=int, default=64)
    args = parser.parse_args()

    print("=" * 70)
    print("PATH-BASED SPATIAL CONCEPTS — EMPIRICAL ANALYSIS")
    print("=" * 70)

    print(f"\nLoading environment from {args.model_dir}...")
    env, data_gen, unflatten_fn, concept_config = load_env_and_data(
        args.model_dir, args.data, args.max_num_objects
    )
    print(f"  max_meters = {concept_config.max_meters}")
    print(f"  num_target_path_points = {concept_config.num_target_path_points}")

    # ── Collect raw path features and compute all candidates ─────────
    reset_fn = jax.jit(env.reset)
    rng = jax.random.PRNGKey(0)

    all_results = {name: [] for name in CANDIDATE_CONCEPTS}
    raw_path_stats = {"x_min": [], "x_max": [], "y_min": [], "y_max": [],
                      "arc_len": [], "chord_len": []}
    n_collected = 0

    print(f"\nProcessing {args.num_scenarios} scenarios...")
    for scenario in data_gen:
        if n_collected >= args.num_scenarios:
            break

        rng, rk = jax.random.split(rng)
        rk = jax.random.split(rk, 1)
        try:
            env_t = reset_fn(scenario, rk)
        except Exception as e:
            print(f"  [WARN] Scenario {n_collected} failed: {e}")
            n_collected += 1
            continue

        obs = env_t.observation
        inp = observation_to_concept_input(obs, unflatten_fn, concept_config)

        # Denormalize path: (1, P, 2) → metres
        path_xy_m = denorm_xy(inp.path_features, inp.config)  # (1, P, 2)

        # Raw path stats
        p = np.array(path_xy_m[0])  # (P, 2)
        raw_path_stats["x_min"].append(p[:, 0].min())
        raw_path_stats["x_max"].append(p[:, 0].max())
        raw_path_stats["y_min"].append(p[:, 1].min())
        raw_path_stats["y_max"].append(p[:, 1].max())
        segs = np.diff(p, axis=0)
        seg_lens = np.sqrt((segs ** 2).sum(axis=1))
        raw_path_stats["arc_len"].append(seg_lens.sum())
        raw_path_stats["chord_len"].append(np.sqrt((p[-1] - p[0]) @ (p[-1] - p[0])))

        # Compute all candidate concepts
        for name, fn in CANDIDATE_CONCEPTS.items():
            val = float(fn(path_xy_m)[0])
            all_results[name].append(val)

        n_collected += 1
        if n_collected % 20 == 0:
            print(f"  {n_collected}/{args.num_scenarios} done")

    print(f"\nCollected {n_collected} scenarios.\n")

    # ── Section A: Raw path feature statistics ───────────────────────
    print("=" * 70)
    print("A. RAW PATH FEATURE STATISTICS (denormalized, metres)")
    print("=" * 70)
    for key, vals in raw_path_stats.items():
        arr = np.array(vals)
        print(f"  {key:12s}: mean={arr.mean():7.2f}  std={arr.std():6.2f}  "
              f"min={arr.min():7.2f}  max={arr.max():7.2f}")

    # ── Section B: Per-concept statistics ─────────────────────────────
    print(f"\n{'=' * 70}")
    print("B. PER-CONCEPT STATISTICS")
    print("=" * 70)
    concept_arrays = {}
    for name, vals in all_results.items():
        arr = np.array(vals)
        concept_arrays[name] = arr
        nan_count = np.isnan(arr).sum()
        inf_count = np.isinf(arr).sum()
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            print(f"\n  {name}: ALL NaN/Inf ({nan_count} NaN, {inf_count} Inf)")
            continue
        print(f"\n  {name}:")
        print(f"    mean={valid.mean():+10.4f}  std={valid.std():8.4f}")
        print(f"    min ={valid.min():+10.4f}  max={valid.max():+10.4f}")
        print(f"    NaN={nan_count}  Inf={inf_count}")

        # Degeneracy check
        if valid.std() < 1e-6:
            print(f"    ** DEGENERATE: near-constant (std < 1e-6) **")
        elif valid.std() / (np.abs(valid.mean()) + 1e-8) < 0.05:
            print(f"    ** LOW VARIANCE: CoV < 0.05 — limited discriminative power **")

        # For binary-ish concepts
        if name in ("is_turning",):
            pos_rate = (valid > 0.5).mean() * 100
            print(f"    positive rate: {pos_rate:.1f}%")

    # ── Section C: Degeneracy summary ────────────────────────────────
    print(f"\n{'=' * 70}")
    print("C. DEGENERACY SUMMARY (concepts that may be useless at t=0)")
    print("=" * 70)
    for name, arr in concept_arrays.items():
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            print(f"  {name:35s}: UNUSABLE (all NaN/Inf)")
        elif valid.std() < 1e-6:
            print(f"  {name:35s}: DEGENERATE (constant={valid.mean():.4f})")
        elif len(np.unique(np.round(valid, 4))) <= 3:
            print(f"  {name:35s}: NEAR-DEGENERATE (<=3 unique values)")
        else:
            cov = valid.std() / (np.abs(valid.mean()) + 1e-8)
            ent = f"CoV={cov:.3f}"
            unique_pct = len(np.unique(np.round(valid, 2))) / len(valid) * 100
            print(f"  {name:35s}: OK ({ent}, {unique_pct:.0f}% unique rounded vals)")

    # ── Section D: Redundancy analysis (pairwise correlation) ────────
    print(f"\n{'=' * 70}")
    print("D. REDUNDANCY ANALYSIS (Pearson correlation, |r| > 0.8 flagged)")
    print("=" * 70)
    names = list(concept_arrays.keys())
    n = len(names)
    # Build matrix of finite values only (use pairwise deletion)
    for i in range(n):
        for j in range(i + 1, n):
            a = concept_arrays[names[i]]
            b = concept_arrays[names[j]]
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 10:
                continue
            r = np.corrcoef(a[mask], b[mask])[0, 1]
            if abs(r) > 0.6:
                flag = "HIGH" if abs(r) > 0.8 else "moderate"
                print(f"  {names[i]:35s} vs {names[j]:35s}: r={r:+.3f} [{flag}]")

    # ── Section E: Existing concept overlap ──────────────────────────
    print(f"\n{'=' * 70}")
    print("E. OVERLAP WITH EXISTING CONCEPTS")
    print("=" * 70)
    existing_path = ["heading_deviation", "progress_along_route"]
    print(f"  Existing path-based concepts: {existing_path}")
    print(f"  heading_deviation — measures ego yaw vs first path segment angle")
    print(f"  progress_along_route — projects origin onto path, returns arc fraction")
    print()
    # Check overlap: heading_deviation ≈ heading_to_path_end?
    # path_straightness captures something progress_along_route doesn't
    print("  Overlap assessment:")
    print("  - heading_to_path_end overlaps with heading_deviation (both measure path direction)")
    print("  - progress_along_route is always ~0 at t=0 (by definition)")
    print("  - dist_to_path overlaps with progress_along_route lateral component")
    print("  - NEW information from: curvature, straightness, total_heading_change, is_turning")

    # ── Section F: Logged-data vs closed-loop considerations ─────────
    print(f"\n{'=' * 70}")
    print("F. LOGGED-DATA (t=0) vs CLOSED-LOOP RL CONSIDERATIONS")
    print("=" * 70)
    print("""
  At t=0 (logged data / offline extraction):
  - Path is fixed from the scenario's roadgraph. Same path for all policies.
  - dist_to_path ≈ 0 always (SDC starts on its path by construction).
  - progress_along_route ≈ 0 always (SDC at origin, path starts ahead).
  - path geometry (curvature, straightness, turns) IS meaningful:
    these vary per scenario and describe the upcoming road geometry.

  During closed-loop RL rollout:
  - Path is STILL fixed (it's from the roadgraph, not replanned).
  - dist_to_path becomes meaningful (SDC can deviate from path).
  - progress_along_route becomes meaningful (SDC moves along path).
  - Curvature/straightness remain the same (path doesn't change).

  Implication for CBM:
  - Path geometry concepts (curvature, straightness, is_turning) are STATIC
    within a rollout but VARY across scenarios. They describe the
    "situation type" — useful as a conditioning variable but won't
    change with policy actions within an episode.
  - Path deviation concepts (dist_to_path, progress) are DYNAMIC within
    a rollout — they change as the SDC moves. These are the concepts
    that capture the agent's behavior relative to the intended path.
  - For CBM, both types are valuable:
    - Static geometry concepts: explain WHY the policy behaves differently
      in different scenarios (curved road → different acceleration profile)
    - Dynamic deviation concepts: explain WHAT the policy is doing
      relative to the intended route (drifting off path, making progress)
""")

    # ── Section G: Recommendation summary ────────────────────────────
    print(f"{'=' * 70}")
    print("G. PRELIMINARY RECOMMENDATION")
    print("=" * 70)

    # Summarize each candidate
    recommendations = []
    for name, arr in concept_arrays.items():
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            recommendations.append((name, "DEFER", "all NaN/Inf"))
            continue
        std = valid.std()
        if std < 1e-6:
            recommendations.append((name, "DEFER", f"degenerate at t=0 (const={valid.mean():.4f})"))
        elif std < 1e-4:
            recommendations.append((name, "MAYBE", f"very low variance (std={std:.6f})"))
        else:
            recommendations.append((name, "KEEP", f"std={std:.4f}, range=[{valid.min():.4f}, {valid.max():.4f}]"))

    for name, status, reason in recommendations:
        marker = {"KEEP": "+", "MAYBE": "?", "DEFER": "-"}[status]
        print(f"  [{marker}] {name:35s}: {status:5s} — {reason}")

    print(f"\n{'=' * 70}")
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
