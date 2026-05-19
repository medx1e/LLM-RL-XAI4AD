"""Feature annotation via z-score and Spearman correlation with telemetry.

For each SAE latent dimension j (out of F = D * expansion_factor):

**Z-score mode (use_correlation=False):**
1. Find the top-K timesteps where feature j activates most strongly.
2. Query the aligned telemetry for those timesteps.
3. Compute z-scores: z = (mean_topK - mean_global) / std_global.
4. Rank telemetry fields by |z| to identify the concept the feature encodes.

**Correlation mode (use_correlation=True, default):**
1. Compute Spearman ρ between the continuous feature activations and each
   telemetry field across ALL timesteps.
2. Rank by |ρ|.  A feature with ρ=0.85 with ego_speed is a much stronger
   finding than a z-score of 1.37 — it uses the full dataset and captures
   graded relationships.

Outputs a JSON file with per-feature annotations and a summary.

Usage:
    python -m sae_interpretability.feature_annotator \\
        --data harvest.h5 \\
        --sae_checkpoint sae_model.pt \\
        --top_k 50 \\
        --output annotations.json

    # Correlation mode (default):
    python -m sae_interpretability.feature_annotator \\
        --data harvest.h5 \\
        --sae_checkpoint sae_model.pt \\
        --output annotations.json --use_correlation
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import torch

from sae_interpretability.sae_model import SparseAutoencoder
from sae_interpretability.feature_analysis import run_analysis


# Which HDF5 telemetry fields to include in z-score analysis.
# Per-agent fields are collapsed to scalar summary statistics first.
#
# NOTE: ego_x / ego_y intentionally excluded — raw map coordinates are not
# driving concepts and absorb SAE features with spurious position variance.
_SCALAR_TEL_KEYS = [
    'ego_speed',
    'ego_lat_accel',
    'ego_yaw_rate',
    'min_ttc',
    # NOTE: 'min_agent_dist' intentionally excluded — this scalar was harvested
    # as all-zeros because the ego vehicle slot is included in the raw distance
    # array at distance 0 and the harvester did not exclude it.  The corrected
    # measure is 'agent_dists_min' computed below with proper ego-exclusion.
    'n_valid_agents',
    'lead_vehicle_dist',
    'lead_vehicle_closing_speed',
]
_BOOL_TEL_KEYS = [
    'is_ttc_critical',
    'is_lead_vehicle_hard_braking',
    'is_following_lead',
    'has_agent_left',
    'has_agent_right',
]
_AGENT_KEYS_TO_SUMMARIZE = [
    ('agent_dists', ['mean', 'min']),
    ('agent_ttcs', ['mean', 'min']),
    ('agent_speeds', ['mean', 'max']),
    ('agent_closing_speed', ['mean', 'max']),
]


def _load_telemetry(hf: h5py.File) -> Dict[str, np.ndarray]:
    """Load and flatten telemetry from an open HDF5 file.

    Per-agent reductions (mean / min / max) are computed only over **valid**
    agent slots.  Without masking, invalid slots are zero-filled by the
    simulator which causes e.g. ``agent_dists_min`` to always be 0.0 —
    a constant field that Spearman cannot use.

    Fallback sentinels when no valid agents exist at a timestep:
      - min  → 999.0  (treat as "no nearby agent")
      - mean → 0.0
      - max  → 0.0
    """
    tel: Dict[str, np.ndarray] = {}

    for k in _SCALAR_TEL_KEYS:
        path = f'telemetry/{k}'
        if path in hf:
            tel[k] = hf[path][:].astype(np.float32)

    for k in _BOOL_TEL_KEYS:
        path = f'telemetry/{k}'
        if path in hf:
            # float for z-score arithmetic
            tel[k] = hf[path][:].astype(np.float32)

    # Load agent validity mask once — shape [N, n_agents], bool.
    # Used to mask out zero-filled invalid slots before reduction.
    agent_valid: Optional[np.ndarray] = None
    if 'telemetry/agent_valid' in hf:
        agent_valid = hf['telemetry/agent_valid'][:].astype(bool)
        valid_frac = agent_valid.mean()
        n_timesteps_no_agents = int((agent_valid.sum(axis=1) == 0).sum())
        print(f"[Telemetry] agent_valid found — shape={agent_valid.shape}, "
              f"valid_frac={valid_frac:.3f}, "
              f"timesteps_with_no_valid_agent={n_timesteps_no_agents}/{agent_valid.shape[0]}")
    else:
        # List what keys ARE present under telemetry/ to help diagnose
        tel_keys_present = [k for k in hf.keys() if k == 'telemetry']
        if 'telemetry' in hf:
            present = sorted(hf['telemetry'].keys())
        else:
            present = []
        print(f"[Telemetry] WARNING: 'telemetry/agent_valid' NOT found in HDF5. "
              f"Will fall back to unmasked reductions (agent_dists_min will be wrong).\n"
              f"           Keys present under telemetry/: {present}")

    # Diagnostic: show unique-value counts for fields known to be problematic
    for _diag_key in ('min_agent_dist',):
        _path = f'telemetry/{_diag_key}'
        if _path in hf:
            _arr = hf[_path][:].astype(np.float32)
            _unique = np.unique(_arr)
            print(f"[Telemetry] {_diag_key}: shape={_arr.shape}, "
                  f"n_unique={len(_unique)}, "
                  f"min={_arr.min():.4f}, max={_arr.max():.4f}, "
                  f"mean={_arr.mean():.4f}"
                  + (f", unique_vals={_unique.tolist()}" if len(_unique) <= 5 else ""))

    for k, stats in _AGENT_KEYS_TO_SUMMARIZE:
        path = f'telemetry/{k}'
        if path not in hf:
            continue
        arr = hf[path][:]  # [N, n_agents]

        if agent_valid is not None:
            if 'mean' in stats:
                # nanmean over valid slots; NaN where no valid agent → 0.0
                masked_mean = np.where(agent_valid, arr, np.nan)
                result = np.nanmean(masked_mean, axis=1)
                tel[f'{k}_mean'] = np.nan_to_num(
                    result, nan=0.0).astype(np.float32)

            if 'min' in stats:
                # Replace invalid slots with large sentinel so they never win.
                # For distance arrays, also exclude near-zero entries (< 0.5 m)
                # which correspond to the ego vehicle itself — it is present in
                # the agent array at index sdc_idx with distance == 0, marked
                # valid, so we must filter it out explicitly.
                valid_for_min = agent_valid.copy()
                if k == 'agent_dists':
                    valid_for_min = valid_for_min & (arr >= 0.5)
                masked_min = np.where(valid_for_min, arr,
                                      np.full_like(arr, 999.0))
                tel[f'{k}_min'] = masked_min.min(axis=1).astype(np.float32)

            if 'max' in stats:
                # Replace invalid slots with -inf so they never win
                masked_max = np.where(agent_valid, arr,
                                      np.full_like(arr, -np.inf))
                result = masked_max.max(axis=1)
                tel[f'{k}_max'] = np.nan_to_num(
                    result, neginf=0.0).astype(np.float32)
        else:
            # No validity mask available — fall back to unmasked reductions
            # (legacy behaviour; will warn if fields are constant)
            if 'mean' in stats:
                tel[f'{k}_mean'] = arr.mean(axis=1)
            if 'min' in stats:
                tel[f'{k}_min'] = arr.min(axis=1)
            if 'max' in stats:
                tel[f'{k}_max'] = arr.max(axis=1)

    return tel


# ---------------------------------------------------------------------------
# Spearman correlation helpers
# ---------------------------------------------------------------------------

def _compute_spearman_correlations(
    features: np.ndarray,
    tel: Dict[str, np.ndarray],
    min_activations: int,
) -> Dict[int, Dict[str, float]]:
    """Compute Spearman ρ between each feature and each telemetry field.

    Only considers features with ≥ min_activations non-zero values.

    Guard conditions applied before each spearmanr call:
    - Skip if the telemetry field is constant (std ≈ 0). This is the cause of
      ConstantInputWarning for boolean fields like is_ttc_critical that are
      all-False in safe-driving datasets.
    - Skip if the feature itself is constant (shouldn't happen given the
      min_activations guard, but defensive).

    For very sparse features (< min_activations_for_spearman activations),
    the Spearman ρ is unreliable because almost all values are tied at 0,
    swamping the rank calculation. Those features still get ρ=0.0 and will be
    ranked by z-score in the caller instead.

    Returns:
        Dict mapping feature_idx → {tel_field: ρ_value}.
    """
    from scipy.stats import spearmanr

    F_dim = features.shape[1]
    tel_keys = list(tel.keys())
    correlations: Dict[int, Dict[str, float]] = {}

    # Pre-compute std for every telemetry field once — O(N) per field.
    tel_stds = {k: float(v.std()) for k, v in tel.items()}
    n_skipped_constant_tel = sum(1 for s in tel_stds.values() if s < 1e-9)
    if n_skipped_constant_tel:
        print(f"[Annotator] Warning: {n_skipped_constant_tel} telemetry field(s) "
              f"are constant across the dataset and will be skipped in Spearman. "
              f"Fields: {[k for k, s in tel_stds.items() if s < 1e-9]}")

    print(f"[Annotator] Computing Spearman ρ for {F_dim} features × "
          f"{len(tel_keys)} telemetry fields ...")

    for j in range(F_dim):
        feat_j = features[:, j]
        n_nonzero = int((feat_j > 0).sum())

        if n_nonzero < min_activations:
            continue

        feat_std = float(feat_j.std())
        rho_dict: Dict[str, float] = {}

        for k in tel_keys:
            # Guard 1: constant telemetry field (e.g. all-False boolean in a
            # dataset with no critical events). spearmanr would emit
            # ConstantInputWarning and return NaN.
            if tel_stds[k] < 1e-9:
                rho_dict[k] = 0.0
                continue

            # Guard 2: constant feature — defensive, already covered by the
            # n_nonzero check above but kept for safety.
            if feat_std < 1e-9:
                rho_dict[k] = 0.0
                continue

            # NOTE: no guard on n_nonzero here. Sparse features work correctly
            # with Spearman because all N-k zeros receive the same tied rank
            # (~N/2), making their contribution to the covariance zero. The
            # signal comes entirely from the k active timesteps. A feature that
            # fires only 5 times but always during hard-braking events will still
            # return ρ ≈ 1.0 with is_lead_vehicle_hard_braking.
            rho, _ = spearmanr(feat_j, tel[k])
            rho_dict[k] = round(float(rho) if np.isfinite(rho) else 0.0, 4)

        correlations[j] = rho_dict

        if (j + 1) % 500 == 0:
            print(f"  ... {j + 1}/{F_dim} features done")

    print(f"[Annotator] Spearman done — {len(correlations)} active features.")
    return correlations


# ---------------------------------------------------------------------------
# Main annotation entry point
# ---------------------------------------------------------------------------

def annotate(
    data_path: str,
    sae_checkpoint: str,
    top_k: int = 50,
    output_path: str = "annotations.json",
    min_activations: int = 10,
    use_correlation: bool = True,
    run_grouped_analysis: bool = True,
    fig_dir: str = "data/sae_interpretability/figures",
) -> List[Dict[str, Any]]:
    """Annotate SAE features by correlating with telemetry.

    Args:
        data_path: Path to harvest.h5.
        sae_checkpoint: Path to trained SAE .pt checkpoint.
        top_k: Number of top-activating timesteps per feature (z-score mode).
        output_path: Where to save the annotations JSON.
        min_activations: Features with fewer non-zero activations are labelled 'dead'.
        use_correlation: If True, use Spearman ρ (continuous, all timesteps) for
            ranking and labelling.  If False, use top-K z-scores.  Both metrics
            are always computed and stored; this flag controls which one drives
            the label and the "most selective" ranking.

    Returns:
        Sorted list of annotation dicts (active features first, then dead).
    """
    mode_str = "Spearman ρ" if use_correlation else "top-K z-score"
    print(f"[Annotator] Annotation mode: {mode_str}")

    model = SparseAutoencoder.from_checkpoint(
        sae_checkpoint, map_location='cpu')
    model.eval()
    print(
        f"[Annotator] SAE: {model.input_dim}D → {model.latent_dim}D  ({sae_checkpoint})")

    print(f"[Annotator] Loading data from {data_path}")
    with h5py.File(data_path, 'r') as hf:
        activations = hf['activations'][:]   # [N, D]
        tel = _load_telemetry(hf)

    N, D = activations.shape
    print(f"[Annotator] {N:,} rows, {D} dims, {len(tel)} telemetry fields")

    # Apply the same whitening used during training (mean subtract + std divide)
    act_mean: Optional[np.ndarray] = None
    act_std: Optional[np.ndarray] = None
    try:
        ckpt = torch.load(sae_checkpoint, map_location='cpu',
                          weights_only=False)
        act_mean = ckpt.get('act_mean', None)
        act_std  = ckpt.get('act_std',  None)
    except Exception:
        pass

    if act_mean is not None and act_std is not None:
        act_in = ((activations - act_mean) / act_std).astype(np.float32)
    elif act_mean is not None:
        act_in = (activations - act_mean).astype(np.float32)
    else:
        act_in = activations.astype(np.float32)

    # Encode in batches to avoid OOM
    x = torch.from_numpy(act_in)
    chunks = []
    with torch.no_grad():
        for i in range(0, N, 8192):
            chunks.append(model.encode(x[i:i + 8192]).numpy())
    features = np.concatenate(chunks, axis=0)  # [N, F]

    F_dim = features.shape[1]
    print(
        f"[Annotator] Encoded → {features.shape}. Annotating {F_dim} features...")

    # Global stats for z-scores (only needed in z-score mode)
    global_stats = None
    if not use_correlation:
        global_stats = {k: (float(v.mean()), float(v.std()) + 1e-8)
                        for k, v in tel.items()}

    # Spearman correlations (only computed in correlation mode)
    spearman_map = None
    if use_correlation:
        spearman_map = _compute_spearman_correlations(
            features, tel, min_activations)

    annotations: List[Dict[str, Any]] = []
    n_active = 0

    for j in range(F_dim):
        feat_j = features[:, j]
        n_nonzero = int((feat_j > 0).sum())

        if n_nonzero < min_activations:
            annotations.append(
                {'feature_idx': j, 'label': 'dead', 'n_activations': n_nonzero})
            continue

        n_active += 1

        if use_correlation:
            # --- Spearman ρ mode ---
            rho_scores = spearman_map.get(j, {})
            ranked = sorted(rho_scores.items(),
                            key=lambda kv: abs(kv[1]), reverse=True)
            top_field, top_val = ranked[0] if ranked else ('', 0.0)
            direction = "high" if top_val > 0 else "low"
            label = f"{direction}_{top_field} (ρ={top_val:+.3f})"
            selectivity_score = round(abs(top_val), 4)

            annotations.append({
                'feature_idx': j,
                'label': label,
                'n_activations': n_nonzero,
                'max_abs_rho': selectivity_score,
                'selectivity_score': selectivity_score,
                'mean_activation_when_active': round(float(feat_j[feat_j > 0].mean()), 5),
                'top_correlated_fields': [{'field': f, 'rho': r} for f, r in ranked[:5]],
                'all_rho_scores': rho_scores,
            })
        else:
            # --- Z-score mode ---
            k_actual = min(top_k, n_nonzero)
            top_idx = np.argpartition(feat_j, -k_actual)[-k_actual:]
            top_idx = top_idx[np.argsort(feat_j[top_idx])[::-1]]

            z_scores: Dict[str, float] = {}
            for k, vals in tel.items():
                topk_mean = float(vals[top_idx].mean())
                g_mean, g_std = global_stats[k]
                z_scores[k] = round((topk_mean - g_mean) / g_std, 4)

            ranked = sorted(z_scores.items(),
                            key=lambda kv: abs(kv[1]), reverse=True)
            top_field, top_val = ranked[0]
            direction = "high" if top_val > 0 else "low"
            label = f"{direction}_{top_field} (z={top_val:+.2f})"
            selectivity_score = round(abs(top_val), 4)

            annotations.append({
                'feature_idx': j,
                'label': label,
                'n_activations': n_nonzero,
                'max_abs_z': selectivity_score,
                'selectivity_score': selectivity_score,
                'mean_activation_when_active': round(float(feat_j[feat_j > 0].mean()), 5),
                'top_correlated_fields': [{'field': f, 'z': z} for f, z in ranked[:5]],
                'all_z_scores': z_scores,
            })

    print(
        f"[Annotator] {n_active}/{F_dim} features active (≥{min_activations} activations)")

    active = sorted(
        [a for a in annotations if a['label'] != 'dead'],
        key=lambda a: a['n_activations'],
        reverse=True,
    )
    dead = [a for a in annotations if a['label'] == 'dead']
    sorted_annotations = active + dead

    # Selective features: sorted by the mode's primary score
    selective = sorted(
        [a for a in annotations if a['label'] != 'dead'],
        key=lambda a: a['selectivity_score'],
        reverse=True,
    )

    metric_key = 'max_abs_rho' if use_correlation else 'max_abs_z'
    metric_label = '|ρ|' if use_correlation else '|z|'
    val_sym = 'ρ' if use_correlation else 'z'

    summary = {
        'total_features': F_dim,
        'active_features': n_active,
        'dead_features': F_dim - n_active,
        'dead_pct': round(100.0 * (F_dim - n_active) / F_dim, 2),
        'annotation_mode': 'spearman' if use_correlation else 'zscore',
        'top_10_most_active': [
            {'idx': a['feature_idx'], 'label': a['label'],
                'n_activations': a['n_activations']}
            for a in active[:10]
        ],
        'top_10_most_selective': [
            {'idx': a['feature_idx'], 'label': a['label'],
                'n_activations': a['n_activations'],
                metric_key: a.get(metric_key, 0)}
            for a in selective[:10]
        ],
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path))
                or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(
            {'summary': summary, 'annotations': sorted_annotations}, f, indent=2)

    print(f"[Annotator] Saved → {output_path}")

    _print_feature_table(
        f"Top 10 Most Selective Features (by {metric_label})",
        selective[:10], metric_label, val_sym)

    # --- Grouped analysis & figures ---
    if run_grouped_analysis and use_correlation:
        tel_keys = list(tel.keys())
        run_analysis(sorted_annotations, tel_keys, fig_dir=fig_dir)
    elif run_grouped_analysis and not use_correlation:
        print("[Annotator] Grouped analysis requires correlation mode "
              "(--use_correlation). Skipping figures.")

    return sorted_annotations


def _print_feature_table(
    title: str,
    features: List[Dict[str, Any]],
    metric_label: str = '|z|',
    val_sym: str = 'z',
) -> None:
    corr_key = 'top_correlated_fields'

    print(f"\n--- {title} ---")
    for ann in features:
        score = ann.get('selectivity_score', 0)
        print(
            f"  [{ann['feature_idx']:4d}] n={ann['n_activations']:5d}  "
            f"{metric_label}={score:.3f}  {ann['label']}")
        for entry in ann.get(corr_key, [])[:3]:
            val = entry.get(val_sym, entry.get('z', entry.get('rho', 0)))
            print(f"           {entry['field']}: {val_sym}={val:+.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Annotate SAE features via telemetry z-scores / Spearman ρ.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--sae_checkpoint", required=True)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--output", default="data/sae_interpretability/annotations.json")
    parser.add_argument("--min_activations", type=int, default=10)
    parser.add_argument(
        "--use_correlation",
        action="store_true",
        default=True,
        help="Use Spearman ρ (all timesteps) instead of top-K z-scores "
             "for labelling and ranking (default: True)")
    parser.add_argument(
        "--no_correlation",
        action="store_true",
        default=False,
        help="Disable Spearman mode and use top-K z-scores instead")
    parser.add_argument(
        "--no_analysis",
        action="store_true",
        default=False,
        help="Skip grouped analysis and figure generation")
    parser.add_argument(
        "--fig_dir",
        default="data/sae_interpretability/figures",
        help="Output directory for analysis figures")
    args = parser.parse_args()

    use_corr = args.use_correlation and not args.no_correlation

    annotate(
        args.data, args.sae_checkpoint, args.top_k,
        args.output, args.min_activations,
        use_correlation=use_corr,
        run_grouped_analysis=not args.no_analysis,
        fig_dir=args.fig_dir,
    )


if __name__ == "__main__":
    main()
