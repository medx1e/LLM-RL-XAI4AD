"""Attention Decomposition Analysis — Single-Feature Correlations of Combined Attention.

Pivot from Head Specialization (which showed both heads doing ~same thing):
Instead of asking "what does each head do?", we ask "what semantic factors explain
the encoder's COMBINED attention, and how does that change with risk?"

Method:
  At each timestep, compute Pearson correlation between combined attention and
  each semantic feature independently across vehicles. This avoids the severe
  multicollinearity that destroyed the multivariate regression (distance, TTC,
  closing_speed are mathematically related — TTC ≈ distance/closing_speed).

  Single-feature correlation with n≈5 vehicles and k=1 is stable: the standardized
  beta from a univariate regression equals the Pearson r, bounded in [-1, 1].

  Additionally, a 2-feature model (distance + TTC only) provides an overall R²
  summary statistic. With n≈5 and k=2, this has 2 degrees of freedom — tight
  but not degenerate.

Three outputs:
  1. Overall decomposition — average single-feature r and 2-feature R²
  2. Risk-conditioned — compare correlations in calm (R < 0.1) vs danger (R > 0.3)
  3. Correlation-risk — does each feature's importance change with risk?

Usage:
    python attention_decomposition_analysis.py \\
        --extraction-dir research-RL-BC/extractions \\
        --output-dir research-RL-BC/figures_decomposition
"""

import argparse
import glob
import os
import pickle
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ==============================================================================
# Data Class
# ==============================================================================

@dataclass
class DecompositionResult:
    """Results from attention decomposition analysis."""
    overall: Dict[str, Any]
    risk_conditioned: Dict[str, Any]
    beta_risk_correlations: Dict[str, Any]
    feature_names: List[str]
    n_episodes: int
    n_total_timesteps: int
    n_skipped_timesteps: int


# ==============================================================================
# Feature Configuration
# ==============================================================================

# All 4 features for single-feature correlations
REGRESSION_FEATURES = ['distance_to_ego', 'ttc', 'closing_speed', 'agent_speeds']

# 2-feature subset for the joint R² summary (least collinear pair)
R2_FEATURES = ['distance_to_ego', 'ttc']

# Human-readable labels for plots
FEATURE_LABELS = {
    'distance_to_ego': 'Distance',
    'ttc': 'TTC',
    'closing_speed': 'Closing Speed',
    'agent_speeds': 'Agent Speed',
}


# ==============================================================================
# Attention Decomposition Analyzer
# ==============================================================================

class AttentionDecompositionAnalyzer:
    """Decompose combined attention into semantic feature contributions
    using single-feature correlations at each timestep.

    Instead of multivariate regression (which suffers from severe multicollinearity
    with only ~5 valid vehicles), we run 4 independent univariate analyses.
    The standardized beta from a single-feature regression equals the Pearson r.

    A 2-feature model (distance + TTC) provides an overall R² summary.
    """

    # Risk thresholds
    CALM_THRESHOLD = 0.1
    DANGER_THRESHOLD = 0.3

    # Minimum valid vehicles for correlation (need >= 3)
    MIN_VALID_VEHICLES = 3

    # Minimum timesteps per episode for correlation-risk analysis
    MIN_TIMESTEPS = 10

    # Minimum qualifying episodes for Fisher z aggregation
    MIN_QUALIFYING_EPISODES = 3

    # Risk variation threshold for correlation-risk analysis
    HV_THRESHOLD = 0.05

    def __init__(self, extraction_dir: str):
        self.extraction_dir = extraction_dir
        self._extraction_data = None
        self._result: Optional[DecompositionResult] = None

    # =========================================================================
    # Data Loading
    # =========================================================================

    def load_data(self, extraction_file: Optional[str] = None):
        """Load extraction data from pickle file."""
        if extraction_file is None:
            pattern = os.path.join(self.extraction_dir, "extraction_*.pkl")
            files = sorted(glob.glob(pattern))
            if not files:
                raise FileNotFoundError(
                    f"No extraction files found in {self.extraction_dir}"
                )

            final_file = None
            max_step = -1
            max_step_file = None

            for fpath in files:
                fname = os.path.basename(fpath)
                if 'final' in fname.lower():
                    final_file = fpath
                match = re.search(r'model_(\d+)', fname)
                if match:
                    step = int(match.group(1))
                    if step > max_step:
                        max_step = step
                        max_step_file = fpath

            extraction_file = final_file or max_step_file or files[0]
            print(f"[Decomp] Auto-selected: {os.path.basename(extraction_file)}")
        else:
            if not os.path.isabs(extraction_file):
                extraction_file = os.path.join(self.extraction_dir, extraction_file)

        with open(extraction_file, 'rb') as f:
            self._extraction_data = pickle.load(f)

        mode = self._extraction_data.get('extraction_mode', 'unknown')
        n = self._extraction_data['n_scenarios']
        print(f"[Decomp] Loaded {n} scenarios ({mode} mode)")
        print(f"[Decomp] Checkpoint: {self._extraction_data['checkpoint']} "
              f"(step {self._extraction_data['step']})")

        if mode != 'rollout':
            print("[Decomp] WARNING: Data was NOT extracted with --rollout mode.")
            print("[Decomp] Decomposition requires multi-timestep data.")

        return self

    # =========================================================================
    # Episode Loading
    # =========================================================================

    def _load_episode(self, scenario_data: Dict) -> Tuple[np.ndarray, np.ndarray, Dict, np.ndarray, int]:
        """Convert a scenario's timestep list into structured arrays.

        Returns:
            combined_attention: (T, n_vehicles) — attention averaged across heads, normalized
            risk_timeseries: (T,)
            semantic_timeseries: dict of arrays, e.g. 'ttc': (T, n_vehicles)
            valid_timeseries: (T, n_vehicles) bool
            sdc_index: int — ego vehicle index
        """
        timesteps = scenario_data['timesteps']
        T = len(timesteps)

        # --- Combined attention ---
        first_dist = timesteps[0]['attention_distributions']
        if 'per_vehicle_attention' not in first_dist:
            raise ValueError("No per_vehicle_attention in extraction data")

        pva_stack = np.stack(
            [ts['attention_distributions']['per_vehicle_attention'] for ts in timesteps],
            axis=0
        )  # (T, n_vehicles, n_heads)

        # Average across heads → (T, n_vehicles)
        combined = pva_stack.mean(axis=2)

        # Normalize each timestep to sum to 1
        row_sums = combined.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        combined = combined / row_sums

        # --- Risk ---
        risk_timeseries = np.array([ts['collision_risk'] for ts in timesteps])

        # --- Semantic features ---
        skip_fields = {'sdc_index', 'timestep', 'valid', 'positions_x', 'positions_y',
                       'ego_speed', 'object_types'}
        semantic_timeseries = {}
        feature_keys = [k for k in timesteps[0]['semantic_features'].keys()
                        if k not in skip_fields]
        for feat in feature_keys:
            try:
                semantic_timeseries[feat] = np.stack(
                    [ts['semantic_features'][feat] for ts in timesteps], axis=0
                )
            except (ValueError, KeyError):
                continue

        # --- Valid mask ---
        if 'valid' in timesteps[0]['semantic_features']:
            valid_timeseries = np.stack(
                [ts['semantic_features']['valid'] for ts in timesteps], axis=0
            ).astype(bool)
        else:
            n_v = combined.shape[1]
            valid_timeseries = np.ones((T, n_v), dtype=bool)

        # --- SDC index ---
        sdc_index = int(timesteps[0]['semantic_features'].get('sdc_index', 0))

        return combined, risk_timeseries, semantic_timeseries, valid_timeseries, sdc_index

    # =========================================================================
    # Vehicle Mask (shared logic)
    # =========================================================================

    def _get_vehicle_mask(self, attention: np.ndarray, valid_mask: np.ndarray,
                          sdc_index: int) -> np.ndarray:
        """Build mask of valid, non-ego, non-padding vehicles at a single timestep.

        Args:
            attention: (n_vehicles,)
            valid_mask: (n_vehicles,) bool
            sdc_index: ego index to exclude

        Returns:
            (n_vehicles,) bool mask
        """
        n_vehicles = len(attention)
        mask = valid_mask.copy()

        # Exclude ego vehicle
        if 0 <= sdc_index < n_vehicles:
            mask[sdc_index] = False

        # Exclude non-finite attention
        mask &= np.isfinite(attention)

        # Exclude padding (zero attention AND could be padding)
        mask &= attention > 1e-10

        return mask

    # =========================================================================
    # Per-Timestep Single-Feature Correlation
    # =========================================================================

    def _correlate_single_feature(
        self,
        attention: np.ndarray,
        feature: np.ndarray,
        mask: np.ndarray,
    ) -> Optional[Dict]:
        """Compute Pearson correlation between attention and a single feature.

        The standardized beta from univariate regression equals Pearson r.
        Bounded in [-1, 1], stable with as few as 3 datapoints.

        Args:
            attention: (n_vehicles,) combined attention fractions
            feature: (n_vehicles,) single semantic feature
            mask: (n_vehicles,) bool — valid vehicles

        Returns:
            dict with 'r' (Pearson), 'pval', 'n_valid'
            or None if can't compute
        """
        # Further filter for finite feature values
        feat_mask = mask & np.isfinite(feature)
        n_valid = feat_mask.sum()

        if n_valid < self.MIN_VALID_VEHICLES:
            return None

        a_valid = attention[feat_mask]
        f_valid = feature[feat_mask]

        # Check variance
        if a_valid.std() < 1e-10 or f_valid.std() < 1e-10:
            return None

        r, pval = stats.pearsonr(a_valid, f_valid)
        if not np.isfinite(r):
            return None

        return {
            'r': float(r),
            'pval': float(pval),
            'n_valid': int(n_valid),
        }

    # =========================================================================
    # Per-Timestep 2-Feature R² (Distance + TTC)
    # =========================================================================

    def _compute_r2_two_feature(
        self,
        attention: np.ndarray,
        semantic_timeseries: Dict,
        t: int,
        mask: np.ndarray,
    ) -> Optional[Dict]:
        """Fit 2-feature standardized regression (distance + TTC) for R² summary.

        With n≈5 and k=2, we have ~2 degrees of freedom — tight but not degenerate.

        Returns:
            dict with 'r_squared', 'r_squared_adj', 'n_valid', 'betas'
            or None if can't fit
        """
        # Check both features exist
        columns = []
        feat_names = []
        for feat in R2_FEATURES:
            if feat not in semantic_timeseries:
                return None
            col = semantic_timeseries[feat][t]
            columns.append(col)
            feat_names.append(feat)

        F = np.column_stack(columns)  # (n_vehicles, 2)

        # Apply mask + filter finite features
        feat_mask = mask.copy()
        feat_mask &= np.all(np.isfinite(F), axis=1)
        n_valid = feat_mask.sum()

        # Need at least k+2 = 4 vehicles
        if n_valid < 4:
            return None

        a_valid = attention[feat_mask]
        F_valid = F[feat_mask]

        if a_valid.std() < 1e-10:
            return None

        # Check feature variance
        keep = []
        for i in range(F_valid.shape[1]):
            if F_valid[:, i].std() > 1e-10:
                keep.append(i)

        if len(keep) == 0:
            return None

        F_valid = F_valid[:, keep]
        feat_names_used = [feat_names[i] for i in keep]
        k = len(keep)

        if n_valid < k + 2:
            return None

        # Standardize
        scaler = StandardScaler()
        F_std = scaler.fit_transform(F_valid)
        a_std = (a_valid - a_valid.mean()) / a_valid.std()

        reg = LinearRegression(fit_intercept=False)
        reg.fit(F_std, a_std)

        r2 = float(reg.score(F_std, a_std))
        n = int(n_valid)
        if n > k + 1:
            r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)
        else:
            r2_adj = float('nan')

        betas = {fn: float(c) for fn, c in zip(feat_names_used, reg.coef_)}

        return {
            'r_squared': r2,
            'r_squared_adj': float(r2_adj),
            'n_valid': n,
            'betas': betas,
        }

    # =========================================================================
    # Run Decomposition Across All Episodes
    # =========================================================================

    def run_decomposition(self) -> List[Dict]:
        """Run per-timestep single-feature correlations across all episodes.

        For each timestep, computes:
          - Pearson r between attention and each feature independently
          - R² from 2-feature model (distance + TTC)

        Returns:
            List of episode results with correlation timeseries.
        """
        if self._extraction_data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        episodes = self._extraction_data['scenarios']
        all_episode_results = []
        total_skipped = 0
        total_computed = 0

        for ep_idx, episode in enumerate(episodes):
            combined_attn, risk, semantics, valid_ts, sdc_idx = self._load_episode(episode)
            T = combined_attn.shape[0]

            # Determine available features
            available_features = [f for f in REGRESSION_FEATURES if f in semantics]
            if len(available_features) == 0:
                continue

            # Initialize timeseries storage
            corr_series = {f: np.full(T, np.nan) for f in available_features}
            pval_series = {f: np.full(T, np.nan) for f in available_features}
            r2_series = np.full(T, np.nan)
            r2_adj_series = np.full(T, np.nan)
            r2_betas = {f: np.full(T, np.nan) for f in R2_FEATURES if f in semantics}
            n_valid_series = np.zeros(T, dtype=int)

            for t in range(T):
                mask = self._get_vehicle_mask(combined_attn[t], valid_ts[t], sdc_idx)
                n_valid = mask.sum()

                if n_valid < self.MIN_VALID_VEHICLES:
                    total_skipped += 1
                    continue

                any_computed = False
                n_valid_series[t] = n_valid

                # Single-feature correlations
                for feat in available_features:
                    feat_vals = semantics[feat][t]
                    result = self._correlate_single_feature(
                        combined_attn[t], feat_vals, mask
                    )
                    if result is not None:
                        corr_series[feat][t] = result['r']
                        pval_series[feat][t] = result['pval']
                        any_computed = True

                # 2-feature R²
                r2_result = self._compute_r2_two_feature(
                    combined_attn[t], semantics, t, mask
                )
                if r2_result is not None:
                    r2_series[t] = r2_result['r_squared']
                    r2_adj_series[t] = r2_result['r_squared_adj']
                    for fn, bv in r2_result['betas'].items():
                        if fn in r2_betas:
                            r2_betas[fn][t] = bv

                if any_computed:
                    total_computed += 1
                else:
                    total_skipped += 1

            all_episode_results.append({
                'episode_idx': ep_idx,
                'corr_timeseries': corr_series,       # {feat: (T,) Pearson r}
                'pval_timeseries': pval_series,       # {feat: (T,) p-values}
                'r2_timeseries': r2_series,           # (T,) 2-feature R²
                'r2_adj_timeseries': r2_adj_series,   # (T,) adjusted R²
                'r2_betas': r2_betas,                 # {feat: (T,)} betas from 2-feat model
                'risk_timeseries': risk,
                'n_valid_timeseries': n_valid_series,
                'combined_attention': combined_attn,
                'n_timesteps': T,
                'feature_names': available_features,
            })

        print(f"[Decomp] Processed {len(all_episode_results)} episodes")
        print(f"[Decomp] {total_computed} timesteps computed, {total_skipped} skipped")
        return all_episode_results

    # =========================================================================
    # Aggregation
    # =========================================================================

    def aggregate_results(self, episode_results: List[Dict]) -> DecompositionResult:
        """Aggregate decomposition results across episodes.

        Produces three outputs:
          1. Overall — mean single-feature r per feature + 2-feature R²
          2. Risk-conditioned — calm vs danger correlations, Mann-Whitney U
          3. Correlation-risk — Fisher z-aggregated Spearman ρ(r(t), R(t))
        """
        feature_names = episode_results[0]['feature_names']

        # =====================================================================
        # OUTPUT 1: Overall decomposition
        # =====================================================================
        all_corrs = {f: [] for f in feature_names}
        all_r2 = []
        all_r2_adj = []

        for ep in episode_results:
            for feat in feature_names:
                vals = ep['corr_timeseries'][feat]
                all_corrs[feat].extend(vals[np.isfinite(vals)].tolist())
            r2 = ep['r2_timeseries']
            all_r2.extend(r2[np.isfinite(r2)].tolist())
            r2a = ep['r2_adj_timeseries']
            all_r2_adj.extend(r2a[np.isfinite(r2a)].tolist())

        overall = {
            'mean_corrs': {f: float(np.mean(v)) for f, v in all_corrs.items() if v},
            'std_corrs': {f: float(np.std(v)) for f, v in all_corrs.items() if v},
            'median_corrs': {f: float(np.median(v)) for f, v in all_corrs.items() if v},
            'mean_r_squared': float(np.mean(all_r2)) if all_r2 else float('nan'),
            'std_r_squared': float(np.std(all_r2)) if all_r2 else float('nan'),
            'mean_r_squared_adj': float(np.mean(all_r2_adj)) if all_r2_adj else float('nan'),
            'std_r_squared_adj': float(np.std(all_r2_adj)) if all_r2_adj else float('nan'),
            'n_total_timesteps': len(all_r2),
            'n_per_feature': {f: len(v) for f, v in all_corrs.items()},
        }

        # =====================================================================
        # OUTPUT 2: Risk-conditioned decomposition
        # =====================================================================
        calm_corrs = {f: [] for f in feature_names}
        danger_corrs = {f: [] for f in feature_names}
        calm_r2 = []
        danger_r2 = []

        for ep in episode_results:
            risk = ep['risk_timeseries']
            r2 = ep['r2_timeseries']

            for t in range(len(risk)):
                is_calm = risk[t] < self.CALM_THRESHOLD
                is_danger = risk[t] > self.DANGER_THRESHOLD

                if not is_calm and not is_danger:
                    continue

                for feat in feature_names:
                    corr_val = ep['corr_timeseries'][feat][t]
                    if not np.isfinite(corr_val):
                        continue
                    if is_calm:
                        calm_corrs[feat].append(corr_val)
                    elif is_danger:
                        danger_corrs[feat].append(corr_val)

                if np.isfinite(r2[t]):
                    if is_calm:
                        calm_r2.append(r2[t])
                    elif is_danger:
                        danger_r2.append(r2[t])

        risk_conditioned = {}
        for feat in feature_names:
            c = calm_corrs[feat]
            d = danger_corrs[feat]
            entry = {
                'calm_mean': float(np.mean(c)) if c else float('nan'),
                'calm_std': float(np.std(c)) if c else float('nan'),
                'danger_mean': float(np.mean(d)) if d else float('nan'),
                'danger_std': float(np.std(d)) if d else float('nan'),
                'shift': float(np.mean(d) - np.mean(c)) if (c and d) else float('nan'),
                'n_calm': len(c),
                'n_danger': len(d),
            }
            # Mann-Whitney U for significance
            if len(c) >= 5 and len(d) >= 5:
                stat, pval = stats.mannwhitneyu(c, d, alternative='two-sided')
                entry['pval'] = float(pval)
                entry['significant'] = pval < 0.05
            else:
                entry['pval'] = float('nan')
                entry['significant'] = False
            risk_conditioned[feat] = entry

        risk_conditioned['_r_squared'] = {
            'calm_mean': float(np.mean(calm_r2)) if calm_r2 else float('nan'),
            'danger_mean': float(np.mean(danger_r2)) if danger_r2 else float('nan'),
            'n_calm': len(calm_r2),
            'n_danger': len(danger_r2),
        }

        # =====================================================================
        # OUTPUT 3: Correlation-risk via Fisher z-transform
        # =====================================================================
        # For each feature, compute within-episode Spearman ρ between
        # the single-feature r(t) timeseries and risk R(t), then aggregate
        beta_risk_correlations = {}
        for feat in feature_names:
            episode_rhos = []
            episode_Ts = []
            episode_pvals = []

            for ep in episode_results:
                corrs = ep['corr_timeseries'].get(feat)
                if corrs is None:
                    continue
                risk = ep['risk_timeseries']

                # Filter to finite values
                valid = np.isfinite(corrs) & np.isfinite(risk)
                c_valid = corrs[valid]
                r_valid = risk[valid]

                if len(c_valid) < self.MIN_TIMESTEPS:
                    continue

                # Variation filter
                if np.std(r_valid) < self.HV_THRESHOLD:
                    continue
                if np.std(c_valid) < 1e-10:
                    continue

                rho, pval = stats.spearmanr(c_valid, r_valid)
                if np.isfinite(rho):
                    episode_rhos.append(float(rho))
                    episode_Ts.append(len(c_valid))
                    episode_pvals.append(float(pval))

            if len(episode_rhos) >= self.MIN_QUALIFYING_EPISODES:
                rhos = np.array(episode_rhos)
                Ts = np.array(episode_Ts)
                pvals = np.array(episode_pvals)

                # Fisher z-transform
                z_vals = np.arctanh(np.clip(rhos, -0.999, 0.999))
                z_mean = float(np.mean(z_vals))
                rho_agg = float(np.tanh(z_mean))

                # SE from 1/(T-3) variance formula
                valid_Ts = Ts[Ts > 3]
                if len(valid_Ts) > 0:
                    variances = 1.0 / (valid_Ts - 3)
                    se = float(np.sqrt(np.mean(variances)))
                else:
                    se = float('nan')

                ci_lower = float(np.tanh(z_mean - 1.96 * se))
                ci_upper = float(np.tanh(z_mean + 1.96 * se))

                beta_risk_correlations[feat] = {
                    'rho_aggregated': rho_agg,
                    'ci_95': (ci_lower, ci_upper),
                    'ci_excludes_zero': (ci_lower > 0) or (ci_upper < 0),
                    'n_episodes': len(episode_rhos),
                    'frac_significant': float(np.mean(np.array(episode_pvals) < 0.05)),
                    'per_episode_rhos': rhos.tolist(),
                }

        n_total = overall['n_total_timesteps']
        n_skipped = sum(ep['n_timesteps'] for ep in episode_results) - n_total

        self._result = DecompositionResult(
            overall=overall,
            risk_conditioned=risk_conditioned,
            beta_risk_correlations=beta_risk_correlations,
            feature_names=feature_names,
            n_episodes=len(episode_results),
            n_total_timesteps=n_total,
            n_skipped_timesteps=n_skipped,
        )

        return self._result

    # =========================================================================
    # Summary Table
    # =========================================================================

    def print_summary(self, result: Optional[DecompositionResult] = None):
        """Print formatted summary table of decomposition results."""
        r = result or self._result
        if r is None:
            print("No results to display. Run aggregate_results() first.")
            return

        print("\n" + "=" * 110)
        print("ATTENTION DECOMPOSITION RESULTS (Single-Feature Correlations)")
        print("=" * 110)
        print(f"Episodes: {r.n_episodes} | "
              f"Timesteps computed: {r.n_total_timesteps} | "
              f"Skipped: {r.n_skipped_timesteps}")
        print(f"2-Feature R² (dist+TTC) = {r.overall['mean_r_squared']:.3f} "
              f"± {r.overall['std_r_squared']:.3f}")
        print(f"2-Feature R²_adj        = {r.overall['mean_r_squared_adj']:.3f} "
              f"± {r.overall['std_r_squared_adj']:.3f}")

        # Risk-conditioned R²
        rc_r2 = r.risk_conditioned.get('_r_squared', {})
        if rc_r2:
            print(f"R² calm: {rc_r2.get('calm_mean', float('nan')):.3f} "
                  f"(n={rc_r2.get('n_calm', 0)}) | "
                  f"R² danger: {rc_r2.get('danger_mean', float('nan')):.3f} "
                  f"(n={rc_r2.get('n_danger', 0)})")

        print()
        header = (f"{'Feature':<18} | {'Mean r':>8} | {'Calm r':>8} | "
                  f"{'Danger r':>8} | {'Shift':>8} | {'p-val':>8} | "
                  f"{'r-Risk ρ̄':>10} | {'CI':>20}")
        print(header)
        print("-" * len(header))

        for feat in r.feature_names:
            label = FEATURE_LABELS.get(feat, feat)
            mean_r = r.overall['mean_corrs'].get(feat, float('nan'))
            rc = r.risk_conditioned.get(feat, {})
            calm_r = rc.get('calm_mean', float('nan'))
            danger_r = rc.get('danger_mean', float('nan'))
            shift = rc.get('shift', float('nan'))
            pval = rc.get('pval', float('nan'))
            sig = '***' if rc.get('significant', False) else ''

            br = r.beta_risk_correlations.get(feat, {})
            rho_agg = br.get('rho_aggregated', float('nan'))
            ci = br.get('ci_95', (float('nan'), float('nan')))
            ci_excl = '*' if br.get('ci_excludes_zero', False) else ''

            print(f"{label:<18} | {mean_r:>+8.3f} | {calm_r:>+8.3f} | "
                  f"{danger_r:>+8.3f} | {shift:>+8.3f} | {pval:>7.4f}{sig} | "
                  f"{rho_agg:>+10.3f} | [{ci[0]:>+.3f}, {ci[1]:>+.3f}]{ci_excl}")

        print("=" * 110)
        print("Mean r = average Pearson r(attention, feature) across timesteps")
        print("Shift = danger_r - calm_r | *** p < 0.05 (Mann-Whitney U)")
        print("r-Risk ρ̄ = Fisher z-aggregated Spearman ρ(r(t), Risk(t)) | * CI excludes zero")
        print()


# ==============================================================================
# Visualization
# ==============================================================================

class DecompositionVisualization:
    """Generate visualizations for attention decomposition analysis."""

    # Professional color palette
    COLORS = {
        'distance_to_ego': '#2196F3',   # Blue
        'ttc': '#F44336',               # Red
        'closing_speed': '#FF9800',      # Orange
        'agent_speeds': '#4CAF50',       # Green
    }
    CALM_COLOR = '#66BB6A'
    DANGER_COLOR = '#EF5350'

    def __init__(self, result: DecompositionResult):
        self.result = result
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 11,
            'figure.dpi': 150,
            'figure.facecolor': 'white',
        })

    # =========================================================================
    # Plot 1: Overall Decomposition Bar Chart
    # =========================================================================

    def plot_overall_decomposition(self, save_path: Optional[str] = None):
        """Bar chart of mean single-feature Pearson r with error bars."""
        r = self.result
        feats = r.feature_names
        labels = [FEATURE_LABELS.get(f, f) for f in feats]
        means = [r.overall['mean_corrs'].get(f, 0) for f in feats]
        stds = [r.overall['std_corrs'].get(f, 0) for f in feats]
        colors = [self.COLORS.get(f, '#9E9E9E') for f in feats]

        fig, ax = plt.subplots(figsize=(8, 5))

        bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors,
                      edgecolor='white', linewidth=1.5, alpha=0.85,
                      error_kw={'linewidth': 1.5})

        ax.axhline(y=0, color='#333333', linewidth=0.8, linestyle='-')
        ax.set_ylabel('Mean Pearson r (attention vs feature)')
        ax.set_title('Attention Decomposition: Single-Feature Correlations')
        ax.set_ylim(-1.05, 1.05)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add R² annotation
        r2_mean = r.overall['mean_r_squared']
        r2_adj = r.overall['mean_r_squared_adj']
        n = r.overall['n_total_timesteps']
        ax.text(0.98, 0.95,
                f'2-feat R² = {r2_mean:.3f}\n'
                f'2-feat R²_adj = {r2_adj:.3f}\n'
                f'n = {n}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5',
                          edgecolor='#cccccc', alpha=0.9))

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        return fig

    # =========================================================================
    # Plot 2: Risk-Conditioned Grouped Bar Chart
    # =========================================================================

    def plot_risk_conditioned(self, save_path: Optional[str] = None):
        """Grouped bar chart: calm vs danger correlations per feature."""
        r = self.result
        feats = r.feature_names
        labels = [FEATURE_LABELS.get(f, f) for f in feats]

        calm_means = [r.risk_conditioned.get(f, {}).get('calm_mean', 0) for f in feats]
        danger_means = [r.risk_conditioned.get(f, {}).get('danger_mean', 0) for f in feats]
        calm_stds = [r.risk_conditioned.get(f, {}).get('calm_std', 0) for f in feats]
        danger_stds = [r.risk_conditioned.get(f, {}).get('danger_std', 0) for f in feats]
        sigs = [r.risk_conditioned.get(f, {}).get('significant', False) for f in feats]

        x = np.arange(len(feats))
        width = 0.35

        fig, ax = plt.subplots(figsize=(9, 5))

        b1 = ax.bar(x - width / 2, calm_means, width, yerr=calm_stds, capsize=4,
                     label='Calm (R < 0.1)',
                     color=self.CALM_COLOR, alpha=0.8, edgecolor='white', linewidth=1.2)
        b2 = ax.bar(x + width / 2, danger_means, width, yerr=danger_stds, capsize=4,
                     label='Danger (R > 0.3)',
                     color=self.DANGER_COLOR, alpha=0.8, edgecolor='white', linewidth=1.2)

        ax.axhline(y=0, color='#333333', linewidth=0.8)

        # Mark significant shifts
        for i, sig in enumerate(sigs):
            if sig:
                y_max = max(abs(calm_means[i]), abs(danger_means[i]))
                y_pos = y_max + max(calm_stds[i], danger_stds[i]) + 0.03
                if calm_means[i] < 0 and danger_means[i] < 0:
                    y_pos = -y_pos
                ax.text(x[i], y_pos, '***', ha='center', va='bottom',
                        fontsize=14, fontweight='bold', color='#D32F2F')

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Mean Pearson r')
        ax.set_title('Risk-Conditioned Feature–Attention Correlations')
        ax.set_ylim(-1.05, 1.05)
        ax.legend(framealpha=0.9, edgecolor='#cccccc')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add sample sizes
        rc_r2 = r.risk_conditioned.get('_r_squared', {})
        n_calm = rc_r2.get('n_calm', 0)
        n_danger = rc_r2.get('n_danger', 0)
        ax.text(0.98, 0.05,
                f'n_calm={n_calm}, n_danger={n_danger}',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=9, color='#666666')

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        return fig

    # =========================================================================
    # Plot 3: Timeseries Diagnostic (4-panel)
    # =========================================================================

    def plot_timeseries_diagnostic(
        self,
        episode_results: List[Dict],
        scenario_idx: Optional[int] = None,
        save_path: Optional[str] = None,
    ):
        """4-panel timeseries for one episode with risk variation.

        Panels:
          1. Risk R over time (with calm/danger thresholds)
          2. 2-feature R² over time
          3. Single-feature correlations over time (one line per feature)
          4. Combined attention distribution per vehicle
        """
        # Pick scenario with most risk variation if not specified
        if scenario_idx is None:
            risk_stds = [np.nanstd(ep['risk_timeseries']) for ep in episode_results]
            scenario_idx = int(np.argmax(risk_stds))
            print(f"  Auto-selected scenario {scenario_idx} "
                  f"(std(R) = {risk_stds[scenario_idx]:.3f})")

        ep = episode_results[scenario_idx]
        T = ep['n_timesteps']
        t_axis = np.arange(T)

        fig = plt.figure(figsize=(14, 12))
        gs = gridspec.GridSpec(4, 1, hspace=0.35, height_ratios=[1, 1, 1.5, 1.5])

        # --- Panel 1: Risk R ---
        ax1 = fig.add_subplot(gs[0])
        risk = ep['risk_timeseries']
        ax1.plot(t_axis, risk, color='#E53935', linewidth=1.5, label='Risk R')
        ax1.axhline(0.1, color=self.CALM_COLOR, linestyle='--', alpha=0.7, label='Calm threshold')
        ax1.axhline(0.3, color=self.DANGER_COLOR, linestyle='--', alpha=0.7, label='Danger threshold')
        ax1.fill_between(t_axis, 0, risk, where=risk < 0.1,
                         color=self.CALM_COLOR, alpha=0.15)
        ax1.fill_between(t_axis, 0, risk, where=risk > 0.3,
                         color=self.DANGER_COLOR, alpha=0.15)
        ax1.set_ylabel('Collision Risk R')
        ax1.set_title(f'Timeseries Diagnostic — Episode {ep["episode_idx"]}')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.set_xlim(0, T - 1)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # --- Panel 2: 2-Feature R² ---
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        r2 = ep['r2_timeseries']
        r2a = ep['r2_adj_timeseries']
        ax2.plot(t_axis, r2, color='#1565C0', linewidth=1.5, label='R² (dist+TTC)')
        ax2.plot(t_axis, r2a, color='#1565C0', linewidth=1.0, linestyle='--',
                 alpha=0.7, label='R²_adj')
        ax2.axhline(0.5, color='#888888', linestyle=':', alpha=0.5)
        ax2.set_ylabel('R²')
        ax2.set_ylim(-0.1, 1.05)
        ax2.legend(loc='upper right', fontsize=8)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # --- Panel 3: Single-feature correlations ---
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        for feat in ep['feature_names']:
            corrs = ep['corr_timeseries'][feat]
            color = self.COLORS.get(feat, '#9E9E9E')
            label = FEATURE_LABELS.get(feat, feat)
            ax3.plot(t_axis, corrs, color=color, linewidth=1.5, label=label, alpha=0.85)
        ax3.axhline(0, color='#333333', linewidth=0.5)
        ax3.set_ylabel('Pearson r')
        ax3.set_ylim(-1.05, 1.05)
        ax3.legend(loc='upper right', fontsize=8, ncol=2)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        # --- Panel 4: Combined attention per vehicle ---
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        combined_attn = ep['combined_attention']  # (T, n_vehicles)
        n_veh = combined_attn.shape[1]
        cmap = plt.cm.tab10
        for v in range(n_veh):
            ax4.plot(t_axis, combined_attn[:, v], color=cmap(v % 10),
                     linewidth=1.0, alpha=0.7, label=f'Vehicle {v}')
        ax4.set_ylabel('Attention Fraction')
        ax4.set_xlabel('Timestep')
        ax4.legend(loc='upper right', fontsize=7, ncol=4)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        return fig

    # =========================================================================
    # Plot 4: Correlation-Risk Summary
    # =========================================================================

    def plot_beta_risk_correlation(self, save_path: Optional[str] = None):
        """Horizontal bar chart of Fisher z-aggregated ρ between r(t) and R(t)."""
        r = self.result
        brc = r.beta_risk_correlations

        if not brc:
            print("  No correlation-risk results to plot (not enough qualifying episodes).")
            return None

        feats = [f for f in r.feature_names if f in brc]
        if not feats:
            print("  No features with correlation-risk results.")
            return None

        labels = [FEATURE_LABELS.get(f, f) for f in feats]
        rhos = [brc[f]['rho_aggregated'] for f in feats]
        ci_lows = [brc[f]['ci_95'][0] for f in feats]
        ci_highs = [brc[f]['ci_95'][1] for f in feats]
        ci_excl = [brc[f]['ci_excludes_zero'] for f in feats]

        # Error bars (asymmetric)
        errors_low = [rho - cl for rho, cl in zip(rhos, ci_lows)]
        errors_high = [ch - rho for rho, ch in zip(rhos, ci_highs)]

        fig, ax = plt.subplots(figsize=(8, 4))

        y_pos = np.arange(len(feats))
        colors = ['#E53935' if rho > 0 else '#1E88E5' for rho in rhos]

        ax.barh(y_pos, rhos, xerr=[errors_low, errors_high], capsize=4,
                color=colors, alpha=0.8, edgecolor='white', linewidth=1.2,
                error_kw={'linewidth': 1.5})

        ax.axvline(0, color='#333333', linewidth=0.8)

        # Mark significant
        for i, excl in enumerate(ci_excl):
            if excl:
                ax.text(rhos[i] + (0.02 if rhos[i] > 0 else -0.04), i,
                        '*', fontsize=16, fontweight='bold',
                        color='#D32F2F', va='center')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Fisher z-aggregated ρ(r, Risk)')
        ax.set_title('Does Feature–Attention Correlation Change with Risk?')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add n_episodes info
        ns = [brc[f]['n_episodes'] for f in feats]
        ax.text(0.98, 0.05, f'n_episodes: {min(ns)}-{max(ns)}',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=9, color='#666666')

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"  Saved: {save_path}")
        return fig


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Attention Decomposition Analysis — '
                    'Single-Feature Correlations of Combined Attention'
    )
    parser.add_argument('--extraction-dir', type=str, required=True,
                        help='Directory containing extraction pickle files')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save figures (default: <extraction-dir>/../figures_decomposition)')
    parser.add_argument('--extraction-file', type=str, default=None,
                        help='Specific extraction pickle file to use')
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(args.extraction_dir), 'figures_decomposition')
    os.makedirs(output_dir, exist_ok=True)

    # --- Run analysis ---
    print("\n" + "=" * 60)
    print("ATTENTION DECOMPOSITION ANALYSIS")
    print("=" * 60)

    analyzer = AttentionDecompositionAnalyzer(args.extraction_dir)
    analyzer.load_data(args.extraction_file)

    print("\n[1/3] Running per-timestep decomposition...")
    episode_results = analyzer.run_decomposition()

    print("\n[2/3] Aggregating results...")
    result = analyzer.aggregate_results(episode_results)

    print("\n[3/3] Generating visualizations...")
    viz = DecompositionVisualization(result)

    viz.plot_overall_decomposition(
        save_path=os.path.join(output_dir, 'decomposition_overall.png')
    )
    viz.plot_risk_conditioned(
        save_path=os.path.join(output_dir, 'decomposition_risk_conditioned.png')
    )
    viz.plot_timeseries_diagnostic(
        episode_results,
        save_path=os.path.join(output_dir, 'decomposition_timeseries.png')
    )
    viz.plot_beta_risk_correlation(
        save_path=os.path.join(output_dir, 'decomposition_beta_risk.png')
    )

    # --- Print summary ---
    analyzer.print_summary(result)

    # Save results pickle for further analysis
    results_path = os.path.join(output_dir, 'decomposition_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump({
            'result': result,
            'episode_results': episode_results,
        }, f)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
