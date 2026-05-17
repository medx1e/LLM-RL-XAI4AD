"""Head Specialization Analysis for XAI — Within-Module Distributional Analysis.

This module analyzes attention head behavior through within-module distributional
analysis, not module-level sums (which are trivially ~1.0 due to softmax).

Key Methodology:
- Within-module cross-entity Spearman correlation at each sim timestep
  (e.g., across 8 vehicles: correlate attention fraction with TTC)
- Two complementary analyses:
  Option A: Fisher z-aggregate of per-timestep ρ → "consistent selectivity"
  Option B: Correlate ρ timeseries with risk R → "risk-dependent selectivity"
- Risk-conditioned profiling: compare attention concentration in calm vs danger

Usage:
    analyzer = HeadSpecializationAnalyzer(extraction_dir="./extractions")
    analyzer.load_data()
    results = analyzer.compute_hsi()
    analyzer.export_registry("head_functions.json")
    analyzer.visualize_all("./hsi_plots")
"""

import argparse
import glob
import json
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

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class HSIResult:
    """Results from Head Specialization Index computation."""
    hsi_scores: np.ndarray                        # Shape: (n_heads,)
    aggregated_correlations: Dict[str, Dict]       # feature_key -> agg stats per head
    primary_features: List[str]                     # Primary feature per head
    primary_correlations: np.ndarray               # ρ̄ value for primary feature
    head_labels: Dict[int, Dict[str, Any]]          # Head index -> label info
    risk_profiles: Dict[int, Dict[str, Any]]       # Head index -> calm/danger profiles
    risk_dependent: Dict[str, Dict]                # feature_key -> risk-dependent analysis
    checkpoint_step: int                           # Training step
    n_qualifying_episodes: Dict[str, int]          # feature_key -> n qualifying episodes


# ==============================================================================
# Feature Labeling Configuration
# ==============================================================================

FEATURE_TO_LABEL = {
    # Type 1: Cross-vehicle correlations within agent module
    'vehicle_attn_vs_ttc': {
        'name': 'Threat-Distance Head',
        'description': 'Within agent module, prioritizes low-TTC vehicles',
        'expected_sign': 'negative',
    },
    'vehicle_attn_vs_distance': {
        'name': 'Proximity Head',
        'description': 'Within agent module, prioritizes nearby vehicles',
        'expected_sign': 'negative',
    },
    'vehicle_attn_vs_closing_speed': {
        'name': 'Approach-Rate Head',
        'description': 'Within agent module, attends to rapidly approaching vehicles',
        'expected_sign': 'positive',
    },
    'vehicle_attn_vs_speed': {
        'name': 'Dynamic Object Head',
        'description': 'Within agent module, attends to fast-moving agents',
        'expected_sign': 'variable',
    },
    # Type 2: Concentration metrics vs. risk
    'roadgraph_concentration_vs_risk': {
        'name': 'Avoidance Planning Head',
        'description': 'Focuses roadgraph attention when risk rises',
        'expected_sign': 'positive',
    },
    'roadgraph_entropy_vs_risk': {
        'name': 'Road Scanning Head',
        'description': 'Broadens road attention under threat (lower concentration)',
        'expected_sign': 'negative',
    },
    'tl_concentration_vs_risk': {
        'name': 'Signal Focus Head',
        'description': 'Focuses traffic light attention when risk rises',
        'expected_sign': 'positive',
    },
    'gps_center_of_mass_vs_risk': {
        'name': 'Navigation Horizon Head',
        'description': 'Shifts GPS attention to nearer waypoints under threat',
        'expected_sign': 'negative',
    },
    'sdc_recency_vs_risk': {
        'name': 'Ego Recency Head',
        'description': 'Shifts to more recent ego observations under threat',
        'expected_sign': 'positive',
    },
    # Type 3: Risk-dependent selectivity (Option B correlations)
    'vehicle_ttc_selectivity_vs_risk': {
        'name': 'Risk-Triggered Safety Head',
        'description': 'Becomes MORE TTC-selective when risk rises',
        'expected_sign': 'negative',
    },
    'vehicle_distance_selectivity_vs_risk': {
        'name': 'Risk-Triggered Proximity Head',
        'description': 'Becomes MORE proximity-selective when risk rises',
        'expected_sign': 'negative',
    },
}

DEFAULT_LABEL = {
    'name': 'General Context Head',
    'description': 'Diffuse attention without strong feature specialization',
}


# ==============================================================================
# Head Specialization Analyzer
# ==============================================================================

class HeadSpecializationAnalyzer:
    """Analyze attention head specialization using within-module distributional analysis.
    
    Two complementary analysis modes:
    - Option A: Consistent selectivity — average cross-vehicle ρ across timesteps
    - Option B: Risk-dependent selectivity — correlate selectivity with risk R
    """
    
    # Minimum HSI threshold for "specialized" heads
    HSI_THRESHOLD = 0.3
    
    # Risk thresholds for calm/danger phase classification
    CALM_THRESHOLD = 0.1
    DANGER_THRESHOLD = 0.4
    
    # Minimum std(R) for high-variation episodes
    HV_THRESHOLD = 0.2
    
    # Minimum timesteps per episode for meaningful correlation
    MIN_TIMESTEPS = 10
    
    # Minimum qualifying episodes for meaningful aggregation
    MIN_QUALIFYING_EPISODES = 3
    
    # Minimum number of valid vehicles per timestep for cross-vehicle correlation
    MIN_VALID_VEHICLES = 3
    
    def __init__(self, extraction_dir: str):
        self.extraction_dir = extraction_dir
        self._extraction_data = None
        self._result: Optional[HSIResult] = None
        
    def load_data(self, extraction_file: Optional[str] = None):
        """Load extraction data from pickle file."""
        if extraction_file is None:
            pattern = os.path.join(self.extraction_dir, "extraction_*.pkl")
            files = sorted(glob.glob(pattern))
            if not files:
                raise FileNotFoundError(f"No extraction files found in {self.extraction_dir}")
            
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
            print(f"[HSI] Auto-selected: {os.path.basename(extraction_file)}")
        else:
            if not os.path.isabs(extraction_file):
                extraction_file = os.path.join(self.extraction_dir, extraction_file)
        
        with open(extraction_file, 'rb') as f:
            self._extraction_data = pickle.load(f)
        
        mode = self._extraction_data.get('extraction_mode', 'unknown')
        print(f"[HSI] Loaded {self._extraction_data['n_scenarios']} scenarios ({mode} mode)")
        print(f"[HSI] Checkpoint: {self._extraction_data['checkpoint']} "
              f"(step {self._extraction_data['step']})")
        
        if mode != 'rollout':
            print("[HSI] WARNING: Data was NOT extracted with --rollout mode.")
            print("[HSI] Within-module analysis requires multi-timestep data.")
        
        return self
    
    # =========================================================================
    # Episode data loading
    # =========================================================================
    
    def _load_episode(self, scenario_data: Dict) -> Tuple[Dict, np.ndarray, Dict]:
        """Convert a scenario's timestep list into structured arrays.
        
        Returns:
            attn_dists: dict of distributional data per timestep
                - 'per_vehicle_attention': (T, n_vehicles, n_heads)
                - 'roadgraph_concentration': (T, n_heads)
                - 'roadgraph_entropy': (T, n_heads)
                - 'per_waypoint_attention': (T, n_waypoints, n_heads) if present
            risk_timeseries: (T,)
            semantic_timeseries: dict of arrays, e.g. 'ttc': (T, n_vehicles)
        """
        timesteps = scenario_data['timesteps']
        T = len(timesteps)
        
        # Determine what keys are available
        first_dist = timesteps[0]['attention_distributions']
        
        attn_dists = {}
        
        # Stack per-vehicle attention: (T, n_vehicles, n_heads)
        if 'per_vehicle_attention' in first_dist:
            attn_dists['per_vehicle_attention'] = np.stack(
                [ts['attention_distributions']['per_vehicle_attention'] for ts in timesteps],
                axis=0
            )
        
        # Stack roadgraph concentration: (T, n_heads)
        if 'roadgraph_concentration' in first_dist:
            attn_dists['roadgraph_concentration'] = np.stack(
                [ts['attention_distributions']['roadgraph_concentration'] for ts in timesteps],
                axis=0
            )
        if 'roadgraph_entropy' in first_dist:
            attn_dists['roadgraph_entropy'] = np.stack(
                [ts['attention_distributions']['roadgraph_entropy'] for ts in timesteps],
                axis=0
            )
        if 'roadgraph_max_fraction' in first_dist:
            attn_dists['roadgraph_max_fraction'] = np.stack(
                [ts['attention_distributions']['roadgraph_max_fraction'] for ts in timesteps],
                axis=0
            )
        
        # Stack per-waypoint: (T, n_waypoints, n_heads)
        if 'per_waypoint_attention' in first_dist:
            attn_dists['per_waypoint_attention'] = np.stack(
                [ts['attention_distributions']['per_waypoint_attention'] for ts in timesteps],
                axis=0
            )
        
        # Stack per-light: (T, n_lights, n_heads)
        if 'per_light_attention' in first_dist:
            attn_dists['per_light_attention'] = np.stack(
                [ts['attention_distributions']['per_light_attention'] for ts in timesteps],
                axis=0
            )
        
        # Per-SDC timestep: (T, n_obs_ts, n_heads)
        if 'per_sdc_timestep_attention' in first_dist:
            attn_dists['per_sdc_timestep_attention'] = np.stack(
                [ts['attention_distributions']['per_sdc_timestep_attention'] for ts in timesteps],
                axis=0
            )
        
        # Risk timeseries: (T,)
        risk_timeseries = np.array([ts['collision_risk'] for ts in timesteps])
        
        # Semantic features: stack per-vehicle features
        skip_fields = {'sdc_index', 'timestep', 'valid', 'positions_x', 'positions_y',
                       'ego_speed', 'object_types'}
        semantic_timeseries = {}
        feature_names = [k for k in timesteps[0]['semantic_features'].keys()
                         if k not in skip_fields]
        for feat in feature_names:
            try:
                semantic_timeseries[feat] = np.stack(
                    [ts['semantic_features'][feat] for ts in timesteps], axis=0
                )
            except (ValueError, KeyError):
                continue
        
        return attn_dists, risk_timeseries, semantic_timeseries
    
    # =========================================================================
    # Option A: Consistent Selectivity — cross-vehicle ρ aggregated over time
    # =========================================================================
    
    def _compute_cross_vehicle_rho_timeseries(
        self, per_vehicle_attn: np.ndarray, feature_series: np.ndarray,
        valid_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """At each timestep, compute Spearman ρ across vehicles for each head.
        
        Args:
            per_vehicle_attn: (T, n_vehicles, n_heads)
            feature_series: (T, n_vehicles) — e.g., TTC or distance
            valid_mask: (T, n_vehicles) bool — which vehicles are valid
            
        Returns:
            rho_timeseries: (T, n_heads) — cross-vehicle ρ at each timestep
            valid_timesteps: (T,) bool — which timesteps had enough valid vehicles
        """
        T, n_vehicles, n_heads = per_vehicle_attn.shape
        rho_timeseries = np.full((T, n_heads), np.nan)
        valid_timesteps = np.zeros(T, dtype=bool)
        
        for t in range(T):
            feat = feature_series[t]
            
            # Build mask
            if valid_mask is not None:
                mask = valid_mask[t].astype(bool)
            else:
                mask = np.ones(n_vehicles, dtype=bool)
            
            # Also exclude NaN/Inf features
            mask &= np.isfinite(feat)
            
            if mask.sum() < self.MIN_VALID_VEHICLES:
                continue
            
            feat_valid = feat[mask]
            
            # Skip if no variation in feature
            if np.std(feat_valid) < 1e-10:
                continue
            
            valid_timesteps[t] = True
            
            for h in range(n_heads):
                attn_valid = per_vehicle_attn[t, mask, h]
                if np.std(attn_valid) < 1e-10:
                    rho_timeseries[t, h] = 0.0
                    continue
                rho, _ = stats.spearmanr(attn_valid, feat_valid)
                rho_timeseries[t, h] = rho if np.isfinite(rho) else 0.0
        
        return rho_timeseries, valid_timesteps
    
    def compute_option_a(self) -> Dict[str, Dict[int, Dict]]:
        """Option A: Consistent selectivity.
        
        For each (head, vehicle_feature) pair, compute cross-vehicle Spearman ρ
        at each timestep, then Fisher z-aggregate across timesteps within each
        episode, then Fisher z-aggregate across episodes.
        
        Returns:
            {feature_key: {head_idx: aggregation_stats}}
        """
        if self._extraction_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        episodes = self._extraction_data['scenarios']
        
        # Determine n_heads from first episode
        first_ep = episodes[0]
        first_dist = first_ep['timesteps'][0]['attention_distributions']
        if 'per_vehicle_attention' not in first_dist:
            print("[HSI] No per_vehicle_attention in data. Skipping Option A.")
            return {}
        n_heads = first_dist['per_vehicle_attention'].shape[1]
        
        # Vehicle features to correlate with attention
        vehicle_features = {
            'vehicle_attn_vs_ttc': 'ttc',
            'vehicle_attn_vs_distance': 'distance_to_ego',
            'vehicle_attn_vs_closing_speed': 'closing_speed',
            'vehicle_attn_vs_speed': 'agent_speeds',
        }
        
        results = {fk: {h: [] for h in range(n_heads)} for fk in vehicle_features}
        
        skipped_short = 0
        
        for ep_idx, episode in enumerate(episodes):
            attn_dists, risk, semantics = self._load_episode(episode)
            T = len(risk)
            
            if T < self.MIN_TIMESTEPS:
                skipped_short += 1
                continue
            
            per_veh_attn = attn_dists.get('per_vehicle_attention')
            if per_veh_attn is None:
                continue
            
            # Get valid mask from semantic features
            valid_series = semantics.get('valid')  # might not exist
            
            for feature_key, sem_name in vehicle_features.items():
                feat_series = semantics.get(sem_name)
                if feat_series is None or feat_series.ndim != 2:
                    continue
                
                # Compute cross-vehicle ρ at each timestep
                rho_ts, valid_ts = self._compute_cross_vehicle_rho_timeseries(
                    per_veh_attn, feat_series, valid_series
                )
                
                n_valid = valid_ts.sum()
                if n_valid < self.MIN_TIMESTEPS:
                    continue
                
                # Fisher z-aggregate over valid timesteps within this episode
                for h in range(n_heads):
                    rhos_valid = rho_ts[valid_ts, h]
                    rhos_valid = rhos_valid[np.isfinite(rhos_valid)]
                    
                    if len(rhos_valid) < self.MIN_TIMESTEPS:
                        continue
                    
                    z_vals = np.arctanh(np.clip(rhos_valid, -0.999, 0.999))
                    z_mean = np.mean(z_vals)
                    rho_episode = float(np.tanh(z_mean))
                    
                    results[feature_key][h].append({
                        'rho': rho_episode,
                        'n_timesteps': int(n_valid),
                        'episode_idx': ep_idx,
                    })
        
        print(f"[HSI] Option A: Skipped {skipped_short} short episodes")
        
        # Cross-episode Fisher z-aggregation
        aggregated = {}
        for feature_key in vehicle_features:
            aggregated[feature_key] = {}
            for h in range(n_heads):
                ep_results = results[feature_key][h]
                n = len(ep_results)
                if n < self.MIN_QUALIFYING_EPISODES:
                    continue
                
                rhos = np.array([r['rho'] for r in ep_results])
                z_vals = np.arctanh(np.clip(rhos, -0.999, 0.999))
                z_mean = np.mean(z_vals)
                rho_agg = float(np.tanh(z_mean))
                
                # CI from z-variance
                if n > 1:
                    z_se = float(np.std(z_vals, ddof=1) / np.sqrt(n))
                    ci_lower = float(np.tanh(z_mean - 1.96 * z_se))
                    ci_upper = float(np.tanh(z_mean + 1.96 * z_se))
                else:
                    z_se = float('nan')
                    ci_lower = float('nan')
                    ci_upper = float('nan')
                
                aggregated[feature_key][h] = {
                    'rho_aggregated': rho_agg,
                    'ci_95': (ci_lower, ci_upper),
                    'se': z_se,
                    'n_episodes': n,
                    'per_episode_rhos': rhos.tolist(),
                }
                
                print(f"  [A] Head {h} | {feature_key:35s} | ρ̄={rho_agg:+.3f} "
                      f"CI=[{ci_lower:+.3f}, {ci_upper:+.3f}] n={n}")
        
        return aggregated
    
    # =========================================================================
    # Option B: Risk-Dependent Selectivity — ρ timeseries vs risk
    # =========================================================================
    
    def compute_option_b(self) -> Dict[str, Dict[int, Dict]]:
        """Option B: Risk-dependent selectivity.
        
        For each (head, vehicle_feature) pair, compute cross-vehicle ρ at each
        timestep, then correlate that ρ timeseries with risk R within each episode.
        Finally aggregate across episodes with Fisher z.
        
        This answers: "Does this head become MORE selective when risk rises?"
        """
        if self._extraction_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        episodes = self._extraction_data['scenarios']
        
        first_dist = episodes[0]['timesteps'][0]['attention_distributions']
        if 'per_vehicle_attention' not in first_dist:
            return {}
        n_heads = first_dist['per_vehicle_attention'].shape[1]
        
        vehicle_features = {
            'vehicle_ttc_selectivity_vs_risk': 'ttc',
            'vehicle_distance_selectivity_vs_risk': 'distance_to_ego',
        }
        
        # Scalar concentration/distribution metrics vs risk
        concentration_features = {}
        if 'roadgraph_concentration' in first_dist:
            concentration_features['roadgraph_concentration_vs_risk'] = 'roadgraph_concentration'
        if 'roadgraph_entropy' in first_dist:
            concentration_features['roadgraph_entropy_vs_risk'] = 'roadgraph_entropy'
        
        # Derived scalar features computed per-episode (GPS center-of-mass, TL concentration, SDC recency)
        derived_scalar_features = []
        if 'per_light_attention' in first_dist:
            derived_scalar_features.append('tl_concentration_vs_risk')
        if 'per_waypoint_attention' in first_dist:
            derived_scalar_features.append('gps_center_of_mass_vs_risk')
        if 'per_sdc_timestep_attention' in first_dist:
            derived_scalar_features.append('sdc_recency_vs_risk')
        
        all_features = {**vehicle_features, **concentration_features}
        for dsf in derived_scalar_features:
            all_features[dsf] = dsf  # placeholder, handled in loop below
        results = {fk: {h: [] for h in range(n_heads)} for fk in all_features}
        
        skipped_low_var = 0
        
        for ep_idx, episode in enumerate(episodes):
            attn_dists, risk, semantics = self._load_episode(episode)
            T = len(risk)
            
            if T < self.MIN_TIMESTEPS:
                continue
            
            # Variation filter on risk
            if np.std(risk) < self.HV_THRESHOLD:
                skipped_low_var += 1
                continue
            
            per_veh_attn = attn_dists.get('per_vehicle_attention')
            valid_series = semantics.get('valid')
            
            # === Vehicle selectivity vs risk ===
            if per_veh_attn is not None:
                for feature_key, sem_name in vehicle_features.items():
                    feat_series = semantics.get(sem_name)
                    if feat_series is None or feat_series.ndim != 2:
                        continue
                    
                    rho_ts, valid_ts = self._compute_cross_vehicle_rho_timeseries(
                        per_veh_attn, feat_series, valid_series
                    )
                    
                    for h in range(n_heads):
                        rho_series = rho_ts[:, h]
                        # Use only timesteps where both rho and risk are valid
                        both_valid = valid_ts & np.isfinite(rho_series)
                        
                        if both_valid.sum() < self.MIN_TIMESTEPS:
                            continue
                        
                        rho_valid = rho_series[both_valid]
                        risk_valid = risk[both_valid]
                        
                        if np.std(rho_valid) < 1e-10:
                            continue
                        
                        corr, pval = stats.spearmanr(rho_valid, risk_valid)
                        if np.isfinite(corr):
                            results[feature_key][h].append({
                                'rho': float(corr),
                                'pval': float(pval),
                                'T': int(both_valid.sum()),
                                'episode_idx': ep_idx,
                            })
            
            # === Concentration features vs risk (roadgraph) ===
            for feature_key, attn_key in concentration_features.items():
                conc_series = attn_dists.get(attn_key)
                if conc_series is None:
                    continue
                
                for h in range(n_heads):
                    series_h = conc_series[:, h]
                    if np.std(series_h) < 1e-10:
                        continue
                    
                    corr, pval = stats.spearmanr(series_h, risk)
                    if np.isfinite(corr):
                        results[feature_key][h].append({
                            'rho': float(corr),
                            'pval': float(pval),
                            'T': T,
                            'episode_idx': ep_idx,
                        })
            
            # === Traffic light concentration vs risk ===
            per_light = attn_dists.get('per_light_attention')  # (T, n_lights, n_heads)
            if per_light is not None and 'tl_concentration_vs_risk' in results:
                for h in range(n_heads):
                    # Concentration = max fraction across lights per timestep
                    tl_max_frac = per_light[:, :, h].max(axis=1)  # (T,)
                    if np.std(tl_max_frac) < 1e-10:
                        continue
                    corr, pval = stats.spearmanr(tl_max_frac, risk)
                    if np.isfinite(corr):
                        results['tl_concentration_vs_risk'][h].append({
                            'rho': float(corr), 'pval': float(pval),
                            'T': T, 'episode_idx': ep_idx,
                        })
            
            # === GPS waypoint center-of-mass vs risk ===
            per_wp = attn_dists.get('per_waypoint_attention')  # (T, n_waypoints, n_heads)
            if per_wp is not None and 'gps_center_of_mass_vs_risk' in results:
                n_wp = per_wp.shape[1]
                wp_indices = np.arange(n_wp, dtype=float)  # 0=nearest, n-1=farthest
                for h in range(n_heads):
                    # Center of mass: weighted average waypoint index
                    com = (per_wp[:, :, h] * wp_indices[None, :]).sum(axis=1)  # (T,)
                    if np.std(com) < 1e-10:
                        continue
                    corr, pval = stats.spearmanr(com, risk)
                    if np.isfinite(corr):
                        results['gps_center_of_mass_vs_risk'][h].append({
                            'rho': float(corr), 'pval': float(pval),
                            'T': T, 'episode_idx': ep_idx,
                        })
            
            # === SDC recency bias vs risk ===
            per_sdc = attn_dists.get('per_sdc_timestep_attention')  # (T, n_obs_ts, n_heads)
            if per_sdc is not None and 'sdc_recency_vs_risk' in results:
                n_obs = per_sdc.shape[1]
                obs_indices = np.arange(n_obs, dtype=float)  # 0=oldest, n-1=most recent
                for h in range(n_heads):
                    # Recency = center of mass over obs timesteps (higher = more recent)
                    recency = (per_sdc[:, :, h] * obs_indices[None, :]).sum(axis=1)  # (T,)
                    if np.std(recency) < 1e-10:
                        continue
                    corr, pval = stats.spearmanr(recency, risk)
                    if np.isfinite(corr):
                        results['sdc_recency_vs_risk'][h].append({
                            'rho': float(corr), 'pval': float(pval),
                            'T': T, 'episode_idx': ep_idx,
                        })
        
        print(f"[HSI] Option B: Skipped {skipped_low_var} low-var episodes (std(R) < {self.HV_THRESHOLD})")
        
        # Cross-episode Fisher z-aggregation
        aggregated = {}
        for feature_key in all_features:
            aggregated[feature_key] = {}
            for h in range(n_heads):
                ep_results = results[feature_key][h]
                n = len(ep_results)
                if n < self.MIN_QUALIFYING_EPISODES:
                    continue
                
                rhos = np.array([r['rho'] for r in ep_results])
                z_vals = np.arctanh(np.clip(rhos, -0.999, 0.999))
                z_mean = np.mean(z_vals)
                rho_agg = float(np.tanh(z_mean))
                
                if n > 1:
                    z_se = float(np.std(z_vals, ddof=1) / np.sqrt(n))
                    ci_lower = float(np.tanh(z_mean - 1.96 * z_se))
                    ci_upper = float(np.tanh(z_mean + 1.96 * z_se))
                else:
                    z_se = ci_lower = ci_upper = float('nan')
                
                frac_sig = float(np.mean([r['pval'] < 0.05 for r in ep_results]))
                
                aggregated[feature_key][h] = {
                    'rho_aggregated': rho_agg,
                    'ci_95': (ci_lower, ci_upper),
                    'se': z_se,
                    'n_episodes': n,
                    'frac_significant': frac_sig,
                    'per_episode_rhos': rhos.tolist(),
                }
                
                print(f"  [B] Head {h} | {feature_key:35s} | ρ̄={rho_agg:+.3f} "
                      f"CI=[{ci_lower:+.3f}, {ci_upper:+.3f}] n={n}")
        
        return aggregated
    
    # =========================================================================
    # Risk-conditioned profiling
    # =========================================================================
    
    def compute_risk_profiles(self) -> Dict[int, Dict[str, Any]]:
        """Compare attention concentration in calm vs danger phases per head.
        
        Covers ALL modules:
        - Agents: max vehicle fraction, entropy
        - Roadgraph: concentration
        - Traffic lights: max light fraction
        - GPS: waypoint center-of-mass
        - SDC: recency (obs-timestep center-of-mass)
        """
        episodes = self._extraction_data['scenarios']
        first_dist = episodes[0]['timesteps'][0]['attention_distributions']
        
        has_vehicles = 'per_vehicle_attention' in first_dist
        has_roadgraph = 'roadgraph_concentration' in first_dist
        has_tl = 'per_light_attention' in first_dist
        has_gps = 'per_waypoint_attention' in first_dist
        has_sdc = 'per_sdc_timestep_attention' in first_dist
        
        # Determine n_heads from whatever is available
        for key in ['per_vehicle_attention', 'per_light_attention', 'per_waypoint_attention',
                     'per_sdc_timestep_attention']:
            if key in first_dist:
                n_heads = first_dist[key].shape[-1]
                break
        else:
            if 'roadgraph_concentration' in first_dist:
                n_heads = first_dist['roadgraph_concentration'].shape[0]
            else:
                print("[HSI] No distributional data for risk profiling.")
                return {}
        
        # Accumulators: per head, calm vs danger
        calm_max_frac = {h: [] for h in range(n_heads)}
        danger_max_frac = {h: [] for h in range(n_heads)}
        calm_veh_entropy = {h: [] for h in range(n_heads)}
        danger_veh_entropy = {h: [] for h in range(n_heads)}
        calm_rg_conc = {h: [] for h in range(n_heads)}
        danger_rg_conc = {h: [] for h in range(n_heads)}
        calm_tl_max = {h: [] for h in range(n_heads)}
        danger_tl_max = {h: [] for h in range(n_heads)}
        calm_gps_com = {h: [] for h in range(n_heads)}
        danger_gps_com = {h: [] for h in range(n_heads)}
        calm_sdc_rec = {h: [] for h in range(n_heads)}
        danger_sdc_rec = {h: [] for h in range(n_heads)}
        
        for episode in episodes:
            attn_dists, risk, _ = self._load_episode(episode)
            
            calm_mask = risk < self.CALM_THRESHOLD
            danger_mask = risk > self.DANGER_THRESHOLD
            
            # Vehicle module
            if has_vehicles and 'per_vehicle_attention' in attn_dists:
                pva = attn_dists['per_vehicle_attention']  # (T, n_vehicles, n_heads)
                for h in range(n_heads):
                    veh_fracs = pva[:, :, h]
                    max_fracs = veh_fracs.max(axis=1)
                    log_fracs = np.where(veh_fracs > 1e-10, np.log(veh_fracs), 0.0)
                    entropies = -(veh_fracs * log_fracs).sum(axis=1)
                    if calm_mask.sum() > 0:
                        calm_max_frac[h].extend(max_fracs[calm_mask].tolist())
                        calm_veh_entropy[h].extend(entropies[calm_mask].tolist())
                    if danger_mask.sum() > 0:
                        danger_max_frac[h].extend(max_fracs[danger_mask].tolist())
                        danger_veh_entropy[h].extend(entropies[danger_mask].tolist())
            
            # Roadgraph
            if has_roadgraph and 'roadgraph_concentration' in attn_dists:
                rg_conc = attn_dists['roadgraph_concentration']
                for h in range(n_heads):
                    if calm_mask.sum() > 0:
                        calm_rg_conc[h].extend(rg_conc[calm_mask, h].tolist())
                    if danger_mask.sum() > 0:
                        danger_rg_conc[h].extend(rg_conc[danger_mask, h].tolist())
            
            # Traffic lights
            if has_tl and 'per_light_attention' in attn_dists:
                pla = attn_dists['per_light_attention']  # (T, n_lights, n_heads)
                for h in range(n_heads):
                    tl_max = pla[:, :, h].max(axis=1)  # max light fraction per timestep
                    if calm_mask.sum() > 0:
                        calm_tl_max[h].extend(tl_max[calm_mask].tolist())
                    if danger_mask.sum() > 0:
                        danger_tl_max[h].extend(tl_max[danger_mask].tolist())
            
            # GPS waypoints
            if has_gps and 'per_waypoint_attention' in attn_dists:
                pwp = attn_dists['per_waypoint_attention']  # (T, n_wp, n_heads)
                n_wp = pwp.shape[1]
                wp_idx = np.arange(n_wp, dtype=float)
                for h in range(n_heads):
                    com = (pwp[:, :, h] * wp_idx[None, :]).sum(axis=1)  # center of mass
                    if calm_mask.sum() > 0:
                        calm_gps_com[h].extend(com[calm_mask].tolist())
                    if danger_mask.sum() > 0:
                        danger_gps_com[h].extend(com[danger_mask].tolist())
            
            # SDC obs-timesteps
            if has_sdc and 'per_sdc_timestep_attention' in attn_dists:
                psa = attn_dists['per_sdc_timestep_attention']  # (T, n_obs, n_heads)
                n_obs = psa.shape[1]
                obs_idx = np.arange(n_obs, dtype=float)
                for h in range(n_heads):
                    recency = (psa[:, :, h] * obs_idx[None, :]).sum(axis=1)
                    if calm_mask.sum() > 0:
                        calm_sdc_rec[h].extend(recency[calm_mask].tolist())
                    if danger_mask.sum() > 0:
                        danger_sdc_rec[h].extend(recency[danger_mask].tolist())
        
        profiles = {}
        for h in range(n_heads):
            profile = {
                'n_calm': len(calm_max_frac[h]) if has_vehicles else
                          len(calm_rg_conc[h]) if has_roadgraph else
                          len(calm_tl_max[h]),
                'n_danger': len(danger_max_frac[h]) if has_vehicles else
                            len(danger_rg_conc[h]) if has_roadgraph else
                            len(danger_tl_max[h]),
            }
            
            if has_vehicles:
                profile['calm_max_vehicle_frac'] = float(np.mean(calm_max_frac[h])) if calm_max_frac[h] else 0.0
                profile['danger_max_vehicle_frac'] = float(np.mean(danger_max_frac[h])) if danger_max_frac[h] else 0.0
                profile['max_frac_gap'] = profile['danger_max_vehicle_frac'] - profile['calm_max_vehicle_frac']
                profile['calm_vehicle_entropy'] = float(np.mean(calm_veh_entropy[h])) if calm_veh_entropy[h] else 0.0
                profile['danger_vehicle_entropy'] = float(np.mean(danger_veh_entropy[h])) if danger_veh_entropy[h] else 0.0
                profile['entropy_gap'] = profile['danger_vehicle_entropy'] - profile['calm_vehicle_entropy']
            
            if has_roadgraph:
                profile['calm_rg_concentration'] = float(np.mean(calm_rg_conc[h])) if calm_rg_conc[h] else 0.0
                profile['danger_rg_concentration'] = float(np.mean(danger_rg_conc[h])) if danger_rg_conc[h] else 0.0
                profile['rg_conc_gap'] = profile['danger_rg_concentration'] - profile['calm_rg_concentration']
            
            if has_tl:
                profile['calm_tl_max_frac'] = float(np.mean(calm_tl_max[h])) if calm_tl_max[h] else 0.0
                profile['danger_tl_max_frac'] = float(np.mean(danger_tl_max[h])) if danger_tl_max[h] else 0.0
                profile['tl_gap'] = profile['danger_tl_max_frac'] - profile['calm_tl_max_frac']
            
            if has_gps:
                profile['calm_gps_com'] = float(np.mean(calm_gps_com[h])) if calm_gps_com[h] else 0.0
                profile['danger_gps_com'] = float(np.mean(danger_gps_com[h])) if danger_gps_com[h] else 0.0
                profile['gps_com_gap'] = profile['danger_gps_com'] - profile['calm_gps_com']
            
            if has_sdc:
                profile['calm_sdc_recency'] = float(np.mean(calm_sdc_rec[h])) if calm_sdc_rec[h] else 0.0
                profile['danger_sdc_recency'] = float(np.mean(danger_sdc_rec[h])) if danger_sdc_rec[h] else 0.0
                profile['sdc_recency_gap'] = profile['danger_sdc_recency'] - profile['calm_sdc_recency']
            
            profiles[h] = profile
            
            print(f"\n  Head {h} Risk Profile:")
            print(f"    Calm: {profile['n_calm']} ts | Danger: {profile['n_danger']} ts")
            if has_vehicles:
                print(f"    Veh max_frac:   calm={profile['calm_max_vehicle_frac']:.4f}  "
                      f"danger={profile['danger_max_vehicle_frac']:.4f}  "
                      f"gap={profile['max_frac_gap']:+.4f}")
                print(f"    Veh entropy:    calm={profile['calm_vehicle_entropy']:.4f}  "
                      f"danger={profile['danger_vehicle_entropy']:.4f}  "
                      f"gap={profile['entropy_gap']:+.4f}")
            if has_roadgraph:
                print(f"    RG conc:        calm={profile['calm_rg_concentration']:.4f}  "
                      f"danger={profile['danger_rg_concentration']:.4f}  "
                      f"gap={profile['rg_conc_gap']:+.4f}")
            if has_tl:
                print(f"    TL max_frac:    calm={profile['calm_tl_max_frac']:.4f}  "
                      f"danger={profile['danger_tl_max_frac']:.4f}  "
                      f"gap={profile['tl_gap']:+.4f}")
            if has_gps:
                print(f"    GPS center:     calm={profile['calm_gps_com']:.4f}  "
                      f"danger={profile['danger_gps_com']:.4f}  "
                      f"gap={profile['gps_com_gap']:+.4f}")
            if has_sdc:
                print(f"    SDC recency:    calm={profile['calm_sdc_recency']:.4f}  "
                      f"danger={profile['danger_sdc_recency']:.4f}  "
                      f"gap={profile['sdc_recency_gap']:+.4f}")
        
        return profiles
    
    # =========================================================================
    # HSI Computation (unified)
    # =========================================================================
    
    def compute_hsi(self) -> HSIResult:
        """Compute Head Specialization Index using both Option A and B.
        
        HSI_h = max over all features of |ρ̄_{h, feature}|
        Specialized if HSI > threshold AND 95% CI excludes zero.
        """
        print("\n" + "=" * 60)
        print("WITHIN-MODULE HSI ANALYSIS")
        print("=" * 60)
        
        # Option A: Consistent selectivity
        print("\n--- Option A: Consistent Selectivity ---")
        option_a = self.compute_option_a()
        
        # Option B: Risk-dependent selectivity
        print("\n--- Option B: Risk-Dependent Selectivity ---")
        option_b = self.compute_option_b()
        
        # Risk profiles
        print("\n--- Risk-Conditioned Profiles ---")
        risk_profiles = self.compute_risk_profiles()
        
        # Merge all correlations
        all_corrs = {}
        for fk, head_dict in {**option_a, **option_b}.items():
            if fk not in all_corrs:
                all_corrs[fk] = {}
            for h, stats_dict in head_dict.items():
                if h not in all_corrs[fk]:
                    all_corrs[fk][h] = stats_dict
        
        # Pivot to per-head view
        n_heads = max(
            max((max(hd.keys()) + 1 for hd in all_corrs.values() if hd), default=0),
            len(risk_profiles)
        )
        
        hsi_scores = np.zeros(n_heads)
        primary_features = []
        primary_correlations = np.zeros(n_heads)
        n_qualifying = {}
        
        # Build per-head aggregated dict
        per_head_agg = {h: {} for h in range(n_heads)}
        for fk, head_dict in all_corrs.items():
            for h, stats_dict in head_dict.items():
                per_head_agg[h][fk] = stats_dict
                n_qualifying[fk] = stats_dict.get('n_episodes', 0)
        
        for h in range(n_heads):
            if not per_head_agg[h]:
                primary_features.append('none')
                continue
            
            best_feature = None
            best_abs_rho = 0.0
            best_rho = 0.0
            
            for fk, sd in per_head_agg[h].items():
                abs_rho = abs(sd['rho_aggregated'])
                if abs_rho > best_abs_rho:
                    best_abs_rho = abs_rho
                    best_rho = sd['rho_aggregated']
                    best_feature = fk
            
            hsi_scores[h] = best_abs_rho
            primary_features.append(best_feature if best_feature else 'none')
            primary_correlations[h] = best_rho
        
        # Assign labels
        head_labels = {}
        for h in range(n_heads):
            feat = primary_features[h]
            
            if feat != 'none' and hsi_scores[h] >= self.HSI_THRESHOLD:
                sd = per_head_agg[h].get(feat, {})
                ci = sd.get('ci_95', (0.0, 0.0))
                ci_excludes_zero = (ci[0] > 0) or (ci[1] < 0)
                
                if ci_excludes_zero:
                    label_info = FEATURE_TO_LABEL.get(feat, DEFAULT_LABEL).copy()
                    label_info['specialized'] = True
                else:
                    label_info = DEFAULT_LABEL.copy()
                    label_info['specialized'] = False
                    label_info['reason'] = 'CI includes zero'
                
                label_info['primary_feature'] = feat
                label_info['hsi'] = float(hsi_scores[h])
                label_info['correlation'] = float(primary_correlations[h])
                label_info['ci_95'] = ci
                label_info['n_qualifying_episodes'] = sd.get('n_episodes', 0)
                label_info['frac_significant'] = sd.get('frac_significant', 0.0)
            else:
                label_info = DEFAULT_LABEL.copy()
                label_info['primary_feature'] = feat if feat != 'none' else None
                label_info['hsi'] = float(hsi_scores[h])
                label_info['correlation'] = float(primary_correlations[h])
                label_info['specialized'] = False
                label_info['reason'] = 'Below threshold'
            
            head_labels[h] = label_info
        
        self._result = HSIResult(
            hsi_scores=hsi_scores,
            aggregated_correlations=per_head_agg,
            primary_features=primary_features,
            primary_correlations=primary_correlations,
            head_labels=head_labels,
            risk_profiles=risk_profiles,
            risk_dependent=option_b,
            checkpoint_step=self._extraction_data['step'],
            n_qualifying_episodes=n_qualifying,
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("HEAD SPECIALIZATION RESULTS")
        print("=" * 60)
        for h, label in head_labels.items():
            tag = "✓ SPECIALIZED" if label.get('specialized') else "✗ General"
            print(f"Head {h}: HSI={label['hsi']:.3f} | {label['name']} [{tag}]")
            if label.get('primary_feature'):
                ci = label.get('ci_95', (0, 0))
                print(f"         Primary: {label['primary_feature']} "
                      f"(ρ̄={label['correlation']:.3f}, CI=[{ci[0]:.3f}, {ci[1]:.3f}])")
        print("=" * 60)
        
        return self._result
    
    def export_registry(self, output_path: str):
        """Export head function registry to JSON file."""
        if self._result is None:
            self.compute_hsi()
        
        registry = {
            'checkpoint_step': self._result.checkpoint_step,
            'checkpoint': self._extraction_data['checkpoint'],
            'methodology': 'within_module_distributional',
            'analysis_modes': ['option_a_consistent', 'option_b_risk_dependent'],
            'hsi_threshold': self.HSI_THRESHOLD,
            'heads': {},
        }
        
        for h, info in self._result.head_labels.items():
            head_data = {k: v for k, v in info.items()}
            if h in self._result.risk_profiles:
                head_data['risk_profile'] = self._result.risk_profiles[h]
            registry['heads'][str(h)] = head_data
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(registry, f, indent=2, default=str)
        print(f"[HSI] Exported registry to {output_path}")
    
    def visualize_all(self, output_dir: str):
        """Generate all visualizations."""
        if self._result is None:
            self.compute_hsi()
        
        os.makedirs(output_dir, exist_ok=True)
        viz = HeadVisualization(self)
        
        viz.plot_correlation_heatmap(
            save_path=os.path.join(output_dir, "correlation_heatmap.png"))
        plt.close()
        
        viz.plot_hsi_bar(
            save_path=os.path.join(output_dir, "hsi_bar_chart.png"))
        plt.close()
        
        for h in range(len(self._result.hsi_scores)):
            viz.plot_risk_profile(
                head_idx=h,
                save_path=os.path.join(output_dir, f"risk_profile_head_{h}.png"))
            plt.close()
        
        for h, label in self._result.head_labels.items():
            if label.get('specialized') and label.get('primary_feature'):
                viz.plot_per_episode_rho_scatter(
                    head_idx=h,
                    feature_key=label['primary_feature'],
                    save_path=os.path.join(output_dir,
                                           f"rho_scatter_head_{h}.png"))
                plt.close()
        
        episodes = self._extraction_data['scenarios']
        for i in range(min(3, len(episodes))):
            viz.plot_timeseries_diagnostic(
                scenario_idx=i,
                save_path=os.path.join(output_dir, f"timeseries_scenario_{i}.png"))
            plt.close()
        
        print(f"[HSI] Saved visualizations to {output_dir}")


# ==============================================================================
# Visualization
# ==============================================================================

class HeadVisualization:
    """Generate visualizations for within-module head specialization analysis."""
    
    def __init__(self, analyzer: HeadSpecializationAnalyzer):
        self.analyzer = analyzer
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 11
    
    def _save_fig(self, save_path: Optional[str] = None, dpi: int = 150):
        """Save the figure as both the original format and a lossless vector PDF."""
        if not save_path:
            return
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        
        # Save as PDF if not already a PDF
        base, ext = os.path.splitext(save_path)
        if ext.lower() != '.pdf':
            pdf_path = base + '.pdf'
            plt.savefig(pdf_path, bbox_inches='tight')
            print(f"[HSI] Saved additional PDF figure to {pdf_path}")
    
    def plot_correlation_heatmap(self, save_path: Optional[str] = None):
        """Heatmap of Fisher z-aggregated ρ̄. * = 95% CI excludes zero."""
        result = self.analyzer._result
        aggregated = result.aggregated_correlations
        n_heads = len(result.hsi_scores)
        
        all_fks = set()
        for h in range(n_heads):
            all_fks.update(aggregated[h].keys())
        feature_keys = sorted(all_fks)
        
        if not feature_keys:
            print("[Viz] No feature keys for heatmap.")
            return
        
        corr_matrix = np.zeros((n_heads, len(feature_keys)))
        sig_matrix = np.zeros((n_heads, len(feature_keys)), dtype=bool)
        
        for h in range(n_heads):
            for j, fk in enumerate(feature_keys):
                if fk in aggregated[h]:
                    sd = aggregated[h][fk]
                    corr_matrix[h, j] = sd['rho_aggregated']
                    ci = sd['ci_95']
                    sig_matrix[h, j] = (ci[0] > 0) or (ci[1] < 0)
        
        fig, ax = plt.subplots(figsize=(max(10, len(feature_keys) * 2.5), max(4, n_heads * 1.5)))
        im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        short_labels = []
        for fk in feature_keys:
            label = fk.replace('vehicle_attn_vs_', 'veh→').replace('_vs_risk', '↔R') \
                      .replace('vehicle_', 'veh_').replace('roadgraph_', 'rg_')
            short_labels.append(label)
        
        ax.set_xticks(range(len(feature_keys)))
        ax.set_xticklabels(short_labels, rotation=45, ha='right')
        ax.set_yticks(range(n_heads))
        ax.set_yticklabels([f"Head {h}" for h in range(n_heads)])
        
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Fisher z-aggregated Spearman ρ̄')
        
        for i in range(n_heads):
            for j in range(len(feature_keys)):
                val = corr_matrix[i, j]
                sig = "*" if sig_matrix[i, j] else ""
                color = 'white' if abs(val) > 0.4 else 'black'
                ax.text(j, i, f'{val:.2f}{sig}', ha='center', va='center',
                        color=color, fontweight='bold' if sig else 'normal')
        
        ax.set_title(f'Within-Module Head-Feature Correlations (Step {result.checkpoint_step})\n'
                     f'* = 95% CI excludes zero | A=consistent, B=risk-dependent')
        plt.tight_layout()
        if save_path:
            self._save_fig(save_path, dpi=150)
        return fig
    
    def plot_hsi_bar(self, save_path: Optional[str] = None):
        """Bar chart of HSI scores per head."""
        result = self.analyzer._result
        n_heads = len(result.hsi_scores)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = []
        for h in range(n_heads):
            if result.head_labels[h].get('specialized'):
                colors.append('#2ecc71')
            elif result.hsi_scores[h] >= self.analyzer.HSI_THRESHOLD:
                colors.append('#f39c12')
            else:
                colors.append('#95a5a6')
        
        bars = ax.bar(range(n_heads), result.hsi_scores, color=colors, edgecolor='black')
        ax.axhline(y=self.analyzer.HSI_THRESHOLD, color='red', linestyle='--',
                   label=f'Threshold ({self.analyzer.HSI_THRESHOLD})')
        
        for i, (bar, label_info) in enumerate(zip(bars, result.head_labels.values())):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    label_info['name'].replace(' ', '\n'), ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Attention Head')
        ax.set_ylabel('HSI = max |ρ̄|')
        ax.set_title(f'Head Specialization Index (Step {result.checkpoint_step})')
        ax.set_xticks(range(n_heads))
        ax.legend()
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        if save_path:
            self._save_fig(save_path, dpi=150)
        return fig
    
    def plot_risk_profile(self, head_idx: int, save_path: Optional[str] = None):
        """Attention concentration in calm vs danger phases.
        
        Left: max vehicle fraction + vehicle entropy in calm vs danger.
        Right: concentration gap (danger - calm).
        """
        result = self.analyzer._result
        if head_idx not in result.risk_profiles:
            return
        
        profile = result.risk_profiles[head_idx]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Collect metrics that exist
        metric_names = []
        calm_vals = []
        danger_vals = []
        gaps = []
        
        if 'calm_max_vehicle_frac' in profile:
            metric_names.append('Max Veh\nFraction')
            calm_vals.append(profile['calm_max_vehicle_frac'])
            danger_vals.append(profile['danger_max_vehicle_frac'])
            gaps.append(profile['max_frac_gap'])
        
        if 'calm_vehicle_entropy' in profile:
            metric_names.append('Vehicle\nEntropy')
            calm_vals.append(profile['calm_vehicle_entropy'])
            danger_vals.append(profile['danger_vehicle_entropy'])
            gaps.append(profile['entropy_gap'])
        
        if 'calm_rg_concentration' in profile:
            metric_names.append('Roadgraph\nConcentration')
            calm_vals.append(profile['calm_rg_concentration'])
            danger_vals.append(profile['danger_rg_concentration'])
            gaps.append(profile['rg_conc_gap'])
        
        if 'calm_tl_max_frac' in profile:
            metric_names.append('TL Max\nFraction')
            calm_vals.append(profile['calm_tl_max_frac'])
            danger_vals.append(profile['danger_tl_max_frac'])
            gaps.append(profile['tl_gap'])
        
        if 'calm_gps_com' in profile:
            metric_names.append('GPS\nCenter-of-Mass')
            calm_vals.append(profile['calm_gps_com'])
            danger_vals.append(profile['danger_gps_com'])
            gaps.append(profile['gps_com_gap'])
        
        if 'calm_sdc_recency' in profile:
            metric_names.append('SDC\nRecency')
            calm_vals.append(profile['calm_sdc_recency'])
            danger_vals.append(profile['danger_sdc_recency'])
            gaps.append(profile['sdc_recency_gap'])
        
        if not metric_names:
            return
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        ax1.bar(x - width/2, calm_vals, width,
                label=f'Calm (R < {self.analyzer.CALM_THRESHOLD})',
                color='#3498db', alpha=0.8)
        ax1.bar(x + width/2, danger_vals, width,
                label=f'Danger (R > {self.analyzer.DANGER_THRESHOLD})',
                color='#e74c3c', alpha=0.8)
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(metric_names)
        ax1.set_ylabel('Mean Value')
        ax1.set_title(f'Head {head_idx}: Attention Concentration by Phase')
        ax1.legend()
        
        gap_colors = ['#e74c3c' if g > 0 else '#3498db' for g in gaps]
        ax2.bar(x, gaps, color=gap_colors, alpha=0.8, edgecolor='black')
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(metric_names)
        ax2.set_ylabel('Gap (danger − calm)')
        ax2.set_title(f'Head {head_idx}: Concentration Change Under Threat')
        
        fig.suptitle(f'Head {head_idx} Risk Profile '
                     f'({profile["n_calm"]} calm / {profile["n_danger"]} danger timesteps)')
        
        plt.tight_layout()
        if save_path:
            self._save_fig(save_path, dpi=150)
        return fig
    
    def plot_per_episode_rho_scatter(self, head_idx: int, feature_key: str,
                                      save_path: Optional[str] = None):
        """Scatter per-episode ρ_i with Fisher aggregate line and CI band."""
        result = self.analyzer._result
        agg = result.aggregated_correlations.get(head_idx, {})
        if feature_key not in agg:
            return
        
        sd = agg[feature_key]
        rhos = np.array(sd['per_episode_rhos'])
        rho_agg = sd['rho_aggregated']
        ci = sd['ci_95']
        n = sd['n_episodes']
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.scatter(range(n), rhos, alpha=0.6, s=30, c='steelblue', zorder=3)
        ax.axhline(y=rho_agg, color='red', linewidth=2, label=f'ρ̄ = {rho_agg:.3f}')
        ax.axhspan(ci[0], ci[1], alpha=0.15, color='red',
                   label=f'95% CI [{ci[0]:.3f}, {ci[1]:.3f}]')
        ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
        
        ax.set_xlabel('Episode Index')
        ax.set_ylabel('Within-episode ρ')
        ax.set_title(f'Head {head_idx} | {feature_key} | n={n}')
        ax.legend(loc='upper right')
        ax.set_ylim(-1.05, 1.05)
        
        plt.tight_layout()
        if save_path:
            self._save_fig(save_path, dpi=150)
        return fig
    
    def plot_timeseries_diagnostic(self, scenario_idx: int = 0,
                                    save_path: Optional[str] = None):
        """Timeseries: risk R, per-vehicle attention, module scalar metrics, and ρ(attn, TTC)."""
        episodes = self.analyzer._extraction_data['scenarios']
        if scenario_idx >= len(episodes):
            return
        
        episode = episodes[scenario_idx]
        attn_dists, risk, semantics = self.analyzer._load_episode(episode)
        T = len(risk)
        time_axis = np.arange(T)
        
        has_vehicles = 'per_vehicle_attention' in attn_dists
        has_rg = 'roadgraph_concentration' in attn_dists
        has_tl = 'per_light_attention' in attn_dists
        has_gps = 'per_waypoint_attention' in attn_dists
        has_sdc = 'per_sdc_timestep_attention' in attn_dists
        
        # Determine n_heads
        if has_vehicles:
            pva = attn_dists['per_vehicle_attention']
            n_heads = pva.shape[2]
            n_vehicles = pva.shape[1]
        else:
            # Try to get n_heads from any available module
            for key in ['roadgraph_concentration', 'per_light_attention',
                        'per_waypoint_attention', 'per_sdc_timestep_attention']:
                if key in attn_dists:
                    n_heads = attn_dists[key].shape[-1]
                    break
            else:
                return
        
        # Compute cross-vehicle ρ(attn, TTC)
        rho_ttc = None
        if has_vehicles:
            ttc_series = semantics.get('ttc')
            if ttc_series is not None and ttc_series.ndim == 2:
                rho_ttc, _ = self.analyzer._compute_cross_vehicle_rho_timeseries(
                    pva, ttc_series
                )
        
        # Count panels: risk + per-head vehicle + module scalars + ρ panel
        n_panels = 1  # risk
        if has_vehicles:
            n_panels += n_heads  # per-head vehicle attention
        has_module_scalars = has_rg or has_tl or has_gps or has_sdc
        if has_module_scalars:
            n_panels += 1  # combined scalar metrics panel
        if rho_ttc is not None:
            n_panels += 1  # ρ(attn, TTC)
        
        fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3 * n_panels), sharex=True)
        if n_panels == 1:
            axes = [axes]
        
        panel_idx = 0
        
        # --- Panel: Risk ---
        ax = axes[panel_idx]
        ax.plot(time_axis, risk, color='black', linewidth=2, label='Risk R')
        ax.axhline(y=self.analyzer.CALM_THRESHOLD, color='green', linestyle='--', alpha=0.5, label='Calm')
        ax.axhline(y=self.analyzer.DANGER_THRESHOLD, color='red', linestyle='--', alpha=0.5, label='Danger')
        ax.fill_between(time_axis, 0, risk, where=risk > self.analyzer.DANGER_THRESHOLD,
                        alpha=0.2, color='red')
        ax.set_ylabel('Risk R')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9, loc='upper right')
        ax.set_title(f'Scenario {scenario_idx}: Multi-Module Attention Diagnostic')
        panel_idx += 1
        
        # --- Panels: per-head vehicle attention ---
        if has_vehicles:
            dist_series = semantics.get('distance_to_ego')
            veh_labels = []
            for v in range(n_vehicles):
                if dist_series is not None and dist_series.ndim == 2:
                    mean_dist = np.nanmean(dist_series[:, v])
                    veh_labels.append(f'Agent {v} ({mean_dist:.0f}m)')
                else:
                    veh_labels.append(f'Agent {v}')
            
            veh_colors = plt.cm.Set1(np.linspace(0, 1, n_vehicles))
            for h in range(n_heads):
                ax = axes[panel_idx]
                for v in range(n_vehicles):
                    ax.plot(time_axis, pva[:, v, h], color=veh_colors[v],
                           linewidth=1.2, alpha=0.7,
                           label=veh_labels[v] if h == 0 else None)
                ax.set_ylabel(f'Head {h}\nVeh Frac')
                if h == 0:
                    ax.legend(loc='upper right', fontsize=7, ncol=min(n_vehicles, 4))
                panel_idx += 1
        
        # --- Panel: Module scalar metrics (all heads overlaid) ---
        if has_module_scalars:
            ax = axes[panel_idx]
            head_colors = plt.cm.tab10(np.linspace(0, 1, n_heads))
            legend_entries = set()
            
            for h in range(n_heads):
                linestyle_idx = 0
                module_styles = ['-', '--', '-.', ':']
                
                if has_rg:
                    rg = attn_dists['roadgraph_concentration'][:, h]
                    label = 'RG conc' if 'RG conc' not in legend_entries else None
                    ax.plot(time_axis, rg, color=head_colors[h],
                           linestyle=module_styles[0], linewidth=1.2, alpha=0.6,
                           label=f'H{h} RG conc')
                
                if has_tl:
                    tl = attn_dists['per_light_attention'][:, :, h].max(axis=1)
                    ax.plot(time_axis, tl, color=head_colors[h],
                           linestyle=module_styles[1], linewidth=1.2, alpha=0.6,
                           label=f'H{h} TL max')
                
                if has_gps:
                    gps = attn_dists['per_waypoint_attention']
                    n_wp = gps.shape[1]
                    wp_idx = np.arange(n_wp, dtype=float) / max(n_wp - 1, 1)  # normalize 0-1
                    com = (gps[:, :, h] * wp_idx[None, :]).sum(axis=1)
                    ax.plot(time_axis, com, color=head_colors[h],
                           linestyle=module_styles[2], linewidth=1.2, alpha=0.6,
                           label=f'H{h} GPS CoM')
                
                if has_sdc:
                    sdc = attn_dists['per_sdc_timestep_attention']
                    n_obs = sdc.shape[1]
                    obs_idx = np.arange(n_obs, dtype=float) / max(n_obs - 1, 1)  # normalize 0-1
                    rec = (sdc[:, :, h] * obs_idx[None, :]).sum(axis=1)
                    ax.plot(time_axis, rec, color=head_colors[h],
                           linestyle=module_styles[3], linewidth=1.2, alpha=0.6,
                           label=f'H{h} SDC rec')
            
            ax.set_ylabel('Module\nMetrics')
            ax.legend(loc='upper right', fontsize=6, ncol=n_heads)
            panel_idx += 1
        
        # --- Panel: ρ(attn, TTC) ---
        if rho_ttc is not None:
            ax = axes[panel_idx]
            for h in range(n_heads):
                ax.plot(time_axis, rho_ttc[:, h], linewidth=1.5, alpha=0.8,
                       label=f'Head {h}')
            ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
            ax.set_ylabel('ρ(attn, TTC)')
            ax.legend(fontsize=9)
            panel_idx += 1
        
        axes[-1].set_xlabel('Timestep')
        
        plt.tight_layout()
        if save_path:
            self._save_fig(save_path, dpi=150)
        return fig


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Head Specialization Analysis — Within-Module Distributional"
    )
    parser.add_argument("--extraction_dir", type=str, required=True,
                        help="Directory with extraction output from offline_extraction.py (--rollout)")
    parser.add_argument("--extraction_file", type=str, default=None,
                        help="Specific extraction file (default: auto-select)")
    parser.add_argument("--output_dir", type=str, default="./hsi_results",
                        help="Output directory (default: ./hsi_results)")
    parser.add_argument("--hsi_threshold", type=float, default=0.3)
    parser.add_argument("--hv_threshold", type=float, default=0.2)
    
    args = parser.parse_args()
    
    analyzer = HeadSpecializationAnalyzer(extraction_dir=args.extraction_dir)
    analyzer.HSI_THRESHOLD = args.hsi_threshold
    analyzer.HV_THRESHOLD = args.hv_threshold
    
    print("=" * 60)
    print("HEAD SPECIALIZATION ANALYSIS — WITHIN-MODULE DISTRIBUTIONAL")
    print("=" * 60)
    
    analyzer.load_data(extraction_file=args.extraction_file)
    results = analyzer.compute_hsi()
    
    os.makedirs(args.output_dir, exist_ok=True)
    analyzer.export_registry(os.path.join(args.output_dir, "head_registry.json"))
    analyzer.visualize_all(os.path.join(args.output_dir, "visualizations"))
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
