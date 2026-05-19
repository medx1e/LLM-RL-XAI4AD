"""Calibration Analysis: Attention Concentration vs Criticality.

This script computes calibration scores measuring how well attention
concentrates on critical vehicles across scenarios.

Usage:
    python calibration_analysis.py --extraction extractions/extraction_model_final.pkl
    python calibration_analysis.py --extraction_dir extractions/ --compare
"""

import argparse
import glob
import os
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# Add project paths
analysis_dir = os.path.dirname(os.path.abspath(__file__))
research_dir = os.path.dirname(analysis_dir)
if research_dir not in sys.path:
    sys.path.insert(0, research_dir)

from analysis.concentration_metrics import (
    gini_coefficient,
    entropy_concentration,
    topk_mass,
    compute_concentration_suite,
)


# ============================================================
# Data Loading
# ============================================================

def load_extraction(path: str) -> Dict[str, Any]:
    """Load a single extraction pickle file."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"[CalibrationAnalysis] Loaded {data['n_scenarios']} scenarios from {path}")
    return data


CONCENTRATION_METRICS = ['gini', 'entropy', 'top3_mass']
METRIC_LABELS = {'gini': 'Gini', 'entropy': 'Entropy', 'top3_mass': 'Top-3 Mass'}


def extract_paired_data(extraction: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Extract paired criticality and concentration data from extraction results.
    
    For each scenario, we compute:
    - Scene-level criticality: max criticality across valid vehicles
    - Scene-level concentration per metric: mean across heads
    - Per-vehicle weighted attention-criticality alignment
    
    Returns:
        Dictionary with arrays:
        - 'scene_criticality': (n_scenarios,) max criticality per scene
        - 'scene_concentration': dict of metric -> (n_scenarios,) arrays
        - 'vehicle_criticality': (n_valid_vehicles,) per-vehicle criticality
        - 'vehicle_attention': (n_valid_vehicles,) per-vehicle attention mass
        - 'n_valid_vehicles': (n_scenarios,) vehicle count per scene
        - 'concentration_per_head': dict of metric -> (n_scenarios, n_heads) arrays
    """
    scenarios = extraction['scenarios']
    
    scene_criticality = []
    # Per-metric scene concentration and per-head arrays
    scene_concentration = {m: [] for m in CONCENTRATION_METRICS}
    concentration_per_head = {m: [] for m in CONCENTRATION_METRICS}
    vehicle_criticality_all = []
    vehicle_attention_all = []
    n_valid_vehicles = []
    
    for scenario in scenarios:
        sem = scenario.get('semantic_features', {})
        attn_per_vehicle = scenario.get('attention_per_vehicle')
        conc_suite = scenario.get('concentration_suite')  # New: dict of metric -> array(n_heads)
        conc_scores = scenario.get('concentration_scores')  # Legacy: Gini-only array(n_heads)
        
        # Skip scenarios with missing data
        if 'criticality' not in sem or attn_per_vehicle is None:
            continue
        
        # Build per-metric scores dict (handle old extractions without suite)
        if conc_suite is not None:
            metric_scores = conc_suite
        elif conc_scores is not None:
            # Legacy: only Gini available, compute others from attention
            metric_scores = {'gini': conc_scores}
            from analysis.concentration_metrics import compute_per_head_concentration
            for m in ['entropy', 'top3_mass']:
                metric_scores[m] = compute_per_head_concentration(attn_per_vehicle, metric=m)
        else:
            continue
        
        criticality = sem['criticality']
        valid = sem.get('valid', np.ones_like(criticality))
        
        # Scene-level metrics
        valid_mask = valid > 0.5
        n_valid = int(valid_mask.sum())
        n_valid_vehicles.append(n_valid)
        
        if n_valid == 0:
            scene_criticality.append(0.0)
            for m in CONCENTRATION_METRICS:
                scores = metric_scores.get(m)
                if scores is not None:
                    scene_concentration[m].append(float(np.mean(scores)))
                    concentration_per_head[m].append(scores)
            continue
        
        # Scene-level criticality: max across valid vehicles
        scene_crit = float(np.max(criticality[valid_mask]))
        scene_criticality.append(scene_crit)
        
        # Scene-level concentration per metric: mean across heads
        for m in CONCENTRATION_METRICS:
            scores = metric_scores.get(m)
            if scores is not None:
                scene_concentration[m].append(float(np.mean(scores)))
                concentration_per_head[m].append(scores)
        
        # Per-vehicle: sum attention across heads, pair with criticality
        # attn_per_vehicle shape: (n_heads, n_vehicles)
        attn_sum = attn_per_vehicle.sum(axis=0)  # (n_vehicles,)
        
        # Normalize attention to sum to 1
        attn_total = attn_sum[valid_mask].sum()
        if attn_total > 0:
            attn_norm = attn_sum[valid_mask] / attn_total
        else:
            attn_norm = np.zeros(n_valid)
        
        vehicle_criticality_all.extend(criticality[valid_mask].tolist())
        vehicle_attention_all.extend(attn_norm.tolist())
    
    # Convert to numpy arrays
    result = {
        'scene_criticality': np.array(scene_criticality),
        'scene_concentration': {m: np.array(v) for m, v in scene_concentration.items()},
        'vehicle_criticality': np.array(vehicle_criticality_all),
        'vehicle_attention': np.array(vehicle_attention_all),
        'n_valid_vehicles': np.array(n_valid_vehicles),
        'concentration_per_head': {
            m: np.array(v) if v else np.array([])
            for m, v in concentration_per_head.items()
        },
    }
    return result


# ============================================================
# Correlation Analysis
# ============================================================

def compute_correlations(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Compute Pearson and Spearman correlations with p-values.
    
    Args:
        x: First variable array.
        y: Second variable array.
        
    Returns:
        Dictionary with correlation coefficients and p-values.
    """
    if len(x) < 3 or len(y) < 3:
        return {
            'pearson_r': float('nan'),
            'pearson_p': float('nan'),
            'spearman_r': float('nan'),
            'spearman_p': float('nan'),
            'n_samples': len(x),
        }
    
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    
    return {
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'n_samples': len(x),
    }


def partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Dict[str, float]:
    """Compute partial correlation between x and y, controlling for z.
    
    Uses residual-based approach:
    1. Regress x on z, get residuals
    2. Regress y on z, get residuals
    3. Correlate residuals
    
    Args:
        x: First variable.
        y: Second variable.
        z: Control variable.
        
    Returns:
        Dictionary with partial correlation coefficient and p-value.
    """
    if len(x) < 4:
        return {'partial_r': float('nan'), 'partial_p': float('nan')}
    
    # Regress out z from x
    slope_xz, intercept_xz, _, _, _ = stats.linregress(z, x)
    x_resid = x - (slope_xz * z + intercept_xz)
    
    # Regress out z from y
    slope_yz, intercept_yz, _, _, _ = stats.linregress(z, y)
    y_resid = y - (slope_yz * z + intercept_yz)
    
    # Correlate residuals
    r, p = stats.pearsonr(x_resid, y_resid)
    
    return {'partial_r': float(r), 'partial_p': float(p)}


def bootstrap_ci(x: np.ndarray, y: np.ndarray, func=stats.pearsonr,
                 n_bootstrap: int = 1000, ci: float = 0.95,
                 seed: int = 42) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for a correlation.
    
    Args:
        x, y: Data arrays.
        func: Correlation function returning (statistic, p-value).
        n_bootstrap: Number of bootstrap samples.
        ci: Confidence level (e.g. 0.95 for 95% CI).
        seed: Random seed.
        
    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    rng = np.random.RandomState(seed)
    n = len(x)
    
    if n < 3:
        return (float('nan'), float('nan'))
    
    boot_stats = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        try:
            r, _ = func(x[idx], y[idx])
            boot_stats.append(r)
        except:
            continue
    
    if len(boot_stats) == 0:
        return (float('nan'), float('nan'))
    
    boot_stats = np.array(boot_stats)
    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_stats, alpha * 100))
    upper = float(np.percentile(boot_stats, (1 - alpha) * 100))
    
    return (lower, upper)


# ============================================================
# Calibration Score
# ============================================================

def compute_calibration_score(paired_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Compute the full calibration analysis for one extraction.
    
    Calibration measures how well attention concentration tracks criticality.
    A well-calibrated model concentrates attention more when situations are critical.
    
    Returns:
        Dictionary with all calibration metrics.
    """
    results = {}
    sc = paired_data['scene_criticality']
    
    # 1. Scene-level correlation per concentration metric
    scene_conc_dict = paired_data['scene_concentration']  # dict of metric -> array
    results['scene_level'] = {}
    for metric_name, sn in scene_conc_dict.items():
        if len(sn) == 0:
            continue
        corr = compute_correlations(sc, sn)
        corr['ci_95'] = bootstrap_ci(sc, sn)
        results['scene_level'][metric_name] = corr
    
    # 2. Vehicle-level correlation: attention vs criticality (metric-independent)
    vc = paired_data['vehicle_criticality']
    va = paired_data['vehicle_attention']
    results['vehicle_level'] = compute_correlations(vc, va)
    results['vehicle_level']['ci_95'] = bootstrap_ci(vc, va)
    
    # 3. Partial correlation per metric, controlling for scene complexity
    results['partial_corr'] = {}
    if len(sc) >= 4:
        n_veh = paired_data['n_valid_vehicles'].astype(float)
        for metric_name, sn in scene_conc_dict.items():
            if len(sn) == 0:
                continue
            results['partial_corr'][metric_name] = partial_correlation(sc, sn, n_veh)
    
    # 4. Summary statistics per metric
    results['summary'] = {
        'n_scenarios': len(sc),
        'n_vehicle_observations': len(vc),
        'mean_criticality': float(np.mean(sc)) if len(sc) > 0 else float('nan'),
        'std_criticality': float(np.std(sc)) if len(sc) > 0 else float('nan'),
    }
    for metric_name, sn in scene_conc_dict.items():
        if len(sn) > 0:
            results['summary'][f'mean_{metric_name}'] = float(np.mean(sn))
            results['summary'][f'std_{metric_name}'] = float(np.std(sn))
    
    # 5. Per-head concentration stats per metric
    conc_per_head_dict = paired_data.get('concentration_per_head', {})
    results['per_head'] = {}
    for metric_name, conc_arr in conc_per_head_dict.items():
        if conc_arr is not None and len(conc_arr) > 0 and conc_arr.ndim == 2:
            n_heads = conc_arr.shape[1]
            # Per-head scene-level correlation with criticality
            head_corrs = []
            for h in range(n_heads):
                head_conc = conc_arr[:, h]
                head_corrs.append(compute_correlations(sc, head_conc))
            
            results['per_head'][metric_name] = {
                'n_heads': n_heads,
                'mean': conc_arr.mean(axis=0).tolist(),
                'std': conc_arr.std(axis=0).tolist(),
                'per_head_corr': head_corrs,
            }
    
    return results


# ============================================================
# Regime Breakdown (Deliverable 5 preview)
# ============================================================

def compute_regime_breakdown(paired_data: Dict[str, np.ndarray],
                             low_threshold: float = 0.3,
                             high_threshold: float = 0.7) -> Dict[str, Any]:
    """Analyze calibration by criticality regime across all metrics.
    
    Args:
        paired_data: Paired data from extract_paired_data.
        low_threshold: Upper bound for "low criticality" regime.
        high_threshold: Lower bound for "high criticality" regime.
        
    Returns:
        Dictionary with per-regime, per-metric calibration metrics.
    """
    sc = paired_data['scene_criticality']
    scene_conc_dict = paired_data['scene_concentration']
    
    low_mask = sc <= low_threshold
    high_mask = sc >= high_threshold
    
    results = {
        'low_criticality': {'n_scenarios': int(low_mask.sum())},
        'high_criticality': {'n_scenarios': int(high_mask.sum())},
    }
    
    # Per-metric regime stats
    results['per_metric'] = {}
    for metric_name, sn in scene_conc_dict.items():
        if len(sn) == 0:
            continue
        
        m_result = {}
        
        if low_mask.sum() >= 3:
            m_result['low'] = {
                'mean': float(np.mean(sn[low_mask])),
                'std': float(np.std(sn[low_mask])),
            }
        
        if high_mask.sum() >= 3:
            m_result['high'] = {
                'mean': float(np.mean(sn[high_mask])),
                'std': float(np.std(sn[high_mask])),
            }
        
        # T-test between regimes
        if low_mask.sum() >= 3 and high_mask.sum() >= 3:
            t_stat, t_pval = stats.ttest_ind(sn[high_mask], sn[low_mask])
            m_result['regime_comparison'] = {
                'diff': float(np.mean(sn[high_mask]) - np.mean(sn[low_mask])),
                't_stat': float(t_stat),
                'p_value': float(t_pval),
                'significant': bool(t_pval < 0.05),
            }
        
        results['per_metric'][metric_name] = m_result
    
    # Criticality stats per regime
    if low_mask.sum() >= 1:
        results['low_criticality']['mean_criticality'] = float(np.mean(sc[low_mask]))
    if high_mask.sum() >= 1:
        results['high_criticality']['mean_criticality'] = float(np.mean(sc[high_mask]))
    
    return results


# ============================================================
# Pretty Printing
# ============================================================

def print_calibration_report(results: Dict[str, Any], model_name: str = "Model"):
    """Print a formatted calibration analysis report."""
    print("\n" + "=" * 70)
    print(f"  CALIBRATION ANALYSIS: {model_name}")
    print("=" * 70)
    
    # Summary
    s = results['summary']
    print(f"\n  Scenarios analyzed: {s['n_scenarios']}")
    print(f"  Vehicle observations: {s['n_vehicle_observations']}")
    print(f"  Mean criticality:    {s['mean_criticality']:.4f} ± {s['std_criticality']:.4f}")
    
    # Critical Scenarios Info
    if 'critical_scenarios_info' in results:
        info = results['critical_scenarios_info']
        crit_list = info['list']
        threshold = info['threshold']
        print(f"  Critical scenarios:  {len(crit_list)} (threshold: {threshold})")
        if crit_list:
            ids = [c['scenario_id'] for c in crit_list]
            # Show up to 20 IDs
            ids_str = ", ".join(map(str, ids[:20]))
            if len(ids) > 20:
                ids_str += " ..."
            print(f"    IDs: {ids_str}")
    
    # Per-metric summary
    print(f"\n  Concentration Metrics (scene-level mean ± std):")
    for m in CONCENTRATION_METRICS:
        mean_key = f'mean_{m}'
        std_key = f'std_{m}'
        if mean_key in s:
            print(f"    {METRIC_LABELS[m]:>10}: {s[mean_key]:.4f} ± {s[std_key]:.4f}")
    
    # Per-head stats per metric
    if 'per_head' in results and results['per_head']:
        print(f"\n  Per-Head Concentration:")
        for metric_name, ph in results['per_head'].items():
            label = METRIC_LABELS.get(metric_name, metric_name)
            n_heads = ph['n_heads']
            print(f"    {label}:")
            for i, (m, sd) in enumerate(zip(ph['mean'], ph['std'])):
                # Show per-head correlation with criticality
                corr_str = ""
                if 'per_head_corr' in ph:
                    hc = ph['per_head_corr'][i]
                    corr_str = f"  | Spearman ρ = {hc['spearman_r']:+.3f} (p={hc['spearman_p']:.3e})"
                print(f"      Head {i}: {m:.4f} ± {sd:.4f}{corr_str}")
    
    # Scene-level correlation per metric
    print("\n" + "-" * 70)
    print("  SCENE-LEVEL: Concentration vs Max Criticality")
    print("-" * 70)
    sl = results.get('scene_level', {})
    print(f"\n  {'Metric':>10} {'Pearson r':>12} {'p-value':>12} {'Spearman ρ':>12} {'p-value':>12} {'95% CI':>20}")
    print("  " + "-" * 80)
    for metric_name in CONCENTRATION_METRICS:
        if metric_name not in sl:
            continue
        mc = sl[metric_name]
        ci = mc.get('ci_95', (float('nan'), float('nan')))
        label = METRIC_LABELS[metric_name]
        print(f"  {label:>10} {mc['pearson_r']:>+12.4f} {mc['pearson_p']:>12.4e} "
              f"{mc['spearman_r']:>+12.4f} {mc['spearman_p']:>12.4e} "
              f"[{ci[0]:+.3f}, {ci[1]:+.3f}]")
    
    # Vehicle-level correlation
    print("\n" + "-" * 70)
    print("  VEHICLE-LEVEL: Attention Mass vs Criticality")
    print("-" * 70)
    vl = results['vehicle_level']
    print(f"  Pearson r:  {vl['pearson_r']:+.4f}  (p = {vl['pearson_p']:.4e})")
    print(f"  Spearman ρ: {vl['spearman_r']:+.4f}  (p = {vl['spearman_p']:.4e})")
    if 'ci_95' in vl:
        lo, hi = vl['ci_95']
        print(f"  95% CI:     [{lo:+.4f}, {hi:+.4f}]")
    
    # Partial correlation per metric
    pc = results.get('partial_corr', {})
    if pc:
        print(f"\n  Partial Correlation (controlling for scene complexity):")
        for metric_name, pc_vals in pc.items():
            label = METRIC_LABELS.get(metric_name, metric_name)
            print(f"    {label:>10}: r = {pc_vals['partial_r']:+.4f}  (p = {pc_vals['partial_p']:.4e})")
    
    print("\n" + "=" * 70)


def print_regime_report(regime_results: Dict[str, Any]):
    """Print regime breakdown report."""
    print("\n" + "-" * 70)
    print("  REGIME BREAKDOWN")
    print("-" * 70)
    
    for regime_name in ['low_criticality', 'high_criticality']:
        r = regime_results.get(regime_name, {})
        label = regime_name.replace('_', ' ').title()
        n = r.get('n_scenarios', 0)
        crit = r.get('mean_criticality', float('nan'))
        print(f"\n  {label}: {n} scenarios (mean criticality: {crit:.4f})")
    
    # Per-metric regime comparison
    per_metric = regime_results.get('per_metric', {})
    if per_metric:
        print(f"\n  {'Metric':>10} {'Low conc.':>12} {'High conc.':>12} {'Diff':>10} {'t-stat':>10} {'p-value':>12} {'Sig?':>6}")
        print("  " + "-" * 75)
        for metric_name in CONCENTRATION_METRICS:
            if metric_name not in per_metric:
                continue
            m = per_metric[metric_name]
            low_str = f"{m['low']['mean']:.4f}" if 'low' in m else 'N/A'
            high_str = f"{m['high']['mean']:.4f}" if 'high' in m else 'N/A'
            label = METRIC_LABELS[metric_name]
            if 'regime_comparison' in m:
                rc = m['regime_comparison']
                sig = "✓" if rc['significant'] else "✗"
                print(f"  {label:>10} {low_str:>12} {high_str:>12} {rc['diff']:>+10.4f} "
                      f"{rc['t_stat']:>10.4f} {rc['p_value']:>12.4e} {sig:>6}")
            else:
                print(f"  {label:>10} {low_str:>12} {high_str:>12} {'—':>10} {'—':>10} {'—':>12} {'—':>6}")
    
    print()


# ============================================================
# Main
# ============================================================

def analyze_single(extraction_path: str, model_name: Optional[str] = None) -> Dict[str, Any]:
    """Run full calibration analysis on a single extraction file."""
    data = load_extraction(extraction_path)
    paired = extract_paired_data(data)
    
    if model_name is None:
        model_name = os.path.basename(os.path.dirname(extraction_path))
    
    # Core calibration
    calibration = compute_calibration_score(paired)
    
    # Add critical scenario info if present in extraction data
    if 'critical_scenarios' in data:
        calibration['critical_scenarios_info'] = {
            'list': data['critical_scenarios'],
            'threshold': data.get('critical_threshold', 0.7)
        }
    
    # Regime breakdown
    regime = compute_regime_breakdown(paired)
    
    # Print reports
    print_calibration_report(calibration, model_name)
    print_regime_report(regime)
    
    return {
        'model_name': model_name,
        'extraction_path': extraction_path,
        'calibration': calibration,
        'regime': regime,
        'paired_data': paired,
    }


def compare_models(extraction_paths: List[str], 
                   model_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Run calibration analysis on multiple extractions and compare."""
    all_results = []
    
    for i, path in enumerate(extraction_paths):
        name = model_names[i] if model_names and i < len(model_names) else None
        result = analyze_single(path, name)
        all_results.append(result)
    
    # Print comparison table
    if len(all_results) > 1:
        print("\n\n" + "=" * 90)
        print("  CALIBRATION COMPARISON ACROSS MODELS")
        print("=" * 90)
        
        for metric_name in CONCENTRATION_METRICS:
            label = METRIC_LABELS[metric_name]
            print(f"\n  Metric: {label}")
            print(f"  {'Model':<25} {'Pearson r':>10} {'Spearman ρ':>12} {'Partial r':>12} {'N':>5}")
            print("  " + "-" * 65)
            for r in all_results:
                cal = r['calibration']
                sl = cal.get('scene_level', {}).get(metric_name, {})
                pc = cal.get('partial_corr', {}).get(metric_name, {})
                pr = sl.get('pearson_r', float('nan'))
                sr = sl.get('spearman_r', float('nan'))
                pcr = pc.get('partial_r', float('nan'))
                n = sl.get('n_samples', 0)
                print(f"  {r['model_name']:<25} {pr:>+10.4f} {sr:>+12.4f} "
                      f"{pcr:>+12.4f} {n:>5}")
        print("=" * 90)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Calibration Analysis: Attention Concentration vs Criticality"
    )
    parser.add_argument(
        "--extraction", type=str, default=None,
        help="Path to a single extraction pickle file"
    )
    parser.add_argument(
        "--extraction_dir", type=str, default=None,
        help="Directory containing extraction pickle files (for comparison)"
    )
    parser.add_argument(
        "--model_name", type=str, default=None,
        help="Name for the model (used in reports)"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare all extractions in the directory"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save results as pickle"
    )
    args = parser.parse_args()
    
    if args.extraction:
        results = analyze_single(args.extraction, args.model_name)
        
        if args.output:
            with open(args.output, 'wb') as f:
                # Don't save paired_data arrays to keep file small
                save_data = {k: v for k, v in results.items() if k != 'paired_data'}
                pickle.dump(save_data, f)
            print(f"\n[CalibrationAnalysis] Results saved to {args.output}")
    
    elif args.extraction_dir and args.compare:
        paths = sorted(glob.glob(os.path.join(args.extraction_dir, "*.pkl")))
        if not paths:
            print(f"No .pkl files found in {args.extraction_dir}")
            return
        
        results = compare_models(paths)
        
        if args.output:
            with open(args.output, 'wb') as f:
                save_data = [{k: v for k, v in r.items() if k != 'paired_data'} for r in results]
                pickle.dump(save_data, f)
            print(f"\n[CalibrationAnalysis] Comparison results saved to {args.output}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
