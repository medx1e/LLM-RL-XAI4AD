"""Calibration Visualization Suite.

Publication-ready visualizations for attention calibration analysis:
  1. Scatter plot: Concentration vs Criticality (scene & vehicle level)
  2. Bar chart: Calibration scores across architectures / metrics
  3. Time-series: Per-head concentration evolution across scenarios
  4. Regime comparison: Box plots by criticality regime
  5. Per-head heatmap: Head-level correlation with criticality

Usage:
    python calibration_visualization.py --extraction extractions/extraction_model_final.pkl
    python calibration_visualization.py --extraction_dir extractions/ --compare
"""

import argparse
import glob
import os
import pickle
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy import stats

# LOWESS for non-linear trend fitting
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    LOWESS_AVAILABLE = True
except ImportError:
    LOWESS_AVAILABLE = False
    print("[Warning] statsmodels not available. Using linear fit instead of LOWESS.")

# Add project paths
analysis_dir = os.path.dirname(os.path.abspath(__file__))
research_dir = os.path.dirname(analysis_dir)
if research_dir not in sys.path:
    sys.path.insert(0, research_dir)

from analysis.calibration_analysis import (
    load_extraction,
    extract_paired_data,
    compute_calibration_score,
    compute_regime_breakdown,
    CONCENTRATION_METRICS,
    METRIC_LABELS,
)


# ============================================================
# Style Configuration
# ============================================================

# Professional color palette
COLORS = {
    'primary': '#2563EB',       # Blue
    'secondary': '#7C3AED',     # Purple
    'accent': '#059669',        # Emerald
    'warning': '#D97706',       # Amber
    'danger': '#DC2626',        # Red
    'muted': '#6B7280',         # Gray
    'bg_dark': '#1F2937',       # Dark gray
    'bg_light': '#F9FAFB',      # Light gray
    'grid': '#E5E7EB',          # Grid lines
}

ARCHITECTURE_COLORS = {
    'lq': '#2563EB',
    'perceiver': '#2563EB',
    'wayformer': '#7C3AED',
    'mtr': '#059669',
}

METRIC_COLORS = {
    'gini': '#2563EB',
    'entropy': '#7C3AED',
    'top3_mass': '#059669',
}

METRIC_MARKERS = {
    'gini': 'o',
    'entropy': 's',
    'top3_mass': '^',
}


def apply_style():
    """Apply publication-ready matplotlib style."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': '#FAFBFC',
        'axes.edgecolor': '#D1D5DB',
        'axes.labelcolor': '#1F2937',
        'axes.titlecolor': '#111827',
        'axes.grid': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.color': '#E5E7EB',
        'grid.alpha': 0.7,
        'grid.linewidth': 0.5,
        'text.color': '#374151',
        'xtick.color': '#6B7280',
        'ytick.color': '#6B7280',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 15,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2,
    })


# ============================================================
# 1. Scatter Plot: Concentration vs Criticality
# ============================================================

def plot_scatter_concentration_criticality(
    paired_data: Dict[str, np.ndarray],
    model_name: str = "Model",
    output_path: Optional[str] = None,
    metric: str = 'gini',
):
    """Scatter plot of concentration vs criticality at scene and vehicle level.
    
    Creates a 1x2 subplot:
    - Left: Scene-level (concentration vs max criticality)
    - Right: Vehicle-level (attention mass vs criticality)
    """
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    label = METRIC_LABELS.get(metric, metric)
    color = METRIC_COLORS.get(metric, COLORS['primary'])
    
    # --- Left panel: Scene-level ---
    ax = axes[0]
    sc = paired_data['scene_criticality']
    sn = paired_data['scene_concentration'].get(metric, np.array([]))
    
    if len(sc) > 0 and len(sn) > 0:
        ax.scatter(sc, sn, c=color, alpha=0.6, s=50, edgecolors='white',
                   linewidth=0.5, zorder=3)
        
        # Trend curve (LOWESS for non-linear monotonic relationships)
        if len(sc) >= 10:
            if LOWESS_AVAILABLE:
                # Sort data for LOWESS
                sort_idx = np.argsort(sc)
                sc_sorted = sc[sort_idx]
                sn_sorted = sn[sort_idx]
                
                # LOWESS smoothing (frac controls smoothness: lower = more local)
                smoothed = lowess(sn_sorted, sc_sorted, frac=0.3, return_sorted=False)
                
                ax.plot(sc_sorted, smoothed, '-', color=color, alpha=0.9, linewidth=2.5,
                        label='LOWESS trend', zorder=4)
            else:
                # Fallback to linear fit
                z = np.polyfit(sc, sn, 1)
                p = np.poly1d(z)
                x_line = np.linspace(sc.min(), sc.max(), 100)
                ax.plot(x_line, p(x_line), '--', color=color, alpha=0.8, linewidth=2,
                        label='Linear fit', zorder=4)
            
            # Correlation annotation
            r, pval = stats.spearmanr(sc, sn)
            sig_str = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
            ax.annotate(f'ρ = {r:+.3f} ({sig_str})\nn = {len(sc)}',
                       xy=(0.05, 0.95), xycoords='axes fraction',
                       fontsize=11, fontweight='bold', color=color,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                edgecolor=color, alpha=0.9))
    
    ax.set_xlabel('Max Scene Criticality', fontweight='medium')
    ax.set_ylabel(f'{label} Concentration', fontweight='medium')
    ax.set_title(f'Scene-Level: {label} vs Criticality', fontweight='bold')
    
    # --- Right panel: Vehicle-level ---
    ax = axes[1]
    vc = paired_data['vehicle_criticality']
    va = paired_data['vehicle_attention']
    
    if len(vc) > 0 and len(va) > 0:
        ax.scatter(vc, va, c=COLORS['secondary'], alpha=0.35, s=25,
                   edgecolors='white', linewidth=0.3, zorder=3)
        
        # Trend curve (LOWESS for non-linear relationships)
        if len(vc) >= 50:  # Vehicle-level has many more data points
            if LOWESS_AVAILABLE:
                # Sort data for LOWESS
                sort_idx = np.argsort(vc)
                vc_sorted = vc[sort_idx]
                va_sorted = va[sort_idx]
                
                # LOWESS smoothing (smaller frac for large datasets)
                smoothed = lowess(va_sorted, vc_sorted, frac=0.15, return_sorted=False)
                
                ax.plot(vc_sorted, smoothed, '-', color=COLORS['secondary'], 
                        alpha=0.9, linewidth=2.5, label='LOWESS trend', zorder=4)
            else:
                # Fallback to linear fit
                z = np.polyfit(vc, va, 1)
                p = np.poly1d(z)
                x_line = np.linspace(vc.min(), vc.max(), 100)
                ax.plot(x_line, p(x_line), '--', color=COLORS['secondary'],
                        alpha=0.8, linewidth=2, zorder=4)
            
            # Correlation
            r, pval = stats.spearmanr(vc, va)
            sig_str = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
            ax.annotate(f'ρ = {r:+.3f} ({sig_str})\nn = {len(vc)}',
                       xy=(0.05, 0.95), xycoords='axes fraction',
                       fontsize=11, fontweight='bold', color=COLORS['secondary'],
                       verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                edgecolor=COLORS['secondary'], alpha=0.9))
    
    ax.set_xlabel('Vehicle Criticality', fontweight='medium')
    ax.set_ylabel('Normalized Attention Mass', fontweight='medium')
    ax.set_title('Vehicle-Level: Attention vs Criticality', fontweight='bold')
    
    fig.suptitle(f'Attention Calibration — {model_name}',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300)
        print(f"[Viz] Saved scatter plot → {output_path}")
    
    plt.close(fig)
    return fig


def plot_multi_metric_scatter(
    paired_data: Dict[str, np.ndarray],
    model_name: str = "Model",
    output_path: Optional[str] = None,
):
    """Scatter plot comparing all three concentration metrics vs criticality."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    sc = paired_data['scene_criticality']
    
    for idx, metric in enumerate(CONCENTRATION_METRICS):
        ax = axes[idx]
        label = METRIC_LABELS[metric]
        color = METRIC_COLORS[metric]
        sn = paired_data['scene_concentration'].get(metric, np.array([]))
        
        if len(sc) == 0 or len(sn) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color=COLORS['muted'])
            ax.set_title(label, fontweight='bold')
            continue
        
        ax.scatter(sc, sn, c=color, alpha=0.55, s=50, edgecolors='white',
                   linewidth=0.5, zorder=3)
        
        # Trend curve (LOWESS for non-linear relationships)
        if len(sc) >= 10:
            if LOWESS_AVAILABLE:
                # Sort data for LOWESS
                sort_idx = np.argsort(sc)
                sc_sorted = sc[sort_idx]
                sn_sorted = sn[sort_idx]
                
                # LOWESS smoothing
                smoothed = lowess(sn_sorted, sc_sorted, frac=0.3, return_sorted=False)
                
                ax.plot(sc_sorted, smoothed, '-', color=color, alpha=0.9, linewidth=2.5,
                        zorder=4)
            else:
                # Fallback to linear fit
                z = np.polyfit(sc, sn, 1)
                p = np.poly1d(z)
                x_line = np.linspace(sc.min(), sc.max(), 100)
                y_line = p(x_line)
                ax.plot(x_line, y_line, '--', color=color, alpha=0.8, linewidth=2)
            
            # Spearman and Pearson
            r_s, p_s = stats.spearmanr(sc, sn)
            r_p, p_p = stats.pearsonr(sc, sn)
            
            sig_p = "***" if p_p < 0.001 else "**" if p_p < 0.01 else "*" if p_p < 0.05 else "n.s."
            sig_s = "***" if p_s < 0.001 else "**" if p_s < 0.01 else "*" if p_s < 0.05 else "n.s."
            
            ax.annotate(
                f'Pearson r = {r_p:+.3f} ({sig_p})\nSpearman ρ = {r_s:+.3f} ({sig_s})',
                xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=10, fontweight='bold', color=color,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor=color, alpha=0.9))
        
        ax.set_xlabel('Max Scene Criticality', fontweight='medium')
        ax.set_ylabel(f'{label} Concentration', fontweight='medium')
        ax.set_title(label, fontweight='bold', color=color)
    
    fig.suptitle(f'Scene-Level Calibration — {model_name}',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300)
        print(f"[Viz] Saved multi-metric scatter → {output_path}")
    
    plt.close(fig)
    return fig


# ============================================================
# 2. Bar Chart: Calibration Scores Across Architectures
# ============================================================

def plot_calibration_comparison_bars(
    all_results: List[Dict[str, Any]],
    output_path: Optional[str] = None,
):
    """Bar chart comparing calibration (Spearman ρ) across architectures and metrics.
    
    Args:
        all_results: List of dicts from analyze_single, each containing
                     'model_name' and 'calibration' keys.
    """
    apply_style()
    
    model_names = [r['model_name'] for r in all_results]
    n_models = len(model_names)
    n_metrics = len(CONCENTRATION_METRICS)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Left panel: Scene-level Spearman ρ per metric ---
    ax = axes[0]
    bar_width = 0.25
    x = np.arange(n_models)
    
    for i, metric in enumerate(CONCENTRATION_METRICS):
        label = METRIC_LABELS[metric]
        color = METRIC_COLORS[metric]
        
        values = []
        ci_lo = []
        ci_hi = []
        for r in all_results:
            sl = r['calibration'].get('scene_level', {}).get(metric, {})
            val = sl.get('spearman_r', float('nan'))
            ci = sl.get('ci_95', (float('nan'), float('nan')))
            values.append(val)
            ci_lo.append(val - ci[0] if not np.isnan(ci[0]) else 0)
            ci_hi.append(ci[1] - val if not np.isnan(ci[1]) else 0)
        
        values = np.array(values)
        errors = np.array([ci_lo, ci_hi])
        
        bars = ax.bar(x + i * bar_width, values, bar_width,
                       label=label, color=color, alpha=0.85,
                       edgecolor='white', linewidth=0.5)
        ax.errorbar(x + i * bar_width, values, yerr=errors,
                    fmt='none', color='#374151', capsize=3, linewidth=1)
    
    ax.set_xlabel('Architecture', fontweight='medium')
    ax.set_ylabel('Spearman ρ', fontweight='medium')
    ax.set_title('Scene-Level Calibration', fontweight='bold')
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(model_names, fontweight='medium')
    ax.axhline(y=0, color=COLORS['muted'], linewidth=0.8, linestyle='-')
    ax.legend(loc='upper left', framealpha=0.9)
    
    # --- Right panel: Vehicle-level Spearman ρ ---
    ax = axes[1]
    values = []
    ci_lo = []
    ci_hi = []
    colors_bars = []
    
    for r in all_results:
        vl = r['calibration'].get('vehicle_level', {})
        val = vl.get('spearman_r', float('nan'))
        ci = vl.get('ci_95', (float('nan'), float('nan')))
        values.append(val)
        ci_lo.append(val - ci[0] if not np.isnan(ci[0]) else 0)
        ci_hi.append(ci[1] - val if not np.isnan(ci[1]) else 0)
        # Use architecture-specific color
        name_lower = r['model_name'].lower().replace('/', '_').split('_')[0]
        colors_bars.append(ARCHITECTURE_COLORS.get(name_lower, COLORS['primary']))
    
    values = np.array(values)
    errors = np.array([ci_lo, ci_hi])
    
    bars = ax.bar(x, values, 0.5, color=colors_bars, alpha=0.85,
                  edgecolor='white', linewidth=0.5)
    ax.errorbar(x, values, yerr=errors,
                fmt='none', color='#374151', capsize=4, linewidth=1)
    
    # Value labels on bars
    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10,
                    fontweight='bold', color='#374151')
    
    ax.set_xlabel('Architecture', fontweight='medium')
    ax.set_ylabel('Spearman ρ', fontweight='medium')
    ax.set_title('Vehicle-Level Calibration', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontweight='medium')
    ax.axhline(y=0, color=COLORS['muted'], linewidth=0.8, linestyle='-')
    
    fig.suptitle('Calibration Comparison Across Architectures',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300)
        print(f"[Viz] Saved calibration bar chart → {output_path}")
    
    plt.close(fig)
    return fig


# ============================================================
# 3. Time-Series: Per-Scenario Concentration & Criticality
# ============================================================

def plot_concentration_criticality_timeseries(
    paired_data: Dict[str, np.ndarray],
    model_name: str = "Model",
    output_path: Optional[str] = None,
    metric: str = 'gini',
):
    """Time-series plot showing concentration and criticality evolution across scenarios.
    
    X-axis: scenario index (ordered by criticality)
    Y-axis (left): Criticality
    Y-axis (right): Concentration
    """
    apply_style()
    
    sc = paired_data['scene_criticality']
    sn = paired_data['scene_concentration'].get(metric, np.array([]))
    label = METRIC_LABELS.get(metric, metric)
    
    if len(sc) == 0 or len(sn) == 0:
        print("[Viz] No data for time-series plot")
        return None
    
    # Sort by criticality for visual clarity
    sort_idx = np.argsort(sc)
    sc_sorted = sc[sort_idx]
    sn_sorted = sn[sort_idx]
    
    fig, ax1 = plt.subplots(figsize=(14, 5))
    
    x = np.arange(len(sc_sorted))
    
    # Criticality bars (background)
    ax1.bar(x, sc_sorted, color=COLORS['danger'], alpha=0.3, width=1.0,
            label='Criticality', zorder=1)
    ax1.set_xlabel('Scenarios (sorted by criticality)', fontweight='medium')
    ax1.set_ylabel('Max Criticality', color=COLORS['danger'], fontweight='medium')
    ax1.tick_params(axis='y', labelcolor=COLORS['danger'])
    ax1.set_ylim(0, max(1.0, sc_sorted.max() * 1.1))
    
    # Concentration line (overlay)
    ax2 = ax1.twinx()
    color = METRIC_COLORS.get(metric, COLORS['primary'])
    ax2.plot(x, sn_sorted, color=color, linewidth=2, alpha=0.9,
             label=f'{label} Concentration', zorder=3)
    
    # Moving average
    window = max(5, len(x) // 10)
    if len(sn_sorted) >= window:
        ma = np.convolve(sn_sorted, np.ones(window) / window, mode='valid')
        ma_x = x[window // 2: window // 2 + len(ma)]
        ax2.plot(ma_x, ma, color=color, linewidth=3, alpha=0.5,
                 linestyle='--', label=f'Moving avg (w={window})')
    
    ax2.set_ylabel(f'{label} Concentration', color=color, fontweight='medium')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
               framealpha=0.9, edgecolor=COLORS['grid'])
    
    ax1.set_title(f'Criticality & {label} Concentration — {model_name}',
                  fontweight='bold', pad=10)
    
    fig.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300)
        print(f"[Viz] Saved time-series → {output_path}")
    
    plt.close(fig)
    return fig


# ============================================================
# 4. Regime Comparison Box Plot
# ============================================================

def plot_regime_boxplot(
    paired_data: Dict[str, np.ndarray],
    model_name: str = "Model",
    output_path: Optional[str] = None,
    low_threshold: float = 0.3,
    high_threshold: float = 0.7,
):
    """Box plots comparing concentration distributions in low vs high criticality regimes."""
    apply_style()
    
    sc = paired_data['scene_criticality']
    low_mask = sc <= low_threshold
    mid_mask = (sc > low_threshold) & (sc < high_threshold)
    high_mask = sc >= high_threshold
    
    n_metrics = len(CONCENTRATION_METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]
    
    regime_colors = ['#93C5FD', '#FCD34D', '#FCA5A5']  # blue, amber, red (light)
    regime_edge = ['#2563EB', '#D97706', '#DC2626']
    regime_labels = ['Low', 'Medium', 'High']
    masks = [low_mask, mid_mask, high_mask]
    
    for idx, metric in enumerate(CONCENTRATION_METRICS):
        ax = axes[idx]
        label = METRIC_LABELS[metric]
        sn = paired_data['scene_concentration'].get(metric, np.array([]))
        
        if len(sn) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(label, fontweight='bold')
            continue
        
        box_data = []
        box_labels = []
        box_colors = []
        box_edges = []
        
        for i, (mask, rname) in enumerate(zip(masks, regime_labels)):
            if mask.sum() > 0:
                box_data.append(sn[mask])
                box_labels.append(f'{rname}\n(n={mask.sum()})')
                box_colors.append(regime_colors[i])
                box_edges.append(regime_edge[i])
        
        if not box_data:
            continue
        
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                        widths=0.6, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='white',
                                      markeredgecolor='#374151', markersize=7),
                        medianprops=dict(color='#374151', linewidth=2),
                        whiskerprops=dict(color='#9CA3AF'),
                        capprops=dict(color='#9CA3AF'),
                        flierprops=dict(marker='o', markerfacecolor='#D1D5DB',
                                       markersize=4, alpha=0.5))
        
        for patch, fc, ec in zip(bp['boxes'], box_colors, box_edges):
            patch.set_facecolor(fc)
            patch.set_edgecolor(ec)
            patch.set_linewidth(1.5)
            patch.set_alpha(0.8)
        
        ax.set_ylabel(f'{label} Concentration', fontweight='medium')
        ax.set_title(label, fontweight='bold', color=METRIC_COLORS[metric])
        
        # T-test annotation between low and high
        if low_mask.sum() >= 3 and high_mask.sum() >= 3:
            t_stat, p_val = stats.ttest_ind(sn[high_mask], sn[low_mask])
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
            diff = np.mean(sn[high_mask]) - np.mean(sn[low_mask])
            ax.annotate(f'Δ = {diff:+.3f} ({sig})',
                       xy=(0.5, 0.02), xycoords='axes fraction',
                       ha='center', fontsize=10, fontweight='bold',
                       color='#374151',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                edgecolor='#D1D5DB', alpha=0.9))
    
    fig.suptitle(f'Concentration by Criticality Regime — {model_name}',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300)
        print(f"[Viz] Saved regime boxplot → {output_path}")
    
    plt.close(fig)
    return fig


# ============================================================
# 5. Per-Head Correlation Heatmap
# ============================================================

def plot_per_head_heatmap(
    calibration_results: Dict[str, Any],
    model_name: str = "Model",
    output_path: Optional[str] = None,
):
    """Heatmap showing per-head Spearman ρ with criticality for each metric."""
    apply_style()
    
    per_head = calibration_results.get('per_head', {})
    if not per_head:
        print("[Viz] No per-head data for heatmap")
        return None
    
    # Determine grid dimensions
    metrics_with_data = [m for m in CONCENTRATION_METRICS if m in per_head]
    if not metrics_with_data:
        return None
    
    n_heads = per_head[metrics_with_data[0]]['n_heads']
    n_metrics = len(metrics_with_data)
    
    # Build correlation matrix: (n_metrics, n_heads)
    corr_matrix = np.zeros((n_metrics, n_heads))
    pval_matrix = np.zeros((n_metrics, n_heads))
    
    for i, metric in enumerate(metrics_with_data):
        ph = per_head[metric]
        for h in range(n_heads):
            hc = ph['per_head_corr'][h]
            corr_matrix[i, h] = hc['spearman_r']
            pval_matrix[i, h] = hc['spearman_p']
    
    fig, ax = plt.subplots(figsize=(max(6, n_heads * 2 + 2), n_metrics * 1.2 + 2))
    
    # Color map: diverging blue-white-red
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
    
    # Annotate with values and significance
    for i in range(n_metrics):
        for h in range(n_heads):
            val = corr_matrix[i, h]
            pval = pval_matrix[i, h]
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            text_color = 'white' if abs(val) > 0.3 else '#374151'
            ax.text(h, i, f'{val:+.3f}\n{sig}', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=text_color)
    
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f'Head {h}' for h in range(n_heads)], fontweight='medium')
    ax.set_yticks(range(n_metrics))
    ax.set_yticklabels([METRIC_LABELS[m] for m in metrics_with_data], fontweight='medium')
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label='Spearman ρ')
    cbar.ax.tick_params(labelsize=9)
    
    ax.set_title(f'Per-Head Calibration — {model_name}',
                 fontweight='bold', pad=15)
    
    fig.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300)
        print(f"[Viz] Saved per-head heatmap → {output_path}")
    
    plt.close(fig)
    return fig


# ============================================================
# 6. Combined Dashboard
# ============================================================

def generate_dashboard(
    extraction_path: str,
    model_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    metric: str = 'gini',
):
    """Generate the full visualization dashboard for one model.
    
    Creates 5 figures:
    1. Multi-metric scatter (scene-level)
    2. Single-metric scatter (scene + vehicle)
    3. Time-series (criticality + concentration)
    4. Regime box plots
    5. Per-head heatmap
    """
    data = load_extraction(extraction_path)
    paired = extract_paired_data(data)
    calibration = compute_calibration_score(paired)
    
    if model_name is None:
        model_name = os.path.basename(os.path.dirname(extraction_path))
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(extraction_path), 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    prefix = model_name.lower().replace('/', '_').replace(' ', '_')
    
    print(f"\n[Viz] Generating dashboard for {model_name}...")
    print(f"[Viz] Output directory: {output_dir}")
    
    # 1. Multi-metric scatter
    plot_multi_metric_scatter(
        paired, model_name,
        os.path.join(output_dir, f'{prefix}_scatter_multi_metric.png'))
    
    # 2. Single-metric scatter (scene + vehicle)
    plot_scatter_concentration_criticality(
        paired, model_name,
        os.path.join(output_dir, f'{prefix}_scatter_{metric}.png'),
        metric=metric)
    
    # 3. Time-series
    plot_concentration_criticality_timeseries(
        paired, model_name,
        os.path.join(output_dir, f'{prefix}_timeseries_{metric}.png'),
        metric=metric)
    
    # 4. Regime box plot
    plot_regime_boxplot(
        paired, model_name,
        os.path.join(output_dir, f'{prefix}_regime_boxplot.png'))
    
    # 5. Per-head heatmap
    plot_per_head_heatmap(
        calibration, model_name,
        os.path.join(output_dir, f'{prefix}_head_heatmap.png'))
    
    print(f"[Viz] Dashboard complete! {5} figures saved to {output_dir}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Calibration Visualization Suite"
    )
    parser.add_argument(
        "--extraction", type=str, default=None,
        help="Path to a single extraction pickle file"
    )
    parser.add_argument(
        "--extraction_dir", type=str, default=None,
        help="Directory containing extraction pickle files"
    )
    parser.add_argument(
        "--model_name", type=str, default=None,
        help="Name for the model (used in titles)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save figures"
    )
    parser.add_argument(
        "--metric", type=str, default='gini',
        choices=['gini', 'entropy', 'top3_mass'],
        help="Primary concentration metric for single-metric plots"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Generate comparison visualizations across all extractions"
    )
    args = parser.parse_args()
    
    if args.extraction:
        generate_dashboard(
            args.extraction,
            model_name=args.model_name,
            output_dir=args.output_dir,
            metric=args.metric,
        )
    
    elif args.extraction_dir and args.compare:
        paths = sorted(glob.glob(os.path.join(args.extraction_dir, "*.pkl")))
        if not paths:
            print(f"No .pkl files found in {args.extraction_dir}")
            return
        
        # Generate individual dashboards
        all_results = []
        for path in paths:
            name = os.path.basename(path).replace('.pkl', '').replace('extraction_', '')
            data = load_extraction(path)
            paired = extract_paired_data(data)
            calibration = compute_calibration_score(paired)
            
            all_results.append({
                'model_name': name,
                'calibration': calibration,
                'paired_data': paired,
            })
            
            generate_dashboard(path, model_name=name,
                             output_dir=args.output_dir, metric=args.metric)
        
        # Comparison bar chart
        if len(all_results) > 1:
            out_dir = args.output_dir or os.path.join(args.extraction_dir, 'figures')
            os.makedirs(out_dir, exist_ok=True)
            plot_calibration_comparison_bars(
                all_results,
                os.path.join(out_dir, 'comparison_calibration_bars.png'))
    
    elif args.extraction_dir:
        # Single model in a directory
        paths = sorted(glob.glob(os.path.join(args.extraction_dir, "*.pkl")))
        for path in paths:
            name = os.path.basename(path).replace('.pkl', '').replace('extraction_', '')
            generate_dashboard(path, model_name=name,
                             output_dir=args.output_dir, metric=args.metric)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
