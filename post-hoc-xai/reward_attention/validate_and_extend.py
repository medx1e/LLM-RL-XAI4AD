#!/usr/bin/env python3
"""
validate_and_extend.py — Pre-scale validation + new analyses for RLC 2026.

Run BEFORE launching 50-scenario experiments on minimal model.
All sections are CPU-only (no GPU needed). All data from existing pkl files.

Usage:
    cd /home/med1e/post-hoc-xai
    eval "$(~/anaconda3/bin/conda shell.bash hook)" && conda activate vmax
    export PYTHONPATH=/home/med1e/post-hoc-xai/V-Max:$PYTHONPATH

    python reward_attention/validate_and_extend.py --all
    python reward_attention/validate_and_extend.py --validate
    python reward_attention/validate_and_extend.py --entropy
    python reward_attention/validate_and_extend.py --leadlag
    python reward_attention/validate_and_extend.py --budget
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Paper-ready style
# ---------------------------------------------------------------------------

def set_paper_style():
    """Publication-quality matplotlib defaults (white background, clean fonts)."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "text.usetex": False,
    })

# Model colors — consistent across all figures
MODEL_COLORS = {
    "complete": "#2166AC",   # deep blue
    "minimal":  "#B2182B",   # deep red
    "basic":    "#7F7F7F",   # gray (de-emphasized)
}
MODEL_LINESTYLES = {
    "complete": "-",
    "minimal":  "--",
    "basic":    ":",
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results/reward_attention")
COMPLETE_PKL = RESULTS_DIR / "womd_sac_road_perceiver_complete_42" / "timestep_data.pkl"
MINIMAL_PKL  = RESULTS_DIR / "womd_sac_road_perceiver_minimal_42"  / "timestep_data.pkl"
BASIC_PKL    = RESULTS_DIR / "womd_sac_road_perceiver_basic_42"    / "timestep_data.pkl"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_scenario(records, scenario_id: int):
    """Return records for one scenario, sorted by timestep."""
    recs = [r for r in records if r.scenario_id == scenario_id]
    return sorted(recs, key=lambda r: r.timestep)


def get_timeseries(recs, field: str) -> np.ndarray:
    """Extract a numpy array of one field from sorted records."""
    return np.array([getattr(r, field) for r in recs])


def scenario_ids(records) -> list[int]:
    return sorted(set(r.scenario_id for r in records))


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: VALIDATION CHECKS
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# 1A: Attention budget invariant
# ---------------------------------------------------------------------------

def check_attention_budget(records, model_name: str) -> bool:
    """Verify attn_sdc + attn_agents + attn_roadgraph + attn_lights + attn_gps = 1.0."""
    violations = []
    sums = []
    for r in records:
        s = r.attn_sdc + r.attn_agents + r.attn_roadgraph + r.attn_lights + r.attn_gps
        sums.append(s)
        if abs(s - 1.0) > 0.01:
            violations.append((r.scenario_id, r.timestep, s))

    sums = np.array(sums)
    print(f"\n{'='*60}")
    print(f"CHECK 1A — ATTENTION BUDGET: {model_name}")
    print(f"{'='*60}")
    print(f"  Records:    {len(records)}")
    print(f"  Sum min:    {sums.min():.6f}")
    print(f"  Sum max:    {sums.max():.6f}")
    print(f"  Sum mean:   {sums.mean():.6f}")
    print(f"  Sum std:    {sums.std():.6f}")
    print(f"  Violations: {len(violations)} / {len(records)}")
    if violations:
        print(f"  First 5 violations:")
        for sid, t, s in violations[:5]:
            print(f"    scenario={sid}, t={t}, sum={s:.6f}")
    passed = len(violations) == 0
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


# ---------------------------------------------------------------------------
# 1B: Cross-model risk profile comparison
# ---------------------------------------------------------------------------

def compare_risk_profiles(complete_recs, minimal_recs, basic_recs=None):
    """Compare collision_risk timeseries across models for overlapping scenarios."""
    overlapping = sorted(set(scenario_ids(complete_recs)) & set(scenario_ids(minimal_recs)))
    if not overlapping:
        print("  No overlapping scenarios between complete and minimal.")
        return

    print(f"\n{'='*60}")
    print(f"CHECK 1B — CROSS-MODEL RISK PROFILES")
    print(f"{'='*60}")
    print(f"  Overlapping scenarios: {overlapping}")

    profile_ok = True
    for sid in overlapping:
        c_recs = get_scenario(complete_recs, sid)
        m_recs = get_scenario(minimal_recs, sid)
        min_len = min(len(c_recs), len(m_recs))
        if min_len == 0:
            print(f"  s{sid:03d}: no overlapping timesteps — SKIP")
            continue

        c_risk = get_timeseries(c_recs[:min_len], "collision_risk")
        m_risk = get_timeseries(m_recs[:min_len], "collision_risk")

        if np.std(c_risk) > 0.01 and np.std(m_risk) > 0.01:
            rho, p = stats.spearmanr(c_risk, m_risk)
        else:
            rho, p = float("nan"), float("nan")

        mad = np.mean(np.abs(c_risk - m_risk))

        print(f"\n  s{sid:03d} (n={min_len} common timesteps):")
        print(f"    Spearman rho(complete_risk, minimal_risk) = {rho:+.3f}  p={p:.4f}")
        print(f"    Mean |delta_risk| = {mad:.4f}")
        print(f"    Complete risk: mean={c_risk.mean():.3f} std={c_risk.std():.3f}")
        print(f"    Minimal  risk: mean={m_risk.mean():.3f} std={m_risk.std():.3f}")

        if rho > 0.8 and mad < 0.1:
            print(f"    -> GOOD: Risk profiles are very similar.")
        elif rho > 0.5:
            print(f"    -> MODERATE: Risk profiles correlate but differ in magnitude.")
        else:
            print(f"    -> WARNING: Risk profiles diverge substantially.")
            profile_ok = False

    # Generate figure
    _plot_risk_comparison(complete_recs, minimal_recs, basic_recs, overlapping)
    return profile_ok


def _plot_risk_comparison(complete_recs, minimal_recs, basic_recs, scenario_ids_list):
    """Publication-quality overlaid risk profiles per scenario."""
    set_paper_style()

    n_plots = len(scenario_ids_list)
    fig, axes = plt.subplots(n_plots, 1, figsize=(6.5, 2.2 * n_plots), sharex=False)
    if n_plots == 1:
        axes = [axes]

    for ax, sid in zip(axes, scenario_ids_list):
        c_recs = get_scenario(complete_recs, sid)
        m_recs = get_scenario(minimal_recs, sid)
        min_len = min(len(c_recs), len(m_recs))
        if min_len == 0:
            ax.set_title(f"s{sid:03d} -- no data")
            continue

        c_risk = get_timeseries(c_recs[:min_len], "collision_risk")
        m_risk = get_timeseries(m_recs[:min_len], "collision_risk")
        t = np.arange(min_len)

        ax.plot(t, c_risk, color=MODEL_COLORS["complete"], linewidth=1.4,
                label="Complete", linestyle=MODEL_LINESTYLES["complete"])
        ax.plot(t, m_risk, color=MODEL_COLORS["minimal"], linewidth=1.4,
                label="Minimal", linestyle=MODEL_LINESTYLES["minimal"])

        if basic_recs is not None:
            b_recs = get_scenario(basic_recs, sid)
            if len(b_recs) > 0:
                b_len = min(min_len, len(b_recs))
                b_risk = get_timeseries(b_recs[:b_len], "collision_risk")
                ax.plot(np.arange(b_len), b_risk, color=MODEL_COLORS["basic"],
                        linewidth=1.2, label="Basic", linestyle=MODEL_LINESTYLES["basic"])

        # Compute rho for annotation
        if np.std(c_risk) > 0.01 and np.std(m_risk) > 0.01:
            rho, _ = stats.spearmanr(c_risk, m_risk)
            ax.text(0.98, 0.92, f"$\\rho$ = {rho:+.2f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))

        ax.set_ylabel("Collision risk")
        ax.set_title(f"Scenario {sid:03d}", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        if sid == scenario_ids_list[0]:
            ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    axes[-1].set_xlabel("Timestep")
    fig.suptitle("Cross-Model Risk Profile Comparison", fontsize=11, y=1.01)
    plt.tight_layout()
    save_path = RESULTS_DIR / "fig_risk_profile_comparison.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n  Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# 1C: Vigilance gap replication
# ---------------------------------------------------------------------------

def check_vigilance_gap(complete_recs, minimal_recs, scenario_ids_list=(0, 2)):
    """Check if complete model maintains higher calm-phase agent attention."""
    print(f"\n{'='*60}")
    print(f"CHECK 1C — VIGILANCE GAP REPLICATION")
    print(f"{'='*60}")

    gap_confirmed = False

    for sid in scenario_ids_list:
        c_recs = get_scenario(complete_recs, sid)
        m_recs = get_scenario(minimal_recs, sid)
        min_len = min(len(c_recs), len(m_recs))
        if min_len < 10:
            print(f"  s{sid:03d}: too few overlapping timesteps ({min_len})")
            continue

        c_attn = get_timeseries(c_recs[:min_len], "attn_agents")
        m_attn = get_timeseries(m_recs[:min_len], "attn_agents")
        c_risk = get_timeseries(c_recs[:min_len], "collision_risk")
        m_risk = get_timeseries(m_recs[:min_len], "collision_risk")
        avg_risk = (c_risk + m_risk) / 2.0

        calm_mask = avg_risk < 0.2
        danger_mask = avg_risk > 0.5

        print(f"\n  s{sid:03d} (n={min_len}):")
        print(f"    Calm timesteps (risk<0.2):   {calm_mask.sum()}")
        print(f"    Danger timesteps (risk>0.5): {danger_mask.sum()}")

        if calm_mask.sum() >= 3:
            c_calm = c_attn[calm_mask].mean()
            m_calm = m_attn[calm_mask].mean()
            gap = c_calm - m_calm
            gap_pct = (gap / m_calm * 100) if m_calm > 0.001 else float("nan")
            print(f"    Calm-phase attn_agents:")
            print(f"      complete = {c_calm:.4f}")
            print(f"      minimal  = {m_calm:.4f}")
            print(f"      gap      = {gap:+.4f}  ({gap_pct:+.1f}%)")
            if gap > 0.01:
                print(f"    -> VIGILANCE GAP CONFIRMED in s{sid:03d}")
                gap_confirmed = True
            elif gap < -0.01:
                print(f"    -> REVERSED: minimal > complete during calm phases")
            else:
                print(f"    -> NEGLIGIBLE gap")
        else:
            print(f"    Not enough calm timesteps to test.")

        if danger_mask.sum() >= 3:
            print(f"    Danger-phase attn_agents:")
            print(f"      complete = {c_attn[danger_mask].mean():.4f}")
            print(f"      minimal  = {m_attn[danger_mask].mean():.4f}")

        c_gps = get_timeseries(c_recs[:min_len], "attn_gps")
        m_gps = get_timeseries(m_recs[:min_len], "attn_gps")
        print(f"    Episode mean attn_agents: complete={c_attn.mean():.4f}, "
              f"minimal={m_attn.mean():.4f}")
        print(f"    Episode mean attn_gps:    complete={c_gps.mean():.4f}, "
              f"minimal={m_gps.mean():.4f}")

    _plot_vigilance_comparison(complete_recs, minimal_recs, scenario_ids_list)
    return gap_confirmed


def _contiguous_regions(mask):
    """Yield (start, end) of contiguous True regions in a boolean array."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return zip(starts, ends)


def _plot_vigilance_comparison(complete_recs, minimal_recs, scenario_ids_list):
    """Publication-quality: attn_agents + collision_risk, two scenarios side-by-side."""
    set_paper_style()

    n_rows = len(scenario_ids_list)
    fig, axes = plt.subplots(n_rows, 2, figsize=(7, 2.8 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row, sid in enumerate(scenario_ids_list):
        c_recs = get_scenario(complete_recs, sid)
        m_recs = get_scenario(minimal_recs, sid)
        min_len = min(len(c_recs), len(m_recs))
        if min_len < 5:
            continue

        c_attn = get_timeseries(c_recs[:min_len], "attn_agents")
        m_attn = get_timeseries(m_recs[:min_len], "attn_agents")
        c_risk = get_timeseries(c_recs[:min_len], "collision_risk")
        m_risk = get_timeseries(m_recs[:min_len], "collision_risk")
        t = np.arange(min_len)

        # Left panel: agent attention
        ax = axes[row, 0]
        ax.plot(t, c_attn, color=MODEL_COLORS["complete"], linewidth=1.4,
                label="Complete (TTC)")
        ax.plot(t, m_attn, color=MODEL_COLORS["minimal"], linewidth=1.4,
                linestyle="--", label="Minimal (no TTC)")

        # Shade calm phases
        calm = c_risk < 0.2
        first_calm = True
        for start, end in _contiguous_regions(calm):
            lbl = "Calm phase" if first_calm else None
            ax.axvspan(start, end, alpha=0.10, color="#4CAF50", label=lbl)
            first_calm = False

        # Fill the vigilance gap
        ax.fill_between(t, m_attn, c_attn, where=(c_attn > m_attn),
                        alpha=0.18, color=MODEL_COLORS["complete"], label="Vigilance gap")

        ax.set_ylabel("Agent attention", fontsize=9)
        ax.set_title(f"Scenario {sid:03d}", fontsize=10)
        if row == 0:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.9)

        # Right panel: collision risk
        ax = axes[row, 1]
        ax.plot(t, c_risk, color=MODEL_COLORS["complete"], linewidth=1.4, label="Complete")
        ax.plot(t, m_risk, color=MODEL_COLORS["minimal"], linewidth=1.4,
                linestyle="--", label="Minimal")
        ax.set_ylabel("Collision risk", fontsize=9)
        ax.set_title(f"Scenario {sid:03d}", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        if row == 0:
            ax.legend(fontsize=7)

    for ax in axes[-1]:
        ax.set_xlabel("Timestep", fontsize=9)

    fig.suptitle("Vigilance Gap: Complete vs. Minimal Agent Attention", fontsize=11, y=1.01)
    plt.tight_layout()
    save_path = RESULTS_DIR / "fig_vigilance_gap_s000_s002.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n  Saved: {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Validation summary
# ---------------------------------------------------------------------------

def validation_summary(budget_ok: bool, risk_ok, vigilance_ok: bool):
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Attention budget sums to 1.0:    {'PASS' if budget_ok else 'FAIL'}")
    print(f"  Cross-model risk profiles:       {'reviewed' if risk_ok else 'needs investigation'}")
    print(f"  Vigilance gap replicates:        {'YES' if vigilance_ok else 'UNCLEAR -- check figure'}")
    print()
    if budget_ok:
        print("  -> Safe to proceed with 50-scenario runs.")
    else:
        print("  -> FIX THE BUG before running anything else.")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: ATTENTION ENTROPY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def compute_category_entropy(records) -> np.ndarray:
    """Shannon entropy of the 5-category attention distribution (bits).

    Max possible = log2(5) = 2.322 bits (uniform over 5 categories).
    """
    entropies = []
    for r in records:
        p = np.array([r.attn_sdc, r.attn_agents, r.attn_roadgraph,
                       r.attn_lights, r.attn_gps])
        p = np.clip(p, 1e-10, None)
        p = p / p.sum()
        H = -np.sum(p * np.log2(p))
        entropies.append(H)
    return np.array(entropies)


def entropy_analysis(records, model_name: str):
    """Full entropy analysis: correlation with risk, phase comparison, figures."""
    entropies = compute_category_entropy(records)
    sids = scenario_ids(records)

    print(f"\n{'='*60}")
    print(f"ENTROPY ANALYSIS: {model_name}")
    print(f"{'='*60}")
    print(f"  Global entropy: mean={entropies.mean():.4f}, "
          f"std={entropies.std():.4f}, "
          f"range=[{entropies.min():.4f}, {entropies.max():.4f}]")
    print(f"  Max possible (5 categories): {np.log2(5):.4f} bits")

    # Within-episode correlation
    rhos = []
    rho_details = []
    for sid in sids:
        idx = [i for i, r in enumerate(records) if r.scenario_id == sid]
        if len(idx) < 10:
            continue
        risk = np.array([records[i].collision_risk for i in idx])
        ent = entropies[np.array(idx)]
        if np.std(risk) < 0.2:
            continue
        rho, p = stats.spearmanr(risk, ent)
        rhos.append(rho)
        rho_details.append((sid, rho, p, len(idx)))
        sig = "**" if p < 0.05 else ""
        print(f"  s{sid:03d}: entropy x risk rho={rho:+.3f} (p={p:.4f}) {sig}")

    if rhos:
        mean_rho = np.mean(rhos)
        print(f"\n  Mean within-episode rho(entropy, collision_risk) = {mean_rho:+.3f}")
        print(f"  n={len(rhos)} high-variation scenarios")
        if mean_rho < -0.1:
            print(f"  -> HYPOTHESIS SUPPORTED: Entropy drops when risk rises.")
        elif mean_rho > 0.1:
            print(f"  -> HYPOTHESIS REJECTED: Entropy rises with risk.")
        else:
            print(f"  -> INCONCLUSIVE: No clear entropy-risk relationship.")

    # Calm vs danger phase comparison
    calm_idx = [i for i, r in enumerate(records) if r.collision_risk < 0.2]
    danger_idx = [i for i, r in enumerate(records) if r.collision_risk > 0.5]
    if len(calm_idx) > 5 and len(danger_idx) > 5:
        calm_ent = entropies[np.array(calm_idx)]
        danger_ent = entropies[np.array(danger_idx)]
        print(f"\n  Phase comparison:")
        print(f"    Calm (risk<0.2):   H={calm_ent.mean():.4f} +/- {calm_ent.std():.4f}  (n={len(calm_ent)})")
        print(f"    Danger (risk>0.5): H={danger_ent.mean():.4f} +/- {danger_ent.std():.4f}  (n={len(danger_ent)})")
        u_stat, u_p = stats.mannwhitneyu(calm_ent, danger_ent, alternative="two-sided")
        print(f"    Mann-Whitney U: p={u_p:.4f} {'(significant)' if u_p < 0.05 else '(n.s.)'}")

    # Generate figures
    _plot_entropy_scatter(records, entropies, model_name)
    _plot_entropy_timeseries(records, entropies, model_name)

    return entropies


def _plot_entropy_scatter(records, entropies, model_name):
    """Scatter: collision_risk vs. entropy, colored by scenario."""
    set_paper_style()

    fig, ax = plt.subplots(figsize=(5.5, 4))
    risks = np.array([r.collision_risk for r in records])
    sids_arr = np.array([r.scenario_id for r in records])

    # Use a qualitative colormap for scenarios
    unique_sids = sorted(set(sids_arr))
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(max(10, len(unique_sids)))

    for i, sid in enumerate(unique_sids):
        mask = sids_arr == sid
        r_std = np.std(risks[mask])
        # Only label high-variation scenarios
        if r_std > 0.2:
            ax.scatter(risks[mask], entropies[mask], alpha=0.35, s=12,
                       color=cmap(i % 10), label=f"s{sid:03d}", zorder=3)
        else:
            ax.scatter(risks[mask], entropies[mask], alpha=0.15, s=8,
                       color="#BBBBBB", zorder=2)

    # Overall trend line
    if np.std(risks) > 0.01:
        slope, intercept, r_val, p_val, _ = stats.linregress(risks, entropies)
        x_fit = np.linspace(0, 1, 100)
        ax.plot(x_fit, slope * x_fit + intercept, "k-", linewidth=1.8,
                label=f"Trend: $r$={r_val:.3f}, $p$={p_val:.1e}", zorder=5)

    ax.set_xlabel("Collision risk")
    ax.set_ylabel("Attention entropy (bits)")
    ax.set_title(f"Attention concentration vs. risk ({model_name})")
    ax.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.9)
    plt.tight_layout()
    save_path = RESULTS_DIR / f"fig_entropy_scatter_{model_name}.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


def _plot_entropy_timeseries(records, entropies, model_name, top_n: int = 3):
    """Timeseries: entropy + collision_risk overlaid for top variable scenarios."""
    set_paper_style()

    sids = sorted(set(r.scenario_id for r in records))
    ranked = []
    for sid in sids:
        idx = [i for i, r in enumerate(records) if r.scenario_id == sid]
        risk_std = np.std([records[i].collision_risk for i in idx])
        ranked.append((sid, risk_std, idx))
    ranked.sort(key=lambda x: -x[1])

    n_plot = min(top_n, len(ranked))
    fig, axes = plt.subplots(n_plot, 1, figsize=(6.5, 2.5 * n_plot))
    if n_plot == 1:
        axes = [axes]

    for ax, (sid, risk_std, idx) in zip(axes, ranked[:n_plot]):
        t = np.array([records[i].timestep for i in idx])
        risk = np.array([records[i].collision_risk for i in idx])
        ent = entropies[np.array(idx)]

        order = np.argsort(t)
        t, risk, ent = t[order], risk[order], ent[order]

        color_ent = "#2166AC"
        color_risk = "#B2182B"

        ax.plot(t, ent, color=color_ent, linewidth=1.4, label="Entropy")
        ax.set_ylabel("Entropy (bits)", color=color_ent, fontsize=9)
        ax.tick_params(axis="y", labelcolor=color_ent)

        ax2 = ax.twinx()
        ax2.plot(t, risk, color=color_risk, linewidth=1.2, alpha=0.7, label="Collision risk")
        ax2.set_ylabel("Collision risk", color=color_risk, fontsize=9)
        ax2.tick_params(axis="y", labelcolor=color_risk)
        ax2.set_ylim(-0.05, 1.05)
        ax2.spines["right"].set_visible(True)

        # Compute within-episode rho
        if np.std(risk) > 0.01 and np.std(ent) > 0.01:
            rho, p = stats.spearmanr(risk, ent)
            ax.text(0.02, 0.92, f"$\\rho$ = {rho:+.2f}" + (" **" if p < 0.05 else ""),
                    transform=ax.transAxes, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))

        ax.set_title(f"s{sid:03d}  (risk std = {risk_std:.3f})", fontsize=9)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Timestep")
    fig.suptitle(f"Attention entropy vs. collision risk ({model_name})", fontsize=11, y=1.01)
    plt.tight_layout()
    save_path = RESULTS_DIR / f"fig_entropy_timeseries_{model_name}.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: PER-SCENARIO LEAD-LAG HISTOGRAM
# ═══════════════════════════════════════════════════════════════════════════

def leadlag_per_scenario(records, lag_range: int = 8, min_risk_std: float = 0.2):
    """Per-scenario best lag analysis.

    lag > 0 → attention LEADS risk (anticipatory)
    lag < 0 → attention LAGS risk (reactive)
    lag = 0 → simultaneous
    """
    sids = sorted(set(r.scenario_id for r in records))

    results = []
    all_lag_profiles = []

    print(f"\n{'='*60}")
    print(f"PER-SCENARIO LEAD-LAG ANALYSIS")
    print(f"{'='*60}")

    for sid in sids:
        recs = get_scenario(records, sid)
        if len(recs) < 2 * lag_range + 5:
            continue

        risk = get_timeseries(recs, "collision_risk")
        attn = get_timeseries(recs, "attn_agents")

        if np.std(risk) < min_risk_std:
            continue

        lag_rhos = []
        for lag in range(-lag_range, lag_range + 1):
            if lag > 0:
                a = attn[:len(attn) - lag]
                r = risk[lag:]
            elif lag < 0:
                a = attn[-lag:]
                r = risk[:len(risk) + lag]
            else:
                a = attn
                r = risk

            if len(a) < 10:
                lag_rhos.append(float("nan"))
                continue
            rho, _ = stats.spearmanr(a, r)
            lag_rhos.append(rho)

        lag_rhos = np.array(lag_rhos)
        all_lag_profiles.append(lag_rhos)

        valid = ~np.isnan(lag_rhos)
        if valid.any():
            best_idx = np.nanargmax(lag_rhos)
            best_lag = int(best_idx) - lag_range
            best_rho = lag_rhos[best_idx]
        else:
            best_lag, best_rho = 0, float("nan")

        results.append((sid, best_lag, best_rho, len(recs)))
        print(f"  s{sid:03d}: best_lag={best_lag:+d}, rho={best_rho:+.3f} (n={len(recs)})")

    if not results:
        print("  No high-variation scenarios found!")
        return results

    best_lags = [r[1] for r in results]
    print(f"\n  Summary (n={len(results)} HV scenarios):")
    print(f"    Mean best lag: {np.mean(best_lags):+.2f}")
    print(f"    Median best lag: {np.median(best_lags):+.1f}")
    print(f"    Std: {np.std(best_lags):.2f}")

    positive = sum(1 for l in best_lags if l > 0)
    zero = sum(1 for l in best_lags if l == 0)
    negative = sum(1 for l in best_lags if l < 0)
    cluster_1_3 = sum(1 for l in best_lags if 1 <= l <= 3)

    print(f"    Attention leads (lag>0): {positive}/{len(best_lags)} "
          f"({100*positive/len(best_lags):.0f}%)")
    print(f"    Simultaneous (lag=0):    {zero}/{len(best_lags)}")
    print(f"    Attention lags (lag<0):  {negative}/{len(best_lags)}")
    print(f"    In lag +1 to +3 range:   {cluster_1_3}/{len(best_lags)} "
          f"({100*cluster_1_3/len(best_lags):.0f}%)")

    if len(best_lags) > 0:
        if cluster_1_3 / len(best_lags) > 0.4:
            print(f"    -> ANTICIPATORY CLAIM SUPPORTED: majority cluster at +1 to +3.")
        elif positive / len(best_lags) > 0.5:
            print(f"    -> ATTENTION LEADS in most scenarios, but spread is wide.")
        else:
            print(f"    -> ANTICIPATORY CLAIM WEAK: no clear clustering.")

    _plot_leadlag_histogram(results, lag_range)
    if all_lag_profiles:
        _plot_leadlag_heatmap(all_lag_profiles, results, lag_range)

    return results


def _plot_leadlag_histogram(results, lag_range):
    """Publication-quality histogram of per-scenario best lags."""
    set_paper_style()

    best_lags = [r[1] for r in results]
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    bins = np.arange(-lag_range - 0.5, lag_range + 1.5, 1)
    counts, _, patches = ax.hist(best_lags, bins=bins, edgecolor="white",
                                  linewidth=0.5, alpha=0.85)

    # Color: green for leads, red for lags, gray for zero
    for patch, left_edge in zip(patches, bins[:-1]):
        center = left_edge + 0.5
        if center > 0:
            patch.set_facecolor("#4CAF50")
        elif center < 0:
            patch.set_facecolor("#E53935")
        else:
            patch.set_facecolor("#9E9E9E")

    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    median_lag = np.median(best_lags)
    ax.axvline(x=median_lag, color="#FF9800", linestyle="-", linewidth=2,
               label=f"Median = {median_lag:+.1f}")

    ax.set_xlabel("Best lag (positive = attention leads risk)")
    ax.set_ylabel("Number of scenarios")
    ax.set_title(f"Per-scenario optimal lead-lag\n($n$ = {len(best_lags)} high-variation scenarios)")
    ax.set_xticks(range(-lag_range, lag_range + 1))
    ax.legend(fontsize=8, framealpha=0.9)

    plt.tight_layout()
    save_path = RESULTS_DIR / "fig_leadlag_histogram.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n  Saved: {save_path}")
    plt.close()


def _plot_leadlag_heatmap(lag_profiles, results, lag_range):
    """Heatmap of correlation profiles across lags per scenario (appendix figure)."""
    set_paper_style()

    sorted_idx = sorted(range(len(results)), key=lambda i: results[i][1])
    matrix = np.array([lag_profiles[i] for i in sorted_idx])
    labels = [f"s{results[i][0]:03d}" for i in sorted_idx]
    lags = list(range(-lag_range, lag_range + 1))

    fig, ax = plt.subplots(figsize=(6.5, max(3, len(labels) * 0.28)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    ax.set_xticks(range(len(lags)))
    ax.set_xticklabels(lags, fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Lag (positive = attention leads)")
    ax.set_ylabel("Scenario")
    ax.set_title("Lead-lag correlation profile per scenario")

    cbar = plt.colorbar(im, ax=ax, label="Spearman $\\rho$", shrink=0.8)
    cbar.ax.tick_params(labelsize=8)

    # Mark best lag per scenario with a star
    for row, i in enumerate(sorted_idx):
        best_lag_idx = results[i][1] + lag_range
        if 0 <= best_lag_idx < len(lags):
            ax.plot(best_lag_idx, row, "k*", markersize=6)

    plt.tight_layout()
    save_path = RESULTS_DIR / "fig_leadlag_heatmap.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: BUDGET REALLOCATION (low-risk vs high-risk stacked bars)
# ═══════════════════════════════════════════════════════════════════════════

# Category order, labels, and colors — consistent with visualization.py
CATEGORY_KEYS   = ["attn_sdc", "attn_agents", "attn_roadgraph", "attn_lights", "attn_gps"]
CATEGORY_LABELS = ["Ego (SDC)", "Other Agents", "Road Graph", "Traffic Lights", "GPS Path"]
CATEGORY_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


def _get_hv_scenario_ids(records, min_std: float = 0.2) -> set[int]:
    """Return set of scenario IDs with collision_risk std > min_std."""
    from collections import defaultdict
    by_sc = defaultdict(list)
    for r in records:
        by_sc[r.scenario_id].append(r.collision_risk)
    return {sid for sid, risks in by_sc.items() if np.std(risks) > min_std}


def budget_reallocation(records, model_name: str,
                        low_thresh: float = 0.2, high_thresh: float = 0.7,
                        hv_only: bool = True):
    """Compute mean attention budget at low vs high risk.

    When hv_only=True, restricts to high-variation scenarios (risk_std > 0.2)
    to avoid dilution from near-constant-risk episodes.
    """
    if hv_only:
        hv_sids = _get_hv_scenario_ids(records)
        pool = [r for r in records if r.scenario_id in hv_sids]
        label_suffix = f" (HV scenarios only, n={len(hv_sids)})"
    else:
        pool = records
        label_suffix = ""

    low_recs  = [r for r in pool if r.collision_risk < low_thresh]
    high_recs = [r for r in pool if r.collision_risk > high_thresh]

    print(f"\n{'='*60}")
    print(f"BUDGET REALLOCATION: {model_name}{label_suffix}")
    print(f"{'='*60}")
    print(f"  Low-risk  (collision_risk < {low_thresh}): n = {len(low_recs)}")
    print(f"  High-risk (collision_risk > {high_thresh}): n = {len(high_recs)}")

    if len(low_recs) < 5 or len(high_recs) < 5:
        print("  Not enough timesteps in one or both phases — skipping.")
        return None

    low_means, high_means = {}, {}
    low_sems, high_sems = {}, {}
    for key in CATEGORY_KEYS:
        low_vals  = np.array([getattr(r, key) for r in low_recs])
        high_vals = np.array([getattr(r, key) for r in high_recs])
        low_means[key]  = float(low_vals.mean())
        high_means[key] = float(high_vals.mean())
        low_sems[key]   = float(low_vals.std() / np.sqrt(len(low_vals)))
        high_sems[key]  = float(high_vals.std() / np.sqrt(len(high_vals)))

    print(f"\n  {'Category':<18s} {'Low risk':>10s} {'High risk':>10s} {'Delta':>10s} {'Rel %':>8s}")
    print(f"  {'-'*56}")
    for key, label in zip(CATEGORY_KEYS, CATEGORY_LABELS):
        lo, hi = low_means[key], high_means[key]
        delta = hi - lo
        rel = (delta / lo * 100) if lo > 0.001 else float("nan")
        print(f"  {label:<18s} {lo:>10.4f} {hi:>10.4f} {delta:>+10.4f} {rel:>+7.1f}%")

    # Effect size: Mann-Whitney for the main trade-off (agents)
    lo_agents  = np.array([r.attn_agents for r in low_recs])
    hi_agents  = np.array([r.attn_agents for r in high_recs])
    u_stat, u_p = stats.mannwhitneyu(lo_agents, hi_agents, alternative="two-sided")
    print(f"\n  Mann-Whitney U (agents low vs high): p = {u_p:.2e}")

    return {"low": low_means, "high": high_means,
            "low_sem": low_sems, "high_sem": high_sems,
            "n_low": len(low_recs), "n_high": len(high_recs)}


def _scenario_rho_table(records, low_thresh=0.2, high_thresh=0.7):
    """Compute per-scenario rho and phase counts. Returns sorted list."""
    from collections import defaultdict
    by_sc = defaultdict(list)
    for r in records:
        by_sc[r.scenario_id].append(r)

    rows = []
    for sid, recs in by_sc.items():
        risk = np.array([r.collision_risk for r in recs])
        agents = np.array([r.attn_agents for r in recs])
        if np.std(risk) < 0.2:
            continue
        n_lo = int(np.sum(risk < low_thresh))
        n_hi = int(np.sum(risk > high_thresh))
        if n_lo < 3 or n_hi < 3:
            continue
        rho, p = stats.spearmanr(risk, agents)
        rows.append({"sid": sid, "rho": rho, "p": p, "n": len(recs),
                      "n_lo": n_lo, "n_hi": n_hi, "risk_std": float(np.std(risk))})
    rows.sort(key=lambda r: r["rho"], reverse=True)
    return rows


def _compute_budget(records, low_thresh=0.2, high_thresh=0.7):
    """Return (low_means, high_means, n_lo, n_hi) dicts for one set of records."""
    lo = [r for r in records if r.collision_risk < low_thresh]
    hi = [r for r in records if r.collision_risk > high_thresh]
    lo_means, hi_means = {}, {}
    for key in CATEGORY_KEYS:
        lo_means[key] = float(np.mean([getattr(r, key) for r in lo])) if lo else 0.0
        hi_means[key] = float(np.mean([getattr(r, key) for r in hi])) if hi else 0.0
    return lo_means, hi_means, len(lo), len(hi)


def _draw_stacked_pair(ax, lo_means, hi_means, n_lo, n_hi, label,
                       rho, bar_width=0.7, annotate=True):
    """Draw a Low|High stacked bar pair on the given axes."""
    positions = np.array([0.0, 1.0])

    bottoms = np.zeros(2)
    for key, cat_label, color in zip(CATEGORY_KEYS, CATEGORY_LABELS, CATEGORY_COLORS):
        vals = np.array([lo_means[key], hi_means[key]])
        ax.bar(positions, vals, bar_width, bottom=bottoms, color=color,
               edgecolor="white", linewidth=0.5)
        # Label segments > 4%
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0.04:
                ax.text(positions[i], b + v / 2, f"{v:.0%}",
                        ha="center", va="center", fontsize=7,
                        color="white", fontweight="bold")
        bottoms += vals

    ax.set_xticks(positions)
    ax.set_xticklabels(["Low\nrisk", "High\nrisk"], fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.axhline(1.0, color="#DDDDDD", linewidth=0.5, zorder=0)

    # Sample sizes
    ax.text(0, -0.065, f"$n$={n_lo}", ha="center", va="top",
            fontsize=7, color="#666", transform=ax.get_xaxis_transform())
    ax.text(1, -0.065, f"$n$={n_hi}", ha="center", va="top",
            fontsize=7, color="#666", transform=ax.get_xaxis_transform())

    # Title with rho
    sig = "***" if abs(rho) > 0.001 else ""  # we already filtered for significance
    rho_color = "#2166AC" if rho > 0 else "#B2182B"
    ax.set_title(f"{label}\n$\\rho$ = {rho:+.2f}", fontsize=9, pad=6,
                 color=rho_color, fontweight="bold")

    if not annotate:
        return

    # Delta annotation for agents
    delta_agents = hi_means["attn_agents"] - lo_means["attn_agents"]
    if abs(delta_agents) > 0.002:
        # Arrow between the agent segments
        lo_bottom = lo_means["attn_sdc"]
        hi_bottom = hi_means["attn_sdc"]
        y_lo = lo_bottom + lo_means["attn_agents"] / 2
        y_hi = hi_bottom + hi_means["attn_agents"] / 2
        ax.annotate("",
                    xy=(1 - bar_width / 2 - 0.02, y_hi),
                    xytext=(0 + bar_width / 2 + 0.02, y_lo),
                    arrowprops=dict(arrowstyle="-|>", color=CATEGORY_COLORS[1],
                                    lw=1.4, shrinkA=1, shrinkB=1,
                                    connectionstyle="arc3,rad=0.18"))
        sign = "+" if delta_agents > 0 else ""
        x_mid = 0.5
        y_arr = max(y_lo, y_hi) + 0.02
        ax.text(x_mid, y_arr, f"{sign}{delta_agents:.1%}",
                ha="center", va="bottom", fontsize=7.5,
                color=CATEGORY_COLORS[1], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.12", fc="white",
                          ec=CATEGORY_COLORS[1], alpha=0.85, lw=0.5))


def budget_reallocation_figure(records, low_thresh=0.2, high_thresh=0.7,
                               n_positive=3, n_counter=2):
    """Per-scenario budget reallocation: top positive-rho + counter-examples.

    Top row: n_positive scenarios with strongest positive rho (clear agent shift).
    Bottom row: n_counter scenarios with strongest negative rho (inverted shift).
    Each scenario gets a Low|High stacked bar pair.
    """
    set_paper_style()
    from collections import defaultdict

    table = _scenario_rho_table(records, low_thresh, high_thresh)
    if not table:
        print("  No qualifying scenarios for per-scenario budget figure.")
        return

    # Select scenarios
    positive = table[:n_positive]
    negative = [r for r in reversed(table) if r["rho"] < 0][:n_counter]

    has_neg = len(negative) > 0
    n_rows = 2 if has_neg else 1
    n_cols = max(len(positive), len(negative)) if has_neg else len(positive)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(2.5 * n_cols + 0.8, 3.6 * n_rows + 0.6),
                              squeeze=False)

    # Group records by scenario
    by_sc = defaultdict(list)
    for r in records:
        by_sc[r.scenario_id].append(r)

    # --- Top row: positive rho scenarios ---
    for col, info in enumerate(positive):
        ax = axes[0, col]
        sc_recs = by_sc[info["sid"]]
        lo_m, hi_m, n_lo, n_hi = _compute_budget(sc_recs, low_thresh, high_thresh)
        _draw_stacked_pair(ax, lo_m, hi_m, n_lo, n_hi,
                           f"s{info['sid']:03d}", info["rho"])
        if col == 0:
            ax.set_ylabel("Attention fraction", fontsize=9)

    # Hide unused top-row axes
    for col in range(len(positive), n_cols):
        axes[0, col].set_visible(False)

    # Row label
    axes[0, 0].text(-0.45, 0.5, "Risk-reactive\n(typical)",
                     transform=axes[0, 0].transAxes, rotation=90,
                     ha="center", va="center", fontsize=9, fontweight="bold",
                     color="#2166AC")

    # --- Bottom row: counter-examples ---
    if has_neg:
        for col, info in enumerate(negative):
            ax = axes[1, col]
            sc_recs = by_sc[info["sid"]]
            lo_m, hi_m, n_lo, n_hi = _compute_budget(sc_recs, low_thresh, high_thresh)
            _draw_stacked_pair(ax, lo_m, hi_m, n_lo, n_hi,
                               f"s{info['sid']:03d}", info["rho"])
            if col == 0:
                ax.set_ylabel("Attention fraction", fontsize=9)

        # Hide unused bottom-row axes
        for col in range(len(negative), n_cols):
            axes[1, col].set_visible(False)

        axes[1, 0].text(-0.45, 0.5, "Counter-\nexamples",
                         transform=axes[1, 0].transAxes, rotation=90,
                         ha="center", va="center", fontsize=9, fontweight="bold",
                         color="#B2182B")

    # Build legend handles manually (axes handles may be fragmented)
    import matplotlib.patches as mpatches
    legend_handles = [mpatches.Patch(facecolor=CATEGORY_COLORS[i], edgecolor="white",
                                      label=CATEGORY_LABELS[i])
                      for i in range(len(CATEGORY_KEYS) - 1, -1, -1)]

    # Place legend in the empty bottom-right cell if available
    if has_neg and n_cols > len(negative):
        leg_ax = axes[1, n_cols - 1]
        leg_ax.set_visible(True)
        leg_ax.axis("off")
        leg_ax.legend(handles=legend_handles, loc="center",
                      fontsize=8.5, frameon=True, framealpha=0.9,
                      title="Token category", title_fontsize=9)
    else:
        fig.legend(handles=legend_handles, loc="upper right",
                   fontsize=8, frameon=True, framealpha=0.9)

    fig.suptitle("Per-scenario attention budget: low risk vs. high risk",
                 fontsize=11)

    plt.tight_layout(rect=[0.05, 0.01, 1, 0.96])
    save_path = RESULTS_DIR / "fig_budget_reallocation.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n  Saved: {save_path}")
    plt.close()

    # Print the scenario selections
    print(f"\n  Top-row (positive rho):")
    for info in positive:
        print(f"    s{info['sid']:03d}: rho={info['rho']:+.3f}  "
              f"n_lo={info['n_lo']} n_hi={info['n_hi']}")
    if negative:
        print(f"  Bottom-row (counter-examples):")
        for info in negative:
            print(f"    s{info['sid']:03d}: rho={info['rho']:+.3f}  "
                  f"n_lo={info['n_lo']} n_hi={info['n_hi']}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pre-scale validation + new analyses for RLC 2026."
    )
    parser.add_argument("--all", action="store_true",
                        help="Run all sections")
    parser.add_argument("--validate", action="store_true",
                        help="Section 1: budget check, risk profiles, vigilance gap")
    parser.add_argument("--entropy", action="store_true",
                        help="Section 2: attention entropy analysis")
    parser.add_argument("--leadlag", action="store_true",
                        help="Section 3: per-scenario lead-lag histogram")
    parser.add_argument("--budget", action="store_true",
                        help="Section 4: budget reallocation (low vs high risk stacked bars)")
    args = parser.parse_args()

    if args.all:
        args.validate = args.entropy = args.leadlag = args.budget = True

    if not any([args.validate, args.entropy, args.leadlag, args.budget]):
        print("Specify --all, --validate, --entropy, or --leadlag")
        return

    # Load data
    print("Loading complete model data...")
    if not COMPLETE_PKL.exists():
        print(f"  ERROR: {COMPLETE_PKL} not found!")
        return
    complete = load_pkl(COMPLETE_PKL)
    print(f"  Loaded {len(complete)} records, "
          f"{len(scenario_ids(complete))} scenarios")

    # Check attn_per_agent availability (for entropy granularity)
    has_per_agent = hasattr(complete[0], "attn_per_agent")
    per_agent_populated = False
    if has_per_agent:
        pa = complete[0].attn_per_agent
        per_agent_populated = pa is not None and any(v != 0.0 for v in pa)
    print(f"  attn_per_agent field: {'present' if has_per_agent else 'MISSING'}"
          f"{' (populated)' if per_agent_populated else ' (zeros/empty)' if has_per_agent else ''}")

    minimal = None
    if MINIMAL_PKL.exists():
        print("Loading minimal model data...")
        minimal = load_pkl(MINIMAL_PKL)
        print(f"  Loaded {len(minimal)} records, "
              f"{len(scenario_ids(minimal))} scenarios")
    else:
        print("  Minimal pkl not found -- cross-model checks will be skipped")

    # ── Section 1: Validation ──
    if args.validate:
        print(f"\n{'='*60}")
        print("SECTION 1: VALIDATION CHECKS")
        print(f"{'='*60}")

        budget_ok = check_attention_budget(complete, "complete")
        if minimal is not None:
            check_attention_budget(minimal, "minimal")

        risk_ok = True
        if minimal is not None:
            risk_ok = compare_risk_profiles(complete, minimal)

        vigilance_ok = False
        if minimal is not None:
            vigilance_ok = check_vigilance_gap(
                complete, minimal, scenario_ids_list=[0, 2]
            )

        validation_summary(budget_ok, risk_ok, vigilance_ok)

    # ── Section 2: Entropy ──
    if args.entropy:
        print(f"\n{'='*60}")
        print("SECTION 2: ATTENTION ENTROPY ANALYSIS")
        print(f"{'='*60}")
        entropy_analysis(complete, "complete")
        if minimal is not None:
            entropy_analysis(minimal, "minimal")

    # ── Section 3: Lead-Lag ──
    if args.leadlag:
        print(f"\n{'='*60}")
        print("SECTION 3: PER-SCENARIO LEAD-LAG")
        print(f"{'='*60}")
        leadlag_per_scenario(complete, lag_range=8, min_risk_std=0.2)

    # ── Section 4: Budget Reallocation ──
    if args.budget:
        print(f"\n{'='*60}")
        print("SECTION 4: BUDGET REALLOCATION")
        print(f"{'='*60}")
        budget_reallocation(complete, "complete")
        if minimal is not None:
            budget_reallocation(minimal, "minimal")
        budget_reallocation_figure(complete)

    print(f"\nAll figures saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
