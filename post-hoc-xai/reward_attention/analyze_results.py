"""Post-processing analysis for completed experiment runs.

Reads timestep_data.pkl and generates ALL visualizations + summary statistics.
Designed to run after run_experiment.py completes — no GPU needed.

Usage:
    # Analyze complete model 50-scenario run:
    python reward_attention/analyze_results.py \
        --pkl results/reward_attention/womd_sac_road_perceiver_complete_42/timestep_data.pkl

    # Compare two models side by side:
    python reward_attention/analyze_results.py \
        --pkl results/reward_attention/womd_sac_road_perceiver_complete_42/timestep_data.pkl \
        --compare results/reward_attention/womd_sac_road_perceiver_minimal_42/timestep_data.pkl \
        --compare-label minimal

    # Three-way comparison:
    python reward_attention/analyze_results.py \
        --pkl results/reward_attention/womd_sac_road_perceiver_complete_42/timestep_data.pkl \
        --compare results/reward_attention/womd_sac_road_perceiver_minimal_42/timestep_data.pkl \
        --compare results/reward_attention/womd_sac_road_perceiver_basic_42/timestep_data.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from scipy import stats

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

SCENARIO_COLORS = [
    "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
    "#A65628", "#F781BF", "#999999", "#66C2A5", "#FC8D62",
]
ATTN_COLS = ["attn_sdc", "attn_agents", "attn_roadgraph", "attn_lights", "attn_gps"]
RISK_COLS = ["collision_risk", "safety_risk", "behavior_risk"]
EVENT_COLORS = {"hazard_onset": "#FF6600", "near_miss": "#FFCC00",
                "evasive_steering": "#9900FF", "collision_imminent": "#FF0000"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_records(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def to_df(records) -> pd.DataFrame:
    # All fields present in pkl records
    base_fields = [
        "scenario_id", "timestep",
        "attn_sdc", "attn_agents", "attn_roadgraph", "attn_lights", "attn_gps",
        "attn_to_nearest", "attn_to_threat",
        "collision_risk", "safety_risk", "navigation_risk", "behavior_risk",
        "min_ttc", "accel", "steering", "ego_speed",
        "num_valid_agents", "nearest_agent_id", "threat_agent_id",
        "is_collision_step", "is_offroad_step",
    ]
    rows = []
    for r in records:
        row = {}
        for f in base_fields:
            row[f] = getattr(r, f, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)

def spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return np.nan, np.nan
    return stats.spearmanr(x, y)

def partial_spearman(x, y, z):
    """Partial Spearman correlation of x~y controlling for z."""
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    if len(x) < 10 or np.std(z) < 1e-10:
        # z is constant — partialling has no effect; fall back to raw
        return spearman(x, y)
    # Rank all three
    rx = stats.rankdata(x).astype(float)
    ry = stats.rankdata(y).astype(float)
    rz = stats.rankdata(z).astype(float)
    # Partial out z from x and y via linear regression residuals
    def resid(a, b):
        slope, intercept, *_ = stats.linregress(b, a)
        return a - (slope * b + intercept)
    return stats.spearmanr(resid(rx, rz), resid(ry, rz))

def per_scenario_rho(df, risk_col, attn_col):
    rows = []
    for sid, sub in df.groupby("scenario_id"):
        x = sub[risk_col].values
        y = sub[attn_col].values
        rho, p = spearman(x, y)
        if np.isfinite(rho):
            rows.append({"scenario_id": sid, "rho": rho, "p": p, "n": len(sub),
                         "risk_std": float(np.std(x)), "risk_mean": float(np.mean(x))})
    return pd.DataFrame(rows)

def fisher_summary(rho_p_pairs):
    """Fisher z-transform summary. Accepts list of (rho, p) tuples or plain rho values."""
    # Support both list-of-rho and list-of-(rho,p)
    if len(rho_p_pairs) == 0:
        return {"mean_rho": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n": 0, "sig_pct": np.nan}
    if isinstance(rho_p_pairs[0], (tuple, list)):
        rhos = np.array([r for r, p in rho_p_pairs if np.isfinite(r)])
        ps   = np.array([p for r, p in rho_p_pairs if np.isfinite(r)])
    else:
        # Legacy: plain rho list — sig_pct unavailable
        rhos = np.array([r for r in rho_p_pairs if np.isfinite(r)])
        ps   = np.full(len(rhos), np.nan)
    if len(rhos) == 0:
        return {"mean_rho": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "n": 0, "sig_pct": np.nan}
    z = np.arctanh(np.clip(rhos, -0.999, 0.999))
    mz = np.mean(z)
    se = np.std(z) / np.sqrt(len(z))
    sig_pct = float(100 * np.mean(ps < 0.05)) if np.all(np.isfinite(ps)) else np.nan
    return {
        "mean_rho": float(np.tanh(mz)),
        "ci_lower": float(np.tanh(mz - 1.96 * se)),
        "ci_upper": float(np.tanh(mz + 1.96 * se)),
        "n": len(rhos),
        "sig_pct": sig_pct,
    }


# ---------------------------------------------------------------------------
# 1. Scenario ranking table
# ---------------------------------------------------------------------------

def scenario_ranking_table(df, out_dir: Path, min_std: float = 0.2):
    """Rank all scenarios by risk variability and per-scenario ρ(risk, agents)."""
    rows = []
    for sid, sub in df.groupby("scenario_id"):
        risk = sub["collision_risk"].values
        agents = sub["attn_agents"].values
        rho, p = spearman(risk, agents)
        rows.append({
            "scenario_id": int(sid),
            "n_steps": len(sub),
            "risk_std": float(np.std(risk)),
            "risk_mean": float(np.mean(risk)),
            "risk_min": float(np.min(risk)),
            "risk_max": float(np.max(risk)),
            "rho_risk_agents": round(rho, 3) if np.isfinite(rho) else np.nan,
            "p_value": round(p, 4) if np.isfinite(p) else np.nan,
            "high_variation": bool(np.std(risk) > min_std),
            "has_calm_phase": bool(np.min(risk) < 0.2),
            "has_danger_phase": bool(np.max(risk) > 0.7),
        })

    ranking = pd.DataFrame(rows).sort_values("risk_std", ascending=False)
    csv_path = out_dir / "scenario_ranking.csv"
    ranking.to_csv(csv_path, index=False)
    print(f"\n[Scenario Ranking] saved to {csv_path}")
    print(ranking[["scenario_id","n_steps","risk_std","risk_mean","rho_risk_agents",
                   "p_value","high_variation","has_calm_phase"]].to_string(index=False))
    return ranking


# ---------------------------------------------------------------------------
# 2. Full within-scenario correlation summary
# ---------------------------------------------------------------------------

def within_scenario_summary(df, out_dir: Path, min_std: float = 0.2):
    """Fisher z aggregated within-episode ρ for all risk×attn pairs."""
    pairs = [
        ("collision_risk",   "attn_agents",      "positive"),
        ("collision_risk",   "attn_roadgraph",   "negative"),
        ("collision_risk",   "attn_gps",         "negative"),
        ("collision_risk",   "attn_sdc",         "unclear"),
        ("collision_risk",   "attn_to_threat",   "positive"),
        ("collision_risk",   "attn_to_nearest",  "positive"),
        ("navigation_risk",  "attn_gps",         "positive"),
        ("safety_risk",      "attn_agents",      "positive"),
        ("behavior_risk",    "attn_roadgraph",   "unclear"),
    ]

    rows = []
    for risk_col, attn_col, direction in pairs:
        if risk_col not in df.columns or attn_col not in df.columns:
            continue
        per_sc = per_scenario_rho(df, risk_col, attn_col)
        if per_sc.empty:
            continue
        # All scenarios — pass (rho, p) pairs for correct sig_pct
        all_pairs = list(zip(per_sc["rho"].values, per_sc["p"].values))
        all_s = fisher_summary(all_pairs)
        # High-variation only
        hv = per_sc[per_sc["risk_std"] > min_std]
        hv_pairs = list(zip(hv["rho"].values, hv["p"].values))
        hv_s = fisher_summary(hv_pairs)
        rows.append({
            "pair": f"{risk_col} × {attn_col}",
            "expected": direction,
            "n_all": all_s["n"],
            "mean_rho_all": all_s["mean_rho"],
            "ci_all": f"[{all_s['ci_lower']:+.3f}, {all_s['ci_upper']:+.3f}]",
            "n_hv": hv_s["n"],
            "mean_rho_hv": hv_s["mean_rho"],
            "ci_hv": f"[{hv_s['ci_lower']:+.3f}, {hv_s['ci_upper']:+.3f}]",
            "sig_pct_hv": hv_s["sig_pct"],
        })

    result = pd.DataFrame(rows)
    csv_path = out_dir / "within_scenario_summary.csv"
    result.to_csv(csv_path, index=False)
    print(f"\n[Within-Scenario Summary] saved to {csv_path}")
    print(result.to_string(index=False))
    return result


# ---------------------------------------------------------------------------
# 3. Scenario-colored scatter
# ---------------------------------------------------------------------------

def plot_scenario_scatter(df, out_dir: Path,
                          risk_col="collision_risk", attn_col="attn_agents"):
    scenario_ids = sorted(df["scenario_id"].unique())
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#111111")

    for i, sid in enumerate(scenario_ids):
        sub = df[df["scenario_id"] == sid]
        x, y = sub[risk_col].values, sub[attn_col].values
        rho, p = spearman(x, y)
        color = SCENARIO_COLORS[i % len(SCENARIO_COLORS)]
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        ax.scatter(x, y, color=color, alpha=0.4, s=18, zorder=3)
        if np.std(x) > 0.05:
            sl, ic, *_ = stats.linregress(x, y)
            xl = np.linspace(x.min(), x.max(), 80)
            ax.plot(xl, sl * xl + ic, color=color, linewidth=1.8, alpha=0.9, zorder=4,
                    label=f"s{sid:03d}  ρ={rho:+.2f}{sig}  n={len(sub)}")

    # Pooled
    rho_pool, _ = spearman(df[risk_col].values, df[attn_col].values)
    sl, ic, *_ = stats.linregress(df[risk_col].values, df[attn_col].values)
    xl = np.linspace(0, 1, 100)
    ax.plot(xl, sl * xl + ic, color="white", linewidth=1.5, linestyle="--",
            alpha=0.5, label=f"Pooled ρ={rho_pool:+.2f} (confounded)")

    ax.set_xlabel(f"{risk_col}", fontsize=11, color="#cccccc")
    ax.set_ylabel(f"{attn_col}", fontsize=11, color="#cccccc")
    ax.set_title(f"Within-Episode Attention Tracks Risk — {len(scenario_ids)} Scenarios\n"
                 f"Pooled ρ={rho_pool:+.2f} masks within-episode effects",
                 fontsize=11, color="white")
    ax.tick_params(colors="#888")
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    ax.legend(fontsize=7, facecolor="#1a1a1a", edgecolor="#444", labelcolor="white",
              loc="upper left", ncol=2)
    plt.tight_layout()
    path = out_dir / f"fig_scenario_scatter_{risk_col}_vs_{attn_col}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# 4. Top-N scenario timeseries
# ---------------------------------------------------------------------------

def plot_top_scenario_timeseries(df, ranking, out_dir: Path,
                                 top_n: int = 5, min_std: float = 0.2):
    events_path = _PROJECT_ROOT / "events" / "test_catalog.json"
    events_by_scenario = {}
    if events_path.exists():
        with open(events_path) as f:
            cat = json.load(f)
        for ev in cat["events"]:
            sid_str = ev["scenario_id"]
            try:
                sid_int = int(sid_str.replace("s", ""))
                events_by_scenario.setdefault(sid_int, []).append(ev)
            except ValueError:
                pass

    good = ranking[ranking["risk_std"] > min_std].head(top_n)
    print(f"\n[Top Scenarios] Generating timeseries for top {len(good)} scenarios "
          f"(risk_std > {min_std}):")

    for _, row in good.iterrows():
        sid = int(row["scenario_id"])
        sub = df[df["scenario_id"] == sid].sort_values("timestep")
        ts = sub["timestep"].values
        risk = sub["collision_risk"].values
        agents = sub["attn_agents"].values
        road = sub["attn_roadgraph"].values
        gps = sub["attn_gps"].values
        sdc = sub["attn_sdc"].values

        rho_ag, _ = spearman(risk, agents)
        rho_rd, _ = spearman(risk, road)

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        fig.patch.set_facecolor("#0d0d0d")

        evs = events_by_scenario.get(sid, [])
        for ax in axes:
            ax.set_facecolor("#111111")
            for ev in evs:
                ec = EVENT_COLORS.get(ev["event_type"], "#fff")
                ax.axvspan(ev["onset"], ev.get("offset", ev["onset"]), alpha=0.10, color=ec)
            ax.tick_params(colors="#888")
            for sp in ax.spines.values(): sp.set_edgecolor("#333")

        ax1 = axes[0]
        ax1.fill_between(ts, risk, alpha=0.3, color="#F44336")
        ax1.plot(ts, risk, color="#F44336", linewidth=2)
        ax1.set_ylabel("Collision Risk", color="#F44336", fontsize=10)
        ax1.set_ylim(-0.02, 1.05)

        ax2 = axes[1]
        ax2.plot(ts, agents, color="#F44336", linewidth=2,
                 label=f"Agents  ρ={rho_ag:+.3f}")
        ax2.plot(ts, road, color="#4CAF50", linewidth=2,
                 label=f"Road    ρ={rho_rd:+.3f}")
        ax2.plot(ts, gps, color="#9C27B0", linewidth=1.8, linestyle="--",
                 label=f"GPS     mean={gps.mean():.3f}")
        ax2.plot(ts, sdc, color="#2196F3", linewidth=1.5, linestyle=":",
                 label=f"Ego     mean={sdc.mean():.3f}")
        ax2.set_ylabel("Attention Fraction", color="#cccccc", fontsize=10)
        ax2.set_xlabel("Timestep", color="#cccccc", fontsize=10)
        ax2.legend(fontsize=8.5, facecolor="#1a1a1a", edgecolor="#444",
                   labelcolor="white", loc="upper right")

        fig.suptitle(
            f"s{sid:03d} — risk_std={row['risk_std']:.3f}  "
            f"ρ(risk,agents)={rho_ag:+.3f}  n={len(sub)} steps",
            color="white", fontsize=10,
        )
        plt.tight_layout()
        path = out_dir / f"fig_timeseries_s{sid:03d}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="#0d0d0d")
        plt.close(fig)
        print(f"  s{sid:03d}: risk_std={row['risk_std']:.3f}  "
              f"ρ(agents)={rho_ag:+.3f}  → {path.name}")


# ---------------------------------------------------------------------------
# 5. Risk-std distribution plot
# ---------------------------------------------------------------------------

def plot_risk_distribution(df, ranking, out_dir: Path):
    """Bar plot of risk_std per scenario — shows which scenarios are analyzable."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0d0d0d")

    # Left: risk_std per scenario
    ax = axes[0]
    ax.set_facecolor("#111111")
    sids = ranking["scenario_id"].values
    stds = ranking["risk_std"].values
    colors = ["#4CAF50" if s > 0.2 else "#F44336" for s in stds]
    bars = ax.bar([f"s{s:03d}" for s in sids[:30]], stds[:30], color=colors[:30], width=0.7)
    ax.axhline(0.2, color="white", linewidth=1.5, linestyle="--", alpha=0.7,
               label="min_std=0.2 threshold")
    ax.set_xlabel("Scenario", color="#cccccc", fontsize=9)
    ax.set_ylabel("Risk Std (collision_risk)", color="#cccccc", fontsize=9)
    ax.set_title("Risk Variability per Scenario\n(green = usable for correlation analysis)",
                 color="white", fontsize=10)
    ax.tick_params(colors="#888", labelsize=7, axis="x", rotation=90)
    ax.legend(fontsize=8, facecolor="#1a1a1a", edgecolor="#444", labelcolor="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#333")

    # Right: scatter of risk_std vs ρ(risk, agents)
    ax2 = axes[1]
    ax2.set_facecolor("#111111")
    valid = ranking.dropna(subset=["rho_risk_agents"])
    colors2 = ["#4CAF50" if s > 0.2 else "#888888" for s in valid["risk_std"]]
    ax2.scatter(valid["risk_std"], valid["rho_risk_agents"],
                c=colors2, s=40, alpha=0.8, zorder=3)
    ax2.axvline(0.2, color="white", linewidth=1.2, linestyle="--", alpha=0.6)
    ax2.axhline(0, color="#555", linewidth=1)
    ax2.set_xlabel("Risk Std", color="#cccccc", fontsize=10)
    ax2.set_ylabel("ρ(collision_risk, attn_agents)", color="#cccccc", fontsize=10)
    ax2.set_title("Risk Variability vs Correlation Strength\n"
                  "(green = above threshold)", color="white", fontsize=10)
    ax2.tick_params(colors="#888")
    for sp in ax2.spines.values(): sp.set_edgecolor("#333")

    plt.tight_layout()
    path = out_dir / "fig_risk_distribution.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# 6. Attention baseline comparison (multi-model)
# ---------------------------------------------------------------------------

def plot_attention_baselines(pkls: list[tuple[str, Path]], out_dir: Path):
    """Bar chart comparing attention baselines across models — the vigilance prior figure."""
    model_data = []
    for label, pkl_path in pkls:
        records = load_records(pkl_path)
        df = to_df(records)
        means = {col: df[col].mean() for col in ATTN_COLS}
        means["label"] = label
        model_data.append(means)

    if len(model_data) < 2:
        return

    cats = ["attn_sdc", "attn_agents", "attn_roadgraph", "attn_lights", "attn_gps"]
    cat_labels = ["Ego (SDC)", "Other Agents", "Road Graph", "Traffic Lights", "GPS Path"]
    x = np.arange(len(cats))
    width = 0.25 if len(model_data) == 3 else 0.35

    model_colors = {"complete": "#F44336", "minimal": "#FF9800", "basic": "#9C27B0"}
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#111111")

    for i, md in enumerate(model_data):
        label = md["label"]
        vals = [md[c] for c in cats]
        offset = (i - len(model_data) / 2 + 0.5) * width
        color = model_colors.get(label, SCENARIO_COLORS[i])
        bars = ax.bar(x + offset, vals, width, label=label.capitalize(),
                      color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=10, color="#cccccc")
    ax.set_ylabel("Mean Attention Fraction (episode average)", fontsize=10, color="#cccccc")
    ax.set_title("Attention Allocation Prior — Shaped by Reward Design\n"
                 "GPS gradient: Minimal > Complete > Basic | Agent baseline: Complete > Minimal",
                 fontsize=11, color="white")
    ax.legend(fontsize=10, facecolor="#1a1a1a", edgecolor="#444", labelcolor="white")
    ax.tick_params(colors="#888")
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    ax.set_ylim(0, 0.65)

    plt.tight_layout()
    path = out_dir / "fig_attention_baselines_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# 7. Temporal lead-lag analysis
# ---------------------------------------------------------------------------

def plot_lead_lag(df, out_dir: Path, risk_col="collision_risk",
                  attn_col="attn_agents", max_lag: int = 8, min_std: float = 0.2):
    """Cross-correlation at lags -max_lag…+max_lag.

    Positive lag = attention LEADS risk by that many steps (anticipatory).
    Negative lag = attention LAGS risk (reactive).
    """
    lag_range = range(-max_lag, max_lag + 1)
    hv_scenario_ids = [
        sid for sid, sub in df.groupby("scenario_id")
        if np.std(sub[risk_col].values) > min_std
    ]

    all_rhos = {lag: [] for lag in lag_range}
    for sid in hv_scenario_ids:
        sub = df[df["scenario_id"] == sid].sort_values("timestep")
        risk = sub[risk_col].values
        attn = sub[attn_col].values
        for lag in lag_range:
            if lag == 0:
                rho, _ = spearman(risk, attn)
            elif lag > 0:
                # attn leads: attn[:-lag] ~ risk[lag:]
                rho, _ = spearman(attn[:-lag], risk[lag:])
            else:
                # attn lags: attn[-lag:] ~ risk[:lag]
                rho, _ = spearman(attn[-lag:], risk[:lag])
            if np.isfinite(rho):
                all_rhos[lag].append(rho)

    lags = list(lag_range)
    means = [np.mean(all_rhos[l]) if all_rhos[l] else np.nan for l in lags]
    stds  = [np.std(all_rhos[l]) if all_rhos[l] else np.nan for l in lags]
    ns    = [len(all_rhos[l]) for l in lags]
    ses   = [s / np.sqrt(n) if n > 0 else np.nan for s, n in zip(stds, ns)]

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#111111")

    means_arr = np.array(means)
    ses_arr   = np.array(ses)
    ax.bar(lags, means_arr, color=["#4CAF50" if m > 0 else "#F44336" for m in means_arr],
           alpha=0.7, zorder=3)
    ax.errorbar(lags, means_arr, yerr=1.96 * ses_arr,
                fmt="none", color="white", capsize=4, linewidth=1.5, zorder=4)
    ax.axvline(0, color="white", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.axhline(0, color="#555", linewidth=1)

    zero_rho = means[lags.index(0)]
    ax.set_xlabel("Lag (steps) — positive = attention leads risk", color="#cccccc", fontsize=11)
    ax.set_ylabel(f"Mean Spearman ρ  ({risk_col} × {attn_col})", color="#cccccc", fontsize=11)
    ax.set_title(
        f"Temporal Lead-Lag: Does Attention Anticipate Risk?\n"
        f"Lag=0 (contemporaneous): ρ={zero_rho:+.3f} | "
        f"n={ns[lags.index(0)]} high-variation scenarios",
        color="white", fontsize=11,
    )
    ax.tick_params(colors="#888")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")

    plt.tight_layout()
    path = out_dir / f"fig_lead_lag_{risk_col}_vs_{attn_col}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    print(f"  Saved {path.name}")

    # Print summary
    best_lag = lags[int(np.nanargmax(means_arr))]
    print(f"  Lead-lag summary: best ρ={np.nanmax(means_arr):+.3f} at lag={best_lag:+d} steps "
          f"({'attention leads' if best_lag > 0 else 'attention lags' if best_lag < 0 else 'contemporaneous'})")
    return {l: m for l, m in zip(lags, means)}


# ---------------------------------------------------------------------------
# 8. ρ distribution histogram (heterogeneity visualization)
# ---------------------------------------------------------------------------

def plot_rho_distribution(df, out_dir: Path, risk_col="collision_risk",
                          attn_col="attn_agents", min_std: float = 0.2):
    """Histogram of per-scenario ρ values — shows heterogeneity across 50 scenarios."""
    per_sc = per_scenario_rho(df, risk_col, attn_col)
    hv = per_sc[per_sc["risk_std"] > min_std]
    lv = per_sc[per_sc["risk_std"] <= min_std]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#111111")

    bins = np.linspace(-1, 1, 21)
    ax.hist(hv["rho"].values, bins=bins, color="#4CAF50", alpha=0.75,
            label=f"High-variation (n={len(hv)}, std>0.2)", zorder=3)
    ax.hist(lv["rho"].values, bins=bins, color="#888888", alpha=0.55,
            label=f"Low-variation (n={len(lv)}, excluded)", zorder=2)

    mean_hv = hv["rho"].mean()
    ax.axvline(mean_hv, color="#FFEB3B", linewidth=2, linestyle="--",
               label=f"Mean ρ (HV) = {mean_hv:+.3f}")
    ax.axvline(0, color="white", linewidth=1, linestyle=":", alpha=0.6)

    # Shade counter-examples
    n_neg = (hv["rho"] < 0).sum()
    n_pos = (hv["rho"] > 0).sum()
    ax.text(0.02, 0.92, f"{n_neg}/{len(hv)} counter-examples (ρ<0)",
            transform=ax.transAxes, color="#F44336", fontsize=9)
    ax.text(0.60, 0.92, f"{n_pos}/{len(hv)} confirmatory (ρ>0)",
            transform=ax.transAxes, color="#4CAF50", fontsize=9)

    ax.set_xlabel(f"Per-scenario Spearman ρ  ({risk_col} × {attn_col})", color="#cccccc", fontsize=11)
    ax.set_ylabel("Number of scenarios", color="#cccccc", fontsize=11)
    ax.set_title(
        f"Heterogeneity of Within-Episode Correlations Across {df['scenario_id'].nunique()} Scenarios\n"
        f"Mean ρ = {mean_hv:+.3f} | std = {hv['rho'].std():.3f}",
        color="white", fontsize=11,
    )
    ax.legend(fontsize=9, facecolor="#1a1a1a", edgecolor="#444", labelcolor="white")
    ax.tick_params(colors="#888")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")

    plt.tight_layout()
    path = out_dir / f"fig_rho_distribution_{risk_col}_vs_{attn_col}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# 9. Agent-count confound analysis
# ---------------------------------------------------------------------------

def plot_agent_count_confound(df, out_dir: Path, min_std: float = 0.2):
    """Check if attn_agents ~ collision_risk is confounded by num_valid_agents.

    If the model attends more agents only because more agents are present (not
    because risk is higher), then partialling out num_valid_agents should reduce ρ
    substantially. If partial ρ stays high, the correlation is genuine attention shift.
    """
    if "num_valid_agents" not in df.columns:
        print("  [confound] num_valid_agents not in data — skipping")
        return

    hv_ids = [
        sid for sid, sub in df.groupby("scenario_id")
        if np.std(sub["collision_risk"].values) > min_std
    ]

    raw_rhos, partial_rhos, sids_out = [], [], []
    for sid in hv_ids:
        sub = df[df["scenario_id"] == sid].sort_values("timestep")
        risk  = sub["collision_risk"].values
        attn  = sub["attn_agents"].values
        nagts = sub["num_valid_agents"].values.astype(float)

        raw, _ = spearman(risk, attn)
        part, _ = partial_spearman(risk, attn, nagts)

        if np.isfinite(raw) and np.isfinite(part):
            raw_rhos.append(raw)
            partial_rhos.append(part)
            sids_out.append(sid)

    if not raw_rhos:
        return

    raw_rhos = np.array(raw_rhos)
    partial_rhos = np.array(partial_rhos)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0d0d0d")

    # Left: scatter of raw vs partial ρ per scenario
    ax = axes[0]
    ax.set_facecolor("#111111")
    colors = ["#4CAF50" if p > 0 else "#F44336" for p in partial_rhos]
    ax.scatter(raw_rhos, partial_rhos, c=colors, s=50, alpha=0.8, zorder=3)
    lim = max(abs(raw_rhos).max(), abs(partial_rhos).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "w--", linewidth=1, alpha=0.5, label="No change")
    ax.axhline(0, color="#555", linewidth=1)
    ax.axvline(0, color="#555", linewidth=1)
    ax.set_xlabel("Raw ρ (risk × attn_agents)", color="#cccccc", fontsize=10)
    ax.set_ylabel("Partial ρ (controlling for num_valid_agents)", color="#cccccc", fontsize=10)
    ax.set_title("Confound Check: Agent Count\nPoints above diagonal → count suppresses true correlation",
                 color="white", fontsize=10)
    ax.legend(fontsize=8, facecolor="#1a1a1a", edgecolor="#444", labelcolor="white")
    ax.tick_params(colors="#888")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")

    # Right: mean comparison bar
    ax2 = axes[1]
    ax2.set_facecolor("#111111")
    means = [raw_rhos.mean(), partial_rhos.mean()]
    ses   = [raw_rhos.std() / np.sqrt(len(raw_rhos)),
             partial_rhos.std() / np.sqrt(len(partial_rhos))]
    bars = ax2.bar(["Raw ρ", "Partial ρ\n(−agent count)"], means,
                   color=["#2196F3", "#FF9800"], alpha=0.85, width=0.4)
    ax2.errorbar([0, 1], means, yerr=[1.96 * s for s in ses],
                 fmt="none", color="white", capsize=6, linewidth=2)
    ax2.axhline(0, color="#555", linewidth=1)
    for bar, val in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:+.3f}", ha="center", va="bottom", color="white", fontsize=11)
    ax2.set_ylabel("Mean Spearman ρ", color="#cccccc", fontsize=11)
    ax2.set_title(f"Mean ρ Before and After\nControlling for Agent Count  (n={len(raw_rhos)} HV scenarios)",
                  color="white", fontsize=10)
    ax2.tick_params(colors="#888")
    for sp in ax2.spines.values():
        sp.set_edgecolor("#333")

    plt.tight_layout()
    path = out_dir / "fig_agent_count_confound.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    print(f"  Saved {path.name}")
    print(f"  Confound check: raw ρ={raw_rhos.mean():+.3f}  partial ρ={partial_rhos.mean():+.3f}  "
          f"Δ={partial_rhos.mean()-raw_rhos.mean():+.3f}")


# ---------------------------------------------------------------------------
# 10. Print terminal summary
# ---------------------------------------------------------------------------

def print_summary(df, ranking, within_summary):
    n_scenarios = df["scenario_id"].nunique()
    n_records = len(df)
    n_hv = (ranking["risk_std"] > 0.2).sum()

    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"  Total records    : {n_records:,}")
    print(f"  Scenarios        : {n_scenarios}")
    print(f"  High-variation   : {n_hv} / {n_scenarios} (risk_std > 0.2)")
    print(f"  Top scenario     : s{int(ranking.iloc[0]['scenario_id']):03d} "
          f"(risk_std={ranking.iloc[0]['risk_std']:.3f})")
    print()
    print("  Within-scenario ρ (high-variation, key results):")
    for _, row in within_summary.iterrows():
        if pd.isna(row["mean_rho_hv"]):
            continue
        sig = "**" if row.get("sig_pct_hv", 0) == 100 else (
              "*"  if row.get("sig_pct_hv", 0) >= 50 else "")
        print(f"    {row['pair']:<42s}  "
              f"ρ={row['mean_rho_hv']:+.3f}{sig}  "
              f"CI={row['ci_hv']}  "
              f"n={int(row['n_hv'])}  [{row['expected']}]")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Post-processing analysis from saved pkl")
    parser.add_argument("--pkl", required=True, help="Path to timestep_data.pkl")
    parser.add_argument("--compare", action="append", default=[],
                        help="Additional pkl paths to compare (can repeat)")
    parser.add_argument("--compare-label", action="append", default=[],
                        help="Labels for compare pkls (same order)")
    parser.add_argument("--top-n", type=int, default=8,
                        help="Number of top scenarios for timeseries")
    parser.add_argument("--min-std", type=float, default=0.2,
                        help="Minimum risk_std to include scenario in analysis")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: same dir as pkl)")
    args = parser.parse_args()

    pkl_path = Path(args.pkl)
    out_dir = Path(args.output) if args.output else pkl_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Infer model label from directory name
    main_label = pkl_path.parent.name.replace("womd_sac_road_perceiver_", "").replace("_42", "")

    print(f"\nLoading {pkl_path} ...")
    records = load_records(pkl_path)
    df = to_df(records)
    print(f"  {len(records):,} records, {df['scenario_id'].nunique()} scenarios")

    # 1. Scenario ranking
    ranking = scenario_ranking_table(df, out_dir, min_std=args.min_std)

    # 2. Within-scenario correlation summary
    within_summary = within_scenario_summary(df, out_dir, min_std=args.min_std)

    # 3. Scenario-colored scatter
    print("\n[Figures] Generating scatter plots ...")
    plot_scenario_scatter(df, out_dir, "collision_risk", "attn_agents")
    plot_scenario_scatter(df, out_dir, "collision_risk", "attn_roadgraph")

    # 4. Risk distribution plot
    plot_risk_distribution(df, ranking, out_dir)

    # 5. Top-N timeseries
    print(f"\n[Figures] Generating timeseries for top {args.top_n} scenarios ...")
    plot_top_scenario_timeseries(df, ranking, out_dir,
                                 top_n=args.top_n, min_std=args.min_std)

    # 6. Multi-model baseline comparison
    compare_pkls = [(main_label, pkl_path)]
    labels = args.compare_label or []
    for i, cmp_pkl in enumerate(args.compare):
        lbl = labels[i] if i < len(labels) else Path(cmp_pkl).parent.name.replace(
            "womd_sac_road_perceiver_", "").replace("_42", "")
        compare_pkls.append((lbl, Path(cmp_pkl)))

    if len(compare_pkls) > 1:
        print("\n[Figures] Generating attention baseline comparison ...")
        plot_attention_baselines(compare_pkls, out_dir)

    # 7. Temporal lead-lag
    print("\n[Figures] Generating temporal lead-lag ...")
    plot_lead_lag(df, out_dir, min_std=args.min_std)

    # 8. ρ distribution histogram
    print("\n[Figures] Generating ρ distribution histogram ...")
    plot_rho_distribution(df, out_dir, min_std=args.min_std)

    # 9. Agent-count confound check
    print("\n[Figures] Running agent-count confound check ...")
    plot_agent_count_confound(df, out_dir, min_std=args.min_std)

    # 10. Terminal summary
    print_summary(df, ranking, within_summary)
    print(f"\nAll outputs in: {out_dir}\n")


if __name__ == "__main__":
    main()
