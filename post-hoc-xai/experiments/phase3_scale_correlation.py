"""Phase 3 — Large-scale attention-attribution correlation.

Runs the attention-vs-attribution correlation study over N_SCENARIOS,
with risk stratification and action-conditioned analysis.

════════════════════════════════════════════════════════════════════════
CONFIGURATION  ← change these to extend the study
════════════════════════════════════════════════════════════════════════
"""

# ── Core config ────────────────────────────────────────────────────────────────
MODEL_KEY   = "complete"   # override via --model arg
N_SCENARIOS = 50           # 50 for local; 500+ for cluster
N_IG_STEPS  = 50           # IG integration steps
# Methods to compute — add "sarfa" for cluster run
METHODS = ["vg", "ig"]

# Attention signals to correlate against — add "norm_weighted" after Phase 1d validates it
ATTENTION_SIGNALS = ["rollout"]

# ── Risk stratification ─────────────────────────────────────────────────────────
RISK_BUCKETS = {
    "calm":     (0.0, 0.2),
    "moderate": (0.2, 0.6),
    "high":     (0.6, 1.0),
}

# ── Action thresholds (calibrated from reward_attention 3,676-timestep dataset)
# accel: mean=-0.083, std=0.389; steering: mean=-0.124, std=0.208
# Each bucket captures ≈25% of timesteps at these thresholds
ACTION_THRESHOLDS = {
    "braking":      lambda a, s: float(a) < -0.3,
    "accelerating": lambda a, s: float(a) > 0.3,
    "steering":     lambda a, s: abs(float(s)) > 0.3 and abs(float(a)) <= 0.3,
    "neutral":      lambda a, s: abs(float(a)) <= 0.3 and abs(float(s)) <= 0.3,
}

# ── Memory management ───────────────────────────────────────────────────────────
# For IG on GTX 1660 Ti (6GB): keep chunk ≤ 40 timesteps.
# For cluster (24GB): increase to 200+.
OBS_CHUNK_SIZE = 40

# ════════════════════════════════════════════════════════════════════════
# Imports & paths
# ════════════════════════════════════════════════════════════════════════

import sys
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
_CBM  = _ROOT.parent / "cbm"

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_CBM))
sys.path.insert(0, str(_CBM / "V-Max"))

import importlib.util as _ilu

import posthoc_xai as xai
from posthoc_xai.utils.ig_baseline import BaselineAccumulator

# Load attention aggregation once at module level (not per-call)
_agg_spec = _ilu.spec_from_file_location(
    "attention_aggregation",
    _ROOT / "posthoc_xai" / "utils" / "attention_aggregation.py",
)
_agg = _ilu.module_from_spec(_agg_spec)
_agg_spec.loader.exec_module(_agg)
_aggregate_attention = _agg.aggregate_attention
from posthoc_xai.visualization.paper_figures import (
    set_paper_style, save_figure,
    plot_risk_stratified_correlation,
    plot_category_heatmap,
    plot_action_conditioned,
    plot_model_comparison,
    plot_correlation_distribution,
)

DATA_PATH = str(_CBM / "data" / "training.tfrecord")

MODELS = {
    "complete": "womd_sac_road_perceiver_complete_42",
    "minimal":  "womd_sac_road_perceiver_minimal_42",
}

CATS = ["sdc_trajectory", "other_agents", "roadgraph", "traffic_lights", "gps_path"]


# ════════════════════════════════════════════════════════════════════════
# Data extraction helpers
# ════════════════════════════════════════════════════════════════════════

def make_adapter(model):
    """Create and prepare a VMaxAdapter once — reuse across all scenarios.

    adapter.prepare() JIT-compiles the step function. Calling it once and
    reusing the adapter avoids re-compilation on every scenario.
    """
    from event_mining.integration.vmax_adapter import VMaxAdapter
    adapter = VMaxAdapter(store_raw_obs=True)
    adapter.prepare(model)
    return adapter


def run_scenario(adapter, model, scenario, scenario_id: int) -> dict | None:
    """Extract per-timestep (obs, attention, risk, action) for one scenario.

    Args:
        adapter: Pre-prepared VMaxAdapter (call make_adapter once in main).
        model:   Loaded ExplainableModel.
        scenario: Raw scenario from data_gen.
        scenario_id: Integer identifier.

    Returns a dict with numpy arrays, or None if extraction fails.
    """
    from reward_attention.risk_metrics import RiskComputer
    from reward_attention.config import AnalysisConfig

    sd = adapter.extract_scenario_data(model, scenario, scenario_id=str(scenario_id))
    if sd.total_steps == 0 or sd.raw_observations is None:
        return None

    cfg  = AnalysisConfig(n_scenarios=N_SCENARIOS)
    risk = RiskComputer.from_scenario_data(sd, cfg)

    return {
        "raw_obs":        np.array(sd.raw_observations),
        "collision_risk": np.array(risk.collision_risk),
        "safety_risk":    np.array(risk.safety_risk),
        "accel":          np.array(sd.ego_accel)    if sd.ego_accel    is not None else np.zeros(sd.total_steps),
        "steering":       np.array(sd.ego_steering) if sd.ego_steering is not None else np.zeros(sd.total_steps),
        "T":              sd.total_steps,
    }


def classify_actions(accel: np.ndarray, steering: np.ndarray) -> list[str]:
    labels = []
    for a, s in zip(accel, steering):
        for name, fn in ACTION_THRESHOLDS.items():
            if fn(a, s):
                labels.append(name); break
        else:
            labels.append("neutral")
    return labels


# ════════════════════════════════════════════════════════════════════════
# Attribution computation
# ════════════════════════════════════════════════════════════════════════

def compute_attention_signals(model, raw_obs: np.ndarray) -> dict[str, np.ndarray]:
    """Batched forward pass → all configured attention signals as (T, 5) category fractions."""
    import jax.numpy as jnp

    obs_batch = jnp.array(raw_obs)
    attn      = model.forward(obs_batch).attention
    signals   = {}

    if "rollout" in ATTENTION_SIGNALS:
        key     = "cross_attn_rollout" if "cross_attn_rollout" in attn else "cross_attn_avg"
        rollout = np.array(attn[key])   # (T, 16, 280)
        result  = np.zeros((raw_obs.shape[0], len(CATS)))
        for t in range(raw_obs.shape[0]):
            d = _aggregate_attention(rollout[t], "rollout")
            result[t] = [d[c] for c in CATS]
        signals["rollout"] = result

    if "norm_weighted" in ATTENTION_SIGNALS and "norm_weighted_attn" in attn:
        nw     = np.array(attn["norm_weighted_attn"])  # (T, 16, 280)
        result = np.zeros((raw_obs.shape[0], len(CATS)))
        for t in range(raw_obs.shape[0]):
            d = _aggregate_attention(nw[t], "mean")
            result[t] = [d[c] for c in CATS]
        signals["norm_weighted"] = result

    return signals


def compute_vg_batch(model, raw_obs: np.ndarray) -> np.ndarray:
    """Batched VG via jax.vmap(jax.grad) — all timesteps in one JIT call."""
    import jax, jax.numpy as jnp

    params, module, action_size = (
        model._policy_params, model._policy_module, model._action_size
    )
    obs_struct = model.observation_structure

    def scalar_fn(obs_1d):
        logits = module.apply(params, obs_1d[None, :])
        return jnp.sum(logits[0, :action_size])

    batched_grad = jax.vmap(jax.grad(scalar_fn))
    grads        = np.array(batched_grad(jnp.array(raw_obs)))  # (T, 1655)

    result = np.zeros((raw_obs.shape[0], len(CATS)))
    for t in range(raw_obs.shape[0]):
        abs_g = np.abs(grads[t])
        total = abs_g.sum() + 1e-10
        result[t] = [abs_g[s:e].sum() / total for _, (s, e) in zip(CATS, obs_struct.values())]
    return result   # (T, 5)


def compute_ig_batch(
    model, raw_obs: np.ndarray, baseline: np.ndarray, **_
) -> np.ndarray:
    """IG with validity-zeroed mean baseline — JIT-compiled once for all T timesteps.

    The previous implementation called ig() per timestep, which caused JAX to
    retrace and recompile the vmap on every call (different closure over 'observation').
    This version makes obs an explicit argument so JAX compiles once for shape (D,)
    and reuses the compiled function for all 80 timesteps.
    Expected speedup: ~80× (from ~10 min → ~30–60 s per scenario).
    """
    import jax, jax.numpy as jnp

    params      = model._policy_params
    module      = model._policy_module
    action_size = model._action_size
    obs_struct  = model.observation_structure
    T           = raw_obs.shape[0]

    baseline_jnp = jnp.array(baseline)
    alphas       = jnp.linspace(0.0, 1.0, N_IG_STEPS + 1)   # (n_steps+1,)

    # JIT once — baseline and alphas are traced constants (same across all calls)
    @jax.jit
    def ig_one(obs_1d: jnp.ndarray) -> jnp.ndarray:
        """Raw IG attribution for a single observation. Shape: (D,)."""
        def grad_at_alpha(alpha):
            interp = baseline_jnp + alpha * (obs_1d - baseline_jnp)
            def scalar_fn(x):
                logits = module.apply(params, x[None, :])
                return jnp.sum(logits[0, :action_size])
            return jax.grad(scalar_fn)(interp)

        path_grads = jax.vmap(grad_at_alpha)(alphas)          # (n_steps+1, D)
        interior   = jnp.sum(path_grads[1:-1], axis=0)
        avg_grads  = (
            path_grads[0] + 2.0 * interior + path_grads[-1]
        ) / (2.0 * N_IG_STEPS)
        return (obs_1d - baseline_jnp) * avg_grads            # (D,)

    obs_batch = jnp.array(raw_obs)   # (T, D)
    result    = np.zeros((T, len(CATS)), dtype=np.float32)

    for t in range(T):
        raw_attr = np.array(ig_one(obs_batch[t]))   # compiled once, reused 80x
        abs_g    = np.abs(raw_attr)
        total    = abs_g.sum() + 1e-10
        result[t] = [abs_g[s:e].sum() / total
                     for c, (s, e) in obs_struct.items() if c in CATS]

    return result   # (T, 5)


def compute_sarfa_batch(model, raw_obs: np.ndarray) -> np.ndarray:
    """Fully-batched SARFA — 6 forward passes total for all T timesteps.

    Uses sarfa_batch() which bypasses capture_intermediates and processes
    all timesteps per category in one JIT call each.  10-50x faster than
    iterating SARFA.__call__ over individual timesteps.
    """
    from posthoc_xai.methods.sarfa import sarfa_batch
    scores = sarfa_batch(model, raw_obs, perturbation_type="zero", target_action=0)
    # scores: (T, n_categories) already normalised — reorder to match CATS
    obs_struct = model.observation_structure
    cat_order  = list(obs_struct.keys())
    result     = np.zeros((raw_obs.shape[0], len(CATS)))
    for i, cat in enumerate(CATS):
        col = cat_order.index(cat)
        result[:, i] = scores[:, col]
    return result


METHOD_FNS = {
    "vg":    compute_vg_batch,
    "ig":    compute_ig_batch,
    "sarfa": compute_sarfa_batch,
}


# ════════════════════════════════════════════════════════════════════════
# Per-scenario correlation
# ════════════════════════════════════════════════════════════════════════

def correlate_series(a: np.ndarray, b: np.ndarray) -> dict:
    """Per-timestep Pearson ρ between two (T, 5) arrays + aggregated stats."""
    from scipy.stats import pearsonr

    T = a.shape[0]

    # Per-category ρ over time
    cat_corr = {}
    for i, cat in enumerate(CATS):
        av, bv = a[:, i], b[:, i]
        if av.std() < 1e-8 or bv.std() < 1e-8:
            cat_corr[cat] = {"r": 0.0, "p": 1.0, "constant": True}
        else:
            r, p = pearsonr(av, bv)
            cat_corr[cat] = {"r": float(r), "p": float(p), "constant": False}

    # Per-timestep Pearson ρ (5-dim vectors)
    ts_rs = []
    for t in range(T):
        av, bv = a[t], b[t]
        if av.std() < 1e-8 or bv.std() < 1e-8:
            ts_rs.append(np.nan)
        else:
            r, _ = pearsonr(av, bv)
            ts_rs.append(float(r))
    ts_rs = np.array(ts_rs)

    return {
        "cat_corr":   cat_corr,
        "mean_r":     float(np.nanmean(ts_rs)),
        "median_r":   float(np.nanmedian(ts_rs)),
        "pct_pos":    float((ts_rs > 0).mean()),
        "ts_rs":      ts_rs.tolist(),
    }


# ════════════════════════════════════════════════════════════════════════
# Per-scenario result JSON
# ════════════════════════════════════════════════════════════════════════

def save_scenario_result(
    scenario_id: int,
    attn: np.ndarray,
    method_results: dict[str, np.ndarray],
    collision_risk: np.ndarray,
    safety_risk: np.ndarray,
    accel: np.ndarray,
    steering: np.ndarray,
    out_dir: Path,
    attention_signal: str = "rollout",
):
    """Save all per-timestep data and correlations for one scenario."""
    action_labels = classify_actions(accel, steering)
    T             = attn.shape[0]

    corr_per_method = {}
    for mname, marr in method_results.items():
        corr_per_method[mname] = correlate_series(attn, marr)

    # Risk-stratified correlations
    risk_strat = {}
    for bucket, (lo, hi) in RISK_BUCKETS.items():
        mask = (collision_risk >= lo) & (collision_risk < hi)
        if mask.sum() < 5:
            risk_strat[bucket] = {"n": int(mask.sum())}
            continue
        bucket_data = {}
        for mname, marr in method_results.items():
            bucket_data[mname] = correlate_series(attn[mask], marr[mask])
        bucket_data["n"] = int(mask.sum())
        risk_strat[bucket] = bucket_data

    # Action-conditioned correlations
    action_strat = {}
    for action_type in ACTION_THRESHOLDS:
        mask = np.array([l == action_type for l in action_labels])
        if mask.sum() < 5:
            action_strat[action_type] = {"n": int(mask.sum())}
            continue
        action_data = {}
        for mname, marr in method_results.items():
            action_data[mname] = correlate_series(attn[mask], marr[mask])
        action_data["n"] = int(mask.sum())
        action_strat[action_type] = action_data

    result = {
        "scenario_id":      scenario_id,
        "attention_signal": attention_signal,
        "T":                T,
        "mean_risk":        float(collision_risk.mean()),
        "std_risk":         float(collision_risk.std()),
        "action_counts":    {k: int(sum(1 for l in action_labels if l == k))
                             for k in ACTION_THRESHOLDS},
        "overall_corr":     corr_per_method,
        "risk_strat":       risk_strat,
        "action_strat":     action_strat,
    }

    path = out_dir / f"scenario_{scenario_id:04d}_{attention_signal}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return result


# ════════════════════════════════════════════════════════════════════════
# Aggregation across scenarios → DataFrames for figure generation
# ════════════════════════════════════════════════════════════════════════

def load_all_results(out_dir: Path) -> list[dict]:
    results = []
    for p in sorted(out_dir.glob("scenario_*_*.json")):
        with open(p) as f:
            results.append(json.load(f))
    return results


def build_dataframes(results: list[dict], model_key: str) -> dict[str, pd.DataFrame]:
    """Convert per-scenario JSONs to tidy DataFrames for figure generation."""
    rows_risk   = []
    rows_action = []
    rows_overall = []
    rows_cat    = []

    for res in results:
        sid    = res["scenario_id"]
        signal = res.get("attention_signal", "rollout")

        # Overall
        for method, corr in res["overall_corr"].items():
            rows_overall.append({
                "model": model_key, "scenario_id": sid,
                "attention_signal": signal,
                "method": method, "pearson_r": corr["mean_r"],
            })
            for cat, cdata in corr["cat_corr"].items():
                rows_cat.append({
                    "model": model_key, "scenario_id": sid,
                    "attention_signal": signal,
                    "method": method, "category": cat,
                    "risk_bucket": "all",
                    "pearson_r": cdata["r"] if not cdata.get("constant") else np.nan,
                })

        # Risk-stratified
        for bucket, bdata in res["risk_strat"].items():
            if bdata.get("n", 0) < 5:
                continue
            for method, corr in bdata.items():
                if method == "n" or not isinstance(corr, dict):
                    continue
                rows_risk.append({
                    "model": model_key, "scenario_id": sid,
                    "attention_signal": signal,
                    "method": method, "risk_bucket": bucket,
                    "pearson_r": corr["mean_r"], "n": bdata["n"],
                })
                for cat, cdata in corr.get("cat_corr", {}).items():
                    rows_cat.append({
                        "model": model_key, "scenario_id": sid,
                        "attention_signal": signal,
                        "method": method, "category": cat,
                        "risk_bucket": bucket,
                        "pearson_r": cdata["r"] if not cdata.get("constant") else np.nan,
                    })

        # Action-conditioned
        for action_type, adata in res["action_strat"].items():
            if adata.get("n", 0) < 5:
                continue
            for method, corr in adata.items():
                if method == "n" or not isinstance(corr, dict):
                    continue
                rows_action.append({
                    "model": model_key, "scenario_id": sid,
                    "attention_signal": signal,
                    "method": method, "action_type": action_type,
                    "pearson_r": corr["mean_r"], "n": adata["n"],
                })

    return {
        "risk":    pd.DataFrame(rows_risk),
        "action":  pd.DataFrame(rows_action),
        "overall": pd.DataFrame(rows_overall),
        "cat":     pd.DataFrame(rows_cat),
    }


# ════════════════════════════════════════════════════════════════════════
# Figure generation
# ════════════════════════════════════════════════════════════════════════

def generate_figures(dfs: dict[str, pd.DataFrame], model_key: str, fig_dir: Path):
    """Generate all Phase 3 paper figures (PDF + PNG), one set per attention signal."""
    set_paper_style()

    risk_df = dfs["risk"]
    signals = (
        risk_df["attention_signal"].unique().tolist()
        if "attention_signal" in risk_df.columns and len(risk_df)
        else ["rollout"]
    )

    for signal in signals:
        def _filt(df, sig=signal):
            if "attention_signal" not in df.columns or not len(df):
                return df
            return df[df["attention_signal"] == sig]

        r_df  = _filt(dfs["risk"])
        a_df  = _filt(dfs["action"])
        c_df  = _filt(dfs["cat"])
        o_df  = _filt(dfs["overall"])
        sfx   = f"{model_key}_{signal}"

        methods = [m for m in METHODS if len(r_df) and m in r_df["method"].unique().tolist()]
        if not methods:
            methods = METHODS

        if len(r_df):
            fig = plot_risk_stratified_correlation(
                r_df, methods,
                title=f"Attention–attribution agreement  [{model_key}, {signal}]",
            )
            save_figure(fig, f"fig1_risk_stratified_{sfx}", fig_dir)

        if len(c_df):
            for method in methods:
                fig = plot_category_heatmap(
                    c_df[c_df["risk_bucket"] != "all"],
                    method=method,
                    title=f"Category-level ρ by risk  [{model_key}, {method}, {signal}]",
                )
                save_figure(fig, f"fig2_cat_heatmap_{sfx}_{method}", fig_dir)

        if len(a_df):
            fig = plot_action_conditioned(
                a_df, methods,
                title=f"Attention–attribution agreement by action  [{model_key}, {signal}]",
            )
            save_figure(fig, f"fig3_action_conditioned_{sfx}", fig_dir)

        if len(o_df):
            fig = plot_correlation_distribution(
                o_df, methods,
                title=f"Per-scenario ρ distribution  [{model_key}, {signal}]",
            )
            save_figure(fig, f"fig4_distribution_{sfx}", fig_dir)

    print(f"  Figures saved to {fig_dir}/")


# ════════════════════════════════════════════════════════════════════════
# Summary printout
# ════════════════════════════════════════════════════════════════════════

def print_summary(results: list[dict], model_key: str):
    from collections import defaultdict
    print(f"\n{'='*60}")
    print(f"PHASE 3 SUMMARY — {model_key.upper()}  ({len(results)} scenarios)")
    print(f"{'='*60}")

    by_signal: dict = defaultdict(list)
    for res in results:
        by_signal[res.get("attention_signal", "rollout")].append(res)

    for signal, s_results in sorted(by_signal.items()):
        print(f"\n  [attention signal: {signal}]  n={len(s_results)}")
        for method in METHODS:
            all_r = [res["overall_corr"][method]["mean_r"]
                     for res in s_results if method in res["overall_corr"]]
            if not all_r:
                continue
            print(f"\n    {method.upper()}:")
            print(f"      Overall: mean ρ={np.mean(all_r):.3f}, "
                  f"median={np.median(all_r):.3f}, "
                  f"ρ>0: {(np.array(all_r)>0).mean()*100:.0f}%")

            for bucket in ["calm", "moderate", "high"]:
                bucket_r = []
                for res in s_results:
                    b = res["risk_strat"].get(bucket, {})
                    if method in b and isinstance(b[method], dict):
                        bucket_r.append(b[method]["mean_r"])
                if bucket_r:
                    print(f"      {bucket:10}: mean ρ={np.mean(bucket_r):.3f} "
                          f"(n_scenarios={len(bucket_r)})")


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 3 — scale correlation")
    parser.add_argument("--model", choices=list(MODELS.keys()),
                        default=MODEL_KEY)
    parser.add_argument("--n-scenarios", type=int, default=N_SCENARIOS)
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Methods to run, e.g. --methods vg ig sarfa. "
                             "Overrides METHODS config constant.")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="IG chunk size (timesteps per GPU batch). "
                             "Overrides OBS_CHUNK_SIZE. Use 200+ on cluster.")
    parser.add_argument("--figures-only", action="store_true",
                        help="Skip computation, only re-generate figures from saved JSONs.")
    parser.add_argument("--attention-signals", nargs="+", default=None,
                        help="Attention signals to compute, e.g. --attention-signals rollout norm_weighted. "
                             "Default: rollout. Add norm_weighted after Phase 1d validation.")
    args = parser.parse_args()

    # CLI overrides beat module-level config constants
    global METHODS, OBS_CHUNK_SIZE, ATTENTION_SIGNALS
    if args.methods is not None:
        METHODS = args.methods
    if args.chunk_size is not None:
        OBS_CHUNK_SIZE = args.chunk_size
    if args.attention_signals is not None:
        ATTENTION_SIGNALS = args.attention_signals

    model_key   = args.model
    n_scenarios = args.n_scenarios
    model_dir   = str(_CBM / "runs_rlc" / MODELS[model_key])

    out_dir = _HERE / "phase3_results" / model_key
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Phase 3 — {model_key}  |  {n_scenarios} scenarios  |  methods: {METHODS}  |  signals: {ATTENTION_SIGNALS}  |  chunk: {OBS_CHUNK_SIZE}")
    print(f"Results: {out_dir}\n")

    # ── Figures-only mode ─────────────────────────────────────────────────
    if args.figures_only:
        results = load_all_results(out_dir)
        if not results:
            print("No results found. Run without --figures-only first.")
            return
        dfs = build_dataframes(results, model_key)
        generate_figures(dfs, model_key, fig_dir)
        print_summary(results, model_key)
        return

    # ── Load model ────────────────────────────────────────────────────────
    print(f"Loading model: {MODELS[model_key]}")
    model = xai.load_model(model_dir, data_path=DATA_PATH)
    print(f"  Model loaded. has_attention={model.has_attention}\n")

    # ── Create adapter once — avoids re-JIT on every scenario ───────────
    print("Preparing VMaxAdapter (JIT compiles step fn once)...")
    adapter  = make_adapter(model)
    print("  Adapter ready.\n")

    # ── Iterate scenarios ─────────────────────────────────────────────────
    data_gen   = model._loaded.data_gen
    acc        = BaselineAccumulator()
    done_count = 0

    for scenario_id in range(n_scenarios):
        pending_signals = [
            s for s in ATTENTION_SIGNALS
            if not (out_dir / f"scenario_{scenario_id:04d}_{s}.json").exists()
        ]
        if not pending_signals:
            print(f"  [skip] scenario {scenario_id:04d} — already done")
            done_count += 1
            try: next(data_gen)
            except StopIteration: break
            continue

        try:
            scenario = next(data_gen)
        except StopIteration:
            print("  Data generator exhausted."); break

        print(f"  Scenario {scenario_id:04d}...", end=" ", flush=True)

        # Extract observations + risk
        ep = run_scenario(adapter, model, scenario, scenario_id)
        if ep is None:
            print("SKIP (empty)")
            continue

        raw_obs        = ep["raw_obs"]
        collision_risk = ep["collision_risk"]

        # Update streaming baseline
        acc.update(raw_obs)
        baseline = acc.finalize()

        # Attention signals (single forward pass, returns all configured signals)
        attn_signals = compute_attention_signals(model, raw_obs)  # dict[signal → (T, 5)]

        # Attribution methods
        method_results = {}
        for mname in METHODS:
            if mname == "vg":
                method_results["vg"] = compute_vg_batch(model, raw_obs)
            elif mname == "ig":
                method_results["ig"] = compute_ig_batch(model, raw_obs, baseline)
            elif mname == "sarfa":
                method_results["sarfa"] = compute_sarfa_batch(model, raw_obs)

        # Save per pending attention signal
        for signal_name in pending_signals:
            if signal_name not in attn_signals:
                continue
            save_scenario_result(
                scenario_id, attn_signals[signal_name], method_results,
                collision_risk, ep["safety_risk"],
                ep["accel"], ep["steering"], out_dir,
                attention_signal=signal_name,
            )
        done_count += 1
        print(f"T={ep['T']}, risk={collision_risk.mean():.2f}")

    # ── Aggregate + figures ───────────────────────────────────────────────
    print(f"\nCompleted {done_count}/{n_scenarios} scenarios.")
    results = load_all_results(out_dir)
    dfs     = build_dataframes(results, model_key)

    generate_figures(dfs, model_key, fig_dir)
    print_summary(results, model_key)

    # Save combined DataFrame for cross-model comparison
    for name, df in dfs.items():
        df.to_csv(out_dir / f"df_{name}.csv", index=False)
    print(f"\nDone. See {out_dir}/figures/ for all PDFs.")


if __name__ == "__main__":
    main()
