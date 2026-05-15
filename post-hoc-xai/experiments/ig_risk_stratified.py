"""IG attribution stratified by collision risk level.

For each of N_SCENARIOS:
  - Run IG at every timestep (batched via lax.scan)
  - Assign event type: normal / elevated / high based on collision risk R
  - Save per-timestep rows to CSV

Output:
  experiments/ig_risk_results/ig_risk_timesteps.csv
  experiments/ig_risk_results/ig_risk_summary_table.csv
  experiments/ig_risk_results/ig_risk_bar_chart.pdf

Usage:
    conda activate vmax
    cd /path/to/post-hoc-xai
    python experiments/ig_risk_stratified.py --model complete --n-scenarios 50 --data-path /path/to/training.tfrecord

Quick feasibility check on 5 scenarios:
    python experiments/ig_risk_stratified.py --model complete --n-scenarios 5 --data-path /path/to/training.tfrecord
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
_CBM  = _ROOT.parent / "cbm"

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_CBM))
sys.path.insert(0, str(_CBM / "V-Max"))

import posthoc_xai as xai
from posthoc_xai.utils.ig_baseline import BaselineAccumulator

# ── Config ─────────────────────────────────────────────────────────────────────
N_SCENARIOS = 50
N_IG_STEPS  = 50
MODEL_KEY   = "complete"

MODELS = {
    "complete": "womd_sac_road_perceiver_complete_42",
    "minimal":  "womd_sac_road_perceiver_minimal_42",
}

CATS = ["sdc_trajectory", "other_agents", "roadgraph", "traffic_lights", "gps_path"]

RISK_THRESHOLDS = {
    "normal":   (0.0, 0.2),
    "elevated": (0.2, 0.7),
    "high":     (0.7, 1.0),
}

OUT_DIR = _HERE / "ig_risk_results"
OUT_DIR.mkdir(exist_ok=True)

# Module-level JIT cache (same pattern as phase3)
_ig_jit_cache: dict = {}


# ── IG computation (reused from phase3) ────────────────────────────────────────

def compute_ig_batch(model, raw_obs: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    import jax, jax.numpy as jnp

    params      = model._policy_params
    module      = model._policy_module
    action_size = model._action_size
    obs_struct  = model.observation_structure
    T           = raw_obs.shape[0]

    baseline_jnp = jnp.array(baseline)
    alphas       = jnp.linspace(0.0, 1.0, N_IG_STEPS + 1)

    cache_key = (id(params), N_IG_STEPS)
    if cache_key not in _ig_jit_cache:
        @jax.jit
        def _ig_scan(obs_batch: jnp.ndarray, bl: jnp.ndarray) -> jnp.ndarray:
            def ig_one_t(carry, obs_1d):
                def grad_at_alpha(alpha):
                    interp = bl + alpha * (obs_1d - bl)
                    def scalar_fn(x):
                        logits = module.apply(params, x[None, :])
                        return jnp.sum(logits[0, :action_size])
                    return jax.grad(scalar_fn)(interp)
                path_grads = jax.vmap(grad_at_alpha)(alphas)
                interior   = jnp.sum(path_grads[1:-1], axis=0)
                avg_grads  = (path_grads[0] + 2.0 * interior + path_grads[-1]) / (2.0 * N_IG_STEPS)
                return carry, (obs_1d - bl) * avg_grads
            _, all_attrs = jax.lax.scan(ig_one_t, None, obs_batch)
            return all_attrs
        _ig_jit_cache[cache_key] = _ig_scan

    all_attrs = np.array(_ig_jit_cache[cache_key](jnp.array(raw_obs), baseline_jnp))  # (T, D)
    result    = np.zeros((T, len(CATS)), dtype=np.float32)
    for t in range(T):
        abs_g     = np.abs(all_attrs[t])
        total     = abs_g.sum() + 1e-10
        result[t] = [abs_g[s:e].sum() / total for _, (s, e) in zip(CATS, obs_struct.values())]
    return result   # (T, 5)


# ── Scenario extraction (reused from phase3) ───────────────────────────────────

def make_adapter(model):
    from event_mining.integration.vmax_adapter import VMaxAdapter
    adapter = VMaxAdapter(store_raw_obs=True)
    adapter.prepare(model)
    return adapter


def run_scenario(adapter, model, scenario, scenario_id):
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
        "T":              sd.total_steps,
    }


def assign_event_type(risk_score: float) -> str:
    if risk_score < 0.2:
        return "normal"
    elif risk_score < 0.7:
        return "elevated"
    return "high"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        choices=list(MODELS.keys()), default=MODEL_KEY)
    parser.add_argument("--n-scenarios",  type=int, default=N_SCENARIOS)
    parser.add_argument("--data-path",    type=str, default=None,
                        help="Path to training.tfrecord")
    parser.add_argument("--runs-rlc",     type=str, default=None,
                        help="Path to runs_rlc/ directory containing model weights")
    args = parser.parse_args()

    data_path = args.data_path or str(_CBM / "data" / "training.tfrecord")
    runs_rlc  = Path(args.runs_rlc) if args.runs_rlc else (_CBM / "runs_rlc")
    model_dir = str(runs_rlc / MODELS[args.model])

    print(f"IG risk stratification | model={args.model} | n={args.n_scenarios} | IG_steps={N_IG_STEPS}")

    model   = xai.load_model(model_dir, data_path=data_path)
    adapter = make_adapter(model)
    data_gen = model._loaded.data_gen
    acc      = BaselineAccumulator()

    rows = []

    for scenario_id in range(args.n_scenarios):
        try:
            scenario = next(data_gen)
        except StopIteration:
            print("Data exhausted."); break

        print(f"  Scenario {scenario_id:04d}...", end=" ", flush=True)
        ep = run_scenario(adapter, model, scenario, scenario_id)
        if ep is None:
            print("SKIP"); continue

        raw_obs        = ep["raw_obs"]
        collision_risk = ep["collision_risk"]
        T              = ep["T"]

        acc.update(raw_obs)
        baseline = acc.finalize()

        ig_attrs = compute_ig_batch(model, raw_obs, baseline)   # (T, 5)

        for t in range(T):
            R          = float(collision_risk[t])
            event_type = assign_event_type(R)
            for c_idx, cat in enumerate(CATS):
                rows.append({
                    "scenario_id":  scenario_id,
                    "timestep":     t,
                    "R":            R,
                    "event_type":   event_type,
                    "category":     cat,
                    "ig_attribution": float(ig_attrs[t, c_idx]),
                })

        print(f"T={T}, mean_risk={collision_risk.mean():.2f}")

    # ── Save full CSV ──────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "ig_risk_timesteps.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}  ({len(df)} rows)")

    # ── Summary table (mean ± std per event_type × category) ──────────────────
    summary = df.groupby(["event_type", "category"])["ig_attribution"].agg(["mean", "std"]).reset_index()
    summary["mean_std"] = summary.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
    table = summary.pivot(index="event_type", columns="category", values="mean_std")
    table = table.loc[["normal", "elevated", "high"], CATS]  # enforce order
    table_path = OUT_DIR / "ig_risk_summary_table.csv"
    table.to_csv(table_path)
    print(f"Saved: {table_path}")
    print("\nSummary table (mean ± std):")
    print(table.to_string())

    # ── Figure: grouped bar chart ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    event_types = ["normal", "elevated", "high"]
    colors      = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    cat_labels  = ["SDC", "Agents", "Road", "TL", "GPS"]
    x           = np.arange(len(event_types))
    width       = 0.15
    offsets     = np.linspace(-2, 2, 5) * width

    for i, (cat, col, lbl, off) in enumerate(zip(CATS, colors, cat_labels, offsets)):
        means = [summary[(summary.event_type == et) & (summary.category == cat)]["mean"].values[0]
                 for et in event_types]
        stds  = [summary[(summary.event_type == et) & (summary.category == cat)]["std"].values[0]
                 for et in event_types]
        ax.bar(x + off, means, width, label=lbl, color=col, yerr=stds, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(["Normal\n(R<0.2)", "Elevated\n(0.2≤R<0.7)", "High\n(R≥0.7)"], fontsize=11)
    ax.set_ylabel("IG attribution fraction", fontsize=11)
    ax.set_title(f"IG Attribution by Risk Level — {args.model} model ({args.n_scenarios} scenarios)",
                 fontsize=11, fontweight="bold")
    ax.legend(title="Category", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.set_ylim(0, 0.75)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig_path = OUT_DIR / "ig_risk_bar_chart.pdf"
    fig.savefig(fig_path, format="pdf", bbox_inches="tight")
    fig.savefig(str(fig_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
