"""B2/B3 — All 7 XAI methods on high-risk vs normal timesteps.

Depends on B1 output: ig_risk_results/ig_risk_timesteps.csv
Loads R values and event-type labels from B1 — does NOT recompute risk.

Outputs (in experiments/b2b3_results/):
  b2b3_raw_attributions.csv          — per-timestep, per-method attribution vectors
  topcategory_highRisk.tex           — frequency table (LaTeX)
  topcategory_normal.tex             — frequency table (LaTeX)
  method_correlation_heatmap.pdf     — 7×7 Spearman heatmap
  sarfa_agents_table.tex             — agents attribution ratio table (LaTeX)

Usage:
    conda activate vmax
    cd /path/to/post-hoc-xai
    python experiments/b2b3_all_methods_risk.py \
        --model complete \
        --data-path /path/to/training.tfrecord \
        --runs-rlc /path/to/runs_rlc

One model per process (Waymax registry constraint).
"""

import sys
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
_CBM  = _ROOT.parent / "cbm"

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_CBM))
sys.path.insert(0, str(_CBM / "V-Max"))

import posthoc_xai as xai
from posthoc_xai.utils.ig_baseline import BaselineAccumulator

# ── Config ─────────────────────────────────────────────────────────────────────
N_IG_STEPS      = 50
N_SG_SAMPLES    = 5       # SmoothGrad noise samples (fewer = faster)
SG_NOISE_LEVEL  = 0.1     # fraction of feature std
HIGH_RISK_THRESH = 0.7
NORMAL_THRESH    = 0.2
FALLBACK_THRESH  = 0.5    # if fewer than 30 high-risk timesteps

MODELS = {
    "complete": "womd_sac_road_perceiver_complete_42",
    "minimal":  "womd_sac_road_perceiver_minimal_42",
}

CATS     = ["sdc_trajectory", "other_agents", "roadgraph", "traffic_lights", "gps_path"]
CAT_COLS = ["sdc_attr", "agents_attr", "roadgraph_attr", "lights_attr", "gps_attr"]
METHODS  = ["vg", "ig", "gxi", "smoothgrad", "perturbation", "feature_ablation", "sarfa"]
METHOD_LABELS = {
    "vg": "VG", "ig": "IG", "gxi": "GxI", "smoothgrad": "SmoothGrad",
    "perturbation": "Perturbation", "feature_ablation": "FeatureAblation", "sarfa": "SARFA",
}
PARADIGM_ORDER = ["vg", "gxi", "ig", "smoothgrad", "perturbation", "feature_ablation", "sarfa"]

B1_CSV  = _HERE / "ig_risk_results" / "ig_risk_timesteps.csv"
OUT_DIR = _HERE / "b2b3_results"
OUT_DIR.mkdir(exist_ok=True)

_ig_jit_cache: dict = {}
_vg_jit_cache: dict = {}


# ── Attribution methods ────────────────────────────────────────────────────────

def _get_raw_grads(model, raw_obs: np.ndarray) -> np.ndarray:
    """(T, D) raw signed gradients (not abs)."""
    import jax, jax.numpy as jnp
    params, module, action_size = model._policy_params, model._policy_module, model._action_size

    cache_key = (id(params), "raw_grad")
    if cache_key not in _vg_jit_cache:
        def scalar_fn(obs_1d):
            logits = module.apply(params, obs_1d[None, :])
            return jnp.sum(logits[0, :action_size])
        _vg_jit_cache[cache_key] = jax.jit(jax.vmap(jax.grad(scalar_fn)))

    return np.array(_vg_jit_cache[cache_key](jnp.array(raw_obs)))  # (T, D)


def _normalize_to_cats(abs_attr: np.ndarray, obs_struct: dict) -> np.ndarray:
    """(T, D) abs attribution → (T, 5) normalized category fractions."""
    T = abs_attr.shape[0]
    result = np.zeros((T, len(CATS)), dtype=np.float32)
    for t in range(T):
        total = abs_attr[t].sum() + 1e-10
        result[t] = [abs_attr[t, s:e].sum() / total for _, (s, e) in zip(CATS, obs_struct.values())]
    return result


def compute_vg(model, raw_obs: np.ndarray) -> np.ndarray:
    grads = _get_raw_grads(model, raw_obs)
    return _normalize_to_cats(np.abs(grads), model.observation_structure)


def compute_gxi(model, raw_obs: np.ndarray) -> np.ndarray:
    grads = _get_raw_grads(model, raw_obs)
    return _normalize_to_cats(np.abs(grads * raw_obs), model.observation_structure)


def compute_smoothgrad(model, raw_obs: np.ndarray) -> np.ndarray:
    import jax.numpy as jnp
    obs_std    = raw_obs.std(axis=0) + 1e-8
    accumulated = np.zeros_like(raw_obs)
    for _ in range(N_SG_SAMPLES):
        noise    = np.random.normal(0, SG_NOISE_LEVEL * obs_std, raw_obs.shape).astype(np.float32)
        noisy    = raw_obs + noise
        grads    = _get_raw_grads(model, noisy)
        accumulated += np.abs(grads)
    return _normalize_to_cats(accumulated / N_SG_SAMPLES, model.observation_structure)


def compute_ig(model, raw_obs: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    import jax, jax.numpy as jnp
    params, module, action_size = model._policy_params, model._policy_module, model._action_size
    obs_struct = model.observation_structure
    T          = raw_obs.shape[0]
    bl         = jnp.array(baseline)
    alphas     = jnp.linspace(0.0, 1.0, N_IG_STEPS + 1)

    cache_key = (id(params), N_IG_STEPS)
    if cache_key not in _ig_jit_cache:
        @jax.jit
        def _ig_scan(obs_batch, bl_arg):
            def ig_one_t(carry, obs_1d):
                def grad_at_alpha(alpha):
                    interp = bl_arg + alpha * (obs_1d - bl_arg)
                    def scalar_fn(x):
                        logits = module.apply(params, x[None, :])
                        return jnp.sum(logits[0, :action_size])
                    return jax.grad(scalar_fn)(interp)
                path_grads = jax.vmap(grad_at_alpha)(alphas)
                interior   = jnp.sum(path_grads[1:-1], axis=0)
                avg_grads  = (path_grads[0] + 2.0 * interior + path_grads[-1]) / (2.0 * N_IG_STEPS)
                return carry, (obs_1d - bl_arg) * avg_grads
            _, all_attrs = jax.lax.scan(ig_one_t, None, obs_batch)
            return all_attrs
        _ig_jit_cache[cache_key] = _ig_scan

    all_attrs = np.array(_ig_jit_cache[cache_key](jnp.array(raw_obs), bl))
    return _normalize_to_cats(np.abs(all_attrs), obs_struct)


def compute_perturbation(model, raw_obs: np.ndarray) -> np.ndarray:
    """Category-level zero perturbation — identical to FeatureAblation at category level."""
    import jax.numpy as jnp
    params, module, action_size = model._policy_params, model._policy_module, model._action_size
    obs_struct = model.observation_structure
    T          = raw_obs.shape[0]

    def get_actions(obs_batch):
        logits = module.apply(params, jnp.array(obs_batch))
        return np.array(logits[:, :action_size])

    baseline_actions = get_actions(raw_obs)          # (T, action_size)
    result           = np.zeros((T, len(CATS)), dtype=np.float32)

    for col_idx, (_, (s, e)) in enumerate(obs_struct.items()):
        perturbed              = raw_obs.copy()
        perturbed[:, s:e]      = 0.0
        perturbed_actions      = get_actions(perturbed)
        change                 = np.abs(baseline_actions - perturbed_actions).sum(axis=1)  # (T,)
        result[:, col_idx]     = change

    row_sums = result.sum(axis=1, keepdims=True) + 1e-10
    return result / row_sums


def compute_sarfa(model, raw_obs: np.ndarray) -> np.ndarray:
    from posthoc_xai.methods.sarfa import sarfa_batch
    return sarfa_batch(model, raw_obs, perturbation_type="zero", target_action=0)


# ── Scenario runner ────────────────────────────────────────────────────────────

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
    cfg  = AnalysisConfig(n_scenarios=50)
    risk = RiskComputer.from_scenario_data(sd, cfg)
    return {"raw_obs": np.array(sd.raw_observations), "collision_risk": np.array(risk.collision_risk), "T": sd.total_steps}


# ── LaTeX helpers ──────────────────────────────────────────────────────────────

def freq_table_to_latex(freq_df: pd.DataFrame, n: int, filename: Path):
    """rows=methods, cols=categories, cells=proportion. Two decimal places."""
    col_labels = ["SDC", "Agents", "Road", "TL", "GPS", "n"]
    lines = [
        r"\begin{tabular}{l" + "c" * 6 + "}",
        r"\toprule",
        "Method & " + " & ".join(col_labels) + r" \\",
        r"\midrule",
    ]
    for method in PARADIGM_ORDER:
        if method not in freq_df.index:
            continue
        row = freq_df.loc[method]
        cells = [f"{row[cat]:.2f}" if cat in row else "—" for cat in CATS]
        cells.append(str(n))
        lines.append(METHOD_LABELS[method] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    filename.write_text("\n".join(lines))
    print(f"  Saved: {filename.name}")


def agents_table_to_latex(table_df: pd.DataFrame, filename: Path):
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & High-risk mean & Normal mean & Ratio \\",
        r"\midrule",
    ]
    sarfa_written = False
    non_sarfa = [m for m in PARADIGM_ORDER if m != "sarfa"]
    for method in non_sarfa:
        if method not in table_df.index:
            continue
        row = table_df.loc[method]
        lines.append(f"{METHOD_LABELS[method]} & {row['high_mean']:.2f} & {row['normal_mean']:.2f} & {row['ratio']:.2f}" + r" \\")
    lines.append(r"\midrule")
    if "sarfa" in table_df.index:
        row = table_df.loc["sarfa"]
        lines.append(f"SARFA & {row['high_mean']:.2f} & {row['normal_mean']:.2f} & {row['ratio']:.2f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    filename.write_text("\n".join(lines))
    print(f"  Saved: {filename.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       choices=list(MODELS.keys()), default="complete")
    parser.add_argument("--data-path",   type=str, default=None)
    parser.add_argument("--runs-rlc",    type=str, default=None)
    parser.add_argument("--n-scenarios", type=int, default=None,
                        help="Number of scenarios to process. Defaults to all scenarios in B1 CSV.")
    args = parser.parse_args()

    data_path = args.data_path or str(_CBM / "data" / "training.tfrecord")
    runs_rlc  = Path(args.runs_rlc) if args.runs_rlc else (_CBM / "runs_rlc")
    model_dir = str(runs_rlc / MODELS[args.model])

    # ── Load B1 event assignments ──────────────────────────────────────────────
    assert B1_CSV.exists(), f"B1 CSV not found: {B1_CSV}. Run ig_risk_stratified.py first."
    b1 = pd.read_csv(B1_CSV)
    # Pivot to get one row per (scenario_id, timestep)
    ts_info = b1.drop_duplicates(subset=["scenario_id", "timestep"])[
        ["scenario_id", "timestep", "R", "event_type"]
    ].copy()

    # ── Threshold decision ─────────────────────────────────────────────────────
    high_thresh   = HIGH_RISK_THRESH
    n_high_global = (ts_info["event_type"] == "high").sum()
    if n_high_global < 30:
        high_thresh = FALLBACK_THRESH
        ts_info["event_type"] = ts_info["R"].apply(
            lambda r: "high" if r >= high_thresh else ("normal" if r < NORMAL_THRESH else "elevated")
        )
        print(f"  NOTE: fewer than 30 high-risk timesteps at R≥0.7 ({n_high_global}). "
              f"Extended threshold to R≥{high_thresh}.")

    high_ts   = ts_info[ts_info["event_type"] == "high"]
    normal_ts = ts_info[ts_info["event_type"] == "normal"]

    # Stratified sampling of normal control set (same scenarios, same count)
    n_high    = len(high_ts)
    scenarios_with_high = high_ts["scenario_id"].unique()
    sampled_normal = []
    for sid in scenarios_with_high:
        pool = normal_ts[normal_ts["scenario_id"] == sid]
        n_needed = len(high_ts[high_ts["scenario_id"] == sid])
        if len(pool) >= n_needed:
            sampled_normal.append(pool.sample(n=n_needed, random_state=42))
        else:
            sampled_normal.append(pool)
    normal_control = pd.concat(sampled_normal).reset_index(drop=True)

    print(f"\nTimestep counts:")
    print(f"  High-risk  (R≥{high_thresh}): {len(high_ts)}")
    print(f"  Normal control (R<{NORMAL_THRESH}): {len(normal_control)}")

    # ── Load model ────────────────────────────────────────────────────────────
    model   = xai.load_model(model_dir, data_path=data_path)
    adapter = make_adapter(model)
    data_gen = model._loaded.data_gen
    acc      = BaselineAccumulator()

    # ── Run all 7 methods over scenarios present in B1 ────────────────────────
    n_scenarios  = args.n_scenarios or int(ts_info["scenario_id"].max() + 1)
    all_rows     = []

    for scenario_id in range(n_scenarios):
        try:
            scenario = next(data_gen)
        except StopIteration:
            break

        ep = run_scenario(adapter, model, scenario, scenario_id)
        if ep is None:
            continue

        raw_obs        = ep["raw_obs"]
        collision_risk = ep["collision_risk"]
        T              = ep["T"]

        # Only process timesteps that appear in high or normal sets for this scenario
        target_ts = set(
            ts_info[
                (ts_info["scenario_id"] == scenario_id) &
                (ts_info["event_type"].isin(["high", "normal"]))
            ]["timestep"].tolist()
        )
        # Also include control set timesteps
        target_ts |= set(normal_control[normal_control["scenario_id"] == scenario_id]["timestep"].tolist())

        if not target_ts:
            continue

        acc.update(raw_obs)
        baseline = acc.finalize()

        print(f"  Scenario {scenario_id:04d}: {len(target_ts)} target timesteps...", flush=True)

        # Compute all 7 methods on full episode (more efficient than slicing)
        attrs = {
            "vg":               compute_vg(model, raw_obs),
            "ig":               compute_ig(model, raw_obs, baseline),
            "gxi":              compute_gxi(model, raw_obs),
            "smoothgrad":       compute_smoothgrad(model, raw_obs),
            "perturbation":     compute_perturbation(model, raw_obs),
            "feature_ablation": compute_perturbation(model, raw_obs),  # identical at category level
            "sarfa":            compute_sarfa(model, raw_obs),
        }

        for t in target_ts:
            if t >= T:
                continue
            R          = float(collision_risk[t])
            event_type = ts_info[
                (ts_info["scenario_id"] == scenario_id) & (ts_info["timestep"] == t)
            ]["event_type"].values
            if len(event_type) == 0:
                continue
            event_type = event_type[0]

            for method, arr in attrs.items():
                top_cat = CATS[int(arr[t].argmax())]
                all_rows.append({
                    "scenario_id":   scenario_id,
                    "timestep":      t,
                    "R":             R,
                    "event_type":    event_type,
                    "method":        method,
                    "sdc_attr":      float(arr[t, 0]),
                    "agents_attr":   float(arr[t, 1]),
                    "roadgraph_attr": float(arr[t, 2]),
                    "lights_attr":   float(arr[t, 3]),
                    "gps_attr":      float(arr[t, 4]),
                    "top_category":  top_cat,
                })

    df = pd.DataFrame(all_rows)
    csv_path = OUT_DIR / "b2b3_raw_attributions.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}  ({len(df)} rows)")

    # ── Artifact 1: Frequency tables ──────────────────────────────────────────
    for event_type, fname in [("high", "topcategory_highRisk.tex"), ("normal", "topcategory_normal.tex")]:
        subset = df[df["event_type"] == event_type]
        n_ts   = len(subset[subset["method"] == "vg"])
        freq   = (
            subset.groupby("method")["top_category"]
            .value_counts(normalize=True)
            .unstack(fill_value=0.0)
        )
        # Ensure all cat columns exist
        for cat in CATS:
            if cat not in freq.columns:
                freq[cat] = 0.0
        freq_table_to_latex(freq[CATS], n_ts, OUT_DIR / fname)

    # ── Artifact 2: Spearman correlation heatmap ──────────────────────────────
    high_df = df[df["event_type"] == "high"]
    corr_matrix = np.zeros((7, 7))
    ordered     = PARADIGM_ORDER

    for i, m1 in enumerate(ordered):
        for j, m2 in enumerate(ordered):
            if i == j:
                corr_matrix[i, j] = 1.0
                continue
            v1 = high_df[high_df["method"] == m1][CAT_COLS].values.flatten()
            v2 = high_df[high_df["method"] == m2][CAT_COLS].values.flatten()
            if len(v1) > 1:
                corr_matrix[i, j] = spearmanr(v1, v2).statistic
            else:
                corr_matrix[i, j] = np.nan

    labels = [METHOD_LABELS[m] for m in ordered]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, label="Spearman ρ")
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    # Annotate cells
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{corr_matrix[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(corr_matrix[i,j]) > 0.6 else "black")

    # Paradigm group separators
    ax.axhline(3.5, color="black", lw=2)
    ax.axvline(3.5, color="black", lw=2)
    ax.axhline(5.5, color="black", lw=2)
    ax.axvline(5.5, color="black", lw=2)

    # Paradigm labels on top
    for label, x_center in [("Gradient", 1.5), ("Occlusion", 4.5), ("RL-specific", 6.0)]:
        ax.text(x_center, -0.8, label, ha="center", fontsize=9, fontweight="bold",
                transform=ax.transData)

    ax.set_title("Method Agreement at High-Risk Timesteps\n(Spearman ρ on 5-category attribution vectors)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    heatmap_path = OUT_DIR / "method_correlation_heatmap.pdf"
    fig.savefig(heatmap_path, format="pdf", bbox_inches="tight")
    fig.savefig(str(heatmap_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: method_correlation_heatmap.pdf")

    # ── Artifact 3: SARFA agents table ────────────────────────────────────────
    normal_df = df[df["event_type"] == "normal"]
    agents_rows = {}
    for method in METHODS:
        h_mean = high_df[high_df["method"] == method]["agents_attr"].mean()
        n_mean = normal_df[normal_df["method"] == method]["agents_attr"].mean()
        ratio  = h_mean / (n_mean + 1e-10)
        agents_rows[method] = {"high_mean": h_mean, "normal_mean": n_mean, "ratio": ratio}

    agents_df = pd.DataFrame(agents_rows).T
    agents_df = agents_df.sort_values("ratio", ascending=False)
    agents_table_to_latex(agents_df, OUT_DIR / "sarfa_agents_table.tex")

    print("\nAgents attribution (high-risk vs normal):")
    print(agents_df.round(3).to_string())
    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
