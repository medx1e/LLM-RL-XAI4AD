"""Phase 2 — Attention-vs-Attribution Correlation Pilot.

Answers: does Perceiver rollout-corrected attention agree with gradient-based
attribution at the category level, and is agreement higher for IG than VG?

Data: platform_cache (2 models × 3 scenarios × 80 timesteps = 480 timesteps total).
Attention: rollout-corrected (cross_attn_rollout), live forward pass.
Attribution: IG and GxI loaded from cache; VG computed live (batched vmap).

Correlation approach:
  A) Per-timestep rank correlation (Kendall τ): rank the 5 categories by
     attention vs by attribution at each timestep. Reports distribution of τ.
  B) Category-level Pearson ρ over time: for each category, how well does
     attention track attribution across the episode?

Run in vmax env (one model per run):
  conda activate vmax
  cd /home/med1e/platform_fyp/post-hoc-xai
  python experiments/phase2_correlation_pilot.py --model complete
  python experiments/phase2_correlation_pilot.py --model minimal

Output: experiments/phase2_results/
"""

import sys
import argparse
import importlib.util
import pickle
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, kendalltau, wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE  = Path(__file__).parent
_ROOT  = _HERE.parent
_CBM   = _ROOT.parent / "cbm"
_CACHE = _ROOT.parent / "platform_cache"
OUT    = _HERE / "phase2_results"
OUT.mkdir(exist_ok=True)

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_CBM))
sys.path.insert(0, str(_CBM / "V-Max"))

# Load aggregation module directly (avoids posthoc_xai JAX __init__ at import time)
_agg_spec = importlib.util.spec_from_file_location(
    "attention_aggregation",
    _ROOT / "posthoc_xai" / "utils" / "attention_aggregation.py",
)
_agg = importlib.util.module_from_spec(_agg_spec)
_agg_spec.loader.exec_module(_agg)
aggregate_attention = _agg.aggregate_attention

import posthoc_xai as xai
from posthoc_xai.utils.ig_baseline import compute_baseline, compute_baseline_stats

CATS   = ["sdc_trajectory", "other_agents", "roadgraph", "traffic_lights", "gps_path"]
LABELS = ["SDC", "Agents", "Road", "TL", "GPS"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

MODELS = {
    "complete": ("womd_sac_road_perceiver_complete_42", "SAC_Perceiver_Complete_WOMD_seed_42"),
    "minimal":  ("womd_sac_road_perceiver_minimal_42",  "SAC_Perceiver_WOMD_seed_42"),
}
SCENARIOS  = [1, 2, 3]
DATA_PATH  = str(_CBM / "data" / "training.tfrecord")
# Cached attribution methods available in platform_cache
CACHE_METHODS = ["integrated_gradients", "gradient_x_input"]

# Action thresholds (calibrated from reward_attention 3,676 timesteps):
#   accel: mean=-0.083, std=0.389, range≈[-1,1]
#   steering: mean=-0.124, std=0.208, range≈[-1,1]
#   Thresholds set at ≈p25/p75 to get ~25% of timesteps per bucket.
ACTION_THRESHOLDS = {
    "braking":      lambda a, s: a < -0.3,
    "accelerating": lambda a, s: a > 0.3,
    "steering":     lambda a, s: abs(s) > 0.3 and abs(a) <= 0.3,
    "neutral":      lambda a, s: abs(a) <= 0.3 and abs(s) <= 0.3,
}


# ── unpickler (same pattern as phase1c) ───────────────────────────────────────

class _Stub:
    def __setstate__(self, state):
        self.__dict__.update(state)

_STUB_MODULES = ("platform.shared", "platform.posthoc", "platform.tabs", "bev_visualizer")

class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if any(module.startswith(m) for m in _STUB_MODULES):
            return _Stub
        return super().find_class(module, name)


# ── data loading ──────────────────────────────────────────────────────────────

def load_raw_obs(model_slug: str, scenario_idx: int) -> np.ndarray:
    """Load raw_observations (T, obs_dim) from artifact pkl."""
    path = _CACHE / model_slug / f"scenario_{scenario_idx:04d}_artifact.pkl"
    with open(path, "rb") as f:
        artifact = _SafeUnpickler(f).load()
    return np.array(artifact.raw_observations)   # (T, 1655)


def load_attribution_cache(
    model_slug: str, scenario_idx: int, method: str
) -> list[dict]:
    """Load cached list[Attribution] → list of category_importance dicts."""
    path = _CACHE / model_slug / f"scenario_{scenario_idx:04d}_attr_{method}.pkl"
    with open(path, "rb") as f:
        attrs = pickle.load(f)   # list[Attribution]; needs vmax for jnp arrays
    return [a.category_importance for a in attrs]


# ── attention & attribution computation ───────────────────────────────────────

def compute_rollout_attention(model, raw_obs: np.ndarray) -> list[dict]:
    """Batched forward pass → rollout-corrected attention per timestep."""
    import jax.numpy as jnp
    obs_batch = jnp.array(raw_obs)                         # (T, obs_dim)
    out       = model.forward(obs_batch)
    attn      = out.attention

    key = "cross_attn_rollout" if "cross_attn_rollout" in attn else "cross_attn_avg"
    rollout = np.array(attn[key])                          # (T, 16, 280)

    return [aggregate_attention(rollout[t], "rollout") for t in range(rollout.shape[0])]


def compute_vg(model, raw_obs: np.ndarray) -> list[dict]:
    """Batched VG via jax.vmap(jax.grad) → category_importance per timestep.

    Uses vmap over the gradient computation — one JIT call for all T observations.
    """
    import jax
    import jax.numpy as jnp

    params = model._policy_params
    module = model._policy_module
    action_size = model._action_size

    def scalar_output(obs_1d):
        """Scalar Q-proxy: sum of action means for a single (unbatched) obs."""
        logits = module.apply(params, obs_1d[None, :])    # (1, action_dim*2)
        return jnp.sum(logits[0, :action_size])            # sum of action means

    grad_fn       = jax.grad(scalar_output)                # obs → (1655,) gradient
    batched_grad  = jax.vmap(grad_fn)                      # (T,1655) → (T,1655)

    obs_batch = jnp.array(raw_obs)                         # (T, 1655)
    grads     = np.array(batched_grad(obs_batch))          # (T, 1655)

    # Aggregate each gradient to category importance (abs, normalize, sum per cat)
    obs_struct = model.observation_structure               # {cat: (start, end)}
    series = []
    for t in range(grads.shape[0]):
        abs_g = np.abs(grads[t])
        total = abs_g.sum() + 1e-10
        cat_imp = {cat: float(abs_g[s:e].sum() / total)
                   for cat, (s, e) in obs_struct.items()}
        series.append(cat_imp)
    return series


# ── baseline & IG with new baseline ──────────────────────────────────────────

def build_baseline(all_raw_obs: list[np.ndarray]) -> np.ndarray:
    """Compute validity-zeroed mean baseline from a list of obs arrays.

    Accepts any number of scenarios — reuses BaselineAccumulator so this
    function can be called again in Phase 3 with 50 scenarios.
    """
    from posthoc_xai.utils.ig_baseline import BaselineAccumulator
    acc = BaselineAccumulator()
    for obs in all_raw_obs:
        acc.update(obs)
    baseline = acc.finalize()

    stats = compute_baseline_stats(np.concatenate(all_raw_obs, axis=0))
    print(f"  Baseline: {stats['n_observations']} obs, "
          f"{stats['n_binary_features']} validity bits zeroed "
          f"({stats['binary_per_category']})")
    return baseline


def compute_ig_new_baseline(
    model, raw_obs: np.ndarray, baseline: np.ndarray
) -> list[dict]:
    """IG with validity-zeroed mean baseline — batched vmap over path."""
    import jax
    import jax.numpy as jnp

    ig = xai.IntegratedGradients(model, n_steps=50, baseline=baseline)
    obs_struct = model.observation_structure

    obs_batch = jnp.array(raw_obs)    # (T, 1655)
    series = []
    for t in range(raw_obs.shape[0]):
        attr  = ig(obs_batch[t])
        series.append(attr.category_importance)
    return series


def extract_actions(model, raw_obs: np.ndarray) -> np.ndarray:
    """Run batched forward pass → action means (T, action_dim).

    Returns numpy array (T, 2) with columns [accel, steering].
    """
    import jax.numpy as jnp
    obs_batch = jnp.array(raw_obs)
    out       = model.forward(obs_batch)
    return np.array(out.action_mean)   # (T, 2) — [accel, steering]


def classify_actions(actions: np.ndarray) -> list[str]:
    """Map (T, 2) action array → list of T action-type strings."""
    labels = []
    for a, s in actions:
        for name, fn in ACTION_THRESHOLDS.items():
            if fn(float(a), float(s)):
                labels.append(name)
                break
        else:
            labels.append("neutral")
    return labels


# ── correlation metrics ───────────────────────────────────────────────────────

def per_timestep_correlation(
    series_a: list[dict], series_b: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-timestep Pearson ρ and Kendall τ between two 5-dim vectors."""
    rs, taus = [], []
    for a, b in zip(series_a, series_b):
        a_vec = np.array([a[c] for c in CATS])
        b_vec = np.array([b[c] for c in CATS])
        r,  _ = pearsonr(a_vec,  b_vec)
        τ,  _ = kendalltau(a_vec, b_vec)
        rs.append(r); taus.append(τ)
    return np.array(rs), np.array(taus)


def category_correlation(
    series_a: list[dict], series_b: list[dict]
) -> dict[str, dict]:
    """Per-category Pearson ρ across all timesteps.
    Returns r=0, p=1 for constant series (e.g. TL=0 when no lights in scene).
    """
    results = {}
    for c in CATS:
        a_vals = np.array([d[c] for d in series_a])
        b_vals = np.array([d[c] for d in series_b])
        if a_vals.std() < 1e-8 or b_vals.std() < 1e-8:
            results[c] = {"r": 0.0, "p": 1.0, "constant": True}
        else:
            r, p = pearsonr(a_vals, b_vals)
            results[c] = {"r": float(r), "p": float(p), "constant": False}
    return results


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_tau_distributions(
    tau_dict: dict[str, np.ndarray], model_label: str, scenario_idx: int
):
    """Violin/box plot of per-timestep Kendall τ distributions per method."""
    fig, ax = plt.subplots(figsize=(7, 4))
    labels  = list(tau_dict.keys())
    data    = [tau_dict[m] for m in labels]
    bp = ax.violinplot(data, positions=range(len(labels)), showmedians=True)
    ax.axhline(0, color="gray", lw=0.8, linestyle="--", label="τ=0 (no agreement)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Kendall τ (per-timestep, 5 categories)")
    ax.set_ylim(-1, 1)
    ax.set_title(
        f"{model_label} — s{scenario_idx:02d}  |  "
        "Attention vs attribution rank agreement"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    name = f"tau_dist_{model_label}_s{scenario_idx:02d}.png"
    fig.savefig(OUT / name, dpi=150); plt.close(fig)
    print(f"  saved: {name}")


def plot_category_correlations(
    cat_corr_dict: dict[str, dict[str, dict]], model_label: str, scenario_idx: int
):
    """Grouped bar chart: per-category Pearson ρ for each method."""
    methods = list(cat_corr_dict.keys())
    x       = np.arange(len(CATS))
    width   = 0.8 / len(methods)
    method_colors = plt.cm.tab10(np.linspace(0, 0.6, len(methods)))

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (method, color) in enumerate(zip(methods, method_colors)):
        rs = [cat_corr_dict[method][c]["r"] for c in CATS]
        ax.bar(x + i * width - 0.4 + width / 2, rs, width,
               label=method, color=color, alpha=0.85)

    ax.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(LABELS)
    ax.set_ylabel("Pearson ρ (over timesteps)")
    ax.set_ylim(-1, 1)
    ax.set_title(
        f"{model_label} — s{scenario_idx:02d}  |  "
        "Category-level attention-attribution correlation"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    name = f"cat_corr_{model_label}_s{scenario_idx:02d}.png"
    fig.savefig(OUT / name, dpi=150); plt.close(fig)
    print(f"  saved: {name}")


def plot_scatter(
    attn_series: list[dict], method_series: list[dict],
    method_name: str, model_label: str, scenario_idx: int
):
    """5-panel scatter: attention vs method per category over timesteps."""
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for ax, cat, lbl, col in zip(axes, CATS, LABELS, COLORS):
        a = np.array([d[cat] for d in attn_series])
        m = np.array([d[cat] for d in method_series])
        if a.std() < 1e-8 or m.std() < 1e-8:
            r, p = 0.0, 1.0
        else:
            r, p = pearsonr(a, m)
        ax.scatter(a, m, s=8, alpha=0.5, color=col)
        ax.set_xlabel("Attention"); ax.set_ylabel(method_name)
        ax.set_title(f"{lbl}\nρ={r:.2f}", fontsize=9)
    fig.suptitle(
        f"{model_label} — s{scenario_idx:02d}  |  Attention vs {method_name}",
        fontsize=10, fontweight="bold"
    )
    fig.tight_layout()
    name = f"scatter_{method_name}_{model_label}_s{scenario_idx:02d}.png"
    fig.savefig(OUT / name, dpi=150); plt.close(fig)
    print(f"  saved: {name}")


def action_conditioned_tau(
    tau_per_method: dict[str, np.ndarray],
    action_labels: list[str],
) -> dict[str, dict[str, float]]:
    """Mean τ per action type per method.

    Returns {method: {action_type: mean_tau, 'n': count}}.
    """
    action_arr = np.array(action_labels)
    results = {}
    for method, taus in tau_per_method.items():
        if method == "IG_vs_VG (calibration)":
            continue
        bucket_stats = {}
        for action_type in ACTION_THRESHOLDS:
            mask = action_arr == action_type
            n = mask.sum()
            bucket_stats[action_type] = {
                "mean_tau": float(taus[mask].mean()) if n > 0 else float("nan"),
                "n": int(n),
            }
        results[method] = bucket_stats
    return results


def plot_action_conditioned(
    ac_results: dict[str, dict],
    model_label: str,
    scenario_idx: int,
):
    """Bar chart: mean τ per action type per method."""
    action_types = list(ACTION_THRESHOLDS.keys())
    methods      = [m for m in ac_results if m != "IG_vs_VG (calibration)"]
    x = np.arange(len(action_types))
    width = 0.8 / len(methods)
    colors = plt.cm.tab10(np.linspace(0, 0.6, len(methods)))

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (method, color) in enumerate(zip(methods, colors)):
        means = []
        ns    = []
        for at in action_types:
            d = ac_results[method].get(at, {})
            means.append(d.get("mean_tau", float("nan")))
            ns.append(d.get("n", 0))
        bars = ax.bar(x + i * width - 0.4 + width / 2, means, width,
                      label=method, color=color, alpha=0.85)
        for bar, n in zip(bars, ns):
            ax.text(bar.get_x() + bar.get_width()/2, 0.02, f"n={n}",
                    ha="center", va="bottom", fontsize=6, rotation=90)

    ax.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(action_types)
    ax.set_ylabel("Mean Kendall τ"); ax.set_ylim(-1, 1)
    ax.set_title(f"{model_label} — s{scenario_idx:02d}  |  Action-conditioned τ")
    ax.legend(fontsize=8)
    fig.tight_layout()
    name = f"action_conditioned_{model_label}_s{scenario_idx:02d}.png"
    fig.savefig(OUT / name, dpi=150); plt.close(fig)
    print(f"  saved: {name}")


def plot_combined_summary(all_results: dict, model_label: str):
    """Aggregate τ distributions across all scenarios for one model."""
    methods = None
    combined = {}

    for s_idx, res in all_results.items():
        if methods is None:
            methods = list(res["tau_per_method"].keys())
            combined = {m: [] for m in methods}
        for m in methods:
            combined[m].extend(res["tau_per_method"][m].tolist())

    fig, ax = plt.subplots(figsize=(8, 4))
    data = [combined[m] for m in methods]
    bp = ax.violinplot(data, positions=range(len(methods)), showmedians=True)
    ax.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=15)
    ax.set_ylabel("Kendall τ")
    ax.set_ylim(-1, 1)
    ax.set_title(
        f"{model_label} — All scenarios combined  |  "
        "Attention vs attribution rank agreement"
    )
    fig.tight_layout()
    name = f"tau_combined_{model_label}.png"
    fig.savefig(OUT / name, dpi=150); plt.close(fig)
    print(f"  saved: {name}")
    return combined


# ── numerical summary ─────────────────────────────────────────────────────────

def print_summary(all_results: dict, model_label: str):
    print(f"\n=== {model_label.upper()} — CORRELATION SUMMARY ===")
    methods = None
    combined_tau = {}

    for s_idx, res in all_results.items():
        if methods is None:
            methods = list(res["tau_per_method"].keys())
            combined_tau = {m: [] for m in methods}
        print(f"\n  Scenario {s_idx}  (T={res['T']})")
        print(f"    {'Method':25}  {'Mean τ':>8}  {'Median τ':>9}  {'τ>0 (%)':>8}")
        for m in methods:
            taus = res["tau_per_method"][m]
            combined_tau[m].extend(taus.tolist())
            print(f"    {m:25}  {taus.mean():8.3f}  {np.median(taus):9.3f}  "
                  f"{(taus > 0).mean()*100:7.1f}%")

    print(f"\n  COMBINED (all scenarios)")
    print(f"    {'Method':25}  {'Mean τ':>8}  {'Median τ':>9}  {'τ>0 (%)':>8}")
    for m in methods:
        taus = np.array(combined_tau[m])
        print(f"    {m:25}  {taus.mean():8.3f}  {np.median(taus):9.3f}  "
              f"{(taus > 0).mean()*100:7.1f}%")

    # Wilcoxon test: is attention-IG τ significantly higher than attention-VG τ?
    if "vanilla_gradient" in combined_tau and "integrated_gradients" in combined_tau:
        vg_taus = np.array(combined_tau["vanilla_gradient"])
        ig_taus = np.array(combined_tau["integrated_gradients"])
        stat, p = wilcoxon(ig_taus, vg_taus, alternative="greater")
        print(f"\n  Wilcoxon test (IG τ > VG τ): stat={stat:.1f}, p={p:.4f} "
              f"({'significant' if p < 0.05 else 'not significant'})")

    return combined_tau


def save_results(model_key: str, all_results: dict, combined_tau: dict):
    """Save JSON summary for cross-model comparison."""
    summary = {}
    methods = list(combined_tau.keys())
    for m in methods:
        taus = np.array(combined_tau[m])
        summary[m] = {
            "mean_tau": float(taus.mean()),
            "median_tau": float(np.median(taus)),
            "pct_positive": float((taus > 0).mean()),
            "n_timesteps": len(taus),
        }
    # Per-scenario category correlations (use first scenario as representative)
    first = list(all_results.values())[0]
    summary["category_corr_ig"] = first.get("cat_corr", {}).get(
        "integrated_gradients", {}
    )
    with open(OUT / f"results_{model_key}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved: results_{model_key}.json")


def write_findings():
    """Write combined findings.md when both model JSONs exist."""
    complete_p = OUT / "results_complete.json"
    minimal_p  = OUT / "results_minimal.json"
    if not (complete_p.exists() and minimal_p.exists()):
        return

    with open(complete_p) as f: c = json.load(f)
    with open(minimal_p)  as f: m = json.load(f)

    lines = ["# Phase 2 Findings — Attention-Attribution Correlation Pilot\n\n"]
    lines.append(
        "## Setup\n\n"
        "- 2 models × 3 scenarios × 80 timesteps = 480 timesteps total\n"
        "- Attention: rollout-corrected (Phase 1c canonical signal)\n"
        "- Methods compared: VG (baseline), IG (primary), GxI (from cache)\n"
        "- Metric: Kendall τ between 5-category attention ranking and attribution ranking\n\n"
    )

    lines.append("## Combined Results (all scenarios)\n\n")
    lines.append("| Method | Complete mean τ | Minimal mean τ | Complete τ>0 | Minimal τ>0 |\n")
    lines.append("|---|---|---|---|---|\n")
    for method in ["vanilla_gradient", "integrated_gradients", "gradient_x_input"]:
        if method in c and method in m:
            lines.append(
                f"| {method} | {c[method]['mean_tau']:.3f} | {m[method]['mean_tau']:.3f} "
                f"| {c[method]['pct_positive']*100:.0f}% | {m[method]['pct_positive']*100:.0f}% |\n"
            )

    lines.append("\n## Interpretation\n\n")

    # Auto-generate interpretation
    ig_c = c.get("integrated_gradients", {}).get("mean_tau", 0)
    vg_c = c.get("vanilla_gradient", {}).get("mean_tau", 0)
    ig_m = m.get("integrated_gradients", {}).get("mean_tau", 0)

    if ig_c > 0.3:
        lines.append(
            f"Strong positive attention-IG agreement (τ={ig_c:.3f} for complete). "
            "Perceiver attention is a faithful importance signal by the Jain & Wallace test.\n\n"
        )
    elif ig_c > 0.1:
        lines.append(
            f"Moderate positive attention-IG agreement (τ={ig_c:.3f} for complete). "
            "Attention provides partial but imperfect explanation signal.\n\n"
        )
    else:
        lines.append(
            f"Weak attention-IG agreement (τ={ig_c:.3f} for complete). "
            "Attention does not reliably reflect gradient importance at category level.\n\n"
        )

    if ig_c > vg_c:
        lines.append(
            f"IG agrees with attention better than VG (τ_IG={ig_c:.3f} vs τ_VG={vg_c:.3f}). "
            "Path-integrated attribution is more consistent with attention than local gradient.\n\n"
        )

    lines.append(
        "## Go/No-Go for Phase 3\n\n"
        f"Mean attention-IG τ = {ig_c:.3f} (complete), {ig_m:.3f} (minimal). "
    )
    if ig_c > 0.1 or ig_m > 0.1:
        lines.append(
            "**GO** — signal is present, scale up to 50 scenarios with risk stratification.\n"
        )
    else:
        lines.append(
            "**RECONSIDER** — weak signal at pilot scale. Review methodology before scaling.\n"
        )

    lines.append(
        "\n## What to write in the thesis\n\n"
        "> We computed the Kendall rank correlation between rollout-corrected Perceiver "
        "attention and gradient attribution at the category level for each timestep. "
        "Across 480 timesteps (2 models × 3 scenarios × 80 steps), mean attention-IG "
        f"τ = {ig_c:.3f} (complete) and {ig_m:.3f} (minimal), with "
        f"{c.get('integrated_gradients',{}).get('pct_positive',0)*100:.0f}% and "
        f"{m.get('integrated_gradients',{}).get('pct_positive',0)*100:.0f}% of timesteps "
        "showing positive agreement respectively. [EXPAND with risk-stratified results "
        "from Phase 3.]\n"
    )

    with open(OUT / "findings.md", "w") as f:
        f.writelines(lines)
    print("  saved: findings.md")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2 — correlation pilot")
    parser.add_argument("--model", choices=["complete", "minimal"], required=True)
    args       = parser.parse_args()
    model_key  = args.model
    model_dir, model_slug = MODELS[model_key]

    print(f"Phase 2 — Attention-Attribution Correlation  [{model_key}]")
    print(f"Output: {OUT}\n")

    model_path = str(_CBM / "runs_rlc" / model_dir)
    print(f"Loading model: {model_dir}")
    model = xai.load_model(model_path, data_path=DATA_PATH)
    print(f"  Model loaded.\n")

    # ── Build validity-zeroed mean baseline from all scenarios ────────────
    print("Building IG baseline from all scenarios...")
    all_raw_obs = [load_raw_obs(model_slug, s) for s in SCENARIOS]
    ig_baseline = build_baseline(all_raw_obs)

    all_results = {}

    for s_idx, raw_obs in zip(SCENARIOS, all_raw_obs):
        print(f"\nScenario {s_idx}...")
        print(f"  raw_obs: {raw_obs.shape}")

        # ── Attention (rollout) ──────────────────────────────────────────
        print("  Computing rollout attention...")
        attn_series = compute_rollout_attention(model, raw_obs)

        # ── Actions ──────────────────────────────────────────────────────
        print("  Extracting actions...")
        actions       = extract_actions(model, raw_obs)   # (T, 2)
        action_labels = classify_actions(actions)
        accel_vals    = actions[:, 0]
        steer_vals    = actions[:, 1]
        bucket_counts = {k: action_labels.count(k) for k in ACTION_THRESHOLDS}
        print(f"  Action buckets: {bucket_counts}")

        # ── Attribution ──────────────────────────────────────────────────
        method_series = {}

        print("  Loading cached IG (zero baseline) and GxI...")
        for m_name in CACHE_METHODS:
            try:
                method_series[m_name] = load_attribution_cache(model_slug, s_idx, m_name)
            except FileNotFoundError:
                print(f"    SKIP {m_name} — not in cache")

        print("  Computing VG (batched vmap)...")
        method_series["vanilla_gradient"] = compute_vg(model, raw_obs)

        print("  Computing IG (new validity-zeroed mean baseline)...")
        method_series["ig_mean_baseline"] = compute_ig_new_baseline(
            model, raw_obs, ig_baseline
        )

        # ── Correlations ─────────────────────────────────────────────────
        tau_per_method = {}
        cat_corr       = {}

        for m_name, m_series in method_series.items():
            _, taus = per_timestep_correlation(attn_series, m_series)
            tau_per_method[m_name] = taus
            cat_corr[m_name]       = category_correlation(attn_series, m_series)

        # ── Calibration: IG(zero) vs IG(mean baseline) ───────────────────
        if "integrated_gradients" in method_series and "ig_mean_baseline" in method_series:
            _, taus_ig_compare = per_timestep_correlation(
                method_series["integrated_gradients"],
                method_series["ig_mean_baseline"]
            )
            tau_per_method["IG_zero_vs_IG_mean (calibration)"] = taus_ig_compare

        # IG-vs-VG calibration
        if "vanilla_gradient" in method_series and "ig_mean_baseline" in method_series:
            _, taus_ig_vg = per_timestep_correlation(
                method_series["ig_mean_baseline"],
                method_series["vanilla_gradient"]
            )
            tau_per_method["IG_vs_VG (calibration)"] = taus_ig_vg

        all_results[s_idx] = {
            "T": len(attn_series),
            "tau_per_method": tau_per_method,
            "cat_corr": cat_corr,
            "action_labels": action_labels,
            "accel": accel_vals.tolist(),
            "steering": steer_vals.tolist(),
        }

        # ── Action-conditioned τ ──────────────────────────────────────────
        ac_results = action_conditioned_tau(tau_per_method, action_labels)

        # ── Plots ─────────────────────────────────────────────────────────
        plot_tau_distributions(tau_per_method, model_key, s_idx)
        plot_category_correlations(
            {m: cat_corr[m] for m in method_series}, model_key, s_idx
        )
        for m_name in ["ig_mean_baseline", "vanilla_gradient"]:
            if m_name in method_series:
                plot_scatter(attn_series, method_series[m_name],
                             m_name, model_key, s_idx)
        plot_action_conditioned(ac_results, model_key, s_idx)

        print(f"  Done — T={len(attn_series)} timesteps")

    # ── Combined summary ──────────────────────────────────────────────────
    combined_tau = print_summary(all_results, model_key)
    plot_combined_summary(all_results, model_key)
    save_results(model_key, all_results, combined_tau)
    write_findings()

    remaining = [k for k in MODELS if k != model_key
                 and not (OUT / f"results_{k}.json").exists()]
    if remaining:
        print(f"\nRun next: python phase2_correlation_pilot.py --model {remaining[0]}")
    print("\nDone.")


if __name__ == "__main__":
    main()
