"""Phase 1d — Norm-weighted attention comparison (Kobayashi et al. 2020).

Validates the norm_weighted_attn signal by comparing it against rollout and
raw cross_attn_avg on the 3 platform_cache scenarios. Also compares Option 1a
(value norms after W_O output projection) vs Option 1b (value norms before W_O).

Run in the vmax conda environment (one model per process — Waymax registry):
  conda activate vmax
  cd /home/med1e/platform_fyp/post-hoc-xai
  python experiments/phase1d_norm_weighted_comparison.py --model complete
  python experiments/phase1d_norm_weighted_comparison.py --model minimal

Output: experiments/phase1d_results/
"""

import sys
import argparse
import importlib.util
import pickle
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).parent
_ROOT     = _HERE.parent
_CBM      = _ROOT.parent / "cbm"
_CACHE    = _ROOT.parent / "platform_cache"
OUT       = _HERE / "phase1d_results"
OUT.mkdir(exist_ok=True)

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_CBM))
sys.path.insert(0, str(_CBM / "V-Max"))

# Load aggregation module directly (avoids triggering JAX __init__)
_agg_spec = importlib.util.spec_from_file_location(
    "attention_aggregation",
    _ROOT / "posthoc_xai" / "utils" / "attention_aggregation.py",
)
_agg = importlib.util.module_from_spec(_agg_spec)
_agg_spec.loader.exec_module(_agg)
aggregate_attention = _agg.aggregate_attention
TOKEN_RANGES = _agg.TOKEN_RANGES

import posthoc_xai as xai

CATS   = list(TOKEN_RANGES.keys())
LABELS = ["SDC", "Agents", "Road", "TL", "GPS"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

MODELS = {
    "complete": ("womd_sac_road_perceiver_complete_42",
                 "SAC_Perceiver_Complete_WOMD_seed_42"),
    "minimal":  ("womd_sac_road_perceiver_minimal_42",
                 "SAC_Perceiver_WOMD_seed_42"),
}
SCENARIOS = [1, 2, 3]
DATA_PATH = str(_CBM / "data" / "training.tfrecord")


# ── data helpers (same stub pattern as phase1c) ────────────────────────────────

class _ArtifactStub:
    def __setstate__(self, state):
        self.__dict__.update(state)


_STUB_MODULES = (
    "platform.shared", "platform.posthoc", "platform.tabs", "bev_visualizer",
)


class _PlatformUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if any(module.startswith(m) for m in _STUB_MODULES):
            return _ArtifactStub
        return super().find_class(module, name)


def load_artifact(model_slug: str, scenario_idx: int):
    path = _CACHE / model_slug / f"scenario_{scenario_idx:04d}_artifact.pkl"
    with open(path, "rb") as f:
        return _PlatformUnpickler(f).load()


# ── forward pass ───────────────────────────────────────────────────────────────

def run_forward_pass(model, raw_obs_np: np.ndarray) -> dict:
    """Single batched forward pass → all attention signals for T timesteps.

    Returns:
        cross_avg     : (T, 16, 280) raw mean-pooled cross-attention
        rollout       : (T, 16, 280) rollout-corrected, or None
        norm_weighted : (T, 16, 280) norm-weighted (Option 1a, with W_O), or None
        f_x_norms     : (T, 280) per-token ‖f(x_j)‖₂ with W_O, or None
        f_v_norms     : (T, 280) per-token ‖v_j‖₂ without W_O, or None
    """
    import jax.numpy as jnp

    obs_batch = jnp.array(raw_obs_np)
    attn = model.forward(obs_batch).attention

    if attn is None:
        raise RuntimeError("No attention returned — wrong model type?")

    return {
        "cross_avg":     np.array(attn["cross_attn_avg"]),
        "rollout":       np.array(attn["cross_attn_rollout"])    if "cross_attn_rollout"  in attn else None,
        "norm_weighted": np.array(attn["norm_weighted_attn"])    if "norm_weighted_attn"  in attn else None,
        "f_x_norms":     np.array(attn["f_x_norms"])            if "f_x_norms"           in attn else None,
        "f_v_norms":     np.array(attn["f_v_norms"])            if "f_v_norms"           in attn else None,
    }


# ── analysis helpers ───────────────────────────────────────────────────────────

def to_cat_series(attn_3d: np.ndarray, mode: str) -> dict[str, list]:
    """Convert (T, 16, 280) → per-category time series via aggregate_attention."""
    T = attn_3d.shape[0]
    series = {c: [] for c in CATS}
    for t in range(T):
        d = aggregate_attention(attn_3d[t], mode)
        for c in CATS:
            series[c].append(d[c])
    return series


def mad(a: dict, b: dict) -> dict[str, float]:
    return {
        c: float(np.mean(np.abs(np.array(a[c]) - np.array(b[c]))))
        for c in CATS
    }


def global_mad(mad_dict: dict) -> float:
    return float(np.mean(list(mad_dict.values())))


def compute_nw_1b(cross_avg_3d: np.ndarray, f_v_norms_2d: np.ndarray) -> np.ndarray:
    """Manually compute Option 1b norm-weighted from cross_avg and v_token norms.

    Option 1b uses ‖v_j‖ (no W_O), whereas the wrapper's norm_weighted_attn uses
    ‖f(x_j)‖ = ‖v_j @ W_O‖ (Option 1a). Comparing the two shows the W_O effect.
    """
    T = cross_avg_3d.shape[0]
    nw_1b = np.zeros_like(cross_avg_3d)
    for t in range(T):
        nw_t     = cross_avg_3d[t] * f_v_norms_2d[t][None, :]  # (16, 280)
        row_sums = nw_t.sum(axis=-1, keepdims=True) + 1e-8
        nw_1b[t] = nw_t / row_sums
    return nw_1b


def analyse_scenario(model, model_slug: str, scenario_idx: int) -> dict:
    artifact = load_artifact(model_slug, scenario_idx)
    raw_obs  = np.array(artifact.raw_observations)
    if raw_obs is None:
        raise ValueError("No raw_observations in artifact")

    fwd = run_forward_pass(model, raw_obs)
    T   = fwd["cross_avg"].shape[0]

    raw_series = to_cat_series(fwd["cross_avg"], "mean")
    result     = {
        "T":              T,
        "has_rollout":      fwd["rollout"]      is not None,
        "has_norm_weighted": fwd["norm_weighted"] is not None,
        "raw_series":     raw_series,
    }

    if fwd["rollout"] is not None:
        result["rollout_series"]    = to_cat_series(fwd["rollout"], "rollout")
        result["mad_rollout_vs_raw"] = mad(result["rollout_series"], raw_series)

    if fwd["norm_weighted"] is not None:
        result["nw_series"]       = to_cat_series(fwd["norm_weighted"], "mean")
        result["mad_nw_vs_raw"]   = mad(result["nw_series"], raw_series)
        if fwd["rollout"] is not None:
            result["mad_nw_vs_rollout"] = mad(result["nw_series"], result["rollout_series"])

    # Option 1b (without W_O) — computed in-script from f_v_norms
    if fwd["f_v_norms"] is not None:
        nw_1b = compute_nw_1b(fwd["cross_avg"], fwd["f_v_norms"])
        result["nw_1b_series"] = to_cat_series(nw_1b, "mean")
        if fwd["norm_weighted"] is not None:
            result["mad_1a_vs_1b"] = mad(result["nw_series"], result["nw_1b_series"])

    # Value-vector norm statistics (diagnostic for CV check — Section G)
    for key, label in [("f_x_norms", "f_x_norms"), ("f_v_norms", "f_v_norms")]:
        if fwd[key] is not None:
            norms = fwd[key]  # (T, 280)
            mean_val = float(norms.mean())
            result[f"{label}_stats"] = {
                "min":  float(norms.min()),
                "max":  float(norms.max()),
                "mean": mean_val,
                "std":  float(norms.std()),
                "cv":   float(norms.std() / (mean_val + 1e-8)),
            }

    # Sanity check 1: row sums of norm_weighted_attn should be ≈ 1
    if fwd["norm_weighted"] is not None:
        row_sums = fwd["norm_weighted"].sum(axis=-1)  # (T, 16)
        result["nw_row_sum_max_dev"] = float(np.abs(row_sums - 1.0).max())

    return result


# ── plots ──────────────────────────────────────────────────────────────────────

def plot_four_way(data: dict, model_label: str, scenario_idx: int):
    """Category time series: raw, rollout, norm_weighted (1a), norm_weighted (1b)."""
    T  = data["T"]
    ts = list(range(T))

    panels = [("Raw (cross_attn_avg)",    data["raw_series"],          "mean")]
    if "rollout_series"  in data: panels.append(("Rollout",              data["rollout_series"],  "rollout"))
    if "nw_series"       in data: panels.append(("Norm-weighted 1a (W_O)", data["nw_series"],     "mean"))
    if "nw_1b_series"    in data: panels.append(("Norm-weighted 1b (no W_O)", data["nw_1b_series"], "mean"))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (title, series, _) in zip(axes, panels):
        for cat, lbl, col in zip(CATS, LABELS, COLORS):
            ax.plot(ts, series[cat], label=lbl, color=col, lw=1.5)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Timestep")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=6)

    axes[0].set_ylabel("Attention fraction")
    fig.suptitle(f"{model_label} — Scenario {scenario_idx}  |  Attention signal comparison",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    name = f"four_way_{model_label}_s{scenario_idx:02d}.png"
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print(f"  saved: {name}")


def plot_delta_nw_vs_rollout(data: dict, model_label: str, scenario_idx: int):
    """Per-category delta: norm_weighted − rollout over time."""
    if "nw_series" not in data or "rollout_series" not in data:
        return
    T  = data["T"]
    ts = list(range(T))

    fig, axes = plt.subplots(1, len(CATS), figsize=(3 * len(CATS), 3), sharey=True)
    for ax, cat, lbl, col in zip(axes, CATS, LABELS, COLORS):
        delta = np.array(data["nw_series"][cat]) - np.array(data["rollout_series"][cat])
        ax.plot(ts, delta, color=col, lw=1.2)
        ax.axhline(0, color="gray", lw=0.7, linestyle="--")
        ax.set_title(f"{lbl}\nMAD={data['mad_nw_vs_rollout'][cat]:.4f}", fontsize=8)
        ax.set_xlabel("Timestep", fontsize=7)

    axes[0].set_ylabel("Norm-weighted − rollout", fontsize=8)
    fig.suptitle(f"{model_label} — Scenario {scenario_idx}  |  NW − rollout",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    name = f"delta_nw_rollout_{model_label}_s{scenario_idx:02d}.png"
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print(f"  saved: {name}")


def plot_mad_summary(all_data: dict):
    """Bar chart: mean MAD across all models/scenarios for each comparison."""
    comparisons = [
        ("mad_nw_vs_raw",      "NW vs raw"),
        ("mad_nw_vs_rollout",  "NW vs rollout"),
        ("mad_1a_vs_1b",       "1a vs 1b (W_O effect)"),
        ("mad_rollout_vs_raw", "Rollout vs raw"),
    ]

    # Aggregate per category
    agg = {key: {c: [] for c in CATS} for key, _ in comparisons}
    for model_data in all_data.values():
        for data in model_data.values():
            for key, _ in comparisons:
                if key in data:
                    for c in CATS:
                        v = data[key].get(c)
                        if v is not None:
                            agg[key][c].append(v)

    valid = [(key, lbl) for key, lbl in comparisons if any(agg[key][c] for c in CATS)]
    if not valid:
        return

    fig, axes = plt.subplots(1, len(valid), figsize=(4 * len(valid), 3.5))
    if len(valid) == 1:
        axes = [axes]

    for ax, (key, lbl) in zip(axes, valid):
        means = [np.mean(agg[key][c]) if agg[key][c] else 0.0 for c in CATS]
        bars  = ax.bar(LABELS, means, color=COLORS)
        ax.set_title(f"{lbl}", fontsize=9)
        ax.set_ylabel("Mean MAD")
        ax.set_ylim(0, max(max(means) * 1.25, 0.005))
        for bar, v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Phase 1d — MAD comparison: norm-weighted vs other attention signals",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    name = "mad_comparison.png"
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print(f"  saved: {name}")


# ── text summary ───────────────────────────────────────────────────────────────

def summarize(all_data: dict):
    print("\n=== PHASE 1d — NORM-WEIGHTED ATTENTION VALIDATION ===\n")

    def _print_mad_table(title: str, key: str):
        vals = {c: [] for c in CATS}
        rows_exist = False
        print(f"  -- {title} --")
        print(f"  {'Model':12}  {'Scen':5}  " + "  ".join(f"{l:>8}" for l in LABELS))
        for ml, model_data in all_data.items():
            for s_idx, data in model_data.items():
                if key not in data:
                    continue
                rows_exist = True
                mads = [data[key].get(c, None) for c in CATS]
                row  = "  ".join(f"{m:8.4f}" if m is not None else f"{'N/A':>8}" for m in mads)
                print(f"  {ml:12}  s{s_idx}     {row}")
                for c, m in zip(CATS, mads):
                    if m is not None:
                        vals[c].append(m)
        if not rows_exist:
            print("  (no data)")
            return None, None
        means  = [np.mean(vals[c]) if vals[c] else None for c in CATS]
        g_mad  = float(np.mean([m for m in means if m is not None]))
        row    = "  ".join(f"{m:8.4f}" if m is not None else f"{'N/A':>8}" for m in means)
        print(f"  {'Overall':12}  {'':5}  {row}")
        print(f"  Global MAD: {g_mad:.4f}\n")
        return means, g_mad

    _, mad_nw_raw     = _print_mad_table("Norm-weighted vs Raw",                 "mad_nw_vs_raw")
    _, mad_nw_rollout = _print_mad_table("Norm-weighted vs Rollout",             "mad_nw_vs_rollout")
    _, mad_1a_1b      = _print_mad_table("Option 1a vs 1b  (W_O effect)",        "mad_1a_vs_1b")
    _print_mad_table("Rollout vs Raw  (reference from Phase 1c)",                "mad_rollout_vs_raw")

    # CV diagnostic
    print("  -- Value-vector norm statistics --")
    print(f"  {'Model':12}  {'Scen':5}  {'CV(f_x)':>10}  {'CV(f_v)':>10}  {'sanity_dev':>12}")
    for ml, model_data in all_data.items():
        for s_idx, data in model_data.items():
            cv_fx  = data.get("f_x_norms_stats", {}).get("cv")
            cv_fv  = data.get("f_v_norms_stats", {}).get("cv")
            dev    = data.get("nw_row_sum_max_dev")
            cv_fx_s  = f"{cv_fx:>10.3f}" if cv_fx  is not None else f"{'N/A':>10}"
            cv_fv_s  = f"{cv_fv:>10.3f}" if cv_fv  is not None else f"{'N/A':>10}"
            dev_s    = f"{dev:>12.2e}"    if dev    is not None else f"{'N/A':>12}"
            print(f"  {ml:12}  s{s_idx}     {cv_fx_s}  {cv_fv_s}  {dev_s}")

    if mad_nw_rollout is not None:
        print(f"\n  Verdict (MAD norm_weighted vs rollout = {mad_nw_rollout:.4f}):")
        if mad_nw_rollout < 0.01:
            print("  → Norm-weighted is nearly identical to rollout at the category level.")
            print("    The W_O output projection does not substantially re-rank tokens.")
            print("    Decision: skip Phase 3 integration; norm-weighted adds no information.")
        elif mad_nw_rollout < 0.03:
            print("  → Moderate difference. Norm-weighted is meaningfully distinct from rollout.")
            print("    Worth integrating into Phase 3 as a parallel signal.")
        else:
            print("  → Large difference (MAD > 0.03). Norm-weighting substantially changes the signal.")
            print("    Integrate into Phase 3; this is the key Kobayashi finding.")

    return mad_nw_raw, mad_nw_rollout, mad_1a_1b


def write_findings(all_data: dict, mad_nw_raw, mad_nw_rollout, mad_1a_1b):
    lines = ["# Phase 1d Findings — Norm-Weighted Attention\n\n"]
    lines.append(
        "## Method\n\n"
        "Kobayashi et al. (EMNLP 2020) propose weighting each input token's attention "
        "by the L2 norm of its value-transformed representation:\n\n"
        "```\n"
        "f(x_j) = v_j @ W_O + b_O        (Option 1a, with output projection)\n"
        "norm_weighted[i,j] = α[i,j] × ‖f(x_j)‖₂   (then row-normalised)\n"
        "```\n\n"
        "Option 1b (without W_O): use ‖v_j‖₂ directly.\n\n"
    )

    def _mad_table(title, key):
        lines.append(f"## {title}\n\n")
        lines.append("| Category | " + " | ".join(LABELS) + " | Global |\n")
        lines.append("|---" * (len(CATS) + 2) + "|\n")
        vals = {c: [] for c in CATS}
        for model_data in all_data.values():
            for data in model_data.values():
                if key not in data:
                    continue
                for c in CATS:
                    v = data[key].get(c)
                    if v is not None:
                        vals[c].append(v)
        means = [np.mean(vals[c]) if vals[c] else None for c in CATS]
        row   = " | ".join(f"{m:.4f}" if m is not None else "N/A" for m in means)
        g     = np.mean([m for m in means if m is not None]) if any(m is not None for m in means) else None
        lines.append(f"| Mean | {row} | {g:.4f if g is not None else 'N/A'} |\n\n")

    _mad_table("MAD: Norm-weighted vs Raw",    "mad_nw_vs_raw")
    _mad_table("MAD: Norm-weighted vs Rollout", "mad_nw_vs_rollout")
    _mad_table("MAD: Option 1a vs 1b",         "mad_1a_vs_1b")

    path = OUT / "findings.md"
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"  saved: findings.md")


# ── persist/resume helpers ─────────────────────────────────────────────────────

def save_model_results(model_key: str, model_data: dict):
    serialisable = {}
    for s_idx, d in model_data.items():
        serialisable[str(s_idx)] = {k: v for k, v in d.items()
                                    if not k.endswith("_series")}
    path = OUT / f"results_{model_key}.json"
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"  saved: results_{model_key}.json")


def load_existing_results() -> dict:
    results = {}
    for model_key in MODELS:
        p = OUT / f"results_{model_key}.json"
        if p.exists():
            with open(p) as f:
                results[model_key] = {int(k): v for k, v in json.load(f).items()}
    return results


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 1d — norm-weighted attention")
    parser.add_argument("--model", choices=["complete", "minimal"], required=True,
                        help="Which model to run (Waymax: one per process).")
    args = parser.parse_args()
    model_key  = args.model
    model_dir, model_slug = MODELS[model_key]

    print(f"Phase 1d — Norm-weighted attention comparison  [{model_key}]")
    print(f"Output: {OUT}\n")

    model_path = str(_CBM / "runs_rlc" / model_dir)
    print(f"Loading model: {model_dir}")
    model = xai.load_model(model_path, data_path=DATA_PATH)
    print(f"  Model loaded. has_attention={model.has_attention}\n")

    model_data = {}
    for s_idx in SCENARIOS:
        print(f"  Scenario {s_idx}...")
        try:
            data = analyse_scenario(model, model_slug, s_idx)
            model_data[s_idx] = data

            # Quick sanity report
            has_nw  = data.get("has_norm_weighted", False)
            dev     = data.get("nw_row_sum_max_dev")
            cv_fx   = data.get("f_x_norms_stats", {}).get("cv")
            print(f"    T={data['T']}, has_nw={has_nw}, "
                  f"row_sum_dev={dev:.2e if dev is not None else 'N/A'}, "
                  f"CV(f_x)={cv_fx:.3f if cv_fx is not None else 'N/A'}")

            if has_nw:
                nw_vs_raw     = data.get("mad_nw_vs_raw",     {})
                nw_vs_rollout = data.get("mad_nw_vs_rollout", {})
                print(f"    MAD(NW vs raw)={global_mad(nw_vs_raw):.4f}  "
                      f"MAD(NW vs rollout)={global_mad(nw_vs_rollout):.4f}")

            plot_four_way(data, model_key, s_idx)
            plot_delta_nw_vs_rollout(data, model_key, s_idx)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()

    save_model_results(model_key, model_data)

    # Combined summary if both models are done
    all_data = load_existing_results()
    # Include current run's series data (not serialised to JSON)
    all_data[model_key] = model_data

    if len(all_data) == len(MODELS):
        print("\nBoth models done — computing combined summary...")
        mad_nw_raw, mad_nw_rollout, mad_1a_1b = summarize(all_data)
        write_findings(all_data, mad_nw_raw, mad_nw_rollout, mad_1a_1b)
        plot_mad_summary(all_data)
    else:
        remaining = [k for k in MODELS if k not in all_data]
        print(f"\nRun next: python phase1d_norm_weighted_comparison.py --model {remaining[0]}")

    print("\nDone. Review phase1d_results/ for all outputs.")


if __name__ == "__main__":
    main()
