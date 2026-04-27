"""Phase 1b — Attention aggregation comparison.

Loads the platform_cache attention pkl files (2 models × 3 scenarios, full
episode length T) and compares three query aggregation strategies:
  mean     — current default (average over 16 queries)
  maxpool  — max over 16 queries per token
  entropy  — sharpness-weighted average (focused queries matter more)

Also analyses query specialization: entropy per query to identify whether
specific queries focus on specific categories.

Run in the vmax conda environment:
  conda activate vmax
  cd /home/med1e/platform_fyp/post-hoc-xai
  python experiments/phase1b_aggregation_comparison.py

Output: experiments/phase1b_results/
"""

import sys
import pickle
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths & imports ───────────────────────────────────────────────────────────
_HERE  = Path(__file__).parent
_ROOT  = _HERE.parent
_CACHE = _ROOT.parent / "platform_cache"
OUT    = _HERE / "phase1b_results"
OUT.mkdir(exist_ok=True)

# Load aggregation module directly (avoids posthoc_xai JAX __init__)
_agg_spec = importlib.util.spec_from_file_location(
    "attention_aggregation",
    _ROOT / "posthoc_xai" / "utils" / "attention_aggregation.py",
)
_agg = importlib.util.module_from_spec(_agg_spec)
_agg_spec.loader.exec_module(_agg)

aggregate_attention_all = _agg.aggregate_attention_all
query_entropy           = _agg.query_entropy
TOKEN_RANGES            = _agg.TOKEN_RANGES

CATS   = list(TOKEN_RANGES.keys())
LABELS = ["SDC", "Agents", "Road", "TL", "GPS"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
METHODS = ["mean", "maxpool", "entropy"]
METHOD_COLORS = {"mean": "#2196F3", "maxpool": "#FF9800", "entropy": "#9C27B0"}

MODELS = {
    "complete": "SAC_Perceiver_Complete_WOMD_seed_42",
    "minimal":  "SAC_Perceiver_WOMD_seed_42",
}
SCENARIOS = [1, 2, 3]


# ── data loading ─────────────────────────────────────────────────────────────

def load_attention_series(model_slug: str, scenario_idx: int) -> list[dict]:
    """Load attention pkl → list[dict], each with 'cross_attn_avg' (1,16,280)."""
    path = _CACHE / model_slug / f"scenario_{scenario_idx:04d}_attention.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_matrix(attn_dict: dict) -> np.ndarray:
    """Get the (16, 280) cross_attn_avg from one timestep dict."""
    arr = attn_dict.get("cross_attn_avg")
    if arr is None:
        raise KeyError("cross_attn_avg not found in attention dict")
    return np.array(arr)  # (1, 16, 280) or (16, 280)


# ── per-scenario analysis ─────────────────────────────────────────────────────

def analyse_scenario(
    model_slug: str,
    scenario_idx: int,
) -> dict:
    """Run all aggregation methods on every timestep of one scenario.

    Returns:
        {
          'T': int,
          'results': {method: {cat: list[float] over T}},
          'entropies': np.ndarray (T, 16)   — per-timestep per-query entropy
        }
    """
    series = load_attention_series(model_slug, scenario_idx)
    T = len(series)

    # Collect per-method, per-category time series
    results = {m: {c: [] for c in CATS} for m in METHODS}
    entropies = []

    for attn_dict in series:
        attn = extract_matrix(attn_dict)       # (1,16,280) → squeezes inside fn
        agg  = aggregate_attention_all(attn)   # {method: {cat: float}}
        for m in METHODS:
            for c in CATS:
                results[m][c].append(agg[m][c])
        entropies.append(query_entropy(attn))  # (16,)

    return {
        "T": T,
        "results": results,
        "entropies": np.array(entropies),  # (T, 16)
    }


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_method_comparison_timeseries(
    data: dict,
    model_label: str,
    scenario_idx: int,
):
    """Line plot: category importance over time for each aggregation method."""
    T = data["T"]
    ts = list(range(T))
    results = data["results"]

    fig, axes = plt.subplots(1, len(METHODS), figsize=(5 * len(METHODS), 4),
                             sharey=True)
    for ax, method in zip(axes, METHODS):
        for cat, lbl, col in zip(CATS, LABELS, COLORS):
            ax.plot(ts, results[method][cat], label=lbl, color=col, linewidth=1.5)
        ax.set_title(f"{method}", fontsize=10)
        ax.set_xlabel("Timestep")
        ax.set_ylim(0, 1)
        if ax is axes[0]:
            ax.set_ylabel("Attention fraction")
        ax.legend(fontsize=7)

    fig.suptitle(
        f"{model_label} — Scenario {scenario_idx}  |  Method comparison",
        fontsize=10, fontweight="bold"
    )
    fig.tight_layout()
    name = f"timeseries_{model_label}_s{scenario_idx:02d}.png"
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print(f"  saved: {name}")


def plot_method_delta(
    data: dict,
    model_label: str,
    scenario_idx: int,
):
    """Show how maxpool and entropy differ from mean (delta over time)."""
    T = data["T"]
    ts = list(range(T))
    results = data["results"]

    fig, axes = plt.subplots(len(CATS), 2, figsize=(10, 2.5 * len(CATS)), sharey=False)
    for row, (cat, lbl) in enumerate(zip(CATS, LABELS)):
        mean_arr = np.array(results["mean"][cat])
        for col, alt_method in enumerate(["maxpool", "entropy"]):
            alt_arr  = np.array(results[alt_method][cat])
            delta    = alt_arr - mean_arr
            ax = axes[row][col]
            ax.plot(ts, delta, color=METHOD_COLORS[alt_method], linewidth=1.2)
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
            ax.set_title(f"{lbl}  |  {alt_method} − mean", fontsize=8)
            ax.set_xlabel("Timestep", fontsize=7)
            ax.set_ylabel("Δ importance", fontsize=7)
            # Mark mean absolute delta
            mad = np.mean(np.abs(delta))
            ax.text(0.98, 0.95, f"MAD={mad:.3f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7)

    fig.suptitle(
        f"{model_label} — Scenario {scenario_idx}  |  Delta from mean-pool",
        fontsize=10, fontweight="bold"
    )
    fig.tight_layout()
    name = f"delta_{model_label}_s{scenario_idx:02d}.png"
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print(f"  saved: {name}")


def plot_query_entropy(
    data: dict,
    model_label: str,
    scenario_idx: int,
):
    """Visualise per-query entropy over time to detect query specialization."""
    entropies = data["entropies"]   # (T, 16)
    T, Q = entropies.shape
    ts = list(range(T))
    max_entropy = np.log2(280)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: heatmap (queries × timesteps)
    im = ax1.imshow(
        entropies.T,           # (16, T)
        aspect="auto",
        origin="lower",
        cmap="viridis_r",      # reversed: dark = low entropy = focused
        vmin=0, vmax=max_entropy,
    )
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Query index")
    ax1.set_title("Query entropy over time\n(dark = focused, light = diffuse)")
    plt.colorbar(im, ax=ax1, label="Entropy (bits)")

    # Right: mean entropy per query + std
    mean_H = entropies.mean(axis=0)  # (16,)
    std_H  = entropies.std(axis=0)
    sorted_idx = np.argsort(mean_H)
    ax2.barh(
        range(Q), mean_H[sorted_idx],
        xerr=std_H[sorted_idx], capsize=3,
        color="#607D8B", alpha=0.8, ecolor="gray",
    )
    ax2.axvline(max_entropy, color="red", linestyle="--", linewidth=0.8,
                label=f"Max entropy ({max_entropy:.1f} bits)")
    ax2.set_yticks(range(Q))
    ax2.set_yticklabels([f"q{sorted_idx[i]}" for i in range(Q)], fontsize=7)
    ax2.set_xlabel("Mean entropy (bits)")
    ax2.set_title("Per-query mean entropy (sorted)\nLow = specialized")
    ax2.legend(fontsize=7)

    fig.suptitle(
        f"{model_label} — Scenario {scenario_idx}  |  Query specialization",
        fontsize=10, fontweight="bold"
    )
    fig.tight_layout()
    name = f"query_entropy_{model_label}_s{scenario_idx:02d}.png"
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print(f"  saved: {name}")


# ── numerical summary ─────────────────────────────────────────────────────────

def compute_global_summary(all_data: dict) -> dict:
    """Aggregate MAD (mean absolute deviation from mean-pool) across all scenarios."""
    # all_data: {model_label: {scenario_idx: data_dict}}
    summary = {m: {c: [] for c in CATS} for m in ["maxpool", "entropy"]}

    for model_data in all_data.values():
        for data in model_data.values():
            results = data["results"]
            mean_arr = {c: np.array(results["mean"][c]) for c in CATS}
            for alt in ["maxpool", "entropy"]:
                for c in CATS:
                    alt_arr = np.array(results[alt][c])
                    mad = float(np.mean(np.abs(alt_arr - mean_arr[c])))
                    summary[alt][c].append(mad)

    # Average MAD across all scenarios/models
    averaged = {}
    for alt in ["maxpool", "entropy"]:
        averaged[alt] = {c: float(np.mean(summary[alt][c])) for c in CATS}
    return averaged


def print_summary(all_data: dict):
    print("\n=== AGGREGATION METHOD COMPARISON ===")
    print("Mean Absolute Deviation from mean-pool (category-level)\n")
    global_mad = compute_global_summary(all_data)

    print(f"  {'Category':20}  {'maxpool MAD':>12}  {'entropy MAD':>12}")
    for cat, lbl in zip(CATS, LABELS):
        mp = global_mad["maxpool"][cat]
        ew = global_mad["entropy"][cat]
        print(f"  {lbl:20}  {mp:12.4f}  {ew:12.4f}")

    overall_mp = np.mean(list(global_mad["maxpool"].values()))
    overall_ew = np.mean(list(global_mad["entropy"].values()))
    print(f"\n  {'Overall':20}  {overall_mp:12.4f}  {overall_ew:12.4f}")

    # Query specialization summary
    print("\n=== QUERY SPECIALIZATION ===")
    for model_label, model_data in all_data.items():
        for s_idx, data in model_data.items():
            H = data["entropies"]           # (T, 16)
            mean_H = H.mean(axis=0)         # (16,)
            max_entropy = np.log2(280)
            focused = (mean_H < max_entropy * 0.6).sum()  # entropy < 60% of max
            print(f"  {model_label} s{s_idx}: "
                  f"mean H={mean_H.mean():.2f} bits  "
                  f"min H={mean_H.min():.2f} (q{mean_H.argmin()})  "
                  f"queries below 60% max: {focused}/16")


def write_findings(all_data: dict):
    global_mad = compute_global_summary(all_data)
    overall_mp = float(np.mean(list(global_mad["maxpool"].values())))
    overall_ew = float(np.mean(list(global_mad["entropy"].values())))

    lines = ["# Phase 1b Findings — Attention Aggregation Comparison\n\n"]
    lines.append(
        "## What we compared\n\n"
        "Three strategies for collapsing the Perceiver's `(16, 280)` cross-attention "
        "matrix into a 5-dim category vector:\n\n"
        "- **mean** (current default): average over 16 queries\n"
        "- **maxpool**: max over 16 queries per token (best-case signal per token)\n"
        "- **entropy**: weight queries by sharpness (1/H) before averaging\n\n"
    )

    lines.append("## Mean Absolute Deviation from mean-pool\n\n")
    lines.append("| Category | maxpool MAD | entropy MAD |\n|---|---|---|\n")
    for cat, lbl in zip(CATS, LABELS):
        lines.append(
            f"| {lbl} | {global_mad['maxpool'][cat]:.4f} | {global_mad['entropy'][cat]:.4f} |\n"
        )
    lines.append(f"| **Overall** | **{overall_mp:.4f}** | **{overall_ew:.4f}** |\n\n")

    # Determine recommendation
    if overall_mp < 0.03 and overall_ew < 0.03:
        decision = (
            "Both alternatives deviate <0.03 from mean-pool on average. "
            "**Keep mean-pool** as the canonical method — the alternatives add "
            "complexity without meaningfully changing results."
        )
    elif overall_mp > 0.05:
        decision = (
            "Max-pool deviates significantly (>0.05) from mean-pool, indicating "
            "genuine specialization in some queries. **Use maxpool** for the "
            "attention-IG correlation experiment to capture the full per-query signal."
        )
    else:
        decision = (
            "Deviations are moderate. Proceed with mean-pool as primary; report "
            "maxpool as robustness check in the thesis."
        )

    lines.append(f"## Decision\n\n{decision}\n\n")

    lines.append(
        "## What to write in the thesis\n\n"
        "> We evaluated three strategies for aggregating the Perceiver's 16 learned "
        "query vectors into a single per-category attention signal: query mean-pooling "
        "(current default), query max-pooling, and sharpness-weighted averaging "
        "(queries with lower attention entropy contribute more). "
        f"The mean absolute deviation between methods was {overall_mp:.3f} (maxpool) "
        f"and {overall_ew:.3f} (entropy-weighted), indicating [INSERT CONCLUSION: "
        "negligible/moderate] sensitivity to the aggregation choice. We use "
        "**mean-pooling** as the primary metric for consistency with the reward-"
        "conditioned attention study, and report maxpool results as a robustness check.\n"
    )

    path = OUT / "findings.md"
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"  saved: findings.md")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("Phase 1b — Attention aggregation comparison")
    print(f"Output: {OUT}\n")

    all_data = {}

    for model_key, model_slug in MODELS.items():
        print(f"\nModel: {model_slug}")
        all_data[model_key] = {}

        for s_idx in SCENARIOS:
            print(f"  Scenario {s_idx}...")
            try:
                data = analyse_scenario(model_slug, s_idx)
                all_data[model_key][s_idx] = data
                print(f"    T={data['T']} timesteps")

                plot_method_comparison_timeseries(data, model_key, s_idx)
                plot_method_delta(data, model_key, s_idx)
                plot_query_entropy(data, model_key, s_idx)

            except FileNotFoundError as e:
                print(f"    SKIP: {e}")

    print_summary(all_data)
    write_findings(all_data)
    print("\nDone. Review phase1b_results/ for all outputs.")


if __name__ == "__main__":
    main()
