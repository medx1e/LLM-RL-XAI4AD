"""Phase 1c — Attention rollout comparison.

Loads platform_cache scenarios, re-runs the forward pass with the updated
_extract_attention() that now returns self-attention weights and cross_attn_rollout,
then compares rollout vs raw cross_attn_avg at the category level.

Run in the vmax conda environment (one model per run — Waymax registry constraint):
  conda activate vmax
  cd /home/med1e/platform_fyp/post-hoc-xai
  python experiments/phase1c_rollout_comparison.py --model complete
  python experiments/phase1c_rollout_comparison.py --model minimal

Output: experiments/phase1c_results/
"""

import sys
import argparse
import importlib.util
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).parent
_ROOT     = _HERE.parent
_CBM      = _ROOT.parent / "cbm"
_CACHE    = _ROOT.parent / "platform_cache"
_PLATFORM = _ROOT.parent   # platform_fyp/ — needed for unpickling platform.shared.*
OUT       = _HERE / "phase1c_results"
OUT.mkdir(exist_ok=True)

sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_CBM))
sys.path.insert(0, str(_CBM / "V-Max"))

# Load aggregation module directly
_agg_spec = importlib.util.spec_from_file_location(
    "attention_aggregation",
    _ROOT / "posthoc_xai" / "utils" / "attention_aggregation.py",
)
_agg = importlib.util.module_from_spec(_agg_spec)
_agg_spec.loader.exec_module(_agg)
aggregate_attention_all = _agg.aggregate_attention_all
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


# ── data helpers ──────────────────────────────────────────────────────────────

class _ArtifactStub:
    """Generic container that captures any pickled object's __dict__.

    Used to deserialise PlatformScenarioArtifact without importing the
    platform.shared package (which shadows stdlib 'platform' in this env).
    All fields — including raw_observations — are accessible as attributes.
    """
    def __setstate__(self, state):
        self.__dict__.update(state)


# Modules that require stubbing when loading platform_cache artifacts.
# - platform.shared / platform.posthoc: our custom package, shadows stdlib 'platform'
# - bev_visualizer: CBM internal, imports jax at module level (transitive issue)
# We only need raw_observations (numpy array at top-level), so stubbing these
# nested objects is safe — they are captured as generic attribute bags.
_STUB_MODULES = (
    "platform.shared", "platform.posthoc", "platform.tabs",
    "bev_visualizer",
)


class _PlatformUnpickler(pickle.Unpickler):
    """Stubs out platform.shared.* classes; lets all other classes load normally."""
    def find_class(self, module, name):
        if any(module.startswith(m) for m in _STUB_MODULES):
            return _ArtifactStub
        return super().find_class(module, name)


def load_artifact(model_slug: str, scenario_idx: int):
    path = _CACHE / model_slug / f"scenario_{scenario_idx:04d}_artifact.pkl"
    with open(path, "rb") as f:
        return _PlatformUnpickler(f).load()


def run_forward_pass(model, raw_obs_np: np.ndarray) -> dict:
    """Run all T observations through the updated forward pass.

    Returns dict:
        'cross_avg'    : (T, 16, 280)
        'rollout'      : (T, 16, 280)  — None if self-attention not captured
        'self_layers'  : list of (T, 16, 16) per self-attn layer
    """
    import jax.numpy as jnp
    import jax

    T = raw_obs_np.shape[0]
    obs_batch = jnp.array(raw_obs_np)       # (T, obs_dim)
    out = model.forward(obs_batch)
    attn = out.attention

    if attn is None:
        raise RuntimeError("No attention returned — wrong model type?")

    result = {
        "cross_avg":   np.array(attn["cross_attn_avg"]),      # (T, 16, 280)
        "rollout":     np.array(attn["cross_attn_rollout"]) if "cross_attn_rollout" in attn else None,
        "self_layers": [
            np.array(attn[f"self_attn_layer_{i}"])
            for i in range(4) if f"self_attn_layer_{i}" in attn
        ],
    }
    return result


# ── analysis ──────────────────────────────────────────────────────────────────

def analyse_scenario(model, model_slug: str, scenario_idx: int) -> dict:
    """Aggregate attention under raw and rollout for one scenario."""
    artifact = load_artifact(model_slug, scenario_idx)
    raw_obs  = artifact.raw_observations     # (T, obs_dim)
    if raw_obs is None:
        raise ValueError("No raw_observations in artifact")

    fwd = run_forward_pass(model, np.array(raw_obs))
    T   = fwd["cross_avg"].shape[0]

    raw_series     = {c: [] for c in CATS}
    rollout_series = {c: [] for c in CATS}

    for t in range(T):
        all_r = aggregate_attention_all(
            fwd["cross_avg"][t],
            fwd["rollout"][t] if fwd["rollout"] is not None else None,
        )
        for c in CATS:
            raw_series[c].append(all_r["mean"][c])
            if "rollout" in all_r:
                rollout_series[c].append(all_r["rollout"][c])

    # MAD between rollout and raw mean per category
    mad = {}
    for c in CATS:
        if rollout_series[c]:
            mad[c] = float(np.mean(np.abs(
                np.array(rollout_series[c]) - np.array(raw_series[c])
            )))
        else:
            mad[c] = None

    return {
        "T": T,
        "raw_series": raw_series,
        "rollout_series": rollout_series,
        "mad": mad,
        "has_rollout": fwd["rollout"] is not None,
        "n_self_layers": len(fwd["self_layers"]),
        "self_layers": fwd["self_layers"],   # list of (T, 16, 16)
    }


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_rollout_comparison(data: dict, model_label: str, scenario_idx: int):
    """Side-by-side: raw cross_attn_avg vs rollout category time series."""
    T  = data["T"]
    ts = list(range(T))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for cat, lbl, col in zip(CATS, LABELS, COLORS):
        ax1.plot(ts, data["raw_series"][cat],     label=lbl, color=col, lw=1.5)
        if data["rollout_series"][cat]:
            ax2.plot(ts, data["rollout_series"][cat], label=lbl, color=col, lw=1.5)

    ax1.set_title("Raw cross_attn_avg (mean-pool over queries)")
    ax2.set_title("Rollout-corrected attention (mean-pool over queries)")
    for ax in (ax1, ax2):
        ax.set_xlabel("Timestep"); ax.set_ylabel("Attention fraction")
        ax.set_ylim(0, 1); ax.legend(fontsize=7)

    fig.suptitle(f"{model_label} — Scenario {scenario_idx}  |  Rollout vs raw",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    name = f"rollout_vs_raw_{model_label}_s{scenario_idx:02d}.png"
    fig.savefig(OUT / name, dpi=150); plt.close(fig)
    print(f"  saved: {name}")


def plot_self_attn_heatmap(data: dict, model_label: str, scenario_idx: int):
    """Mean self-attention matrix (16×16) over all timesteps."""
    if not data["self_layers"]:
        return
    n_layers = data["n_self_layers"]
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    for i, (ax, layer_data) in enumerate(zip(axes, data["self_layers"])):
        mean_A = layer_data.mean(axis=0)   # (16, 16)
        im = ax.imshow(mean_A, cmap="Blues", vmin=0, vmax=mean_A.max())
        ax.set_title(f"Self-attn layer {i}", fontsize=9)
        ax.set_xlabel("Key query"); ax.set_ylabel("Query")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        f"{model_label} — Scenario {scenario_idx}  |  "
        "Mean self-attention (how queries mix)",
        fontsize=10, fontweight="bold"
    )
    fig.tight_layout()
    name = f"self_attn_heatmap_{model_label}_s{scenario_idx:02d}.png"
    fig.savefig(OUT / name, dpi=150); plt.close(fig)
    print(f"  saved: {name}")


def plot_delta(data: dict, model_label: str, scenario_idx: int):
    """Rollout − raw delta per category over time."""
    if not data["rollout_series"][CATS[0]]:
        return
    T  = data["T"]
    ts = list(range(T))

    fig, axes = plt.subplots(1, len(CATS), figsize=(3 * len(CATS), 3), sharey=True)
    for ax, cat, lbl, col in zip(axes, CATS, LABELS, COLORS):
        delta = np.array(data["rollout_series"][cat]) - np.array(data["raw_series"][cat])
        ax.plot(ts, delta, color=col, lw=1.2)
        ax.axhline(0, color="gray", lw=0.7, linestyle="--")
        ax.set_title(f"{lbl}\nMAD={data['mad'][cat]:.4f}", fontsize=8)
        ax.set_xlabel("Timestep", fontsize=7)

    axes[0].set_ylabel("Rollout − raw", fontsize=8)
    fig.suptitle(f"{model_label} — Scenario {scenario_idx}  |  Delta: rollout − raw",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    name = f"delta_rollout_{model_label}_s{scenario_idx:02d}.png"
    fig.savefig(OUT / name, dpi=150); plt.close(fig)
    print(f"  saved: {name}")


# ── summary ───────────────────────────────────────────────────────────────────

def summarize(all_data: dict):
    print("\n=== ROLLOUT vs RAW CROSS-ATTENTION ===")
    print("MAD (rollout − raw mean-pool) per category\n")
    print(f"  {'Model':12}  {'Scen':6}  " + "  ".join(f"{l:>8}" for l in LABELS))
    for ml, model_data in all_data.items():
        for s_idx, data in model_data.items():
            if not data["has_rollout"]:
                print(f"  {ml:12}  s{s_idx}     ROLLOUT NOT AVAILABLE")
                continue
            mads = [data["mad"][c] for c in CATS]
            row  = "  ".join(f"{m:8.4f}" if m is not None else f"{'N/A':>8}" for m in mads)
            print(f"  {ml:12}  s{s_idx}     {row}")

    # Overall
    all_mads = {c: [] for c in CATS}
    for model_data in all_data.values():
        for data in model_data.values():
            if data["has_rollout"]:
                for c in CATS:
                    if data["mad"][c] is not None:
                        all_mads[c].append(data["mad"][c])
    overall = {c: float(np.mean(v)) if v else None for c, v in all_mads.items()}
    row = "  ".join(f"{overall[c]:8.4f}" if overall[c] is not None else f"{'N/A':>8}" for c in CATS)
    print(f"\n  {'Overall':12}  {'':6}  {row}")
    global_mad = np.mean([v for v in overall.values() if v is not None])
    print(f"\n  Global MAD (rollout vs raw): {global_mad:.4f}")
    return overall, global_mad


def write_findings(all_data: dict, overall: dict, global_mad: float):
    lines = ["# Phase 1c Findings — Attention Rollout\n\n"]
    lines.append(
        "## What attention rollout does\n\n"
        "The Perceiver runs 4 blocks of: cross-attention (queries→tokens) + "
        "self-attention (queries→queries). Raw `cross_attn_avg` ignores that "
        "self-attention mixes information between queries after each cross-attention "
        "step. Rollout corrects for this by chaining residual-corrected self-attention "
        "matrices and applying the result to the mean cross-attention:\n\n"
        "```\n"
        "A_eff[l] = 0.5*I + 0.5*A_self[l]  (residual correction per layer)\n"
        "R        = A_eff[3] @ A_eff[2] @ A_eff[1] @ A_eff[0]  (16×16)\n"
        "rollout  = R @ cross_attn_avg                           (16×280)\n"
        "```\n\n"
    )
    lines.append("## MAD: rollout vs raw mean-pool\n\n")
    lines.append("| Category | Overall MAD |\n|---|---|\n")
    for cat, lbl in zip(CATS, LABELS):
        v = overall[cat]
        lines.append(f"| {lbl} | {v:.4f} |\n" if v else f"| {lbl} | N/A |\n")
    lines.append(f"| **Global** | **{global_mad:.4f}** |\n\n")

    if global_mad < 0.02:
        conclusion = (
            "Rollout barely changes the category-level signal (global MAD < 0.02). "
            "Self-attention mixes queries but does not substantially redistribute "
            "which input tokens influence the final representation. "
            "**Use raw cross_attn_avg as the canonical signal** — rollout adds "
            "complexity without meaningful benefit."
        )
    elif global_mad < 0.05:
        conclusion = (
            f"Rollout introduces a moderate shift (global MAD={global_mad:.3f}). "
            "Self-attention mixing is non-negligible but modest. Report rollout as "
            "a robustness check; use raw mean-pool as primary."
        )
    else:
        conclusion = (
            f"Rollout substantially changes the attention signal (global MAD={global_mad:.3f}). "
            "Self-attention layers significantly redistribute query information. "
            "**Prefer rollout** for the attention-IG correlation experiment."
        )
    lines.append(f"## Decision\n\n{conclusion}\n\n")

    lines.append(
        "## What to write in the thesis\n\n"
        "> **Methodological note — attention rollout:**\n"
        "> The Perceiver processes input tokens through 4 interleaved blocks of "
        "cross-attention (queries attend to input tokens) and self-attention "
        "(queries attend to each other). Raw cross-attention weights do not "
        "account for the information mixing that occurs in self-attention layers. "
        "We implemented attention rollout (Abnar & Zuidema 2020), which chains "
        "residual-corrected self-attention matrices to compute an effective "
        "attention from the final representation back to the input tokens. "
        f"The mean absolute deviation between rolled-out and raw attention was "
        f"{global_mad:.3f} at the category level, indicating [INSERT CONCLUSION]. "
        "We [use raw cross-attention / use rollout] as the primary attention "
        "signal throughout this chapter.\n"
    )

    path = OUT / "findings.md"
    with open(path, "w") as f:
        f.writelines(lines)
    print(f"  saved: findings.md")


# ── main ──────────────────────────────────────────────────────────────────────

def load_existing_results() -> dict:
    """Load per-model MAD JSONs written by previous runs."""
    import json
    results = {}
    for model_key in MODELS:
        p = OUT / f"results_{model_key}.json"
        if p.exists():
            with open(p) as f:
                results[model_key] = json.load(f)
    return results


def save_model_results(model_key: str, model_data: dict):
    import json
    p = OUT / f"results_{model_key}.json"
    # Convert non-serialisable values to plain Python
    serialisable = {}
    for s_idx, d in model_data.items():
        serialisable[str(s_idx)] = {
            "T": d["T"],
            "has_rollout": d["has_rollout"],
            "n_self_layers": d["n_self_layers"],
            "mad": d["mad"],
        }
    with open(p, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"  saved: results_{model_key}.json")


def main():
    parser = argparse.ArgumentParser(description="Phase 1c — attention rollout")
    parser.add_argument(
        "--model", choices=["complete", "minimal"], required=True,
        help="Which model to run (Waymax allows only one model per process).",
    )
    args = parser.parse_args()
    model_key = args.model
    model_dir, model_slug = MODELS[model_key]

    print(f"Phase 1c — Attention rollout comparison  [{model_key}]")
    print(f"Output: {OUT}\n")

    model_path = str(_CBM / "runs_rlc" / model_dir)
    print(f"Loading model: {model_dir}")
    model = xai.load_model(model_path, data_path=DATA_PATH)
    print(f"  Model loaded. has_attention={model.has_attention}")

    model_data = {}
    for s_idx in SCENARIOS:
        print(f"  Scenario {s_idx}...")
        try:
            data = analyse_scenario(model, model_slug, s_idx)
            model_data[s_idx] = data
            print(f"    T={data['T']}, "
                  f"has_rollout={data['has_rollout']}, "
                  f"self_layers={data['n_self_layers']}")
            plot_rollout_comparison(data, model_key, s_idx)
            plot_self_attn_heatmap(data, model_key, s_idx)
            plot_delta(data, model_key, s_idx)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()

    save_model_results(model_key, model_data)

    # If both models are done, print combined summary + write findings
    all_data = load_existing_results()
    all_data[model_key] = {
        int(k): v for k, v in
        {str(s): model_data[s] for s in model_data}.items()
    }
    if len(all_data) == len(MODELS):
        print("\nBoth models done — computing combined summary...")
        overall, global_mad = summarize(all_data)
        write_findings(all_data, overall, global_mad)
    else:
        remaining = [k for k in MODELS if k not in all_data]
        print(f"\nRun next: python phase1c_rollout_comparison.py --model {remaining[0]}")

    print("\nDone. Review phase1c_results/ for all outputs.")


if __name__ == "__main__":
    main()
