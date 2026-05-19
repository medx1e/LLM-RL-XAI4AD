# Copyright 2025 Valeo.

"""SAE Hyperparameter Tuner.

Runs a grid search over the tuning values defined in the [tune] section of
sae_config.yaml, calling train() from sae_trainer.py for each combination.
Results are written to a CSV so you can inspect the L0/dead-neuron trade-off
across runs and pick the best checkpoint.

Usage:
    python -m sae_interpretability.sae_tuner \\
        --data  data/sae_interpretability/harvest.h5 \\
        --config sae_interpretability/sae_config.yaml \\
        --output_dir runs/sae_tuning
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import time
from copy import deepcopy
from typing import Any, Dict, List

import yaml

from sae_interpretability.config import SAEConfig
from sae_interpretability.sae_trainer import train


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_tune_grid(config_path: str) -> Dict[str, List[Any]]:
    """Read the 'tune' block from sae_config.yaml.

    Returns a dict mapping parameter name → list of values to try.
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    tune_block = raw.get("tune", {})
    if not tune_block:
        raise ValueError(
            "No 'tune:' block found in config. "
            "Add candidate values under the 'tune:' key (see sae_config.yaml)."
        )
    return tune_block


def _build_configs(base_cfg: SAEConfig,
                   grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Cartesian product of all grid values → list of override dicts."""
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    return [dict(zip(keys, combo)) for combo in combos]


def _apply_overrides(base_cfg: SAEConfig,
                     overrides: Dict[str, Any]) -> SAEConfig:
    """Return a new SAEConfig with the override values applied."""
    cfg = deepcopy(base_cfg)
    # SAEConfig fields map 1-to-1 to the override keys defined in sae_config.yaml
    _FIELD_MAP = {
        "l1_coeff":        "sae_l1_coeff",
        "expansion_factor": "sae_expansion_factor",
        "learning_rate":   "sae_learning_rate",
        "batch_size":      "sae_batch_size",
        "epochs":          "sae_epochs",
    }
    for key, val in overrides.items():
        attr = _FIELD_MAP.get(key, key)   # fall back to raw key
        if hasattr(cfg, attr):
            setattr(cfg, attr, val)
        else:
            print(f"  [Tuner] Warning: unknown config field '{attr}', skipping.")
    return cfg


# ---------------------------------------------------------------------------
# Main tuning loop
# ---------------------------------------------------------------------------

def run_tuning(
    data_path: str,
    config_path: str,
    output_dir: str,
) -> None:
    """Run the full grid search and write results to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "tuning_results.csv")

    # Load base config
    base_cfg = SAEConfig.from_yaml(config_path)

    # Load tuning grid
    grid = _load_tune_grid(config_path)
    combos = _build_configs(base_cfg, grid)

    print(f"[Tuner] Grid search: {len(combos)} combinations")
    print(f"[Tuner] Parameters : {list(grid.keys())}")
    print(f"[Tuner] Output dir : {output_dir}\n")

    results = []
    best_global_loss = float("inf")
    best_global_path = ""

    fieldnames = list(grid.keys()) + [
        "final_loss", "best_l0", "best_dead_pct", "best_epoch", "run_time_s", "checkpoint"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for run_idx, overrides in enumerate(combos, start=1):
        cfg = _apply_overrides(base_cfg, overrides)

        # Build a descriptive run name from the overrides
        tag = "_".join(f"{k}={v}" for k, v in overrides.items())
        ckpt_path = os.path.join(output_dir, f"run{run_idx:03d}_{tag}.pt")

        print(f"{'─'*60}")
        print(f"[Tuner] Run {run_idx}/{len(combos)}  →  {tag}")
        print(f"{'─'*60}")

        t0 = time.time()
        try:
            # Each run gets its own TensorBoard sub-directory
            tb_dir = os.path.join(output_dir, f"tb_run{run_idx:03d}_{tag}")
            model = train(data_path, ckpt_path, cfg, log_dir=tb_dir)

            # Read the best loss and L0 back from the saved checkpoint
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            final_loss = float(ckpt.get("loss", float("inf")))
            best_l0 = float(ckpt.get("best_l0", -1))
            best_dead_pct = float(ckpt.get("best_dead_pct", -1))
            best_epoch = int(ckpt.get("epoch", cfg.sae_epochs))
        except Exception as e:
            print(f"[Tuner] Run {run_idx} FAILED: {e}")
            final_loss = float("inf")
            best_l0 = -1.0
            best_dead_pct = -1.0
            best_epoch = -1

        elapsed = time.time() - t0

        row = {**overrides,
               "final_loss": f"{final_loss:.6f}",
               "best_l0":    f"{best_l0:.1f}",
               "best_dead_pct": f"{best_dead_pct:.1f}",
               "best_epoch":  best_epoch,
               "run_time_s":  f"{elapsed:.1f}",
               "checkpoint":  ckpt_path}
        results.append(row)

        # Append to CSV immediately (crash-safe)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        if final_loss < best_global_loss:
            best_global_loss = final_loss
            best_global_path = ckpt_path

        print(f"\n[Tuner] Run {run_idx} done  loss={final_loss:.6f}  "
              f"L0={best_l0:.1f}  dead={best_dead_pct:.1f}%  elapsed={elapsed:.1f}s\n")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"{'='*60}")
    print(f"[Tuner] Grid search complete.  {len(combos)} runs.")
    print(f"[Tuner] Results CSV  → {csv_path}")
    print(f"[Tuner] Best run     → {best_global_path}")
    print(f"[Tuner] Best loss    → {best_global_loss:.6f}")
    print(f"{'='*60}")

    results_sorted = sorted(results, key=lambda x: float(x["final_loss"]))
    print(f"\n{'Rank':<5} {'Loss':<12} {'L0':<8} {'Dead%':<8} Config")
    for rank, r in enumerate(results_sorted[:10], 1):
        cfg_str = "  ".join(f"{k}={r[k]}" for k in grid.keys())
        print(f"{rank:<5} {r['final_loss']:<12} {r['best_l0']:<8} {r['best_dead_pct']:<8} {cfg_str}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Grid-search hyperparameters for SAE training."
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to harvest.h5 (output of harvester.py)")
    parser.add_argument(
        "--config",
        default="sae_interpretability/sae_config.yaml",
        help="Path to sae_config.yaml containing the 'tune:' block")
    parser.add_argument(
        "--output_dir",
        default="runs/sae_tuning",
        help="Directory to save per-run checkpoints and results CSV")
    args = parser.parse_args()

    run_tuning(args.data, args.config, args.output_dir)


if __name__ == "__main__":
    main()
