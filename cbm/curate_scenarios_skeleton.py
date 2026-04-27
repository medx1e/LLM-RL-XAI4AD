"""
Skeleton Script for Pre-computing / Curating Scenarios
======================================================
Run this to pre-compute rollouts for "golden" scenarios and save
the raw data to disk. The Streamlit UI then loads instantly.

Usage:
    python curate_scenarios_skeleton.py
"""

import os
import sys
import pickle
from pathlib import Path

# Provide access to bev_visualizer package
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "V-Max") not in sys.path:
    sys.path.insert(0, str(_ROOT / "V-Max"))

from bev_visualizer import run_rollout

# ── Configuration ─────────────────────────────────────────────────────────────
TARGET_SCENARIOS = [0, 1, 2]
DATA_PATH        = "data/training.tfrecord"
OUTPUT_DIR       = _ROOT / "curated_scenarios"

MODELS_TO_CACHE = [
    "SAC Complete (WOMD seed 42)",
    "SAC Minimal (WOMD seed 42)",
]

def main():
    print("==============================================")
    print(" Starting Skeleton Headless Curation Engine")
    print("==============================================")

    for model_key in MODELS_TO_CACHE:
        model_dir_name = model_key.replace(" ", "_").replace("/", "-")
        output_dir = OUTPUT_DIR / model_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*50}")
        print(f" Model: {model_key}")
        print(f"{'='*50}")

        for idx in TARGET_SCENARIOS:
            out_file = output_dir / f"scenario_{idx:04d}_cache.pkl"

            if out_file.exists():
                print(f"\n[SKIP] Scenario {idx} already cached → {out_file.name}")
                continue

            print(f"\n[EVAL] Running rollout for Scenario {idx}...")

            scenario_data = run_rollout(
                model_key=model_key,
                data_path=DATA_PATH,
                scenario_idx=idx,
                num_steps=80,
            )

            print(f"[SAVE] Saving → {out_file.name}...")
            with open(out_file, "wb") as f:
                pickle.dump(scenario_data, f)

            print(f"       ✅ Cached scenario {idx} successfully.")

    print("\n==============================================")
    print(" Curation Complete.")
    print(f" Check {OUTPUT_DIR} for cached artifacts.")
    print("==============================================")

if __name__ == "__main__":
    main()
