"""
XAI Scenario Miner
==================
Runs the CBM model over a batch of scenarios and automatically tags them
based on a causal chain of Concept (Perception) -> Action (Decision) -> Outcome.

Outputs a JSON index that the Streamlit UI can use to filter scenarios.
"""

import sys
import os
import json
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "V-Max") not in sys.path:
    sys.path.insert(0, str(_ROOT / "V-Max"))

from bev_visualizer.rollout_engine import run_rollout

# ── Config
NUM_SCENARIOS = 50
MODEL_KEY = "CBM-V2 Frozen — 150GB (15 concepts)"
DATA_PATH = "data/training.tfrecord"
OUTPUT_INDEX = _ROOT / "XAI_curated_index.json"

def classify_scenario(data) -> list[str]:
    """Returns a list of tags for a given ScenarioData object."""
    tags = []
    
    if data.pred_concepts is None:
        return ["Baseline Model (No Concepts)"]
        
    names = data.concept_names
    T = len(data.dones)
    
    # Early stop detection
    n_frames = len(data.frame_states)
    dones = data.dones.flatten()
    first_done = int(dones.argmax()) if dones.max() > 0.5 else n_frames
    crashed = first_done < (n_frames - 1)
    
    # Extract concepts
    def c(name, source="true"):
        if name not in names: return np.zeros(T)
        idx = names.index(name)
        return data.true_concepts[:, idx] if source == "true" else data.pred_concepts[:, idx]

    tl_red_true = c("traffic_light_red", "true")
    tl_red_pred = c("traffic_light_red", "pred")
    dist_obj = c("dist_nearest_object", "true")
    path_curve = c("path_curvature_max", "true")
    
    speed = c("ego_speed", "true") # normalized [0, 1]
    accel = data.ego_actions[:, 0]
    steer = data.ego_actions[:, 1]
    
    # ── Heuristic: Rear-end Collision (Ignored Concept)
    if crashed and dist_obj[first_done] < 0.05:
        # Check if they failed to brake
        avg_accel_before_crash = accel[max(0, first_done-10):first_done].mean()
        if avg_accel_before_crash > -0.2:
            tags.append("Rear-end Collision (Failed to brake)")
        else:
            tags.append("Rear-end Collision (Braked too late)")

    # ── Heuristic: Red Light Stop
    if (tl_red_true > 0.8).any():
        if not crashed and speed[-1] < 0.05:
            tags.append("Correct Red Light Stop")
        elif crashed:
            tags.append("Red Light Violation Collision")

    # ── Heuristic: Off-road Curve Failure
    if crashed and (path_curve > 0.5).any():
        if "Rear-end Collision" not in " ".join(tags):
            tags.append("Off-road Curve Failure")

    # ── Heuristic: Perfect Navigation
    if not crashed:
        progress = c("progress_along_route", "true")
        if progress[-1] > 0.8:
            if not tags: # if it didn't trigger a red light stop
                tags.append("Perfect Route Navigation")

    if not tags:
        tags.append("Untagged")
        
    return tags

def main():
    print("==============================================")
    print(" Starting XAI Causal Miner")
    print("==============================================")
    
    index = {} # tag -> list of scenarios
    
    for idx in range(NUM_SCENARIOS):
        print(f"\n[MINER] Scraping Scenario {idx} / {NUM_SCENARIOS}...")
        try:
            data = run_rollout(
                model_key=MODEL_KEY,
                data_path=DATA_PATH,
                scenario_idx=idx,
                num_steps=80
            )
            
            tags = classify_scenario(data)
            print(f"        -> Identified Tags: {tags}")
            
            for t in tags:
                if t not in index:
                    index[t] = []
                index[t].append(idx)
                
        except Exception as e:
            print(f"        -> Evaluator crashed on scenario {idx}: {e}")

    print("\n==============================================")
    print(" Summary of Mined Categories:")
    for tag, scens in index.items():
        print(f"  - {tag}: {len(scens)} examples")
        
    print(f"\n[SAVE] Exporting index to {OUTPUT_INDEX.name}")
    with open(OUTPUT_INDEX, "w") as f:
        json.dump(index, f, indent=4)

if __name__ == "__main__":
    main()
