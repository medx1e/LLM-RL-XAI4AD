# Causal XAI Scenario Mining — Context & Definitions

This document details the logic behind our automated Scenario Miner (`mine_scenarios.py`).

## The Goal
Instead of manually searching through thousands of Waymax scenarios to find interesting edge cases, we built an automated miner. The core philosophy is to evaluate the model's behavior using a causal chain:
**Concept (What it saw) ➔ Action (What it did) ➔ Outcome (What happened).**

By analyzing this chain, we can automatically extract specific subsets of scenarios that demonstrate where our Concept Bottleneck Model (CBM) excels or fails.

---

## 1. Tracking the Variables

> **System Modification Note:** To make this causal analysis possible, the core `bev_visualizer/rollout_engine.py` was officially modified to pull data out of the GPU. Inside the `jax.lax.scan` loop, it now calls the model's `encode_and_predict_concepts()` method and the concept adapter `extract_all_concepts()`, returning them alongside `ego_actions` inside the `ScenarioData` object. 

Inside `rollout_engine.py`, the `ScenarioData` object was upgraded to track:

### A. The Outcomes
- `dones` (Shape: T) — 1.0 if the episode terminated early (usually due to a collision `overlap` or `offroad`).
- `rewards` (Shape: T) — The scalar reward per step.
- `ego_speed` / `ego_xy` / `agents_xy` — General physics metrics.

### B. The Actions
- `ego_actions` (Shape: T × Action_Dim) — The raw commands the model outputted. In `InvertibleBicycleModel`, this is typically `[acceleration, steering]`. If action[0] is highly negative, the model is braking hard.

### C. The Concepts
- `true_concepts` (Shape: T × Num_Concepts) — What was *actually* happening in the environment, extracted directly from the ground-truth observation pipeline (`extract_all_concepts`).
- `pred_concepts` (Shape: T × Num_Concepts) — What the CBM *believed* was happening, extracted via `encode_and_predict_concepts`.

---

## 2. Causal Heuristics (The Tags)

The miner script defines heuristic functions that scan the time-series arrays inside `ScenarioData` and yield boolean tags. 

*Note: These definitions outline the design. The specific threshold values are defined dynamically inside `mine_scenarios.py`.*

### Tag: "Correct Red Light Stop"
A scenario where a red light exists, the model accurately predicts it, applies brakes, and survives.
- **Concept:** `true_concepts['traffic_light_red']` > 0.8  AND  `pred_concepts['traffic_light_red']` > 0.8
- **Action:** `ego_actions[:, 0]` (acceleration) is negative during the braking phase.
- **Outcome:** `ego_speed` approaches 0, and `dones` remains 0 (no collision).

### Tag: "Rear-end Collision (Ignored Concept)"
A scenario where the model crashes into a lead vehicle because it failed to apply the brakes, despite the distance closing.
- **Concept:** `true_concepts['dist_nearest_object']` approaches 0.
- **Action:** `ego_actions[:, 0]` stays positive or mostly positive (failed to brake).
- **Outcome:** `dones` hits 1.0 prematurely.

### Tag: "Off-road Curve Failure"
A failure to navigate a sharp turn.
- **Concept:** `true_concepts['path_curvature_max']` is high.
- **Action:** `ego_actions[:, 1]` (steering) was insufficient or delayed.
- **Outcome:** `dones` hits 1.0 prematurely.

### Tag: "Flawless Navigation"
A baseline scenario for comparative visualization where everything goes perfectly.
- **Concept:** N/A (Standard driving situation)
- **Action:** Smooth acceleration/steering.
- **Outcome:** `dones` is False at step 80, and total route progress > 0.95.

---

## 3. The Output (`XAI_curated_index.json`)

When the miner runs, it drops a JSON file that categorizes the dataset indices based on these tags. The Streamlit UI specifically looks for this file to power its "Filter by Tag" dropdown, allowing a researcher to instantly pull up 3 perfectly categorized "Rear-end Collision" videos for XAI analysis.
