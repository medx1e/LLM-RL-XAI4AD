# Virtual-Max (V-Max) Concept Bottleneck Model Project
**Global Context & Architecture Reference**

This document serves as the high-level map of the `cbm` project directory. It explains how the core simulation framework (Waymax) connects to the neural network training loops, concept extraction, and UI visualization.

---

## 🏗 High-Level Architecture

The system is built on top of **V-Max** (a Waymax Reinforcement Learning framework). The core objective of this project is to inject **Explainable AI (XAI)** into Autonomous Vehicle driving policies by training a Concept Bottleneck Model (CBM). 

Instead of a black-box Neural Network taking in Lidar and outputting Steering, the CBM predicts human-interpretable concepts (e.g., "Is the light red?", "Is the car ahead braking?") and then makes driving decisions based *strictly* on those concepts.

---

## 📁 Directory Structure Overview

### 1. The Core Simulation & Base Agents
- **`V-Max/`**: The underlying Google framework for executing JAX-based RL. Contains identical wrappers, environments, and basic SAC (Soft Actor-Critic) agent implementations.

### 2. The Concept Bottleneck Engine
- **`cbm_v1/`**: The entire neural network training architecture for our custom models.
  - `networks.py` & `cbm_sac_factory.py`: Defines the Neural Network structure (Encoder -> Bottleneck -> Actor/Critic).
  - `cbm_trainer.py`: The massive JAX PPO/SAC training loop that calculates the $L_{concept}$ and $L_{task}$ loss functions.
  - `eval_cbm.py`: Fast GPU-based evaluation scripts.
  - `config.py`: The `CBMConfig` schema.

### 3. The Heuristic Concept Registry
- **`concepts/`**: The deterministic "Ground Truth" extraction pipeline. It contains functions that parse the raw `SimulatorState` (Waymax 3D bounding boxes, traffic lights) and mathematically calculate the 15 concepts over 3 phases (e.g. `at_intersection`, `path_curvature_max`).
  - `registry.py`: The master list of all 15 concepts and their normalization limits.
  - `adapters.py`: Converts complex Waymax dicts into a flat array structure.

### 4. The Visualization Platform
- **`bev_visualizer/`**: The bridge between JAX/Waymax and the user screen. 
  - `rollout_engine.py`: Runs a fast `jax.lax.scan` rollout directly on the GPU, returning a pure Numpy `ScenarioData` object containing Agent locations, Ego Actions, and CBM Concept arrays.
  - `bev_renderer.py`: A Matplotlib engine to draw the Bird's Eye View map. Supports an `overlay_fn` for injecting XAI Neuron heatmaps.
  - `streamlit_app.py`: A decoupled, fast web dashboard that reads pre-cached rollouts and displays the BEV animation and reward curves.
  - `model_registry.py`: The database of trained model paths.

### 5. Curation & Causal Mining
- **`curate_scenarios_skeleton.py`**: A caching script that iterates through a subset of datasets and silently dumps Numpy `.pkl` files to disk to prevent Streamlit from exploding system RAM.
- **`mine_scenarios.py`**: The XAI Causal Miner. It dynamically runs the CBM over datasets, tracks the Causal Chain (`Concept -> Action -> Outcome`), and mathematically tags scenarios (e.g., "Rear-end Collision because it didn't brake"). Saves outputs to `XAI_curated_index.json`.

---

## 🔑 Key Model Paradigms

1. **SAC Baseline**: Standard Soft Actor-Critic agent trained purely on Reward (Task loss).
2. **CBM Joint**: Encoder, Concepts, and Policy layers are all trained together (Concept loss + Task loss). High reward but often ignores concepts (concept collapse).
3. **CBM Frozen ("Minimal" / "Complete")**: The backbone Encoder is locked. Only the Concept Head and Actor/Critic are trained. This forces the model to heavily rely on the injected logic, representing our state-of-the-art interpretable model.

---

## 🛠 Active Workflows

If you are modifying or analyzing this codebase:
- **Adding a Concept:** Edit `concepts/registry.py` to add the mathematical extractor. The `CBMConfig` will auto-detect it.
- **Registering a Trained Model:** Add its directory to `bev_visualizer/model_registry.py`.
- **Finding Edge-Cases:** Run `python mine_scenarios.py` and review the output JSON file.
- **Rendering the UI:** Run `streamlit run bev_visualizer/streamlit_app.py`.
