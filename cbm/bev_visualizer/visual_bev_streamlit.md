# BEV Visualizer & Scenario Curation Architecture

This document describes the decoupled, high-performance architecture built for visualizing closed-loop simulation rollouts using V-Max and Waymax. 

Because executing deep RL models + simulation steps live inside a Streamlit web app is prone to extreme memory leaks (VRAM exhaustion) and slow user experiences, we have implemented a **Factory-Gallery Paradigm (Headless Curation + Cached UI)**.

---

## 🏗 The Architecture

The system is split into two completely decoupled phases:

### Phase 1: The Headless Factory (`curate_scenarios_skeleton.py`)
A fast, backend CLI script that allocates the GPU, builds the JAX policy, and uses a highly optimized `jax.lax.scan` loop to run a full 80-step rollout statically on the GPU in <1 second. 
The resulting `ScenarioData` (a pure Numpy dataclass) is serialized to disk via `pickle`.

### Phase 2: The Gallery UI (`streamlit_app.py`)
A lightweight frontend dashboard. It **does not import JAX, Waymax, or any model weights**. It purely scans the `curated_scenarios/` directory for pre-computed `.pkl` files, unpickles them instantly (requiring only CPU and a few MBs of RAM), and passes the states to the Matplotlib backend (`bev_renderer.py`).

---

## 📁 Folder Structure

```
cbm/
├── bev_visualizer/
│   ├── __init__.py           ← Public API exports (run_rollout, render_episode)
│   ├── model_registry.py     ← THE ONLY FILE YOU EDIT to register new models
│   ├── rollout_engine.py     ← Core JAX/Waymax loop. Returns numpy `ScenarioData`
│   ├── bev_renderer.py       ← Matplotlib BEV renderer + overlay protocol
│   └── streamlit_app.py      ← JAX-free Streamlit UI reading from curated_scenarios/
│
├── curate_scenarios_skeleton.py  ← Headless script to cache scenarios to disk
└── curated_scenarios/            ← Cached data output directory
    ├── SAC_Complete_(WOMD_seed_42)/
    │   ├── scenario_0000_cache.pkl
    │   └── scenario_0001_cache.pkl
    └── SAC_Minimal_(WOMD_seed_42)/
```

---

## 🚀 How to Use

### 1. Generating Golden Scenarios (Curation)
If you want to observe how a model behaves on Scenario X, you *pre-compute* it.
Open `curate_scenarios_skeleton.py` and modify `TARGET_SCENARIOS` and `MODELS_TO_CACHE`. 
Run it in the terminal:
```bash
conda activate vmax
python curate_scenarios_skeleton.py
```
This script handles $O(1)$ memory dataset iteration, runs the episode, and saves the `.pkl` artifact.

### 2. Viewing Results (Streamlit)
To visualize the generated caches, spin up the decoupled UI:
```bash
streamlit run bev_visualizer/streamlit_app.py
```
You can instantly select models and scenarios from the dropdowns with zero loading time.

---

## 🔧 Extending the System

### 1. Adding a New Model
Open `bev_visualizer/model_registry.py` and append to `MODEL_REGISTRY`:
```python
"My New CBM Model": {
    "type": "cbm",
    "checkpoint": _p("cbm_v2_frozen_womd_150gb/checkpoints/model_final.pkl"),
    "pretrained_dir": _p("runs_rlc/womd_sac_road_perceiver_complete_42"),
    "num_concepts": 15,
    "concept_phases": [1, 2, 3],
    "description": "15-concept CBM, frozen encoder.",
},
```
You can then add `"My New CBM Model"` to the `MODELS_TO_CACHE` list in the curation script.

### 2. The Overlay Protocol (For CBM Heatmaps/Neurons)
The long-term goal of this platform is to visualize **Why** the model made a decision by rendering Concept Neurons or Attention visualizers on top of the BEV.

The `render_episode` function accepts an `overlay_fn`:
```python
def draw_concept_influences(ax, step: int):
    """ax is the current Matplotlib Axes. step is the frame index."""
    ax.plot(x, y, 'ro', markersize=8)

frames = render_episode(data.frame_states, overlay_fn=draw_concept_influences)
```
Because the Streamlit UI only relies on `render_episode`, you can pass your custom `overlay_fn` directly inside Streamlit once your concepts are unpacked!

---

## 📦 ScenarioData Reference

The core hand-off object between the Engine and the UI is `ScenarioData`. It is heavily optimized (pure numpy).

| Field          | Shape        | Description                                        |
|----------------|--------------|----------------------------------------------------|
| `ego_xy`       | `(T, 2)`     | Model-driven SDC position (x, y) per step          |
| `ego_yaw`      | `(T,)`       | SDC heading in radians per step                    |
| `agents_xy`    | `(T, A, 2)`  | All agent positions per step                       |
| `agents_valid` | `(T, A)`     | Boolean validity mask per agent per step           |
| `agents_types` | `(A,)`       | Static object type IDs                             |
| `frame_states` | `list[T]`    | Extracted CPU-side Waymax SimulatorState per step  |
| `rewards`      | `(T,)`       | True per-step scalar reward (`linear` config)      |
| `dones`        | `(T,)`       | Episode termination flag (e.g., collisions)        |
| `model_key`    | `str`        | Which model produced this rollout                  |
| `scenario_idx` | `int`        | Which scenario dataset index                       |

---

## 🎯 Design Principles Summary

1. **JAX Belongs in the Backend:** The UI layer (`streamlit_app`) must never initialize JAX, cuSolver, or XLA memory. All compute is front-loaded in the curation script.
2. **Instant State Retrieval:** Re-running a live rollout just to "rewind the video" is banned. Scenarios are `ScenarioData` `.pkl` objects.
3. **Robust Evaluation Logging:** The UI detects `Early Stop` precisely so researchers know if the baseline failed the intersection natively (before relying on XAI overlays).
