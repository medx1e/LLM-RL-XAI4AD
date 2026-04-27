# Post-Hoc XAI Framework — Overview

## What Is This?

This framework answers a simple question: **why did the self-driving car make that decision?**

We work with pretrained autonomous driving models (V-MAX) that take in a driving scene — nearby vehicles, road layout, traffic lights, GPS path — and output a steering/acceleration action. These models are neural networks, which means they are black boxes. Our framework opens that black box.

## What Does It Do?

Given any driving scenario and any pretrained model, the framework can:

1. **Rank what mattered most** — Was the model paying attention to the road layout? A nearby car? A traffic light? The GPS route? We compute an importance score for each input category.

2. **Identify individual entities** — Not just "other vehicles mattered" but specifically "vehicle 3 was the most important agent (26%), vehicle 1 barely mattered (0.7%)". Same for individual traffic lights, road points, and GPS waypoints. Invalid entities (e.g. no traffic light in the scene) correctly show zero importance.

3. **Compare explanation methods** — Different explainability techniques can give different perspectives. We implemented 7 methods and can run them side by side to see where they agree and disagree.

4. **Measure explanation quality** — Are the explanations faithful to what the model actually does? Are they sparse (focused) or spread thin? Are they consistent across similar scenarios? We have metrics for all of this.

5. **Visualize everything** — Bar charts for per-entity importance, method comparisons, deletion/insertion curves. All generated with one function call.

## The 7 Explainability Methods

| Method | How It Works (Plain English) | Speed |
|--------|------------------------------|-------|
| **Vanilla Gradient** | Asks "if I slightly change this input, how much does the output change?" | Fast |
| **Integrated Gradients** | Same idea, but accumulated along a path from a blank input to the real input. More theoretically grounded. | Medium |
| **SmoothGrad** | Averages gradients over many slightly noisy versions of the input. Produces cleaner results. | Medium |
| **Gradient x Input** | Multiplies the gradient by the input value itself. Highlights features that are both sensitive and large. | Fast |
| **Perturbation** | Actually zeroes out features and measures how much the output drops. Direct but slower. | Slow |
| **Feature Ablation** | Removes entire categories (e.g. all traffic light data) and measures impact. Good for high-level understanding. | Fast |
| **SARFA** | Designed specifically for RL agents. Measures both whether a feature affects the chosen action AND whether it specifically affects that action vs alternatives. | Slow |

## What Models Are Supported?

All 5 V-MAX encoder architectures:

| Architecture | Description | Status |
|-------------|-------------|--------|
| **Perceiver** | Cross-attention from learned queries to input tokens | Fully supported (gradient + attention) |
| **MTR** | Local k-nearest-neighbor attention | Supported (gradient-based) |
| **Wayformer** | Late fusion with per-modality processing | Supported (gradient-based) |
| **MGAIL** | Hierarchical cross-attention | Supported (gradient-based) |
| **None/MLP** | Simple feedforward baseline (no encoder) | Supported (gradient-based) |

## What the Observation Contains

Each driving scenario is represented as a flat vector of 1,655 features, broken down into:

| Category | Features | What It Captures |
|----------|----------|-----------------|
| **SDC Trajectory** | 40 | The ego vehicle's own recent trajectory (5 timesteps) |
| **Other Agents** | 320 | 8 nearest vehicles: position, velocity, heading, size over 5 timesteps |
| **Roadgraph** | 1,000 | 200 road points: lane boundaries, directions |
| **Traffic Lights** | 275 | 5 nearest traffic lights: position and state over 5 timesteps |
| **GPS Path** | 20 | 10 target waypoints the vehicle should follow |

The framework breaks this down to individual entities — so you see importance per vehicle, per traffic light, per waypoint.

## Evaluation Metrics

- **Faithfulness** — If we remove the features the method says are important, does the model output actually change? (Deletion/insertion curves)
- **Sparsity** — Is the explanation focused on a few key features, or spread thinly across everything? (Gini coefficient, entropy)
- **Consistency** — Does the method give similar explanations for similar driving scenarios? (Pairwise correlation)

## Key Findings So Far

### Cross-Architecture Findings (from initial testing)

- **Roadgraph dominates** for most architectures (60-70% importance) — the road layout is the primary input the models rely on
- **MGAIL is the exception** — it uniquely relies heavily on GPS path (41%) alongside roadgraph (38%)
- **None/MLP is the most uniform** — without an attention mechanism, importance is spread more evenly across all categories
- **Per-agent importance varies dramatically** — in a given scenario, typically 1-2 vehicles matter significantly while the rest are near zero
- **Invalid entities get zero importance** — traffic lights not present in the scene correctly receive 0% attribution
- **Nearest GPS waypoint dominates** — waypoint_0 consistently gets 80%+ of GPS path importance

### Temporal XAI Findings (from Event Mining + XAI experiment)

We mined 153 driving events across 5 scenarios, then ran vanilla_gradient and integrated_gradients at timesteps spanning the 3 most critical events. Key findings:

- **Vanilla gradient underestimates agent importance by up to 50–144x** compared to integrated gradients — in one evasive steering event, VG attributed <2% to other agents while IG attributed 22–35%. This is because VG misses feature importance in saturated activation regions. Strong argument for using IG in driving model explainability.
- **Temporal attribution shifts during hazards are real and interpretable** — during a critical hazard onset (t=9→t=35), the model shows a clear detect → attend → commit → execute arc: other_agents spikes from 5% to 23% at hazard onset, then declines as roadgraph climbs to 85% as the model commits to an avoidance path.
- **The model detects which specific agent is dangerous** — the causal agent's importance jumps from 0.004% to 22.4% in 4 timesteps at hazard onset, then declines as the threat is managed.
- **GPS path importance increases during evasive maneuvers** — the model checks route deviation when steering aggressively (GPS triples from 5.5% to 14.9% during an evasive steer).
- **Ego state (SDC trajectory) is surprisingly unimportant** (2–10%) — the model is externally-focused.

Full analysis: see `experiments/results_analysis.md`.

## Event Mining Module

The `event_mining/` module detects **frame-level critical driving events** from model rollouts — finding the exact timestep a hazard begins, which agent caused it, and extracting analysis windows for targeted XAI.

### What It Does

1. **Extracts per-step data** from V-Max episode rollouts (ego trajectory, agent positions, actions, metrics)
2. **Detects 7 event types**: hazard onset, collision imminent, hard brake, evasive steering, near miss, collision, off-road
3. **Classifies severity** (LOW → CRITICAL) and identifies the causal agent
4. **Provides analysis windows** for temporal XAI investigation
5. **Renders BEV videos** using Waymax's native visualization with event overlays

### Usage

```python
from event_mining import EventMiner, EventCatalog
from event_mining.integration.vmax_adapter import VMaxAdapter
from event_mining.visualization.bev_video import render_model_video

# Mine events
adapter = VMaxAdapter(store_raw_obs=True)
adapter.prepare(model)
sd = adapter.extract_scenario_data(model, scenario, "s000", rng_seed=0)
events = EventMiner().mine_scenario(sd)

# Render BEV video with event overlays
render_model_video(model, scenario, events, "output.gif", fps=10)
```

Full documentation: see `Post-hoc-xai-framework-docs/event_mining.md`.

---

## Experiment Pipelines

### Pipeline 1: Scenario Scanner + XAI Analysis

Manually scripting each analysis is tedious. The experiment pipeline automates the full workflow:

### What It Does

1. **Scans** N scenarios — runs the model on each, records metrics (collision, offroad, route completion), tags scenarios (crowded, has traffic lights, failure)
2. **Selects** the most interesting scenarios — failures first, then ranked by number of agents and traffic lights
3. **Analyzes** selected scenarios at key timesteps — runs all 7 XAI methods, computes sparsity and faithfulness metrics, measures cross-method agreement
4. **Generates temporal plots** — tracks how feature importance shifts across the episode (e.g. roadgraph importance dropping while agent importance spikes before a collision)
5. **Summarizes** — aggregates results (mean/std per method, per category) and prints tables
6. **Compares** across models — generates grouped bar charts and heatmaps showing how different architectures explain their decisions

### Key Properties

- **Resume-friendly** — if the process crashes mid-analysis, restart picks up where it left off (completed results are cached as JSON)
- **One model per run** — respects the Waymax one-model-per-process constraint. Run the pipeline N times for N models, then compare without loading any model
- **Presets** — `quick` (10 scenarios, 3 fast methods, ~15 min) for testing; `standard` (50 scenarios, all 7 methods, faithfulness metrics) for real experiments

### Usage

```python
from posthoc_xai.experiments import ExperimentConfig, run_experiment, compare_experiments

# Quick test on one model
config = ExperimentConfig.quick("runs_rlc/womd_sac_road_perceiver_minimal_42")
run_experiment(config)

# Standard run
config = ExperimentConfig.standard("runs_rlc/womd_sac_road_mtr_minimal_42")
run_experiment(config)

# Compare all models (reads saved JSON, no model loading)
compare_experiments("results/default")
```

Or via CLI:

```bash
python -m posthoc_xai.experiments.runner run --model runs_rlc/womd_sac_road_perceiver_minimal_42 --preset quick
python -m posthoc_xai.experiments.runner compare --output results/default
```

### Output Structure

```
results/default/
├── catalog.json                           # All scanned scenarios with tags
├── womd_sac_road_perceiver_minimal_42/
│   ├── analysis/
│   │   ├── s000_t000.json                 # Per-(scenario, timestep) results
│   │   ├── s000_t040.json
│   │   └── ...
│   ├── plots/                             # Per-point visualizations
│   └── summary.json                       # Aggregated metrics
├── womd_sac_road_mtr_minimal_42/
│   └── ...
└── comparison/
    ├── summary.json                       # Cross-model comparison
    └── plots/                             # Grouped bar charts, heatmaps
```

---

### Pipeline 2: Event Mining + Temporal XAI

A targeted experiment that mines critical events, then runs XAI methods across event windows:

1. **Mine events** from N scenarios using all 6 detectors
2. **Select the most interesting events** (highest severity, diverse types)
3. **Run XAI methods** at timesteps spanning each event window (onset, peak, offset + surrounding context)
4. **Generate temporal plots** showing how attributions shift during critical moments

```bash
# Run the experiment (in conda vmax env)
PYTHONPATH=/path/to/V-Max:$PYTHONPATH python experiments/event_xai_experiment.py
```

Output: `experiments/event_xai_results/` — JSON data + category importance plots, agent importance plots, stacked composition charts.

Full results analysis: see `experiments/results_analysis.md`.

---

## Limitations

- Each model must be loaded in a separate Python process (a Waymax simulator constraint)
- `sac_seed0/42/69` models cannot be loaded (they use a `speed_limit` feature not available in Waymax)
- Architecture-specific attention extraction is only implemented for Perceiver; other architectures use gradient-based methods only
- Event mining thresholds need recalibration for normalized observation space (V-Max observations are ~[-1,1], not real-world meters)
- BEV video rendering requires Waymax native visualization (model rollout mode only; logged trajectory mode is incompatible with the current data format)
