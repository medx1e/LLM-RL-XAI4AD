# Post-Hoc XAI Framework — Technical Guide

## Project Structure

```
posthoc_xai/                           # Core XAI framework (JAX-native)
├── __init__.py                        # Top-level API: load_model(), explain()
├── models/
│   ├── __init__.py
│   ├── base.py                        # ExplainableModel ABC, ModelOutput
│   ├── loader.py                      # load_vmax_model() — handles all 5 compatibility fixes
│   ├── _obs_structure.py              # Shared: per-category + per-entity obs structure computation
│   ├── perceiver_wrapper.py           # PerceiverWrapper (gradient + attention extraction)
│   └── generic_wrapper.py             # GenericWrapper (all other encoders, gradient-based)
├── methods/
│   ├── __init__.py                    # METHOD_REGISTRY: name → class mapping
│   ├── base.py                        # Attribution dataclass, AttributionMethod ABC
│   ├── vanilla_gradient.py
│   ├── integrated_gradients.py
│   ├── smooth_grad.py
│   ├── gradient_x_input.py
│   ├── perturbation.py
│   ├── feature_ablation.py
│   └── sarfa.py
├── metrics/
│   ├── __init__.py
│   ├── faithfulness.py                # Deletion/insertion curves, AUC, correlation
│   ├── sparsity.py                    # Gini, top-k concentration, entropy
│   └── consistency.py                 # Pairwise attribution/category correlation
├── visualization/
│   ├── __init__.py
│   └── heatmaps.py                    # All plot functions (bar, temporal, curves)
├── utils/
│   └── __init__.py
└── experiments/
    ├── __init__.py                    # Exports: ExperimentConfig, run_experiment, etc.
    ├── config.py                      # ExperimentConfig dataclass + quick/standard presets
    ├── scanner.py                     # ScenarioInfo, scan_scenarios(), select_scenarios()
    ├── analyzer.py                    # AnalysisResult, analyze_timestep(), analyze_scenarios()
    ├── reporter.py                    # summarize_model(), compare_models(), plots
    └── runner.py                      # run_experiment(), compare_experiments(), CLI

event_mining/                          # Event detection + BEV visualization module
├── __init__.py                        # Top-level: EventMiner, EventCatalog, mine_events
├── __main__.py                        # python -m event_mining support
├── events/
│   ├── __init__.py                    # ALL_DETECTORS list, re-exports
│   ├── base.py                        # Event, EventType, Severity, ScenarioData, EventDetector ABC
│   ├── safety.py                      # HazardOnsetDetector, NearMissDetector
│   ├── action.py                      # HardBrakeDetector, EvasiveSteeringDetector
│   └── outcome.py                     # CollisionDetector, OffRoadDetector
├── catalog.py                         # EventCatalog (queryable, JSON-serializable)
├── miner.py                           # EventMiner orchestrator + mine_events()
├── metrics.py                         # TTC, distance, criticality (numpy)
├── integration/
│   ├── __init__.py                    # Lazy imports (deferred JAX dependency)
│   ├── vmax_adapter.py                # VMaxAdapter: extract ScenarioData from V-Max rollouts
│   └── xai_bridge.py                  # XAIBridge: feed events into posthoc_xai pipeline
├── visualization/
│   ├── __init__.py
│   └── bev_video.py                   # Waymax-based BEV renderer + event overlays
└── cli.py                             # CLI: mine, summary, export, render commands

experiments/                           # Standalone experiment scripts
├── event_xai_experiment.py            # Event mining → temporal XAI analysis pipeline
└── event_xai_results/                 # Output: JSON data + temporal plots
```

---

## Quick Start

### Setup

```bash
eval "$(~/anaconda3/bin/conda shell.bash hook)" && conda activate vmax
```

Make sure `V-Max/` is in the project root and `data/training.tfrecord` exists.

### Load a Model

```python
import posthoc_xai as xai

model = xai.load_model(
    "runs_rlc/womd_sac_road_perceiver_minimal_42",
    data_path="data/training.tfrecord",
)
```

`load_model()` auto-selects the wrapper:
- Perceiver/LQ models → `PerceiverWrapper` (supports attention extraction)
- Everything else → `GenericWrapper`

### Get an Observation

```python
scenario = next(model._loaded.data_gen)
state = model._loaded.env.reset(scenario)
obs = state.observation.reshape(-1)  # shape: (1655,)
```

### Run a Single Method

```python
vg = xai.VanillaGradient(model)
attr = vg(obs)

print(attr.category_importance)
# {'sdc_trajectory': 0.029, 'other_agents': 0.013, 'roadgraph': 0.708, ...}

print(attr.entity_importance["other_agents"])
# {'agent_0': 0.0001, 'agent_1': 0.0125, 'agent_2': 0.0001, ...}
```

### Run Multiple Methods at Once

```python
results = xai.explain(model, obs)
# Returns: {'vanilla_gradient': Attribution, 'integrated_gradients': Attribution, 'perturbation': Attribution}

# Or specify which methods:
results = xai.explain(model, obs, methods=["vanilla_gradient", "smooth_grad", "sarfa"])
```

### Check Entity Validity

```python
validity = model.get_entity_validity(obs)
# {'sdc_trajectory': {'sdc_0': True},
#  'other_agents': {'agent_0': True, 'agent_1': True, ...},
#  'traffic_lights': {'light_0': False, 'light_1': False, ...},
#  ...}
```

### Visualize

```python
from posthoc_xai.visualization.heatmaps import (
    plot_category_importance,
    plot_entity_importance,
    plot_agent_comparison,
    plot_method_comparison,
    plot_deletion_insertion_curves,
)

# Per-entity importance (agents, lights, waypoints)
fig = plot_entity_importance(attr, validity)
fig.savefig("entity_importance.png", dpi=150, bbox_inches="tight")

# Category-level bar chart
fig = plot_category_importance(attr)

# Compare methods side by side (category level)
fig = plot_method_comparison([attr_vg, attr_ig, attr_sg])

# Compare methods per agent
fig = plot_agent_comparison([attr_vg, attr_ig, attr_sg], validity)
```

---

## Core Classes

### `ExplainableModel` (ABC) — `posthoc_xai/models/base.py`

Abstract interface every wrapper implements.

| Method / Property | Signature | Description |
|---|---|---|
| `forward(obs)` | `jnp.ndarray → ModelOutput` | Full forward pass. Accepts `(obs_dim,)` or `(batch, obs_dim)`. |
| `get_action_value(obs, action_idx=None)` | `jnp.ndarray → scalar` | Scalar output for `jax.grad`. Unbatched input only. |
| `get_embedding(obs)` | `jnp.ndarray → jnp.ndarray` | Encoder output vector. |
| `get_attention(obs)` | `jnp.ndarray → dict or None` | Attention weights (Perceiver only). |
| `observation_structure` | `dict[str, (int, int)]` | Category name → `(start_idx, end_idx)` in flat obs. |
| `observation_structure_detailed` | `dict[str, dict]` | Per-entity ranges within each category. |
| `get_entity_validity(obs)` | `jnp.ndarray → dict[str, dict[str, bool]]` | Which entities exist in the scene. |
| `has_attention` | `bool` | Whether attention extraction is available. |
| `name` | `str` | Human-readable model identifier. |

### `ModelOutput` — `posthoc_xai/models/base.py`

```python
ModelOutput(
    action_mean,    # (action_dim,) — deterministic action
    action_std,     # (action_dim,) — action distribution std
    value=None,     # V(s) if available
    embedding=None, # encoder output
    attention=None,  # dict of attention weight arrays
)
```

### `Attribution` (dataclass) — `posthoc_xai/methods/base.py`

Returned by every XAI method.

| Field | Type | Description |
|---|---|---|
| `raw` | `jnp.ndarray` | Raw attribution, same shape as input obs |
| `normalized` | `jnp.ndarray` | `abs(raw)` normalized to sum to 1 |
| `category_importance` | `dict[str, float]` | Per-category aggregated importance |
| `entity_importance` | `dict[str, dict[str, float]]` | Per-entity importance within each category |
| `method_name` | `str` | Method identifier |
| `target_action` | `int or None` | Which action dim was targeted |
| `computation_time_ms` | `float` | Wall-clock time |
| `extras` | `dict or None` | Method-specific data |

### `AttributionMethod` (ABC) — `posthoc_xai/methods/base.py`

Base class for all methods.

| Method | Description |
|---|---|
| `__init__(model, **kwargs)` | Stores model reference and config |
| `compute_raw_attribution(obs, target_action)` | **Abstract** — returns raw gradient/perturbation array |
| `name` (property) | **Abstract** — method identifier string |
| `normalize(raw)` | `abs(raw) / sum(abs(raw))` |
| `aggregate_by_category(normalized)` | Sum per observation category |
| `aggregate_by_entity(normalized)` | Sum per entity within each category |
| `__call__(obs, target_action)` | Full pipeline: raw → normalize → aggregate → `Attribution` |

---

## XAI Methods Reference

### VanillaGradient

```python
vg = xai.VanillaGradient(model)
attr = vg(obs)
```

Computes `df(x)/dx` via `jax.grad`. Fastest method.

### IntegratedGradients

```python
ig = xai.IntegratedGradients(model, n_steps=50, baseline="zero")
attr = ig(obs)
```

| Param | Default | Description |
|---|---|---|
| `n_steps` | 50 | Number of interpolation steps along the path |
| `baseline` | `"zero"` | Starting point: `"zero"`, `"noise"`, or a `jnp.ndarray` |

Path integral from baseline to input. Satisfies completeness axiom (attributions sum to `f(x) - f(baseline)`).

### SmoothGrad

```python
sg = xai.SmoothGrad(model, n_samples=50, noise_std=0.1, seed=42)
attr = sg(obs)
```

| Param | Default | Description |
|---|---|---|
| `n_samples` | 50 | Number of noisy copies |
| `noise_std` | 0.1 | Gaussian noise standard deviation |
| `seed` | 42 | RNG seed for reproducibility |

Averages gradients over noisy copies. Produces visually smoother explanations.

### GradientXInput

```python
gxi = xai.GradientXInput(model)
attr = gxi(obs)
```

No extra params. Computes `x * grad(f(x))`. Highlights features that are both sensitive and have large values.

### PerturbationAttribution

```python
pa = xai.PerturbationAttribution(model, perturbation_type="zero", per_category=True)
attr = pa(obs)
```

| Param | Default | Description |
|---|---|---|
| `perturbation_type` | `"zero"` | How to mask: `"zero"`, `"mean"`, or `"noise"` |
| `per_category` | `True` | If `True`, perturb whole categories (fast). If `False`, per-feature (slow). |

Occlusion-based: masks features and measures output change. Model-agnostic (no gradients needed).

### FeatureAblation

```python
fa = xai.FeatureAblation(model, replacement="zero")
attr = fa(obs)
```

| Param | Default | Description |
|---|---|---|
| `replacement` | `"zero"` | `"zero"` or `"mean"` |

Removes entire categories one at a time. Good for high-level "what type of information matters" analysis. Also has a convenience method:

```python
cat_importance = fa.compute_category_importance(obs)
# {'sdc_trajectory': 0.03, 'other_agents': 0.25, ...}
```

### SARFA

```python
sarfa = xai.SARFA(model, perturbation_type="zero", per_category=True)
attr = sarfa(obs)
```

| Param | Default | Description |
|---|---|---|
| `perturbation_type` | `"zero"` | `"zero"` or `"mean"` |
| `per_category` | `True` | Category-level (fast) or per-feature |

RL-specific method. Computes `relevance * specificity` where:
- **Relevance** = does this feature affect the chosen action?
- **Specificity** = does it affect this action more than others?

---

## Metrics Reference

### Faithfulness — `posthoc_xai/metrics/faithfulness.py`

```python
from posthoc_xai.metrics.faithfulness import (
    deletion_curve,
    insertion_curve,
    area_under_deletion_curve,
    area_under_insertion_curve,
    attention_gradient_correlation,
)

# Deletion curve: remove most important features progressively
pcts, outputs = deletion_curve(model, obs, attribution, n_steps=20)
auc_del = area_under_deletion_curve(outputs)  # lower is better

# Insertion curve: add most important features progressively
pcts, outputs = insertion_curve(model, obs, attribution, n_steps=20)
auc_ins = area_under_insertion_curve(outputs)  # higher is better

# Compare two attribution vectors
corr = attention_gradient_correlation(attr_a, attr_b)
```

### Sparsity — `posthoc_xai/metrics/sparsity.py`

```python
from posthoc_xai.metrics.sparsity import gini_coefficient, top_k_concentration, entropy, compute_all

gini = gini_coefficient(attribution)          # 0=uniform, 1=sparse
top10 = top_k_concentration(attribution, k=10)  # fraction in top 10 features
ent = entropy(attribution)                     # 0=concentrated, 1=uniform

all_metrics = compute_all(attribution)
# {'gini': 0.85, 'top_10_concentration': 0.42, 'top_50_concentration': 0.78, 'entropy': 0.31}
```

### Consistency — `posthoc_xai/metrics/consistency.py`

```python
from posthoc_xai.metrics.consistency import attribution_consistency, category_consistency

# Run same method on multiple scenarios
attrs = [method(obs1), method(obs2), method(obs3)]

# Feature-level pairwise Pearson correlation (1.0 = perfectly consistent)
feat_corr = attribution_consistency(attrs)

# Category-level correlation (more robust)
cat_corr = category_consistency(attrs)
```

---

## Visualization Reference

All functions are in `posthoc_xai.visualization.heatmaps`.

### `plot_category_importance(attribution, title=None, ax=None) → Figure`

Bar chart of per-category importance for a single method.

### `plot_entity_importance(attribution, validity=None, categories=None, title=None) → Figure`

Horizontal bar chart showing per-entity importance. Invalid entities are grayed out. Default categories: `["other_agents", "traffic_lights", "gps_path"]`.

### `plot_agent_comparison(attributions, validity=None, title=...) → Figure`

Grouped bar chart comparing per-agent importance across multiple methods. Invalid agents get a light background.

### `plot_method_comparison(attributions, title=..., ax=None) → Figure`

Grouped bar chart comparing category-level importance across methods.

### `plot_deletion_insertion_curves(deletion_data, insertion_data) → Figure`

Side-by-side line plots for deletion and insertion curves. Each data entry is `(method_name, percentages_array, outputs_array)`.

### Temporal Plots

These functions visualize how feature importance changes across timesteps within an episode. All accept simple dict/list data structures (not `Attribution` objects).

#### `plot_temporal_category(timesteps, category_series, method_name, scenario_idx=None, events=None) → Figure`

Line plot with one line per category, showing importance over time. `events` is an optional `{timestep: label}` dict to mark events like collisions.

#### `plot_temporal_category_stacked(timesteps, category_series, method_name, scenario_idx=None, events=None) → Figure`

Stacked area chart of the same data — shows relative composition shifting over time.

#### `plot_temporal_entity(timesteps, entity_series, category_name, method_name, scenario_idx=None, validity_series=None, top_n=5) → Figure`

Line plot of the top-N entities (by max importance) over time. Useful for tracking when specific agents become important.

#### `plot_temporal_sparsity(timesteps, sparsity_series, metric_name, scenario_idx=None) → Figure`

Line plot with one line per method, showing how sparsity (e.g. Gini coefficient) evolves.

#### `plot_temporal_multi_method(timesteps, method_category_series, scenario_idx=None) → Figure`

Grid of subplots (one per category), with one line per method. Shows whether different XAI methods agree on temporal dynamics.

#### Data format

```python
timesteps = [0, 20, 40, 60, 79]

# For category plots
category_series = {
    "sdc_trajectory": [0.05, 0.04, 0.06, 0.08, 0.10],
    "other_agents":   [0.10, 0.15, 0.25, 0.35, 0.45],
    "roadgraph":      [0.70, 0.65, 0.50, 0.35, 0.25],
    ...
}

# For entity plots
entity_series = {
    "agent_0": [0.01, 0.02, 0.08, 0.20, 0.35],
    "agent_1": [0.05, 0.08, 0.10, 0.08, 0.05],
    ...
}

# For sparsity plots
sparsity_series = {
    "vanilla_gradient": [0.85, 0.82, 0.78, 0.75, 0.72],
    "feature_ablation": [0.60, 0.62, 0.58, 0.55, 0.50],
}
```

#### Standalone temporal plot generation

To generate temporal plots from already-saved analysis JSON files (no model loading needed):

```python
from posthoc_xai.experiments import generate_temporal_plots_from_saved

generate_temporal_plots_from_saved("results/default", "womd_sac_road_perceiver_minimal_42")
# Or for specific scenarios:
generate_temporal_plots_from_saved("results/default", "womd_sac_road_perceiver_minimal_42",
                                   scenario_indices=[3, 7, 12])
```

#### Auto-generated during pipeline

When `save_plots=True` (default), the experiment pipeline automatically generates temporal plots after completing all timesteps of each scenario. These are saved as:
```
plots/s003_temporal_vanilla_gradient_categories.png   # line plot per category
plots/s003_temporal_vanilla_gradient_stacked.png      # stacked area
plots/s003_temporal_vanilla_gradient_other_agents.png  # top-5 agents over time
plots/s003_temporal_feature_ablation_categories.png
plots/s003_temporal_sparsity_gini.png                 # Gini across methods
plots/s003_temporal_multi_method.png                  # grid comparison
```

---

## Model Loader Details — `posthoc_xai/models/loader.py`

`load_vmax_model()` handles 5 known compatibility issues automatically:

| Issue | Problem | Fix |
|---|---|---|
| 1. Pickle paths | Saved weights reference old module paths | `load_params()` + tensorboardX |
| 2. Encoder aliases | `perceiver` / `mgail` not in current registry | Remap to `lq` / `lqh` |
| 3. Obs type aliases | `road` / `lane` not recognized | Remap to `vec` |
| 4. Param keys | `perceiver_attention` in saved weights | Remap to `lq_attention` |
| 5. speed_limit | `sac_seed*` models use unavailable feature | Raises error with clear message |

### Available Models

```
runs_rlc/
├── womd_sac_road_perceiver_minimal_{42,69,99}   # Perceiver/LQ
├── womd_sac_road_mtr_minimal_{42,69,99}          # MTR
├── womd_sac_road_wayformer_minimal_{42,69,99}    # Wayformer
├── womd_sac_road_mgail_minimal_{42,69,99}        # MGAIL/LQH
├── womd_sac_road_none_minimal_{42,69,99}         # None/MLP
├── sac_seed{0,42,69}                              # BROKEN (speed_limit)
└── ... (other variants)
```

---

## Observation Structure

### Category Level

```python
model.observation_structure
# {
#     'sdc_trajectory':  (0, 40),      # 40 features
#     'other_agents':    (40, 360),     # 320 features (8 agents x 40)
#     'roadgraph':       (360, 1360),   # 1000 features (200 points x 5)
#     'traffic_lights':  (1360, 1635),  # 275 features (5 lights x 55)
#     'gps_path':        (1635, 1655),  # 20 features (10 waypoints x 2)
# }
```

### Entity Level

```python
model.observation_structure_detailed
# {
#     'other_agents': {
#         'num_entities': 8,
#         'features_per_entity': 40,
#         'entities': {
#             'agent_0': (40, 80),
#             'agent_1': (80, 120),
#             ...
#         }
#     },
#     ...
# }
```

Per entity: `features_per_entity` includes the validity mask columns. For `other_agents`, each agent has 5 timesteps x (7 features + 1 valid flag) = 40 values.

---

## Experiment Pipeline — `posthoc_xai/experiments/`

The experiment pipeline automates the full workflow: scan scenarios, run XAI methods, compute metrics, and generate comparison plots.

### Quick Start

```python
from posthoc_xai.experiments import ExperimentConfig, run_experiment, compare_experiments

# Quick test (~15 min including JIT)
config = ExperimentConfig.quick("runs_rlc/womd_sac_road_perceiver_minimal_42")
run_experiment(config)

# Standard experiment
config = ExperimentConfig.standard("runs_rlc/womd_sac_road_perceiver_minimal_42")
run_experiment(config)

# Compare across models (no model loading, reads saved JSON)
compare_experiments("results/default")
```

### CLI

```bash
# Full pipeline for one model
python -m posthoc_xai.experiments.runner run \
    --model runs_rlc/womd_sac_road_perceiver_minimal_42 \
    --data data/training.tfrecord \
    --n-scenarios 50 \
    --output results

# Quick test
python -m posthoc_xai.experiments.runner run \
    --model runs_rlc/womd_sac_road_perceiver_minimal_42 \
    --preset quick

# Cross-model comparison
python -m posthoc_xai.experiments.runner compare --output results/default
```

### `ExperimentConfig` — `experiments/config.py`

| Field | Default | Description |
|---|---|---|
| `model_dir` | (required) | Path to model directory |
| `data_path` | `"data/training.tfrecord"` | Path to tfrecord |
| `n_scenarios` | 50 | Number of scenarios to scan |
| `min_valid_agents` | 0 | Filter: min agents (0 = no filter) |
| `require_traffic_lights` | `False` | Filter: must have traffic lights |
| `include_failures` | `True` | Always include collision/offroad scenarios |
| `max_selected` | 15 | Max scenarios for analysis |
| `methods` | all 7 | List of method names from `METHOD_REGISTRY` |
| `timestep_strategy` | `"key_moments"` | `"key_moments"`, `"fixed_interval"`, or `"all"` |
| `timestep_interval` | 20 | Interval for `"fixed_interval"` strategy |
| `target_action` | `None` | `None`=all, `0`=steering, `1`=accel |
| `compute_faithfulness` | `True` | Compute deletion/insertion AUC |
| `faithfulness_steps` | 10 | Steps for deletion/insertion curves |
| `save_raw_arrays` | `False` | Save .npz with raw attribution arrays |
| `save_plots` | `True` | Generate per-analysis-point plots |
| `output_dir` | `"results"` | Base output directory |
| `experiment_name` | `"default"` | Experiment subdirectory name |

**Presets:**
- `ExperimentConfig.quick(model_dir)` — 10 scenarios, 5 selected, 3 fast methods (`vanilla_gradient`, `gradient_x_input`, `feature_ablation`), no faithfulness
- `ExperimentConfig.standard(model_dir)` — 50 scenarios, 15 selected, all 7 methods, faithfulness enabled

### `ScenarioInfo` — `experiments/scanner.py`

Produced by scanning. Contains:

| Field | Description |
|---|---|
| `index` | Position in the data generator |
| `num_valid_agents` | Count of valid agents in the scene |
| `num_traffic_lights` | Count of valid traffic lights |
| `total_steps` | Episode length |
| `collision`, `offroad`, `comfort_violation`, `ran_red_light` | Boolean episode outcomes |
| `route_completion` | 0.0–1.0 |
| `tags` | Auto-assigned: `"collision"`, `"offroad"`, `"has_lights"`, `"crowded"`, `"full_completion"`, `"normal"` |
| `key_timesteps` | Timesteps selected for XAI analysis |
| `saved_obs_path` | Path to `.npz` with saved observations at key timesteps |

**Timestep strategies:**
- `"key_moments"` — first (0), middle, last, and pre-failure step. Typically 3–4 timesteps per episode.
- `"fixed_interval"` — every N steps (e.g. 0, 20, 40, 60, 79). Always includes the final step.
- `"all"` — every timestep. Comprehensive but slow.

### `AnalysisResult` — `experiments/analyzer.py`

Produced per (scenario, timestep) pair. Contains:

| Field | Description |
|---|---|
| `scenario_idx`, `timestep`, `model_name` | Identifiers |
| `attributions` | `{method: {category_importance, entity_importance, computation_time_ms}}` |
| `validity` | `{category: {entity: bool}}` |
| `sparsity` | `{method: {gini, entropy, top_10_concentration, top_50_concentration}}` |
| `faithfulness` | `{method: {deletion_auc, insertion_auc}}` |
| `method_agreement` | `{method_a_vs_method_b: pearson_r}` |

Each result is saved as JSON immediately after computation. On restart, existing results are loaded from disk (resume-friendly).

### Scanning & Selection — `experiments/scanner.py`

```python
from posthoc_xai.experiments.scanner import scan_scenarios, select_scenarios

catalog = scan_scenarios(model, config)
# Prints: [1/50] scenario_000: 80 steps, collision=False, agents=5, tags=[crowded]

selected = select_scenarios(catalog, config)
# Prints: Selected 12/50 scenarios (2 failures, 10 others)
```

`select_scenarios` prioritizes:
1. Failure scenarios (collision/offroad) — always included
2. Remaining ranked by interestingness (more agents + traffic lights = higher)
3. Capped at `max_selected`

### Analysis — `experiments/analyzer.py`

```python
from posthoc_xai.experiments.analyzer import analyze_scenarios

results = analyze_scenarios(model, selected, config)
# Prints: [3/45] scenario_007 step_40: 7 methods, faithfulness=True (12.3s)
```

For each (scenario, timestep):
1. Load saved observation from `.npz`
2. Run each configured XAI method → `Attribution`
3. Compute sparsity metrics (Gini, entropy, top-k)
4. Optionally compute faithfulness (deletion/insertion AUC)
5. Compute pairwise method agreement (category-level Pearson r)
6. Save JSON result; optionally generate plots

### Reporting — `experiments/reporter.py`

```python
from posthoc_xai.experiments.reporter import (
    summarize_model, compare_models, generate_comparison_plots, print_summary,
)

# Single model summary
summary = summarize_model("results/default", "womd_sac_road_perceiver_minimal_42")
print_summary("results/default", "womd_sac_road_perceiver_minimal_42")

# Cross-model comparison
compare_models("results/default")
generate_comparison_plots("results/default")
print_summary("results/default")  # model_name=None prints comparison
```

**Generated plots** (saved to `comparison/plots/`):
- `category_<method>.png` — grouped bar chart of category importance per model
- `faithfulness_deletion_auc.png` / `faithfulness_insertion_auc.png` — faithfulness ranking
- `sparsity_gini.png` — Gini coefficient per method per model
- `agreement_<model>.png` — method agreement heatmap per model

### Output Directory Structure

```
results/default/
├── catalog.json                           # All scanned scenarios
├── observations/
│   ├── s000.npz                           # Saved observations per scenario
│   ├── s001.npz
│   └── ...
├── womd_sac_road_perceiver_minimal_42/
│   ├── analysis/
│   │   ├── s000_t000.json                 # Results per (scenario, timestep)
│   │   ├── s000_t040.json
│   │   └── ...
│   ├── plots/
│   │   ├── s000_t000_method_comparison.png
│   │   ├── s000_t000_vanilla_gradient_categories.png
│   │   ├── s000_t000_vanilla_gradient_entities.png
│   │   ├── s000_t000_faithfulness.png
│   │   └── ...
│   └── summary.json                       # Aggregated mean/std across all points
├── womd_sac_road_mtr_minimal_42/
│   └── ...
└── comparison/
    ├── summary.json                       # Cross-model comparison tables
    └── plots/
        ├── category_vanilla_gradient.png
        ├── faithfulness_deletion_auc.png
        ├── sparsity_gini.png
        └── agreement_<model>.png
```

### Resume Behavior

The pipeline is designed to survive crashes:

1. **Catalog**: if `catalog.json` exists, scanning is skipped entirely
2. **Analysis**: each `s<IDX>_t<STEP>.json` is checked before computing. Existing files are loaded from disk.
3. **Summary/comparison**: always recomputed from analysis JSONs (cheap)

To force a full re-run, delete the relevant directory.

---

## Event Mining Module

### Overview

The `event_mining/` module detects frame-level critical driving events from V-Max rollouts. It extracts per-step trajectory data via the `VMaxAdapter`, runs 6 event detectors, and produces an `EventCatalog` of timestamped events with severity and causal agent annotations.

Full documentation: `Post-hoc-xai-framework-docs/event_mining.md`

### Quick Start

```python
import posthoc_xai as xai
from event_mining import EventMiner, EventCatalog
from event_mining.integration.vmax_adapter import VMaxAdapter

model = xai.load_model("runs_rlc/womd_sac_road_perceiver_minimal_42",
                        data_path="data/training.tfrecord")

# Prepare adapter (JIT-compiles step functions once)
adapter = VMaxAdapter(store_raw_obs=True)
adapter.prepare(model)

# Extract data from a scenario
scenario = next(model._loaded.data_gen)
sd = adapter.extract_scenario_data(model, scenario, "s000", rng_seed=0)

# Mine events
miner = EventMiner()
events = miner.mine_scenario(sd)
print(f"Found {len(events)} events")

# Build catalog
catalog = EventCatalog(events)
catalog.save("events/catalog.json")
```

### VMaxAdapter — Data Extraction

The adapter runs a full episode rollout and extracts per-step data from **unflattened observations**:

```python
# At each timestep:
features, masks = unflatten_fn(obs)
sdc_feat, other_feat, rg_feat, tl_feat, gps_feat = features

# Key shapes (after squeeze):
# sdc_feat:   (n_timesteps, 7)     → [x, y, vx, vy, yaw, length, width]
# other_feat: (N_agents, n_timesteps, 7)
# rg_feat:    (N_rg_points, 4)
# tl_feat:    (N_tl, n_timesteps, 10)
# gps_feat:   (N_gps, 2)
```

**Important:** All coordinates are **ego-relative and normalized** (~[-1, 1]), not real-world meters. This affects detector thresholds.

The `prepare()` method JIT-compiles `env.reset` and the step function once, avoiding recompilation per scenario. Critical on the 6GB GTX 1660 Ti.

### Event Detectors

| Detector | Trigger | Threshold |
|----------|---------|-----------|
| HazardOnsetDetector | TTC < threshold for N consecutive steps | ttc < 3.0s, min_duration=3 |
| NearMissDetector | Distance < threshold without collision | distance < 3.0m |
| HardBrakeDetector | Acceleration < threshold sustained | accel < -3.0 m/s² |
| EvasiveSteeringDetector | |Steering| > threshold sustained | steering > 0.3 rad |
| CollisionDetector | Episode has collision | traces TTC backwards |
| OffRoadDetector | Step-level offroad flags | continuous windows |

**Note:** Thresholds need recalibration for normalized observation space.

### BEV Video Rendering

Uses Waymax's native `plot_simulator_state()` for the base frame (proper world coordinates, full road geometry, vehicle polygons) and overlays event markers, info text, and a timeline bar.

```python
from event_mining.visualization.bev_video import render_model_video, render_event_clip

# Full scenario video with event overlays
render_model_video(model, scenario, events, "output.gif", fps=10, rng_seed=0)

# Focused clip around one event (±padding steps)
render_event_clip(model, scenario, event, "clip.gif", fps=10, padding=5, rng_seed=0)
```

Supports MP4 (via mediapy or ffmpeg) and GIF (via PIL) output.

---

## Event Mining + Temporal XAI Experiment

### Script: `experiments/event_xai_experiment.py`

End-to-end pipeline that connects event mining with temporal XAI analysis:

1. **Load model** via `posthoc_xai.load_model()`
2. **Mine events** from N scenarios using VMaxAdapter + EventMiner
3. **Select top events** by priority (collision > offroad > evasive_steering > hard_brake > hazard_onset) and severity
4. **Run XAI methods** at timesteps spanning each event window (onset, peak, offset + padding, strided)
5. **Generate temporal plots**: category importance lines, agent importance lines, stacked area composition

### Configuration

```python
MODEL_DIR = "runs_rlc/womd_sac_road_perceiver_minimal_42"
DATA_PATH = "data/training.tfrecord"
N_SCENARIOS = 5
N_EVENTS_TO_ANALYZE = 3
TIMESTEP_STRIDE = 3
WINDOW_PADDING = 5
XAI_METHODS = ["vanilla_gradient", "integrated_gradients"]
```

### Running

```bash
eval "$(~/anaconda3/bin/conda shell.bash hook)" && conda activate vmax
PYTHONPATH=/path/to/V-Max:$PYTHONPATH python experiments/event_xai_experiment.py 2>/dev/null
```

### Output: `experiments/event_xai_results/`

| File Pattern | Description |
|-------------|-------------|
| `catalog.json` | Full event catalog (all mined events) |
| `event_XX_sYYY.json` | Raw attribution time-series data per event |
| `event_XX_categories.png` | Category importance over time (one subplot per method) |
| `event_XX_agents.png` | Top-5 agent importance over time (causal agent highlighted) |
| `event_XX_stacked.png` | Stacked area composition chart (one panel per method) |

### Key Results

See `experiments/results_analysis.md` for the full analysis. Summary:
- Roadgraph dominates (50-85%) across all events
- Integrated gradients reveals 20-50% agent importance that vanilla gradient misses (up to 144x underestimation)
- Temporal attribution shifts show a detect→attend→commit→execute arc during hazards
- Per-agent attribution confirms causal agent detection by the model

---

## Important Constraints

### One Model Per Process

Waymax's metric registry is global and does not support re-registration. Loading a second model in the same process will fail with `"Metric run_red_light has already been registered"`. To compare architectures, run each in a separate process:

```python
# In separate scripts or subprocesses:
model_a = xai.load_model("runs_rlc/womd_sac_road_perceiver_minimal_42", ...)
model_b = xai.load_model("runs_rlc/womd_sac_road_mtr_minimal_42", ...)  # different process!
```

### GPU Memory

The GTX 1660 Ti has 6GB VRAM. First JIT compilation takes ~10 minutes. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` (the loader does this automatically) to avoid OOM.

### Target Action

All methods accept `target_action: int | None`:
- `None` (default): explains the sum of all action dimensions
- `0`: explains steering specifically
- `1`: explains acceleration specifically

```python
attr_steer = vg(obs, target_action=0)   # steering explanation
attr_accel = vg(obs, target_action=1)   # acceleration explanation
```

---

## METHOD_REGISTRY

String-to-class mapping for use with `explain()`:

```python
from posthoc_xai.methods import METHOD_REGISTRY

print(list(METHOD_REGISTRY.keys()))
# ['vanilla_gradient', 'integrated_gradients', 'smooth_grad',
#  'gradient_x_input', 'perturbation', 'feature_ablation', 'sarfa']

# Used internally by:
results = xai.explain(model, obs, methods=["vanilla_gradient", "sarfa"])
```

---

## Full Example

```python
import posthoc_xai as xai
from posthoc_xai.visualization.heatmaps import plot_entity_importance, plot_agent_comparison
from posthoc_xai.metrics.sparsity import compute_all as sparsity_metrics

# 1. Load
model = xai.load_model(
    "runs_rlc/womd_sac_road_perceiver_minimal_42",
    data_path="data/training.tfrecord",
)

# 2. Get observation
scenario = next(model._loaded.data_gen)
state = model._loaded.env.reset(scenario)
obs = state.observation.reshape(-1)

# 3. Run methods
vg = xai.VanillaGradient(model)
ig = xai.IntegratedGradients(model, n_steps=30)
attr_vg = vg(obs)
attr_ig = ig(obs)

# 4. Inspect per-agent importance
for name, imp in attr_ig.entity_importance["other_agents"].items():
    print(f"  {name}: {imp:.4f}")

# 5. Check validity
validity = model.get_entity_validity(obs)

# 6. Visualize
fig = plot_entity_importance(attr_ig, validity)
fig.savefig("entity_importance.png", dpi=150, bbox_inches="tight")

fig = plot_agent_comparison([attr_vg, attr_ig], validity)
fig.savefig("agent_comparison.png", dpi=150, bbox_inches="tight")

# 7. Metrics
metrics = sparsity_metrics(attr_ig)
print(metrics)  # {'gini': 0.85, 'top_10_concentration': 0.42, ...}
```
