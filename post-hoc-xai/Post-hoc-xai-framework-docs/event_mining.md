# Event Mining Module — Implementation Documentation

## Overview

The `event_mining/` module detects **frame-level critical driving events** from V-Max autonomous driving model rollouts. While the existing scanner (`posthoc_xai/experiments/scanner.py`) only captures episode-level flags (collision yes/no, offroad yes/no), event mining adds:

- **Exact timestep** a hazard begins
- **Which agent** caused it
- **Severity classification** (LOW → CRITICAL)
- **Analysis windows** around critical moments for targeted XAI
- **BEV video rendering** of scenarios with event overlays

---

## Architecture

```
event_mining/
├── __init__.py                        # Top-level: EventMiner, EventCatalog, mine_events
├── __main__.py                        # python -m event_mining support
├── events/
│   ├── __init__.py                    # ALL_DETECTORS list, re-exports
│   ├── base.py                        # Event, EventType, Severity, ScenarioData, EventDetector ABC
│   ├── safety.py                      # HazardOnsetDetector, NearMissDetector
│   ├── action.py                      # HardBrakeDetector, EvasiveSteeringDetector
│   └── outcome.py                     # CollisionDetector, OffRoadDetector
├── catalog.py                         # EventCatalog (queryable, JSON-serializable)
├── miner.py                           # EventMiner orchestrator + mine_events() convenience fn
├── metrics.py                         # TTC, distance, criticality computations (numpy)
├── integration/
│   ├── __init__.py                    # Lazy imports (deferred JAX dependency)
│   ├── vmax_adapter.py                # VMaxAdapter: extracts ScenarioData from V-Max rollouts
│   └── xai_bridge.py                  # XAIBridge: feeds events into posthoc_xai pipeline
├── visualization/
│   ├── __init__.py
│   └── bev_video.py                   # BEVRenderer, render_scenario_video, render_event_clip
└── cli.py                             # CLI: mine, summary, export, render commands
```

**16 files total.** No existing files were modified.

---

## Core Data Structures

### EventType (enum)

| Value | Description |
|-------|-------------|
| `HAZARD_ONSET` | TTC drops below threshold — ego is on a collision course |
| `COLLISION_IMMINENT` | Reserved for very low TTC (future use) |
| `HARD_BRAKE` | Large negative acceleration (< -3.0 m/s²) |
| `EVASIVE_STEERING` | Large steering magnitude (> 0.3 rad) |
| `NEAR_MISS` | Agent comes very close without collision (< 3.0m) |
| `COLLISION` | Actual collision occurred |
| `OFF_ROAD` | Ego vehicle left the road |

### Severity (enum)

| Level | Score | Meaning |
|-------|-------|---------|
| `LOW` | 0.25 | Minor, informational |
| `MEDIUM` | 0.50 | Notable, worth reviewing |
| `HIGH` | 0.75 | Dangerous situation |
| `CRITICAL` | 1.00 | Collision or imminent collision |

### Event (dataclass)

Each detected event has:

```python
Event(
    event_type=EventType.HAZARD_ONSET,
    severity=Severity.HIGH,
    onset=35,           # first timestep of event
    peak=40,            # most intense timestep (best for XAI)
    offset=45,          # last timestep of event
    window=(25, 55),    # analysis window with padding
    scenario_id="s003",
    causal_agent_id=2,  # which other agent is involved
    metadata={          # detector-specific info
        "min_ttc": 0.8,
        "ttc_threshold": 3.0,
    },
)
```

- `event.duration` → `offset - onset + 1`
- `event.severity_score` → float in [0.25, 1.0]
- `event.to_dict()` / `Event.from_dict()` for JSON serialization

### ScenarioData (dataclass)

All per-timestep data extracted from a V-Max episode. All arrays are numpy.

```
Ego trajectory:      ego_x, ego_y, ego_vx, ego_vy, ego_yaw, ego_length, ego_width  — all (T,)
Ego actions:         ego_accel, ego_steering  — (T,)
Other agents:        other_agents (T, N_agents, 7)  — [x, y, vx, vy, yaw, length, width]
                     other_agents_valid (T, N_agents) bool
Road graph:          road_graph (N_rg, features), road_graph_valid — static
Traffic lights:      traffic_lights (T, N_tl, features), traffic_lights_valid
GPS path:            gps_path (N_gps, 2) — static route

Derived metrics (computed by metrics.py):
    ttc              (T, N_agents) time-to-collision
    min_distance     (T, N_agents) Euclidean distance per agent
    nearest_agent_id (T,) index of closest agent
    criticality      (T,) composite score in [0, 1]

Per-step flags:      step_collision (T,), step_offroad (T,)
Episode outcomes:    has_collision, collision_time, has_offroad, offroad_time, route_completion
Raw observations:    raw_observations (T, obs_dim) — optional, for XAI methods
```

Coordinates are **ego-relative** as provided by V-Max's observation unflattener.

---

## Event Detectors

All detectors inherit from `EventDetector` ABC and implement `detect(ScenarioData) -> List[Event]`.

### HazardOnsetDetector (`events/safety.py`)
- **Triggers when:** TTC to any agent < `ttc_threshold` (default 3.0s) for `min_duration` (default 3) consecutive steps
- **Severity:** Based on minimum TTC reached — CRITICAL if < 1.0s, HIGH if < 1.5s, etc.
- **Causal agent:** The specific agent with low TTC
- **Iterates:** Per-agent TTC columns independently

### NearMissDetector (`events/safety.py`)
- **Triggers when:** Distance to agent < `distance_threshold` (default 3.0m) WITHOUT a collision
- **Skips:** If `has_collision` is True (that's a CollisionDetector's job)
- **Severity:** Based on minimum distance — CRITICAL if < 1.0m

### HardBrakeDetector (`events/action.py`)
- **Triggers when:** `ego_accel` < `accel_threshold` (default -3.0 m/s²) for `min_duration` (default 2) steps
- **Severity:** HIGH if accel < -6.0 m/s²
- **Causal agent:** Nearest agent at peak timestep

### EvasiveSteeringDetector (`events/action.py`)
- **Triggers when:** `|ego_steering|` > `steering_threshold` (default 0.3 rad) for `min_duration` (default 2) steps
- **Severity:** HIGH if > 0.6 rad
- **Causal agent:** Nearest agent at peak timestep

### CollisionDetector (`events/outcome.py`)
- **Triggers when:** `has_collision` is True
- **Onset detection:** Traces TTC backwards from collision time to find when the dangerous approach began
- **Severity:** Always CRITICAL
- **Causal agent:** Nearest valid agent at collision time

### OffRoadDetector (`events/outcome.py`)
- **Triggers when:** `has_offroad` is True, uses `step_offroad` per-step flags
- **Finds:** Continuous offroad windows
- **Severity:** CRITICAL if >= 10 steps, HIGH if >= 5, else MEDIUM

### Shared Helpers (in `EventDetector` base class)
- `_find_continuous_windows(condition, min_duration)` — finds runs of True in a boolean array
- `_compute_window(onset, offset, total_steps, padding)` — adds padding for analysis context
- `_find_event_peak(signal, onset, offset, mode)` — finds min/max within event window

---

## Safety Metrics (`metrics.py`)

All functions are pure numpy — no JAX dependency.

### `compute_distances(ego_x, ego_y, other_x, other_y, valid) → (T, N)`
Euclidean distance from ego to each agent. Invalid agents get `inf`.

### `compute_ttc(ego_*, other_*, valid, max_ttc=10.0) → (T, N)`
Simplified point-mass linear TTC: `TTC = -dot(dp, dv) / dot(dv, dv)` for closing pairs. Non-closing or invalid pairs get `max_ttc`.

### `compute_criticality(ttc, min_distance, ego_speed) → (T,)`
Composite score in [0, 1] combining:
- 50% TTC component (1 when TTC=0, 0 when TTC ≥ 5s)
- 30% distance component (1 when dist=0, 0 when dist ≥ 20m)
- 20% speed component (normalized by 30 m/s)

---

## V-Max Adapter (`integration/vmax_adapter.py`)

### How Data Extraction Works

The adapter runs a full episode rollout and collects per-step data from **unflattened observations**:

```python
# At each timestep:
obs = env_transition.observation         # (1, obs_dim) flat
features, masks = unflatten_fn(obs)      # structured
sdc_feat, other_feat, rg_feat, tl_feat, gps_feat = features

# other_feat shape: (1, N_agents, N_timesteps, 7)
# features = [x, y, vx, vy, yaw, length, width]
# Take most recent timestep (index -1) for current positions
```

This is necessary because V-Max doesn't expose raw `sim_state.x/y` — we get ego-relative coordinates from the observation pipeline.

### VMaxAdapter API

```python
adapter = VMaxAdapter(store_raw_obs=True)

# Pre-compile JAX functions (call once, reuse across scenarios)
adapter.prepare(model)

# Extract data from one scenario
scenario_data = adapter.extract_scenario_data(model, scenario, "s000", rng_seed=0)
```

The `prepare()` method JIT-compiles `env.reset` and the step function once, avoiding re-compilation per scenario. This is important on the 6GB GTX 1660 Ti where memory is tight.

### What Gets Extracted Per Step
1. **Ego state** from `sdc_feat` (most recent timestep in history window)
2. **Other agent states** from `other_feat` (most recent timestep, per agent)
3. **Actions** from `transition.action` (acceleration, steering)
4. **Metrics** from `env_transition.metrics` (collision, offroad flags)
5. **Traffic lights** and **road graph** from observations
6. **Raw flat observations** (optional, for later XAI analysis)

After collection, the adapter computes derived metrics (TTC, distances, criticality) automatically.

---

## EventCatalog (`catalog.py`)

Queryable collection with filtering and serialization:

```python
catalog = EventCatalog(events)

# Filtering
critical = catalog.filter(min_severity_score=0.75)
collisions = catalog.filter(event_type="collision")
s003_events = catalog.filter(scenario_id="s003")

# Grouping
by_scenario = catalog.by_scenario()   # dict[str, list[Event]]
by_type = catalog.by_type()           # dict[EventType, list[Event]]

# Analysis points for XAI
windows = catalog.get_windows()        # [(scenario_id, start, end, event), ...]
points = catalog.get_analysis_points() # [(scenario_id, peak_timestep, event), ...]

# Stats
catalog.summary()
# {'total_events': 23, 'unique_scenarios': 12,
#  'by_type': {'hazard_onset': 8, 'hard_brake': 5, ...},
#  'by_severity': {'critical': 3, 'high': 7, ...}}

# Serialization
catalog.save("events/catalog.json")
loaded = EventCatalog.load("events/catalog.json")
```

---

## EventMiner (`miner.py`)

Orchestrator that runs all detectors:

```python
# Basic: run detectors on pre-extracted data
miner = EventMiner()                    # uses ALL_DETECTORS by default
events = miner.mine_scenario(scenario_data)

# Full pipeline: load scenarios, run episodes, detect events
catalog = miner.mine_from_model(model, n_scenarios=50, save_path="events/catalog.json")

# Convenience function (loads model if given a path)
from event_mining import mine_events
catalog = mine_events("runs_rlc/womd_sac_road_perceiver_minimal_42", n_scenarios=50)
```

Custom detectors:

```python
miner = EventMiner(detectors=[
    HazardOnsetDetector(ttc_threshold=2.0),  # stricter
    CollisionDetector(),
    HardBrakeDetector(accel_threshold=-4.0),  # stricter
])
```

---

## XAI Bridge (`integration/xai_bridge.py`)

Connects mined events to the posthoc_xai framework:

```python
from event_mining.integration.xai_bridge import XAIBridge

bridge = XAIBridge(catalog)

# Iterate over key timesteps for XAI analysis
for scenario_id, timestep, metadata in bridge.iter_event_timesteps():
    # metadata has: event_type, severity, role (onset/peak), causal_agent_id
    obs = get_obs_at_timestep(model, scenario_id, timestep)
    attributions = xai.explain(model, obs)

# Iterate over analysis windows
for scenario_id, start, end, event in bridge.iter_analysis_windows(min_severity_score=0.5):
    # Run XAI over entire window for temporal analysis
    ...

# Get timeline for one scenario
timeline = bridge.get_scenario_event_timeline("s003")
# [{'onset': 35, 'peak': 40, 'type': 'hazard_onset', ...}, ...]

# Export to pandas
df = bridge.to_dataframe()
df.to_csv("events.csv")
```

---

## BEV Video Visualization (`visualization/bev_video.py`)

Uses **Waymax's native `plot_simulator_state()`** for the base frame (proper world coordinates, full road geometry, vehicle polygons) and overlays event markers on top.

### Why Waymax Native Visualization

The original implementation used a custom matplotlib BEV renderer drawing from extracted ScenarioData. However, V-Max observations are ego-relative and normalized (~[-1, 1]), producing unrealistic visualizations. Switching to Waymax's native renderer gives:
- Proper world-frame coordinates
- Full road geometry with lane markings
- Correctly sized and positioned vehicle polygons
- Professional-quality frames

### Key Implementation Details

- States must be **unbatched** before passing to Waymax's visualizer: `jtu.tree_map(lambda x: x[0], env_transition.state)`
- The overlay system adds event info text, colored borders during active events, and a timeline bar at the bottom
- Supports both model rollout mode and (partially) logged trajectory mode

### Rendering Functions

```python
from event_mining.visualization.bev_video import render_model_video, render_event_clip

# Full scenario video with event overlays (model rollout)
render_model_video(model, scenario, events, "output/s003.gif", fps=10, rng_seed=0)

# Focused clip around one event (±padding steps)
render_event_clip(model, scenario, event, "output/s003_hazard.gif", fps=10, padding=5, rng_seed=0)

# Logged trajectory mode (no model needed, but data format must be compatible)
from event_mining.visualization.bev_video import render_logged_video
render_logged_video("data/training.tfrecord", record_index=0, events=events, output_path="logged.gif")
```

**Note:** Logged trajectory mode may fail with some tfrecord formats that don't include `roadgraph_samples/*` keys. Model rollout mode works reliably.

### Event Color Coding

| Event Type | Color |
|-----------|-------|
| HAZARD_ONSET | Orange (#FF6600) |
| COLLISION_IMMINENT | Red (#FF0000) |
| HARD_BRAKE | Purple (#CC00CC) |
| EVASIVE_STEERING | Violet (#9900FF) |
| NEAR_MISS | Yellow (#FFCC00) |
| COLLISION | Red (#FF0000) |
| OFF_ROAD | Cyan (#00CCFF) |

---

## CLI

```bash
# Mine events from model rollouts
python -m event_mining mine \
    --model runs_rlc/womd_sac_road_perceiver_minimal_42 \
    --data data/training.tfrecord \
    --n-scenarios 50 \
    --output events/catalog.json

# View summary
python -m event_mining summary --catalog events/catalog.json

# Export to CSV
python -m event_mining export --catalog events/catalog.json --output events.csv

# Render BEV videos for a scenario
python -m event_mining render \
    --catalog events/catalog.json \
    --model runs_rlc/womd_sac_road_perceiver_minimal_42 \
    --scenario s003 \
    --output videos/
```

---

## Import Structure & Dependencies

### Standalone (no JAX needed)
Everything except `vmax_adapter.py` works without JAX:

```python
from event_mining import EventMiner, EventCatalog, mine_events
from event_mining.events.base import Event, EventType, Severity, ScenarioData
from event_mining.events import ALL_DETECTORS
from event_mining.catalog import EventCatalog
from event_mining.metrics import compute_ttc, compute_distances
from event_mining.integration.xai_bridge import XAIBridge
from event_mining.visualization.bev_video import BEVRenderer
```

### With V-Max (conda vmax env)
```python
from event_mining.integration.vmax_adapter import VMaxAdapter  # needs JAX
from event_mining import mine_events  # calls VMaxAdapter internally
```

The `integration/__init__.py` uses `__getattr__` for lazy imports so importing `event_mining` never fails due to missing JAX.

### External Dependencies
- **numpy** — all metric computations and data storage
- **matplotlib** — BEV rendering
- **pillow** — GIF output (optional, fallback if ffmpeg unavailable)
- **ffmpeg** — MP4 output (optional)
- **pandas** — `XAIBridge.to_dataframe()` only
- **jax, vmax, waymax** — `VMaxAdapter` only (available in conda vmax env)

---

## Testing Summary

### Unit Tests Passing
- 6 detectors on synthetic data → 5 events detected correctly (hazard onset, near miss, hard brake, evasive steering)
- JSON catalog serialization round-trips
- XAI bridge yields correct analysis points (onset + peak per event)
- BEV renderer produces GIF from synthetic data

### Integration Test (PASSED)

Ran on 5 scenarios from `data/training.tfrecord` with `womd_sac_road_perceiver_minimal_42`:
- **153 events detected** across 5 scenarios
- Event types found: hazard_onset, near_miss, hard_brake, evasive_steering
- No collisions or offroad in these 5 scenarios
- Catalog saved to `events/test_catalog.json` (51 KB)

**Note:** High event count due to detector thresholds being calibrated for real-world meters while observations are normalized. Many events are false positives (especially near_miss spanning entire episodes). Threshold recalibration needed.

### End-to-End Experiment (PASSED)

Ran `experiments/event_xai_experiment.py`: mine events → select top 3 → run vanilla_gradient + integrated_gradients across event windows → generate temporal plots.
- 9 plots generated in `experiments/event_xai_results/`
- Full analysis: `experiments/results_analysis.md`

---

## Relationship to Existing Framework

```
posthoc_xai/                          event_mining/
├── experiments/scanner.py  ←------→  integration/vmax_adapter.py
│   (episode-level flags)             (frame-level trajectories)
│                                      ↓
├── methods/ (VanillaGrad, IG, etc)   events/ (detectors)
│                                      ↓
├── experiments/runner.py  ←--------  integration/xai_bridge.py
│   (runs XAI at timesteps)           (feeds event timesteps)
│                                      ↓
└── (future) visualization  ←------  visualization/bev_video.py
                                      (BEV rendering)
```

The event_mining module sits **between** the scanner (which finds scenarios) and the XAI methods (which explain decisions). It answers: **"which timesteps in which scenarios are worth explaining?"**

---

## Known Issues

### 1. Detector Threshold Calibration
V-Max observations are ego-relative and normalized to ~[-1, 1], not real-world meters. Current thresholds (TTC < 3.0s, distance < 3.0m, accel < -3.0 m/s²) are for real-world units, causing oversensitive detection. Suggested recalibrated thresholds for normalized space:
- Distance: ~0.15 (instead of 3.0)
- TTC: ~0.5 (instead of 3.0)
- Acceleration: ~-0.3 (instead of -3.0)
- Steering: ~0.5 (instead of 0.3)

### 2. Causal Agent Identification
All detected events tend to flag agent_0 as causal, likely because it's the nearest agent by observation ordering and always has the lowest normalized distance/TTC.

### 3. Logged Trajectory BEV Rendering
`render_logged_video()` may fail with some tfrecord formats that don't include `roadgraph_samples/*` keys (the V-Max data uses a different schema). Model rollout mode via `render_model_video()` works reliably.
