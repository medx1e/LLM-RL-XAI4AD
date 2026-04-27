# CBM Platform Tab — Implementation Guide

This guide is for the agent implementing the CBM tab in the Streamlit platform at `/home/med1e/platform_fyp/`.

---

## Platform Architecture (read this first)

The platform lives at `/home/med1e/platform_fyp/`. Entry point: `app.py`.

**How the existing tabs work:**
```
app.py
  → registers tabs in _TABS dict
  → calls tab_module.render() based on sidebar selection

platform/tabs/tab_posthoc.py   ← existing post-hoc XAI tab (reference)
platform/tabs/tab_home.py      ← existing home tab

platform/shared/
  model_catalog.py   — PLATFORM_MODELS dict: model name → ModelEntry (path, encoder, etc.)
  scenario_store.py  — load_artifact() / save_artifact() for PlatformScenarioArtifact
  contracts.py       — PlatformScenarioArtifact dataclass (wraps ScenarioData + metadata)
  bev_component.py   — render_bev_player(artifact) Streamlit component
```

**How existing scenarios are served:**
- Pre-computed artifacts live in `platform_cache/{model_slug}/scenario_{idx:04d}_artifact.pkl`
- Each artifact is a `PlatformScenarioArtifact` containing a `ScenarioData` (BEV frames, rewards, dones)
- Tabs call `load_artifact(model_key, scenario_idx)` from `scenario_store.py`

**The CBM tab is different — simpler, standalone:**
- Do NOT touch model_catalog, scenario_store, or bev_component for this tab
- The CBM tab reads directly from two pre-built files (already on disk):
  - `/home/med1e/platform_fyp/cbm/curated_scenarios.json`
  - `/home/med1e/platform_fyp/cbm/curated_scenarios_data.npz`
- No JAX, no Waymax, no model loading — pure numpy + matplotlib + streamlit

---

## The Pre-Built Data Files

### `curated_scenarios.json`
Structure:
```json
{
  "model": "cbm_scratch_v2_lambda05",
  "n_total": 400,
  "top_k": 10,
  "archetypes": {
    "red_light_stop": [
      {"rank": 1, "local_idx": 42, "scenario_idx": 246, "score": 0.9491,
       "progress": 1.0, "no_at_fault": true, "episode_done": false},
      ...
    ],
    "ttc_success":     [...],
    "curvature_nav":   [...],
    "concept_failure": [...]
  }
}
```
- `local_idx`: index into the npz arrays (0–399 range of the 400 val scenarios)
- `scenario_idx`: the actual WOMD scenario number
- 10 entries per archetype, ranked by score

### `curated_scenarios_data.npz`
Arrays for the **36 unique top scenarios** only (not all 400):
```python
data = np.load("curated_scenarios_data.npz", allow_pickle=True)

data["pred_concepts"]    # (80, 36, 15)  float32 — model's concept predictions per step
data["true_concepts"]    # (80, 36, 15)  float32 — ground-truth concept values per step
data["valid_mask"]       # (80, 36, 15)  bool    — is concept valid at this step?
data["ego_actions"]      # (80, 36, 2)   float32 — [acceleration, steering] per step
data["dones"]            # (80, 36)      float32 — episode terminated flag per step
data["rewards"]          # (80, 36)      float32 — reward per step
data["driving_metrics"]  # (80, 36, 14)  float32 — all Waymax driving metrics per step
data["local_indices"]    # (36,)         int     — which local_idx each column corresponds to
data["scenario_indices"] # (36,)         int     — WOMD scenario numbers
data["concept_names"]    # (15,)         str     — concept name per column index
data["driving_metric_keys"] # (14,)      str     — driving metric name per column index
```

**How to get the npz column for a given local_idx from the JSON:**
```python
local_idx = entry["local_idx"]  # from JSON
col = np.where(data["local_indices"] == local_idx)[0][0]  # column in npz arrays
pred = data["pred_concepts"][:, col, :]   # (80, 15) — this scenario's predictions
```

### Concept names (column order in npz):
```
0  ego_speed            Phase 1 continuous
1  ego_acceleration     Phase 1 continuous
2  dist_nearest_object  Phase 1 continuous
3  num_objects_within_10m Phase 1 continuous
4  traffic_light_red    Phase 1 BINARY
5  dist_to_traffic_light Phase 1 continuous
6  heading_deviation    Phase 1 continuous
7  progress_along_route Phase 1 continuous
8  ttc_lead_vehicle     Phase 2 continuous
9  lead_vehicle_decelerating Phase 2 BINARY
10 at_intersection      Phase 2 BINARY
11 path_curvature_max   Phase 3 continuous
12 path_net_heading_change Phase 3 continuous
13 path_straightness    Phase 3 continuous
14 heading_to_path_end  Phase 3 continuous
```

### Top curated scenarios (for hardcoding / reference):

**red_light_stop** — best 3:
| Rank | WOMD Scenario | Score | Progress |
|---|---|---|---|
| 1 | 246 | 0.949 | 1.000 ✓ |
| 2 | 198 | 0.939 | 1.000 ✓ |
| 3 | 329 | 0.929 | 1.000 ✓ |

**ttc_success** — best 3:
| Rank | WOMD Scenario | Score | Progress |
|---|---|---|---|
| 1 | 327 | 0.876 | 0.844 ✓ |
| 2 | 381 | 0.842 | 1.000 ✓ |
| 3 | 307 | 0.801 | 1.000 ✓ |

**curvature_nav** — best 3:
| Rank | WOMD Scenario | Score | Progress |
|---|---|---|---|
| 1 | 244 | 0.978 | 1.000 ✓ |
| 2 | 57  | 0.924 | 1.000 ✓ |
| 3 | 109 | 0.920 | 1.000 ✓ |

**concept_failure** — best 3:
| Rank | WOMD Scenario | Score | Progress |
|---|---|---|---|
| 1 | 142 | 0.842 | 0.546 ✗ collision |
| 2 | 278 | 0.816 | 1.000 ✗ |
| 3 | 309 | 0.815 | 1.000 ✗ |

---

## What to Build

### Files to create:
1. **`platform/tabs/tab_cbm.py`** — the new CBM tab (main implementation)

### Files to modify:
2. **`app.py`** — register the new tab in `_TABS`

That's it. No other files need touching.

---

## tab_cbm.py — Detailed Spec

### Layout

```
Sidebar:
  - Archetype selector (radio): red_light_stop | ttc_success | curvature_nav | concept_failure
  - Scenario rank selector (slider 1–10): selects which of the top-10 to show
  - Key concepts to display (multiselect): pre-filled with archetype-relevant defaults

Main area (2 columns):
  Left (40%):
    - Scenario info card (WOMD idx, score, progress, collision status)
    - Archetype description
    - Ego Actions plot (acceleration + steering over 80 steps)
    - Key driving metrics (route progress, collision) as st.metric

  Right (60%):
    - Concept Timeline plot (main visual):
        One subplot per selected concept
        Blue line = ground truth, orange dashed = CBM prediction
        Shaded where valid_mask=False
        Red vertical line = current timestep (from slider)
    - Timestep slider (0–79) to scrub through the episode
```

### Archetype descriptions and default concepts:

```python
ARCHETYPE_META = {
    "red_light_stop": {
        "label": "Red Light Stop",
        "description": "The CBM correctly identifies a red traffic light and applies brakes. "
                       "Concept: traffic_light_red predicted high → ego decelerates.",
        "default_concepts": ["traffic_light_red", "dist_to_traffic_light", "ego_speed"],
        "action_focus": "acceleration",   # highlight braking
    },
    "ttc_success": {
        "label": "TTC Success",
        "description": "A lead vehicle slows down. The CBM tracks Time-to-Collision and "
                       "brakes before impact.",
        "default_concepts": ["ttc_lead_vehicle", "lead_vehicle_decelerating", "ego_speed"],
        "action_focus": "acceleration",
    },
    "curvature_nav": {
        "label": "Curve Navigation",
        "description": "The planned path has high curvature. The CBM detects it via "
                       "path_curvature_max and steers appropriately.",
        "default_concepts": ["path_curvature_max", "path_straightness", "heading_deviation"],
        "action_focus": "steering",
    },
    "concept_failure": {
        "label": "Concept Failure",
        "description": "The CBM misreads a key concept (pred ≠ true) leading to a "
                       "suboptimal or dangerous outcome.",
        "default_concepts": ["traffic_light_red", "dist_nearest_object", "ttc_lead_vehicle"],
        "action_focus": "acceleration",
    },
}
```

### Concept Timeline Plot spec:

```python
# For each selected concept, one subplot:
# - x axis: timestep 0–79
# - y axis: [0, 1] (all concepts normalized)
# - Blue solid line: true_concepts[:, col, concept_idx]
# - Orange dashed line: pred_concepts[:, col, concept_idx]
# - Grey shading: where valid_mask[:, col, concept_idx] == False
# - Red vertical dashed line: current_step from slider
# - Title: concept name + "(binary)" or "(continuous)"
# - Annotation: R² or accuracy in top-right corner of subplot
# - Legend: "Ground Truth" / "CBM Prediction" (once, at figure level)
```

### Action Timeline Plot spec:

```python
# Single figure, 2 subplots stacked:
# Top: ego_actions[:, col, 0]  — acceleration (-1 to +1)
#   - Fill red below 0 (braking), fill green above 0 (accelerating)
#   - Red vertical dashed line: current_step
# Bottom: ego_actions[:, col, 1]  — steering (-1 to +1)
#   - Fill blue for left turns (negative), orange for right (positive)
#   - Red vertical dashed line: current_step
```

### Data loading (put in module scope, cached):

```python
import json
import numpy as np
import streamlit as st
from pathlib import Path

_CBM_ROOT = Path(__file__).resolve().parent.parent.parent / "cbm"

@st.cache_data
def _load_cbm_data():
    with open(_CBM_ROOT / "curated_scenarios.json") as f:
        index = json.load(f)
    data = np.load(_CBM_ROOT / "curated_scenarios_data.npz", allow_pickle=True)
    return index, data

def _get_scenario_col(data, local_idx: int) -> int:
    """Map local_idx (JSON field) to npz column index."""
    matches = np.where(data["local_indices"] == local_idx)[0]
    return int(matches[0]) if len(matches) else 0
```

### app.py change:

```python
# In _TABS dict, add:
from platform.tabs import tab_home, tab_posthoc, tab_cbm

_TABS = {
    "Home": tab_home,
    "Post-hoc XAI": tab_posthoc,
    "CBM Explorer": tab_cbm,       # ← add this line
}
```

---

## Implementation Notes

1. **No JAX/Waymax imports** — the tab is pure numpy + matplotlib + streamlit. This means it loads instantly without any GPU/model initialization.

2. **`@st.cache_data` on `_load_cbm_data()`** — load the npz and json once, cache across rerenders.

3. **Matplotlib figures** — use `st.pyplot(fig, use_container_width=True)` and `plt.close(fig)` after each. Match the style of the existing tabs.

4. **Concept multiselect** — let the user add/remove concepts dynamically. Default should be the 3 archetype-relevant ones. The multiselect options are all 15 concept names from `data["concept_names"]`.

5. **Valid mask shading** — grey out timesteps where `valid_mask[:, col, concept_idx]` is False. Use `ax.fill_between(steps, 0, 1, where=~valid, color='grey', alpha=0.15, label='Not valid')`.

6. **No BEV rendering** — skip it for now. The concept timeline IS the main visual for the CBM tab. BEV can be added later once we decide on the rendering pipeline.

7. **Archetype color coding**:
   - red_light_stop → red accent
   - ttc_success → orange accent
   - curvature_nav → green accent
   - concept_failure → purple accent

---

## What NOT to do

- Do not touch `model_catalog.py`, `scenario_store.py`, `bev_component.py`, `contracts.py`
- Do not import JAX, Waymax, or anything from `cbm_v1/` — the data is already precomputed
- Do not try to run live rollouts — load from npz only
- Do not modify the existing tabs

---

## How to Test

```bash
cd /home/med1e/platform_fyp
streamlit run app.py
```

Navigate to "CBM Explorer" tab. Select archetype, select rank, observe concept timelines. Change concept selection via multiselect. Scrub timestep slider.

Expected: plots load instantly (no spinner), all 4 archetypes switch smoothly, concept timelines show pred vs true with valid mask shading.
