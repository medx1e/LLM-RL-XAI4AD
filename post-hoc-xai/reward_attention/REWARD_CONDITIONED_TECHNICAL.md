# Reward-Conditioned Attention — Technical Reference

> Session context file. Read at start of every session working on this research direction.

---

## 1. Directory Structure

```
post-hoc-xai/
├── reward_attention/               # Main analysis package
│   ├── __init__.py
│   ├── probe.py                    # Phase 0: verify attention extraction
│   ├── config.py                   # AnalysisConfig, TOKEN_RANGES, TimestepRecord
│   ├── extractor.py                # AttentionTimestepCollector (batched inference)
│   ├── risk_metrics.py             # RiskComputer → collision/safety/navigation/behavior risk
│   ├── correlation.py              # CorrelationAnalyzer (per-scenario + pooled + Fisher z)
│   ├── temporal.py                 # TemporalAnalyzer (event catalog window trajectories)
│   ├── visualization.py            # All 4 figure types (scatter, heatmap, temporal, bar)
│   ├── run_experiment.py           # Main CLI: one model → full results + figures
│   └── bev_attention.py            # BEV + scenario-colored scatter + timeseries
│
├── results/reward_attention/
│   ├── womd_sac_road_perceiver_complete_42/
│   │   ├── timestep_data.pkl       # 393 records (5 scenarios × ~80 steps)
│   │   ├── results.json
│   │   ├── summary_table.csv
│   │   ├── fig1_scatter_*.png
│   │   ├── fig2_correlation_heatmap.png
│   │   ├── fig4_action_attention.png
│   │   ├── fig_scenario_scatter.png        # scenario-colored scatter (key figure)
│   │   ├── fig_timeseries_s000.png
│   │   ├── fig_timeseries_s002.png         # most informative timeseries
│   │   ├── fig_bev_panel_s000.png
│   │   └── fig_bev_panel_s002.png
│   ├── womd_sac_road_perceiver_minimal_42/
│   │   ├── timestep_data.pkl       # 240 records (3 scenarios)
│   │   ├── fig_timeseries_s002.png
│   │   └── ...
│   ├── womd_sac_road_perceiver_basic_42/
│   │   ├── timestep_data.pkl       # 157 records (3 scenarios, 2 early terminations)
│   │   └── ...
│   └── fig_complete_vs_minimal_s002.png    # 2-model overlay (key figure)
│       fig_3way_comparison_s002.png        # 3-model comparison (KEY paper figure)
│
├── events/test_catalog.json        # 152 events from 5 scenarios (s000–s004)
├── logs/                           # Experiment logs
└── REWARD_CONDITIONED_*.md         # This context file set
```

---

## 2. Environment Setup

```bash
# Always activate before running anything
eval "$(~/anaconda3/bin/conda shell.bash hook)" && conda activate vmax
export PYTHONPATH=/home/med1e/post-hoc-xai/V-Max:$PYTHONPATH
cd /home/med1e/post-hoc-xai
```

---

## 3. CLI Commands

### Run full experiment (one model, N scenarios)
```bash
python reward_attention/run_experiment.py \
    --model runs_rlc/womd_sac_road_perceiver_complete_42 \
    --data data/training.tfrecord \
    --n-scenarios 50 \
    --output results/reward_attention

# Quick flags:
#   --quick          skip temporal analysis (saves ~20%)
#   --n-scenarios 1  single-scenario test
#   --n-scenarios 3  reach s002 (index 2 in data generator)
```

### Generate BEV + scenario-colored scatter (no GPU for scatter/timeseries)
```bash
# Scatter + timeseries only (fast, no GPU):
python reward_attention/bev_attention.py --scatter-only --scenario-idx 2

# Full: scatter + timeseries + BEV key-timestep panel:
python reward_attention/bev_attention.py --scenario-idx 2 --no-video

# Full GIF animation (slow, ~80 frames):
python reward_attention/bev_attention.py --scenario-idx 2
```

### Generate timeseries for a specific model pkl (no GPU)
```python
from reward_attention.bev_attention import plot_attention_risk_timeseries
from pathlib import Path

plot_attention_risk_timeseries(
    pkl_path=Path('results/reward_attention/womd_sac_road_perceiver_minimal_42/timestep_data.pkl'),
    scenario_idx=2,
    save_path=Path('results/reward_attention/womd_sac_road_perceiver_minimal_42/fig_timeseries_s002.png'),
)
```

### Run on specific model variants
```bash
# Complete (TTC reward):
python reward_attention/run_experiment.py --model runs_rlc/womd_sac_road_perceiver_complete_42 ...

# Minimal (no TTC):
python reward_attention/run_experiment.py --model runs_rlc/womd_sac_road_perceiver_minimal_42 ...

# Basic (collision+offroad+redlight only):
python reward_attention/run_experiment.py --model runs_rlc/womd_sac_road_perceiver_basic_42 ...
```

### Load pkl and run custom analysis
```python
import pickle
import numpy as np
from scipy import stats

with open('results/reward_attention/womd_sac_road_perceiver_complete_42/timestep_data.pkl', 'rb') as f:
    records = pickle.load(f)

# Filter to scenario 2
recs = sorted([r for r in records if r.scenario_id == 2], key=lambda r: r.timestep)

# Available fields on TimestepRecord:
# scenario_id, timestep
# attn_sdc, attn_agents, attn_roadgraph, attn_lights, attn_gps
# attn_to_nearest, attn_to_threat
# collision_risk, safety_risk, navigation_risk, behavior_risk
# min_ttc, accel, steering, ego_speed
# num_valid_agents, is_collision_step, is_offroad_step
```

---

## 4. Key Technical Facts

### Token Structure (280 tokens total)
```python
TOKEN_RANGES = {
    "sdc":            (0,   5),    # ego × 5 timesteps
    "other_agents":   (5,   45),   # 8 agents × 5 timesteps
    "roadgraph":      (45,  245),  # 200 road points × 1
    "traffic_lights": (245, 270),  # 5 lights × 5 timesteps
    "gps_path":       (270, 280),  # 10 waypoints × 1
}
```

### Attention Extraction
- **Source**: `posthoc_xai/models/perceiver_wrapper.py` → `_extract_attention()`
- **Method**: reconstruct from Q (Dense_0) and K (Dense_1) intermediates
  ```
  scores = einsum('bqhd,bthd->bhqt', Q, K) / sqrt(head_dim)
  attn = softmax(scores, axis=-1)  # (batch, n_heads, n_queries, n_tokens)
  attn_avg = mean over heads → (batch, n_queries, n_tokens)
  ```
- **Output keys**: `cross_attn_layer_0` through `cross_attn_layer_3`, `cross_attn_avg`
- **Architecture**: 4 cross-attention layers, 2 heads, head_dim=16, 16 latent queries

### Risk Metrics
```python
collision_risk  = clip(1 - min_TTC / 3.0, 0, 1)     # 0=safe, 1=imminent collision
safety_risk     = max(collision_risk, offroad_risk)
navigation_risk = step_offroad.astype(float)          # binary proxy
behavior_risk   = clip(|ego_accel| / 0.3, 0, 1)
```

### Correlation Method
- **Primary**: Spearman ρ (rank-based, robust to normalized coordinate data)
- **Multi-scenario aggregation**: Fisher z-transform for proper averaging
- **Key filter**: `min_x_std=0.2` — only include scenarios where risk varies sufficiently
- **Subgroups**: "all", "high_risk" (safety_risk ≥ 0.5), "braking" (accel < -0.15)

### Constraints
- **One model per process**: Waymax global metric registry. Cannot load two models simultaneously.
- **GPU**: GTX 1660 Ti, 6GB VRAM
- **JIT compilation**: ~10 minutes on first run per model
- **Episode loop speed**: ~400–500s per scenario (Python bottleneck, not GPU)
- **Scenario indexing**: scenario_id in pkl = 0-indexed position in data generator

### Reward Configurations
| Config | Reward Terms | Key difference |
|--------|-------------|----------------|
| basic | collision + offroad + red_light | No navigation, no TTC |
| minimal | basic + off_route + progression | Adds navigation incentive |
| complete | minimal + speed + TTC(1.5s, w=0.3) + comfort | Adds continuous proximity penalty |

---

## 5. Data Generator Behavior

The data generator (`model._loaded.data_gen`) is sequential and stateful:
- `next(data_gen)` → scenario 0, then 1, then 2, ...
- To reach scenario index N, must advance through 0..N-1 first
- Resetting requires reloading the model (expensive — JIT recompile)
- `run_experiment.py --n-scenarios 3` gives scenarios 0, 1, 2 (s000, s001, s002)

### Scenario Map (data/training.tfrecord)
| Index | Event catalog ID | Events | Critical | Risk std | Notes |
|-------|-----------------|--------|----------|----------|-------|
| 0 | s000 | 34 | 22 | 0.438 | High variation, 2 risk cycles |
| 1 | s001 | 45 | 35 | 0.158 | Near-constant high risk |
| 2 | s002 | 17 | 15 | 0.395 | **Most interesting**: 2 clear cycles, ρ=+0.769** |
| 3 | s003 | 8 | 8 | 0.097 | Near-constant low risk |
| 4 | s004 | 48 | 42 | 0.082 | Near-constant max risk |

**Rule**: High-variation filter (std > 0.2) keeps s000 and s002. s001, s003, s004 are filtered out.

---

## 6. Scenario Selection for XAI Analysis

**Do not use event count as the primary selection criterion.**

Good scenarios for correlation analysis require:
```python
std(collision_risk) > 0.3      # sufficient variation for correlation
min(collision_risk) < 0.2      # has calm phases (needed for contrast)
max(collision_risk) > 0.7      # has dangerous phases
# bonus: multiple risk cycles (like s002) → stronger evidence
```

The event mining module currently ranks by event count — this selects s004 (48 events) and s001 (45 events), both near-constant high-risk and useless for XAI. **TODO**: update event mining to score by risk dynamic range.

---

## 7. Files Modified from Base Framework

```
posthoc_xai/models/perceiver_wrapper.py   MODIFIED:
  - _extract_attention(): real softmax(Q@K^T/sqrt(d)) from Dense_0/Dense_1
  - observation_structure_detailed property added
```

---

## 8. Pending Experiments

| Experiment | Command | Status | Purpose |
|-----------|---------|--------|---------|
| 50-scenario complete | `run_experiment.py --n-scenarios 50` | **TODO (tonight)** | Statistical power |
| 50-scenario minimal | same, minimal model | TODO | Comparison at scale |
| Event mining fix | refactor scoring | TODO | Better scenario selection |
| BEV GIF animation | `bev_attention.py --scenario-idx 2` | TODO | Visualization |
