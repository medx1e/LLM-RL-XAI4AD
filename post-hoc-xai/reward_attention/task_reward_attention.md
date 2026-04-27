# Reward-Conditioned Attention Framework — Implementation Checkpoint

> Read this file at the start of every session to resume work.
> Mark tasks `[x]` as they are completed.

---

## 1. What We Are Building

A pipeline that answers: **does the Perceiver model's attention reflect what it was rewarded for?**

The Perceiver encoder has 16 learned latent queries that cross-attend over all input entities at every timestep. This produces a `(16, N_tokens)` attention matrix — a direct window into what the model is "looking at." We correlate this attention with continuous risk metrics derived from the scenario state (TTC-based collision risk, route deviation, etc.) to test whether the model's attention is semantically aligned with its reward components.

**Core hypothesis:** An RL agent trained on safety/navigation rewards implicitly learns to direct attention toward features that determine those rewards — without being explicitly told to.

---

## 2. Strategy

- **Start with one model:** `womd_sac_road_perceiver_complete_42`
- **Why complete?** It has an explicit TTC penalty (threshold=1.5s, weight=0.3). Highest prior probability of finding vehicle attention ↔ TTC correlation.
- **Scale later:** run `minimal` and `basic` configs to compare. Expected gradient: complete > minimal > basic in vehicle-attention correlation strength.
- **No cross-architecture comparison** for now. Perceiver only.

---

## 3. Expected Outputs

1. **Correlation scatter plots** — safety_risk vs attn_trajectory, nav_risk vs attn_path. With regression line + r-value + p-value annotation.
2. **Correlation heatmap** — full matrix: all risk metrics × all attention categories.
3. **Temporal event plots** — attention categories + safety risk over event windows (from existing event catalog).
4. **Action-conditioned attention bar chart** — what does the model attend to when braking vs steering vs neutral?
5. **Summary table** — Pearson r, Spearman ρ, p-value, n_samples for each (risk, attention) pair. Publishable result.

---

## 4. Key Technical Facts

### Model
- Path: `runs_rlc/womd_sac_road_perceiver_complete_42`
- Encoder: `perceiver` (remapped to `lq` internally)
- Wrapper: `PerceiverWrapper` (has `get_attention()` via `capture_intermediates`)
- Reward: collision(-1) + offroad(-1) + red_light(-1) + off_route(-0.6) + progression(+0.2) + **TTC(-1, threshold=1.5s, weight=0.3)** + speed(0.3) + comfort(0.3)
- Load with: `posthoc_xai.load_model("runs_rlc/womd_sac_road_perceiver_complete_42", data_path="data/training.tfrecord")`

### Observation Structure (flat, 1655 features)
```
sdc_trajectory:  (0,   40)    — 40 features (1 ego × 5 timesteps × 8 feats)
other_agents:    (40,  360)   — 320 features (8 agents × 5 timesteps × 8 feats)
roadgraph:       (360, 1360)  — 1000 features (200 points × 5 feats)
traffic_lights:  (1360,1635)  — 275 features (5 lights × 5 timesteps × 11 feats)
gps_path:        (1635,1655)  — 20 features (10 waypoints × 2 feats)
```

### Attention Structure (TOKEN space, not feature space)
The LQ encoder tokenizes entities BEFORE the attention. Attention is `(16 queries, N_tokens)`.
Token order mirrors observation unflattening order:
```
token 0:           sdc (ego)
tokens 1-8:        other_agents (8 agents)
tokens 9-208:      roadgraph (200 road points)
tokens 209-213:    traffic_lights (5 lights)
tokens 214-223:    gps_path (10 waypoints)
Total: 224 tokens
```
**⚠ MUST VERIFY:** Run probe.py first to confirm (a) attention extraction returns non-empty dict, (b) shape is (batch, n_queries, n_tokens) or similar, (c) token order is as above.

### Risk Metrics (computed from ScenarioData, normalized coordinates)
- `collision_risk = clip(1 - min_TTC / 3.0, 0, 1)` — using VMaxAdapter's computed TTC
- `safety_risk = max(collision_risk, offroad_risk)`
- `navigation_risk` — proxy from route deviation (GPS path alignment)
- `behavior_risk = clip(|accel| / threshold, 0, 1)`
- Use **Spearman correlation** (rank-based, safe for normalized coordinate data)

### Key Constraints
- **One model per process** — Waymax global metric registry. Cannot load two models in same process.
- **GPU:** GTX 1660 Ti, 6GB VRAM. JIT compilation ~10 min on first run.
- **Conda env:** `vmax`. Activate: `eval "$(~/anaconda3/bin/conda shell.bash hook)" && conda activate vmax`
- **PYTHONPATH:** Must include `V-Max/` directory.
- **Coordinates:** All positions/velocities are ego-relative and normalized (~[-1,1]). Not real-world meters.

### Existing Infrastructure to Reuse
- `event_mining/integration/vmax_adapter.py` → `VMaxAdapter` — runs full episode, extracts per-step (ego state, agent states, TTC, distances, actions). **Reuse this directly.**
- `event_mining/metrics.py` → `compute_ttc()`, `compute_distances()`, `compute_criticality()` — already implemented.
- `events/test_catalog.json` — existing event catalog (152+ events from 5 scenarios). Use for temporal analysis.
- `posthoc_xai/visualization/heatmaps.py` → temporal plot functions already built.
- `posthoc_xai/models/loader.py` → `load_vmax_model()` — handles all 5 compatibility fixes.

---

## 5. Directory Structure to Build

```
reward_attention/
├── __init__.py
├── probe.py                  # STEP 0: verify attention extraction, print intermediates tree
├── config.py                 # AnalysisConfig dataclass, token ranges, risk thresholds
├── extractor.py              # AttentionTimestepCollector: run episode, collect (obs, attention, state) per step
├── risk_metrics.py           # RiskComputer: ScenarioData → continuous risk metrics [0,1]
├── correlation.py            # CorrelationAnalyzer: Pearson + Spearman, subgroup filtering
├── temporal.py               # TemporalAnalyzer: event-window attention trajectories
├── visualization.py          # All paper figures (scatter, heatmap, temporal, action-conditioned)
└── run_experiment.py         # Main script: one model → full results + figures
```

Output:
```
results/reward_attention/
└── womd_sac_road_perceiver_complete_42/
    ├── timestep_data.pkl     # raw (attention, risk) per timestep
    ├── results.json          # summary statistics + correlations
    ├── fig1_scatter_safety.png
    ├── fig2_correlation_heatmap.png
    ├── fig3_temporal_*.png   # one per analyzed event
    ├── fig4_action_attention.png
    └── summary_table.csv     # publishable correlation table
```

---

## 6. Implementation Tasks

### Phase 0: Probe & Verify ✅ COMPLETE
- [x] **0.1** Create `reward_attention/probe.py` — done, ran successfully
- [x] **0.2** Confirm attention tensor shape — confirmed: reconstructed as `(batch, 16, 280)` via Q@K^T softmax
- [x] **0.3** Confirm token count — **CORRECTED: 280 tokens** (not 224): sdc[0:5], agents[5:45], road[45:245], lights[245:270], gps[270:280]
- [x] **0.4** Fixed `_extract_attention` in `perceiver_wrapper.py` — now computes real softmax attention from Q (Dense_0, 1×16×32) and K (Dense_1, 1×280×32) via `einsum('bqhd,bthd->bhqt', Q, K) / sqrt(16)` + softmax + avg over 2 heads
- [x] **0.5** `TOKEN_RANGES` in `config.py` with correct 280-token mapping

### Phase 1: Core Data Structures ✅ COMPLETE
- [x] **1.1** `reward_attention/__init__.py`
- [x] **1.2** `reward_attention/config.py` — TOKEN_RANGES (280), AnalysisConfig, constants
- [x] **1.3** `TimestepRecord` dataclass — all fields implemented

### Phase 2: Attention Extraction ✅ COMPLETE
- [x] **2.1** `reward_attention/extractor.py` — `AttentionTimestepCollector`:
  - Uses VMaxAdapter for episode rollout (store_raw_obs=True)
  - Batched forward pass: `model.forward(obs_batch)` for all T steps at once
  - Aggregates by TOKEN_RANGES, normalizes to fractions

### Phase 3: Risk Computation ✅ COMPLETE
- [x] **3.1** `reward_attention/risk_metrics.py` — `RiskComputer.from_scenario_data()` implemented

### Phase 4: Correlation Analysis ✅ COMPLETE
- [x] **4.1** `reward_attention/correlation.py` — `CorrelationAnalyzer`:
  - Pearson + Spearman with subgroup filtering
  - **NEW: `compute_per_scenario_correlations()` + `compute_per_scenario_summary()`** — Fisher z-transform for proper averaging
  - Full correlation matrix, action-conditioned attention

### Phase 5: Temporal Analysis ✅ COMPLETE
- [x] **5.1** `reward_attention/temporal.py` — `TemporalAnalyzer` implemented

### Phase 6: Visualization ✅ COMPLETE
- [x] **6.1** `reward_attention/visualization.py` — all 4 figure types implemented

### Phase 7: Main Experiment Script ✅ COMPLETE
- [x] **7.1** `reward_attention/run_experiment.py` — full pipeline with per-scenario summary

### Phase 8: Testing & Validation (IN PROGRESS)
- [x] **8.1** 1-scenario test: pipeline runs end-to-end, 80 records, all figures generated
- [x] **8.2** Attention non-uniform: confirmed (std=0.008-0.035 across tokens)
- [x] **8.3** Risk metrics vary: collision_risk varies 0→1 across episode timesteps
- [ ] **8.4** 5-scenario run: IN PROGRESS (running in background, ~400s/scenario)
- [ ] **8.5** Run on 50 scenarios — pending background run
- [x] **8.6** Figures look publication-ready (heatmap, scatter, bar chart all generated)

---

## 7. Hypothesized Correlations (Test These)

| Risk Metric | Attention Category | Expected Direction | Rationale |
|-------------|-------------------|-------------------|-----------|
| `collision_risk` | `attn_trajectory` | **positive** | Higher TTC risk → more agent attention |
| `collision_risk` | `attn_to_threat` | **positive** | Model should look at the threatening agent |
| `safety_risk` | `attn_trajectory` | **positive** | Safety risk driven by vehicle proximity |
| `navigation_risk` | `attn_gps_path` | **positive** | Off-route → more path attention |
| `behavior_risk` | `attn_roadgraph` | **unclear** | Comfort-related, may not localize |
| `collision_risk` | `attn_roadgraph` | **negative** | When focused on agents, less road attention |

---

## 8. Key Design Decisions Already Made

1. **Spearman correlation** (not Pearson) — safer for normalized coordinate data
2. **Token-level aggregation** — sum attention weights per token category, normalize to [0,1]
3. **Average across queries** — average the (16, N_tokens) matrix across the 16 query dimension before category aggregation, giving (N_tokens,) importance per token
4. **Reuse VMaxAdapter** — don't rewrite the episode loop; extend it to also call `get_attention()`
5. **Save raw TimestepRecords to disk** — enables re-analysis without GPU rerunning
6. **One model per process** — run one complete experiment, save results, then scale

---

## 9. Files Already Existing (Do Not Recreate)

```
posthoc_xai/models/perceiver_wrapper.py  — PerceiverWrapper with get_attention()
posthoc_xai/models/loader.py             — load_vmax_model() with all 5 fixes
event_mining/integration/vmax_adapter.py — VMaxAdapter (episode rollout + state)
event_mining/metrics.py                  — compute_ttc(), compute_distances()
events/test_catalog.json                 — existing event catalog (152+ events)
posthoc_xai/visualization/heatmaps.py   — temporal plot functions (reusable)
```

---

## 10. Run Commands

```bash
# Activate environment
eval "$(~/anaconda3/bin/conda shell.bash hook)" && conda activate vmax

# Set PYTHONPATH
export PYTHONPATH=/home/med1e/post-hoc-xai/V-Max:$PYTHONPATH

# Working directory
cd /home/med1e/post-hoc-xai

# Phase 0: probe (run first, always)
python reward_attention/probe.py

# Phase 7: main experiment
python reward_attention/run_experiment.py \
    --model runs_rlc/womd_sac_road_perceiver_complete_42 \
    --data data/training.tfrecord \
    --n-scenarios 50 \
    --output results/reward_attention
```

---

## 11. Session Log

| Date | Work Done | Next Step |
|------|-----------|-----------|
| 2026-02-19 | Context analysis, architecture design, this checkpoint file created | Start Phase 0: probe.py |
| 2026-02-19 | Phases 0-7 fully implemented. Pipeline tested on 1-scenario (80 steps). 5-scenario run in background. | Wait for background run, then launch 50-scenario overnight |

---

## 12. Results (5 scenarios, 393 timesteps)

### Key publication result: Within-episode correlations (high-variation scenarios)
*Filtered to episodes where collision_risk std > 0.2 (episodes with meaningful risk variation)*

| Risk × Attention | mean_ρ | 95% CI | sig in | Expected |
|-----------------|--------|--------|--------|----------|
| collision_risk × attn_agents | **+0.701** | [+0.580, +0.792] | 100% | ✓ positive |
| collision_risk × attn_to_threat | **+0.486** | [+0.265, +0.659] | 100% | ✓ positive |
| collision_risk × attn_roadgraph | **-0.535** | [-0.553, -0.516] | 100% | ✓ negative |
| safety_risk × attn_agents | **+0.701** | [+0.580, +0.792] | 100% | ✓ positive |

### Pooled correlations (all 5 scenarios, n=393, between-scenario confounds present)

| Risk Metric | Ego (SDC) | Other Agents | Road Graph | Traffic Lights | GPS Path |
|-------------|-----------|-------------|-----------|----------------|----------|
| Collision Risk | **+0.52** | +0.02 | **-0.48** | **-0.52** | +0.37 |
| Behavior Risk | **+0.47** | -0.14 | -0.34 | -0.31 | **+0.47** |

*Between-scenario note: collision_risk×attn_agents is ρ=+0.02 pooled because 2 scenarios have constant very high risk. Within each episode, the correlation is ρ=+0.70.*

### Action-conditioned attention (n=393)
| Action Type | n | Agents | Road Graph | GPS Path | Ego (SDC) |
|------------|---|--------|-----------|----------|-----------|
| Braking | 121 | **0.096** | 0.419 | 0.210 | 0.249 |
| Steering | 47 | **0.098** | 0.510 | 0.160 | 0.202 |
| Neutral | 225 | 0.066 | 0.494 | 0.196 | 0.220 |

*Active actions → 30-40% more agent attention than neutral driving*

### Unexpected finding
- **collision_risk × attn_sdc = +0.52** — model attends to its own ego trajectory MORE when at risk (self-monitoring for emergency maneuver planning)
- **collision_risk × attn_lights = -0.52** — when at risk, much less traffic light attention (attentional competition: safety > traffic rules in danger)
- **behavior_risk × attn_gps = +0.47** — hard braking/acceleration → more GPS path attention (looking ahead at route when taking aggressive action)

### Performance note
~400s/scenario (VMaxAdapter Python episode loop is the bottleneck, not model inference). For 50 scenarios, plan an overnight run (~6 hours).

## 13. Files Created / Modified

```
reward_attention/
├── __init__.py
├── probe.py                    # Phase 0 probe (run to verify)
├── config.py                   # TOKEN_RANGES (280 tokens), AnalysisConfig, TimestepRecord
├── extractor.py                # AttentionTimestepCollector (batched inference)
├── risk_metrics.py             # RiskComputer → collision/safety/navigation/behavior risk
├── correlation.py              # CorrelationAnalyzer (per-scenario + pooled + Fisher z)
├── temporal.py                 # TemporalAnalyzer (event catalog window trajectories)
├── visualization.py            # All 4 figure types (scatter, heatmap, temporal, bar)
└── run_experiment.py           # Main: --n-scenarios N --quick

posthoc_xai/models/perceiver_wrapper.py  # MODIFIED:
  - _extract_attention(): real softmax(Q@K^T/sqrt(d)) from Dense_0/Dense_1
  - NEW: observation_structure_detailed property
```

---

*This file is the single source of truth for implementation state. Update the session log and check off tasks as work progresses.*
