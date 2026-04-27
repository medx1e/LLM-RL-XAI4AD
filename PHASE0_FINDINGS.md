# Phase 0 Audit Findings

> Completed: 2026-04-24
> Purpose: Ground truth on what is cached, what is missing, and what is
> technically feasible before writing any Phase 1/2/3 code.

---

## 1. Cached Data Inventory

### 1a. event_xai_results (3 JSON files)

Location: `post-hoc-xai/experiments/event_xai_results/`

| File | Event type | Timesteps | Methods |
|------|-----------|-----------|---------|
| event_00_s001.json | evasive_steering (MEDIUM) | ~14 | VG, IG |
| event_01_s000.json | evasive_steering (LOW) | ~12 | VG, IG |
| event_02_s000.json | hazard_onset (CRITICAL) | 15 | VG, IG |

**Format:** Each JSON has `timesteps` (list), `event` (metadata), and `methods`
(dict keyed by method name). Each method has `category_series` (dict: category →
list of floats, one per timestep) and `entity_series` (dict: agent_id → list).

**Critical:** NO attention data in any of these files. Only VG + IG attribution.

---

### 1b. reward_attention timestep_data.pkl (3 model pkl files)

Location: `post-hoc-xai/results/reward_attention/`

| Model | Timesteps | Scenarios |
|-------|-----------|-----------|
| perceiver_complete_42 | 3,676 | 50 |
| perceiver_minimal_42 | 3,718 | 50 |
| perceiver_basic_42 | 157 | ~3 |

**Format:** List of `TimestepData` objects. Fields per timestep:
- Aggregated attention: `attn_sdc`, `attn_agents`, `attn_roadgraph`, `attn_lights`, `attn_gps`
- Per-agent attention: `attn_per_agent` (list[8]), `attn_to_nearest`, `attn_to_threat`
- Risk: `collision_risk`, `safety_risk`, `navigation_risk`, `behavior_risk`, `min_ttc`
- Action/state: `accel`, `steering`, `ego_speed`, `num_valid_agents`
- Labels: `is_collision_step`, `is_offroad_step`

**Critical:** NO gradient attribution (IG, VG, etc.) in these files. Attention only.

---

### 1c. scenario002_all_methods

Location: `post-hoc-xai/experiments/scenario002_all_methods/`

A single-timestep (t=35, scenario s002) analysis with all 7 methods. Has `summary_metrics.csv`
with category importances + faithfulness metrics (deletion_auc, insertion_auc) + sparsity (gini, entropy).

**Critical:** Single timestep only. No attention data. Only useful as reference for
the "3 camps" method-divergence finding.

---

### 1d. platform_cache (KEY FINDING)

Location: `platform_cache/`

| Model slug | Scenarios | Attention | Attribution methods |
|---|---|---|---|
| SAC_Perceiver_Complete_WOMD_seed_42 | 3 (idx 0001–0003) | ✓ | IG, GxI, FeatureAblation |
| SAC_Perceiver_WOMD_seed_42 (=minimal) | 3 (idx 0001–0003) | ✓ | IG, GxI, FeatureAblation |

**Format per scenario:**
- `scenario_{idx:04d}_artifact.pkl` → PlatformScenarioArtifact (includes raw_observations)
- `scenario_{idx:04d}_attention.pkl` → list[dict] of length T (full episode)
  - Each dict has keys: `cross_attn_layer_0`, `cross_attn_layer_1`, `cross_attn_layer_2`,
    `cross_attn_layer_3`, `cross_attn_avg` — all shape `(1, 16, 280)` (batch, queries, tokens)
- `scenario_{idx:04d}_attr_{method}.pkl` → list[Attribution] of length T (full episode)
  - Each Attribution has `raw` (1655,), `normalized` (1655,), `category_importance` (dict),
    `entity_importance` (dict), `method_name`, etc.
- `scenario_{idx:04d}_frames.pkl` → list[np.ndarray] BEV frames

**THIS IS THE ONLY EXISTING DATASET WHERE BOTH ATTENTION AND ATTRIBUTION
ARE CACHED AT THE SAME TIMESTEPS.**

Both lists have length T (full episode), so every timestep has:
- Attention: `cross_attn_avg` shape (1, 16, 280) → aggregatable to 5 categories
- IG attribution: `category_importance` dict → already 5-dim vector

This is what Phase 2 (pilot) will use. No model loading required.

---

## 2. Data Overlap Summary

| Dataset | Has attention | Has attribution | Overlap at same timesteps? |
|---|---|---|---|
| event_xai (3 events, ~41 timesteps) | ✗ | ✓ (VG, IG) | No |
| reward_attention (3 models, ~7,500 timesteps) | ✓ | ✗ | No |
| scenario002_all_methods (1 timestep) | ✗ | ✓ (7 methods) | No |
| **platform_cache (2 models, 3 scenarios each)** | **✓** | **✓ (IG, GxI, FA)** | **YES — full episode T** |

**Immediate zero-compute pilot:** platform_cache (Phase 2). No model loading needed.

**Full-scale study (Phase 3):** Requires reloading models and running IG on the
3,700 reward_attention timesteps (those have risk data for stratification).

---

## 3. PerceiverWrapper Attention Extraction — Verified

### What is currently extracted (cross-attention only)

The `_extract_attention()` method reconstructs softmax attention by capturing
Q and K Dense projections from `capture_intermediates` and recomputing:
`softmax(Q @ K^T / sqrt(head_dim))`.

Returns per forward pass:
- `cross_attn_layer_0` … `cross_attn_layer_3`: per-layer cross-attention, shape `(batch, 16, 280)`, averaged over 2 heads
- `cross_attn_avg`: mean across all 4 layers, shape `(batch, 16, 280)`

Current aggregation to 5 categories (done in reward_attention extractor): mean over 16 queries,
then sum over token ranges per category. This is the **query-mean aggregation** — the one flagged
as potentially destroying query specialization.

### Self-attention — not extracted, but feasible

**Architecture (confirmed from config.yaml):**
- `tie_layer_weights=True` for all models we use
- `num_latents=16`, `encoder_depth=4`
- `cross_num_heads=2`, `cross_head_features=16` → cross-attn: (batch, 16, 280) per layer
- `latent_num_heads=2`, `latent_head_features=16` → self-attn: (batch, 16, 16) per layer

**Where self-attention lives in the intermediates tree:**
```
intermediates
  └── encoder_layer
        └── lq_attention
              ├── cross_attn          ← currently extracted
              │     ├── Dense_0.__call__  [list of 4 tensors: Q per layer]
              │     └── Dense_1.__call__  [list of 4 tensors: K per layer]
              └── self_attn           ← NOT yet extracted
                    ├── Dense_0.__call__  [list of 4 tensors: Q per layer, shape (B,16,32)]
                    └── Dense_1.__call__  [list of 4 tensors: K per layer, shape (B,16,32)]
```

**Attention rollout feasibility:** FEASIBLE. Extracting self-attention requires ~20 lines
added to `_extract_attention()`. The rollout computation itself (Abnar & Zuidema 2020) is
a chain of matrix multiplications, ~10 lines.

Rollout formula (with residual correction):
```
For each layer l: A_eff[l] = 0.5 * I + 0.5 * self_attn[l]   # (16, 16)
Rollout = A_eff[3] @ A_eff[2] @ A_eff[1] @ A_eff[0]          # (16, 16)
final_attention = Rollout @ cross_attn_layer_0                 # (16, 280)
```

---

## 4. Decisions for the Implementation Plan

### Phase 2 pilot: USE platform_cache
- 2 models × 3 scenarios × T timesteps = immediate overlap dataset
- Access: load pkl directly with `pickle.load` (uses JAX arrays, needs vmax env)
- Risk metric: platform_cache artifacts include raw_observations but NOT precomputed
  risk scores. For a simple pilot, use timestep index as a proxy (or compute
  a simple TTC-proxy from raw_obs). Full risk-stratification requires Phase 3.

### Phase 3 scale-up: RERUN IG on reward_attention timesteps
- The reward_attention pkl has risk scores already. Add IG computation at each
  timestep during a second pass. Saves results back alongside the existing pkl.
- Approach: new script `experiments/compute_ig_for_ra_timesteps.py` that loads
  each model, iterates the 3,700 cached timesteps, runs IG, writes a companion
  pkl `timestep_ig_data.pkl` alongside the existing `timestep_data.pkl`.

### Attention aggregation: start with current mean, add alternatives in Phase 1
- Phase 1b adds max-pool and attention-weighted aggregation as alternatives
- Compare all three on the platform_cache pilot before committing to one for Phase 3

### Attention rollout: INCLUDE in Phase 1c
- Self-attention IS capturable (confirmed from architecture)
- Requires adding ~30 lines to `_extract_attention()` + rerunning precompute
  on platform_cache scenarios (3 scenarios × 2 models — quick, ~2 hrs total)
- Decision: implement in Phase 1c, re-cache platform_cache attention files to
  include rolled-out attention as an additional key `cross_attn_rollout`

---

## 5. Token Structure Reference (confirmed)

The 280 input tokens to the cross-attention are:
| Category | Tokens | Range |
|---|---|---|
| sdc_trajectory | 5 (1 entity × 5 timesteps) | [0, 5) |
| other_agents | 40 (8 entities × 5 timesteps) | [5, 45) |
| roadgraph | 200 (200 points) | [45, 245) |
| traffic_lights | 25 (5 lights × 5 timesteps) | [245, 270) |
| gps_path | 10 (10 waypoints) | [270, 280) |
| **Total** | **280** | |

Note: this is the token-level structure (280 tokens) vs the observation-level
structure (1,655 flat features). Attribution methods operate on the 1,655-dim
observation; attention operates on the 280 encoded tokens. Size-corrected
normalization for attribution uses the 1,655-dim feature counts (1000 for
roadgraph, 20 for gps). Size-corrected normalization for attention uses the
280-dim token counts (200 for roadgraph, 10 for gps).

---

## 6. Go/No-Go on Rollout

**GO.** Self-attention is capturable from `lq_attention > self_attn` in the intermediates
tree (identical structure to cross_attn, confirmed from `lq.py` and `attention_utils.py`).
`tie_layer_weights=True` means a single `self_attn` module is called 4 times, so intermediates
will capture a list of 4 Q/K tensors — exactly the same pattern as cross_attn extraction.

Rollout will be implemented in Phase 1c alongside the other aggregation alternatives.
