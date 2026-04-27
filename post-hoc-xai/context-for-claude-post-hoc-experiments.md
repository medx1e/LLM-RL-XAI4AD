# Context for Claude — Post-Hoc XAI Attention Study

> Hand this file to a new Claude instance to continue working on this codebase.
> Last updated: 2026-04-25

---

## 1. What This Project Is

**Final-year project** on explainable RL for autonomous driving. The main research
track covered here is **post-hoc XAI** applied to V-MAX models (JAX/Flax RL agents
trained on Waymo Open Motion Dataset). The platform is a Streamlit showcase app
(separate track, mostly done).

**Core research question being answered here:**
> Does Perceiver cross-attention agree with gradient-based feature importance,
> and is agreement higher during safety-critical moments?
> (Cross-domain test of Jain & Wallace 2019 "Attention is not Explanation")

---

## 2. Codebase Location & Structure

```
/home/med1e/platform_fyp/
├── post-hoc-xai/                  ← main research code (THIS repo)
│   ├── posthoc_xai/               ← XAI framework package
│   │   ├── methods/               ← VG, IG, SmoothGrad, GxI, Perturbation, FeatureAblation, SARFA
│   │   ├── models/                ← PerceiverWrapper, GenericWrapper, loader
│   │   ├── utils/
│   │   │   ├── normalization.py   ← size_correct_attribution, size_correct_attention
│   │   │   ├── attention_aggregation.py  ← aggregate_attention (mean/maxpool/entropy/rollout)
│   │   │   └── ig_baseline.py     ← compute_baseline, BaselineAccumulator
│   │   └── visualization/
│   │       └── paper_figures.py   ← ALL thesis/paper figures (PDF, 300 DPI, reusable)
│   ├── reward_attention/          ← reward-conditioned attention study (paper SUBMITTED)
│   ├── experiments/
│   │   ├── phase1a_size_correction.py
│   │   ├── phase1b_aggregation_comparison.py
│   │   ├── phase1c_rollout_comparison.py
│   │   ├── phase2_correlation_pilot.py    ← 3-scenario pilot
│   │   ├── phase3_scale_correlation.py    ← MAIN experiment (50 scenarios)
│   │   └── phase3_cluster.py              ← cluster entry point (calls phase3 with SARFA)
│   ├── THESIS_NOTES.md            ← living document, thesis writing guidance
│   ├── PHASE0_FINDINGS.md         ← data audit results
│   ├── QUERY_SPECIALIZATION_NOTES.md
│   └── context-for-claude-post-hoc-experiments.md  ← THIS FILE
└── cbm/
    ├── runs_rlc/                  ← 35+ pretrained V-MAX model weights
    ├── data/training.tfrecord     ← Waymo dataset (~1GB)
    └── V-Max/                     ← V-Max simulator (JAX/Waymax)
```

---

## 3. Models

**Architecture:** Perceiver/LQ encoder with SAC policy.
- `tie_layer_weights=True`, `num_latents=16`, `encoder_depth=4`
- `cross_num_heads=2`, `cross_head_features=16`
- **16 learned query tokens** cross-attend to **280 input tokens**

**Key models used in experiments:**
- `womd_sac_road_perceiver_complete_42` — complete reward config (TTC penalty included)
- `womd_sac_road_perceiver_minimal_42` — minimal reward config (no TTC penalty)

**Broken models (do not use):** `sac_seed0/42/69` — speed_limit feature missing.

**Observation:** 1,655 flat features (ego-relative, normalized ≈ [-1, 1]):
| Category | Features | Tokens (attention space) |
|---|---|---|
| sdc_trajectory | 40 | 5 |
| other_agents | 320 | 40 |
| roadgraph | 1,000 | 200 |
| traffic_lights | 275 | 25 |
| gps_path | 20 | 10 |
| **Total** | **1,655** | **280** |

**Action space:** 2D continuous: [acceleration, steering], SAC tanh output ≈ [-1, 1].

---

## 4. Environment

```bash
conda activate vmax   # Python 3.10, JAX 0.5.3, GTX 1660 Ti (4603 MB)

# Key paths:
_ROOT = /home/med1e/platform_fyp/post-hoc-xai
_CBM  = /home/med1e/platform_fyp/cbm
DATA_PATH = cbm/data/training.tfrecord
```

**One model per process** — Waymax metric registry constraint. Never load two
models in the same Python process.

---

## 5. What Was Built (Phase Summary)

### Phase 0 — Data Audit
- Mapped all cached data across experiments
- **Key finding:** `platform_cache/` is the only place with BOTH attention AND
  attribution at the same timesteps (2 models × 3 scenarios × 80 ts)
- Confirmed `tie_layer_weights=True` → self-attention extractable

### Phase 1a — Size-Corrected Normalization
- **File:** `posthoc_xai/utils/normalization.py`
- `size_correct_attribution()` — divides by feature count before renormalizing
- **Finding:** Roadgraph drops from 55% to 8% after correction; GPS jumps to 51%
- **Decision:** Use original total attribution as primary; size-corrected as
  methodological note in thesis

### Phase 1b — Attention Aggregation Alternatives
- **File:** `posthoc_xai/utils/attention_aggregation.py`
- Three strategies: `mean`, `maxpool`, `entropy` (+ `rollout` added in 1c)
- **Results:** entropy ≈ mean (MAD=0.011), maxpool differs more (MAD=0.055)
- **Query specialization finding:** Mean entropy = 4.6/8.13 bits — queries are
  NOT diffuse. Minimal model query 1: entropy=1.62 bits (20% of max) — very sharp.
  Up to 13/16 queries below 60% max entropy in minimal model.
- **Decision:** mean-pool as primary (consistent with reward_attention paper)

### Phase 1c — Attention Rollout
- **Modified:** `posthoc_xai/models/perceiver_wrapper.py` — `_extract_attention()`
- Now returns `self_attn_layer_{0..3}` (B,16,16) AND `cross_attn_rollout` (B,16,280)
- **Rollout formula:** `A_eff[l] = 0.5*I + 0.5*A_self[l]`; `R = prod(A_eff)`; `rollout = R @ cross_attn_avg`
- **Results:** Global MAD rollout vs raw = 0.033. Traffic lights (0.063) and road (0.047) affected most.
- **Decision:** Use rollout as canonical attention signal for the correlation study

### Phase 2 — Pilot Correlation (3 scenarios, platform_cache)
- **File:** `experiments/phase2_correlation_pilot.py`
- Methods: VG (live), IG zero baseline (cached), GxI (cached), IG mean baseline (live)
- **Metric:** Kendall τ per timestep (5-dim rank correlation) + Pearson ρ per category over time
- **Key finding:** VG agrees with attention more than IG (τ_VG=0.453 vs τ_IG=0.058 for complete)
- **IG baseline fix:** `compute_baseline()` in `ig_baseline.py` — validity-zeroed mean.
  However: IG_zero_vs_IG_mean calibration τ=0.80+ → baseline doesn't matter much for IG
- **Root cause of IG divergence:** In traffic-light/braking scenarios, IG and VG themselves
  disagree (τ_IG_VG=0.27), and attention sides with VG. Not a baseline artifact.

### Phase 3 — Scale Correlation (50 scenarios)
- **File:** `experiments/phase3_scale_correlation.py`
- **Config at top of script** — easy to tweak (N_SCENARIOS, METHODS, OBS_CHUNK_SIZE)
- **CLI:** `--model`, `--n-scenarios`, `--methods vg ig sarfa`, `--chunk-size`, `--figures-only`
- **Resume-friendly:** per-scenario JSON saved; skips completed scenarios on restart
- **Methods:** VG (batched vmap, fast), IG (JIT-compiled per-timestep, ~5 min/scenario on 6GB),
  SARFA (6 batched forward passes, fast)
- **Output:** `experiments/phase3_results/{model}/` — JSONs + CSVs + figures/

**Results so far (2 scenarios, complete model, VG+IG only):**
```
VG: overall ρ=0.652, calm=0.935, moderate=0.698, high=0.631
IG: overall ρ=0.349, calm=0.912, moderate=0.419, high=0.257
```
Pattern: calm > high (unexpected — both signals converge on road during calm;
during risk, attention reallocates to agents but VG doesn't fully track this).

---

## 6. SARFA — Key Details

**File:** `posthoc_xai/methods/sarfa.py`

**Formula:** `SARFA(f, a) = |Δaction_mean(a)| × (1 - H(|Δaction_mean|/‖Δaction_mean‖) / log|A|)`
- Relevance: how much does zeroing the category change the target action?
- Specificity: how exclusively does it affect the target action vs both?
- With 2 action dims (accel, steering), specificity is coarse but meaningful

**Fast function:** `sarfa_batch(model, raw_obs)` — 6 batched forward passes for all T timesteps:
1 baseline + 5 category perturbations. Import: `from posthoc_xai.methods.sarfa import sarfa_batch`

**Why SARFA should outperform VG:** Both attention and SARFA ask "what specifically
drives the chosen action?" — VG asks "what changes the output generally?"
If SARFA-attention ρ > VG-attention ρ, that's the key publishable result.

---

## 7. IG Baseline

**File:** `posthoc_xai/utils/ig_baseline.py`

Zero baseline = semantically wrong in normalized space. `compute_baseline(obs_array)`:
- Computes mean over provided observations
- Detects validity/mask bits (binary features) via data inspection
- Zeros out all validity bits in the mean → "average road, no agents, no TL"

`BaselineAccumulator` for streaming (large datasets):
```python
acc = BaselineAccumulator()
for batch in data_stream:
    acc.update(batch)
baseline = acc.finalize()
```

**IntegratedGradients** accepts `baseline=np.ndarray` — converts to jnp at init time.

---

## 8. Paper Figures Module

**File:** `posthoc_xai/visualization/paper_figures.py`

Call `set_paper_style()` once at script top. Save with `save_figure(fig, name, dir)` → PDF+PNG.

Functions:
- `plot_risk_stratified_correlation(data_df, methods, ...)` — main Phase 3 figure
- `plot_category_heatmap(data_df, method, ...)` — categories × risk buckets
- `plot_action_conditioned(data_df, methods, ...)` — braking/accel/steering/neutral
- `plot_model_comparison(data_df, models, method, ...)` — complete vs minimal
- `plot_correlation_distribution(data_df, methods, ...)` — violin of per-scenario ρ
- `plot_temporal_series(...)` — attention + attribution over time (qualitative)

---

## 9. Action Thresholds (calibrated from 3,676 reward_attention timesteps)

```python
# accel range ≈ [-1, 1], mean=-0.083, std=0.389
# steering range ≈ [-1, 1], mean=-0.124, std=0.208
# Each bucket ≈ 25% of timesteps
"braking":      accel < -0.3
"accelerating": accel > 0.3
"steering":     |steering| > 0.3 AND |accel| ≤ 0.3
"neutral":      |accel| ≤ 0.3 AND |steering| ≤ 0.3
```

---

## 10. What Still Needs to Be Done

### Immediate (local)
- [ ] Run Phase 3 with VG only on 50 scenarios (fast, ~10 min per model):
  ```bash
  python experiments/phase3_scale_correlation.py --model complete --n-scenarios 50 --methods vg
  python experiments/phase3_scale_correlation.py --model minimal  --n-scenarios 50 --methods vg
  ```
- [ ] If running IG locally: set `N_IG_STEPS = 10` at top of script (~2h total)

### Cluster run (with friend)
- [ ] Run `phase3_cluster.py` with VG+IG+SARFA on 50 scenarios per model
  (see `run_attention_experiments_post_hoc.md` for full instructions)
- [ ] SARFA is the expected publishable result — if SARFA-attention ρ > VG-attention ρ
  on agents, that's the headline

### After cluster results
- [ ] Generate cross-model comparison figure (complete vs minimal)
  → `plot_model_comparison` already in paper_figures.py, needs wiring in a script
- [ ] Write `post_hoc_xai_attention_study.md` with all final numbers
- [ ] Temporal series figure — one qualitative figure showing co-variation during hazard
  → `plot_temporal_series` in paper_figures.py, use event_xai data

### Platform demo (separate track)
- [ ] Run `scripts/precompute_posthoc_demo.py` to validate Streamlit platform end-to-end
- [ ] Streamlit app: `streamlit run app.py`

---

## 11. Key Technical Gotchas

1. **One model per process** — Waymax metric registry can't be re-registered
2. **platform_cache artifact loading** — use `_PlatformUnpickler` (stubs platform.shared,
   bev_visualizer) when loading outside the platform package context
3. **IG is slow on 6GB** — 5 min/scenario with 50 steps. Use `N_IG_STEPS=10` locally or cluster
4. **VMaxAdapter.prepare()** must be called ONCE and reused — `make_adapter(model)` in phase3 script
5. **attention_aggregation module** is loaded via importlib in some scripts (avoids JAX __init__)
6. **Rollout attention key:** `cross_attn_rollout` in attn dict; falls back to `cross_attn_avg`
7. **IG baseline:** zero baseline is semantically wrong but changing it doesn't help much
   (IG_zero_vs_IG_mean calibration τ=0.80+). The real issue is IG vs VG divergence in
   saturated scenarios, not baseline choice.

---

## 12. Key Research Findings So Far

| Finding | Evidence | Status |
|---|---|---|
| VG agrees with attention better than IG overall | Phase 2+3 ρ | Confirmed |
| Attention-attribution agreement highest during calm (not risk) | Phase 3 ρ stratified | Preliminary (2 scenarios) |
| IG and VG diverge in braking/TL scenarios — attention sides with VG | Phase 2 cat-level ρ | Confirmed |
| Baseline choice barely affects IG | IG_zero vs IG_mean τ=0.80+ | Confirmed |
| Road and agents are most consistently correlated categories | Phase 2+3 cat heatmaps | Confirmed |
| Query specialization: minimal model sharper than complete | Phase 1b entropy analysis | Confirmed |
| SARFA expected to outperform VG for attention agreement | Theoretical + paper | Pending cluster run |

---

## 13. The Published Paper Context

**Reward-conditioned attention study** (`reward_attention/`) is **already submitted**.
Its key findings:
- Budget reallocation reversal: complete +38.2%, minimal -16.6% (agents, low→high risk)
- GPS gradient: minimal 33.5% vs complete 16.4% GPS attention (2× difference)
- Within-episode ρ(collision_risk, attn_agents) = +0.291 for complete model
- Vigilance gap: complete model maintains agent surveillance during calm (+134%)

The current study (attention-attribution correlation) is **separate** and extends the
reward-conditioned findings into the XAI domain. The two studies use different
attention signals: reward_attention used `cross_attn_avg` (mean-pool, no rollout);
current study uses `cross_attn_rollout` (rollout-corrected). Note this in thesis.

---

## 14. Thesis Chapter Structure (recommended)

1. Framework: 7 attribution methods + attention extraction
2. Methodological contributions: size correction, rollout, query specialization
3. Method divergence: 3-camp finding (gradient vs occlusion vs SARFA)
4. Temporal attribution arc: detect→attend→commit→execute
5. Attention-attribution correlation study:
   - Pilot (Phase 2): methodology validation
   - Scale (Phase 3): risk-stratified + action-conditioned results
   - Key finding: [depends on cluster SARFA results]
