# Context for Claude — CBM Project

*Read this before doing anything. It contains the full state of the project as of 2026-04-26.*

---

## What This Project Is

A **Hard Concept Bottleneck Model (CBM)** injected into a Soft Actor-Critic (SAC) agent for autonomous driving. Built on top of V-Max (Google RL framework) + Waymax simulator (WOMD dataset). The goal: instead of a black-box policy, force the agent to reason through 15 human-interpretable concepts before acting.

Architecture:
```
obs (1655-d) → [LQ Encoder] → z (128-d) → [Concept Head] → c (15-d) → [Actor/Critic] → action
```
The actor and critic see ONLY `c`, never `z`. Hard bottleneck — provable interpretability.

**Working directory:** `/home/med1e/platform_fyp/cbm/`
**Platform directory:** `/home/med1e/platform_fyp/`
**Environment:** always `conda activate vmax` before running anything

---

## Directory Structure (key files only)

```
cbm/
├── cbm_v1/
│   ├── train_cbm.py              ← training launcher
│   ├── cbm_trainer.py            ← training loop
│   ├── cbm_sac_factory.py        ← network + loss + SGD step
│   ├── config.py                 ← CBMConfig (auto-derives concept indices)
│   ├── networks.py               ← CBMPolicyNetwork, ConceptHead
│   ├── concept_loss.py           ← BCE + Huber masked loss
│   ├── eval_cbm_v2.py            ← evaluation script (use this, not eval_cbm.py)
│   ├── config_womd_scratch.yaml  ← cluster scratch config (15 concepts, λ=0.5)
│   ├── config_womd_scratch_short.yaml  ← local smoke test config
│   ├── config_womd_frozen_v2_150gb.yaml ← cluster frozen V2 config
│   └── smoke_test_v2.py          ← 30-check verification (15 concepts)
│
├── concepts/
│   ├── registry.py               ← OrderedDict of all 15 concepts + extract_all_concepts()
│   ├── extractors.py             ← one function per concept (pure JAX)
│   ├── adapters.py               ← ONLY file touching V-Max internals
│   ├── geometry.py               ← menger_curvature + other helpers
│   └── schema.py                 ← ConceptSchema, ConceptType enum
│
├── data/training.tfrecord        ← local ~950MB WOMD subset
│
├── cbm_scratch_v2_lambda05/      ← lambda ablation winner (10GB, 15 concepts)
│   └── checkpoints/model_final.pkl
│
├── eval_model_final.json         ← 400-scenario validation eval results (from cluster)
├── eval_model_final_cache/       ← per-step arrays for 400 val scenarios
│   ├── pred_concepts.npy         (80, 400, 15)
│   ├── true_concepts.npy
│   ├── valid_mask.npy
│   ├── ego_actions.npy
│   ├── rewards.npy
│   ├── dones.npy
│   ├── driving_metrics.npy
│   └── concept_names.npy
│
├── curated_scenarios.json        ← 36 top scenarios per archetype (from find_demo_scenarios.py)
├── curated_scenarios_data.npz    ← concept/action arrays for those 36 scenarios
├── figures/                      ← generated thesis figures (PDF + PNG)
├── find_demo_scenarios.py        ← scenario curation script
├── generate_figures.py           ← thesis figure generation (fig1–fig4)
│
├── cbm-detailed-context-for-thesis-writing.md  ← FULL thesis context + writing prompt
├── to-do-tasks-for-cbm.md        ← structured task list
└── train_scratch_v2.md           ← cluster training guide
```

---

## Concept Set (15 concepts, 3 phases)

| # | Name | Type | Phase |
|---|---|---|---|
| 0 | ego_speed | continuous | 1 |
| 1 | ego_acceleration | continuous | 1 |
| 2 | dist_nearest_object | continuous | 1 |
| 3 | num_objects_within_10m | continuous | 1 |
| 4 | traffic_light_red | **binary** | 1 |
| 5 | dist_to_traffic_light | continuous | 1 |
| 6 | heading_deviation | continuous | 1 |
| 7 | progress_along_route | continuous | 1 |
| 8 | ttc_lead_vehicle | continuous | 2 |
| 9 | lead_vehicle_decelerating | **binary** | 2 |
| 10 | at_intersection | **binary** | 2 |
| 11 | path_curvature_max | continuous | 3 |
| 12 | path_net_heading_change | continuous | 3 |
| 13 | path_straightness | continuous | 3 |
| 14 | heading_to_path_end | continuous | 3 |

V1 = phases (1,2) = 11 concepts. V2 = phases (1,2,3) = 15 concepts.

---

## Training Paradigms

| Paradigm | Encoder | Key property |
|---|---|---|
| SAC Baseline | Pretrained, trainable | Black box, best performance (~97.5% route progress) |
| **CBM Frozen** | Pretrained, **frozen** | Concept head reads fixed RL-optimal representations |
| CBM Joint | Pretrained, trainable | Risk of concept collapse (Goodhart's Law) — failed |
| **CBM Scratch** | Random init, trainable | Encoder shaped entirely by CBM + RL — main contribution |

---

## Trained Models Status

| Model | Paradigm | Concepts | Data | Status |
|---|---|---|---|---|
| `runs_rlc/womd_sac_road_perceiver_minimal_42` | SAC Baseline | 0 | 150GB | Done ✓ |
| `cbm_model/` | Frozen V1 | 11 | ~10GB | Done ✓ |
| `cbm_model_joint/` | Joint V1 | 11 | ~10GB | Done ✓ |
| `cbm_v1_frozen_150GB/` | Frozen V1 | 11 | 150GB | Done ✓ |
| `cbm_scratch/` | Scratch (11c, bug run) | 11 | 150GB | Done ✓ (old, ignore) |
| `cbm_v2_frozen_womd_150gb/` | Frozen V2 | 15 | 150GB | Done ✓ |
| `cbm_scratch_v2_lambda05/` | **Scratch V2, λ=0.5** | 15 | 10GB | **Done ✓ — primary result** |
| `runs_cbm/cbm_scratch_v2_150gb_42/` | Scratch V2, λ=0.5 | 15 | 150GB | **Training on cluster** |
| SAC Scratch baseline | SAC, no CBM | 0 | 10GB | **Training on cluster** |

---

## Key Results So Far (primary model: Scratch V2, 10GB, λ=0.5, 400 val scenarios)

**Binary concept accuracy (validation):**
- traffic_light_red: 99.94%
- at_intersection: 99.12%
- lead_vehicle_decelerating: 98.14%

**Continuous concept quality (validation):**
- Best: dist_to_traffic_light R²=0.914, path_straightness R²=0.629
- Weak: heading_deviation R²=-1.374, dist_nearest_object R²=-0.035 (needs 150GB)

**Task performance (400 val scenarios):**
- Route progress: 0.876 (vs SAC baseline 0.975)
- At-fault collision: 0.080
- **Run red light: 0.000** ← key finding, causally tied to 99.94% TL accuracy
- Offroad: 0.015

---

## Lambda Ablation Decision

Ran 3 short runs (10GB, 1M steps): λ ∈ {0.01, 0.1, 0.5}

- λ=0.01: eliminated — TL red accuracy 76%, at_intersection 70% → safety failure
- λ=0.1 vs λ=0.5: equivalent task performance, λ=0.5 wins on all concept quality metrics
- **Decision: λ=0.5** for 150GB run

---

## Bugs Fixed (important to know)

**Bug 1 (critical, fixed):** `concept_phases` was never passed through the training pipeline.
- `train_cbm.py` read `num_concepts` from YAML but not `concept_phases`
- `CBMConfig` defaulted to `concept_phases=(1,2)` regardless of YAML
- `extract_all_concepts()` used default phases=(1,2,3) → 15-d targets but 11-concept loss
- **Fix:** `train_cbm.py` line 233, `cbm_trainer.py` lines 120 + 244 + 252
- This is why the old `cbm_scratch/` model has only 11 concepts despite intending 15

---

## Diagnostic Metrics Added to Training (cbm_sac_factory.py)

These now appear in TensorBoard every logged iteration:
- `train/concept_loss/<name>` — per-concept loss
- `train/concept_mean/<name>` — mean predicted value (dead neuron check)
- `train/concept_std/<name>` — std predicted value (→0 = dead neuron)
- `train/concept_accuracy/<name>` — binary accuracy for the 3 binary concepts
- `train/concept_task_grad_ratio` — concept_head update norm / actor_fc update norm

---

## Evaluation Script

Use `cbm_v1/eval_cbm_v2.py` (NOT `eval_cbm.py`).

**For frozen/joint models (have pretrained_dir):**
```bash
python cbm_v1/eval_cbm_v2.py \
    --checkpoint <path>/model_final.pkl \
    --pretrained_dir runs_rlc/womd_sac_road_perceiver_minimal_42 \
    --data data/validation.tfrecord \
    --num_scenarios 400 --mode frozen --num_concepts 11 --concept_phases 1 2
```

**For scratch models (no pretrained_dir):**
```bash
python cbm_v1/eval_cbm_v2.py \
    --checkpoint cbm_scratch_v2_lambda05/checkpoints/model_final.pkl \
    --config cbm_v1/config_womd_scratch.yaml \
    --data data/validation.tfrecord \
    --num_scenarios 400 --mode scratch --num_concepts 15 --concept_phases 1 2 3
```

Outputs: `eval_model_final.json` + `eval_model_final_cache/` (per-step arrays, 80×N×15).

See `eval_cbms_guide.md` for exact commands for every model.

---

## Figure Generation

```bash
python generate_figures.py
# Outputs: figures/fig1_concept_quality.pdf/png
#          figures/fig4_concept_temporal.pdf/png
# (fig2 was dropped — not useful. fig3 was dropped — better as table.)
```

Run after collecting all eval results. The script reads `eval_model_final_cache/` and `eval_model_final.json`.

---

## Scenario Curation (demo scenarios for platform + defense)

```bash
python find_demo_scenarios.py
# Reads: eval_model_final_cache/ (must exist)
# Writes: curated_scenarios.json (ranked top-10 per archetype)
#         curated_scenarios_data.npz (concept/action arrays for 36 top scenarios)
```

**4 archetypes:** red_light_stop, ttc_success, curvature_nav, concept_failure

**Top scenario indices (local_idx in the 400-scenario batch):**
- red_light_stop rank 1: local_idx=246, score=0.949, progress=1.0 ✓
- ttc_success rank 1: local_idx=327, score=0.876, progress=0.844 ✓
- curvature_nav rank 1: local_idx=244, score=0.978, progress=1.0 ✓
- concept_failure rank 1: local_idx=142, score=0.842, progress=0.546 ✗

---

## Platform (Streamlit)

**Location:** `/home/med1e/platform_fyp/`
**Run:** `cd /home/med1e/platform_fyp && streamlit run app.py`

**Tabs:**
- Home
- Post-hoc XAI (existing, SAC models with attention + attribution)
- **CBM Explorer** (new tab added in `platform/tabs/tab_cbm.py`)

**CBM Explorer status:**
- Without precomputed BEV artifacts: shows static figures (fig1, fig4) + warning message
- With precomputed artifacts: shows BEV player + concept timeline synced to slider

**Precompute script** (needs validation tfrecord + validation artifacts on disk):
```bash
conda activate vmax
cd /home/med1e/platform_fyp
python scripts/precompute_cbm_demo.py \
    --data cbm/data/validation.tfrecord \
    --top_k 3
```
- Processes top-3 per archetype = ~12 unique scenarios
- Saves to `platform_cache/CBM_Scratch_V2/`
- BEV frames have "★ Key event" annotation baked in at interesting timesteps
- Concept timeline shows shaded bands at same steps

**CBM model registered in:** `platform/shared/model_catalog.py` as `"CBM Scratch V2 — λ=0.5"`

---

## Pending Tasks (priority order)

1. **Wait for 150GB scratch V2 training to finish** → run eval on validation set
2. **Wait for SAC Scratch baseline to finish** → run V-Max eval for comparison
3. **Collect validation tfrecord locally** → run eval for all remaining models + precompute platform
4. **Fill in `[PENDING]` rows in thesis results table** (in `cbm-detailed-context-for-thesis-writing.md`)
5. **Regenerate figures** once all model evals are done
6. **Write thesis CBM chapter** using `cbm-detailed-context-for-thesis-writing.md` + writing prompt

---

## Key Documents

| File | Purpose |
|---|---|
| `cbm-detailed-context-for-thesis-writing.md` | Full thesis chapter context + writing prompt at end |
| `to-do-tasks-for-cbm.md` | Structured task list with phases |
| `eval_cbms_guide.md` | Exact eval commands for every model |
| `train_scratch_v2.md` | Cluster training guide |
| `guide_for_cbm_platform.md` | Platform CBM tab implementation guide |

---

## Publishability Assessment

**Target: ITSC 2026** (deadline typically June–July). Realistic if 150GB results arrive within 2–3 weeks.

**Contributions that hold regardless of 150GB results:**
1. Hard-bottleneck CBM in SAC loop for AD — novel architecture
2. Annotation-free concept supervision from simulator state — scalable, observation-faithful
3. From-scratch CBM training (zero red lights from random init on 10GB)
4. The zero red light finding — causally provable, not just correlational

**What 150GB results will add:**
- Quantify interpretability tax (CBM vs SAC-from-scratch, apples-to-apples)
- Demonstrate scaling: does concept quality improve with more data?
- Close the performance gap with pretrained SAC baseline
