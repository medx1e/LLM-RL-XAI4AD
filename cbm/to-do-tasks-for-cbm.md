# CBM Project — To-Do Task List

*Priority order: top = most blocking. Update status as tasks complete.*

---

## PHASE 0 — Bug Fix (MUST DO BEFORE ANY NEW TRAINING)

- [ ] **Fix `concept_phases` plumbing bug**
  - `cbm_v1/train_cbm.py`: read `concept_phases` from YAML config, add to `train()` call
  - `cbm_v1/cbm_trainer.py`: add `concept_phases` param to `train()` signature, pass to `CBMConfig`, pass to `extract_all_concepts(inp, phases=concept_phases)` in `concept_targets_fn`
  - **Files:** `cbm_v1/train_cbm.py`, `cbm_v1/cbm_trainer.py`
  - **Expected outcome:** Running with `concept_phases: [1, 2, 3]` now properly supervises all 15 concepts

- [ ] **Smoke test the fix locally (500 steps)**
  - Run: `conda activate vmax && python cbm_v1/train_cbm.py --config cbm_v1/config_womd_frozen_short.yaml`
  - Verify TensorBoard shows 15 per-concept loss entries under `train/concept_loss/`
  - Confirm `concept_loss/path_curvature_max` and other Phase 3 entries are non-zero and decreasing
  - Confirm no `nan` losses
  - Check concept head shape in saved checkpoint: should be 15

---

## PHASE 1 — Scratch V2 Training (15 concepts)

- [ ] **Launch scratch V2 training on cluster (150GB, 15M steps)**
  - Config: `cbm_v1/config_womd_scratch.yaml` (already set to `num_concepts: 15`, `concept_phases: [1,2,3]`)
  - Command: `conda activate vmax && python cbm_v1/train_cbm.py --config cbm_v1/config_womd_scratch.yaml`
  - Monitor: TensorBoard `train/concept_loss`, `metrics/progress_ratio_nuplan`, `metrics/at_fault_collision`
  - Expected output dir: `runs_cbm/cbm_scratch_womd_42/`

- [ ] **Monitor training health at 500k, 1M, 5M steps**
  - Per-concept loss converging (especially Phase 3 concepts 11–14)
  - Policy loss trending negative
  - At-fault collision < 0.01 by 3M steps

---

## PHASE 2 — Evaluation Pipeline

### 2.1 Quantitative Evaluation (all models)

Run eval on the **same held-out WOMD validation split** for all trained models to get comparable numbers.

- [ ] **Define evaluation protocol** (fixed scenario subset, fixed number of episodes)
  - Pick N=100 validation scenarios (or use WOMD standard val split)
  - Run `eval_cbm.py` for each model with identical settings

- [ ] **Evaluate all trained models and log results**

  | Model | Config | Expected checkpoint |
  |---|---|---|
  | SAC Baseline | — | `runs_rlc/womd_sac_road_perceiver_minimal_42/model/model_final.pkl` |
  | CBM Frozen V1 (10GB) | 11 concepts | `cbm_model/checkpoints/model_final.pkl` |
  | CBM Joint V1 (10GB) | 11 concepts | `cbm_model_joint/checkpoints/model_final.pkl` |
  | CBM Frozen V1 (150GB) | 11 concepts | `cbm_v1_frozen_150GB/checkpoints/model_final.pkl` |
  | CBM Scratch V1 (150GB) | 11 concepts | `cbm_scratch/checkpoints/model_final.pkl` |
  | CBM Frozen V2 (150GB) | 15 concepts | `cbm_v2_frozen_womd_150gb/checkpoints/model_final.pkl` |
  | CBM Scratch V2 (150GB) | 15 concepts | pending |

- [ ] **Collect these metrics for each model:**
  - Route progress ratio (nuplan)
  - At-fault collision rate
  - Offroad rate
  - Average episode reward
  - Concept loss at convergence (CBM models only)
  - Per-concept accuracy / R² (CBM models only — see 2.2)

### 2.2 Concept Quality Evaluation

Write a short evaluation script that runs `extract_all_concepts` and `encode_and_predict_concepts` on the validation set and computes:

- [ ] **Binary concepts:** accuracy (% correct), precision, recall for each of (4, 9, 10)
- [ ] **Continuous concepts:** R², MSE, Pearson correlation for each of (0–8, 11–14)
- [ ] **Concept validity rate:** fraction of steps where each concept is valid (to flag rarely-used concepts)

These numbers go into the thesis concept evaluation table.

---

## PHASE 3 — Figures for Thesis

These are the priority figures. Each has a clear story.

### 3.1 Architecture Diagram
- [ ] **CBM Architecture figure** — clean diagram of the pipeline:
  `obs → Encoder → z → Concept Head → c → Actor/Critic`
  With annotation: stop_gradient in frozen mode, concept supervision arrow, hard bottleneck label.
  *Tool: draw.io, Inkscape, or matplotlib. Should be vector format (PDF/SVG).*

### 3.2 Training Curves (from TensorBoard)
- [ ] **Concept loss over training steps** — per-concept curves for best CBM model
  Shows which concepts are learned fast vs slow.
- [ ] **Policy loss comparison** — SAC vs CBM Frozen vs CBM Joint vs CBM Scratch on same axes
  Shows the cost of the bottleneck (performance gap).
- [ ] **Collision rate over training** — all models on same axes
  Shows CBM Scratch achieves near-zero collisions even without pretraining.

### 3.3 Concept Quality Bar Charts
- [ ] **Binary concept accuracy bar chart** — accuracy for TL red, lead deceleration, at_intersection
  Per model (Frozen V1, Joint, Scratch).
- [ ] **Continuous concept R² bar chart** — for all 8 continuous V1 concepts
  Shows which concepts the bottleneck captures well vs poorly.

### 3.4 Performance Comparison Table / Bar Chart
- [ ] **Main results table**: rows = models, columns = route progress / collision rate / concept loss
  The central thesis result. Include SAC Baseline as reference ceiling.

### 3.5 Scenario Visualizations (for XAI demo section)
- [ ] **4 archetype scenarios** (BEV video frames + concept timeline):
  - Correct red light stop (true TL red → model brakes)
  - TTC success (lead decelerates → model catches it)
  - Curve navigation (high `path_curvature_max` → correct steering) [V2 model]
  - Concept-action failure (true vs pred concept diverges → wrong action)
- [ ] **Side-by-side comparison panel**: same scenario, SAC baseline vs CBM, showing interpretability

### 3.6 Concept Distribution Plots
- [ ] **Concept value distributions at t=0** across WOMD validation
  Shows which concepts are informative (high variance) vs degenerate at initialization.
  Particularly relevant for justifying the deferred `dist_to_path`.

---

## PHASE 4 — Demo Preparation (for Defense)

- [ ] **Write targeted scenario finder script** (`find_demo_scenarios.py`)
  - Scans N validation scenarios
  - Runs `extract_all_concepts` on each
  - Outputs ranked lists per archetype:
    - Top-20 highest `traffic_light_red` + low TTC + no collision
    - Top-20 highest `path_curvature_max` + successful navigation
    - Top-20 cases where `pred_concept` diverges most from `true_concept`
  - Runtime: ~30 min on local GPU
  - Output: JSON index with scenario IDs and their scores

- [ ] **Curate final demo scenarios** — manually review top candidates, pick 4–6 best
- [ ] **Render demo videos** for each archetype
  - BEV animation
  - Concept timeline panel below the video
  - Side-by-side with SAC baseline if possible

---

## PHASE 5 — Thesis Writing Tasks

- [ ] **Methods chapter**: architecture description (use `cbm-detailed-context-for-thesis-writing.md` as source)
  - Network architecture diagram
  - Concept supervision pipeline
  - Training paradigms comparison
  - Loss function derivation

- [ ] **Experiments chapter**:
  - Dataset description (WOMD)
  - Evaluation protocol
  - Main results table
  - Concept quality results
  - Ablation: Frozen vs Joint vs Scratch

- [ ] **Discussion chapter**:
  - Performance tradeoff (interpretability cost)
  - Concept quality analysis (which concepts work, which don't)
  - Failure mode analysis (using curated scenarios)
  - Open questions from `cbm-detailed-context-for-thesis-writing.md` §11

---

## PHASE 6 — Paper Prep (if results are strong)

Required for a publishable paper (targeting ITSC 2026 / IV 2026 / NeurIPS XAI workshop):

- [ ] **Confirm novelty**: do a final literature check for "CBM + SAC + autonomous driving"
- [ ] **Run ablations**:
  - λ_concept sensitivity: {0.01, 0.05, 0.1, 0.5}
  - Concept count sensitivity: 5, 11, 15 concepts
  - Frozen vs Joint vs Scratch performance curves
- [ ] **Statistical significance**: run at least 2–3 seeds for the main result
  (seeds 42, 69, 99 already have SAC baselines; run CBM Frozen V2 on same seeds)
- [ ] **Write 6-page paper draft** — use thesis methods + experiments sections as source

---

## Quick Reference — Key Commands

```bash
# Always activate environment first
conda activate vmax

# Smoke test (500 steps, local)
python cbm_v1/train_cbm.py --config cbm_v1/config_womd_frozen_short.yaml

# V2 smoke test (15 concepts, 500 steps, local)
python cbm_v1/train_cbm.py --config cbm_v1/config_womd_frozen_short.yaml \
    --num_concepts 15 --concept_phases "[1,2,3]" --run_name cbm_v2_smoke_local

# V2 frozen 150GB (cluster)
python cbm_v1/train_cbm.py --config cbm_v1/config_womd_frozen_v2_150gb.yaml

# Scratch V2 150GB (cluster)
python cbm_v1/train_cbm.py --config cbm_v1/config_womd_scratch.yaml

# TensorBoard
tensorboard --logdir runs_cbm/ --port 6006

# V2 smoke test (code verification)
python cbm_v1/smoke_test_v2.py
```
