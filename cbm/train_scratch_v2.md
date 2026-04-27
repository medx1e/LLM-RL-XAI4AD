# CBM Scratch V2 Training Guide

Train the CBM from scratch (no pretrained encoder) with the full **15-concept** set (Phase 1 + 2 + 3). This fixes the earlier scratch run which only trained 11 concepts due to a plumbing bug — `concept_phases` now flows correctly through the pipeline and all 15 concepts are supervised.

---

## 1. What Changed Since Last Scratch Run

### Bug Fix (must be pushed)
- `concept_phases` is now read from the YAML and passed end-to-end: `train_cbm.py → train() → CBMConfig → extract_all_concepts()`. Before, it defaulted to `(1, 2)` regardless of what the YAML said, so Phase 3 concepts had zero concept loss gradient.

### New Diagnostic Metrics (for TensorBoard)
Added to `cbm_sac_factory.py`:
- `train/concept_mean/<name>` and `train/concept_std/<name>` — per-neuron prediction statistics. Std collapsing to ~0 flags a dead/saturated concept.
- `train/concept_accuracy/<name>` — binary classification accuracy (threshold 0.5) for `traffic_light_red`, `lead_vehicle_decelerating`, `at_intersection`.
- `train/concept_task_grad_ratio` — ratio of concept_head update norm to actor_fc update norm. Signals whether `lambda_concept` is strong enough.

---

## 2. Files to Push to Cluster

| File | Change |
|---|---|
| `cbm_v1/train_cbm.py` | Reads `concept_phases` from config, passes to `train()` |
| `cbm_v1/cbm_trainer.py` | New `concept_phases` parameter, passed to `CBMConfig` and `extract_all_concepts()`; added startup print |
| `cbm_v1/cbm_sac_factory.py` | Added `_concept_pred_stats`, `_binary_concept_accuracy`, `_param_update_ratio`; wired into `sgd_step` metrics |
| `cbm_v1/config_womd_scratch_short.yaml` | **NEW** — local smoke test config (memory-safe) |
| `cbm_v1/config_womd_scratch.yaml` | No changes (already has `num_concepts: 15`, `concept_phases: [1,2,3]`) |
| `train_scratch_v2.md` | **NEW** — this file |
| `cbm-detailed-context-for-thesis-writing.md` | **NEW** — thesis reference |
| `to-do-tasks-for-cbm.md` | **NEW** — task list |

All other files (registry, extractors, adapters, concept_loss, networks, config.py) are unchanged.

---

## 3. Local Smoke Test (ALREADY VERIFIED ✓)

Before pushing, already ran:
```bash
conda activate vmax
python cbm_v1/train_cbm.py --config cbm_v1/config_womd_scratch_short.yaml
```
Result: 500 steps in ~3 minutes, `concept_loss` dropped 0.169 → 0.028, 15 concepts active, no NaN. The pipeline is healthy.

---

## 4. Cluster Training Plan

### 4.1 Lambda Ablation (10GB, cluster)

Before the full 150GB run, do the 4 short experiments to pick the best `lambda_concept`:

```bash
conda activate vmax

# Run 1: lambda = 0.01
python cbm_v1/train_cbm.py --config cbm_v1/config_womd_scratch.yaml \
    --lambda_concept 0.01 --run_name cbm_scratch_v2_lambda001 \
    --data_path <path-to-10gb-tfrecord> --total_timesteps 1_000_000

# Run 2: lambda = 0.1 (default)
python cbm_v1/train_cbm.py --config cbm_v1/config_womd_scratch.yaml \
    --lambda_concept 0.1 --run_name cbm_scratch_v2_lambda01 \
    --data_path <path-to-10gb-tfrecord> --total_timesteps 1_000_000

# Run 3: lambda = 0.5
python cbm_v1/train_cbm.py --config cbm_v1/config_womd_scratch.yaml \
    --lambda_concept 0.5 --run_name cbm_scratch_v2_lambda05 \
    --data_path <path-to-10gb-tfrecord> --total_timesteps 1_000_000

# Run 4 (after picking winner): warmup variant of best lambda
# (requires implementing lambda warmup — TODO, see to-do-tasks-for-cbm.md)
```

**Decision criteria after 4.1:**
- Compare `train/concept_loss` final value (lower is better)
- Compare `metrics/progress_ratio_nuplan` (higher is better)
- Compare `train/concept_task_grad_ratio` (should be > 0.01 — if tiny, lambda is too small)
- Pick the lambda on the Pareto frontier of (concept quality, task performance)

### 4.2 Full 150GB Scratch V2 Run

Once the winner lambda is picked:

```bash
conda activate vmax
python cbm_v1/train_cbm.py --config cbm_v1/config_womd_scratch.yaml \
    --lambda_concept <winner> --run_name cbm_scratch_v2_150gb_42
```

The config is already tuned for the cluster (A100/RTX 3090-class GPU):
- `total_timesteps: 15_000_000` (~1 epoch over 150GB)
- `buffer_size: 1_000_000`
- `batch_size: 128`, `num_episodes_per_env: 8`
- `save_freq: 1000` (checkpoint every ~640k env steps)

Expected wall time: multi-day on a single cluster GPU.

---

## 5. What to Monitor (TensorBoard)

**Critical signals (must be healthy):**
- `train/concept_loss` — monotonic decrease, especially in first 500k steps
- `train/concept_loss/path_curvature_max`, `path_straightness`, `path_net_heading_change`, `heading_to_path_end` — these are the Phase 3 concepts; they MUST drop, confirming the bug fix works
- `train/policy_loss` — trending more negative
- `metrics/at_fault_collision` — trending to 0
- `metrics/progress_ratio_nuplan` — trending up

**Diagnostic signals (new):**
- `train/concept_accuracy/traffic_light_red` — should reach >90% within 500k steps
- `train/concept_std/<any>` — if any std collapses near 0, that concept neuron is dead
- `train/concept_task_grad_ratio` — if < 0.01 consistently, lambda is too weak

**Red flags:**
- Any NaN in any loss — stop, investigate
- `concept_loss` plateaus at ~0.15 after 1M steps — concept head is not learning; check data and lambda
- `policy_loss` exploding positive — reduce learning rate or check reward scale

---

## 6. After Training — Evaluation

Run `eval_cbm.py` on the saved `model_final.pkl` against the WOMD validation set. Collect:
- Route progress ratio (nuplan)
- At-fault collision rate
- Per-concept accuracy / R² on validation data
- Compare against: SAC baseline, CBM Frozen V1 (150GB), CBM Frozen V2 (150GB)

See `to-do-tasks-for-cbm.md` Phase 2 for the full evaluation protocol and Phase 3 for the figures that must be generated for the thesis.

---

## 7. Quick Reference

```bash
# Environment (always first)
conda activate vmax

# Local smoke (already verified)
python cbm_v1/train_cbm.py --config cbm_v1/config_womd_scratch_short.yaml

# Cluster lambda ablation (10GB, ~1M steps each)
python cbm_v1/train_cbm.py --config cbm_v1/config_womd_scratch.yaml \
    --lambda_concept <val> --run_name cbm_scratch_v2_lambda<val> \
    --data_path <10gb-path> --total_timesteps 1_000_000

# Cluster full run (150GB)
python cbm_v1/train_cbm.py --config cbm_v1/config_womd_scratch.yaml \
    --lambda_concept <winner> --run_name cbm_scratch_v2_150gb_42

# TensorBoard
tensorboard --logdir runs_cbm/ --port 6006
```
