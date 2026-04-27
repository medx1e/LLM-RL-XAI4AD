# CBM Evaluation Guide

Run `eval_cbm_v2.py` on the WOMD **validation** split for each trained model. Produces a JSON results file and an optional rollout cache (`.npz`) for scenario curation and visualization.

---

## Setup

```bash
conda activate vmax
cd /home/med1e/platform_fyp/cbm
```

Confirm validation data exists:
```bash
ls -lh data/validation.tfrecord   # should be present on the cluster
```

If only `data/training.tfrecord` is available locally, use that path instead (results will be slightly optimistic but still comparable across models).

---

## What You Get

For each run you get two output files saved next to the checkpoint:

| File | Contents |
|---|---|
| `eval_model_final.json` | Aggregated results: concept accuracy, MAE, R², route progress, collision rate, per-scenario breakdown |
| `eval_model_final_cache.npz` | Full rollout: pred concepts, true concepts, ego actions, rewards, dones — per step per scenario. Used for curation and visualization. |

---

## Model-by-Model Commands

Run these one at a time. Each takes ~5–15 minutes on the cluster depending on `--num_scenarios`.

---

### 1. CBM Frozen V1 — 10GB (11 concepts)

```bash
python cbm_v1/eval_cbm_v2.py \
    --checkpoint cbm_model/checkpoints/model_final.pkl \
    --pretrained_dir runs_rlc/womd_sac_road_perceiver_minimal_42 \
    --data data/validation.tfrecord \
    --num_scenarios 200 \
    --mode frozen \
    --num_concepts 11 \
    --concept_phases 1 2
```

---

### 2. CBM Joint V1 — 10GB (11 concepts)

```bash
python cbm_v1/eval_cbm_v2.py \
    --checkpoint cbm_model_joint/checkpoints/model_final.pkl \
    --pretrained_dir runs_rlc/womd_sac_road_perceiver_minimal_42 \
    --data data/validation.tfrecord \
    --num_scenarios 200 \
    --mode joint \
    --num_concepts 11 \
    --concept_phases 1 2
```

---

### 3. CBM Frozen V1 — 150GB (11 concepts)

```bash
python cbm_v1/eval_cbm_v2.py \
    --checkpoint cbm_v1_frozen_150GB/checkpoints/model_final.pkl \
    --pretrained_dir runs_rlc/womd_sac_road_perceiver_minimal_42 \
    --data data/validation.tfrecord \
    --num_scenarios 200 \
    --mode frozen \
    --num_concepts 11 \
    --concept_phases 1 2
```

---

### 4. CBM Scratch V1 — 150GB (11 concepts, old run)

```bash
python cbm_v1/eval_cbm_v2.py \
    --checkpoint cbm_scratch/checkpoints/model_final.pkl \
    --config cbm_v1/config_womd_scratch.yaml \
    --data data/validation.tfrecord \
    --num_scenarios 200 \
    --mode scratch \
    --num_concepts 11 \
    --concept_phases 1 2
```

> Note: this model was trained with 11 concepts despite the current YAML saying 15 — override with `--num_concepts 11 --concept_phases 1 2`.

---

### 5. CBM Frozen V2 — 150GB (15 concepts)

```bash
python cbm_v1/eval_cbm_v2.py \
    --checkpoint cbm_v2_frozen_womd_150gb/checkpoints/model_final.pkl \
    --pretrained_dir runs_rlc/womd_sac_road_perceiver_minimal_42 \
    --data data/validation.tfrecord \
    --num_scenarios 200 \
    --mode frozen \
    --num_concepts 15 \
    --concept_phases 1 2 3
```

---

### 6. CBM Scratch V2 — 10GB lambda ablation winner (15 concepts)

```bash
python cbm_v1/eval_cbm_v2.py \
    --checkpoint cbm_scratch_v2_lambda05/checkpoints/model_final.pkl \
    --config cbm_v1/config_womd_scratch.yaml \
    --data data/validation.tfrecord \
    --num_scenarios 200 \
    --mode scratch \
    --num_concepts 15 \
    --concept_phases 1 2 3
```

---

### 7. CBM Scratch V2 — 150GB (15 concepts) ← run after training completes

```bash
python cbm_v1/eval_cbm_v2.py \
    --checkpoint runs_cbm/cbm_scratch_v2_150gb_42/checkpoints/model_final.pkl \
    --config cbm_v1/config_womd_scratch.yaml \
    --data data/validation.tfrecord \
    --num_scenarios 200 \
    --mode scratch \
    --num_concepts 15 \
    --concept_phases 1 2 3
```

---

## Key Flags

| Flag | Default | When to change |
|---|---|---|
| `--num_scenarios` | 200 | Increase to 500+ for final thesis numbers; reduce to 50 for a quick check |
| `--chunk_size` | 10 | Reduce to 5 if you get OOM during rollout |
| `--no_cache` | off | Add this flag if disk space is tight (skips the `.npz` file) |
| `--output_dir` | checkpoint dir | Set to a central folder to collect all JSONs together |

---

## Reading the Output

The console prints three sections:

**Concept accuracy table** — one row per concept:
- Binary concepts: `accuracy` — should be >95% for a well-trained model
- Continuous concepts: `MAE` (lower is better) + `R²` (higher is better, 1.0 = perfect)
- `Valid%` — fraction of timesteps where this concept was applicable. Low valid% (e.g., TTC at ~33%) is expected and not a problem.

**Task metrics** — key ones to check:
- `progress_ratio_nuplan` — fraction of route completed (higher is better, SAC baseline is ~0.97)
- `at_fault_collision` — fraction of episodes with a collision caused by ego (lower is better, target <0.01)
- `accuracy` — fraction of episodes with no early termination (higher is better)
- `offroad` — fraction of episodes that went offroad

**Summary block** — one-glance overview at the bottom.

---

## Collecting All Results into One Place

To keep all JSON results together, use `--output_dir`:

```bash
mkdir -p eval_results

python cbm_v1/eval_cbm_v2.py \
    --checkpoint cbm_model/checkpoints/model_final.pkl \
    --pretrained_dir runs_rlc/womd_sac_road_perceiver_minimal_42 \
    --data data/validation.tfrecord \
    --num_scenarios 200 --mode frozen --num_concepts 11 --concept_phases 1 2 \
    --output_dir eval_results

# repeat for each model with --output_dir eval_results
```

Then compare them all:
```bash
python - << 'EOF'
import json, glob, os
for p in sorted(glob.glob("eval_results/eval_*.json")):
    with open(p) as f: r = json.load(f)
    prog = r["task_metrics"].get("progress_ratio_nuplan", float("nan"))
    coll = r["task_metrics"].get("at_fault_collision", float("nan"))
    ckpt = os.path.basename(r["checkpoint"])
    nc   = r["num_concepts"]
    print(f"{ckpt:<45} concepts={nc}  progress={prog:.4f}  collision={coll:.4f}")
EOF
```

---

## Using the Cache for Scenario Curation

The `.npz` cache contains the full rollout data. Load it with:

```python
import numpy as np

cache = np.load("eval_results/eval_model_final_cache.npz", allow_pickle=True)

pred_concepts   = cache["pred_concepts"]    # (80, N, C) — model predictions
true_concepts   = cache["true_concepts"]    # (80, N, C) — ground truth
ego_actions     = cache["ego_actions"]      # (80, N, 2) — accel, steer per step
dones           = cache["dones"]            # (80, N)
concept_names   = list(cache["concept_names"])

# Example: find scenarios where model saw red light and correctly braked
tl_idx = concept_names.index("traffic_light_red")
tl_true = true_concepts[:, :, tl_idx]     # (80, N)
accel   = ego_actions[:, :, 0]            # (80, N)

# Scenarios with high TL confidence AND braking action at some step
tl_active  = (tl_true > 0.8).any(axis=0)  # (N,)
braked     = (accel < -0.3).any(axis=0)   # (N,)
candidates = np.where(tl_active & braked & (dones.max(axis=0) < 0.5))[0]
print(f"Found {len(candidates)} 'correct red light stop' scenarios: {candidates}")
```

---

## Troubleshooting

**`FileNotFoundError: No .hydra/config.yaml`** — you used `--pretrained_dir` on a scratch model. Switch to `--config cbm_v1/config_womd_scratch.yaml`.

**`RESOURCE_EXHAUSTED: Out of memory`** — reduce `--chunk_size` from 10 to 5 or 2.

**`nan` in concept accuracy** — the concept had zero valid entries in the evaluated scenarios. Not a bug; check `Valid%` column. Normal for sparse concepts like `ttc_lead_vehicle`.

**`KeyError` in network build** — `--num_concepts` or `--concept_phases` don't match what the checkpoint was trained with. Check the training config or TensorBoard logs for the correct values.
