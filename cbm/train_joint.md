# CBM-V1 Joint Mode — Training Guide

This guide is for launching the Phase 2 (Joint Mode) training run of CBM-V1.
Read it fully before touching anything.

---

## What You Are Doing

Phase 1 (Frozen Mode) trained the Concept Head + Actor/Critic while the encoder
was locked. Phase 2 (Joint Mode) **unfreezes the encoder** for end-to-end fine-tuning.
This should:
- Fix the negative R² on continuous concepts (progress_along_route, ttc_lead_vehicle)
- Bring task accuracy back up toward the 0.97 baseline
- Keep concept accuracy high (binary concepts were already at 92.6%)

You are **initialising from the Phase 1 frozen checkpoint**, not from scratch.

---

## Prerequisites

### 1. Environment

Use the `vmax` conda environment:

```bash
/home/med1e/anaconda3/envs/vmax/bin/python
```

Verify GPU is available:
```bash
/home/med1e/anaconda3/envs/vmax/bin/python -c "import jax; print(jax.devices())"
# Expected: [CudaDevice(id=0)]  or similar
```

### 2. Starting Checkpoint

The joint run **must** be initialised from the Phase 1 frozen CBM checkpoint.
Make sure the following files exist:

```bash
ls cbm_model/cbm_v1_frozen_womd_42/checkpoints/model_final.pkl   # weights
ls cbm_model/cbm_v1_frozen_womd_42/.hydra/config.yaml            # network arch
ls cbm_model/cbm_v1_frozen_womd_42/model_final.pkl               # symlink (required)
```

If the symlink is missing, create it:
```bash
ln -s /absolute/path/to/cbm_model/cbm_v1_frozen_womd_42/checkpoints/model_final.pkl \
      cbm_model/cbm_v1_frozen_womd_42/model_final.pkl
```

### 3. Data

```bash
ls -lh data/training.tfrecord
# Expected: ~951 MB WOMD TFRecord file
```

---

## Step 1 — Smoke Test (Always Run This First)

Runs 500 steps (~3 minutes). Verifies the full joint stack works on your machine.

```bash
cd /path/to/cbm
/home/med1e/anaconda3/envs/vmax/bin/python cbm_v1/train_cbm.py \
    --config cbm_v1/config_womd_joint_short.yaml
```

**Expected output:**
```
--> Warming up cuSolver handle...
    Done.
-> Prefilling replay buffer (200 steps)...
   Prefill done in Xs
CBM Training:  10%| | step=50   | policy=-0.25 value=0.05 concept=0.14 reward=0.000 sps=N
CBM Training: 100%| | step=500  | policy=-0.27 value=0.04 concept=0.04 reward=0.000 sps=N
-> Final checkpoint saved: runs_cbm/cbm_v1_joint_short_verify/checkpoints/model_final.pkl
```

**Check:**
- All three losses are finite (not `nan`)
- `concept_loss` decreases over the run
- "Warming up cuSolver handle..." appears and passes without error
- Final checkpoint is saved

---

## Step 2 — Config Tuning for Your GPU

The production config is at `cbm_v1/config_womd_joint.yaml`.
The defaults are set for a **GTX 1660 Ti (6 GB)**. If you have a better GPU,
increase the following to fully utilise your VRAM:

| Parameter | Default (6GB) | Recommended (≥12GB) | Recommended (≥24GB) |
|---|---|---|---|
| `num_episodes_per_env` | 2 (short) / 4 (prod) | 8 | 16 |
| `batch_size` | 32 | 64–128 | 256 |
| `grad_updates_per_step` | 2 | 4 | 8 |
| `buffer_size` | 200_000 | 500_000 | 1_000_000 |

Edit `cbm_v1/config_womd_joint.yaml` accordingly before launching the full run.

Also update `run_name` if you want a different output directory:
```yaml
run_name: "cbm_v1_joint_womd_42"     # outputs go to runs_cbm/cbm_v1_joint_womd_42/
```

---

## Step 3 — Full Training Run

```bash
/home/med1e/anaconda3/envs/vmax/bin/python cbm_v1/train_cbm.py \
    --config cbm_v1/config_womd_joint.yaml
```

This trains for **2,000,000 steps** in joint mode.
Estimated time: **6–12 hours** depending on GPU.

**Outputs:**
- Checkpoints every 500 iterations → `runs_cbm/cbm_v1_joint_womd_42/checkpoints/`
- Final checkpoint: `runs_cbm/cbm_v1_joint_womd_42/checkpoints/model_final.pkl`
- TensorBoard logs: `runs_cbm/cbm_v1_joint_womd_42/tb/`

---

## Step 4 — Monitoring

### TensorBoard

```bash
/home/med1e/anaconda3/envs/vmax/bin/tensorboard \
    --logdir runs_cbm/cbm_v1_joint_womd_42/tb --port 6006
```

Open `http://localhost:6006`. Key metrics to watch:

| TensorBoard Tag | What to look for |
|---|---|
| `metrics/vmax_score` | 🎯 **The main score** — should trend toward ~0.97 (baseline). Starts low, climbs. |
| `metrics/at_fault_collision` | Should decrease and stay near 0 |
| `train/concept_loss` | Should continue decreasing from frozen endpoint (~0.057) |
| `train/policy_loss` | Should become more negative over time |
| `train/value_loss` | Should stabilize near 0 |

> **Note**: `metrics/vmax_score = 1.0 - at_fault_collision_rate`. This is the same
> formula as the `runs_rlc/runs_accuracy.txt` leaderboard. The baseline is **0.9747**.

### GPU Monitoring (separate terminal)

```bash
watch -n 2 nvidia-smi
```

Expected VRAM usage: ~4–5 GB on a 6GB card, more on larger GPUs.

---

## Step 5 — Evaluation After Training

Run the evaluation script to get:
- Per-concept accuracy (MAE, R², binary accuracy)
- Task accuracy vs baseline comparison

```bash
/home/med1e/anaconda3/envs/vmax/bin/python cbm_v1/eval_cbm.py \
    --checkpoint runs_cbm/cbm_v1_joint_womd_42/checkpoints/model_final.pkl \
    --pretrained_dir cbm_model/cbm_v1_frozen_womd_42 \
    --data data/training.tfrecord \
    --num_scenarios 64
```

> **Note on `--num_scenarios`**: On a 6GB GPU, use 16. On 12GB+, use 64. On 24GB+, use 256.

Results are saved to:
```
runs_cbm/cbm_v1_joint_womd_42/checkpoints/eval_model_final.json
```

---

## Frozen Phase 1 Baseline (for comparison)

| Metric | Frozen Run (Phase 1) | Joint Target |
|---|---|---|
| Binary concept accuracy | 92.6% | ≥90% (maintain) |
| Continuous concept R² | -0.245 (poor) | >0.5 (goal) |
| Task accuracy | 81.25% | ~97.5% (baseline) |
| `vmax_score` | ~0.875 | ~0.97 |

---

## Troubleshooting

**`cuSolver internal error` at startup**
Check that the warm-up printed correctly:
```
--> Warming up cuSolver handle...
    Done.
```
If it still fails, reduce `num_episodes_per_env` further or add
`export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75` before the python command.

**`RESOURCE_EXHAUSTED: Out of memory`**
Reduce in `config_womd_joint.yaml`:
- `batch_size: 32 → 16`
- `num_episodes_per_env: 4 → 2`
- `buffer_size: 200_000 → 100_000`

**`FileNotFoundError: No .pkl checkpoint found`**
The symlink `cbm_model/cbm_v1_frozen_womd_42/model_final.pkl` may be missing.
Re-create it (see Prerequisites section above).

**`concept_loss` is flat / not decreasing in joint mode**
This is unusual. The joint encoder has gradients flowing all the way back.
Check `lambda_concept` is still 0.1 and that you started from the frozen checkpoint
(not from scratch).
