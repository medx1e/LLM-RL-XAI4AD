# CBM-V1 Training Guide

This guide covers everything you need to launch a CBM-V1 training run from scratch. Read it top to bottom before touching anything.

---

## What You Are Training

CBM-V1 is a Hard Concept Bottleneck Model built on top of V-Max's SAC agent. The architecture is:

```
observation (1655-d)
    │
    ▼
LQ Encoder  ← pretrained, FROZEN in default mode
    │
    ▼ z (128-d latent)
    │
    ▼
Concept Head  → c (11-d concept vector, values in [0,1])
    │
    ├──▶ Actor FC → action distribution
    └──▶ Critic FC (+ action) → Q-value
```

The policy and critic **only see the 11-d concept vector**, never the raw latent. The 11 concepts are:

| Index | Name | Type | Loss |
|-------|------|------|------|
| 0 | ego_speed | continuous | Huber |
| 1 | ego_acceleration | continuous | Huber |
| 2 | dist_nearest_object | continuous | Huber |
| 3 | num_objects_within_10m | continuous | Huber |
| 4 | traffic_light_red | binary | BCE |
| 5 | dist_to_traffic_light | continuous | Huber |
| 6 | heading_deviation | continuous | Huber |
| 7 | progress_along_route | continuous | Huber |
| 8 | ttc_lead_vehicle | continuous | Huber |
| 9 | lead_vehicle_decelerating | binary | BCE |
| 10 | at_intersection | binary | BCE |

**Total loss** = SAC_loss + 0.1 × concept_loss

---

## Directory Layout

```
cbm/
├── cbm_v1/
│   ├── train_cbm.py              ← launcher (run this)
│   ├── cbm_trainer.py            ← training loop
│   ├── cbm_sac_factory.py        ← network/loss factory
│   ├── config_womd_frozen.yaml   ← PRODUCTION config (2M steps)
│   ├── config_womd_frozen_short.yaml  ← smoke-test config (500 steps)
│   ├── concept_loss.py
│   ├── config.py
│   └── networks.py
│
├── data/
│   └── training.tfrecord         ← WOMD training data (951 MB)
│
├── runs_rlc/
│   └── womd_sac_road_perceiver_minimal_42/
│       ├── model/
│       │   └── model_final.pkl   ← pretrained encoder checkpoint
│       └── .hydra/
│           └── config.yaml       ← pretrained run config (READ-ONLY)
│
└── runs_cbm/                     ← outputs go here (created on first run)
    └── <run_name>/
        ├── checkpoints/
        │   ├── model_<step>.pkl
        │   └── model_final.pkl
        └── tb/                   ← TensorBoard logs
```

---

## Step 1 — Environment Setup

Everything runs inside the `vmax` conda environment. **Do not use `conda activate`** — use the full Python path:

```bash
/home/med1e/anaconda3/envs/vmax/bin/python
```

Verify it works:

```bash
/home/med1e/anaconda3/envs/vmax/bin/python -c "import jax; print(jax.devices())"
# Expected: [CudaDevice(id=0)]
```

All commands below assume you are in the project root:

```bash
cd /home/med1e/cbm
```

---

## Step 2 — Set the Data Path

The training data is a single WOMD TFRecord file. Confirm it exists:

```bash
ls -lh data/training.tfrecord
# Expected: -rw-r--r-- ... 908M ... data/training.tfrecord
```

The path is set in the config YAML:

```yaml
# cbm_v1/config_womd_frozen.yaml
data_path: "data/training.tfrecord"
```

**If your data is in a different location**, open `cbm_v1/config_womd_frozen.yaml` and change `data_path` to the absolute path of your TFRecord file. The file must be a WOMD-format TFRecord with SDC paths included.

```yaml
data_path: "/your/path/to/womd_training.tfrecord"
```

---

## Step 3 — Set the Pretrained Checkpoint Path

The encoder is initialized from a pretrained V-Max SAC run. Confirm it exists:

```bash
ls runs_rlc/womd_sac_road_perceiver_minimal_42/model/
# Expected: model_final.pkl
```

This is set in the config:

```yaml
pretrained_dir: "runs_rlc/womd_sac_road_perceiver_minimal_42"
```

The launcher will automatically find `model_final.pkl` inside that directory's `model/` subfolder.

**If you want to use a different pretrained run**, change `pretrained_dir` to any V-Max run that used the `perceiver` encoder. The `encoder_remap` field handles the `perceiver → lq` name translation automatically — leave it as-is.

---

## Step 4 — Smoke Test (Run This First)

Always run the short config before committing to the full run. It trains for 500 steps and takes about 3–4 minutes.

```bash
/home/med1e/anaconda3/envs/vmax/bin/python cbm_v1/train_cbm.py \
    --config cbm_v1/config_womd_frozen_short.yaml
```

**Expected output:**

```
CBM Training:  10%| | step=50   | policy=-0.26 value=0.04 concept=0.14 reward=0.00 sps=27
CBM Training:  50%| | step=250  | policy=-0.46 value=0.02 concept=0.12 reward=0.00 sps=26
CBM Training: 100%| | step=500  | policy=-0.83 value=0.02 concept=0.11 reward=0.00 sps=24
-> Final checkpoint saved: runs_cbm/cbm_v1_frozen_short_verify/checkpoints/model_final.pkl
```

**What to check:**
- All three losses (`policy`, `value`, `concept`) are finite numbers, not `nan`
- `concept_loss` is decreasing over the run (concept head is learning)
- Final checkpoint file appears under `runs_cbm/`

If you see `nan` for all losses, something is wrong with the data path or the pretrained checkpoint. Stop and debug before continuing.

---

## Step 5 — Full Training Run

Once the smoke test passes:

```bash
/home/med1e/anaconda3/envs/vmax/bin/python cbm_v1/train_cbm.py \
    --config cbm_v1/config_womd_frozen.yaml
```

This trains for **2,000,000 environment steps** in `frozen` mode (encoder fixed). Estimated wall time: **6–8 hours** on a single GTX 1660 Ti.

**Outputs:**
- Checkpoints every 500 iterations (~160k steps each): `runs_cbm/cbm_v1_frozen_womd_42/checkpoints/model_<step>.pkl`
- Final checkpoint: `runs_cbm/cbm_v1_frozen_womd_42/checkpoints/model_final.pkl`
- TensorBoard logs: `runs_cbm/cbm_v1_frozen_womd_42/tb/`

**View TensorBoard:**
```bash
/home/med1e/anaconda3/envs/vmax/bin/tensorboard \
    --logdir runs_cbm/cbm_v1_frozen_womd_42/tb --port 6006
```

---

## Config Reference

All settings are in `cbm_v1/config_womd_frozen.yaml`. The ones you are most likely to change:

| Field | Default | What it does |
|-------|---------|-------------|
| `data_path` | `"data/training.tfrecord"` | Path to WOMD TFRecord |
| `pretrained_dir` | `"runs_rlc/womd_sac_road_perceiver_minimal_42"` | V-Max run dir for encoder init |
| `output_dir` | `"runs_cbm"` | Root dir for outputs |
| `run_name` | `"cbm_v1_frozen_womd_42"` | Subdirectory for this run |
| `mode` | `"frozen"` | `"frozen"` = encoder fixed; `"joint"` = encoder trains too |
| `total_timesteps` | `2_000_000` | Total env steps |
| `lambda_concept` | `0.1` | Weight of concept supervision in total loss |
| `learning_rate` | `0.0001` | Adam LR for all networks |
| `seed` | `42` | RNG seed |
| `save_freq` | `500` | Checkpoint every N training iterations |
| `log_freq` | `10` | Log metrics every N iterations |

You can also override any config field from the command line:

```bash
/home/med1e/anaconda3/envs/vmax/bin/python cbm_v1/train_cbm.py \
    --config cbm_v1/config_womd_frozen.yaml \
    --seed 1 \
    --run_name cbm_v1_frozen_womd_seed1 \
    --total_timesteps 5000000
```

---

## Running Multiple Seeds

To run seeds 42, 69, 99 (matching the pretrained baselines):

```bash
# Seed 42 — uses the default config
/home/med1e/anaconda3/envs/vmax/bin/python cbm_v1/train_cbm.py \
    --config cbm_v1/config_womd_frozen.yaml \
    --seed 42 --run_name cbm_v1_frozen_womd_42

# Seed 69
/home/med1e/anaconda3/envs/vmax/bin/python cbm_v1/train_cbm.py \
    --config cbm_v1/config_womd_frozen.yaml \
    --seed 69 --run_name cbm_v1_frozen_womd_69 \
    --pretrained_dir runs_rlc/womd_sac_road_perceiver_minimal_69

# Seed 99
/home/med1e/anaconda3/envs/vmax/bin/python cbm_v1/train_cbm.py \
    --config cbm_v1/config_womd_frozen.yaml \
    --seed 99 --run_name cbm_v1_frozen_womd_99 \
    --pretrained_dir runs_rlc/womd_sac_road_perceiver_minimal_99
```

---

## What to Monitor

Watch these three losses during training. All should be finite from the very first logged iteration.

| Metric | Key | Expected behavior |
|--------|-----|-------------------|
| `policy` | `train/policy_loss` | Starts around −0.2, should decrease (become more negative) over time |
| `value` | `train/value_loss` | Starts ~0.05, should decrease and stabilize near 0 |
| `concept` | `train/concept_loss` | Starts ~0.15–0.20, should decrease steadily — this is the key CBM signal |

**Red flags:**
- Any metric showing `nan` — stop, check data path and pretrained checkpoint
- `concept_loss` not decreasing after 50k steps — check the concept extraction pipeline
- `policy_loss` exploding positive — learning rate may be too high

---

## Frozen vs. Joint Mode

The default config uses `mode: frozen`. Here is the difference:

| | `frozen` | `joint` |
|--|---------|---------|
| Encoder gradients | 0 (locked) | Non-zero (fine-tuned) |
| What learns | concept_head + actor FC + critic FC | Everything |
| When to use | First run, concept extraction validation | After frozen converges |
| Risk | Encoder bottleneck limits task performance | Can destabilize concept alignment |

**Recommended sequence:**
1. Train frozen to 2M steps → verify concept_loss converges
2. Initialize joint run from the frozen checkpoint → train another 1–2M steps

To run joint mode, change `mode: joint` in the config, point `pretrained_dir` to your frozen run's checkpoint directory, and give it a new `run_name`.

---

## Baseline Reference

The pretrained V-Max SAC baselines (no concept bottleneck) achieved these scores on WOMD evaluation:

| Run | Score | at-fault collision score |
|-----|-------|--------------------------|
| `womd_sac_road_perceiver_minimal_42` | 0.9747 | 0.9825 |
| `womd_sac_road_perceiver_minimal_69` | 0.9744 | 0.9800 |
| `womd_sac_road_perceiver_minimal_99` | 0.9687 | 0.9754 |

CBM-V1 uses the encoder from these runs. The goal is to match or exceed these scores while also achieving low `concept_loss` (concept head accuracy).

---

## Troubleshooting

**`FileNotFoundError: No .pkl checkpoint found`**
The launcher looks for the checkpoint inside `<pretrained_dir>/model/`. Verify:
```bash
ls runs_rlc/womd_sac_road_perceiver_minimal_42/model/
# Must contain: model_final.pkl
```

**`FileNotFoundError: Dataset not found`**
Check `data_path` in your config. Use an absolute path if relative paths cause issues.

**`nan` losses from iteration 1**
The pretrained encoder weights failed to load. Run the smoke test first and watch for the line:
```
INFO Loading pretrained params from .../model_final.pkl
Remapping: vmax.learning.algorithms.rl.sac.sac_factory -> ...
```
If either line is missing, the checkpoint wasn't loaded.

**`CUDA out of memory`**
Reduce `batch_size` (from 64 to 32) and `buffer_size` (from 500_000 to 200_000) in the config. The GTX 1660 Ti has 6 GB VRAM; the current config targets ~4.5 GB peak usage.

**`concept_loss` is constant / not decreasing**
Verify the concept extraction pipeline works independently:
```bash
/home/med1e/anaconda3/envs/vmax/bin/python cbm_v1/smoke_test.py
# All 22 checks should PASS
```
