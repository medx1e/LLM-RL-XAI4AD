# Cluster Run Guide — Phase 3 Attention-Attribution Correlation

> **Who this is for:** teammate running the full-scale experiment on the cluster GPU.
> No prior knowledge of the codebase is assumed.
> Last updated: 2026-04-28

---

## What This Experiment Does

We have two pretrained autonomous driving models (V-MAX / Perceiver-SAC, trained on
Waymo WOMD). For each model we run 50 driving scenarios and, at every timestep, compute:

- **Attention signals** — where the model is "looking" in input space (2 variants)
- **Attribution methods** — which input features actually drive the action (3 methods)

We then correlate attention with attribution per scenario, stratified by risk level and
action type. The key research question: does attention agree with gradient-based feature
importance, and does that agreement change during safety-critical moments?

**Runtime estimate on a 24 GB GPU (A100/A6000/similar):**

| Step | Model | Time |
|---|---|---|
| JIT compile (one-off) | — | ~10 min |
| 50 scenarios × complete model | ~3–4 h | |
| 50 scenarios × minimal model | ~3–4 h | |
| **Total** | **~8 h** | |

Run both models overnight in **separate processes** (Waymax constraint — details below).

---

## 1. Codebase Structure

```
platform_fyp/
├── cbm/
│   ├── V-Max/                    ← V-Max simulator (JAX/Waymax)
│   ├── runs_rlc/                 ← pretrained model weights
│   └── data/training.tfrecord    ← Waymo dataset (~1 GB)
└── post-hoc-xai/
    ├── posthoc_xai/              ← XAI framework package
    ├── experiments/
    │   ├── phase3_cluster.py     ← ENTRY POINT — use this
    │   ├── phase3_scale_correlation.py  ← called by cluster.py
    │   └── phase3_results/       ← output (created automatically)
    └── reward_attention/         ← separate study, ignore
```

The two models:

| Key | Directory name |
|---|---|
| `complete` | `womd_sac_road_perceiver_complete_42` (TTC penalty reward) |
| `minimal`  | `womd_sac_road_perceiver_minimal_42`  (no TTC penalty) |

---

## 2. Environment Setup

### 2a. Clone / transfer the repo

You need the full `platform_fyp/` directory on the cluster, including:
- `cbm/V-Max/` (git submodule — make sure it is populated)
- `cbm/runs_rlc/` (model weights)
- `cbm/data/training.tfrecord` (Waymo dataset)

If transferring with rsync:
```bash
rsync -avz --progress platform_fyp/ cluster:~/platform_fyp/
```

### 2b. Create the conda environment

```bash
# On the cluster
conda create -n vmax python=3.10 -y
conda activate vmax

# Install V-Max and its dependencies (from the submodule)
cd ~/platform_fyp/cbm/V-Max
pip install -e .

# Required extras
pip install tensorboardX scipy pandas matplotlib
```

> **If the cluster already has a `vmax` environment from a previous run, just activate it
> and skip this step.**

### 2c. Verify the install

```bash
conda activate vmax
cd ~/platform_fyp/post-hoc-xai
python -c "import posthoc_xai; import jax; print('JAX devices:', jax.devices())"
```

Expected: prints `JAX devices: [CudaDevice(id=0)]` (or similar GPU device).

---

## 3. Running the Experiments

> **Critical constraint:** Waymax registers metrics globally at process startup.
> You **cannot** load two models in the same Python process.
> Run `complete` and `minimal` in separate terminal sessions (or as separate SLURM jobs).

### 3a. Complete model (first session / job)

```bash
conda activate vmax
cd ~/platform_fyp/post-hoc-xai
python experiments/phase3_cluster.py --model complete --data-path /path/to/training.tfrecord
```

### 3b. Minimal model (second session / job)

```bash
conda activate vmax
cd ~/platform_fyp/post-hoc-xai
python experiments/phase3_cluster.py --model minimal --data-path /path/to/training.tfrecord
```

If the dataset is already at `cbm/data/training.tfrecord` relative to the repo root, `--data-path` can be omitted.

That's it. The cluster entry point (`phase3_cluster.py`) automatically sets:

| Setting | Value | Why |
|---|---|---|
| Methods | `vg ig sarfa` | All three attribution methods |
| Attention signals | `rollout norm_weighted` | Both attention variants |
| IG chunk size | `200` | Exploits 24 GB VRAM (safe) |
| N scenarios | `50` | Default; override with `--n-scenarios` |

---

## 4. What the Script Produces

Results land in `experiments/phase3_results/{complete,minimal}/`:

```
phase3_results/
├── complete/
│   ├── scenario_0000_rollout.json        ← per-scenario correlation data (rollout attn)
│   ├── scenario_0000_norm_weighted.json  ← same, with norm-weighted attention
│   ├── scenario_0001_rollout.json
│   ├── ... (50 × 2 = 100 JSON files)
│   ├── df_risk.csv      ← tidy DataFrame: risk-stratified ρ
│   ├── df_action.csv    ← tidy DataFrame: action-conditioned ρ
│   ├── df_overall.csv   ← tidy DataFrame: per-scenario ρ
│   ├── df_cat.csv       ← tidy DataFrame: per-category ρ
│   └── figures/
│       ├── fig1_risk_stratified_complete_rollout.pdf
│       ├── fig1_risk_stratified_complete_norm_weighted.pdf
│       ├── fig2_cat_heatmap_*.pdf
│       ├── fig3_action_conditioned_*.pdf
│       └── fig4_distribution_*.pdf
└── minimal/
    └── ... (same structure)
```

Each JSON contains: scenario ID, attention signal used, per-method Pearson ρ overall,
risk-stratified (calm / moderate / high), and action-conditioned (braking / accelerating /
steering / neutral).

---

## 5. Resume-Friendly — Safe to Interrupt

The script checks for existing JSON files before processing each scenario.
If interrupted, just re-run the **same command** — it will skip completed scenarios
and continue from where it left off.

```
[skip] scenario 0000 — already done
[skip] scenario 0001 — already done
  Scenario 0002...   T=80, risk=0.31
```

The skip check is **per attention signal**: if `scenario_0005_rollout.json` exists but
`scenario_0005_norm_weighted.json` does not, the script will re-run the forward pass
and compute only the missing norm_weighted signal. Attribution results (VG/IG/SARFA)
are always recomputed for the pending signals.

---

## 6. Memory / OOM Troubleshooting

If you see an **out-of-memory error**:

1. Reduce the IG chunk size. Edit `phase3_cluster.py` line ~14:
   ```python
   CLUSTER_CHUNK_SIZE = 100   # was 200
   ```

2. If still OOM, reduce further to `40` (same as local 6 GB setting).

3. IG is the memory-hungry method. If you want fast results first, run VG + SARFA only:
   ```bash
   python experiments/phase3_scale_correlation.py \
       --model complete \
       --methods vg sarfa \
       --attention-signals rollout norm_weighted \
       --chunk-size 200
   ```
   Then add IG in a second pass (the resume logic handles this cleanly).

---

## 7. Checking Progress

While the script runs, watch the terminal output:

```
Phase 3 — complete  |  50 scenarios  |  methods: ['vg', 'ig', 'sarfa']  |  signals: ['rollout', 'norm_weighted']  |  chunk: 200
...
  Scenario 0000...   T=80, risk=0.18
  Scenario 0001...   T=80, risk=0.43
  ...
```

Or count completed JSONs from another terminal:
```bash
ls experiments/phase3_results/complete/scenario_*_rollout.json | wc -l
ls experiments/phase3_results/minimal/scenario_*_rollout.json  | wc -l
```

---

## 8. After Both Runs Complete

The script automatically generates figures and prints a summary at the end.
To re-generate figures without recomputing anything:

```bash
python experiments/phase3_cluster.py --model complete --figures-only
python experiments/phase3_cluster.py --model minimal  --figures-only
```

Then **transfer the results back**:

```bash
# From your local machine
rsync -avz cluster:~/platform_fyp/post-hoc-xai/experiments/phase3_results/ \
      ./post-hoc-xai/experiments/phase3_results/
```

The key files to transfer: all `*.json`, `*.csv`, and the `figures/` directories.

---

## 9. SLURM Job Script (if the cluster uses SLURM)

Save as `run_phase3.sh`, submit with `sbatch run_phase3.sh complete` and
`sbatch run_phase3.sh minimal`:

```bash
#!/bin/bash
#SBATCH --job-name=phase3_vmax
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=logs/phase3_%x_%j.log

MODEL=$1   # "complete" or "minimal"

if [ -z "$MODEL" ]; then
    echo "Usage: sbatch run_phase3.sh complete|minimal"
    exit 1
fi

source ~/anaconda3/etc/profile.d/conda.sh
conda activate vmax

cd ~/platform_fyp/post-hoc-xai
echo "Starting Phase 3 — model=$MODEL  $(date)"

python experiments/phase3_cluster.py --model $MODEL

echo "Done — $(date)"
```

Submit both jobs:
```bash
mkdir -p logs
sbatch run_phase3.sh complete
sbatch run_phase3.sh minimal
```

Check status:
```bash
squeue -u $USER
```

---

## 10. Quick Reference — All CLI Options

```
# Full cluster run (what phase3_cluster.py does internally):
python experiments/phase3_scale_correlation.py \
    --model complete \
    --n-scenarios 50 \
    --methods vg ig sarfa \
    --attention-signals rollout norm_weighted \
    --chunk-size 200

# VG + SARFA only (fast, ~30 min per model):
python experiments/phase3_scale_correlation.py \
    --model complete \
    --methods vg sarfa \
    --attention-signals rollout norm_weighted \
    --chunk-size 200

# Figures only (no recomputation):
python experiments/phase3_cluster.py --model complete --figures-only

# Custom scenario count:
python experiments/phase3_cluster.py --model complete --n-scenarios 100
```

---

## 11. What to Send Back

Once both models finish, zip and send:

```bash
cd ~/platform_fyp/post-hoc-xai/experiments
zip -r phase3_results_$(date +%Y%m%d).zip phase3_results/
```

The full result set is ~50–100 MB. The key outputs are:

| File | Used for |
|---|---|
| `phase3_results/*/df_*.csv` | Cross-model comparison figures (local) |
| `phase3_results/*/figures/*.pdf` | Thesis figures (300 DPI, ready to include) |
| `phase3_results/*/scenario_*.json` | Raw data for any additional analysis |

---

## Common Errors

| Error | Cause | Fix |
|---|---|---|
| `MetricRegistry already registered` | Two models in one process | Use separate terminals / SLURM jobs |
| `CUDA out of memory` | IG chunk too large | Reduce `CLUSTER_CHUNK_SIZE` to 100 |
| `No module named posthoc_xai` | Wrong working directory | `cd ~/platform_fyp/post-hoc-xai` first |
| `No module named tensorboardX` | Missing dependency | `pip install tensorboardX` |
| `FileNotFoundError: training.tfrecord` | Dataset not transferred | Copy `cbm/data/training.tfrecord` |
| `KeyError: speed_limit` | Wrong model loaded | Only use `complete` or `minimal`, never `sac_seed*` |
