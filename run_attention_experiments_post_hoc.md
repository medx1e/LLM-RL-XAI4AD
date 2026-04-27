# Post-Hoc XAI Experiments — Cluster Run Guide

**Purpose:** Run the large-scale attention-attribution correlation study
(Phase 3) on the GPU cluster. This experiment correlates Perceiver
cross-attention with three attribution methods (VG, IG, SARFA) across
50 driving scenarios per model, with risk and action stratification.

**Expected runtime per model (24GB GPU):** 45–90 minutes  
**Output:** PDF figures + JSON results — transfer back to local machine when done.

---

## 1. Repository Setup

Clone or copy the full `platform_fyp` directory to your cluster home. The
structure must be:

```
platform_fyp/
├── post-hoc-xai/          ← main experiment code
│   ├── experiments/
│   │   ├── phase3_cluster.py          ← run this
│   │   └── phase3_scale_correlation.py
│   ├── posthoc_xai/
│   └── ...
└── cbm/
    ├── runs_rlc/           ← pretrained model weights (35+ folders)
    ├── data/
    │   └── training.tfrecord   ← Waymo dataset (~1GB)
    └── V-Max/              ← V-Max simulator (git submodule)
```

> If you only have the `post-hoc-xai/` directory, you also need `cbm/` with
> the model weights and dataset. Ask the owner to transfer `cbm/runs_rlc/`
> and `cbm/data/training.tfrecord`.

---

## 2. Environment Setup

The project uses a conda environment called `vmax`.

```bash
# If the environment does not exist yet, create it:
conda create -n vmax python=3.10
conda activate vmax

# Install V-Max dependencies (from inside cbm/V-Max/):
cd ~/platform_fyp/cbm/V-Max
pip install -e ".[dev]"

# Additional dependencies:
pip install tensorboardX scipy seaborn pandas
```

If the environment already exists on the cluster:
```bash
conda activate vmax
python -c "import jax; print(jax.__version__)"  # should print 0.5.x
```

---

## 3. Verify GPU Access

```bash
conda activate vmax
python -c "import jax; print(jax.devices())"
# Should show: [CudaDevice(id=0)] or similar
```

If JAX sees the GPU, you're ready.

---

## 4. Run the Experiments

Navigate to the experiments directory:

```bash
cd ~/platform_fyp/post-hoc-xai/experiments
conda activate vmax
```

**Run complete model first:**
```bash
python phase3_cluster.py --model complete
```

**Then minimal model:**
```bash
python phase3_cluster.py --model minimal
```

Each run saves results incrementally — if it crashes, just re-run the same
command and it resumes from where it left off (completed scenarios are skipped).

**To run in background (recommended for long jobs):**
```bash
nohup python phase3_cluster.py --model complete > logs_complete.txt 2>&1 &
nohup python phase3_cluster.py --model minimal  > logs_minimal.txt  2>&1 &
```

Check progress:
```bash
tail -f logs_complete.txt
# or count completed scenarios:
ls phase3_results/complete/scenario_*.json | wc -l
```

---

## 5. Configuration (if you need to change anything)

Open `phase3_cluster.py` — the only things you should ever change are at
the top of the file:

```python
CLUSTER_METHODS    = ["vg", "ig", "sarfa"]   # methods to run
CLUSTER_CHUNK_SIZE = 200   # IG batch size — reduce to 100 if OOM
```

To run more scenarios:
```bash
python phase3_cluster.py --model complete --n-scenarios 100
```

---

## 6. Memory Issues (OOM)

If you get CUDA out-of-memory errors during IG computation, reduce the
chunk size:

```python
CLUSTER_CHUNK_SIZE = 100  # in phase3_cluster.py
```

VG and SARFA are fully batched and should not OOM on 24GB.

---

## 7. What the Script Computes

For each of the 50 scenarios (per model):

1. **Rollout-corrected attention** — forward pass through updated Perceiver
   extractor that chains self-attention through cross-attention (Abnar &
   Zuidema 2020 rollout).

2. **VG** (Vanilla Gradient) — batched `jax.vmap(jax.grad(...))` over all
   timesteps in one JIT call. Very fast.

3. **IG** (Integrated Gradients) — validity-zeroed mean baseline (not the
   naive zero baseline). Chunked for GPU memory.

4. **SARFA** (Specific and Relevant Feature Attribution) — RL-specific method.
   6 batched forward passes for all timesteps (1 baseline + 5 category
   perturbations). Fast on 24GB.

For each scenario, three analyses are saved:
- **Overall correlation** (Pearson ρ, attention vs each method)
- **Risk-stratified** (calm / moderate / high collision risk)
- **Action-conditioned** (braking / accelerating / steering / neutral)

---

## 8. Output Structure

```
post-hoc-xai/experiments/phase3_results/
├── complete/
│   ├── scenario_0000.json   ← per-scenario results
│   ├── scenario_0001.json
│   ├── ...
│   ├── scenario_0049.json
│   ├── df_risk.csv          ← tidy DataFrames for analysis
│   ├── df_action.csv
│   ├── df_overall.csv
│   ├── df_cat.csv
│   └── figures/
│       ├── fig1_risk_stratified_complete.pdf   ← THESIS FIGURES
│       ├── fig1_risk_stratified_complete.png
│       ├── fig2_cat_heatmap_complete_vg.pdf
│       ├── fig2_cat_heatmap_complete_ig.pdf
│       ├── fig2_cat_heatmap_complete_sarfa.pdf
│       ├── fig3_action_conditioned_complete.pdf
│       └── fig4_distribution_complete.pdf
└── minimal/
    └── (same structure)
```

All figures are saved as **PDF at 300 DPI with embedded fonts** — ready for
direct inclusion in LaTeX thesis or paper.

---

## 9. Regenerate Figures Without Recomputing

If computation is complete but you want to tweak figure style:

```bash
python phase3_cluster.py --model complete --figures-only
python phase3_cluster.py --model minimal  --figures-only
```

---

## 10. Transfer Results Back

Once both models are done, transfer the results directory:

```bash
# From your local machine:
rsync -avz cluster_user@cluster_host:~/platform_fyp/post-hoc-xai/experiments/phase3_results/ \
      ~/platform_fyp/post-hoc-xai/experiments/phase3_results/
```

Or compress and download:
```bash
# On cluster:
cd ~/platform_fyp/post-hoc-xai/experiments
tar -czf phase3_results.tar.gz phase3_results/

# Locally:
scp cluster_user@cluster_host:~/platform_fyp/post-hoc-xai/experiments/phase3_results.tar.gz .
tar -xzf phase3_results.tar.gz -C ~/platform_fyp/post-hoc-xai/experiments/
```

---

## 11. Key Things to Verify After the Run

Look at the printed summary at the end of each run. The headline numbers to
check:

```
  VG:
    Overall:  mean ρ = ?       (expected: 0.3–0.5)
    calm:     mean ρ = ?
    moderate: mean ρ = ?
    high:     mean ρ = ?       (expected: higher than calm)

  IG:
    ...

  SARFA:
    Overall:  mean ρ = ?       (expected: highest of the three — SARFA is RL-specific)
    high:     mean ρ = ?       (expected: strongest signal here)
```

If `high` risk ρ > `calm` risk ρ for any method, the risk-stratification
finding is confirmed — attention is conditionally explanatory.

If SARFA ρ > VG ρ, that is the key publishable result connecting attention
to action-specific RL feature importance.

---

## 12. Contact

If anything breaks, share the error log (`logs_complete.txt`) with the
project owner. Most likely issues:
- Path to `cbm/` not found → check the directory structure in Section 1
- `No module named 'vmax'` → V-Max not installed, see Section 2
- OOM during IG → reduce `CLUSTER_CHUNK_SIZE` to 100, see Section 6
