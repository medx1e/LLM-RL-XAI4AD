# Reward-Conditioned Attention — Next Steps (RLC 2026)

> Resume file. Read alongside REWARD_CONDITIONED_FINDINGS.md, REWARD_CONDITIONED_OVERVIEW.md,
> REWARD_CONDITIONED_TECHNICAL.md before starting work.
> Last updated: 2026-02-22

---

## Current State (as of this writing)

- **Complete model**: 50 scenarios / 3,676 timesteps DONE ✓
- **Minimal model**: 3 scenarios only (needs scale-up)
- **Basic model**: 3 scenarios only (needs scale-up)
- **analyze_results.py**: enhanced with lead-lag, ρ histogram, agent-count confound ✓
- **Key bug fixed**: `sig_pct` in fisher_summary was comparing rho < 0.05 instead of p < 0.05 (now 80.6% not 26%) ✓

### Key numbers to remember
```
collision_risk × attn_agents:  mean ρ=+0.291  CI=[+0.125,+0.442]  sig=80.6%  n=31 HV scenarios
Lead-lag:                       best ρ=+0.281 at lag=+2 (attention leads risk by 2 steps)
Agent-count confound:           raw ρ=+0.262  partial ρ=+0.247  Δ=−0.014  (confound is negligible)
Counter-examples:               s009 (ρ=−0.383), s031 (ρ=−0.559) — unexplained
GPS gradient (3 scenarios):     minimal=0.314 > complete=0.166 > basic=0.092
Vigilance prior (3 scenarios):  complete attn_agents mean=0.173 vs minimal=0.117 (−32%)
```

---

## Priority 1 — Must Do (paper not submittable without these)

### 1a. Run minimal model on 50 scenarios

```bash
cd /home/med1e/post-hoc-xai
eval "$(~/anaconda3/bin/conda shell.bash hook)" && conda activate vmax
export PYTHONPATH=/home/med1e/post-hoc-xai/V-Max:$PYTHONPATH

nohup python reward_attention/run_experiment.py \
    --model runs_rlc/womd_sac_road_perceiver_minimal_42 \
    --data data/training.tfrecord \
    --n-scenarios 50 \
    --output results/reward_attention \
    > logs/minimal_50scenarios.log 2>&1 &
```

**Why**: GPS gradient and vigilance prior findings currently rest on 3 scenarios — anecdotal.
This confirms (or refutes) them at scale. Single most important pending experiment.

**After it finishes**, run the 3-model comparison:
```bash
python reward_attention/analyze_results.py \
    --pkl results/reward_attention/womd_sac_road_perceiver_complete_42/timestep_data.pkl \
    --compare results/reward_attention/womd_sac_road_perceiver_minimal_42/timestep_data.pkl \
    --compare-label minimal \
    --top-n 10
```

---

### 1b. Investigate counter-examples (s009, s031)

These scenarios have **negative** within-episode ρ (attention drops when risk rises):
- s009: risk_std=0.396, ρ=−0.383
- s031: risk_std=0.379, ρ=−0.559

**What to do**:
1. Generate timeseries for both (already done — check `fig_timeseries_s009.png`, `fig_timeseries_s031.png`)
2. Generate BEV panels for both (needs GPU):
```bash
python reward_attention/bev_attention.py --scenario-idx 9 --no-video
python reward_attention/bev_attention.py --scenario-idx 31 --no-video
```
3. Manually inspect: are these lane changes? High-speed scenarios? Committed maneuvers?
4. Label the mechanism — if you can say *why* attention inverts, it becomes a finding not a gap.

---

### 1c. Lead-lag per-scenario distribution

The aggregate lead-lag shows best ρ at lag=+2. But is that consistent or dominated by outliers?

**What to add to `analyze_results.py`**: a histogram of per-scenario *best lag* values.
For each HV scenario, find the lag (−8 to +8) that gives the highest ρ. Plot distribution.
- If most scenarios cluster at lag=+1 to +3 → anticipatory claim is strong
- If spread is wide → aggregate lag=+2 is noise

This is ~1 hour of coding in `analyze_results.py`, no GPU needed.

---

## Priority 2 — Should Do (significantly strengthens paper)

### 2a. Run basic model on 50 scenarios

```bash
nohup python reward_attention/run_experiment.py \
    --model runs_rlc/womd_sac_road_perceiver_basic_42 \
    --data data/training.tfrecord \
    --n-scenarios 50 \
    --output results/reward_attention \
    > logs/basic_50scenarios.log 2>&1 &
```

Note: basic model crashes in ~40% of scenarios — early terminations expected.
Still worth running to confirm GPS gradient at scale.

---

### 2b. Attention entropy analysis

Instead of mean attention per token group, measure **Shannon entropy** of the full attention
distribution across all 280 tokens at each timestep.

**Hypothesis**: entropy drops under danger (attention concentrates on fewer agents).
If true: the model doesn't just shift attention *toward* agents — it *focuses* on fewer of them.

No GPU needed — compute from pkl:
```python
import numpy as np
# attn_per_agent is already in the pkl records (shape: n_agents,)
# entropy = -sum(p * log(p)) for the attention distribution
```

Check field `attn_per_agent` is in the records (it is — verified).
This is ~2 hours coding + generates one strong figure.

---

## Priority 3 — Nice to Have (if time allows)

### 3a. One architecture comparison (Wayformer or MTR)

Run the same 50-scenario experiment on `womd_sac_road_wayformer_minimal_42`.
If risk→attention correlation appears in Wayformer too → claim generalizes beyond Perceiver.
Not essential for RLC but upgrades paper substantially.

```bash
python reward_attention/run_experiment.py \
    --model runs_rlc/womd_sac_road_wayformer_minimal_42 \
    --n-scenarios 50
```

---

## What NOT to Do

- Do not run more than 50 scenarios — statistical power already sufficient
- Do not attempt causal intervention study — too heavyweight for RLC
- Do not wait for all results before writing — paper writing is the actual bottleneck

---

## Paper Writing Checklist

Start writing in parallel with the experiments. Sections that can be written now:

- [x] Findings documented (REWARD_CONDITIONED_FINDINGS.md)
- [ ] **Introduction** — RL reward shapes encoder attention; within-episode vs pooled confound
- [ ] **Related work** — XAI for RL, attention in autonomous driving, Waymax/V-MAX
- [ ] **Method** — Perceiver encoder, token structure, attention extraction, risk metrics
- [ ] **Experiments** — 50-scenario setup, correlation analysis, lead-lag, confound check
- [ ] **Results** — write after minimal 50-scenario run completes
- [ ] **Conclusion**

### Draft paper claim (update after minimal run):
> Reward design in RL-trained autonomous driving agents predictably shapes the Perceiver
> encoder's attentional prior. Within-episode correlation between collision risk and agent
> attention is positive (ρ=+0.291, CI=[+0.125,+0.442], 80.6% of 31 high-variation scenarios
> significant), with attention leading risk by ~2 steps. Pooled cross-episode analysis
> (ρ=+0.088) confounds this 3.3×. Navigation rewards increase GPS attention 3.4×; TTC
> penalties elevate resting agent surveillance 48%. Agent-count confound is negligible (Δρ=−0.014).

---

## File Map

```
results/reward_attention/
  womd_sac_road_perceiver_complete_42/
    timestep_data.pkl          ← 50-scenario data (3,676 records)
    within_scenario_summary.csv
    scenario_ranking.csv
    fig_lead_lag_*.png          ← NEW: attention leads risk by +2 steps
    fig_rho_distribution_*.png  ← NEW: heterogeneity histogram
    fig_agent_count_confound.png← NEW: confound Δ=−0.014
    fig_timeseries_s009.png     ← counter-example (needs investigation)
    fig_timeseries_s031.png     ← counter-example (needs investigation)
  womd_sac_road_perceiver_minimal_42/
    timestep_data.pkl          ← 3-scenario only (needs 50-scenario run)
  womd_sac_road_perceiver_basic_42/
    timestep_data.pkl          ← 3-scenario only (needs 50-scenario run)

reward_attention/
  run_experiment.py            ← main GPU experiment runner
  analyze_results.py           ← CPU post-processing (enhanced, all bugs fixed)
  bev_attention.py             ← BEV visualization (needs GPU)
  correlation.py               ← CorrelationAnalyzer class

logs/
  complete_50scenarios.log     ← DONE
  minimal_50scenarios.log      ← create when running Priority 1a
  basic_50scenarios.log        ← create when running Priority 2a
```

---

## Immediate First Action When Resuming

```bash
# 1. Check if any runs are still going
ps aux | grep run_experiment | grep -v grep

# 2. Check what pkl files exist and their sizes
ls -lh results/reward_attention/*/timestep_data.pkl

# 3. If minimal 50-scenario not done, launch it (see Priority 1a above)
# 4. While it runs, investigate counter-examples (Priority 1b)
```
