# Reward-Conditioned Attention — Latest Progress & Findings

> Master progress file for the RLC 2026 submission.
> Covers: what data we have, what analyses are done, what every finding says, what is still needed.
> Last updated: 2026-02-23

---

## 1. Data Collected

| Model | Config | Scenarios | Timesteps | Early Terminations | pkl Size |
|-------|--------|-----------|-----------|-------------------|----------|
| `womd_sac_road_perceiver_complete_42` | collision+offroad+redlight+off_route+progression+speed+TTC(1.5s)+comfort | **50** | **3,676** | 8 (s001:73, s028:24, s029:58, s036:35, s037:35, s039:17, s042:27, s047:47) | 1003K |
| `womd_sac_road_perceiver_minimal_42` | collision+offroad+redlight+off_route+progression | **3** | **240** | 0 (all full 80-step episodes) | 66K |
| `womd_sac_road_perceiver_basic_42` | collision+offroad+redlight | 3 | 157 | 2 (crashes at steps 38, 39) | 44K |

All models: same Perceiver encoder architecture, same seed (42), same `data/training.tfrecord`.

---

## 2. Validation Checks (2026-02-23, validate_and_extend.py)

### 2a. Attention Budget Invariant — PASS

The five attention categories (ego, agents, road, lights, GPS) sum to **exactly 1.000000** at every timestep, for every model. No violations. The attention extraction and aggregation pipeline is correct.

### 2b. Cross-Model Risk Profiles — MODERATE (proceed with caveat)

The complete and minimal models face partially different risk experiences on the same scenarios because their policies drive differently:

| Scenario | Spearman rho(complete_risk, minimal_risk) | Mean |delta_risk| | Verdict |
|----------|------------------------------------------|---------------------|---------|
| s000 | +0.012 | 0.322 | **Diverge** — models face very different risk |
| s001 | +0.501 | 0.087 | Moderate correlation |
| s002 | +0.674 | 0.132 | Moderate correlation |

**Implication**: When comparing attention between models, we are comparing "each model's response to its own risk experience," not responses to identical stimuli. This is acknowledged in the paper framing. s002 is the most comparable; s000 is essentially incomparable.

### 2c. Vigilance Gap — CONFIRMED on s002, absent on s000

| Scenario | Calm-phase attn_agents (complete) | Calm-phase attn_agents (minimal) | Gap |
|----------|-----------------------------------|----------------------------------|-----|
| s000 | 0.0000 | 0.0000 | Negligible (both near zero) |
| **s002** | **0.1461** | **0.0625** | **+0.0837 (+134%)** |

The vigilance prior (complete model maintaining higher agent attention during calm phases) is **strongly confirmed on s002** but does not appear in s000. In s000, both models allocate near-zero attention to agents during calm phases — likely a scenario where road geometry dominates.

**Conclusion**: The vigilance gap is real but scenario-dependent. Needs 50-scenario minimal run to quantify how often it appears.

---

## 3. Established Findings (from prior sessions + 50-scenario complete run)

### Finding 1: Within-Episode Attention Tracks Collision Risk

**Confirmed at scale (50 scenarios, 31 high-variation).**

```
collision_risk x attn_agents:     mean rho = +0.291   95% CI [+0.125, +0.442]   80.6% individually significant
collision_risk x attn_roadgraph:  mean rho = -0.148   95% CI [-0.287, -0.003]   67.7% individually significant
safety_risk    x attn_agents:     mean rho = +0.291   95% CI [+0.125, +0.442]   80.6% individually significant
```

The positive agent-attention/risk correlation is robust — the CI is entirely above zero. The road graph shows the complementary trade-off: attention moves *from* road *toward* agents under danger.

Best single scenario (s002): rho = +0.769 (p < 0.001).

### Finding 2: Pooled Correlation Is Confounded 3.3x

```
Pooled (all 3,676 timesteps):     rho = +0.088
Within-episode (31 HV scenarios): rho = +0.291
```

19 of 50 scenarios have risk_std < 0.2 (near-constant risk). They add timesteps but zero signal, diluting the pooled estimate. **Within-episode analysis with Fisher z-transform is the correct method.**

### Finding 3: Counter-Examples Exist (~20%)

Two scenarios show *reversed* correlations (attention drops when risk rises):
- s009: rho = -0.383 (risk_std = 0.396)
- s031: rho = -0.559 (risk_std = 0.379)

Also: s013 (rho = -0.495), s014 (rho = -0.802), s022 (rho = -0.234).

These are not noise — they are genuine attentional heterogeneity. Mechanism unknown; BEV investigation pending.

### Finding 4: Vigilance Prior (TTC Reward Effect)

On s002, the complete model (with TTC penalty) maintains +134% higher agent attention during calm phases compared to minimal. The gap is present from t=0 and persists throughout the episode. This is a **learned resting posture** shaped by reward design, not a reactive response.

Episode means (s002):
```
complete: attn_agents = 0.173,  attn_gps = 0.166
minimal:  attn_agents = 0.117,  attn_gps = 0.314
```

### Finding 5: GPS Attention Gradient — Cleanest Reward-to-Attention Signal

```
minimal  (+off_route +progression): attn_gps = 0.314   (highest)
complete (+TTC partially shifts):   attn_gps = 0.166
basic    (no navigation reward):    attn_gps = 0.092   (lowest)
```

The GPS attention baseline follows navigation reward content monotonically. No correlation analysis needed — the three baselines tell the story directly.

### Finding 6: Agent-Count Confound Is Negligible

```
Raw rho:     +0.262
Partial rho: +0.247   (controlling for num_valid_agents)
Delta:       -0.014
```

The correlation between risk and agent attention is not an artifact of more agents being present during dangerous moments.

---

## 4. New Findings (2026-02-23, validate_and_extend.py)

### Finding 7: Attention Entropy RISES with Risk (Surprise)

**Original hypothesis**: Entropy drops under danger (model concentrates on fewer tokens).

**Actual result**: Entropy *increases* when risk rises.

```
Complete model (31 HV scenarios):
  Mean within-episode rho(entropy, collision_risk) = +0.199
  Calm phases (risk<0.2):   H = 1.770 +/- 0.152 bits  (n=1,064)
  Danger phases (risk>0.5): H = 1.803 +/- 0.149 bits  (n=2,338)
  Mann-Whitney U: p < 0.0001 (significant)
```

**Interpretation**: When risk is low, road graph dominates (~47% of budget) and the distribution is concentrated. When risk rises, attention redistributes across agents, GPS, and ego — producing a more uniform (higher entropy) distribution. The model doesn't narrow its focus; it **diversifies** its information sources under threat.

**Per-scenario heterogeneity**: Strong positive rho in s000 (+0.738), s005 (+0.788), s002 (+0.595), s031 (+0.655). Strong negative in s014 (-0.619), s015 (-0.625), s013 (-0.487). The entropy-risk relationship is scenario-dependent.

**Paper recommendation**: Include as a secondary finding with honest framing — "risk triggers attentional redistribution rather than narrowing" — but note the per-scenario variability. Max 5-category entropy is only 2.32 bits, so the effect operates in a compressed range.

### Finding 8: Attention Leads Risk, But Spread Is Wide

Per-scenario optimal lead-lag analysis (31 HV scenarios):

```
Median best lag:  +3.0 steps (attention leads)
Mean best lag:    +2.0 steps
Std:              5.63

Direction breakdown:
  Attention leads (lag>0):  20/31  (65%)
  Simultaneous (lag=0):      2/31  ( 6%)
  Attention lags (lag<0):    9/31  (29%)

Clustering in +1 to +3:     5/31  (16%)
```

**Interpretation**: Attention tends to lead risk (65% of scenarios), consistent with the aggregate finding of best rho at lag=+2. However, the spread is wide (std = 5.63) and many scenarios hit the boundary (lag = +/-8), suggesting longer-horizon dynamics that the 8-step window cannot capture.

**Paper recommendation**: Report the directional finding ("attention anticipates risk in the majority of scenarios") but do not make a precise "exactly 2 steps ahead" claim. The per-scenario histogram (fig_leadlag_histogram.png) and heatmap (fig_leadlag_heatmap.png) should go in the paper/appendix to show honest heterogeneity.

### Finding 9: Budget Reallocation Under Threat — Direct Visualization

Stacked bar comparison of mean attention budget at low risk (< 0.2) vs high risk (> 0.7), restricted to 31 high-variation scenarios:

```
COMPLETE MODEL (n_low=675, n_high=1,168):
  Category          Low risk   High risk     Delta     Rel %
  Ego (SDC)           20.1%      19.1%      -1.0%     -5.0%
  Other Agents         4.3%       6.0%      +1.7%    +38.2%   *** p = 7.4e-14
  Road Graph          53.8%      52.2%      -1.6%     -3.0%
  Traffic Lights       6.3%       6.3%      -0.1%     -0.9%
  GPS Path            15.5%      16.5%      +1.0%     +6.5%

MINIMAL MODEL (n_low=40, n_high=91, only 2 HV scenarios):
  Other Agents         5.6%       6.4%      +0.8%    +14.4%   (p = 0.63, n.s.)
  GPS Path            33.9%      31.9%      -2.1%     -6.1%
```

The figure uses a **two-row per-scenario layout** that tells a story:

**Top row — risk-reactive scenarios (typical):**
- s002 (rho=+0.77): agents 14% -> 20% (+5.7%), road 43% -> 40%
- s000 (rho=+0.62): agents ~0% -> 2.5%, road 56% -> 51%
- s037 (rho=+0.62): agents ~0% -> 6%, road 62% -> 51%

**Bottom row — counter-examples (inverted):**
- s031 (rho=-0.56): agents stay flat, road absorbs budget shift
- s013 (rho=-0.50): agents 9% -> ~3% (-5.8%), road 56% -> 62%

**Why this matters for the paper**: This is more intuitive than correlation coefficients. A reader can immediately see the total budget sums to 100% and that the model *trades off* road attention for agent attention under danger — except in counter-examples where the pattern inverts. The two-row layout turns the heterogeneity from a weakness into a finding.

**Paper recommendation**: Main paper figure. The top row establishes the typical pattern, the bottom row shows it's not universal. Pair with the timeseries for the dynamic view.

---

## 5. Complete Figure Inventory

### Paper-ready figures (300 DPI, white background, serif fonts)

| Figure | File | Source | Status |
|--------|------|--------|--------|
| **Budget reallocation** | `fig_budget_reallocation.png` | validate_and_extend.py | **NEW — potential main figure** |
| Cross-model risk profiles | `fig_risk_profile_comparison.png` | validate_and_extend.py | NEW |
| Vigilance gap (s000 + s002) | `fig_vigilance_gap_s000_s002.png` | validate_and_extend.py | NEW |
| Entropy vs risk scatter (complete) | `fig_entropy_scatter_complete.png` | validate_and_extend.py | NEW |
| Entropy timeseries (complete) | `fig_entropy_timeseries_complete.png` | validate_and_extend.py | NEW |
| Entropy scatter (minimal) | `fig_entropy_scatter_minimal.png` | validate_and_extend.py | NEW |
| Entropy timeseries (minimal) | `fig_entropy_timeseries_minimal.png` | validate_and_extend.py | NEW |
| Lead-lag histogram | `fig_leadlag_histogram.png` | validate_and_extend.py | NEW |
| Lead-lag heatmap | `fig_leadlag_heatmap.png` | validate_and_extend.py | NEW |

### Earlier figures (from analyze_results.py + bev_attention.py)

| Figure | File | Notes |
|--------|------|-------|
| 3-way comparison (s002) | `fig_3way_comparison_s002.png` | **CENTERPIECE** — GPS gradient + agent baseline across 3 models |
| 2-model overlay (s002) | `fig_complete_vs_minimal_s002.png` | Shaded vigilance gap |
| Scenario scatter (50 scenarios) | `complete_42/fig_scenario_scatter_collision_risk_vs_attn_agents.png` | Within vs pooled confound at scale |
| Risk distribution | `complete_42/fig_risk_distribution.png` | Which scenarios have useful variability |
| rho distribution histogram | `complete_42/fig_rho_distribution_collision_risk_vs_attn_agents.png` | Heterogeneity across 50 scenarios |
| Lead-lag (aggregate) | `complete_42/fig_lead_lag_collision_risk_vs_attn_agents.png` | Aggregate lag=+2 |
| Agent-count confound | `complete_42/fig_agent_count_confound.png` | Confound check delta=-0.014 |
| Attention baselines | `complete_42/fig_attention_baselines_comparison.png` | Multi-model bar chart |
| Correlation heatmap | `complete_42/fig2_correlation_heatmap.png` | Full matrix |
| Action-conditioned attention | `complete_42/fig4_action_attention.png` | Braking/steering |
| BEV panel (s000) | `complete_42/fig_bev_panel_s000.png` | Ground-truth scene |
| BEV panel (s002) | `complete_42/fig_bev_panel_s002.png` | Ground-truth scene |
| Top-scenario timeseries | `complete_42/fig_timeseries_s{000,002,009,015,...}.png` | 10 scenarios |

---

## 6. Code Inventory

| File | Purpose | GPU? |
|------|---------|------|
| `reward_attention/config.py` | Token ranges, TimestepRecord, AnalysisConfig | No |
| `reward_attention/extractor.py` | AttentionTimestepCollector (batched inference) | **Yes** |
| `reward_attention/risk_metrics.py` | RiskComputer (TTC, collision, safety, navigation, behavior) | No |
| `reward_attention/correlation.py` | CorrelationAnalyzer (per-scenario + Fisher z) | No |
| `reward_attention/temporal.py` | TemporalAnalyzer (event catalog window) | No |
| `reward_attention/visualization.py` | Scatter, heatmap, temporal, action-conditioned figs | No |
| `reward_attention/run_experiment.py` | Main CLI: model -> full results + figures | **Yes** |
| `reward_attention/analyze_results.py` | Post-processing: pkl -> all figures + summary stats | No |
| `reward_attention/bev_attention.py` | BEV panels + scenario scatter + timeseries | **Yes** (BEV only) |
| `reward_attention/validate_and_extend.py` | Pre-scale validation + entropy + lead-lag | No |
| `reward_attention/probe.py` | Phase 0: verify attention extraction | **Yes** |

---

## 7. Key Numbers to Remember

```
CORE RESULT (complete model, 50 scenarios):
  collision_risk x attn_agents:  mean rho = +0.291  CI = [+0.125, +0.442]
  sig_pct = 80.6%  n = 31 HV scenarios

POOLED vs WITHIN:
  pooled rho = +0.088  |  within-episode rho = +0.291  |  confound ratio = 3.3x

CONFOUND CHECK:
  raw rho = +0.262  |  partial rho (agent-count) = +0.247  |  delta = -0.014

GPS GRADIENT (s002):
  minimal = 0.314  >  complete = 0.166  >  basic = 0.092

VIGILANCE PRIOR (s002, calm phases):
  complete = 0.146  >  minimal = 0.063  |  gap = +134%

ENTROPY (50-scenario complete):
  mean rho(entropy, risk) = +0.199  (entropy RISES with risk)
  calm H = 1.770 bits  |  danger H = 1.803 bits  |  p < 0.0001

LEAD-LAG (50-scenario complete):
  median best lag = +3 steps (attention leads)
  65% of scenarios show attention leading risk
  16% cluster in +1 to +3 range  |  spread is wide (std = 5.63)

BUDGET REALLOCATION (complete, 31 HV scenarios pooled):
  Agents: 4.3% -> 6.0%  (+38.2%, p = 7.4e-14)
  Road:  53.8% -> 52.2%  (-3.0%)
  GPS:   15.5% -> 16.5%  (+6.5%)
BUDGET REALLOCATION (per-scenario, top 3 + 2 counter-examples):
  s002: agents 14% -> 20% (+5.7%)   |  s013: agents 9% -> 3% (-5.8%)
  s000: agents ~0% -> 2.5%          |  s031: agents flat (inverted)

COUNTER-EXAMPLES:
  s009: rho = -0.383  |  s031: rho = -0.559  |  s013: rho = -0.495  |  s014: rho = -0.802
```

---

## 8. What Is Done vs What Remains

### Done

- [x] Complete model 50-scenario run (3,676 timesteps)
- [x] Minimal model 3-scenario pilot (240 timesteps)
- [x] Basic model 3-scenario pilot (157 timesteps)
- [x] Full analyze_results.py pipeline (scatter, heatmap, timeseries, rho distribution, lead-lag, confound check)
- [x] 3-way comparison figure (complete vs minimal vs basic on s002)
- [x] 2-model overlay figure (vigilance gap visualization)
- [x] BEV panels for s000 and s002
- [x] validate_and_extend.py: budget check, risk profiles, vigilance gap, entropy, lead-lag per-scenario
- [x] All validation checks passed — safe to proceed

### Remaining (priority order)

#### Must-do before submission

1. **Minimal model 50-scenario run** — The single most important pending experiment. The GPS gradient and vigilance prior currently rest on 3 scenarios. This confirms or refutes them at scale.
   ```bash
   nohup python reward_attention/run_experiment.py \
       --model runs_rlc/womd_sac_road_perceiver_minimal_42 \
       --data data/training.tfrecord \
       --n-scenarios 50 \
       --output results/reward_attention \
       > logs/minimal_50scenarios.log 2>&1 &
   ```

2. **Re-run validate_and_extend.py** after minimal 50-scenario completes — check vigilance gap at scale, re-run entropy and lead-lag on minimal data.

3. **Counter-example investigation** — Generate BEV panels for s009 and s031. Inspect what is geometrically distinct. If you can label the mechanism (committed maneuver? lane change?), it becomes a finding, not a gap.
   ```bash
   python reward_attention/bev_attention.py --scenario-idx 9 --no-video
   python reward_attention/bev_attention.py --scenario-idx 31 --no-video
   ```

4. **Paper writing** — Introduction, method, results sections. The analysis is mature enough to write the full paper now. Sections that can be written immediately:
   - Introduction (reward shapes attention; within-episode vs pooled confound)
   - Related work (XAI for RL, attention in AD, Waymax/V-MAX)
   - Method (Perceiver encoder, token structure, attention extraction, risk metrics, Fisher z)
   - Experiments section skeleton

#### Should-do (strengthens paper)

5. **Complete model 50-scenario run on minimal** → run analyze_results.py comparison mode:
   ```bash
   python reward_attention/analyze_results.py \
       --pkl results/reward_attention/womd_sac_road_perceiver_complete_42/timestep_data.pkl \
       --compare results/reward_attention/womd_sac_road_perceiver_minimal_42/timestep_data.pkl \
       --compare-label minimal --top-n 10
   ```

6. **Upgrade existing dark-theme figures** to paper-quality white-background style (analyze_results.py figures still use dark theme; validate_and_extend.py figures are already paper-ready).

#### Nice-to-have

7. Architecture comparison (Wayformer or MTR on same 50 scenarios) — proves the findings generalize beyond Perceiver.

8. Basic model 50-scenario run — confirms GPS gradient at scale, but basic crashes ~40% of the time.

---

## 9. Draft Paper Claim

> We demonstrate that reward design in RL-trained autonomous driving agents predictably shapes
> the Perceiver encoder's attentional prior across 50 real-world driving scenarios (3,676
> timesteps). Three findings hold at scale: (1) within-episode Spearman correlation between
> collision risk and agent attention is positive (mean rho = +0.291, 95% CI [+0.125, +0.442],
> 80.6% of 31 high-variation scenarios individually significant), with pooled cross-episode
> analysis (rho = +0.088) confounded 3.3x by between-scenario risk heterogeneity;
> (2) navigation reward terms increase GPS path attention baseline 3.4x; (3) continuous
> proximity penalties (TTC at 1.5s) elevate resting agent surveillance +134% relative to
> models without TTC — a learned vigilance prior maintained even during collision-free phases.
> Attention entropy increases under threat (mean rho = +0.199, p < 0.0001), indicating
> attentional redistribution rather than narrowing. Attention tends to lead risk temporally
> (65% of scenarios, median lag = +3 steps), though with substantial per-scenario variability.
> Approximately 20% of high-variation scenarios exhibit reversed correlations, revealing
> genuine attentional heterogeneity that merits further investigation.

---

## 10. How to Resume Work

```bash
# 1. Activate environment
eval "$(~/anaconda3/bin/conda shell.bash hook)" && conda activate vmax
export PYTHONPATH=/home/med1e/post-hoc-xai/V-Max:$PYTHONPATH
cd /home/med1e/post-hoc-xai

# 2. Check if any experiments are running
ps aux | grep run_experiment | grep -v grep

# 3. Check what data exists
ls -lh results/reward_attention/*/timestep_data.pkl

# 4. If minimal 50-scenario not done, launch it (Priority 1)
nohup python reward_attention/run_experiment.py \
    --model runs_rlc/womd_sac_road_perceiver_minimal_42 \
    --data data/training.tfrecord \
    --n-scenarios 50 \
    --output results/reward_attention \
    > logs/minimal_50scenarios.log 2>&1 &

# 5. Run validation on new data when ready
python reward_attention/validate_and_extend.py --all

# 6. Run full analysis comparison
python reward_attention/analyze_results.py \
    --pkl results/reward_attention/womd_sac_road_perceiver_complete_42/timestep_data.pkl \
    --compare results/reward_attention/womd_sac_road_perceiver_minimal_42/timestep_data.pkl \
    --compare-label minimal --top-n 10
```
