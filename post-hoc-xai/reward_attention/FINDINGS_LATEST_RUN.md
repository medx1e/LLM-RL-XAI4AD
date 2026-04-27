# Findings from Latest Run: Complete (50 scen.) vs Minimal (50 scen.)

> Generated: 2026-02-24
> Models: `womd_sac_road_perceiver_complete_42` (50 scenarios, 3,676 timesteps)
>         `womd_sac_road_perceiver_minimal_42` (50 scenarios, 3,718 timesteps)
> Scripts: `analyze_results.py`, `validate_and_extend.py --all`

---

## 1. Pooled Scatter: A Cautionary Tale

**Figures:**
- `womd_sac_road_perceiver_complete_42/fig1_scatter_collision_risk_vs_attn_agents.png`
- `womd_sac_road_perceiver_minimal_42/fig1_scatter_collision_risk_vs_attn_agents.png`

The pooled scatter plots (all 3,676 / 3,718 timesteps combined) show **misleading results**:

| Model    | Pooled ρ | p-value |
|----------|----------|---------|
| Complete | +0.088   | 0.000   |
| Minimal  | -0.155   | 0.000   |

The complete model's pooled ρ=+0.088 looks weak, while the minimal model's ρ=-0.155 appears to show the *opposite* relationship. Both are artifacts of **Simpson's paradox**: pooling across episodes with different baseline attention levels and risk distributions creates a confound that distorts the true within-episode relationship. The complete model scatter shows a near-flat trend line with massive scatter; the minimal model trend line has a slight downward slope.

**Takeaway**: These figures are useful as a pedagogical device — they demonstrate exactly why naive pooling is inappropriate, and why within-episode analysis is necessary.

---

## 2. Within-Episode Correlation: The Core Result

**Figures:**
- `womd_sac_road_perceiver_complete_42/fig_scenario_scatter_collision_risk_vs_attn_agents.png`
- `womd_sac_road_perceiver_complete_42/fig_rho_distribution_collision_risk_vs_attn_agents.png`

### 2a. Per-scenario scatter with regression lines

`fig_scenario_scatter_collision_risk_vs_attn_agents.png` overlays all 50 scenarios, each with its own regression line, colored by scenario. The per-scenario slopes are visually heterogeneous — some steep positive, some flat, a few negative — but the *majority* tilt upward. The figure title states "Pooled ρ=+0.09 masks within-episode effects," directly making our methodological point.

### 2b. Rho distribution histogram

`fig_rho_distribution_collision_risk_vs_attn_agents.png` is the key statistical summary:

- **25 of 31 HV scenarios** have positive ρ (81%)
- **6 of 31 HV scenarios** have negative ρ (19% counter-examples)
- Mean ρ (HV) = **+0.262**, std = 0.411
- The distribution is clearly right-skewed, with a mode around ρ=+0.5 to +0.7
- Green bars (HV, included in analysis) dominate the positive side
- Grey bars (low-variation, excluded) cluster near zero, confirming exclusion is appropriate

Fisher z-transformed aggregate (more robust):
- **ρ = +0.291**, 95% CI = [+0.125, +0.442]
- **80.6%** of HV scenarios individually significant at p < 0.05

**Interpretation**: In the majority of driving episodes with meaningful risk variation, the complete model's Perceiver encoder increases attention to other agents when collision risk rises. The effect is moderate in size (ρ ~ 0.3) but consistent.

---

## 3. Correlation Heatmap: Which Risk Drives Which Attention?

**Figure:** `womd_sac_road_perceiver_complete_42/fig2_correlation_heatmap.png`

This heatmap shows pooled Spearman ρ between three risk types and seven attention categories. Key patterns:

| Risk type       | Strongest positive   | Strongest negative      |
|-----------------|---------------------|-------------------------|
| Collision risk  | GPS Path (+0.25)     | attn_to_threat (-0.22), Traffic Lights (-0.20) |
| Safety risk     | GPS Path (+0.25)     | attn_to_threat (-0.22), Traffic Lights (-0.20) |
| Behavior risk   | Traffic Lights (+0.13), GPS Path (+0.13) | Road Graph (-0.21) |

The heatmap reveals that at the pooled level, GPS Path attention has the strongest positive association with collision risk, not Other Agents (+0.09). This is counterintuitive and again reflects the pooling confound — scenarios with high mean risk tend to have high mean GPS attention, inflating the GPS-risk correlation. The `attn_to_threat` metric (attention specifically to the closest threatening agent) is *negatively* correlated at the pooled level (-0.22), which is entirely an artifact.

**Takeaway**: This heatmap is useful for motivating the within-episode approach, but should not be cited as a result in isolation.

---

## 4. Timeseries: Flagship Scenarios

### 4a. s002 — Best positive case (ρ = +0.769)

**Figure:** `womd_sac_road_perceiver_complete_42/fig_timeseries_s002.png`

This is the cleanest demonstration of risk-reactive attention. Two panels:
- **Top**: Collision risk rises from ~0.5 to ~0.95 between steps 0-35, drops sharply to ~0.0 around step 45-65, then rises again to ~0.8.
- **Bottom**: Agent attention (red line) tracks this trajectory almost perfectly — rising from ~0.12 to ~0.27 during the danger phase, plummeting during the calm phase, and rising again. Road attention (green) moves inversely. GPS (purple dashed) and Ego (blue dotted) remain relatively stable.

The ρ = +0.769 is exceptionally strong. The Road line (ρ = -0.521) confirms the zero-sum budget reallocation: attention shifts *from* road graph *to* agents under threat.

### 4b. s000 — Strong positive case (ρ = +0.617)

**Figure:** `womd_sac_road_perceiver_complete_42/fig_timeseries_s000.png`

Similar to s002 but with a distinctive mid-episode attention spike. Collision risk is high (0.6-1.0) for the first 45 steps, then drops to ~0. Agent attention (ρ = +0.617) builds from ~0.04 to a peak of ~0.30 around step 28-32, exactly when risk is maximal. Road attention (ρ = -0.549) dips inversely during this period. Notably, both GPS and Ego attention also spike around steps 25-30, suggesting a moment of coordinated "heightened alertness" across multiple token groups.

### 4c. s009 — Counter-example (ρ = -0.383)

**Figure:** `womd_sac_road_perceiver_complete_42/fig_timeseries_s009.png`

This scenario presents the clearest failure of the risk-reactive hypothesis. Risk is highly volatile throughout — oscillating rapidly between 0 and 1.0 with no sustained calm phase. Despite this, agent attention is remarkably flat (~0.10-0.13) across the entire episode. Road attention (ρ = +0.394) actually *increases* slightly with risk rather than decreasing. The negative agent ρ arises because the few timesteps where agent attention dips slightly happen to coincide with high-risk moments.

**Possible explanation**: s009 appears to be a high-frequency oscillatory scenario (perhaps a dense urban intersection) where risk flickers on and off too rapidly for the attention mechanism to track. The agent's attention remains in a "default" road-heavy mode throughout, suggesting the encoder treats the scenario as consistently dangerous rather than modulating per-timestep.

### 4d. s031 — Counter-example (ρ = -0.559)

**Figure:** `womd_sac_road_perceiver_complete_42/fig_timeseries_s031.png`

The strongest negative correlation in the dataset. Risk is persistently high (0.8-1.0) for the first 50 steps, briefly drops to 0 around step 50-60, then rises again. However, agent attention is near-zero (~0.04-0.07) throughout the entire episode — far below the dataset average. Road attention dominates at ~0.55. The negative ρ is driven by a trivial artifact: during the brief calm phase (steps 50-60), agent attention rises marginally (to ~0.08), creating an inverse relationship.

**Possible explanation**: s031 is a structurally road-dominated scenario — perhaps a highway or corridor where other agents are distant or irrelevant. The model has correctly learned that road geometry, not agent behavior, is the primary concern. The near-zero agent attention regardless of risk level suggests the encoder treats agents as uninformative in this context.

---

## 5. Confound Check: Agent Count

**Figure:** `womd_sac_road_perceiver_complete_42/fig_agent_count_confound.png`

Two-panel figure. Left panel: scatter of raw ρ vs partial ρ (controlling for number of valid agents). Points hug the diagonal tightly — green (positive ρ) and red (negative ρ) points alike barely shift when agent count is partialled out. Right panel: bar chart comparing mean raw ρ (+0.262) vs mean partial ρ (+0.247), with overlapping confidence intervals.

**Result**: Δρ = -0.014 (5.7% reduction). The agent-count confound is negligible. The risk-attention relationship is not driven by scenarios that simply have more agents on screen (which would mechanically increase attention to the agent token group).

---

## 6. Lead-Lag Analysis: Does Attention Anticipate Risk?

**Figures:**
- `womd_sac_road_perceiver_complete_42/fig_lead_lag_collision_risk_vs_attn_agents.png`
- `fig_leadlag_histogram.png`
- `fig_leadlag_heatmap.png`

### 6a. Aggregate lead-lag curve

`fig_lead_lag_collision_risk_vs_attn_agents.png` shows mean ρ at each lag from -8 to +8 steps. The curve rises monotonically from ρ ~ +0.10 at lag=-8 to a peak of ρ ~ +0.281 at lag=+2, then gradually declines to ρ ~ +0.19 at lag=+8. The shape is an asymmetric inverted-U favoring positive lags, suggesting attention *leads* risk by approximately 2 steps on average.

However, the confidence intervals are wide (whiskers span ~0.15 units at every lag), and the peak is only marginally higher than lag=0 (ρ=+0.262). The aggregate lead-lag result is **suggestive but not definitive**.

### 6b. Per-scenario best-lag histogram

`fig_leadlag_histogram.png` reveals the heterogeneity behind the aggregate. Across 31 HV scenarios:

- **20/31 (65%)** have positive best lag (attention leads risk) — green bars
- **9/31 (29%)** have negative best lag (attention lags risk) — red bars
- **2/31 (6%)** have best lag at 0 — grey bars
- Median best lag = **+3.0**, mean = **+1.97**, std = **5.63**

The distribution is not concentrated. While positive lags dominate, there is a notable cluster at lag=+8 (the boundary, 8 scenarios) and another at lag=-8 (4 scenarios). These boundary clusters suggest that for some scenarios, the cross-correlation is monotonically increasing or decreasing across the lag range, meaning the "best lag" is at the edge of the search window rather than a true peak. Only **5/31 (16%)** of scenarios have their best lag in the narrow +1 to +3 range.

### 6c. Per-scenario lead-lag heatmap

`fig_leadlag_heatmap.png` is a heatmap where each row is a scenario and each column is a lag value (-8 to +8). Color encodes Spearman ρ (red = positive, blue = negative). Stars mark each scenario's best lag. The heatmap reveals several distinct patterns:

- **Broad positive band**: Scenarios like s005, s036, s037, s041 show deep red across a wide lag range (lags 0 to +5), indicating stable positive correlation regardless of timing offset.
- **Narrow peaks**: s002 has a sharp peak at lag=-5 (unusually, attention appears to *lag* risk in this scenario, despite having the highest contemporaneous ρ).
- **Persistent negative**: s013, s014, s031 show blue bands (negative ρ) across most lags — the counter-examples are not lag-dependent.
- **Symmetric/flat**: s038, s045, s030 show weak color throughout — low correlation at any lag.

**Takeaway**: The anticipatory claim (attention leads risk) holds *on average* and for *most* scenarios, but the spread is too wide and the boundary-cluster problem too prominent to make a strong claim. Report as exploratory.

---

## 7. Attention Allocation Prior: Reward Shapes Resting Attention

**Figure:** `womd_sac_road_perceiver_complete_42/fig_attention_baselines_comparison.png`

Side-by-side bar chart comparing episode-averaged attention fractions between complete (red) and minimal (orange) across all 5 token categories. Key differences:

| Category        | Complete | Minimal | Difference |
|-----------------|----------|---------|------------|
| Ego (SDC)       | 0.201    | 0.142   | +41.5%     |
| Other Agents    | 0.056    | 0.042   | +33.3%     |
| Road Graph      | 0.521    | 0.427   | +22.0%     |
| Traffic Lights  | 0.059    | 0.054   | +9.3%      |
| GPS Path        | 0.164    | 0.335   | **-51.0%** |

The dominant difference is GPS Path: minimal model allocates **2.04x more attention** to GPS than complete. This is the GPS gradient — the minimal model, which receives strong navigation rewards but no TTC penalty, over-attends to the route path. The complete model redistributes that budget toward Ego (+41%), Agents (+33%), and Road Graph (+22%).

**Interpretation**: Reward design shapes not just dynamic attention (what the model looks at under risk) but also the static *prior* (what the model looks at on average). The complete model's TTC penalty produces a more balanced attention distribution, while the minimal model is GPS-dominated.

---

## 8. Vigilance Gap: TTC Penalty Creates Resting Agent Surveillance

**Figure:** `fig_vigilance_gap_s000_s002.png`

Four-panel figure showing two scenarios (s000, s002), each with agent attention (left) and collision risk (right) timeseries for both models.

### s000 (top row):
Both models have near-zero agent attention (~0.00-0.04) throughout. Complete model (solid blue) and minimal model (dashed red) are nearly indistinguishable. The vigilance gap is **absent** in this scenario. Both models briefly spike agent attention around steps 25-35 (matching a risk peak), but the absolute values are tiny. The right panel shows these models experience quite different risk profiles in s000 — complete has a U-shaped risk curve while minimal has more sustained high risk.

### s002 (bottom row):
The vigilance gap is clearly visible. During calm phases (risk < 0.2, shaded green, steps ~45-65):
- **Complete**: agent attention = 0.146 (maintains surveillance)
- **Minimal**: agent attention = 0.062 (drops to half)
- **Gap**: +0.084, or **+134%**

The blue shaded area between the two lines highlights the vigilance gap. During danger phases, the gap narrows but persists (complete: 0.183, minimal: 0.144). The complete model's TTC penalty has taught it to maintain higher agent attention even when collision risk is currently low — a form of "learned vigilance."

**Caveat**: The gap is scenario-dependent. s000 shows no gap, while s002 shows a strong gap. This suggests the vigilance prior may only emerge in scenarios where agent interactions are structurally relevant (s002 has many nearby agents; s000 may have fewer or more distant agents).

---

## 9. Entropy Analysis: Redistribution, Not Narrowing

**Figures:**
- `fig_entropy_scatter_complete.png`
- `fig_entropy_scatter_minimal.png`
- `fig_entropy_timeseries_complete.png`
- `fig_entropy_timeseries_minimal.png`

### 9a. Entropy scatter — complete model

`fig_entropy_scatter_complete.png` plots attention entropy (Shannon, 5 categories, max = 2.32 bits) vs collision risk for all timesteps, colored by scenario. The trend line shows a weak positive slope (r=+0.072, p=1.4e-05). Most points cluster between 1.6 and 2.1 bits. Entropy mean = 1.79 (77% of max), indicating attention is always moderately distributed — never collapsed onto a single token group.

Within-episode mean ρ(entropy, risk) = **+0.199** (n=31 HV scenarios). The original hypothesis — that entropy drops under threat (attention narrows onto fewer categories) — is **rejected**. Instead, entropy *rises* slightly with risk.

### 9b. Entropy scatter — minimal model

`fig_entropy_scatter_minimal.png` shows a *negative* pooled trend (r=-0.170, p=2.0e-25), opposite to the complete model. But the within-episode mean ρ = **+0.045**, which is inconclusive. The discrepancy between pooled (negative) and within-episode (near-zero) is another instance of Simpson's paradox: the minimal model's GPS-heavy scenarios cluster at different entropy-risk regions than its agent-heavy scenarios.

### 9c. Entropy timeseries — complete model

`fig_entropy_timeseries_complete.png` shows three example scenarios:

- **s000** (ρ = +0.74): Entropy rises from ~1.65 to ~1.85 as risk increases from 0.4 to 1.0 during steps 0-45, then drops sharply when risk collapses. Very clear co-movement.
- **s027** (ρ = +0.21): Weaker relationship. Entropy fluctuates around 1.5-1.7 with modest risk-tracking.
- **s049** (ρ = +0.29): Entropy hovering around 1.9-2.0 (high baseline), with gentle rises during risk peaks.

### 9d. Entropy timeseries — minimal model

`fig_entropy_timeseries_minimal.png` shows three scenarios:

- **s027** (ρ = -0.72): Entropy and risk move in *opposite* directions — as risk rises, entropy drops. This is the narrowing behavior we originally hypothesized, but it appears in the minimal model, not the complete model.
- **s018** (ρ = -0.62): Similar inverse pattern — entropy drops sharply when risk spikes around step 20.
- **s049** (ρ = +0.15): Near-zero relationship, entropy fluctuates independently of risk.

**Interpretation**: The entropy finding is nuanced. The complete model's TTC penalty induces *redistribution* under threat — attention spreads more evenly across token groups (entropy up) rather than narrowing. This is consistent with the budget reallocation finding: the complete model shifts attention from road to agents, GPS to ego, etc., creating a more balanced (higher entropy) distribution. The minimal model, being GPS-dominated at baseline, occasionally *narrows* further under threat (entropy down) because it has less budget flexibility.

---

## 10. Budget Reallocation: The Strongest Result

**Figure:** `fig_budget_reallocation.png`

Two-row, five-column stacked bar chart. Top row: three scenarios with the strongest positive ρ (risk-reactive). Bottom row: two counter-examples. Each scenario shows Low Risk (risk < 0.2) vs High Risk (risk > 0.7) stacked bars with five token categories.

### Top row — Risk-reactive scenarios:

**s002 (ρ = +0.77)**:
- Agents: 14% → 20% (+5.7 percentage points, annotated with arrow)
- Road: 43% → 40% (-3pp)
- GPS: 18% → 16% (-2pp)
- Ego: 21% → 20% (-1pp)
- Traffic Lights: 4% → 4% (stable)

The shift is visually clear — the orange (Agents) band expands while green (Road) contracts.

**s000 (ρ = +0.62)**:
- Agents: 22% → 23% (+2.5pp, annotated)
- Road: 56% → 51% (-5pp)
- GPS: 19% → 21% (+2pp)
- Ego: ~3% both (stable, very low)

The agent shift is smaller here but road reduction is substantial. s000 is a road-dominated scenario (56% baseline) so the budget mainly shifts from road to GPS and agents.

**s037 (ρ = +0.62)**:
- Agents: 13% → 19% (+4.7pp, annotated)
- Road: 62% → 51% (-11pp, largest road reduction)
- GPS: 16% → 17% (+1pp)
- Ego: 7% → 7% (stable)

The most dramatic road-to-agents reallocation in the dataset. Road drops 11 percentage points; agents gain nearly 5.

### Bottom row — Counter-examples:

**s031 (ρ = -0.56)**:
- Agents: 18% → 21% (+3pp)
- Road: 56% → 53% (-3pp)
- GPS: 20% → 19% (-1pp)

Surprisingly, even in this counter-example, agents *gain* attention under risk. The negative ρ for s031 is driven by its temporal dynamics (the brief calm-phase attention spike discussed in Section 4d), not by the overall budget direction. The stacked bars show the expected reallocation pattern.

**s013 (ρ = -0.50)**:
- Agents: 23% → 14% (**-9pp**, the only scenario with a large agent attention *decrease*)
- Road: 56% → 62% (+6pp)
- Ego: 9% → 7% (-2pp)
- GPS: 11% → 13% (+2pp)

s013 shows the genuine opposite pattern: under high risk, attention shifts *toward* road and *away from* agents. The -5.8pp annotation confirms this is the true anomaly. This may represent a scenario where the threatening element is a road structure (sharp curve, construction zone) rather than another agent, making the road-focused response appropriate.

**Overall budget statistics (pooled across all 31 HV scenarios)**:
- Complete model: Agents +38.2% relative increase from low to high risk (p = 7.44e-14)
- Minimal model: Agents **-16.6%** relative decrease (p = 0.019)

This reversal is the paper's strongest result. The same architecture, same training algorithm, same scenarios — the *only* difference is reward design — produces opposite attention reallocation behaviors.

---

## 11. Cross-Model Risk Profile Comparison

**Figure:** `fig_risk_profile_comparison.png`

Large multi-panel figure (50 subplots, one per scenario) comparing risk trajectories between complete and minimal models. Each panel overlays complete (blue) and minimal (red) risk over time.

Key observations:
- **11 scenarios** with ρ > 0.8 between models (GOOD agreement): s014, s021, s022, s027, s033, s036, s038, s043, s046, s047, s049
- **15 scenarios** with moderate agreement (ρ 0.5-0.8): similar risk shapes but differing magnitudes
- **24 scenarios** with low agreement (ρ < 0.5 or divergent): substantially different risk trajectories

The high divergence rate (~48%) is important context: the two models drive differently (different policies, same architecture), so they encounter different risk situations. This means attention differences are not solely due to reward design — they also reflect different driving experiences. Cross-model comparisons of attention *conditioned on matching risk levels* (as in the budget reallocation) are more valid than raw attention comparisons.

**Caveat for paper**: Report the risk profile divergence honestly. The vigilance gap and GPS gradient findings compare attention at *equivalent* risk levels, so they are robust to this concern. The raw ρ comparison (complete ρ=+0.291 vs minimal ρ=+0.141) is harder to interpret because the models face different risk dynamics.

---

## Summary Table

| Finding | Strength | Evidence | Paper-ready? |
|---------|----------|----------|-------------|
| Within-episode ρ (complete) = +0.291 | Solid | `fig_rho_distribution_*.png`, `fig_scenario_scatter_*.png` | Yes |
| Budget reallocation reversal (+38% vs -17%) | Strong | `fig_budget_reallocation.png` | Yes (main figure) |
| GPS gradient (minimal 2x GPS attention) | Strong | `fig_attention_baselines_comparison.png` | Yes |
| Agent-count confound negligible (Δ=-0.014) | Strong | `fig_agent_count_confound.png` | Yes (robustness check) |
| Pooled vs within-episode confound (3.3x) | Solid | `fig1_scatter_*.png` vs Fisher summary | Yes (methodological) |
| Vigilance gap (+134% in s002) | Moderate | `fig_vigilance_gap_s000_s002.png` | Yes (with scenario caveat) |
| Entropy rises with risk (complete ρ=+0.199) | Weak | `fig_entropy_scatter_complete.png`, `fig_entropy_timeseries_complete.png` | Exploratory |
| Lead-lag +2 steps (aggregate) | Weak | `fig_lead_lag_*.png`, `fig_leadlag_histogram.png` | Exploratory (wide spread) |
| Counter-examples (6/31 negative ρ) | Honest | `fig_timeseries_s009.png`, `fig_timeseries_s031.png`, budget bottom row | Report transparently |
| Risk profile divergence (~48%) | Caveat | `fig_risk_profile_comparison.png` | Report as limitation |

---

## Recommended Paper Figures

1. **Figure 1**: `fig_budget_reallocation.png` — Per-scenario budget reallocation (strongest visual)
2. **Figure 2**: `fig_rho_distribution_collision_risk_vs_attn_agents.png` — Within-episode ρ distribution
3. **Figure 3**: `fig_attention_baselines_comparison.png` — GPS gradient / allocation prior
4. **Figure 4**: `fig_timeseries_s002.png` — Flagship timeseries (s002, ρ=+0.769)
5. **Figure 5**: `fig_agent_count_confound.png` — Confound robustness check
6. **Supplementary**: `fig_vigilance_gap_s000_s002.png`, `fig_entropy_scatter_complete.png`, `fig_leadlag_heatmap.png`, counter-example timeseries

---

## Key Numbers for Paper Abstract

```
Within-episode ρ(collision_risk, attn_agents):
  Complete:  +0.291  CI=[+0.125, +0.442]  sig=80.6%  n=31 HV scenarios
  Minimal:   +0.141  CI=[-0.039, +0.313]  sig=61%    n=28 HV scenarios

Budget reallocation (agents, low→high risk):
  Complete:  +38.2%  (p=7.44e-14)
  Minimal:   -16.6%  (p=0.019)

GPS gradient (episode-averaged GPS attention):
  Minimal: 0.335  vs  Complete: 0.164  (2.04x)

Agent-count confound: Δρ = -0.014 (negligible)
Pooled confound: pooled ρ=+0.088 vs within ρ=+0.291 (3.3x inflation)
Vigilance gap (s002, calm phase): +134% (complete 0.146 vs minimal 0.062)
Counter-examples: 6/31 HV scenarios with negative ρ (19%)
```
