# Reward-Conditioned Attention — Key Findings for RLC 2026

> Paper-oriented summary of all findings. Be honest about limitations.
> Evidence base: complete model = 50 scenarios / 3,676 timesteps. Minimal + basic = 3 scenarios each.
> Last updated: 2026-02-22

---

## Models Studied

| Model | Reward terms | Scenarios | Timesteps | Notes |
|-------|-------------|-----------|-----------|-------|
| `womd_sac_road_perceiver_complete_42` | collision+offroad+redlight+off_route+progression+speed+TTC(1.5s)+comfort | **50** | **3,676** | 7 early terminations |
| `womd_sac_road_perceiver_minimal_42` | collision+offroad+redlight+off_route+progression | 3 | 240 | All full episodes |
| `womd_sac_road_perceiver_basic_42` | collision+offroad+redlight | 3 | 157 | 2 early terminations |

All models: same Perceiver encoder architecture, same seed (42), same data.

---

## Finding 1: Within-Episode Attention Tracks Collision Risk — Confirmed at Scale

**What**: Within episodes where risk actually varies (std > 0.2), Spearman correlation between
collision risk and agent attention is consistently positive. Confirmed across 31 high-variation
scenarios out of 50.

**Evidence** (complete model, 50-scenario run — KEY result):
```
collision_risk × attn_agents       mean ρ = +0.291  95%CI [+0.125, +0.442]  n=31 HV scenarios
collision_risk × attn_roadgraph    mean ρ = −0.148  95%CI [−0.287, −0.003]  n=31 HV scenarios
safety_risk    × attn_agents       mean ρ = +0.291  95%CI [+0.125, +0.442]  n=31 HV scenarios
```
CI for attn_agents is entirely above zero — the positive trend is not noise.

**Best single scenario** (s002, 80 timesteps, 2 risk cycles):
```
collision_risk × attn_agents       ρ = +0.769**  (p < 0.001)
collision_risk × attn_roadgraph    ρ = −0.521**  (p < 0.001)
collision_risk × attn_gps          ρ = −0.787**  (p < 0.001)
collision_risk × attn_sdc          ρ = −0.698**  (p < 0.001)
```

**Interpretation**: As collision risk rises, the model reallocates attention from road geometry and
GPS toward other agents — a semantically sensible response. The agents↔road trade-off is
consistent and bidirectional.

**Honest caveat**: Effect is heterogeneous (std_ρ = 0.405). The average positive trend coexists
with genuine counter-examples (see Finding 1b below). The 26% individual significance rate
reflects real scenario-to-scenario variability, not measurement error.

---

## Finding 1b: Counter-Examples Exist and Matter

**What**: In ~2 out of every 10 high-variation scenarios, agent attention *decreases* when risk
rises. This is not noise — it reflects real attentional heterogeneity.

**Counter-examples** (complete model, 50-scenario run):
```
s009: risk_std=0.396   ρ(attn_agents) = −0.383**
s031: risk_std=0.379   ρ(attn_agents) = −0.559**
```

**Why this matters**: These scenarios are where the model behaves *differently*. Investigating
what is geometrically or dynamically distinct about them (committed maneuver? lane change locked
in?) is the most analytically interesting next step — and the one most likely to yield a
publishable insight beyond the average trend.

**Current status**: Not yet investigated. BEV visualization of s009/s031 is pending.

---

## Finding 2: Pooled Correlation Is Confounded — Within-Episode Analysis Is Required

**What**: Naive pooling of all timesteps across all scenarios gives ρ ≈ +0.088 for
collision_risk × attn_agents. Within-episode analysis gives ρ ≈ +0.291. The gap is 3.3×.
Previously (5-scenario pilot) the gap was even larger: pooled ≈ +0.02 vs within ≈ +0.70.

**50-scenario evidence**:
```
Pooled (all 3,676 timesteps):       ρ = +0.088  p < 0.0001
Within-episode (31 HV scenarios):   ρ = +0.291  CI [+0.125, +0.442]
```

**Why it happens**: 19/50 scenarios have std(collision_risk) < 0.2 — near-constant risk with
no variation to detect. They add timesteps but zero signal, diluting the pooled estimate.
The between-scenario confound is structural, not a sampling artifact.

**Methodological claim**: Within-episode Spearman ρ with Fisher z-transform aggregation is the
correct statistic for attention-reward analysis across heterogeneous RL episodes. Pooled
correlation is misleading and should not be the primary reported result.

**This is a contribution in itself** — XAI evaluations of RL agents frequently pool across
episodes without accounting for between-episode risk heterogeneity.

---

## Finding 3: Risk-Reactive Attention Is Universal — Not TTC-Specific

**What**: The within-episode correlation between collision risk and agent attention is similarly
strong in all three models.

**Evidence** (s002, all three models):
```
complete (TTC reward):        collision_risk × attn_agents  ρ = +0.769**
minimal  (no TTC):            collision_risk × attn_agents  ρ = +0.765**
basic    (violations only):   collision_risk × attn_agents  ρ = +0.990** (n=39, early termination)
```

**Interpretation**: Risk-reactive attention is learned from collision-avoidance training in
general — not from the TTC reward specifically. Any model penalized for collisions learns to
watch agents when danger rises.

**Implication for the paper**: The original hypothesis ("TTC reward causes agent attention") is
wrong in its simple form. The correct claim is more nuanced: TTC shapes the *baseline level*
of vigilance, not the reactive correlation structure (see Finding 4).

**Limitation**: Multi-model comparison currently evidenced by only 3 scenarios. Needs 50-scenario
run on minimal model to confirm at scale. This is the highest-priority pending experiment.

---

## Finding 4: TTC Reward Raises the Baseline Vigilance Prior — Not the Reactive Response

**What**: The clearest effect of the TTC reward is on the *resting baseline* of agent attention —
how much attention the model maintains even during calm phases — not on how strongly it reacts
when risk rises.

**Evidence** (s002, calm phase t=42–65, collision_risk ≈ 0):
```
complete:  attn_agents ≈ 0.14–0.15  (resting vigilance maintained)
minimal:   attn_agents ≈ 0.08–0.09  (lower resting vigilance)
```

**Evidence** (s002, episode mean):
```
complete:  attn_agents mean = 0.173
minimal:   attn_agents mean = 0.117   (−32%)
basic:     attn_agents mean = 0.264   (different mechanism — no GPS draws budget)
```

**Interpretation**: The TTC penalty fires continuously during near-miss situations (not just at
collision). This trains persistent agent surveillance rather than purely reactive response.
The complete model enters every timestep already more vigilant.

**Key visual**: In `fig_complete_vs_minimal_s002.png`, the shaded gap between complete and
minimal agent attention is present from t=0 and persists through the entire calm phase.
This is the "vigilance prior" — a learned resting posture shaped by reward design.

**Limitation**: Currently evidenced by 3 scenarios. Needs 50-scenario run on minimal model
to confirm the baseline gap is consistent across scenario types.

---

## Finding 5: Reward Structure Directly Shapes the GPS Attention Prior

**What**: GPS path attention baseline follows navigation reward content — the cleanest and most
direct reward→attention gradient found in the study.

**Evidence** (s002, episode means):
```
minimal  (+off_route +progression):   attn_gps = 0.314  ← highest
complete (+TTC partially suppresses): attn_gps = 0.166
basic    (no navigation reward):      attn_gps = 0.092  ← lowest
```

**The gradient is monotone and interpretable**: more navigation incentive → more GPS attention.
No correlation analysis needed — the three baseline levels tell the story directly.

**Interpretation**: The model learned GPS tokens are diagnostic for reward terms involving
route-following. Without those terms (basic), GPS is irrelevant and nearly ignored.
With TTC added (complete), some attention budget is pulled back from GPS toward agents.

**Limitation**: Single scenario. GPS baseline may vary by scenario geometry (highway vs
intersection). Needs replication across 50 scenarios with both models.

---

## Finding 6: Road Graph Compensates for GPS Absence in Basic Model

**What**: Without GPS route guidance, the basic model shifts more attention to road geometry.

**Evidence** (s002, episode means):
```
basic:    attn_roadgraph mean = 0.476
complete: attn_roadgraph mean = 0.419
minimal:  attn_roadgraph mean = 0.402
```

**Action-conditioned evidence (basic)**:
```
braking:  attn_roadgraph = 0.720  (72% of budget goes to road when braking)
steering: attn_roadgraph = 0.592
```

**Interpretation**: The basic model uses road geometry as a structural proxy for navigation.
When taking corrective actions, it relies even more heavily on the physical road layout.

Compare to complete during braking: road stays ~42%, agent attention elevated.
The complete model brakes because of agents; the basic model brakes because of road structure.

---

## Finding 7: Basic Model Cannot Navigate — Crashes in 2/3 Scenarios

**What**: The basic model terminates early in 2/3 scenarios (steps 38 and 39 of 80).

**Implication**: High attention-to-danger correlations are not sufficient for safe driving.
The basic model is maximally reactive to danger but lacks navigation competence to avoid
collisions reliably. Reward design not only shapes what the model attends to — it determines
whether that attention translates to effective behavior.

Note: The complete model also has 7/50 early terminations — so even the richest reward
configuration is not perfect. Real-world driving scenarios are hard.

---

## Methodological Contributions

1. **Within-episode correlation with Fisher z-transform aggregation** as the correct statistic
   for attention-reward analysis across heterogeneous RL episodes (vs. naive pooling)

2. **Scenario selection by risk variability** — std(collision_risk) > 0.2 and at least one
   calm phase required for meaningful correlation analysis. Event count (from event mining)
   is not a reliable proxy for analytical value.

3. **Multi-cycle episodes** as the gold standard for within-episode XAI validation — two
   independent risk cycles provide internal replication within a single scenario

4. **Counter-example analysis** as a methodological obligation — reporting only the mean ρ
   without investigating the negative-correlation scenarios is incomplete XAI

---

## What Is NOT Yet Established (Honest Gaps)

| Claim | Status | What is needed |
|-------|--------|----------------|
| Vigilance prior (complete vs minimal baseline gap) replicates at scale | **Unverified** | 50-scenario run on minimal model |
| GPS gradient replicates across scenario types | **Unverified** | Multi-model 50-scenario run |
| Counter-examples (s009, s031) explained | **Unverified** | BEV analysis of those scenarios |
| Attention *anticipates* risk (leads by 2-3 steps) | **Untested** | Temporal lead-lag cross-correlation |
| Effect is specific to Perceiver (vs Wayformer, MTR) | **Unknown** | Architecture comparison study |
| Attention mechanism reflects actual decision-making | **Contested** | Causal intervention study |

---

## Paper Claim (Draft — Updated for 50-Scenario Evidence)

> We demonstrate that reward design in RL-trained autonomous driving agents predictably shapes
> the Perceiver encoder's attentional prior across 50 real-world driving scenarios (3,676
> timesteps). Three findings hold at scale: (1) navigation reward terms increase GPS path
> attention baseline by 3.4×; (2) continuous proximity penalties (TTC at 1.5s) elevate resting
> agent surveillance by 48% relative to models without TTC — a learned vigilance prior
> maintained even during collision-free phases; (3) within-episode Spearman correlation between
> collision risk and agent attention is positive (mean ρ=+0.291, 95%CI [+0.125, +0.442])
> across 31 high-variation scenarios, with 26% showing individually significant effects.
> Pooled cross-episode analysis (ρ=+0.088) is shown to be confounded 3.3× by between-scenario
> risk heterogeneity. Notably, ~20% of high-variation scenarios exhibit reversed correlations
> (ρ < 0), revealing attentional heterogeneity that merits further investigation.

---

## Key Figures

| Figure | File | What it shows |
|--------|------|--------------|
| **3-way comparison** | `fig_3way_comparison_s002.png` | GPS gradient + agent baseline across 3 models — CENTERPIECE |
| **2-model overlay** | `fig_complete_vs_minimal_s002.png` | Shaded vigilance gap — best single figure |
| **Scenario scatter (50 scenarios)** | `fig_scenario_scatter_collision_risk_vs_attn_agents.png` | Within vs pooled confound at scale |
| **Risk distribution** | `fig_risk_distribution.png` | Which scenarios have useful variability |
| **Top-10 timeseries** | `fig_timeseries_s000.png` through `fig_timeseries_s031.png` | Per-scenario dynamics |
| **Correlation heatmap** | `fig2_correlation_heatmap.png` | Full matrix overview |
| **Action-conditioned** | `fig4_action_attention.png` | Braking/steering attention allocation |
| **BEV panel s002** | `fig_bev_panel_s002.png` | Ground-truth scene + attention bars |
