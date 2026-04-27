# XAI Research Context: Post-Hoc Attribution + Attention in Autonomous Driving RL

> **Purpose of this document:** Share the complete technical and empirical context of our XAI
> research with a research AI agent, to generate ideas for experiments, improvements, and
> publishable contributions.

---

## 1. Platform Overview

We built a **Streamlit showcase platform** for explainable reinforcement learning on V-MAX
autonomous driving policies. V-MAX is an RL framework (JAX/Flax) trained on the Waymo Open
Motion Dataset (WOMD) using Soft Actor-Critic (SAC). We received 35+ pretrained model weights
from the paper authors.

**Research axes we are pursuing:**
- Post-hoc XAI (attribution methods + attention analysis) ← **primary focus of this document**
- Concept Bottleneck Models (CBM) — separate module, already implemented
- Counterfactual explanations — future work
- LLM explanation layer — future work

**Platform goal:** Demo platform for FYP defense. The post-hoc XAI tab is the first complete
research module.

---

## 2. The Driving Model Setup

### Observation Space (1,655 flat features, ego-relative, normalized ≈ [-1, 1])

| Category | Features | Entities | Content |
|---|---|---|---|
| `sdc_trajectory` | 40 | 1 (ego) | Own trajectory: 5 timesteps × (x,y,vx,vy,yaw,l,w,valid) |
| `other_agents` | 320 | 8 agents | Nearest 8 vehicles: 5 timesteps × 8 features each |
| `roadgraph` | 1,000 | 200 points | Lane boundaries, directions (5 features/point) |
| `traffic_lights` | 275 | 5 lights | Position + state: 5 timesteps × 11 features each |
| `gps_path` | 20 | 10 waypoints | Target route waypoints (2 features each) |

### Action Space
2D continuous: `(acceleration, steering)`.

### Models Available (35+ pretrained weights, 5 architectures)

| Architecture | Internal name | Notes |
|---|---|---|
| **Perceiver/LQ** | `lq` | Cross-attention from learned queries to input tokens — only one with accessible attention weights |
| **MTR** | `mtr` | Local k-NN attention |
| **Wayformer** | `wayformer` | Late fusion per-modality |
| **MGAIL/LQH** | `lqh` | Hierarchical cross-attention |
| **None/MLP** | `none` | Feedforward baseline |

**Reward configurations per architecture (Perceiver has all 3):**
- `minimal` — safety + navigation only (collision, offroad, red_light, off_route, progress)
- `basic` — minimal + partial behavior
- `complete` — adds comfort and overspeed penalties

**Seeds:** 42, 69, 99 (3 seeds per config).  
**Broken models:** `sac_seed0/42/69` — incompatible `speed_limit` feature, excluded everywhere.

---

## 3. Post-Hoc XAI Framework — What We Built

### 3.1 The 7 Attribution Methods

| Method | Class | How it works | Speed | Key property |
|---|---|---|---|---|
| `vanilla_gradient` | VanillaGradient | `∂f/∂x` at operating point | Fast | Baseline; saturates in flat regions |
| `integrated_gradients` | IntegratedGradients | Path integral from baseline to input (n_steps=50) | Medium | Completeness axiom: Σattr = f(x) - f(baseline) |
| `smooth_grad` | SmoothGrad | Average gradients over N noisy copies | Medium | Reduces gradient noise |
| `gradient_x_input` | GradientXInput | `x * ∂f/∂x` | Fast | Highlights large + sensitive features |
| `perturbation` | PerturbationAttribution | Zero out features, measure Δoutput | Slow | Model-agnostic, occlusion-based |
| `feature_ablation` | FeatureAblation | Remove whole categories, measure Δoutput | Fast | High-level category importance |
| `sarfa` | SARFA | Relevance × Specificity (RL-specific) | Slow | Designed for Q-value action specificity |

All methods output an `Attribution` dataclass:
```
raw            : jnp.ndarray shape (1655,)
normalized     : abs(raw) / sum(abs(raw))
category_importance : dict[str, float]   e.g. {"roadgraph": 0.68, "other_agents": 0.15, ...}
entity_importance   : dict[str, dict[str, float]]  e.g. {"other_agents": {"agent_0": 0.12, ...}}
```

### 3.2 Attention Extraction (Perceiver Only)

The Perceiver encoder uses cross-attention from 16 learned query tokens to the input tokens.
We extract the full attention matrix `(num_queries=16, num_tokens)` at every timestep.

We aggregate attention to the same 5 categories as attribution:
```
attn_agents     = sum of attention weight over tokens [40:360]
attn_roadgraph  = sum of attention weight over tokens [360:1360]
attn_tl         = sum of attention weight over tokens [1360:1635]
attn_gps        = sum of attention weight over tokens [1635:1655]
attn_sdc        = sum of attention weight over tokens [0:40]
```

We also compute:
- **Per-agent attention** (attention to each of 8 agents separately)
- **Per-query attention** per category (useful for query specialization analysis)
- **attn_to_threat** = attention to the agent with lowest TTC

### 3.3 Evaluation Metrics

**Faithfulness** (are explanations faithful to what the model actually does?):
- Deletion curve: progressively remove most-important features → measure output drop (lower AUC = better)
- Insertion curve: progressively add most-important features → measure output rise (higher AUC = better)

**Sparsity** (are explanations focused?):
- Gini coefficient (0 = uniform, 1 = sparse)
- Shannon entropy
- Top-k concentration (fraction of importance in top k features)

**Consistency** (same method on similar scenarios):
- Pairwise Pearson correlation across attribution vectors
- Category-level correlation (more robust)

**Cross-method agreement:**
- Pairwise category-level Pearson r between methods
- Used to identify which method pairs agree and which diverge

### 3.4 Event Mining Module

We detect **7 frame-level driving event types** from model rollouts:
- `hazard_onset`, `near_miss`, `hard_brake`, `evasive_steering`, `collision`, `offroad`, `collision_imminent`

The `VMaxAdapter` extracts per-step unflattened observation data (ego state, agent states, road info).
Each event carries: onset/peak/offset timestep, severity (LOW→CRITICAL), causal agent ID.

Mined catalog: **153 events across 5 scenarios** (from initial experiment).

---

## 4. Experiments Completed and Key Findings

### 4.1 Experiment 1: Event Mining + Temporal XAI (2 methods, 3 events)

Model: `womd_sac_road_perceiver_minimal_42`. Methods: vanilla_gradient, integrated_gradients.

**Finding 1 — VG underestimates agent importance by 50–144×:**
During evasive maneuvers, vanilla_gradient attributes <2% to other agents while integrated_gradients
attributes 22–35%. This is explained by gradient saturation: VG computes the local derivative at
the operating point, but activation functions are flat in their saturation region. IG integrates
along the path from a zero baseline, bypassing saturation.

*Implication: Research using only vanilla_gradient dramatically underestimates agent influence.*

**Finding 2 — Temporal attribution arc during hazard onset:**
Critical hazard (min_TTC=0.049 in normalized space): a clear detect→attend→commit→execute pattern:
- Pre-hazard (t=4–7): roadgraph ~55%, gps_path ~32%, other_agents ~5%
- Hazard detection (t=9–13): other_agents spikes to 23%, roadgraph drops to ~47%
- Hazard resolution (t=28–40): roadgraph climbs to 85%, other_agents falls to <5%

*VG and IG tell complementary stories: VG shows the "attention arc", IG shows the model
continues to depend on agents throughout (saturation masking the late-hazard dependency).*

**Finding 3 — Per-agent causal detection:**
Causal agent (agent_0) goes from 0.004% to 22.4% importance in 4 timesteps at hazard onset.
Agent_6 emerges later as secondary threat during evasion (from 0% to 46% as ego steers toward it).

**Finding 4 — GPS path importance increases during evasive maneuvers:**
GPS importance triples at event peak — model checks route deviation during aggressive steering.

**Finding 5 — Method divergence at t=35 (all 7 methods, scenario s002):**
Three distinct camps:
- Gradient methods (VG, GxI, IG, SmoothGrad): roadgraph dominates (0.35–0.69)
- Occlusion methods (Perturbation, FeatureAblation): sdc_trajectory dominates (0.56) — but identical results; they are redundant at category level
- SARFA: other_agents dominates (0.71) — aligns with human intuition for a near-miss scenario

---

### 4.2 Experiment 2: Reward-Conditioned Attention Analysis (50 scenarios, 2 models)

Models: `perceiver_complete_42` and `perceiver_minimal_42`.  
Scale: 50 scenarios each, ~3,700 timesteps each.  
Method: Attention extraction + continuous risk metrics (TTC-based collision_risk, route deviation, etc.)

**Finding 1 — GPS Gradient (attention prior shaped by reward):**
| Model | GPS attention | Agent attention | Road attention |
|---|---|---|---|
| minimal | 33.5% | 4.2% | 42.7% |
| complete | 16.4% | 5.6% | 52.1% |

Minimal model allocates 2× more attention to GPS than complete. The TTC penalty in the complete
reward config redistributes attention away from route-following toward other agents and road geometry.

**Finding 2 — Budget Reallocation Reversal (strongest result, paper-ready):**
Low-risk → High-risk transition in agent attention:
- Complete model: agents **+38.2%** relative increase (p=7.44e-14)
- Minimal model: agents **−16.6%** relative decrease (p=0.019)

Same architecture, same algorithm — *only* reward design differs — produces *opposite* reallocation
behaviors. This is the key publishable finding.

**Finding 3 — Within-episode ρ (Simpson's paradox warning):**
Pooled ρ(collision_risk, attn_agents):
- Complete: +0.088 (looks weak)
- Minimal: −0.155 (looks opposite!)

Within-episode ρ (Fisher z-aggregate across high-variation scenarios):
- Complete: **+0.291**, CI=[+0.125, +0.442], 80.6% of scenarios individually significant
- Minimal: **+0.141**, CI=[−0.039, +0.313], 61% significant

Pooling across episodes with different baselines creates a confound (Simpson's paradox).
Within-episode analysis is necessary.

**Finding 4 — Vigilance Gap:**
During calm phases (risk < 0.2), the complete model maintains 0.146 agent attention vs 0.062 for
minimal (+134% gap in scenario s002). The TTC penalty teaches the complete model to maintain
"resting surveillance" even when there is no immediate threat.

**Finding 5 — Entropy rises with risk (redistribution, not narrowing):**
Within-episode ρ(entropy, collision_risk) = +0.199 for complete model.
The model redistributes attention more evenly under threat (higher entropy) rather than narrowing
onto fewer categories. Minimal model shows the opposite in some scenarios.

**Finding 6 — Lead-lag (exploratory):**
Aggregate lead-lag curve peaks at lag=+2 steps (attention leads risk by ~2 steps), suggesting
predictive attention allocation. But the spread is wide (std=5.63) — boundary clusters at ±8
suggest monotonic correlations in edge cases. Report as exploratory.

**Counter-examples (19% of scenarios):**
6/31 high-variation scenarios show negative ρ (attention decreases with risk). Detailed analysis
shows two types: (a) oscillatory scenarios where risk flickers too fast for attention to track,
(b) road-hazard scenarios where road geometry, not agents, is the real threat. The model responds
correctly in these cases; the XAI metric is misleading.

---

## 5. What We Have (Infrastructure for Future Experiments)

### 5.1 Data
- 35+ pretrained V-MAX models (5 architectures × 3 configs × 3 seeds + extras)
- `data/training.tfrecord` — Waymo Open Dataset scenarios (44,000+ available)
- Event catalog: 153 events (expandable)
- Results: JSON analysis files for 50 scenarios × 2 models (reward_attention experiment)
- Precomputed attribution series for platform demo scenarios

### 5.2 Code Infrastructure
- **7 attribution methods**, all JAX-native, all tested
- **Attention extraction** for Perceiver (cross-attention + self-attention)
- **3 metric families**: faithfulness (deletion/insertion AUC), sparsity (Gini/entropy), consistency
- **Event mining pipeline**: 6 detectors, 7 event types, BEV video rendering
- **Temporal XAI pipeline**: event-conditioned temporal attribution extraction
- **Reward-conditioned attention analysis**: full end-to-end pipeline (risk_metrics, correlation_analyzer, temporal_analyzer, lead-lag analysis)
- **Experiment pipeline**: resume-friendly JSON caching, per-timestep analysis, cross-model comparison
- **Visualization**: 15+ plot types including paper-quality figures

### 5.3 Known Gaps / Limitations
- Attention extraction only implemented for Perceiver; MTR/Wayformer/MGAIL need architecture-specific adapters
- Event detector thresholds calibrated for real-world units, not normalized observation space (~[-1,1]) — needs recalibration
- VG and IG only compared in temporal XAI; all 7 methods compared only in single-timestep analysis
- Reward-conditioned attention done for complete and minimal (seed 42 each); not yet run for MTR/Wayformer/MGAIL or for other seeds
- No formal statistical test for temporal patterns (lead-lag, arc detection)
- `attention_gradient_correlation()` exists in faithfulness.py but has never been used in a real study

---

## 6. The "Is Attention Explanation?" Question

The seminal paper by Jain & Wallace (2019) "Attention is not Explanation" argued that attention
weights do not correlate with gradient-based feature importance and should not be interpreted as
explanations. Wiegreffe & Pinter (2019) "Attention is not Not Explanation" argued the opposite
using different tests. This debate is directly relevant to us:

**What we have that speaks to this question:**
- `attention_gradient_correlation(attr_a, attr_b)` — a function that computes Pearson correlation
  between any two attribution vectors (works for attention-as-attribution vs gradient-as-attribution)
- Perceiver attention weights at every timestep for every scenario
- 7 gradient/perturbation attribution methods at the same timesteps
- Temporal context (same scenario, multiple timesteps)
- Driving-specific ground truth: event labels, causal agent annotations

**We can directly test:**
1. Does Perceiver attention (as a feature importance signal) correlate with IG attribution?
2. Does the correlation change during critical events (hazard onset, near-miss)?
3. Which specific categories diverge most (agents? GPS? road)?
4. Is per-agent attention rank correlated with per-agent IG importance rank?
5. How does attention-IG correlation compare to VG-IG correlation and VG-SARFA correlation?

This framing positions our work in a live debate in the XAI community.

---

## 7. The Paper Being Written (reward_attention)

Working title: **"Reward-Conditioned Attention: How Multi-Objective RL Agents Learn to Focus on
What Matters"**

Status: Analysis complete (50 scenarios, 2 models). Key figures generated. LaTeX draft (`main_v4.tex`).

**Strongest publishable findings (in priority order):**
1. Budget reallocation reversal (+38.2% vs −16.6%) — same architecture, different reward → opposite attention behavior
2. GPS gradient — reward design shapes the static attention prior (2× GPS difference)
3. Within-episode ρ = +0.291 for complete model (80.6% of scenarios significant)
4. Vigilance gap — complete model maintains agent surveillance during calm phases (+134%)
5. Simpson's paradox demonstration — pooled ρ=+0.088 masks within-episode ρ=+0.291 (methodological contribution)

**What would make the paper stronger:**
- More architectures (does MTR show the same budget reallocation? it has no extractable attention, so gradient methods only)
- More seeds (do seeds 69 and 99 replicate the findings?)
- Formal causal framing (does the attention-risk link reflect the actual causal structure?)
- Connection to attention-vs-gradient debate (Jain & Wallace 2019)
- Any normalization or post-processing of attention/attribution that increases predictive power

---

## 8. Open Questions We Want to Investigate

1. **Attention vs attribution consistency:** Does Perceiver attention agree with IG at the category level? At the entity level? How does this compare to IG vs VG agreement?

2. **Normalization strategies:** Raw attribution values span many orders of magnitude (VG: 1e-20 to 0.46 per agent). Would softmax normalization, rank-based normalization, or z-score normalization of the raw attribution array (before aggregation) improve faithfulness or consistency metrics?

3. **Cross-architecture attribution patterns:** We have preliminary results only for Perceiver. Does roadgraph domination hold for MTR, Wayformer, MGAIL? Does the detect→attend→commit→execute arc appear in gradient attributions for non-attention architectures?

4. **Temporal aggregation:** Currently we analyze single timesteps. What if we aggregate attribution over a time window? Rolling average? Event-window average? Would this produce more stable or more informative explanations?

5. **Attention head specialization:** The Perceiver has 16 query tokens. Do different queries specialize for different input categories? Can we identify a "safety head" and a "navigation head"?

6. **Semantic consistency of attribution:** Do similar driving scenarios (same event type, same severity) produce similar attribution patterns? Can attribution serve as a scenario fingerprint?

7. **Faithfulness of attention:** Deletion/insertion curves are implemented for attribution methods. We have not tested them on attention-as-attribution. How does Perceiver attention score on faithfulness compared to IG?

8. **SARFA vs attention:** SARFA is the only method that explicitly measures action specificity (does this feature change the *chosen* action specifically?). Does SARFA attribution correlate with attention weights better than other gradient methods?
