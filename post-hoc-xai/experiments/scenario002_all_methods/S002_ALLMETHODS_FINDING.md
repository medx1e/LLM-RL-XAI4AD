# Scenario 002 — All-Methods XAI Findings

**Date:** 2026-02-22
**Model:** `womd_sac_road_perceiver_minimal_42`
**Script:** `experiments/scenario002_all_methods.py`
**Output:** `experiments/scenario002_all_methods/`

---

## What Scenario 002 Was

The SDC **did not move** (0% route completion) but **survived** — no collision, no off-road. It was surrounded by 8 agents simultaneously generating critical near-misses throughout the episode. The model essentially froze in place: a conservative, defensive response that was technically correct.

- **80 steps total**
- **17 events mined** — all hazard onsets and near-misses, all critical severity
- **Focal timestep: t=35** — near-miss peak (agent 0), chosen for single-timestep analysis

---

## The 3 Camps of Methods

At t=35, the 7 methods split into three distinct interpretive camps:

### Camp 1 — Gradient methods (VG, GxI, IG, SmoothGrad)
**Say: roadgraph matters most (0.35–0.69)**

The model is consulting road structure heavily — likely to determine where it *cannot* go. Makes sense for a frozen SDC in a constrained space.

| Method | other_agents | roadgraph | sdc_trajectory |
|---|---|---|---|
| vanilla_gradient | 0.510 | 0.150 | 0.199 |
| gradient_x_input | 0.372 | **0.495** | 0.031 |
| integrated_gradients | 0.261 | **0.689** | 0.005 |
| smooth_grad | 0.407 | 0.355 | 0.097 |

### Camp 2 — Occlusion methods (Perturbation, FeatureAblation)
**Say: sdc_trajectory matters most (0.56)**

Zeroing out the SDC's own past trajectory changes the output most. The model needs to know where it has been. However, see faithfulness critique below.

| Method | sdc_trajectory | other_agents | roadgraph |
|---|---|---|---|
| perturbation | **0.560** | 0.205 | 0.014 |
| feature_ablation | **0.560** | 0.205 | 0.014 |

> Note: Perturbation and FeatureAblation give **identical results** at category level — both zero out whole categories. They are redundant; no need to run both.

### Camp 3 — SARFA
**Says: other_agents matters most (0.71)**

Aligns with human intuition — near-miss means other cars are the threat. SARFA is designed to highlight RL-relevant features (what actually changes the Q-value decision), not just what the model is sensitive to.

| Method | other_agents | gps_path | traffic_lights |
|---|---|---|---|
| sarfa | **0.713** | 0.196 | 0.078 |

---

## Which Methods to Trust — Faithfulness

| Metric | What it measures | Best method |
|---|---|---|
| Deletion AUC (lower = more faithful) | Removing top features should hurt output fast | GradientXInput (−0.263) |
| Insertion AUC (higher = more faithful) | Adding top features should recover output fast | SmoothGrad (+0.089) |

Full table at t=35:

| Method | Gini ↑ | Entropy | Del AUC | Ins AUC | ms/step |
|---|---|---|---|---|---|
| vanilla_gradient | 0.951 | 0.555 | −0.227 | +0.021 | 574 |
| gradient_x_input | 0.916 | 0.662 | **−0.263** | −0.054 | 566 |
| integrated_gradients | 0.867 | 0.745 | −0.159 | +0.075 | 791 |
| smooth_grad | 0.825 | 0.789 | −0.195 | **+0.089** | 753 |
| perturbation | 0.870 | 0.716 | −0.211 | −0.326 | 1635 |
| feature_ablation | 0.870 | 0.716 | −0.211 | −0.326 | 1645 |
| sarfa | 0.784 | 0.814 | −0.101 | −0.090 | 1687 |

**Perturbation/FeatureAblation score worst on insertion (−0.326)** — their "sdc_trajectory first" story is likely misleading. The feature ordering does not match how the model actually uses information.

**SARFA has the least-negative deletion AUC (−0.101)** — removing its top features hurts output the least, meaning it is the least faithful to raw model behaviour by this metric. Despite giving the most interpretable story, its ranking is the weakest by quantitative faithfulness.

---

## The 3 Real Conclusions

**1. The methods answer different questions — all legitimately.**

- Gradient methods reveal what the model is *sensitive to* (local Jacobian)
- SARFA reveals what *changes the policy decision* (RL-weighted relevance)
- Occlusion reveals what is *load-bearing* for the output value

These are three different questions. The disagreement is not noise — it is signal about the model.

**2. Perturbation = FeatureAblation at category granularity.**

Both zero out whole categories. At this level of analysis they are identical and one should be dropped from future experiments to save ~1700 ms/step.

**3. For near-miss safety analysis, SARFA gives the most actionable story.**

Even though it scores lower on faithfulness metrics, it surfaces the *causal* factor (the threatening agents at 0.71) — which is what you want an XAI system to highlight for post-hoc safety investigation. For rigorous faithfulness evaluation, prefer SmoothGrad or IntegratedGradients.

---

## Practical Recommendations

| Use case | Recommended method |
|---|---|
| Safety event analysis / causal explanation | **SARFA** |
| Rigorous faithfulness benchmark | **SmoothGrad** or **IntegratedGradients** |
| Fast exploration / baseline | **VanillaGradient** (566 ms, simplest) |
| Drop entirely | **FeatureAblation** (redundant with Perturbation) |

---

## Output Files

| File | Description |
|---|---|
| `A_temporal_category_all_methods.png` | Category importance over time, 7 subplots |
| `B_temporal_per_category_grid.png` | Per-category grid, all methods overlaid |
| `C_stacked_all_methods.png` | Stacked area composition, all methods |
| `D_agent_importance_all_methods.png` | Per-agent importance over time |
| `E_sparsity_over_time.png` | Gini + entropy over time, all methods |
| `F_peak_timestep_comparison.png` | Bar chart comparison at t=35 |
| `G_deletion_insertion_curves.png` | Faithfulness deletion/insertion curves |
| `H_importance_heatmap.png` | Category × method heatmap at t=35 |
| `summary_metrics.csv` | Full numeric table |
