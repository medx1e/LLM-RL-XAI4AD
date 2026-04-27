# Phase 2 Findings — Attention-Attribution Correlation Pilot

## Setup

- 2 models × 3 scenarios × 80 timesteps = 480 timesteps total
- Attention: rollout-corrected (Phase 1c canonical signal)
- Methods compared: VG (baseline), IG (primary), GxI (from cache)
- Metric: Kendall τ between 5-category attention ranking and attribution ranking

## Combined Results (all scenarios)

| Method | Complete mean τ | Minimal mean τ | Complete τ>0 | Minimal τ>0 |
|---|---|---|---|---|
| vanilla_gradient | 0.453 | 0.207 | 96% | 67% |
| integrated_gradients | 0.058 | 0.120 | 50% | 50% |
| gradient_x_input | 0.333 | 0.210 | 88% | 73% |

## Interpretation

Weak attention-IG agreement (τ=0.058 for complete). Attention does not reliably reflect gradient importance at category level.

## Go/No-Go for Phase 3

Mean attention-IG τ = 0.058 (complete), 0.120 (minimal). **GO** — signal is present, scale up to 50 scenarios with risk stratification.

## What to write in the thesis

> We computed the Kendall rank correlation between rollout-corrected Perceiver attention and gradient attribution at the category level for each timestep. Across 480 timesteps (2 models × 3 scenarios × 80 steps), mean attention-IG τ = 0.058 (complete) and 0.120 (minimal), with 50% and 50% of timesteps showing positive agreement respectively. [EXPAND with risk-stratified results from Phase 3.]
