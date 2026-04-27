# Phase 1a Findings — Size-Corrected Attribution
## What size correction does

Current pipeline: `cat_imp[c] = sum(abs(raw[c])) / sum(abs(raw_all))` — sums over ALL features in the category. Larger categories accumulate more attribution purely from count.

Correction: `corrected[c] = (cat_imp[c] / n_c) / Σ(cat_imp[c'] / n_c')` — divides by feature count before renormalizing. Answers 'which category has the most influence **per input dimension**?' instead of total.

## Correction factors (relative to GPS = 20 features)

| Category | Features | Multiplier |
|---|---|---|
| SDC | 40 | ×0.500 |
| Agents | 320 | ×0.062 |
| Road | 1000 | ×0.020 |
| TL | 275 | ×0.073 |
| GPS | 20 | ×1.000 |

Roadgraph (1000 features) gets a ×0.020 multiplier — its importance is divided by 50× relative to GPS.

## Event_02 IG at peak (t=35)

| Category | Original | Corrected | Change |
|---|---|---|---|
| SDC | 0.086 | 0.294 | +0.209 |
| Agents | 0.288 | 0.124 | -0.164 |
| Road | 0.553 | 0.076 | -0.477 |
| TL | 0.000 | 0.000 | +0.000 |
| GPS | 0.073 | 0.506 | +0.432 |

## Scenario 002 t=35 — Roadgraph original vs corrected

| Method | Road (orig) | Road (corr) | GPS (orig) | GPS (corr) |
|---|---|---|---|---|
| vanilla_gradient | 0.150 | 0.012 | 0.124 | 0.477 |
| gradient_x_input | 0.495 | 0.081 | 0.070 | 0.579 |
| integrated_gradients | 0.689 | 0.260 | 0.018 | 0.347 |
| smooth_grad | 0.355 | 0.056 | 0.038 | 0.298 |
| perturbation | 0.014 | 0.001 | 0.170 | 0.363 |
| feature_ablation | 0.014 | 0.001 | 0.170 | 0.363 |
| sarfa | 0.006 | 0.000 | 0.196 | 0.784 |

## Interpretation

Size correction dramatically changes the *ranking* of categories. Roadgraph drops from dominant (~50–70% original) to near-zero per-feature importance, while GPS rises sharply.

**What this means for the thesis:**
- The 'roadgraph dominates' finding is correct as a statement about *total* attribution — the model's output is most sensitive to the aggregate road geometry signal.
- But each individual GPS waypoint carries far more information per feature than each roadgraph point — GPS is a high-density, compact signal.
- Both perspectives should be reported. The thesis chapter should clarify which normalization is used and why total attribution is the appropriate primary metric (the model processes all 1000 roadgraph features together, not one at a time).
- Size-corrected numbers should be included as a robustness note or table.
