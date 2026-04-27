# Phase 1b Findings — Attention Aggregation Comparison

## What we compared

Three strategies for collapsing the Perceiver's `(16, 280)` cross-attention matrix into a 5-dim category vector:

- **mean** (current default): average over 16 queries
- **maxpool**: max over 16 queries per token (best-case signal per token)
- **entropy**: weight queries by sharpness (1/H) before averaging

## Mean Absolute Deviation from mean-pool

| Category | maxpool MAD | entropy MAD |
|---|---|---|
| SDC | 0.0448 | 0.0150 |
| Agents | 0.0198 | 0.0079 |
| Road | 0.0689 | 0.0157 |
| TL | 0.1022 | 0.0070 |
| GPS | 0.0395 | 0.0073 |
| **Overall** | **0.0551** | **0.0106** |

## Decision

**Primary: mean-pool. Secondary (robustness check): maxpool.**

Entropy-weighted (MAD=0.011) is essentially identical to mean-pool — no reason to prefer it.
Maxpool (MAD=0.055) deviates meaningfully, especially for traffic lights (0.102) and roadgraph
(0.069). This deviation is real — it reflects the fact that some queries specialize on those
categories and are diluted when averaged with the other 15 queries. However, we keep mean-pool
as the primary metric for two reasons:
1. Consistency with the reward-conditioned attention paper (already submitted).
2. Mean-pool is theoretically interpretable as "what the ensemble of queries attends to on average."

Maxpool is reported as a robustness check: if the attention-IG correlation finding holds under
both mean-pool and maxpool, it is robust to query specialization.

## Query specialization findings

This is a new, thesis-worthy observation:

- Mean entropy across all scenarios: ~4.6 bits out of 8.13 max (57%) — queries are NOT diffuse
- **Complete model**: query 3 is consistently the most focused (entropy ≈3.1 bits, 38% of max)
- **Minimal model**: query 1 is dramatically focused (entropy as low as 1.62 bits, 20% of max)
- 6–13 out of 16 queries fall below 60% of max entropy — moderate-to-strong specialization
- The minimal model shows MORE query specialization than the complete model (13/16 vs 7-8/16
  below 60% in some scenarios) — reward design shapes not just attention allocation but also
  how the queries organize themselves internally

## What to write in the thesis

> **Methodological note — aggregation choice:**
> The Perceiver uses 16 learned query vectors that cross-attend to 280 input tokens. We
> evaluated three aggregation strategies: query mean-pooling, max-pooling (best-case
> attention per token), and sharpness-weighted averaging. Entropy-weighted pooling was
> nearly identical to mean-pool (MAD=0.011). Max-pooling differed more substantially
> (MAD=0.055), particularly for traffic lights (MAD=0.102) and roadgraph (MAD=0.069).
> We use mean-pooling as the primary metric for consistency with the reward-conditioned
> attention study, and report max-pool results as a robustness check.

> **Exploratory finding — query specialization:**
> We measured the Shannon entropy of each query's attention distribution (lower entropy =
> more focused). Mean query entropy was 4.6 bits (57% of the 8.13-bit maximum), indicating
> that queries do not attend uniformly. In the complete model, query 3 consistently showed
> the sharpest focus (entropy ≈3.1 bits). In the minimal model, query 1 was even more
> dramatically specialized (entropy as low as 1.62 bits, 20% of maximum), with 6–13 of 16
> queries falling below 60% of maximum entropy across scenarios. That the minimal model
> exhibits greater query specialization than the complete model suggests that reward
> configuration shapes not only the overall attention allocation (the GPS gradient finding)
> but also the internal organization of the latent query representations.
