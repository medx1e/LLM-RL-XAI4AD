# Phase 1c Findings — Attention Rollout

## What attention rollout does

The Perceiver runs 4 blocks of: cross-attention (queries→tokens) + self-attention (queries→queries). Raw `cross_attn_avg` ignores that self-attention mixes information between queries after each cross-attention step. Rollout corrects for this by chaining residual-corrected self-attention matrices and applying the result to the mean cross-attention:

```
A_eff[l] = 0.5*I + 0.5*A_self[l]  (residual correction per layer)
R        = A_eff[3] @ A_eff[2] @ A_eff[1] @ A_eff[0]  (16×16)
rollout  = R @ cross_attn_avg                           (16×280)
```

## MAD: rollout vs raw mean-pool

| Category | Overall MAD |
|---|---|
| SDC | 0.0184 |
| Agents | 0.0119 |
| Road | 0.0467 |
| TL | 0.0630 |
| GPS | 0.0246 |
| **Global** | **0.0329** |

## Decision

Rollout introduces a moderate shift (global MAD=0.033). Self-attention mixing is non-negligible but modest. Report rollout as a robustness check; use raw mean-pool as primary.

## What to write in the thesis

> **Methodological note — attention rollout:**
> The Perceiver processes input tokens through 4 interleaved blocks of cross-attention (queries attend to input tokens) and self-attention (queries attend to each other). Raw cross-attention weights do not account for the information mixing that occurs in self-attention layers. We implemented attention rollout (Abnar & Zuidema 2020), which chains residual-corrected self-attention matrices to compute an effective attention from the final representation back to the input tokens. The mean absolute deviation between rolled-out and raw attention was 0.033 at the category level, indicating [INSERT CONCLUSION]. We [use raw cross-attention / use rollout] as the primary attention signal throughout this chapter.
