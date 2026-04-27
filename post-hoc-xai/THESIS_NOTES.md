# Thesis Notes — Post-Hoc XAI Chapter
> Living document. Updated throughout implementation.
> Captures: research decisions, technical findings, what to write in the thesis.

---

## Research Framing

**Core contribution of this chapter:**
A post-hoc XAI framework for RL-based autonomous driving, applied to V-MAX Perceiver models.
We go beyond running attribution methods — we validate them methodologically (size correction,
method divergence analysis) and connect to a live research debate: *is attention a faithful
explanation in driving RL?*

**The debate we enter:**
Jain & Wallace (2019) showed attention weights in NLP models are often uncorrelated with
gradient-based importance and should not be interpreted as explanations. Wiegreffe & Pinter
(2019) challenged this. Bibal et al. (2022) called for cross-domain testing beyond NLP.
**We answer that call** — same question, different domain (RL + autonomous driving +
Perceiver cross-attention instead of RNN self-attention).

---

## What We Have Built

### Attribution methods (7 total)
| Method | Type | Key property |
|---|---|---|
| VanillaGradient | Gradient | Fast; saturates in flat regions |
| IntegratedGradients | Path-integrated gradient | Completeness axiom: Σattr = f(x) − f(baseline) |
| SmoothGrad | Averaged gradient | Reduces gradient noise |
| GradientXInput | Gradient × input | Captures magnitude + sensitivity |
| PerturbationAttribution | Occlusion | Model-agnostic |
| FeatureAblation | Occlusion (category-level) | High-level category importance |
| SARFA | RL-specific | Relevance × specificity; action-discriminative |

### Attention extraction (Perceiver only)
Cross-attention reconstructed from Q and K Dense projections captured via
`capture_intermediates`. Returns per-layer and averaged cross-attention: shape `(16, 280)`
(16 learned queries × 280 input tokens). Architecture: `tie_layer_weights=True`,
`num_latents=16`, `encoder_depth=4`, `cross_num_heads=2`.

### Evaluation metrics
- **Faithfulness**: deletion/insertion AUC
- **Sparsity**: Gini coefficient, Shannon entropy, top-k concentration
- **Consistency**: pairwise Pearson correlation across scenarios
- **Cross-method agreement**: category-level Pearson r between methods

---

## Key Empirical Findings (existing experiments)

### Method divergence — three camps (scenario s002, t=35, near-miss)
At a near-miss timestep, 7 methods split into three interpretive camps:
- **Gradient methods** (VG, GxI, IG, SmoothGrad): roadgraph dominates (0.35–0.69)
- **Occlusion methods** (Perturbation, FeatureAblation): SDC trajectory dominates (0.56) — and they give **identical results** at category level, making one redundant
- **SARFA**: other_agents dominates (0.71) — aligns with human intuition for a near-miss

> **Thesis note:** This method divergence is a contribution in itself. Three methods from
> the same "gradient" family broadly agree; the RL-specific method (SARFA) points to
> agents, which is what a human expert would expect during a near-miss. This motivates
> using SARFA as the RL-appropriate comparator when evaluating attention faithfulness.

### Temporal attribution arc — hazard onset (event_02)
Critical hazard (min_TTC≈0.05): a detect→attend→commit→execute behavioral arc under VG:
- Pre-hazard: roadgraph ~55%, GPS ~32%, agents ~5%
- Hazard detection: agents spike to 23%, roadgraph drops to ~47%
- Resolution: roadgraph climbs to 85%, agents fall to <5%

IG tells a complementary story: agents stay at 40–50% throughout (gradient saturation
masks the sustained dependency in VG).

### VG underestimates agents by 50–144× vs IG
During evasive maneuvers, VG attributes <2% to agents while IG attributes 22–35%.
Cause: VG computes the local gradient at the operating point; saturated activations
give near-zero gradients even for functionally important features. IG integrates along
the full path from baseline, bypassing saturation.

> **Thesis note:** This is a strong methodological finding. State it explicitly:
> "research using only vanilla gradient to explain driving RL models will dramatically
> underestimate the model's reliance on other agents."

### Per-agent causal detection
Agent_0 (causal agent in event_02) goes from 0.004% → 22.4% importance in 4 timesteps
at hazard onset. Agent_6 emerges from 0% → 46% as the ego vehicle steers toward it
(secondary threat). This is evidence the model tracks which specific agent is dangerous.

---

## Phase 0 — Data Audit (completed 2026-04-24)

### Critical finding: only one overlap dataset
The only existing cache where **both attention AND attribution are at the same timesteps**
is `platform_cache/`:
- 2 models (complete_42 and minimal_42)
- 3 scenarios each, full episode length T
- Methods cached: IG, GxI, FeatureAblation
- Attention cached: `cross_attn_avg` shape (1, 16, 280) per timestep

This is what Phase 2 (pilot correlation) will use. No model loading required.

### Self-attention rollout: confirmed feasible
Config confirmed: `tie_layer_weights=True` for all models. Self-attention Q/K live at
`lq_attention > self_attn > Dense_0/__call__` in the intermediates tree — same pattern
as the already-working cross-attention extraction. Implementation: ~30 lines.

### Token structure (280 tokens)
| Category | Tokens | Range |
|---|---|---|
| SDC trajectory | 5 | [0, 5) |
| Other agents | 40 | [5, 45) |
| Roadgraph | 200 | [45, 245) |
| Traffic lights | 25 | [245, 270) |
| GPS path | 10 | [270, 280) |

---

## Phase 1a — Size-Corrected Normalization (completed 2026-04-24)

### What we did
Added `size_correct_attribution()` and `size_correct_attention()` to
`posthoc_xai/utils/normalization.py`. Applied to all cached category importance data
(event_xai JSONs + scenario002 CSV). Zero compute — pure post-processing.

### The math
Current pipeline: `cat_imp[c] = Σ|raw[c]| / Σ|raw_all|` — sums over ALL features.
Size-corrected: `corrected[c] = (cat_imp[c] / n_c) / Σ(cat_imp[c'] / n_c')`.

### Correction factors (relative to GPS = 20 features, the smallest)
| Category | Features | Multiplier |
|---|---|---|
| GPS path | 20 | ×1.000 |
| SDC trajectory | 40 | ×0.500 |
| Traffic lights | 275 | ×0.073 |
| Other agents | 320 | ×0.063 |
| Roadgraph | 1,000 | ×0.020 |

Roadgraph gets divided by 50× relative to GPS.

### Key result: IG at event_02 peak (t=35)
| Category | Original | Size-corrected |
|---|---|---|
| GPS path | 7% | **51%** |
| Roadgraph | **55%** | 8% |
| Other agents | 29% | 12% |
| SDC | 9% | 29% |

### What to write in the thesis

> **Methodological note (recommended):** The default L1-aggregation sums attribution over
> all features in a category. Larger categories (roadgraph: 1,000 features) accumulate
> more attribution than smaller ones (GPS: 20 features) by construction. We apply a
> per-feature size correction dividing each category's sum by its feature count before
> renormalizing. Under this correction, roadgraph drops from ~55% to ~8% while GPS rises
> to ~51%.
>
> We use **total attribution as the primary metric** throughout this chapter because the
> model processes categories holistically — the 1,000 roadgraph features form a unified
> geometric representation, not 1,000 independent votes. The size-corrected view answers
> a different but complementary question: *which category has the most influence per input
> dimension?* The answer (GPS dominates per-feature) reflects the fact that each GPS
> waypoint is a compact, high-information signal (target coordinates), while each roadgraph
> point contributes a small fraction of the overall lane geometry. Both perspectives are
> valid; the distinction matters for interpretation.

### Decision
Primary metric: **original total attribution**. Size-corrected version: reported as
methodological validation note. Strengthens rather than undermines the roadgraph claim.

---

---

## Phase 1b — Attention Aggregation Alternatives (implemented 2026-04-24)

### What we did
Added `posthoc_xai/utils/attention_aggregation.py` with three aggregation strategies
for collapsing the Perceiver's `(16, 280)` cross-attention matrix into a 5-dim
category vector. Also added `query_entropy()` for query specialization analysis.

| Strategy | How it works | Motivation |
|---|---|---|
| **mean** | Average over 16 queries per token | Current default; simple baseline |
| **maxpool** | Max over 16 queries per token | Captures "at least one query cares" |
| **entropy-weighted** | Weight queries by sharpness (1/entropy) | Focused queries matter more |

Gradient-weighted (weight by `|∂output/∂latent_k|`) is deferred to Phase 1c —
requires architectural access to the latent vectors mid-network.

### Results (run 2026-04-24 on platform_cache, 2 models × 3 scenarios × 80 timesteps)

**MAD from mean-pool:**
| Category | maxpool | entropy-weighted |
|---|---|---|
| SDC | 0.0448 | 0.0150 |
| Agents | 0.0198 | 0.0079 |
| Road | 0.0689 | 0.0157 |
| Traffic lights | **0.1022** | 0.0070 |
| GPS | 0.0395 | 0.0073 |
| **Overall** | **0.0551** | **0.0106** |

Entropy-weighted ≈ mean-pool (no reason to use it). Max-pool deviates meaningfully,
especially for TL and roadgraph — driven by specialized queries being diluted in the mean.

**Decision:** mean-pool as primary (consistency with reward_attention paper), maxpool as
robustness check in the correlation experiment.

### Query specialization — new thesis finding

| Model | Mean entropy | Min entropy (query) | Queries below 60% max |
|---|---|---|---|
| complete | ~4.7 bits / 8.13 max | ~3.1 bits (q3) | 7–8 / 16 |
| minimal | ~4.4 bits / 8.13 max | **1.62 bits (q1)** | 6–13 / 16 |

Queries are NOT diffuse — mean entropy is only 57% of maximum. The minimal model shows
greater query specialization than the complete model (up to 13/16 queries focused below
60% max). This is a reward-design effect: the minimal model's GPS-heavy policy concentrates
its representational bandwidth more sharply than the complete model.

**This is worth 2–3 sentences in the thesis** as an exploratory finding supporting
the reward-conditioned attention narrative.

### What to write in the thesis

> **Methodological note — aggregation:**
> We evaluated three aggregation strategies for the Perceiver's 16 query vectors: mean-pool,
> max-pool, and sharpness-weighted averaging. Entropy-weighted pooling was nearly identical
> to mean-pool (overall MAD=0.011). Max-pooling differed more substantially (MAD=0.055),
> particularly for traffic lights (0.102) and roadgraph (0.069). We use mean-pooling as
> the primary metric for consistency with prior work and report max-pool as a robustness check.

> **Exploratory finding — query specialization:**
> Mean query entropy was 4.6 bits (57% of the 8.13-bit maximum), indicating that the
> Perceiver's learned queries do not attend uniformly. In the complete model, query 3
> consistently showed the sharpest focus (entropy ≈3.1 bits). In the minimal model,
> query 1 was dramatically specialized (entropy as low as 1.62 bits, 20% of maximum),
> with up to 13 of 16 queries falling below 60% of maximum entropy. That the minimal
> model exhibits greater query specialization than the complete model suggests reward
> configuration shapes not only overall attention allocation but also the internal
> organization of the latent query representations.

---

## Phase 1c — Attention Rollout (implemented 2026-04-24)

### What we did
Extended `_extract_attention()` in `perceiver_wrapper.py` to additionally capture
self-attention Q/K projections (`lq_attention > self_attn > Dense_0/1`) and compute
attention rollout through all 4 self-attention layers.

**New keys returned per forward pass:**
- `self_attn_layer_{0..3}` : `(B, 16, 16)` per self-attention layer (queries → queries)
- `cross_attn_rollout`     : `(B, 16, 280)` rollout-corrected effective attention

**Rollout formula:**
```
A_eff[l] = 0.5 * I  +  0.5 * A_self[l]   # residual correction
R         = A_eff[3] @ A_eff[2] @ A_eff[1] @ A_eff[0]   # (16, 16)
rollout   = R @ cross_attn_avg             # (16, 280)
```

**Verified:** rollout is row-stochastic (rows sum to 1), math correct.

**Approximation note:** treats cross-attention as a single operation (using layer-averaged
cross_attn_avg) rather than tracking the interleaved cross+self structure per block.
Appropriate for category-level analysis; state clearly in the thesis.

### Results (run 2026-04-24, 2 models × 3 scenarios × 80 timesteps)

| Category | Overall MAD | Note |
|---|---|---|
| Traffic lights | 0.063 | Substantial — up to 0.115 in minimal s3 |
| Road | 0.047 | Meaningful |
| GPS | 0.025 | Moderate |
| SDC | 0.018 | Negligible |
| Agents | 0.012 | Negligible |
| **Global** | **0.033** | Moderate zone |

The categories most affected (TL, road) are the same ones with highest query
specialization (Phase 1b) — confirms that specialized queries are being partially
diluted by mean-pooling, and rollout corrects this.

**Decision: rollout as primary for Phase 2 (correlation study).**
Raw `cross_attn_avg` stays in the reward_attention paper (already submitted).
The two studies use different signals — note this explicitly in the thesis.

### What to write in the thesis

> **Methodological note — attention rollout:**
> The Perceiver processes inputs through 4 interleaved blocks of cross-attention
> (queries attend to input tokens) and self-attention (queries attend to each other).
> Raw cross-attention weights do not account for the information mixing in
> self-attention. We implemented attention rollout (Abnar & Zuidema 2020): for each
> self-attention layer l, we compute the residual-corrected matrix
> `A_eff[l] = 0.5*I + 0.5*A_self[l]`, chain these across 4 layers to obtain a
> routing matrix R ∈ ℝ^{16×16}, and apply R to mean cross-attention to recover
> effective token-level influence. The category-level MAD between rolled-out and
> raw attention was 0.033 globally, with the largest differences in traffic lights
> (MAD=0.063, up to 0.115) and roadgraph (MAD=0.047) — the same categories showing
> highest query specialization. We use rollout-corrected attention as the primary
> signal in the attribution correlation study. Note: the reward-conditioned attention
> study (Chapter X) used raw cross-attention for historical consistency; differences
> between the two studies in TL and roadgraph fractions reflect this methodological
> distinction.

---

## Phase 2 — Correlation Pilot Results (2026-04-24)

### Setup
2 models × 3 scenarios × 80 timesteps = 480 total.
Attention: rollout-corrected. Methods: IG (path-integrated), GxI (gradient×input), VG (local gradient).
Metric: per-timestep Kendall τ between 5-dim attention ranking and attribution ranking.

### Combined results

| Method | Complete τ | Minimal τ | Combined τ>0 |
|---|---|---|---|
| IG | 0.058 | 0.120 | 50% |
| GxI | 0.333 | 0.210 | 80% |
| VG | 0.453 | 0.207 | 82% |
| IG_vs_VG (calibration) | 0.537 | 0.580 | 92% |

### The outlier: scenario 2
Both models collapse to near-zero τ in scenario 2 (the "frozen SDC" scenario — 0% route
completion, 8 surrounding agents, all near-misses). Flat attention distribution across
timesteps makes rank correlations meaningless. This scenario contaminates combined averages.

**Scenario 1 without outlier:** IG τ = 0.305–0.388, VG τ = 0.305–0.490 — moderate genuine agreement.

### Key finding: VG agrees with attention more than IG
Complete model: VG (0.453) >> IG (0.058). The IG_vs_VG calibration is 0.537, meaning
IG and VG themselves agree moderately — yet attention aligns far better with VG.
Root cause: zero-baseline for IG. In scenario 2, IG τ = -0.317 while VG τ = +0.427
in the same scenario. The path from zero to observation crosses semantically wrong
regions in that scenario, inverting the IG category rankings.

**Implication:** For the "Is Attention Explanation?" study, attention aligns with local
gradient sensitivity (VG) more than path-integrated importance (IG). This is interpretable:
both attention and VG reflect the model's current operating point; IG reflects something
more global that is corrupted by the wrong baseline.

### Connection to Jain & Wallace
Mean attention-IG τ = 0.058–0.120 — weak overall, consistent with their NLP finding
(low attention-gradient correlation). BUT in behaviorally rich scenarios (s1): τ ≈ 0.3–0.4,
suggesting the relationship is scenario-dependent. More nuanced than a binary "is/isn't."

### Go/No-Go for Phase 3
**GO** — signal exists in good scenarios. Phase 3 priorities:
1. Stratify by scenario type (frozen/edge-case excluded or flagged)
2. Stratify by risk level — does τ increase during high-risk moments?
3. Add SARFA — expected to align better with attention than VG (both track action-relevance)
4. Scale to 50 scenarios to get statistical power beyond 3 anecdotal scenarios

### What to write in the thesis

> **Pilot results — attention-attribution correlation:**
> Across 480 timesteps (2 models × 3 scenarios), mean Kendall τ between rollout-corrected
> Perceiver attention and IG attribution was 0.058 (complete) and 0.120 (minimal), consistent
> with Jain & Wallace (2019) who found low attention-gradient correlation in NLP models.
> However, in scenarios with clear behavioral dynamics (scenario 1), agreement was moderate
> (τ_IG = 0.305–0.388), suggesting attention is informative during behaviorally rich episodes.
> Notably, attention agreed more strongly with vanilla gradient (τ_VG = 0.207–0.453) than
> with path-integrated gradients (τ_IG = 0.058–0.120). We attribute this to the zero-vector
> IG baseline, which in normalized observation space does not represent an uninformative scene
> and corrupts the integration path for certain scenario types. Scenario 2 — a stationary
> vehicle surrounded by 8 simultaneous near-misses — produced near-zero or negative τ for
> all methods, indicating that flat attention distributions in edge-case scenarios confound
> rank-based correlation metrics. We exclude such scenarios in the full-scale analysis (Phase 3)
> and report risk-stratified results.

---

## Implementation Constraints (apply from Phase 2 onwards)

### IG baseline
Current IG uses a zero-vector baseline. In V-MAX's normalized observation space,
zero ≠ "empty scene" — it means "all features at the center of the range" (e.g.
agent at ego position with zero velocity). This is a known limitation. Acknowledge
in the thesis; sensitivity analysis with a mean-observation baseline is future work.
Do not block Phase 2 on this.

### GPU efficiency — JAX-native batching
- All forward passes must be **batched**: pass `(T, obs_dim)` tensors to `model.forward()`
  in one call rather than looping over timesteps in Python. Already done in phase1c.
- For IG specifically: the path-integral requires `n_steps` forward passes per observation.
  Use `jax.vmap` over the interpolation path (or `lax.scan`) instead of a Python loop.
  This vectorizes the entire IG computation over both the batch dimension AND the
  n_steps dimension, maximising GPU utilisation.
- For the scale-up (Phase 3, 3700 timesteps): use `lax.scan` for the timestep loop
  instead of Python `for t in range(T)` — avoids Python overhead and keeps everything
  on-device. The existing event_xai experiment uses `lax.scan` for rollouts; reuse that
  pattern.

### Practical note on GTX 1660 Ti (6GB)
- Batch size for IG: n_steps=50 × batch=80 timesteps = 4,000 forward passes per scenario.
  May need to chunk into sub-batches of 20–40 timesteps if OOM. Profile first.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` is already set by the model loader.

---

## Up Next

- **Phase 2:** Pilot attention-vs-IG correlation on platform_cache
  Canonical attention signal: rollout (decided in Phase 1c, global MAD=0.033)
