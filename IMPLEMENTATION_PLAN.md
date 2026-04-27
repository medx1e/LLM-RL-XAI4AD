# Implementation Plan — Post-Hoc XAI Thesis Chapter

> **Scope:** Deepen the post-hoc attribution chapter by adding one new research
> contribution (attention-as-attribution faithfulness) plus methodological
> validation (size-corrected normalization, better attention aggregation).
>
> **Not in scope:** Multi-seed replication, cross-architecture study, Granger
> causality, signed attribution paper. These are either for the already-submitted
> RCA paper or for separate future work.

---

## Research Framing

The chapter's main new contribution will be positioned as a cross-domain test of
the *"Is Attention Explanation?"* debate (Jain & Wallace 2019, Wiegreffe & Pinter
2019, Bibal et al. 2022). The debate originated in NLP with RNN self-attention on
classification tasks. We extend it to **RL + autonomous driving + cross-attention
(Perceiver)** — a genuinely new setting that may produce different conclusions.

Core research question: **Does Perceiver cross-attention agree with gradient-based
feature importance in driving RL, and does agreement depend on risk level?**

---

## What We Already Have

- `attention_gradient_correlation()` function in `posthoc_xai/metrics/faithfulness.py` — implemented, **never run on real data**
- Cached attention series for 3,676 timesteps (complete model) and similar for minimal/basic, but **no cached gradient attribution at those timesteps**
- Cached VG+IG attribution at 3 events (event_xai experiment) — ~45 timesteps total, but **attention not saved in those JSONs**
- All 7 attribution methods at scenario 002 t=35 (single timestep only)
- Full experiment pipeline infrastructure (scanner, analyzer, reporter, resume-friendly)

**Implication:** we currently have attention OR gradient attribution at scale, but never both at the same timesteps in a reusable form. Phase 2 will address this.

---

## Phase 0 — Audit & Setup (½ day)

**Goal:** confirm exactly what is cached and what is missing before touching code.

1. Enumerate all cached attention and attribution data across the three experiments (event_xai, reward_attention, scenario002_all_methods) and document timestep overlap
2. Decide whether to reuse reward_attention's cached attention for the correlation experiment or to recompute everything end-to-end in a single new pipeline
3. Confirm that Perceiver self-attention weights (layers 1–4) are actually extractable — this determines whether attention rollout is feasible

**Go/no-go check:** if self-attention weights are not captured, attention rollout drops out of the plan; we proceed with cross-attention only.

---

## Phase 1 — Methodological Validation (1–2 days, zero compute)

Purely post-processing on cached arrays. These are chapter *validation* steps, not standalone contributions, but they either confirm or qualify existing findings before we build new claims on top.

### 1a. Size-corrected category normalization
Current aggregation sums absolute attribution within a category, which may inflate the roadgraph (1,000 features) vs GPS (20 features) comparison. Add a size-corrected variant that divides by category feature count before renormalizing. Re-generate the category importance numbers for the key existing findings and check whether the roadgraph-dominates conclusion holds.

**Expected outcome:** either confirms the existing finding ("roadgraph dominance is real even after correction") or qualifies it ("dominance is partly a count artifact, with GPS and agents taking a larger share under correction"). Either outcome is chapter-worthy as a methodological note.

### 1b. Attention aggregation alternatives
Currently we average attention across the 16 Perceiver queries before category aggregation. Add three alternatives:
- **Max-pool across queries** (standard in vision transformer interpretability)
- **Attention-weighted pooling** — weight each query's contribution by how much that query's latent representation influences the output (the theoretically most motivated option)
- **Per-query analysis** — keep the 16 queries separate and look for query specialization patterns

Decide which aggregation to use as the "canonical" attention signal for the Phase 2 experiment based on which gives the most stable (least noisy) category distribution.

### 1c. Attention rollout (conditional on Phase 0 check)
If self-attention weights are accessible, implement rollout through the 4 self-attention layers. Compare rolled-out attention vs raw cross-attention on a few timesteps to see whether rollout meaningfully changes the category distribution. If the difference is substantial, rollout becomes the canonical signal for Phase 2.

---

## Phase 2 — Pilot: Attention-vs-IG Correlation on Cached Events (1–2 days)

**Goal:** a small-scale, zero-compute pilot of the main experiment, using the 3 events already cached in the event_xai experiment. Establish the methodology and produce a preliminary signal before committing to larger compute.

1. At each of the ~45 cached event timesteps, extract the 5-dim category importance vector for attention (via the best aggregation from Phase 1b) and for IG
2. Compute per-timestep Pearson ρ and Kendall τ between the two vectors
3. Add calibration baselines: ρ(VG, IG), ρ(SARFA, IG) where SARFA is already cached at those timesteps if available (check in Phase 0)
4. Plot the distribution of per-timestep correlations with a dashed line marking the VG-IG baseline

**Decision point:** if attention-IG correlation is consistently near zero or consistently near 1 across all 45 timesteps, the signal is either uninteresting or trivial. If it varies across events (e.g., high during hazard peak, low during calm) then scaling up to Phase 3 is justified.

---

## Phase 3 — Scale-Up: Risk-Stratified Correlation (3–5 days compute)

**Run only if Phase 2 shows a signal worth scaling.**

**Goal:** compute attention-IG agreement across the full ~3,676 timesteps per model that reward_attention already analyzed, stratified by risk level, producing the chapter's main new figure.

1. Reload each Perceiver model and compute IG attribution at every timestep where reward_attention cached attention data (this is the main compute cost — roughly 1 day per model × 3 models on the GTX 1660 Ti)
2. For each timestep, compute ρ(attention, IG) at category level
3. Stratify results by collision_risk bucket (calm: risk < 0.2, moderate: 0.2–0.6, high: > 0.6) using the risk values already cached
4. Test whether ρ is significantly higher in high-risk buckets than in calm buckets (Wilcoxon rank-sum, within-episode Fisher z-aggregate — reuse the RCA paper's statistical infrastructure)
5. Compare across the three reward configs (minimal / basic / complete) — does attention-IG agreement also depend on reward design?

**Chapter-level deliverables from this phase:**
- Figure: ρ(attention, IG) distribution stratified by risk, with VG-IG baseline
- Figure: ρ comparison across reward configs
- One-number headline: "Perceiver attention–IG correlation is ρ = X during high-risk timesteps vs ρ = Y during calm, p < 0.001"

---

## Phase 4 — Faithfulness via Deletion Curves (2–3 days, optional)

**Only if Phase 3 results are ambiguous or reviewers would demand it.**

Correlation alone does not prove attention is a *faithful* importance signal. The stronger test is feature-removal faithfulness: delete features in order of attention rank vs IG rank vs random, measure how fast the model's output degrades.

1. Pick a subsample of ~200 timesteps balanced across risk buckets
2. Run deletion curves using three orderings: attention-ranked, IG-ranked, random-ranked
3. Compare deletion AUCs — if attention-ranked AUC is within 10% of IG-ranked AUC, attention is functionally faithful even where correlation is moderate

This is the test Bastings & Filippova (2020) argue is more meaningful than correlation.

---

## Phase 5 — Writeup & Chapter Integration (3–5 days)

Fold the new results into the existing post-hoc attribution chapter.

**Proposed chapter structure:**
1. Existing content — 7-method framework, method divergence camps (3 camps finding), temporal attribution arc, event-conditioned analysis
2. *New section* — methodological validation (size-corrected normalization results from Phase 1a)
3. *New section* — attention-as-attribution: does Perceiver attention agree with IG? (Phase 2 + 3 results)
4. *New section* — faithfulness comparison via deletion curves (Phase 4, if run)
5. Discussion — position findings relative to Jain & Wallace debate; discuss what makes driving RL different from NLP

---

## Honest Risk Assessment

- **Phase 2 could produce a null result.** If attention-IG correlation is uniformly near 0.5 with no risk-dependence, there is no narrative. Mitigation: Phase 1 work still produces methodological content for the chapter; Phase 2 becomes a short "what we tried" section.
- **Phase 3 compute could slip.** JIT compilation + IG at 3,676 timesteps × 3 models is optimistic on a 6GB GPU. Allocate buffer or run on a subset (e.g., 500 timesteps per model with risk-balanced sampling).
- **Phase 4 may conflict with platform work.** The Streamlit platform precompute script is also resource-intensive. Schedule carefully; don't run both experiments in parallel.
- **Chapter scope creep.** Every idea in the research brief is interesting. Stay disciplined — Phases 1–3 are the core; 4 is optional; anything beyond is outside this chapter.

---

## Timeline Summary

| Phase | Effort | Compute | Deliverable |
|---|---|---|---|
| 0. Audit | ½ day | none | Decision on data reuse + rollout feasibility |
| 1. Methodological validation | 1–2 days | none | Size-corrected findings, chosen attention aggregation |
| 2. Pilot correlation | 1–2 days | none | Go/no-go signal for Phase 3 |
| 3. Scale-up (if go) | 3–5 days | ~3 days GPU | Main chapter figure + stratified ρ numbers |
| 4. Deletion curves (optional) | 2–3 days | ~1 day GPU | Faithfulness validation |
| 5. Writeup | 3–5 days | none | Updated chapter sections |

**Total realistic duration: 2–3 weeks** with some parallelism between writeup and compute, assuming no major engineering surprises.
