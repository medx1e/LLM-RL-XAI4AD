# Query Specialization — Research Notes
> Cross-reference: Phase 1b results, reward_attention paper, future work candidates.
> Update this file whenever new evidence appears.

---

## What We Found (Phase 1b, 2026-04-24)

Measured Shannon entropy of each query's attention distribution across 2 models × 3 scenarios × 80 timesteps.

| Model | Mean entropy | Min entropy | Query | Queries < 60% max | Max possible |
|---|---|---|---|---|---|
| complete_42 | ~4.7 bits | ~3.1 bits | q3 | 7–8 / 16 | 8.13 bits |
| minimal_42 | ~4.4 bits | **1.62 bits** | q1 | 6–13 / 16 | 8.13 bits |

Key numbers to remember:
- 60% of max entropy = 4.88 bits. Below this = meaningfully focused.
- Query 1 of the minimal model at 1.62 bits = **20% of maximum** — attending to ~3–5 tokens out of 280.
- This was consistent across all 3 scenarios for both models — not a one-off.

---

## Why This Matters

### 1. It is evidence the Perceiver learned meaningful internal structure

The 16 queries are initialized randomly and trained end-to-end with no explicit supervision on what they should attend to. The emergence of focused queries — especially one with 80% less entropy than the maximum — means the training process found it useful to dedicate specific queries to specific input regions. This is non-trivial and publishable as an observation.

### 2. It directly extends the reward-conditioned attention story

The reward_attention paper shows that reward configuration shapes the **aggregate** attention allocation (GPS gradient, budget reallocation reversal). The query specialization finding adds a deeper layer: reward configuration also shapes the **internal organization** of the 16 latent queries.

- Complete model (TTC penalty): moderate specialization, 7–8 queries focused. The TTC penalty forces the model to process multiple input types (agents, road, GPS) simultaneously → queries stay more distributed.
- Minimal model (no TTC): higher specialization, up to 13 queries focused, with one extremely sharp query (1.62 bits). The minimal model is GPS-dominant → the GPS-focused query becomes very sharp because it carries most of the navigation signal.

**One sentence for the paper:** "The minimal model's reward configuration, which lacks a TTC penalty and produces GPS-dominant attention allocation, also induces greater internal query specialization — up to 13 of 16 queries fall below 60% of maximum entropy, compared to 7–8 in the complete model."

### 3. It is a methodological concern for attention-IG correlation

If queries specialize, then mean-pooling over all 16 queries when computing attention-as-attribution is *lossy*. The correlation between mean-pooled attention and IG may be moderate simply because most queries are irrelevant to the category being compared — only the specialized query matters. This motivates:
- Running the attention-IG correlation with max-pool (captures the most focused query per token) in addition to mean-pool
- Checking whether ρ(attention_maxpool, IG) > ρ(attention_mean, IG) — if yes, specialization is real and max-pool is the right comparator

### 4. Connection to the existing "poor results" section

You mentioned having a query specialization section with poor results. The likely issue: you were probably trying to identify WHICH category each query specializes in (e.g., "query 3 = roadgraph query, query 7 = GPS query") by looking at which token range receives the most attention per query. This is hard because:
- Categories have very different sizes (200 roadgraph tokens vs 5 SDC tokens) — a query attending uniformly will always seem to "specialize" in roadgraph just from count
- At any single timestep, the specialization may shift (a query might focus on agents during a hazard and road during calm driving)

The entropy approach sidesteps both problems — it doesn't ask "what does this query attend to?" but rather "how focused is this query, regardless of where it points?" The entropy finding is robust even without knowing the semantic content of the focus.

**How to rescue the existing section:** Combine entropy-based evidence (what we now have) with a time-averaged "attention fingerprint" per query (average attention to each of the 5 categories across all timesteps → 16 × 5 matrix, row-normalize → cluster rows). This gives a visual representation of each query's "specialty" that is more interpretable than looking at raw token-level attention.

---

## What the Existing Section Probably Needs

If your current query specialization section has poor results, it likely needs:

1. **Entropy as the primary metric** (not raw attention to categories) — this is what we now have
2. **Cross-scenario consistency** — show that the same queries are focused across different scenarios (we see q3 for complete and q1 for minimal are consistently focused across all 3 scenarios → strong evidence)
3. **Reward-config comparison** — complete vs minimal entropy distributions side by side (we have this)
4. **Temporal stability** — are the focused queries the same at t=0 vs t=79? (not yet checked, easy to add from Phase 1b data)

---

## Potential Future Work / Paper Ideas

### A. "Reward-Induced Query Specialization in Perceiver-based RL Agents"

Core finding: reward config shapes query specialization, not just aggregate attention.
Method: entropy analysis across multiple configs (minimal/basic/complete) and seeds.
Compute cost: entropy is computed from cached attention — zero compute beyond Phase 1b.
Missing: basic model entropy + seeds 69/99. Basic model pkl exists (157 timesteps). Seeds require model reload.

### B. "Identifying Functionally Specialized Queries in the Perceiver"

Goal: go beyond entropy → identify WHICH category each query focuses on.
Method:
1. For each query k, compute mean attention to each of 5 categories across all timesteps → "attention fingerprint" vector (5-dim)
2. Cluster the 16 fingerprints → natural groupings emerge
3. Validate: during hazard events, do "agent-focused" queries increase their entropy (attention spreads when agents become collectively important) or decrease (narrow to the specific threatening agent)?
Missing: need temporal attention data at event timesteps (requires Phase 2/3 infrastructure).

### C. Add to reward_attention paper (if revision happens)

The query specialization finding is a cheap addition to the reward_attention paper. It requires only:
1. Computing entropy from the already-cached TimestepData attention fields — but wait, TimestepData only stores aggregated attention (5 categories), NOT the raw (16, 280) matrix. So we can't compute per-query entropy from the existing pkl.
2. We CAN compute it from the platform_cache attention pkl (which stores cross_attn_avg shape (1, 16, 280)) — already done in Phase 1b.
3. For the full 50-scenario version: would need to re-run the experiment with raw attention storage. This is a non-trivial change to the experiment pipeline.

**Verdict:** For the current chapter, report the 3-scenario entropy finding as an exploratory observation. For a paper revision or follow-up, run the 50-scenario version.

---

## Numbers to Cite

When writing the thesis or paper, use these exact numbers (from Phase 1b run):

```
Max possible entropy:        8.13 bits  (log2(280))
60% threshold:               4.88 bits

Complete model (3 scenarios, 80 ts each):
  Mean entropy:              4.64–4.79 bits
  Consistently focused query: q3  (~3.1 bits, 38% of max)
  Queries below 60% max:     7–8 / 16

Minimal model (3 scenarios, 80 ts each):
  Mean entropy:              4.27–4.58 bits
  Consistently focused query: q1  (1.62–2.18 bits, 20–27% of max)
  Queries below 60% max:     6–13 / 16  (s3 = most specialized)
```

---

## One-Paragraph Summary (ready to paste into thesis)

> Analysis of per-query Shannon entropy reveals that the Perceiver's 16 learned queries
> do not attend uniformly to the 280 input tokens. Mean query entropy was 4.6 bits
> (57% of the 8.13-bit maximum), and specific queries showed consistent, strong
> specialization across all analyzed scenarios. In the complete model (with TTC penalty),
> query 3 was the most focused (entropy ≈3.1 bits, 38% of maximum), with 7–8 of 16
> queries falling below 60% of maximum entropy. In the minimal model (without TTC
> penalty), specialization was more pronounced: query 1 reached a minimum entropy of
> 1.62 bits (20% of maximum), and up to 13 of 16 queries fell below the 60% threshold.
> This suggests that reward configuration shapes not only the aggregate attention
> allocation — as reported in the reward-conditioned attention study — but also the
> internal representational structure of the latent queries. The minimal model's
> GPS-dominated policy appears to concentrate representational bandwidth more sharply,
> while the complete model's richer reward signal produces more distributed query usage.
> We note this as an exploratory finding; a systematic query specialization study across
> all reward configurations and seeds is left as future work.
