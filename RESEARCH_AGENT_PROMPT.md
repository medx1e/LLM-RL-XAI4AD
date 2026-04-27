# Research Agent Prompt

---

## Context

I am a researcher working on a final-year project on **explainable reinforcement learning for
autonomous driving**. I have built a complete post-hoc XAI framework and run experiments on
pretrained RL driving agents (V-MAX, trained with SAC on Waymo Open Motion Dataset). The full
technical and empirical context is in the attached document `XAI_RESEARCH_CONTEXT.md`.

Read that document fully before answering. The key things to know:
- We have 7 gradient/perturbation attribution methods + Perceiver attention extraction
- We have 35+ pretrained models (5 architectures, 3 reward configs, 3 seeds)
- We have ~3,700 analyzed timesteps across 50 driving scenarios with computed attributions and attention
- We have specific empirical findings (see Sections 4.1 and 4.2 of the context doc)
- A paper is being written on reward-conditioned attention; we are looking to make it stronger and
  to identify new publishable contributions

---

## What I Need From You

Produce a structured research brief with the following sections. Be specific, technical, and
actionable — not generic. Everything you suggest should be runnable with the infrastructure we
already have (the code is JAX-based; data is already cached as JSON and pkl files).

---

### Section 1: Attribution Normalization & Post-Processing

Critically analyze our current normalization pipeline: we compute `abs(raw_attribution)` then
normalize to sum 1. Identify specific failure modes. Then suggest concrete normalization or
transformation strategies (e.g., softmax temperature, rank-based, log-scale, z-score, signed
attribution, baseline subtraction) that could improve faithfulness or cross-method agreement.
For each: explain the theoretical motivation, describe the exact transformation, and identify what
experiment would validate it (e.g., does it improve deletion-curve AUC? does it improve
attention-gradient correlation?).

---

### Section 2: Connecting to the "Is Attention Explanation?" Literature

The Jain & Wallace (2019) / Wiegreffe & Pinter (2019) debate on whether attention is a valid
explanation is directly relevant to us: we have Perceiver attention weights AND gradient attribution
for the same timesteps and scenarios.

1. Summarize the key papers in this debate (Jain & Wallace 2019, Wiegreffe & Pinter 2019, and any
   important follow-ups including domain-specific work, especially in RL or sequential decision-making)
2. Describe the specific tests they used (attention-gradient correlation, counterfactual attention,
   etc.) and identify which of those tests we can run with our existing code
3. Propose the exact experiment design: which models, which timesteps, which XAI method pairs,
   what correlation metric, what statistical test, what would constitute evidence for or against
   attention-as-explanation in our driving setting
4. Identify what is *novel* about the driving RL context vs NLP (where the debate originated) —
   what are the unique structural features of our setup that could produce different conclusions?

---

### Section 3: Ideas to Strengthen the Reward-Conditioned Attention Paper

We have these strong findings: budget reallocation reversal (+38.2% vs −16.6% between complete and
minimal reward configs), GPS gradient (2× GPS attention difference), within-episode ρ=+0.291 for
complete model, vigilance gap (+134%), and a Simpson's paradox demonstration.

The paper currently covers only Perceiver architecture, 2 reward configs (minimal and complete),
seed 42 only.

1. What experiments would most efficiently strengthen the paper given our infrastructure? Prioritize
   by expected impact vs compute cost (each model run takes ~30 min on a GTX 1660 Ti, 6GB VRAM).
2. Are there specific statistical tests or framing choices that would make the reward-attention
   causal argument stronger? (We currently have correlational evidence; what would approach causal?)
3. Suggest a framing for the lead-lag finding (attention leads risk by ~2 steps, but wide spread).
   What analysis or subgroup would sharpen this claim?
4. The entropy finding (redistribution not narrowing) is unexpected. Are there papers that predict
   or explain this pattern? What does it suggest about the Perceiver's information integration
   mechanism under stress?

---

### Section 4: New Publishable Directions

Given everything in the context document, suggest 2–3 concrete new research directions that:
- Could produce a publishable contribution (workshop paper, conference short paper, or chapter
  contribution) within ~3 months of focused experimentation
- Use existing infrastructure (no new model training required — we only analyze existing weights)
- Go beyond what the current paper covers

For each direction:
- Give it a working title and research question
- Describe the key experiment (what you run, what you measure, what you compare)
- Identify the closest related papers and how this would differentiate
- Estimate compute cost (number of model runs × scenarios)
- Be honest about risks: what might make this fail or produce null results?

Candidate directions to consider (but don't limit yourself to these):
- Attention head specialization in the Perceiver (do the 16 query tokens specialize?)
- Cross-architecture study: do architectures without attention (MTR, Wayformer) show the same
  temporal attribution patterns via gradient methods alone?
- Attribution as a driving scenario fingerprint (semantic consistency under similar event types)
- Faithfulness of attention: running deletion/insertion curves on attention-as-attribution vs IG
- Signed attribution analysis: separating positive and negative contributions (currently we use abs)
- Multi-seed robustness: do our key findings replicate across seeds 42, 69, 99?

---

### Section 5: Technical Gotchas and Methodological Warnings

Based on our experimental findings and limitations (see Section 5.3 in the context doc), identify:
1. Specific methodological mistakes that would invalidate claims if we are not careful
2. Common pitfalls in XAI evaluation literature that we are at risk of (e.g., using faithfulness
   metrics that are circular, over-interpreting spurious correlations, ignoring the effect of
   baseline choice in IG)
3. Any specific concern about how we aggregate Perceiver attention (we average across the 16 query
   heads before comparing to gradient attribution — is this the right aggregation? what alternatives exist?)
4. One technical suggestion for improving the attention extraction that could meaningfully change
   our results (e.g., using layer-wise attention, using attention rollout, using raw logits before
   softmax)

---

### Output Format Requirements

- Use numbered lists inside each section for individual ideas/papers/suggestions
- For each paper cited: give full citation (authors, title, venue, year), 1-sentence summary of
  the finding most relevant to us, and 1 sentence on how we can use or replicate it
- For each proposed experiment: be specific enough that a developer can implement it — describe
  the loop, the metric, the comparison, the expected output
- Be honest about what we do not have: if a suggestion requires something we cannot do (e.g., model
  retraining, new datasets), say so explicitly
- Prioritize: put the highest-impact, most feasible ideas first within each section
- Total length: aim for depth over breadth — 3 great ideas per section beats 10 vague ones
