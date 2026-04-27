# Research Brief: Strengthening the XAI Framework for V-MAX Autonomous Driving

**Prepared for:** FYP Researcher — Explainable RL for Autonomous Driving  
**Date:** April 2026  
**Scope:** Actionable experiments, literature connections, and new directions using existing infrastructure

---

## Section 1: Attribution Normalization & Post-Processing

### 1.1 Diagnosis of the Current Pipeline

Your current normalization is: `normalized = abs(raw) / sum(abs(raw))`. This L1-normalization of absolute values has three specific failure modes in your setup:

1. **Roadgraph magnitude dominance.** The roadgraph category has 1,000 features (200 points × 5) compared to 20 for GPS. Even if per-feature importance is equal, the sum of absolute values for roadgraph will dominate purely from count. Your finding that roadgraph importance is ~52–69% across all methods and models may partly reflect this size artifact rather than true model reliance. The token-normalized attention density you already compute for the RCA paper (â_c = a_c / w_c) proves this concern is real: road graph drops from ~52% raw to 0.73 density while GPS rises from ~7% to 4.56 density. You are *not* applying the equivalent correction to gradient attributions.

2. **Sign cancellation in abs().** Taking absolute values before summing discards the direction of attribution. A feature with raw attribution -0.4 (strongly *decreases* Q-value) is treated identically to +0.4 (strongly *increases* Q-value). This matters for RL: during evasive maneuvers, some features push toward braking while others push toward steering — these are qualitatively different contributions that abs() collapses. Your SARFA finding (agent_0 attribution = 71% at near-miss) likely benefits from SARFA's built-in action-specificity, which partly preserves this information.

3. **Dynamic range collapse across methods.** Your vanilla gradient values span 1e-20 to 0.46 per agent — a 10^19 range. After L1-norm on abs(), the tiny values effectively vanish even though they may be stably nonzero (i.e., the model does attend to them, but weakly). Integrated Gradients, by contrast, integrates over the path and produces a much narrower range. L1-norm after abs() thus produces very different distributional shapes for VG vs IG, making cross-method comparisons misleading even when both methods point to the same category.

### 1.2 Concrete Alternative Transformations

**Strategy A — Per-token (size-corrected) normalization for gradient methods.**

- *Motivation:* Analogous to your token-normalized attention density. Categories with more features accumulate more raw attribution by construction. Dividing by category size before comparing categories creates a fair comparison.
- *Transformation:* For each category c with n_c features: `category_importance_corrected[c] = (1/n_c) * sum(abs(raw[features_in_c]))`. Then renormalize across categories so they sum to 1.
- *Validation experiment:* Compare the rank ordering of categories under size-corrected vs raw L1-norm for IG across all 50 scenarios. Measure: (a) does the corrected version increase agreement with attention density rankings (Spearman ρ between corrected-IG and token-normalized attention)? (b) does it change which category dominates during hazard events? If GPS and agent importance rise under correction, this confirms the count artifact.
- *Cost:* Zero additional model runs — purely post-processing on cached attribution arrays.

**Strategy B — Log-scale transformation before normalization.**

- *Motivation:* Compresses the 10^19 dynamic range of VG, making small-but-nonzero features visible. This is standard in spectral analysis and information retrieval (TF-IDF uses log term frequency).
- *Transformation:* `log_attr[i] = log(1 + abs(raw[i]) / epsilon)`, where epsilon = median of nonzero abs(raw). Then L1-normalize the log_attr vector. The epsilon scaling anchors the log to the typical magnitude, preventing numerical instability.
- *Validation experiment:* Recompute deletion-curve AUC for VG with and without log-scaling. If log-VG achieves lower deletion AUC (higher faithfulness), the transformation is recovering signal that raw VG was squashing. Also check: does log-VG produce agent importance closer to IG during hazard events (reducing the 50–144× underestimation gap)?
- *Cost:* Zero additional model runs. Deletion curves need recomputation (~5 minutes per scenario).

**Strategy C — Signed attribution with positive/negative decomposition.**

- *Motivation:* For SAC (your algorithm), the actor outputs a mean and log-std for acceleration and steering. Features that push acceleration negative (braking) vs positive (accelerating) are functionally different. Collapsing them with abs() loses the action-direction structure. Signed attribution could reveal, e.g., that during hazard onset, agent_0 pushes strongly toward braking while roadgraph features push toward steering to avoid — currently invisible.
- *Transformation:* Decompose `raw` into `pos = max(0, raw)` and `neg = min(0, raw)`. Normalize each separately: `pos_norm = pos / sum(pos)`, `neg_norm = abs(neg) / sum(abs(neg))`. Report both distributions per timestep. At category level: `positive_fraction[c] = sum(pos[c]) / sum(pos)` and analogously for negative.
- *Validation experiment:* During the 3 temporal-XAI events already analyzed (hazard onset, near-miss, evasive steering), plot signed category importance timeseries. Measure whether positive and negative attributions point to different categories (e.g., agents dominate negative attribution = braking signal, roadgraph dominates positive = steering signal). If they do, this is a paper-ready finding on its own: "Signed Attribution Reveals Action-Specific Feature Roles in RL Driving Policies."
- *Cost:* Zero additional model runs. Reprocessing cached raw attribution arrays.

### 1.3 What NOT to Do

- **Softmax temperature normalization** (softmax(raw / τ)): Tempting but dangerous — softmax is permutation-invariant and creates a false competition between features that may not actually compete. In attention, softmax is baked into the mechanism; applying it to gradients imposes an architectural assumption that does not exist.
- **Baseline subtraction for IG normalization:** You already use a zero baseline for IG. Changing the baseline (e.g., to a dataset-mean observation) would change the IG values themselves, not just the normalization — this is a separate experiment worth doing but should not be conflated with post-processing.

---

## Section 2: Connecting to the "Is Attention Explanation?" Literature

### 2.1 Key Papers in the Debate

1. **Jain & Wallace (2019).** "Attention is not Explanation." NAACL 2019, pp. 3543–3556.  
   *Key finding:* Attention weights in NLP models (text classification, QA) are frequently uncorrelated with gradient-based feature importance, and adversarial attention distributions can be found that produce nearly identical predictions.  
   *Relevance to us:* Their attention–gradient correlation test is directly implementable with your `attention_gradient_correlation()` function. Their adversarial attention test is not (it requires retraining), but the correlation test is the most cited result.

2. **Wiegreffe & Pinter (2019).** "Attention is not not Explanation." EMNLP 2019, pp. 11–20.  
   *Key finding:* Proposed four counter-tests — uniform-weights baseline, variance calibration across seeds, frozen-weight diagnostics, and adversarial training protocol — showing that attention *can* be explanatory under proper validation. The adversarial distributions found by Jain & Wallace are often not learnable by a model trained end-to-end.  
   *Relevance to us:* Their multi-seed variance calibration maps directly to your 3-seed setup (42, 69, 99). If attention–gradient correlation is stable across seeds but the attention *distribution* varies, that distinguishes explanation-relevant from explanation-irrelevant variation.

3. **Abnar & Zuidema (2020).** "Quantifying Attention Flow in Transformers." ACL 2020, pp. 4190–4197.  
   *Key finding:* Raw attention weights in multi-layer transformers become unreliable as explanation probes because information mixes across layers. They proposed attention rollout (recursively multiplying attention matrices across layers) and attention flow (max-flow computation) as better approximations of how input tokens influence final representations.  
   *Relevance to us:* Your Perceiver has 4 self-attention layers after the cross-attention. You currently extract only the cross-attention matrix. Attention rollout through the self-attention layers would give a more accurate picture of which input tokens actually influence the final representation that feeds the actor. This is directly implementable with your existing `capture_intermediates` infrastructure.

4. **Bibal et al. (2022).** "Is Attention Explanation? An Introduction to the Debate." ACL 2022, pp. 3889–3900.  
   *Key finding:* Survey paper organizing the debate into necessary vs. sufficient conditions, identifying that attention can be explanatory in specific architectural and task conditions. Argues that the debate's conclusions are domain-dependent and that cross-domain testing is needed.  
   *Relevance to us:* Explicitly calls for testing the debate's conclusions outside NLP. Your driving RL setup is a strong candidate for this — different domain, different architecture (Perceiver cross-attention vs. RNN self-attention), different task (sequential decision-making vs. classification).

5. **Bastings & Filippova (2020).** "The Elephant in the Interpretability Room: Why Use Attention as Explanation?" BlackboxNLP Workshop, EMNLP 2020, pp. 149–155.  
   *Key finding:* Argued that the entire framing is confused because "explanation" is underspecified. Attention is best interpreted as a data-routing mechanism, not a causal explanation. Recommended deletion/erasure tests over correlation tests.  
   *Relevance to us:* Supports using your existing deletion-curve infrastructure as the primary faithfulness metric for attention-as-attribution, rather than relying solely on attention–gradient correlation.

6. **Puri et al. (2020) — SARFA.** "Explain Your Move: Understanding Agent Actions Using Specific and Relevant Feature Attribution." ICLR 2020.  
   *Key finding:* RL-specific attribution method decomposing saliency into specificity (action-discriminative) and relevance (output-affecting). Designed to address the fact that gradient methods in RL conflate "changes the output generally" with "changes which action is chosen."  
   *Relevance to us:* Your finding that SARFA attributes 71% to agents during near-miss while gradient methods attribute to roadgraph suggests SARFA may be the best gradient-side comparator for attention in the RL setting. If attention correlates better with SARFA than with IG, that is evidence that attention in the Perceiver is tracking action-relevant (not just output-sensitive) features.

7. **Greydanus et al. (2018).** "Visualizing and Understanding Atari Agents." ICML 2018 Workshop on Visualization for Deep Learning.  
   *Key finding:* First systematic perturbation-based saliency maps for RL agents (Atari). Found that saliency maps show meaningful game-relevant focus, but did not validate against attention or test faithfulness rigorously.  
   *Relevance to us:* The first XAI-for-RL baseline. Your framework already surpasses this by having 7 methods + attention + faithfulness metrics, but it is the canonical citation for "post-hoc XAI applied to RL."

### 2.2 Tests from the Literature and Your Implementability

| Test | Source | What it Measures | Can You Run It? |
|------|--------|-----------------|----------------|
| Attention–gradient Pearson ρ | Jain & Wallace 2019 | Correlation between attention weights and gradient importance per feature | **Yes** — `attention_gradient_correlation()` exists, untested |
| Attention–gradient Kendall τ | Jain & Wallace 2019 | Rank correlation (more robust to outliers) | **Yes** — trivial modification |
| Adversarial attention | Jain & Wallace 2019 | Can you find a different attention distribution that produces the same output? | **No** — requires retraining or architecture modification |
| Uniform-weights baseline | Wiegreffe & Pinter 2019 | Does uniform attention degrade performance vs. learned attention? | **Partially** — you can zero-out attention variation and measure output change, but this requires a forward pass with modified attention weights |
| Multi-seed variance | Wiegreffe & Pinter 2019 | Is attention–gradient correlation stable across seeds? | **Yes** — seeds 42, 69, 99 available |
| Attention rollout | Abnar & Zuidema 2020 | Does accounting for information mixing across self-attention layers improve attention-as-attribution? | **Yes** — you capture self-attention weights via `capture_intermediates` |
| Deletion curve on attention-as-attribution | Bastings & Filippova 2020 | Is attention a faithful importance signal when tested by feature removal? | **Yes** — deletion curves implemented, just need to use attention weights as the ordering criterion |
| Insertion curve on attention-as-attribution | Bastings & Filippova 2020 | Same but additive | **Yes** |

### 2.3 Proposed Experiment Design

**Experiment: "Is Perceiver Cross-Attention Faithful in Driving RL?"**

*Models:* perceiver_complete_42, perceiver_minimal_42 (already analyzed). Extend to perceiver_complete_69, perceiver_complete_99 for multi-seed validation.

*Timesteps:* Use the ~3,700 timesteps already cached per model. Subsample: (a) all timesteps from 10 high-variation scenarios, (b) all event-onset timesteps from the event catalog.

*Method pairs to compare:*
- Attention (mean over 16 queries, aggregated to 5 categories) vs. IG (category-level)
- Attention vs. VG (category-level)
- Attention vs. SARFA (category-level)
- For calibration: IG vs. VG, IG vs. SARFA (method–method agreement baseline)

*Metrics:*
- Pearson ρ and Kendall τ at category level (5 categories) per timestep — yields a distribution of correlations
- Fisher z-transform aggregation across timesteps (same methodology as your RCA paper — reuse code)
- Stratified by risk level: calm (risk < 0.2), moderate (0.2–0.6), high (> 0.6)
- Deletion AUC: rank features by attention weight, delete in order, measure Q-value drop. Compare to IG-ordered deletion and random-ordered deletion.

*Statistical test:* Paired Wilcoxon signed-rank test comparing per-timestep ρ(attention, IG) vs. ρ(VG, IG). If attention–IG correlation is significantly higher than VG–IG correlation, that is evidence that attention carries explanatory signal beyond what the simplest gradient method provides.

*What constitutes evidence:*
- **For** attention-as-explanation: mean ρ(attention, IG) > 0.5 at category level, and attention-ordered deletion AUC is within 10% of IG-ordered deletion AUC.
- **Against:** ρ(attention, IG) < 0.2 or not significantly different from random, and attention-ordered deletion AUC is close to random-ordered deletion AUC.
- **Nuanced:** ρ is high during critical events but low during calm phases → attention is explanatory when it matters most (event-conditioned faithfulness).

*Expected output:* A figure showing ρ(attention, IG) distribution stratified by risk level, with horizontal lines for ρ(VG, IG) and ρ(SARFA, IG) as calibration baselines.

### 2.4 What Is Novel About Your Setting vs. NLP

The entire Jain & Wallace debate occurred in the context of text classification with RNN-based attention over a single input sequence. Your setup differs structurally in ways that could produce fundamentally different conclusions:

1. **Cross-attention, not self-attention.** Your Perceiver uses cross-attention from learned latent queries to input tokens. The query vectors have no "input identity" — they are purely learned slots. This means attention weights are a direct allocation of representational bandwidth, not a mixing of input representations with themselves. Jain & Wallace's concern about "alternative attention distributions producing the same output" is architecturally less likely when the queries are learned bottlenecks rather than input tokens.

2. **Temporal structure.** NLP classification is a single forward pass. Your driving agent processes a sequence of observations across timesteps. Attention patterns have temporal dynamics (the detect→attend→commit→execute arc). Correlation computed at a single timestep may miss the temporal structure. This motivates your within-episode analysis and creates a unique opportunity: testing whether attention *leads* gradient importance temporally (attention at t predicts which features IG will emphasize at t+k).

3. **Grounded semantics with external ground truth.** In NLP, there is no "correct" attention — we can only compare attention to other importance measures. In driving, you have event labels with causal agent annotations. You can test: does attention to agent_0 increase when agent_0 is annotated as the causal threat? This is a stronger test than anything available in the NLP debate.

4. **Closed budget under a simplex constraint.** Attention weights sum to 1 across all 280 tokens. When attention increases to agents, it must decrease elsewhere. This budget constraint means reward-conditioned differences are zero-sum reallocations, not independent increases — producing the reversal effects you observe and making correlation analysis more complex than in NLP where attention can be arbitrarily concentrated.

---

## Section 3: Ideas to Strengthen the Reward-Conditioned Attention Paper

### 3.1 Highest-Impact Experiments (Ordered by Impact/Cost)

1. **Multi-seed replication (seeds 69 and 99).** The current paper uses only seed 42. Reviewers will immediately ask whether the budget reallocation reversal (+38.2% vs −16.6%) is a seed-specific artifact. Running the same 50-scenario pipeline on perceiver_complete_69, perceiver_minimal_69, perceiver_complete_99, perceiver_minimal_99 would produce 4 additional columns in your main results table.
   - *Cost:* 4 models × 50 scenarios × ~30 min = ~100 hours total. On GTX 1660 Ti, this is ~4 days of continuous computation. This is your highest-ROI investment.
   - *Expected outcome:* If the reallocation direction (positive for complete, negative for minimal) replicates in 2/2 additional seeds, you can report p-values from a permutation test across seeds. If it fails in one seed, you learn something even more interesting (seed-dependent attention strategies).
   - *What to report:* For each seed, report the same within-episode Fisher z-aggregated ρ and the budget reallocation percentage. A forest plot showing effect size ± CI for each seed would be compelling.

2. **Adding the basic reward configuration.** You have three Perceiver configs (minimal, basic, complete) but only compare minimal vs. complete. The basic config is intermediate — it includes partial behavior rewards but not the full comfort/overspeed penalties. If GPS attention and vigilance gap show a monotonic gradient across minimal → basic → complete, that is much stronger evidence of dose-response than a binary comparison.
   - *Cost:* 1 model × 50 scenarios × ~30 min = ~25 hours. 1 day of compute.
   - *Expected outcome:* GPS attention: minimal > basic > complete. Vigilance gap: complete > basic > minimal. Budget reallocation direction: positive for complete, neutral or weakly positive for basic, negative for minimal. A monotonic dose-response curve across 3 reward tiers is far more convincing than a 2-point comparison.

3. **Granger causality test for the lead-lag claim.** You report that attention leads risk by ~2 steps but note the wide spread (std=5.63). A Granger causality test formalizes this: test whether past attention (lags 1–5) significantly predicts current risk after controlling for past risk, and vice versa. This can be run per-episode on the high-variation scenarios.
   - *Transformation:* For each qualifying episode (>40 timesteps, sufficient risk variation), fit two VAR(p) models: (a) risk ~ risk_lags + attn_lags, (b) risk ~ risk_lags only. F-test for the joint significance of the attention lags.
   - *Cost:* Zero additional model runs — purely statistical analysis on cached timeseries. A few hours of coding.
   - *Expected outcome:* If attention Granger-causes risk (i.e., attention shifts predict upcoming risk changes), that is strong evidence of predictive attention allocation. If the reverse is also true (risk Granger-causes attention), you have bidirectional coupling — still interesting but weaker for the "attention leads" claim.

### 3.2 Strengthening the Causal Argument

Your evidence is currently correlational: reward configs differ, attention patterns differ. The causal chain (reward → learned weights → attention pattern) is plausible but not formally established. Three strategies to approach causal language more carefully:

1. **Natural experiment framing.** Since the three reward configs form a strict hierarchy (each is a superset of the previous), the minimal→basic→complete comparison is a natural experiment with a "dosage" manipulation. Frame the paper as: "We exploit the hierarchical reward structure of V-MAX as a natural experiment, where the 'treatment' is the addition of specific reward terms (TTC penalty, comfort penalty) while holding architecture, data, seed, and training procedure constant." This is stronger than saying "we correlate reward with attention."

2. **Per-reward-term attribution.** If you can identify which specific timesteps have non-zero TTC penalty vs non-zero offroad penalty in the complete reward, you could test: on timesteps where TTC penalty is active, is agent attention higher than on timesteps where only offroad penalty is active? This decomposes the reward signal into components and tests each one's attention effect separately. This requires access to the reward function's internal terms per timestep — check whether V-MAX exposes per-component reward in the training logs or if you can compute them from observation state.

3. **Negative control.** Run the same attention analysis on the MLP/None architecture (which has no attention mechanism). Show that gradient-based category importance in MLP does *not* show the same budget reallocation pattern, or shows it more weakly. This would support the claim that the attention mechanism is the *mediator* of the reward-conditioned effect, not just a side-effect of changed policy behavior.

### 3.3 Sharpening the Lead-Lag Finding

The current lead-lag result (attention leads risk by ~2 steps, std=5.63) is noisy enough that a reviewer could dismiss it. To sharpen:

1. **Subgroup by scenario topology.** The wide spread may reflect two distinct subpopulations: (a) scenarios where the threat approaches gradually (attention has time to lead), and (b) scenarios where the threat appears suddenly (attention and risk rise simultaneously or attention lags). Split scenarios by the rise-time of the collision risk signal: "slow onset" (risk takes >10 timesteps from 0.1 to 0.5) vs. "sudden onset" (<5 timesteps). The lead-lag should be stronger and more consistent in slow-onset scenarios.

2. **Event-locked analysis.** Instead of computing lag over the full episode, lock the analysis to a window around each hazard_onset event: [onset-10, onset+10]. Within this window, compute the cross-correlation between attention_agents and collision_risk. This eliminates the long calm phases (where both signals are flat and contribute noise to the lag estimate) and focuses on the transition dynamics.

3. **Report the finding as conditional.** "In slow-onset hazard scenarios, attention reliably leads risk by 2.1 ± 1.3 steps (N=18, p=0.003). In sudden-onset scenarios, no significant lead is detected (N=13, p=0.41)." This is honest, informative, and a stronger claim than the unconditional aggregate.

### 3.4 Interpreting the Entropy Finding

The finding that attention entropy *increases* with risk (ρ=+0.199, redistribution rather than narrowing) is counterintuitive but interpretable. Two relevant literatures:

1. **Distributed coding in neuroscience.** Population coding theory predicts that complex stimuli activate more neurons, not fewer. Under threat, the driving scene becomes more complex (agents need tracking, escape routes need evaluation, road geometry becomes safety-critical). The Perceiver may be doing the computational equivalent of recruiting more representational resources rather than focusing on one thing. This aligns with the finding that the complete model increases agent attention *without* proportionally decreasing other categories' absolute attention — the budget reallocation is relative, but the overall allocation becomes more evenly spread.

2. **Information-theoretic attention.** Goyal et al. (2020, "Inductive Biases for Deep Learning of Higher-Level Cognition," arXiv:2011.15091) discuss the role of attention as an information bottleneck. Under high risk, the optimal information bottleneck may be *wider* (higher entropy) because more input categories become simultaneously relevant. The complete model's TTC penalty effectively tells the encoder: "all of these things can hurt you" — the rational response is to spread attention rather than concentrate it.

3. **What experiment would clarify:** Decompose entropy into between-category and within-category components. If entropy rises primarily *between* categories (road, agents, GPS all become co-important), that supports the distributed coding interpretation. If it rises primarily *within* categories (e.g., all 8 agents become equally attended instead of focusing on the threatening one), that suggests the model is uncertain about which agent is dangerous — a less favorable interpretation. You have per-agent attention data to compute this.

---

## Section 4: New Publishable Directions

### 4.1 Direction 1 — "Attention Faithfulness in Driving RL: A Cross-Domain Test of the Jain & Wallace Hypothesis"

**Working title:** "Is Attention Explanation in Autonomous Driving? Testing the Jain & Wallace Hypothesis Beyond NLP"

**Research question:** Does Perceiver cross-attention correlate with gradient-based feature importance in an RL driving setting, and does this correlation depend on risk level, reward configuration, or driving scenario type?

**Key experiment:**
- For each of the ~3,700 cached timesteps per model (perceiver_complete_42, perceiver_minimal_42), compute attention-as-attribution (mean over 16 queries, aggregated to 5 categories) and IG attribution (category-level).
- Compute per-timestep Pearson ρ and Kendall τ between the two 5-dimensional vectors.
- Run deletion curves using attention-ranked features vs. IG-ranked features vs. random. Compare AUCs.
- Stratify by risk level (calm/moderate/high) and by reward config.
- Add attention rollout (Abnar & Zuidema 2020) through the 4 self-attention layers as an improved attention signal; re-run all comparisons.

**Closest related papers:**
- Jain & Wallace 2019 (NLP, found low correlation) — we test in driving RL.
- Bibal et al. 2022 (survey, called for cross-domain testing) — we answer their call.
- Greydanus et al. 2018 (RL saliency, no attention comparison) — we add the attention–attribution comparison they lacked.

**Differentiation:** First systematic attention-vs-gradient faithfulness comparison in RL for autonomous driving. Novel finding potential: attention may be *more* faithful in cross-attention architectures (Perceiver) than in the self-attention RNNs tested by Jain & Wallace, due to the bottleneck structure of learned queries.

**Compute cost:** 0 additional model runs for the correlation analysis (data cached). Deletion curves: ~50 scenarios × ~80 timesteps × 1655 features (progressive deletion) ≈ significant but parallelizable. Estimate: 2–3 days on GTX 1660 Ti if deletion is batched. Attention rollout computation is nearly free (matrix multiplications on extracted weights).

**Risks:**
- ρ could be moderate (0.3–0.5) across the board, which is publishable but not dramatic. The stratification by risk level is the key: if ρ jumps from 0.3 (calm) to 0.7 (high risk), that is the headline finding.
- Category-level correlation with only 5 dimensions has limited statistical power per timestep. Mitigate by aggregating across many timesteps and reporting distributions.
- If attention rollout does not improve over raw cross-attention, that is still a finding (Perceiver cross-attention is already the bottleneck, so self-attention layers may not change the input-level attribution).

### 4.2 Direction 2 — "Cross-Architecture Attribution Fingerprints: Do Different Encoders Explain the Same Scenario the Same Way?"

**Working title:** "Same Policy, Different Lens: How Encoder Architecture Shapes Attribution Structure in RL Driving"

**Research question:** Given the same driving scenario and reward configuration, do Perceiver, MTR, Wayformer, MGAIL, and MLP produce the same category-level attribution patterns? Do temporal attribution arcs (detect→attend→commit→execute) appear in architectures without attention?

**Key experiment:**
- Select 10 scenarios spanning different event types (2 hazard onset, 2 near-miss, 2 hard-brake, 2 evasive, 2 calm).
- For each scenario, run IG and VG attribution on all 5 architectures under the minimal reward config (all have minimal_42 weights available).
- Compute per-scenario, per-timestep category importance vectors (5D) for each architecture.
- Compute pairwise architecture agreement: Pearson ρ between category importance vectors of Perceiver-IG vs. MTR-IG, Perceiver-IG vs. MLP-IG, etc.
- For the Perceiver, add attention-as-attribution as a 6th "method" to the comparison matrix.
- For temporal analysis: extract the full timeseries of agent-category importance across the episode for each architecture. Compute the onset-lag of the agent-importance spike relative to the hazard_onset event timestamp. Compare across architectures.

**Closest related papers:**
- Charraut et al. 2025 (V-MAX paper) compared architectures on behavioral metrics only. We compare them on *explanatory* structure.
- Adebayo et al. 2018 ("Sanity Checks for Saliency Maps," NeurIPS) showed that some attribution methods produce identical results for trained and untrained models. We extend the sanity-check concept to across-architecture comparison.

**Differentiation:** No prior work compares attribution patterns across multiple encoder architectures in RL. If MTR and Wayformer show the same temporal attribution arc as Perceiver (via gradients alone), that suggests the arc reflects the *policy's learned behavior* rather than the *attention mechanism*. If only the Perceiver shows the arc, attention is doing something unique.

**Compute cost:** 5 architectures × 10 scenarios × 2 methods × ~30 min = ~50 hours. About 2 days.

**Risks:**
- The cross-architecture Waymax registry conflict (cannot load multiple models in one process) adds engineering friction. Workaround: separate processes with JSON caching (you already do this).
- If all architectures produce similar attribution patterns, the finding is "attribution structure is architecture-invariant" — still publishable but less exciting. If they diverge strongly, the finding is richer.
- Event-detector thresholds are calibrated for real-world units, not normalized space. Needs recalibration before running on new scenarios. Estimate 1 day of engineering work.

### 4.3 Direction 3 — "Signed Attribution Reveals Action-Specific Feature Roles in RL Driving Policies"

**Working title:** "Push and Pull: Signed Feature Attribution Decomposes Acceleration and Steering Influences in Autonomous Driving RL"

**Research question:** Do positive and negative attributions (features pushing toward vs. away from the chosen action) point to different input categories, and does this decomposition change during safety-critical events?

**Key experiment:**
- For the Perceiver (complete and minimal, seed 42), compute IG attribution for the acceleration and steering action dimensions separately.
- Decompose each into positive and negative components (Strategy C from Section 1).
- For each timestep: report the category composition of positive attribution (what pushes toward the chosen action?) and negative attribution (what pushes against it?).
- During hazard onset: test whether agents dominate negative attribution for acceleration (pushing toward braking) while roadgraph dominates positive attribution for steering (pushing toward the escape route).
- Compute a "conflict index": the fraction of timesteps where the top category differs between positive and negative attribution. High conflict index during hazard events means the model is resolving competing inputs.

**Closest related papers:**
- SARFA (Puri et al. 2020) decomposes importance into specificity and relevance but not into signed directional contributions.
- Judd et al. 2019 ("Additive Explanations for Anomalies Detected from Multivariate Temporal Data") used signed Shapley values to separate promoting vs. suppressing features. Same idea, different domain.
- Greydanus et al. 2018 used unsigned saliency. Our signed decomposition is strictly more informative.

**Differentiation:** First analysis of signed attribution in RL for driving. The key novelty is connecting the sign of attribution to the *physical meaning* of the action dimension: positive acceleration attribution means "this feature promotes speeding up," which is directly interpretable by a safety engineer.

**Compute cost:** Zero additional model runs if you cache the full raw attribution vector (which you do). Requires modifying the post-processing code to preserve sign. ~1 day of implementation, then analysis on cached data.

**Risks:**
- IG attributions can have noisy signs due to the integration path traversing regions where the gradient flips. This is a known issue with signed IG. Mitigate by using SmoothGrad on top of IG (smooth the signed attribution) or by using GradientXInput (which naturally has a sign from the input values).
- If positive and negative attributions point to the same categories (just with different magnitudes), there is no decomposition story. The experiment should first check whether the positive/negative category distributions are significantly different (e.g., Jensen-Shannon divergence > 0.1) before committing to the full analysis.
- The 2D action space (acceleration, steering) means you need per-action-dimension attribution. IG supports this (just differentiate with respect to one action dimension at a time), but it doubles the computation.

---

## Section 5: Technical Gotchas and Methodological Warnings

### 5.1 Specific Mistakes That Would Invalidate Claims

1. **Comparing attention and IG at different granularities without acknowledging the comparison is structural.** You aggregate attention over 16 query tokens (mean) before comparing to IG. But IG operates on the 1,655-dimensional raw observation, then gets summed to 5 categories. These are different compression paths. If one method assigns 30% to "other_agents" and the other assigns 25%, the 5-percentage-point difference could be within the noise of the aggregation procedure. Always report the standard deviation of per-query attention across the 16 queries — if the query-level variance is high, the mean is a poor summary.

2. **Simpson's paradox can also affect the attention-faithfulness comparison.** You correctly identified Simpson's paradox for pooled vs. within-episode correlation for the RCA paper. The same issue applies to any attention–gradient correlation computed by pooling across episodes. If calm episodes have low attention–gradient ρ and high-risk episodes have high ρ, pooling will underestimate the relationship. Use within-episode analysis for this too.

3. **Circular faithfulness evaluation.** Your deletion curve removes features in order of attributed importance and measures output change. If you use IG-ranked deletion to evaluate IG, the metric is partially circular: IG satisfies the completeness axiom (attributions sum to output difference from baseline), so IG-ordered deletion is guaranteed to be efficient. The valid comparison is *cross-method*: use attention-ranked deletion to evaluate attention, and compare the AUC to IG-ranked deletion AUC. Never evaluate a method using features ranked by the same method without comparing to an independent ranking.

4. **Zero baseline for IG is the observation of a non-existent scene.** Your IG baseline is a zero vector. In V-MAX's normalized observation space, zero means "the center of the normalization range" — which does not correspond to "no input" or "uninformative input." For agent features, zero might mean "agent at ego's position with zero velocity," which is a strong (and weird) signal, not the absence of signal. This affects IG attributions for the agent category in particular. You cannot fix this without deciding what the "correct" baseline is (which is an open research question for IG in RL), but you should *acknowledge* it and potentially run a sensitivity analysis with an alternative baseline (e.g., the mean observation across the dataset, or the observation at t=0 of each scenario).

### 5.2 Common XAI Evaluation Pitfalls You Face

1. **Over-interpreting single-scenario case studies.** The detect→attend→commit→execute arc is beautifully illustrated in your temporal-XAI experiment, but it comes from a single critical hazard (min_TTC=0.049) in one scenario. Before claiming this is a general pattern, you need to show it appears across at least 5–10 hazard events with different topologies. The event mining infrastructure supports this, but the analysis has not been run.

2. **Confounding attribution importance with feature magnitude.** GradientXInput multiplies gradient by input. For features with large absolute values (e.g., roadgraph points that are far from ego), GxI will produce large attributions even if the gradient is small. This is not a bug (the feature's value *does* interact with the gradient), but it means GxI importance partially reflects the scale of the observation, not just model sensitivity. When comparing GxI to IG or VG, this scale effect can dominate.

3. **The "top-k feature" fallacy.** Reporting "top-10 features account for X% of importance" is sparsity, not faithfulness. A highly sparse explanation can be completely wrong (pointing to irrelevant features). Always pair sparsity metrics with faithfulness metrics.

### 5.3 The Attention Aggregation Question

You currently average attention across the 16 Perceiver query tokens before comparing to gradient attribution. This is problematic for several reasons:

**Why averaging is risky:** The 16 queries are learned latent vectors. They may specialize — some queries may attend to agents while others attend to road geometry. Averaging them destroys this specialization structure. If query 3 attends 80% to agents and query 7 attends 80% to roadgraph, the average is ~40% each, which describes no actual query's behavior.

**Alternative aggregation strategies (in order of recommendation):**

1. **Max-pooling across queries.** For each input token, take the maximum attention weight across all 16 queries. This gives the "best case" attention: "at least one query attends strongly to this token." Max-pool is the standard aggregation for comparing attention to gradient methods in vision transformers (Chefer et al., 2021, "Transformer Interpretability Beyond Attention Visualization," CVPR).

2. **Attention-weighted by downstream gradient.** Weight each query's attention by the gradient of the output with respect to that query's latent representation. This gives attention × relevance, keeping only the attention that actually influences the output. Implementation: extract the 16 latent vectors, compute ∂output/∂latent_k for each query k, use |∂output/∂latent_k| as the weight for query k's attention row before averaging. This is computationally cheap (16 extra gradient computations) and theoretically motivated.

3. **Per-query analysis without aggregation.** Instead of aggregating at all, report the full 16-query attention matrix. Cluster queries by their attention profiles (which categories they attend to most). If natural clusters emerge (a "safety cluster" and a "navigation cluster"), that is a finding worth reporting on its own and is more informative than any aggregation.

### 5.4 One Technical Suggestion That Could Meaningfully Change Results

**Implement attention rollout through the Perceiver's self-attention layers.**

Currently, you extract cross-attention weights from layer 0 (the cross-attention layer where 16 learned queries attend to 280 input tokens). But the Perceiver then passes these 16 latent representations through 4 self-attention layers (2 heads each) before the final representation feeds into the actor/critic networks. These self-attention layers mix information across the 16 queries.

A query that initially attends to agents may, after self-attention, route that agent information to a different query that initially attended to roadgraph. The final representation that feeds the actor thus reflects a *combination* of cross-attention and self-attention, and the raw cross-attention weights alone do not capture this.

**Attention rollout (Abnar & Zuidema 2020) for the Perceiver:**
- Let A_cross be the 16×280 cross-attention matrix (queries × input tokens).
- Let A_self^(l) be the 16×16 self-attention matrix at self-attention layer l (queries × queries), for l=1,...,4.
- Rollout: R = A_self^(4) × A_self^(3) × A_self^(2) × A_self^(1) × A_cross.
- R is 16×280 and represents the effective attention from each final-layer query to each input token, accounting for all information mixing through self-attention.
- Then aggregate R across queries (using any of the methods in 5.3) for comparison to gradient attribution.

**Why this could meaningfully change results:** If the self-attention layers substantially redistribute information across queries, the raw cross-attention may overstate the importance of tokens attended to by one query while understating the importance of tokens whose information gets relayed via self-attention. In the RCA paper's vigilance gap analysis, this could matter: if the complete model's agent-monitoring query redistributes its information to other queries during calm phases (via self-attention), the raw cross-attention would show agent attention but the rolled-out attention would show it differently.

**Implementation:** You already extract self-attention weights via `capture_intermediates`. The rollout computation is a chain of matrix multiplications — ~10 lines of JAX code. Add the identity matrix to each self-attention layer before multiplying (to account for residual connections): R_self^(l) = 0.5 * I + 0.5 * A_self^(l). This is the standard rollout convention.

---

*End of Research Brief. Total estimated compute for the highest-priority experiments: multi-seed replication (~4 days), basic reward config (~1 day), cross-architecture comparison (~2 days), attention faithfulness (~2–3 days for deletion curves). All other analyses are zero-compute post-processing on cached data.*
