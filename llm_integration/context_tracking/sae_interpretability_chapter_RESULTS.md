# Results & Analysis — Structure Document

## Chapter: SAE-Based Mechanistic Interpretability 

> **Purpose of this file**: Blueprint for the results section. No experimental values are filled in yet. Each subsection states *what* to report, *which figure or table* belongs there, and *what interpretive claim* it supports. Fill in figures/numbers once results are available.

---

## High-Level Narrative Arc

The results section tells a single progressive story:

1. The SAE successfully decomposes the 128-D residual stream into sparse, interpretable features.
2. Those features align statistically with meaningful driving concepts (annotation stage).
3. A curated subset of those features causally governs the policy's behaviour — proven by intervention.

Each stage's results both stand alone *and* motivate the next, mirroring the four-stage pipeline laid out in the methodology.

---

## Section 1 — SAE Training Outcomes

### 1.1 Hyperparameter Selection

**What to report**

- The full 18-configuration grid (λ ∈ {0.02, 0.05, 0.07}, γ ∈ {1, 4, 16}, η₀ ∈ {3×10⁻⁴, 10⁻³}).
- The winning configuration: selected expansion factor γ*, sparsity coefficient λ*, learning rate η₀*, and epoch checkpoint.
- Primary selection criterion value (combined loss L*) for the winning config.

**Table: Hyperparameter grid results**
Columns: γ | λ | η₀ | L_recon | L_sparse | L_combined | Dead% | L₀
One row per configuration; winning row highlighted.

**Interpretive claim**: Justify why the selected config achieves the best trade-off — reconstruction fidelity vs. sparsity vs. dead-feature prevalence.

---

### 1.2 Training Dynamics

**What to report**

- Loss curves (reconstruction loss, L1 sparsity, combined loss) across epochs for the winning configuration.
- Dead-feature percentage curve over epochs.
- L₀ sparsity curve over epochs.

**Figure: Training curves (3-panel)**
Panel 1: Reconstruction loss vs. epoch.
Panel 2: L₁ diagnostic and L₀ vs. epoch (dual-axis or two subplots).
Panel 3: Dead feature % vs. epoch.

**Interpretive claim**: Identify the phases of learning — rapid feature formation, selective pruning, convergence — and relate them to theoretical expectations from Section 3.2.

---

### 1.3 Final Model Statistics

**What to report** (single summary table)

| Statistic                              | Value   |
| -------------------------------------- | ------- |
| Input dimension D                      | 128     |
| Expansion factor γ*                   | INSET   |
| Latent dimension F                     | INSET   |
| Sparsity coefficient λ*               | INSET   |
| Final reconstruction loss              | INSET   |
| Final L₀ (mean active features / obs) | INSET   |
| Dead feature %                         | INSET   |
| Active features (n ≥ 10 activations)  | INSET   |
| Training corpus size                   | 790,936 |

**Interpretive claim**: The final L₀ value should be ≪ F, confirming a genuinely sparse regime. Dead feature % should be below ~50% to confirm the dictionary is not over-expanded.

---

## Section 2 — Feature Discovery and Semantic Annotation


### 2.2 Concept Coverage — Which Driving Concepts Are Encoded?

**What to report**

- For each discovered concept: best feature index, peak |ρ|, direction (positive/negative), number of SAE dimensions encoding it.
- Highlight concepts that are strongly-selective.
- Note any surprising concepts — concepts that appear without an obvious prior hypothesis.

**Figure: Best feature per concept** (`best_per_concept.png/pdf`)
Horizontal bar chart. Each bar = one telemetry concept. Bar length = max |ρ| of the best SAE feature for that concept. Feature index and signed ρ annotated on each bar. Vertical reference lines at 0.3 and 0.5. This is the key "what does the SAE know?" figure.

**Figure: Concept correlation summary** (`concept_correlation_summary.png/pdf`)
Lollipop + bubble chart. Each row = unique driving concept. Bubble area ∝ number of features encoding that concept. Colour = direction of correlation (teal = positive, orange = negative). This is the compact PhD-thesis summary figure of the entire feature vocabulary.

**Interpretive claim**: Connect back to Section 2.2 (superposition hypothesis). The SAE has effectively disentangled the 128-D residual stream into a richer set of F monosemantic directions. Identify which core driving signals (speed, TTC, following, lateral agents) are robustly encoded, and comment on any that are weakly or not encoded.

---

## Section 3 — Causal Feature Steering

### 3.1 Feature Selection Rationale

**What to report**

- The process by which 4–5 features were selected from the full annotation set for causal analysis.
- Selection criteria: high selectivity score, conceptual interest (safety-critical vs. routine), diversity of concept type (speed vs. proximity vs. event vs. lateral).
- A summary table of the 4–5 selected features with their annotation metadata.

**Table: Selected features for causal analysis**
Columns: Feature name | Feature idx | Semantic label | |ρ| | Activation count | Concept type

This table serves as the reader's anchor for the rest of Section 3 — all subsequent subsections refer back to it.

---

### 3.3 Closed-Loop Rollout Results — Feature-by-Feature

For each of the 4–5 selected features, dedicate a subsection with the following structure:

---

#### 3.3.X Feature: [Human Name] (f[idx], ρ=[value] with [telemetry field])

**Narrative introduction** (1–2 sentences)
What concept does this feature encode? Why is it interesting from a safety or driving behaviour perspective?

**Figure: Aggregate Δmetric per α** (`aggregate_metrics.png/pdf`)
The multi-panel bar chart from `plot_rollout_metrics`. One subplot per metric (collision, TTC, comfort, speed violation, lane discipline, driving direction, progress). Bar colour: teal = positive Δ, orange = negative Δ. Error bars = ±std across scenarios.
This is the primary evidence figure for causal behavioural influence.

**Table: Cross-scenario summary for α = [best positive] and α = [best negative]**
Rows = metrics. Columns = baseline | steered (α+) | Δ(α+) | steered (α−) | Δ(α−).
Isolates the most interpretively clean temperature levels.

**Scenario spotlight** (1–2 representative scenarios)

- Choose one scenario where the intervention has a clear, safety-relevant effect (e.g., amplifying a safety-margin feature prevents a collision; suppressing it causes one).
- **Figure: Action trajectories** (`scenario_XX_alpha_YY.png/pdf`): baseline vs. steered acceleration and steering angle over time.
- Brief prose description of what happens in the episode: does the steered agent brake earlier? Does it change lanes? Does it collide?

**Interpretive claim**: State the causal verdict for this feature:

- Is the effect monotone with α? (supports linear readout assumption)
- Is the sign-flip fraction low? (supports cross-scenario consistency)
- Is the safety metric change directionally consistent with the annotation? (confirms semantic fidelity)
- Note any anomalies or limitations (e.g., large std, reversal at high |α| suggesting OOD effects).

---

Repeat 3.3.X for each of the 4–5 selected features. Suggested ordering: speed/comfort features first (clearest causal signal expected), then proximity/TTC features (safety-critical), then event-driven features (rarest, potentially noisiest).

---

### 3.4 Cross-Feature Comparison

**What to report**

- A single aggregate comparison table across all 4–5 features at the most informative α level(s).
- Ranking of features by causal impact (e.g., by |ΔCollision| or |ΔTTC|).
- Patterns: do higher-selectivity features (higher |ρ|) produce larger or more consistent causal effects? This directly tests whether annotation quality predicts causal strength.

**Figure: Cross-feature causal fingerprint heatmap**
Rows = selected features. Columns = driving metrics.
Cell value = mean Δmetric at α = +5 (or the most informative α).
Diverging colormap centred at 0.
This figure provides a bird's-eye view of what each feature controls.

**Interpretive claim**: The cross-feature comparison provides evidence for the overall pipeline's validity — features with strong Spearman correlation produce stronger, semantically coherent causal effects, while low-selectivity features produce weaker and noisier responses. This connects annotation quality to causal reliability.

---

## Section 4 — Discussion and Synthesis

*(Part of the results chapter but more interpretive — can be a bridging section before the formal Limitations section already written in the methodology.)*

### 4.1 What the SAE Learned About the Driving Policy

Synthesise which driving concepts are robustly encoded, which are absent or weakly encoded, and what this reveals about what the Wayformer policy "knows" in its residual stream.

### 4.2 Correlation vs. Causation: What the Steering Experiments Add

Discuss the step from statistical annotation (Section 2) to causal intervention (Section 3). Which features survive the causal test? Are there features with high |ρ| but low causal effect (epiphenomenal features)? Vice versa?

### 4.3 Safety Implications

Identify any features whose causal manipulation directly affects collision rate or TTC. Frame the implication: if these features can be monitored or steered at runtime, they constitute a mechanistic safety interface into the policy.

## Writing Notes

- **Order of presentation**: lead with SAE quality (training + annotation) before steering. Readers must trust the SAE before trusting the interventions.
- **Avoid over-claiming**: the Limitations section (already written) covers SAE imperfection, annotation ambiguity, and OOD intervention effects. Do not repeat them here — simply reference them when relevant (e.g., "this result should be interpreted in light of the distribution-shift limitation discussed in Section 4.2").
- **Consistency with notation**: use the notation established in the theoretical background (f_j for feature j, ρ_{jk} for Spearman correlation, Δa_accel / Δv for intervention effects, α for steering temperature).
- **Feature naming convention in prose**: refer to features by their human-readable name (e.g., "the ego-speed feature") with the index in parentheses (f[idx]), not just by index.
