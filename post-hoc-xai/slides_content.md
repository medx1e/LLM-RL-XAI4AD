# Post-Hoc XAI Framework for Autonomous Driving — Slide Content

---

## Slide 1: Overview

**What we built:** A modular, JAX-native framework for applying and comparing post-hoc explainability methods on pretrained V-Max autonomous driving policies.

**Target models:** 35+ pretrained V-Max model weights (Perceiver, MTR, Wayformer, MGAIL, MLP encoders) trained on the Waymo Open Motion Dataset.

**Core question the framework answers:** *What does the model "look at" when making a driving decision — and does this change over time during critical events?*

---

## Slide 2: Supported Models

All 5 V-Max encoder architectures are supported:

| Encoder | Architecture style |
|---------|-------------------|
| Perceiver / LQ | Global cross-attention (16 latent queries) |
| MTR | Local k-NN attention (k=8 neighbors) |
| Wayformer | Late fusion, per-modality attention |
| MGAIL / LQH | Hierarchical encoder |
| MLP / None | Fully-connected baseline |

Each model is wrapped with a unified interface — any XAI method works on any encoder without modification.

---

## Slide 3: Input Observation Structure

The policy receives a flat observation vector that the framework automatically decomposes into 5 semantic categories:

| Category | What it encodes |
|----------|----------------|
| `roadgraph` | Lane boundaries, road geometry |
| `other_agents` | Positions, velocities of surrounding vehicles |
| `gps_path` | Intended route waypoints |
| `sdc_trajectory` | Ego vehicle's own kinematic history |
| `traffic_lights` | Signal state and positions |

Additionally, `other_agents` is further decomposed **per individual agent**, enabling agent-level attribution.

---

## Slide 4: XAI Methods Implemented (7 total)

**Gradient-based:**
- **Vanilla Gradient** — local sensitivity `∂f/∂x`
- **Integrated Gradients** — path-integrated (satisfies completeness axiom)
- **SmoothGrad** — averaged over noisy input copies
- **Gradient × Input** — element-wise gradient weighting

**Perturbation-based:**
- **Perturbation Attribution** — occlusion per feature/category
- **Feature Ablation** — zero out entire input categories

**RL-specific:**
- **SARFA** — relevance × specificity (Puri et al. 2020); designed for Q-value-based policies

All methods return a uniform `Attribution` object — pluggable into any metric or visualization.

---

## Slide 5: Temporal (Per-Timestep) Analysis

A key feature of the framework: run XAI not just at one snapshot, but **across every timestep of an episode**.

This produces:
- **Category importance over time** — how much does roadgraph vs. agents vs. GPS path matter at each step?
- **Per-agent importance over time** — which specific agent does the model attend to, and when?
- **Stacked attribution composition** — full feature budget across the episode timeline

This allows us to answer: *Does the model's attention shift during a dangerous moment?*

---

## Slide 6: Event Mining Module

To identify **which timesteps are worth explaining**, we built an event mining pipeline on top of the rollout data.

**6 event detectors:**
| Event Type | Trigger condition |
|------------|------------------|
| Hazard Onset | TTC to any agent drops below threshold |
| Near Miss | Agent comes very close without collision |
| Hard Brake | Large negative acceleration |
| Evasive Steering | Large steering magnitude |
| Collision | Actual collision occurred |
| Off-Road | Ego leaves the road |

Each detected event includes: onset/peak/offset timestep, severity (LOW → CRITICAL), causal agent ID, and an analysis window (with padding) for XAI.

**Integration:** The `XAIBridge` feeds event timesteps directly into the XAI pipeline. The `EventCatalog` is queryable and JSON-serializable.

---

## Slide 7: Full Pipeline

```
Pretrained V-Max model
        ↓
  Run episode rollout (80 timesteps)
        ↓
  Event Mining → catalog of critical moments
        ↓
  XAI Analysis at event timesteps (temporal)
        ↓
  Category importance / per-agent importance / stacked composition plots
        ↓
  Faithfulness / Sparsity / Consistency metrics
```

Also: **BEV video rendering** of scenarios with event overlays (Waymax native visualization).

---

## Slide 8: Evaluation Metrics

Three families of XAI quality metrics are implemented:

**Faithfulness** — do the attributed features actually drive model output?
- Deletion curve: progressively mask top features, measure output drop
- Insertion curve: start from zero, add top features back
- Attention–gradient correlation (for Perceiver)

**Sparsity** — how concentrated are the attributions?
- Gini coefficient, top-k concentration, entropy

**Consistency** — are explanations stable across similar scenarios?
- Pairwise and category-level Pearson correlation across scenarios

---

## Slide 9: Key Experimental Findings (5 scenarios, Perceiver model)

1. **Roadgraph dominance:** The model allocates 50–85% of its decision sensitivity to road geometry — it is primarily a road-follower.

2. **Method disagreement on agents:** Vanilla Gradient attributes <2% to other agents during evasive maneuvers; Integrated Gradients attributes 20–50% — a **50–144× discrepancy**. VG misses saturated-region importance that IG captures.

3. **Temporal attribution shift during hazard onset:**
   - Pre-hazard: roadgraph ~55%, agents ~5%
   - Hazard detection: agents spike to ~23%
   - Hazard resolution: roadgraph climbs to ~85%, agents drop to <5%
   - Narrative: **detect → attend → commit → execute**

4. **Causal agent identification:** The specific flagged agent goes from 0.004% → 22.4% importance in 4 timesteps at hazard onset — confirming the model detects which vehicle is dangerous.

5. **GPS path increases during evasive steering** — the model checks route deviation when maneuvering aggressively.

---

## Slide 10: Summary of What the Framework Enables

| Analysis type | What you can ask |
|---------------|-----------------|
| Category importance | Which input modality drives decisions? |
| Temporal analysis | How does attention shift during an episode? |
| Per-agent attribution | Which specific agent does the model react to? |
| Event-targeted XAI | What is the model doing during near-misses or hard brakes? |
| Method comparison | Do different XAI methods agree? |
| Cross-model comparison | Does a Perceiver use features differently than a Wayformer? |
| Faithfulness evaluation | Are the attributions actually faithful to the model? |
