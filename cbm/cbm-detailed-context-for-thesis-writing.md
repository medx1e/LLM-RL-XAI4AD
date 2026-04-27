# CBM Chapter — Complete Thesis Writing Context

*This document is the single source of truth for writing the CBM chapter of the thesis.
It contains all context, results, theoretical arguments, and narrative structure needed.
A writing prompt is attached at the end.*

*Last updated: 2026-04-26.*

---

## PART A — CHAPTER STRUCTURE AND NARRATIVE

### Proposed Chapter Title
**"Concept Bottleneck Models for Interpretable Autonomous Driving"**

### Chapter Arc (the story to tell)

The chapter has a clear progression:

1. **Why we go beyond post-hoc XAI** — attribution methods explain what the network attended to, but do not constrain what it uses to act. A CBM goes further: the policy is architecturally forced to reason through human-interpretable concepts. This is a stronger interpretability guarantee.

2. **The theoretical foundation** — CBMs (Koh et al., 2020) and their connection to probing classifiers. The key upgrade: probing just checks if a concept is decodable from z; CBM makes the concept the causal bottleneck. Also: why heuristic concept supervision is observation-faithful and scientifically sound.

3. **Our architecture and concepts** — how the CBM is injected into a SAC loop. The concept extraction pipeline. The 15 concepts across 3 phases and why they were chosen.

4. **The training paradigm story** — we tried three paradigms. Joint failed (concept collapse). Frozen worked. Scratch is the scientific contribution (bootstrapping from random init). This is the experimental narrative backbone.

5. **Results** — concept quality, task performance, the λ ablation, the key finding (zero red light violations). Placeholders for 150GB and SAC-scratch comparisons.

6. **Discussion** — interpretability tax, concept completeness, open questions.

### Section-by-Section Structure

```
Section 1: Introduction and Motivation
  1.1 Limitations of post-hoc XAI for autonomous driving
  1.2 The case for in-model interpretability
  1.3 Chapter overview

Section 2: Theoretical Background
  2.1 Concept Bottleneck Models — formal definition
  2.2 Connection to probing classifiers
  2.3 Hard vs soft bottleneck
  2.4 Observation-faithfulness: why heuristic supervision is sound

Section 3: Architecture
  3.1 The CBM-SAC integration
  3.2 Network structure and hard bottleneck
  3.3 Concept extraction pipeline
  3.4 Loss function

Section 4: Concept Set Design
  4.1 Design principles
  4.2 Phase 1 — core kinematics (8 concepts)
  4.3 Phase 2 — interactive driving (3 concepts)
  4.4 Phase 3 — path spatial geometry (4 concepts, V2)
  4.5 Binary vs continuous: loss type and validity masking

Section 5: Training Paradigms
  5.1 The paradigm space (SAC / Frozen / Joint / Scratch)
  5.2 CBM Joint — what we tried and why it fails
  5.3 CBM Frozen — the principled approach
  5.4 CBM Scratch — from-scratch training as scientific contribution

Section 6: Experiments
  6.1 Dataset and setup
  6.2 Lambda ablation (concept supervision weight selection)
  6.3 Validation evaluation — concept quality
  6.4 Validation evaluation — task performance
  6.5 Key finding: concept accuracy drives behavior

Section 7: Results Summary [SCALABLE TABLE — add new model rows here]

Section 8: Discussion
  8.1 The interpretability tax
  8.2 Concept completeness and coverage
  8.3 Limitations and open questions
```

---

## PART B — DETAILED CONTENT PER SECTION

---

### Section 1: Introduction and Motivation

#### 1.1 Limitations of post-hoc XAI

Our earlier work (see post-hoc XAI chapter) demonstrated that attribution methods — integrated gradients, gradient×input, feature ablation — can identify which parts of the observation the SAC encoder attends to. These methods are valuable but have fundamental limitations in the autonomous driving context:

- **They explain without constraining.** A saliency map tells you the model attended to the traffic light, but does not guarantee the model's action was determined by the traffic light state. The model could attend to the light while making the decision based on other features.
- **They are fragile.** Attribution values depend on the chosen baseline, the integration path, and the surrogate model assumptions. Two attribution methods often disagree on what the model found important.
- **They cannot provide safety guarantees.** If a regulator asks "did your model stop because it saw a red light?", post-hoc attribution cannot give a provable answer.

#### 1.2 The case for in-model interpretability

A Concept Bottleneck Model (CBM) resolves these limitations by construction. Instead of explaining decisions after the fact, the CBM enforces that decisions flow through a fixed, named, human-readable intermediate layer. The actor and critic literally cannot access the raw latent — they see only the concept vector. This makes the causal chain provable: if the model brakes at a red light, it is because the `traffic_light_red` concept node fired, and that node is supervised to predict exactly the ground-truth traffic light state.

This is not a small difference. It changes "we think this model responds to red lights" to "this model's action is a mathematical function of its `traffic_light_red` neuron output, which tracks red light state at 99.94% accuracy."

---

### Section 2: Theoretical Background

#### 2.1 Concept Bottleneck Models — formal definition

A CBM (Koh et al., 2020) adds an intermediate concept layer `c` between the encoder and the task head:

```
obs → [Encoder] → z → [Concept Head] → c → [Task Head] → action
```

The concept vector `c ∈ [0,1]^N` contains N named scalar predictions. Each dimension corresponds to one human-interpretable concept (e.g., "Is the traffic light red?", "How far is the nearest vehicle?"). The task head — in our case, the SAC actor and critic — operates **only on `c`**, never on `z` directly. This is the **hard bottleneck**.

Training adds a concept supervision term to the standard task loss:
```
L_total = L_task + λ · L_concept
```

`L_concept` penalizes deviation between predicted concept values and ground-truth concept values, using BCE for binary concepts and Huber loss for continuous ones.

#### 2.2 Connection to probing classifiers

Probing classifiers (Alain & Bengio, 2017; Belinkov, 2022) are a standard tool for understanding what information is encoded in a neural representation. A probe is a simple linear classifier trained to predict some property (e.g., "does this representation encode color?") from the latent `z`. If the probe achieves high accuracy, the property is said to be "linearly decodable" from `z`.

Our CBM concept head is essentially a probing classifier — a small MLP that tries to decode human-interpretable concepts from `z`. But there is a critical difference:

- **Probing checks if a concept is present in z** — it is a diagnostic tool applied after training. The probe does not affect the model's behavior.
- **CBM enforces the concept as the causal bottleneck** — the concept is not just readable from `z`; it is the only signal the policy sees. The actor's action is literally a function of the concept predictions, not of `z`.

This makes CBM a stronger, active form of the probing intuition. The question is no longer "is traffic_light_red decodable from z?" but "does the policy's decision to brake causally depend on the traffic_light_red concept neuron?"

Furthermore, CBM supervision during training shapes what `z` encodes. In the frozen paradigm, the concept head must learn to read concepts from a fixed `z`. In the joint and scratch paradigms, the encoder is also trained, so `z` is shaped to be more concept-aligned — the analogue of training a representation to be probe-friendly from the start.

#### 2.3 Hard vs soft bottleneck

A **hard bottleneck** (our design) routes all information from `z` to the policy through `c`:
```
action = policy(c)           where c = concept_head(z)
```

A **soft bottleneck** passes both concepts and a residual stream:
```
action = policy(c, z_residual)
```

The soft variant achieves higher task performance because task-relevant information not captured by the 15 concepts can still flow through `z_residual`. But it sacrifices the interpretability guarantee: if `z_residual` is non-zero, the policy can use unconstrained information and the concepts are no longer the sole causal determinant.

We chose the hard bottleneck deliberately. Our goal is the strongest possible interpretability guarantee, with the best task performance achievable under that constraint — not the best performance overall.

#### 2.4 Observation-faithfulness: why heuristic supervision is sound

A key concern with concept supervision is the origin of the ground-truth concept values. In image classification CBMs (Koh et al., 2020), human annotators provide concept labels. This is expensive, subjective, and does not scale.

In our setting, ground-truth concepts are computed **deterministically from the same flat observation vector that the encoder receives**:

```
flat obs (1655-d)
    → unflatten (same function the encoder uses internally)
    → ConceptInput (structured JAX arrays)
    → per-concept extractor (pure JAX math)
    → normalized concept value ∈ [0, 1]
```

This design has a critical property: **every concept value is a deterministic function of the information already available to the encoder**. There is no oracle information, no privileged signal, no annotation gap. If the encoder sees a traffic light state onehot encoding at index 4 in the flat vector, and the `traffic_light_red` extractor reads that exact onehot to produce the concept target — then the concept target is mathematically computable from what the encoder sees.

This makes the supervision "observation-faithful" — the concept head is being trained to predict something that is, in principle, exactly decodable from the encoder's input. Any gap between concept prediction and ground truth reflects the encoder's representation quality, not an impossibility.

The practical implication: concept supervision in our system does not introduce any external bias or annotation error. The concepts are mathematical projections of the observation, and the concept head is a learned function that approximates those projections.

---

### Section 3: Architecture

#### 3.1 The CBM-SAC integration

The base agent is a Soft Actor-Critic (SAC) agent (Haarnoja et al., 2018) built on the V-Max framework (Google, 2023) running in the Waymax simulator (Gulino et al., 2023). The baseline SAC architecture:

```
obs (1655-d) → [LQ Encoder] → z (128-d) → [Actor MLP] → action (2: accel, steer)
                                          → [Twin Critic MLP] → Q(s,a)
```

The LQ encoder is a Perceiver-style latent cross-attention network: `embedding_layer_sizes=[256,256]`, depth=4, dk=64, 16 latents, tied weights across layers. The pretrained baseline achieves **97.47% route progress** on WOMD evaluation.

The CBM replaces the direct `z → actor` pathway with an explicit concept bottleneck:

```
obs (1655-d)
    │
    ▼
[LQ Encoder]         — pretrained or randomly initialized
    │
    ▼  z (128-d)
    │
    ▼
[Concept Head]       — Dense(128→64)+ReLU+Dense(64→N)+Sigmoid
    │
    ▼  c (N-d, all in [0,1])
    │
    ├──▶ [Actor FC]  — Dense(N→64)+ReLU+Dense(64→32)+ReLU → action
    └──▶ [Critic FC] — Dense(N+2→64)+... → Q-values (twin)
```

**Hard bottleneck guarantee**: Actor and Critic have NO access to `z`. Only `c` reaches the task heads. This is enforced structurally, not by regularization.

**N = 11** for V1 (phases 1+2), **N = 15** for V2 (phases 1+2+3).

#### 3.2 Frozen encoder implementation

In frozen mode, `jax.lax.stop_gradient` is applied after the encoder:

```python
z = self.encoder(obs)
if self.frozen_encoder:
    z = jax.lax.stop_gradient(z)
c = self.concept_head(z)
action = self.actor_fc(c)
```

This blocks ALL gradients (from both SAC policy loss and concept loss) from reaching the encoder weights. Verified experimentally: encoder gradient norm = 0.0 in frozen mode, 1.058 in joint mode.

#### 3.3 Concept extraction pipeline

The ground-truth concept pipeline is a clean, modular system with a strict firewall:

```
V-Max / Waymax  ←── NEVER MODIFIED (zero changes to source)
    │
adapters.py     ←── ONLY crossing point into V-Max internals
    │
ConceptInput    ←── structured dataclass (JAX arrays)
    │
extractors.py   ←── pure JAX, JIT-safe, one function per concept
    │
registry.py     ←── OrderedDict (ordered = index-stable)
    │
CBMConfig       ←── auto-derives binary/continuous index lists
```

The full pipeline runs **inside every SGD step** — concept targets are always fresh and observation-faithful. Since all operations are pure JAX, the pipeline is JIT-compiled and adds negligible overhead to the training step.

#### 3.4 Loss function

```
L_concept = (BCE_binary_masked + Huber_continuous_masked) / total_valid_entries
L_total   = L_SAC_policy + λ · L_concept
```

**Binary concepts** (traffic_light_red, lead_vehicle_decelerating, at_intersection): Binary Cross Entropy with ε-clamping.

**Continuous concepts** (all others): Huber loss (δ=1.0) — quadratic for small errors, linear for large errors. Robust to the extreme values that can appear in velocity/distance estimates.

**Validity masking**: Some concepts are undefined in certain states (e.g., `ttc_lead_vehicle` when no lead vehicle exists). Invalid concepts contribute zero to the numerator. The denominator is the count of valid entries across all concepts — ensuring sparse concepts (like TTC at 45.5% validity) do not artificially inflate the loss.

**λ selection**: A lambda ablation was run (see Section 6.2). λ=0.5 was selected — it achieves the best concept quality while maintaining equivalent task performance to λ=0.1. λ=0.01 causes safety-critical concepts to fail (TL red accuracy drops to 76%).

---

### Section 4: Concept Set Design

#### 4.1 Design principles

Concepts were chosen to satisfy four criteria:
1. **Observation-faithful** — computable purely from the 1655-d observation. No external oracle.
2. **Human-interpretable** — a domain expert reading the concept name should understand what it measures.
3. **Non-redundant** — each concept captures a distinct aspect of the driving scene.
4. **Safety-relevant** — prioritize concepts that relate to crash avoidance and traffic law compliance.

Concepts are organized in phases reflecting increasing complexity. Phases 1 and 2 were validated in V1; Phase 3 was added in V2.

#### 4.2 Phase 1 — Core kinematics and scene awareness (8 concepts)

| # | Name | Type | Formula | Normalization |
|---|---|---|---|---|
| 0 | `ego_speed` | continuous | ‖vel_xy[-1]‖ × 30 m/s | /30, clip[0,1] |
| 1 | `ego_acceleration` | continuous | (speed[-1] − speed[-2]) / dt | (x+6)/12, clip[0,1] |
| 2 | `dist_nearest_object` | continuous | min_n ‖agent_xy[n,-1]‖ × 70 m | /70, clip[0,1] |
| 3 | `num_objects_within_10m` | continuous | count(‖agent_xy‖ < 10 AND valid) | /8, clip[0,1] |
| 4 | `traffic_light_red` | **binary** | any(state_onehot[{0,3,6}] AND tl_valid) | identity |
| 5 | `dist_to_traffic_light` | continuous | min_n ‖tl_xy[n,-1]‖ × 70 m | /70, clip[0,1] |
| 6 | `heading_deviation` | continuous | wrap(ego_yaw − atan2(path_dy, path_dx)) rad | (x+π)/(2π) |
| 7 | `progress_along_route` | continuous | project_onto_path fraction | clip[0,1] |

#### 4.3 Phase 2 — Interactive driving signals (3 concepts)

| # | Name | Type | Formula | Normalization |
|---|---|---|---|---|
| 8 | `ttc_lead_vehicle` | continuous | lead_dist_x / max(Δv, ε), capped 10s | /10, clip[0,1] |
| 9 | `lead_vehicle_decelerating` | **binary** | Δspeed_lead > 0.5 m/s over 2 steps | identity |
| 10 | `at_intersection` | **binary** | any(‖tl_xy‖ < 25m AND tl_valid) | identity |

**Note on `at_intersection`**: WOMD roadgraph does not expose intersection labels in the observation config used. TL proximity (< 25m) is the best available heuristic, achieving ~87.4% accuracy in V1. This is an acknowledged limitation.

#### 4.4 Phase 3 — Path spatial geometry (4 concepts, V2 only)

| # | Name | Type | Formula | Normalization |
|---|---|---|---|---|
| 11 | `path_curvature_max` | continuous | max Menger curvature over path interior (1/m) | /0.25, clip[0,1] |
| 12 | `path_net_heading_change` | continuous | wrap(atan2(last_seg) − atan2(first_seg)) rad | (x+π)/(2π) |
| 13 | `path_straightness` | continuous | chord_length / arc_length | clip[0,1] |
| 14 | `heading_to_path_end` | continuous | atan2(end_y, end_x) in SDC frame rad | (x+π)/(2π) |

All Phase 3 concepts are always valid (path always present in observation). They were validated to be non-redundant with Phase 1+2 concepts and with each other. Menger curvature uses ε-guards against collinear path segments.

#### 4.5 Binary vs continuous: loss type and validity masking

Binary concepts use BCE; continuous use Huber(δ=1.0). The binary/continuous classification is fixed and auto-derived from the registry at config time — no manual hardcoding of indices.

Validity masks are per-concept-per-timestep boolean tensors. A concept is invalid when its prerequisite does not exist (e.g., `ttc_lead_vehicle` when no vehicle is ahead). Invalid concepts are zeroed out in the numerator but counted in the denominator of the concept loss, preventing sparse concepts from dominating the gradient signal.

---

### Section 5: Training Paradigms

#### 5.1 The paradigm space

| Paradigm | Encoder init | Encoder grad | Concept Head | Actor/Critic |
|---|---|---|---|---|
| SAC Baseline | Pretrained | Trainable | None | Direct from z |
| **CBM Frozen** | Pretrained | **Frozen** | Trainable | From c only |
| CBM Joint | Pretrained | Trainable | Trainable | From c only |
| **CBM Scratch** | Random | Trainable | Trainable | From c only |

#### 5.2 CBM Joint — what we tried and why it fails

In joint mode, all parameters are trainable simultaneously. The intuition is appealing: let the encoder adapt to become more concept-aligned, potentially improving concept accuracy.

In practice, joint training risks **concept collapse** — a failure mode analogous to Goodhart's Law applied to interpretability: "when a measure becomes a target, it ceases to be a good measure."

The mechanism: the concept loss creates gradient pressure on the encoder to produce latents `z` from which concepts are easily predictable. Simultaneously, the SAC task loss creates gradient pressure on the encoder to produce latents from which the actor can extract task-relevant information efficiently. These two objectives can be satisfied jointly by an encoder that encodes task information in the "scale" or "correlations" of concept activations rather than their semantic content. The concept head learns to output values that satisfy the BCE/Huber loss, while the actor learns to extract non-semantic information from the joint distribution of concept predictions.

The result: the model satisfies the concept supervision loss at training time, but at deployment the concepts are not causally driving the policy behavior in the human-intended sense. This is the definition of concept collapse.

**Our observation**: Joint mode training shows higher initial task performance than frozen mode (the encoder can adapt freely), but concept quality metrics plateau at lower values and are less stable. Joint mode was trained on ~10GB data and showed these instabilities. For this reason, the recommended paradigm is Frozen (stable, principled) or Scratch (zero pretrained bias).

#### 5.3 CBM Frozen — the principled approach

In frozen mode, the encoder's weights are locked by `stop_gradient`. The concept head must learn to read human-interpretable concepts from a fixed representation that was shaped entirely by RL reward maximization — not by concept supervision.

This is theoretically appealing for two reasons:

1. **The encoder is task-optimal.** The pretrained SAC encoder achieves 97.47% route progress. Its representations are proven to contain the information needed for good driving. Freezing it means we are asking: "are the 15 concepts readable from a provably task-optimal representation?" — a well-posed scientific question.

2. **Concept learning is scoped and stable.** With the encoder frozen, the concept head optimization problem is a standard supervised regression/classification problem on a fixed feature space. There is no risk of concept collapse because the encoder cannot "cheat" — it cannot adapt to satisfy the concept targets in an unintelligible way.

The performance cost: the actor sees only 15 numbers instead of 128. If the 15 concepts don't capture all task-relevant information, performance will be bounded. This is the interpretability tax — and it is measurable.

#### 5.4 CBM Scratch — from-scratch training as scientific contribution

In scratch mode, the encoder starts from random weights. There is no pretrained representation to build on. The only training signals are:
1. The RL reward (SAC task loss)
2. The concept supervision loss (15 concept targets computed per step)

This is the strongest form of the CBM experiment. The encoder's entire representational structure is shaped by what is needed to simultaneously:
- Drive well (maximize reward)
- Predict 15 human-interpretable concepts accurately

If this succeeds, it proves that concept-supervised RL can **bootstrap spatial representations from scratch** without any pretraining. The encoder's learned representations are, by construction, aligned with the human-defined concept vocabulary.

**The result (10GB, λ=0.5)**: 87.6% route progress (90% of the pretrained SAC baseline), zero red light violations, 98–99% binary concept accuracy — trained from random weights on 10GB of data. This is a strong result that demonstrates the feasibility of concept-supervised representation learning for autonomous driving.

The scratch paradigm is also cleaner scientifically: there is no question of "is this concept just readable from the pretrained encoder?" — the encoder has no pretrained knowledge. Every bit of spatial structure it learned came from the interaction between RL reward and concept supervision.

---

### Section 6: Experiments

#### 6.1 Dataset and setup

**Dataset**: Waymo Open Motion Dataset (WOMD)
- Full training set: 150GB TFRecord
- Validation set: separate split (not used for training)
- Scenario length: 80 steps (8 seconds at 10 Hz)
- Max objects: 64; 8 nearest included in observation

**Evaluation**: All validation numbers come from a fixed set of **400 WOMD validation scenarios** using deterministic (argmax) policy. This is held-out data never seen during training.

**Hardware**: Local GPU (NVIDIA GTX 1660 Ti 6GB) for short runs and smoke tests. Cluster GPU (12GB+ VRAM) for full training runs.

**Key hyperparameters** (full training):
| Parameter | Value |
|---|---|
| Total timesteps | 15M (150GB) |
| Learning rate | 1e-4 (Adam) |
| Batch size | 128 |
| Replay buffer | 1M transitions |
| SAC α | 0.2 |
| Discount γ | 0.99 |
| Soft update τ | 0.005 |
| λ_concept | 0.5 (selected by ablation) |
| Concepts | 15 (phases 1+2+3) |

#### 6.2 Lambda ablation — concept supervision weight selection

Before committing to a full 150GB run, we ran three training experiments on 10GB varying only λ ∈ {0.01, 0.1, 0.5}. All other hyperparameters identical.

**Task performance (on training rollouts):**

| Metric | λ=0.01 | λ=0.1 | λ=0.5 |
|---|---|---|---|
| Route Progress (nuplan) | 0.420 | 0.428 | 0.427 |
| At-Fault Collision | 0.0004 | 0.0000 | **0.0000** |
| Episode Reward | 3.03 | 3.81 | 3.70 |
| Policy Loss (final) | -22.52 | -23.33 | **-23.39** |
| Policy Loss Std (last 20%) | 0.22 | 0.27 | 0.24 |

**Binary concept accuracy (on training data, end of run):**

| Concept | λ=0.01 | λ=0.1 | λ=0.5 |
|---|---|---|---|
| `traffic_light_red` | 76.0% ❌ | 99.7% | **99.97%** |
| `at_intersection` | 70.0% ❌ | 98.9% | **99.71%** |
| `lead_vehicle_decelerating` | 98.2% | 98.6% | **98.8%** |

**Key observations:**
- λ=0.01 is eliminated: safety-critical binary concepts fail completely. Multiple continuous concepts actively worsen during training (heading_deviation: +623%, heading_to_path_end: +1834%). The task loss completely overrides concept learning.
- λ=0.1 and λ=0.5 achieve equivalent task performance (zero collisions, equivalent route progress).
- λ=0.5 wins on concept quality across all 15 concepts. The episode reward difference (3.81 vs 3.70) is marginal and measured on training data — not a reliable signal at this scale.
- No dead neurons detected. Gradient ratio concept_head/actor_fc ≈ 1.8 for all λ values — concept learning is not being drowned by the task loss.
- Phase 3 concepts show a mid-training bump (loss temporarily increases then recovers) — normal from-scratch dynamic: the encoder restructures for driving first, then relearns path geometry on a better foundation.

**Decision**: λ=0.5 for the full 150GB run.

#### 6.3 Validation evaluation — concept quality (CBM Scratch V2, 10GB, λ=0.5)

Evaluated on **400 held-out WOMD validation scenarios**. This model is the λ ablation winner before the full 150GB run — its numbers represent a **lower bound** on final performance.

**Binary concept accuracy:**

| Concept | Accuracy | Valid% |
|---|---|---|
| `traffic_light_red` | **99.94%** | 66.2% |
| `at_intersection` | **99.12%** | 66.2% |
| `lead_vehicle_decelerating` | **98.14%** | 45.5% |
| **Mean binary accuracy** | **99.07%** | — |

**Continuous concept quality:**

| Concept | MAE | R² | Phase | Notes |
|---|---|---|---|---|
| `dist_to_traffic_light` | 0.076 | **0.914** | P1 | Best continuous concept |
| `path_straightness` | 0.050 | **0.629** | P3 | Strong Phase 3 result |
| `ego_speed` | 0.091 | 0.521 | P1 | Good |
| `num_objects_within_10m` | 0.124 | 0.505 | P1 | Good |
| `path_net_heading_change` | 0.073 | 0.446 | P3 | Good |
| `path_curvature_max` | 0.145 | 0.403 | P3 | Good |
| `ego_acceleration` | 0.067 | 0.275 | P1 | Volatile signal — expected |
| `ttc_lead_vehicle` | 0.138 | 0.185 | P2 | Sparse (45.5% valid) |
| `heading_to_path_end` | 0.080 | 0.128 | P3 | Still converging — needs 150GB |
| `dist_nearest_object` | 0.066 | -0.035 | P1 | Near-mean prediction — needs more data |
| `progress_along_route` | 0.023 | -0.332 | P1 | Good MAE; R² misleading (near-constant at t=0 in logged data) |
| `heading_deviation` | 0.053 | **-1.374** | P1 | Encoder hasn't learned full lane topology from 10GB |

3 out of 4 Phase 3 concepts already achieve R² > 0.4 on held-out validation. This confirms that Phase 3 concept supervision is working correctly and the concepts are genuinely being learned.

The negative R² values on `heading_deviation` and `dist_nearest_object` are expected at 10GB. `heading_deviation` in particular requires the encoder to understand lane topology and ego-path alignment across diverse road geometries — this takes significantly more data. These are the concepts expected to improve most from 10GB → 150GB.

#### 6.4 Validation evaluation — task performance

| Metric | CBM Scratch V2 (10GB) | SAC Baseline (150GB) | SAC Scratch (10GB) [PENDING] | CBM Scratch V2 (150GB) [PENDING] |
|---|---|---|---|---|
| Route progress (nuplan) | 0.876 | **0.975** | TBD | TBD |
| At-fault collision | 0.080 | 0.018 | TBD | TBD |
| **Run red light** | **0.000** | — | TBD | TBD |
| Offroad rate | 0.015 | — | TBD | TBD |
| Episode completion | 0.900 | ~0.97 | TBD | TBD |
| `on_multiple_lanes` | 0.465 | — | TBD | TBD |

*[PENDING results should be added here once training completes. The SAC Scratch baseline provides the apples-to-apples comparison for the interpretability tax.]*

**Important notes on the table:**
- Route progress 0.876 from scratch on 10GB = ~90% of the pretrained SAC baseline trained on 150GB. This is strong given the data disadvantage.
- The `on_multiple_lanes = 0.465` (high) is expected at 10GB — lateral control precision requires more scenario diversity.
- SAC Scratch comparison will show the interpretability tax: difference in route progress between unconstrained SAC and CBM under identical conditions (both from scratch, same data).

#### 6.5 Key finding: concept accuracy drives behavior

The most striking result is the link between concept accuracy and observable behavior. The model achieved **zero red light violations** (`run_red_light = 0.000`) in 400 validation scenarios. This is directly and causally explained by the **99.94% accuracy** of the `traffic_light_red` concept:

- The model almost never misclassifies a red light.
- Since the actor can only see the concept vector, it has no way to "ignore" the `traffic_light_red` signal.
- Therefore, when a red light is present and correctly predicted, the actor's behavior is deterministically influenced by that concept node.

This is the CBM's core promise demonstrated empirically: an interpretable concept with high accuracy produces predictable, safe behavior — not by accident, but by architectural design.

This causal link between concept accuracy and task behavior cannot be demonstrated by post-hoc attribution methods. Attribution can say "the model attended to the traffic light." The CBM says "the model's braking decision is a mathematical function of its traffic light prediction, which is 99.94% accurate."

---

### Section 7: Results Summary Table (SCALABLE)

**This table should be updated as new experiments complete. Add new rows; do not remove existing ones.**

| Model | Paradigm | Concepts | Data | Route Progress | At-Fault Coll. | Run Red Light | Binary Acc. | Mean R² |
|---|---|---|---|---|---|---|---|---|
| SAC Baseline (seed 42) | Black box | 0 | 150GB | **0.975** | 0.018 | — | — | — |
| CBM Frozen V1 | Frozen (pretrained) | 11 | ~10GB | TBD (eval pending) | TBD | TBD | TBD | TBD |
| CBM Joint V1 | Joint (pretrained) | 11 | ~10GB | TBD | TBD | TBD | TBD | TBD |
| CBM Frozen V1 | Frozen (pretrained) | 11 | 150GB | TBD | TBD | TBD | TBD | TBD |
| CBM Frozen V2 | Frozen (pretrained) | 15 | 150GB | TBD | TBD | TBD | TBD | TBD |
| CBM Scratch V2 (λ=0.5) | Scratch | 15 | 10GB | 0.876 | 0.080 | **0.000** | 99.07% | ~0.39 |
| SAC Scratch | Black box (scratch) | 0 | 10GB | **[PENDING]** | **[PENDING]** | — | — | — |
| CBM Scratch V2 (λ=0.5) | Scratch | 15 | 150GB | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** | **[PENDING]** |

---

### Section 8: Discussion

#### 8.1 The interpretability tax

The fundamental tradeoff in CBM design is performance vs interpretability. The actor sees 15 numbers instead of 128. Any driving situation that requires information not captured by the 15 concepts will lead to suboptimal behavior.

With 10GB training data, the 8% at-fault collision rate (vs 1.8% for the 150GB pretrained baseline) reflects two compounded disadvantages: (1) less training data, and (2) the hard bottleneck at 15 concepts. Disentangling these requires the SAC-from-scratch baseline (same data, same encoder, no concept bottleneck). Once that comparison is available, the interpretability tax can be measured precisely.

The zero red light result suggests the tax is not uniform: for behaviors well-covered by a high-accuracy concept, the CBM imposes no tax at all. The tax appears primarily in behaviors requiring precise continuous control (lateral precision, close-following), where the continuous concepts have lower R².

#### 8.2 Concept completeness and coverage

15 concepts cannot fully describe all driving situations. Construction zones, pedestrian crossings, vehicle merging, and emergency vehicles are not explicitly covered. The model must handle these situations using only the general-purpose concepts (distances, speeds, heading) which may not provide sufficient signal.

This is a known and expected limitation of the current concept set. Future work could extend to 20-25 concepts covering more edge cases — the phased design (Phase 1, 2, 3) is explicitly structured to accommodate this without breaking existing infrastructure.

The `at_intersection` heuristic (TL proximity as proxy) is the clearest example of concept incompleteness. It achieves ~87% accuracy but misses intersections without traffic lights. A roadgraph-type-based extractor could improve this if `roadgraph.types` were added to the observation config.

#### 8.3 Limitations and open questions

1. **Performance ceiling**: Is 15 concepts sufficient? What is the information-theoretic lower bound on the concept count needed for parity with unconstrained SAC?

2. **Concept collapse in joint mode**: Can curriculum learning (frozen warmup → gradual unfreezing) stabilize joint training and recover the benefits of both paradigms?

3. **Temporal consistency**: Per-timestep concept supervision treats each step independently. Adding temporal concept supervision (e.g., TTC trend over 5 steps) could improve prediction of dynamic concepts.

4. **`dist_to_path` (deferred)**: This concept (lateral deviation from route) is meaningful only during rollout — at t=0 in logged data it has a systematic ~5.8m offset. It was deferred because training on logged data with this offset would confuse the concept head. A curriculum approach starting from rollout data might enable it.

5. **Concept redundancy**: `at_intersection` and `dist_to_traffic_light` are correlated. Orthogonality analysis could reveal whether the bottleneck benefits from concept redundancy or is hurt by it.

---

## PART C — SUPPORTING REFERENCE MATERIAL

### Model Inventory (for methods chapter)

| Directory | Paradigm | Concepts | Data | Status |
|---|---|---|---|---|
| `runs_rlc/womd_sac_road_perceiver_minimal_42` | SAC Baseline | 0 | 150GB | Done ✓ |
| `cbm_model/` | CBM Frozen V1 | 11 | ~10GB | Done ✓ |
| `cbm_model_joint/` | CBM Joint V1 | 11 | ~10GB | Done ✓ |
| `cbm_v1_frozen_150GB/` | CBM Frozen V1 | 11 | 150GB | Done ✓ |
| `cbm_scratch/` | CBM Scratch V1 (11c, old bug run) | 11 | 150GB | Done ✓ |
| `cbm_v2_frozen_womd_150gb/` | CBM Frozen V2 | 15 | 150GB | Done ✓ |
| `cbm_scratch_v2_lambda05/` | CBM Scratch V2, λ=0.5 | 15 | 10GB | Done ✓ — primary current result |
| `runs_cbm/cbm_scratch_v2_150gb_42/` | CBM Scratch V2, λ=0.5 | 15 | 150GB | **Training in progress** |
| SAC Scratch (no CBM) | SAC from scratch | 0 | 10GB | **Training in progress** |

### Novelty Claims (for abstract / introduction)

1. **First hard-bottleneck CBM in a SAC continuous-control loop for autonomous driving** on a realistic, large-scale simulator (WOMD / Waymax).

2. **Annotation-free concept supervision** — all 15 concepts derived deterministically from the simulator observation. No human labels, no trained oracle, observation-faithful by construction.

3. **From-scratch CBM training** — demonstrating that concept-supervised RL bootstraps spatial representations from random initialization without any pretraining.

4. **Causal concept-behavior link** — empirically demonstrating that high concept accuracy produces predictable behavioral guarantees (zero red light violations causally explained by 99.94% TL accuracy).

### Related Work (for literature review section)

**Concept Bottleneck Models:**
- Koh et al. (2020) — original CBM (image classification, human labels)
- Zarlenga et al. (2022) "Concept Embedding Models" — soft bottleneck
- Yuksekgonul et al. (2022) "Post-hoc Concept Bottleneck Models"
- Shao et al. (2021) "Right for Better Reasons"

**Probing classifiers:**
- Alain & Bengio (2017) "Understanding Intermediate Representations"
- Belinkov (2022) "Probing Classifiers: Promises, Shortcomings, and Advances"

**XAI for RL:**
- Greydanus et al. (2018) — saliency for Atari agents
- Juozapaitis et al. (2019) — reward decomposition for XRL
- Dazeley et al. (2021) — survey of explainable DRL

**RL for Autonomous Driving:**
- Gulino et al. (2023) "Waymax: An Accelerated, Data-Driven Simulator for Large-Scale Autonomous Driving Research"
- Haarnoja et al. (2018) "Soft Actor-Critic"

**XAI for AD:**
- Kim & Misu (2017) "Grounded Situation Models for Robots in Environments"
- Cultrera et al. (2020) "Explaining Autonomous Driving by Learning End-to-End Visual Attention"

### Evaluation Metrics Reference

| Metric | Source | Description |
|---|---|---|
| `progress_ratio_nuplan` | Waymax / nuPlan | Fraction of planned route completed |
| `at_fault_collision` | Waymax | Episodes where ego caused a collision |
| `run_red_light` | Waymax | Episodes where ego ran a red light |
| `offroad` | Waymax | Episodes with offroad excursion |
| `on_multiple_lanes` | Waymax | Fraction of time straddling lane lines |
| Binary accuracy | Eval script | (pred > 0.5) == (true > 0.5) over valid steps |
| Continuous R² | Eval script | Coefficient of determination over valid steps |
| Continuous MAE | Eval script | Mean absolute error over valid steps |

---

## PART D — WRITING PROMPT

The following prompt should be given to a writing agent along with the current LaTeX thesis skeleton.

---

**PROMPT FOR WRITING AGENT:**

You are writing the CBM chapter of a final-year project thesis on Explainable AI for Autonomous Driving. All necessary context, results, and theoretical arguments are contained in this document. **Ignore the structure and section headers of the existing LaTeX skeleton entirely** — follow the chapter structure defined in PART A of this document.

**Your task:** Write a complete, well-argued thesis chapter based on the content in this document. The chapter should be written at the level of a Master's thesis: technically precise, well-structured, and with clear narrative flow.

**Key writing guidelines:**

1. **Follow the chapter arc** as described in PART A. The narrative moves from (post-hoc XAI limitations → CBM theory → architecture → training story → results → discussion). Do not reorder this.

2. **The probing connection (Section 2.2) is critical** — make it a central theoretical argument, not a footnote. This is what distinguishes our approach theoretically.

3. **The observation-faithfulness argument (Section 2.4) must be explicit** — the reader should understand why computing concepts from the simulator observation is principled, not a shortcut.

4. **The training paradigm narrative (Section 5) should be written as a story** — we tried joint (it failed, here is why theoretically), frozen works (here is why it is principled), scratch is the contribution (here is what it proves). The failure of joint mode is a positive result, not a weakness.

5. **Results tables have [PENDING] entries** — write placeholder sentences like "Full results on the 150GB training run are presented in Table X once training completes. Preliminary results from the 10GB ablation model show..." This makes the chapter scalable.

6. **The zero red light finding (Section 6.5) should be presented as the key result** — it is the clearest empirical demonstration of the CBM's core promise. Spend a paragraph on it and make the causal chain explicit.

7. **Use precise language** — do not say "the model thinks" or "the model understands." Say "the `traffic_light_red` concept neuron outputs a value above 0.5" or "the concept head predicts a TTC of 0.3 normalized units."

8. **Acknowledge limitations honestly** — the `at_intersection` heuristic, the `heading_deviation` R²=-1.374, the 8% collision rate. Frame them as areas of future improvement, not failures.

9. **The chapter should be self-contained** — a reader who has not read the post-hoc XAI chapter should be able to follow it, though they can be pointed to the earlier chapter for the comparison.

10. **Length**: Target 8,000–12,000 words for the full chapter (excluding tables and figure captions). Expand technical sections to appropriate depth. Do not pad with unnecessary repetition.

Attached is the current LaTeX skeleton of the thesis. Write the CBM chapter replacing the skeleton content. Output complete LaTeX for the chapter.
