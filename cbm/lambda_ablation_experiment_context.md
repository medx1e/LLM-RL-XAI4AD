# CBM Lambda Ablation Experiment — Context for External Review

## What We Are Building

A **Hard Concept Bottleneck Model (CBM)** for autonomous driving, built on top of a SAC (Soft Actor-Critic) RL agent running in the Waymax simulator (Google's JAX-based AV simulator using Waymo Open Motion Dataset).

**Architecture:**
```
obs (1655-d) → [LQ Encoder] → z (128-d) → [Concept Head] → c (15-d) → [Actor/Critic] → action
```

The actor and critic see **only** the 15-d concept vector — never the raw latent z. This is a hard bottleneck. The 15 concepts are human-interpretable (e.g., ego speed, distance to nearest object, traffic light red, path curvature).

**Training loss:**
```
L_total = L_SAC + λ × L_concept
```

`L_concept` is a supervised loss aligning the concept head output with ground-truth concept values computed deterministically from the simulator observation (no human labels). Binary concepts use BCE, continuous use Huber loss.

**This experiment: training from scratch** (no pretrained encoder). The encoder starts from random weights and must learn spatial representations guided entirely by the concept supervision + RL reward. This is distinct from our "frozen" paradigm where a pretrained encoder is locked and only the concept head + actor/critic are trained.

---

## The Experiment: Lambda Ablation

We ran 3 training runs on 10GB of WOMD data (~1M environment steps each), varying only `λ` (the concept loss weight):

- **λ=0.01** (weak concept supervision)
- **λ=0.1** (default)
- **λ=0.5** (strong concept supervision)

All other hyperparameters identical: SAC with α=0.2, lr=1e-4, batch=128, buffer=1M, 15 concepts (8 Phase1 + 3 Phase2 + 4 Phase3 path geometry).

---

## Results

### Task Performance (all on training rollouts — NOT held-out val)

| Metric | λ=0.01 | λ=0.1 | λ=0.5 |
|---|---|---|---|
| Route Progress (nuplan) | 0.420 | **0.428** | 0.427 |
| At-Fault Collision | 0.0004 | **0.0000** | **0.0000** |
| Episode Reward (final) | 3.03 | **3.81** | 3.70 |
| Policy Loss (final) | -22.52 | -23.33 | **-23.39** |
| Policy Loss Std (last 20%) | 0.22 | 0.27 | 0.24 |

### Concept Quality

**Binary concept accuracy (final):**

| Concept | λ=0.01 | λ=0.1 | λ=0.5 |
|---|---|---|---|
| `traffic_light_red` | 76.0% ❌ | 99.7% | **99.97%** |
| `at_intersection` | 70.0% ❌ | 98.9% | **99.7%** |
| `lead_vehicle_decelerating` | 98.2% | 98.6% | **98.8%** |

**Concept loss (final, lower = better):**

| Concept | λ=0.01 | λ=0.1 | λ=0.5 |
|---|---|---|---|
| `traffic_light_red` | 0.5199 ❌ | 0.0125 | **0.0017** |
| `at_intersection` | 0.5656 ❌ | 0.0317 | **0.0086** |
| `ego_speed` | 0.0158 | 0.0110 | **0.0049** |
| `dist_nearest_object` | 0.0291 | 0.0085 | **0.0046** |
| `ttc_lead_vehicle` | 0.0968 | 0.0365 | **0.0315** |
| `path_curvature_max` (P3) | 0.0513 | 0.0341 | **0.0159** |
| `path_straightness` (P3) | 0.0167 | 0.0071 | **0.0032** |
| `heading_to_path_end` (P3) | 0.0537 | 0.0108 | **0.0078** |

**Continuous concept Δ% during training (λ=0.01 instability):**

Several concepts actively worsen with λ=0.01:
- `heading_deviation`: +623% (started good, got much worse)
- `path_net_heading_change`: +436%
- `heading_to_path_end`: +1834%
- `ego_acceleration`: +115%

With λ=0.1 and λ=0.5, continuous concepts generally improve or stay stable.

**Concept/Task gradient ratio (final):** ~1.8 for all three — concept head receives substantial gradient signal even at λ=0.01, but the quality of what gets learned differs dramatically.

**Concept prediction std (dead neuron check):**
- `progress_along_route`: low std (~0.02) across all — known issue, near-constant at t=0 in logged data, becomes dynamic during rollout
- `lead_vehicle_decelerating`: low std (~0.015-0.019) — sparse binary concept (~33% validity), expected behavior
- No truly dead neurons found

### Training Stability

Value loss shows a peak-then-converge pattern (normal for SAC critics). Route progress oscillates during training due to varying scenario difficulty in the dataset — not instability. Policy loss is monotonically improving and stable across all three lambdas.

---

## Our Decision: λ=0.5

Rationale:
- λ=0.01 is eliminated: binary concept accuracy fails completely for safety-critical concepts (TL red: 76%, at_intersection: 70%), and several continuous concepts actively worsen
- λ=0.5 vs λ=0.1: λ=0.5 wins on all concept quality metrics significantly; task performance is equivalent (zero collision, same route progress); episode reward slightly lower (3.70 vs 3.81) but this gap is on training data and considered noise at this scale
- For a CBM where **interpretability is the primary contribution**, concept quality takes priority over a marginal reward difference

---

## What Comes Next

- **Full 150GB run** with λ=0.5 (~15M steps, cluster GPU)
- **SAC-from-scratch baseline** (no CBM, no concept loss) in parallel — to quantify the "interpretability tax" (performance cost of the hard bottleneck)
- **Evaluation** on WOMD held-out validation split using `eval_cbm.py` — the training rollout metrics above are not reliable final numbers
- **Compare against**: SAC pretrained baseline (97.47% route progress), CBM Frozen V2 (15 concepts, 150GB, done), CBM Scratch V2 (15 concepts, this run)

---

## Questions for Review

1. Does λ=0.5 seem like the right choice given the tradeoff between concept quality and task performance?
2. Is the mid-training bump in Phase 3 concept losses (loss goes up then comes back down) a normal learning dynamic for from-scratch training, or a sign of something wrong?
3. Are there red flags in these numbers that suggest the 150GB run will fail or diverge?
4. Is training from scratch (no pretrained encoder) a scientifically interesting contribution on top of the frozen-encoder CBM, or redundant?
5. The concept/task gradient ratio stays ~1.8 across all lambdas — is this expected or should we be concerned that the ratio doesn't differentiate between λ values?
