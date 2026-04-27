# Post-Hoc XAI Framework — Full Context

**Project Title:** LLM-based Explainable Deep Reinforcement Learning for Autonomous Driving  
**Date:** 2026-04-20  
**Author:** med1e (Final Year Project)

---

## 1. What Is Built

A JAX-native post-hoc explainability framework for V-MAX autonomous driving policies trained with reinforcement learning on the Waymo Open Motion Dataset. The framework wraps pretrained model weights and exposes 7 XAI methods, 3 evaluation metric families, visualization tools, event mining, and an attention-reward correlation analysis module.

**The key value proposition:** You can take any of the 35+ pretrained V-MAX models, pick a driving scenario and timestep, and get an explanation of *what the model was attending to and why it made that action* — broken down by feature category (ego history, other agents, road graph, traffic lights, GPS route) and by individual entities (agent_0, agent_1, ...).

---

## 2. Implemented Components

### 2.1 Core XAI Methods (`posthoc_xai/methods/`)

All 7 methods return a standardized `Attribution` dataclass:
- `raw_attribution` — unnormalized per-feature scores
- `normalized` — L1-normalized (sum to 1)
- `category_importance` — dict keyed by category (sdc_trajectory, other_agents, roadgraph, traffic_lights, gps_path)
- `entity_importance` — per-entity breakdown (agent_0, agent_1, ...)
- `method_name`, `computation_time_ms`

| Method | Class | Algorithm | Cost |
|--------|-------|-----------|------|
| Vanilla Gradient | `VanillaGradient` | `∂f(x)/∂x` | 1 forward+backward |
| Integrated Gradients | `IntegratedGradients` | Trapezoidal sum over α ∈ [0,1] path | n_steps × 2 (default 50 steps) |
| SmoothGrad | `SmoothGrad` | Average gradients over noisy inputs | n_samples passes (default 50) |
| Gradient × Input | `GradientXInput` | `x ⊙ ∂f(x)/∂x` | 1 forward+backward |
| Perturbation | `PerturbationAttribution` | Mask category → measure output drop | n_categories passes |
| Feature Ablation | `FeatureAblation` | Remove category, normalize by size | n_categories passes |
| SARFA | `SARFA` | Relevance × Specificity (Puri et al. 2020) | n_categories passes |

**Top-level API:**
```python
import posthoc_xai as xai
model = xai.load_model("runs_rlc/womd_sac_road_perceiver_minimal_42")
results = xai.explain(model, observation)  # dict[str, Attribution]
```

### 2.2 Model Wrappers (`posthoc_xai/models/`)

| Wrapper | Encoders | Attention Extraction |
|---------|----------|---------------------|
| `PerceiverWrapper` | LQ / Perceiver | Full: reconstructs softmax(QKᵀ/√d) at all 4 layers via `capture_intermediates` |
| `GenericWrapper` | MTR, Wayformer, MGAIL/LQH, MLP/None | Gradient-based only; no native attention extraction yet |

**Key feature:** Observation structure is discovered *dynamically* — the framework probes the model's `unflatten_fn` to find category boundaries and per-entity index ranges. No hardcoding.

### 2.3 Evaluation Metrics (`posthoc_xai/metrics/`)

| File | Metrics |
|------|---------|
| `faithfulness.py` | Deletion curve AUC, Insertion curve AUC, attention-gradient correlation |
| `sparsity.py` | Gini coefficient, top-k concentration, entropy |
| `consistency.py` | Cross-scenario Pearson correlation of attributions |

### 2.4 Visualization (`posthoc_xai/visualization/`)

- `plot_category_importance()` — bar chart of importance per category
- `plot_method_comparison()` — side-by-side multi-method bar charts
- `plot_deletion_insertion_curves()` — faithfulness curves
- `plot_entity_importance()` — per-agent heatmap with validity coloring
- Temporal attention traces, BEV overlay (`bev_attention.py` in reward_attention/)

### 2.5 Experiment Pipeline (`posthoc_xai/experiments/`)

Full orchestration: scan scenarios → select interesting ones → run all methods → compute metrics → generate reports. Config-driven via `ExperimentConfig`.

### 2.6 Event Mining (`event_mining/`)

Detects critical driving events from rollout trajectories:
- `HAZARD_ONSET` (TTC threshold crossed)
- `NEAR_MISS` (min distance < threshold)
- `HARD_BRAKE` (|accel| > threshold)
- `EVASIVE_STEERING` (|steering| > threshold)
- `COLLISION`, `OFF_ROAD`

`EventCatalog` is JSON-serializable and queryable. `XAIBridge` connects event mining directly to the XAI pipeline — you can ask: "give me all near-miss timesteps and run explanations on them."

### 2.7 Reward-Attention Correlation (`reward_attention/`)

The core research contribution: does the model's attention correlate with *risk*?

- 4 risk metrics: collision_risk (TTC-based), safety_risk (proximity), navigation_risk (route deviation), behavior_risk (acceleration/steering)
- Attention fractions tracked: attn_agents, attn_roadgraph, attn_gps, attn_traffic_lights
- Hypotheses tested with Pearson + Spearman correlation:
  - `collision_risk ↔ attn_agents` → positive expected (danger → focus agents)
  - `navigation_risk ↔ attn_gps` → positive expected (off-route → focus GPS)
  - `collision_risk ↔ attn_roadgraph` → negative expected (when focused on agents, less road)
- Temporal analysis around critical events (attention shift before/after hazard)
- BEV overlay visualization

---

## 3. Verified Working

| Encoder | Example Model | Status |
|---------|--------------|--------|
| LQ/Perceiver | womd_sac_road_perceiver_minimal_42 | PASS — roadgraph 69.1%, gps 22.0% |
| MTR | womd_sac_road_mtr_minimal_42 | PASS — roadgraph 59.6%, agents 27.7% |
| Wayformer | womd_sac_road_wayformer_minimal_42 | PASS — roadgraph 70.5%, agents 11.9% |
| MGAIL/LQH | womd_sac_road_mgail_minimal_42 | PASS — gps 41.2%, roadgraph 38.5% |
| MLP/None | womd_sac_road_none_minimal_42 | PASS — roadgraph 32.1%, agents 27.8% |

All 7 XAI methods verified on Perceiver model. All 3 metric families verified.

---

## 4. Known Gaps & Limitations

| Gap | Severity | Workaround |
|-----|----------|-----------|
| Cannot load multiple models in same process (Waymax registry conflict) | Medium | Use separate processes or restart kernel between models |
| SAC seed0/42/69 models broken (speed_limit feature missing in Waymax) | Low | Use `womd_*` models only — 32+ models still available |
| MTR/Wayformer native attention extraction missing | Low | Gradient-based methods work fine; only lose interpretable attention heads |
| Event detector thresholds need calibration | Medium | Run on a validation set and tune before final demo |
| No ground-truth explanation labels for formal validation | High | See validation strategy below |

---

## 5. Validation Strategy — The Hard Question

**The core problem:** There is no ground truth for "what should the model have paid attention to." Standard XAI benchmarks don't exist for RL driving policies.

### What We Can Do Instead

#### 5.1 Quantitative Metrics (Already Implemented)
- **Faithfulness (Deletion/Insertion AUC):** If roadgraph tokens are truly important and we delete them, does model performance degrade? Lower deletion AUC = more faithful.
- **Sparsity:** Are explanations focused (high Gini) or diffuse? Focused explanations are more human-interpretable.
- **Consistency:** Does the same model produce similar explanations across similar scenarios? High Pearson correlation = stable explanation.
- **Attention-Gradient Correlation:** For Perceiver models, does gradient attribution agree with raw attention weights? High correlation = internal coherence.

#### 5.2 Qualitative Sanity Checks (What Actually Convinces a Jury)
The most convincing validation is: **find scenarios where the explanation makes obvious intuitive sense, and others where it reveals something non-obvious.**

Examples to look for:
- **Near-miss avoidance:** At the timestep the model steers sharply, agents attribution should spike
- **GPS following on open road:** When there are no agents around, GPS path should dominate
- **Traffic light response:** Attention to traffic_lights should increase as ego approaches intersection
- **Offroad failure (Wayformer):** Look at what the model ignores right before going offroad — does it fail to attend to roadgraph?

#### 5.3 Cross-Model Comparison (Strongest Contribution)
Compare explanations across encoder architectures on the *same scenario and timestep*:
- Perceiver vs. Wayformer: do they focus on the same things?
- Models with different accuracy: does the better model have more faithful / more focused explanations?
- This is genuinely interesting and publishable — nobody has done this across V-MAX encoder variants.

#### 5.4 Method Agreement Analysis
If 4 out of 7 methods agree that `other_agents` is the top category in a dangerous scenario, that's strong convergent evidence — even without ground truth.

---

## 6. Streamlit Demo Plan

### Recommended Architecture

```
streamlit_app/
├── app.py                    # Main entrypoint
├── pages/
│   ├── 01_scenario_browser.py   # Pick scenario + run model
│   ├── 02_explanation_viewer.py # Run XAI, view results
│   ├── 03_method_comparison.py  # Side-by-side XAI methods
│   ├── 04_model_comparison.py   # Same scenario, different models
│   └── 05_attention_reward.py   # Reward-attention correlation
├── cache/
│   └── precomputed_results.pkl  # Pre-run to avoid 10min JIT at demo
└── assets/
    └── scenario_thumbnails/
```

### Page 1 — Scenario Browser
- Show a dropdown/list of pre-selected "interesting" scenarios with thumbnails
- Display: scenario ID, length, event labels (near-miss, offroad, etc.)
- Button: "Run this scenario" → stores rollout in session state
- **Key UX:** Show a BEV animation of the scenario playing out

### Page 2 — Explanation Viewer
- Select timestep via slider (with event markers highlighted — e.g., red dot = near-miss)
- Select XAI method via radio buttons
- Display: bar chart of category importance + entity importance heatmap
- Show raw observation context alongside (ego speed, nearest agent distance, etc.)
- Optional: show attention heads (Perceiver only)

### Page 3 — Method Comparison
- Fixed scenario + timestep
- Run all 7 methods in parallel (or show precomputed)
- Side-by-side bar charts (already implemented in `plot_method_comparison()`)
- Highlight agreement / disagreement between methods

### Page 4 — Model Comparison
- Fixed scenario + timestep
- Select 2–3 models via multiselect
- Same XAI method applied to all → radar chart or grouped bars
- Show model accuracy alongside for context

### Page 5 — Attention–Risk Correlation
- Time series plot: risk metrics + attention fractions over a full episode
- Scatter plot: attn_agents vs. collision_risk (colored by timestep)
- Show correlation coefficients with interpretation

### Implementation Notes
- **Precompute everything.** JIT takes 10 min on first run. Run all scenarios × methods × models overnight, pickle results. The demo loads from cache.
- Use `st.cache_data` on all heavy functions
- Keep model loaded in `st.session_state` — don't reload between pages
- One process per model (Waymax registry issue) — pre-run and store results
- Target: < 5 seconds to show any explanation once caching is warm

---

## 7. Curated Scenario Selection Strategy

The goal: find 5–8 scenarios that together tell a coherent story. Run the event miner + XAI pipeline on 50–100 scenarios first, then curate manually.

### Scenario Archetypes to Find

| Archetype | What to Look For | XAI Story |
|-----------|-----------------|-----------|
| **Clean cruise** | No events, smooth driving | Baseline: roadgraph + GPS dominate |
| **Agent avoidance** | Near-miss or hard brake | Agents attribution spikes exactly when needed |
| **Intersection** | Traffic light interaction | Traffic light attention increases as ego approaches |
| **Route following** | GPS deviation followed by correction | GPS attribution drives correction behavior |
| **Model failure** | Wayformer offroad at step 46 (already found!) | What did the model ignore? — story of bad attention |
| **Encoder disagreement** | Same scenario, different attention patterns | What makes Perceiver better than Wayformer? |

### Workflow to Find Them
1. Run event miner across 100 scenarios with Perceiver model
2. Filter: keep scenarios with at least 1 `NEAR_MISS` or `HARD_BRAKE` or `OFF_ROAD` event
3. For each, run all 7 XAI methods at the event timestep
4. Score by: explanation coherence (high Gini), method agreement, visual clarity on BEV
5. Pick 5–8 that cover diverse archetypes
6. Manually verify each makes intuitive sense
7. Lock them in as the demo scenario set

---

## 8. Narrative for Defense

The story arc for the presentation:

1. **Problem:** RL-trained driving policies are black boxes. They work, but we don't know *why*. Regulators and safety engineers need explanations.

2. **Approach:** Post-hoc XAI — we don't open the box during training, we analyze it after. This is practical because the models already exist.

3. **What we built:** A unified framework that applies 7 XAI techniques to any V-MAX encoder. First known XAI comparison across all 5 V-MAX encoder architectures.

4. **Key findings:**
   - Model attention correlates with risk metrics (attention-reward correlation results)
   - Different encoder architectures have fundamentally different explanation patterns
   - Integrated Gradients and SmoothGrad agree with each other more than with SARFA — what does that mean?
   - The Wayformer failure case: the model went offroad because it stopped attending to roadgraph tokens

5. **The LLM angle:** Use an LLM to narrate the explanations in natural language — "At timestep 46, the model primarily attended to agent_2 (the vehicle cutting in from the left) and suddenly shifted attention away from the road boundary, leading to lane departure." This bridges the gap between the XAI output (numbers) and a human-understandable explanation.

---

## 9. The LLM Integration Layer (Next Priority)

The project title says "LLM-based explainable DRL" — the LLM needs to be more than decoration.

### Minimal Viable LLM Integration
- Take the `Attribution` output from any method
- Format into a structured prompt with: scenario context, timestep info, category importances, entity importances, event labels
- Ask Claude/GPT to generate: (a) a natural language explanation of what the model focused on, (b) whether the explanation is concerning or reassuring, (c) what a safety engineer should examine

### Stronger LLM Integration
- Give the LLM the *sequence* of attributions over time → ask it to narrate the episode
- Multi-modal: feed BEV frame + attribution numbers → visual question answering
- LLM as evaluator: given two explanations, which is more faithful? (Substitute for missing ground truth)

### Positioning
Frame the LLM as a "translation layer" between quantitative XAI output and human-understandable safety reports. This is a real gap in the XAI literature — most methods produce numbers, not sentences.

---

## 10. Immediate Next Steps (Prioritized)

| Priority | Task | Time Estimate |
|----------|------|--------------|
| 1 | Run event miner on 50+ scenarios, build curated scenario catalog | 1 day |
| 2 | Precompute XAI results for curated scenarios (all methods × 2–3 models) | 1 day |
| 3 | Build Streamlit app skeleton (pages 1–3 minimum) | 2 days |
| 4 | Add LLM narration to Page 2 (Claude API call on Attribution output) | 1 day |
| 5 | Run reward-attention correlation on 100+ scenarios, get correlation plots | 1 day |
| 6 | Write up validation story (faithfulness curves + coherence analysis) | 1 day |
| 7 | Record demo video as backup (in case live demo fails at defense) | half day |

---

## 11. File Map Quick Reference

```
post-hoc-xai/
├── posthoc_xai/
│   ├── __init__.py              # load_model(), explain() top-level API
│   ├── models/
│   │   ├── base.py              # ExplainableModel ABC, ModelOutput
│   │   ├── loader.py            # load_vmax_model(), all 5 compatibility fixes
│   │   ├── perceiver_wrapper.py # Full attention extraction for LQ/Perceiver
│   │   ├── generic_wrapper.py   # MTR, Wayformer, MGAIL, MLP
│   │   └── _obs_structure.py    # Dynamic observation layout discovery
│   ├── methods/
│   │   ├── base.py              # Attribution dataclass, AttributionMethod ABC
│   │   ├── vanilla_gradient.py
│   │   ├── integrated_gradients.py
│   │   ├── smooth_grad.py
│   │   ├── gradient_x_input.py
│   │   ├── perturbation.py
│   │   ├── feature_ablation.py
│   │   └── sarfa.py
│   ├── metrics/
│   │   ├── faithfulness.py      # Deletion/Insertion AUC, attention-gradient corr
│   │   ├── sparsity.py          # Gini, top-k, entropy
│   │   └── consistency.py       # Cross-scenario Pearson correlation
│   ├── visualization/
│   │   └── heatmaps.py          # All plot functions
│   └── experiments/
│       ├── config.py            # ExperimentConfig dataclass
│       ├── scanner.py           # Scenario scanning and selection
│       ├── analyzer.py          # Run methods + metrics on scenarios
│       ├── reporter.py          # Summary JSON + comparison plots
│       └── runner.py            # run_experiment(), compare_experiments()
├── event_mining/
│   ├── events/                  # base.py, safety.py, action.py, outcome.py
│   ├── miner.py                 # EventMiner orchestrator
│   ├── catalog.py               # EventCatalog (queryable, JSON)
│   ├── metrics.py               # TTC, distance, criticality
│   └── integration/
│       ├── vmax_adapter.py      # V-MAX rollout → ScenarioData
│       └── xai_bridge.py        # EventCatalog → XAI pipeline
├── reward_attention/
│   ├── config.py                # AnalysisConfig, token structure (280 tokens)
│   ├── extractor.py             # AttentionTimestepCollector
│   ├── correlation.py           # CorrelationAnalyzer, Pearson+Spearman
│   ├── temporal.py              # Attention trajectories around events
│   ├── risk_metrics.py          # RiskComputer (4 risk types)
│   ├── visualization.py         # Temporal traces, heatmaps
│   ├── bev_attention.py         # BEV overlay
│   └── run_experiment.py        # Full pipeline orchestration
├── runs_rlc/                    # 35+ pretrained model weights
├── data/training.tfrecord       # Waymo dataset (1GB)
├── explore_vmax.py              # Full exploration script
├── run_inference.py             # Quick inference script
├── .xai_progress.md             # Implementation progress tracker
└── CLAUDE.md                    # Environment + compatibility notes
```

---

## 12. For the AI Agent Reading This

The framework is feature-complete for the core XAI pipeline. The immediate work is:

1. **Streamlit demo** — the architecture is in Section 6, precomputed caching is essential
2. **LLM integration** — see Section 9; the `Attribution` dataclass is the input to the LLM prompt
3. **Scenario curation** — run `event_mining` first, then pick scenarios from Section 7
4. **Validation story** — use faithfulness curves + method agreement as the quantitative case; use curated scenario visualizations as the qualitative case

The model to default to: `womd_sac_road_perceiver_minimal_42` (uses `PerceiverWrapper`, has full attention extraction, 97.47% accuracy, verified working).

The single most important Streamlit page for a defense demo: **Page 2 (Explanation Viewer)** with a timestep slider, event markers, category importance bar chart, and LLM narration box.
