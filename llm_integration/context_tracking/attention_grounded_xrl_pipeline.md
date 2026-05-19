# Phase Overview

**Phase Name:** Attention-Grounded Counterfactual XRL Pipeline Implementation
**Objective:** To finalize a modular, attention-grounded counterfactual explainability (XRL) pipeline for the V-Max autonomous driving agent. The goal is to accurately assess the necessity of driving decisions, ground those assessments in the model's internal attention mechanisms, and generate coherent, natural language rationales for the model's behavior.

# Theoretical Foundation & Design Goals

The core problem addressed in this phase is the opacity of end-to-end autonomous driving models. While models like the Wayformer can make safe and efficient decisions most of the time, stakeholders require an understanding of *why* those decisions are made, particularly in critical driving scenarios.

The pipeline is built upon three primary design paradigms:
*   **Counterfactual Reasoning:** By simulating alternative futures (counterfactual trajectories) within the V-Max simulator, the system evaluates the constraint (or "Necessity") surrounding the chosen action. This establishes the objective danger of a given driving context.
*   **Attention Grounding:** Objective danger alone does not explain the *model's* reasoning. By correlating the objective severity of approaching threats with the model's internal cross-attention allocations, the system evaluates whether the model properly prioritized those threats.
*   **Modular Narration:** To make these mathematical metrics understandable to human stakeholders, a structured reporting, routing, and narration mechanism acts as an interface between raw signals and high-level natural language generation via LLMs.

# Implementation Summary

The XRL pipeline was fully implemented and divided into two decoupled layers coordinated by a central orchestrator:

**Computation Layer (JAX-based)**
*   **Semantic Graph Builder (`semantic_graph.py`):** Translates raw simulator states into relational graphs containing context categories (e.g., following, threat approaching).
*   **Adaptive Action Grid (`adaptive_action_grid.py`):** Dynamically builds a contextual grid of action perturbations based on the current semantic context to ensure relevant counterfactual testing.
*   **Counterfactual Explainer (`counterfactual_explainer.py`):** Runs parallel JAX `vmap` rollouts over the dynamic grid to project future outcomes, identifying imminent collision risks and extracting specific threat agent IDs via pairwise bounding box overlaps.
*   **Necessity Scorer (`necessity_scorer.py`):** Computes the constraint of the chosen action based on the failure ratio of tested alternatives.
*   **Attention Grounder (`attention_grounder.py`):** Computes a grounding score through a dot product of the model's attention mass (extracted from the encoder) and the objective severity of the threat agents.
*   **Decision Classifier (`decision_classifier.py`):** Classifies the moment into a 2x2 grid (e.g., `GROUNDED_CRITICAL`, `UNGROUNDED_CRITICAL`).

**Narration Layer (Python-based)**
*   **Report Builder & Router (`report_builder.py`, `narration_router.py`):** Assembles a structured JSON snapshot of the step and routes the classification to appropriate language templates.
*   **Prompt Builder & Narrator (`prompt_builder.py`, `llm_narrator.py`):** Uses the templates to format zero-shot prompts and securely interfaces with the OpenRouter API to fetch natural language explanations.

**Orchestrator (`run_xai_eval.py`)**
*   Executes the pipeline within the existing V-Max evaluation loop.
*   Integrates dynamic frequency controls configurable via `xai_config.yaml` to decouple the execution rate of metric computations, report generation, and LLM API calls.
*   Surfaces raw attention weights directly from the underlying SAC/PPO/BC policy inference functions.

# Key Decisions & Rationale

*   **Extraction of Attention from Forward Pass:** Rather than executing a secondary encoder-only forward pass to obtain attention matrices, the algorithm factories (SAC, PPO, BC) were modified to surface their internal `encoder_attn_weights` through the `rl_transition.extras` dictionary. This guarantees temporal consistency (the grounded attention matches the exact weights used for the decision) and avoids redundant computational payload during inference.
*   **Pairwise Overlaps for Threat Identification:** The vectorized counterfactual explainer was designed to identify the exact agent causing a collision condition. Since standard overlap metrics return an array of colliding entities rather than the agent pair, the vectorized kernel uses matrix argmax operations iteratively against zeroed-out ego-overlap rows. This ensures the deterministic identification of threat objects required by the attention grounding module.
*   **Adaptive Context-Driven Action Grids:** Instead of running an exhaustive Cartesian product of all possible steering and acceleration combinations universally, the system first constructs a semantic graph to narrow down the viable action space (e.g., generating avoidance maneuvers only when a threat is actively approaching). This architectural choice dramatically cuts the memory required by JAX's `vmap` rollouts and keeps the reasoning space aligned with human behavioral intuition.
*   **Decoupled Metric Frequencies:** Computations such as plotting, counterfactual rollouts, and LLM narration are computationally asymmetrical. Adding a `frequencies` configuration block to `xai_config.yaml` enables the system to run basic telemetry every step but limit LLM calls or graph parsing to sub-sampled intervals, ensuring the overarching simulation loop maintains high performance.
