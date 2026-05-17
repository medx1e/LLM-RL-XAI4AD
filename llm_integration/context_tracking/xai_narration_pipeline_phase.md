# Project Context: Attention-Grounded XRL Narration Pipeline

## 1. Phase Overview
**Phase:** XAI Narration Pipeline Finalization
**Primary Objective:** To finalize and operationalize the LLM-based narration layer of the Attention-Grounded Explainable RL (XRL) pipeline. This phase successfully bridged the Computation Layer (JAX-based metric extraction) with the Narration Layer, enabling both online and offline natural language explanations of the autonomous driving agent's behaviors out of structured semantic graphs and attention metrics.

## 2. Theoretical Foundation & Design Goals
**Problem Statement:** Complex reinforcement learning policies for autonomous driving suffer from opacity. It is crucial to translate raw mathematical abstractions (e.g., cross-attention weights, counterfactual collisions, and scene graphs) into human-readable causal narratives that assert whether an agent was genuinely attentive to the threats it successfully evaded.

**Design Goals & Principles:**
*   **Modular Separation of Concerns:** Strongly decouple mathematical metric computation (JAX) from language generation logic (Python/HTTP) to ensure pipeline scalability and to avoid slowing down critical simulation loops.
*   **Agnostic Offline/Online Execution:** Support generating narratives "online" during policy evaluation or "offline" in batch modes using serialized JSON state reports.
*   **Deterministic Prompt Curation:** Enforce a strict "No Speculation" rule by feeding the LLM only pre-computed, deterministic facts filtering out noise.
*   **Contextual Awareness:** Synthesize the 5 closest agents natively into the prompt context to ensure the language model has a tangible grasp of the ego-vehicle's immediate surroundings.

## 3. Implementation Summary
A comprehensive suite of updates was applied to finalize the end-to-end integration:
*   **Offline Narration Orchestrator (`run_offline_narration.py`):** Developed a standalone processing engine to iteratively parse raw generated JSON scenario reports, apply LLM templates, and securely persist narratives back to disk iteratively while capturing network/latency metrics (resolution to 4 decimals).
*   **Robust LLM Client (`llm_narrator.py`):** Migrated the HTTP implementation to the `requests` library. Integrated standard `User-Agent` headers for high-fidelity compatibility with enterprise API gateways (e.g., Cloudflare) and integrated elapsed generation metrics to support system profiling.
*   **Report & Prompt Integration (`report_builder.py` & `prompt_builder.py`):** Refined the report builder to synthesize raw `semantic_edges` into a unified `scene_description` (distance, speed, relative alignment, and Time-to-Collision). This data is pipelined into `prompt_builder` where it gets automatically translated into precise English string precursors before reaching the LLM.
*   **Adaptive Encoder Detection (`run_xai_eval.py`):** Modernized the orchestrator to dynamically introspect the agent's actual underlying Hydra `observation_config` (e.g., deriving 1811 vs 280 exact token capacities dynamically).
*   **Attention Grounding Refinement:** Engineered an auto-transposition feature inside `AttentionGrounder` to gracefully support dynamically sequenced axis shapes out of `LQEncoder` outputs.

## 4. Key Decisions & Rationale
*   **Filtering Semantic Graph to the Closest 5 Agents:**
    *   *Rationale:* Passing the entire dense semantic graph (potentially 64+ objects) to the LLM directly invokes token-bloat, increasing latency and prompt hallucination risks. By strictly filtering context to the tightest spatial radius prior to LLM injection, language outputs remain surgically targeted, concise, and temporally highly performant.
    *   *Consideration:* Handled strictly inside the computation pipeline (`report_builder.py`) so the standalone offline LLM script requires no re-computation dependencies.
*   **Switching from `urllib` to `requests`:**
    *   *Rationale:* Advanced AI services are guarded by sophisticated Bot Management solutions. The standardized connection pooling, robust header negotiation (Spoofed Browser User-Agents), and native telemetry (e.g., `requests.elapsed`) afforded straightforward operational stability and API analytics tracking.
*   **Dynamic Token Layout and Axis Detection:**
    *   *Rationale:* Hardcoding token counts caused pipeline fragility across varying Wayformer/Perceiver architectural footprints. By introspecting the Hydra configuration live, the `AttentionGrounder` dynamically scales to any experimental model checkpoint securely natively adapting to diverse output matrices.
