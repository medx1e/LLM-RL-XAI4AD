# Weekly Research Presentation: Refining Attention Head Specialization

## [SLIDE 1: PHASE OVERVIEW & PRIMARY OBJECTIVE]
*   **The Mission:** Rebuilding the Head Specialization Index (HSI) pipeline to move from coarse, module-level attention sums to granular, entity-level temporal dynamics.
*   **Why it Matters:** The Wayformer model uses late-fusion. Module-level sums are mathematically constrained to ~1.0, masking the "selectivity" within the module (e.g., which specific vehicle is being prioritized).
*   **Objective:** Implement a distributional analysis that identifies *why* a head attends to specific entities across all input modalities (Agents, Roadgraph, GPS, Traffic Lights, SDC History).

## [SLIDE 2: METHODOLOGY: WITHIN-MODULE DISTRIBUTIONAL ANALYSIS]
*   **The Core Shift:** Instead of measuring "how much attention the agent module gets," we measure "how attention is distributed *among agents* within that module."
*   **Implementation:** 
    *   Extract attention weight matrices at every timestep of an expert log-replay (rollout).
    *   Compute fractions per entity (e.g., Vehicle 1 gets 40% of the head's "agent-module" budget).
*   *Concept Clarification:*
    *   **HSI (Head Specialization Index):** A metric (0.0 to 1.0) representing the maximum correlation between a head's attention and a semantic feature. Higher = more specialized.
    *   **Selectivity:** The model’s ability to "pick" one entity over others based on its attributes (like being the closest vehicle).

## [SLIDE 3: OPTION A: CONSISTENT SELECTIVITY]
*   **The Approach:** Correlating attention with semantic features (TTC, Distance, Speed) across entities at *each* timestep.
*   **Aggregation:** We aggregate these correlations over the entire episode using Fisher Z-transforms.
*   *Concept Clarification:*
    *   **Spearman Rank Correlation:** A non-parametric measure of the relationship between two variables. It handles non-linear relationships better than standard correlation.
    *   **Fisher Z-Transform:** A statistical method used to aggregate multiple correlation coefficients to avoid the bias of simply averaging them.
*   **Analogy:** If a head consistently focuses on the driver who just cut you off at every second of the video, it shows "Consistent Selectivity."

## [SLIDE 4: OPTION B: RISK-DEPENDENT SELECTIVITY]
*   **The Approach:** Observing how a head's "behavior" changes as the scene gets dangerous.
*   **Mechanism:** Correlate the *Selectivity timeseries* (from Option A) with the *Risk timeseries* ($R$).
*   *Concept Clarification:*
    *   **Collision Risk ($R$):** A scene-level metric (0.0 to 1.0) derived from Time-to-Collision (TTC). $R=1$ means an imminent collision.
    *   **Recency Bias:** The tendency of a head to focus on more recent observations in the SDC history as danger increases.
*   **Example:** A head that "wakes up" and starts focusing on your navigation waypoints only when you're entering a sharp turn at high speed.

## [SLIDE 5: EXPERIMENTAL RESULTS & OBSERVED OUTCOMES]
*   **Current Status:** The multi-timestep extraction and Option A/B analysis engine is fully functional and validated on the Wayformer architecture.
*   **Quantitative Results:**
    *   Replaced "Sum-to-1.0" artifacts with true Selective distributions (Max vehicle fractions now range from 0.1 to 0.9 across heads).
    *   Statistical significance ($p < 0.05$) now markers the most especializados heads in the registry.
*   **Qualitative Results:**
    *   Diagnostics now show clear "Risk-Triggered Safety Heads" that increase their focus on low-TTC vehicles only during $R > 0.4$ phases.
    *   Improved visualization: Legends now show semantic context (e.g., "Agent 0 (12m)") allowing instant validation of "Proximity Heads."

## [SLIDE 6: PROPOSED NEXT STEPS & IMMEDIATE FOCUS]
*   **Immediate Focus:** Execute the pipeline across multiple model checkpoints (e.g., Step 50k, 100k, 500k) to observe the *evolution* of specialization.
*   **Feature Expansion:** Correlate roadgraph attention with "Distance to Intersection" to identify "Topology Heads."
*   **Refinement:** Adjust the "High Variation" threshold ($\text{std}(R) > 0.2$) to ensure we only analyze episodes with meaningful driving dynamics.
*   **Final Goal:** Export the `head_registry.json` to the online dashboard for real-time model debugging.
