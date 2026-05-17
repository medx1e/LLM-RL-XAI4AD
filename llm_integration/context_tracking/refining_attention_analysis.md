# Development Phase: Refinement of Attention Head Specialization Analysis

## Phase Overview
This development phase focused on rebuilding the **Head Specialization Index (HSI)** pipeline to accurately characterize the functional roles of attention heads within the Wayformer architecture. The primary objective was to replace flawed module-level aggregation with a **within-module distributional analysis** that captures the temporal dynamics and entity-level selectivity of the model across diverse driving scenarios.

## Theoretical Foundation & Design Goals
The work is grounded in the observation that global attention sums are uninformative in late-fusion transformer encoders, where each modality (vehicles, roadgraph, etc.) undergoes independent normalization.

*   **Design Paradigm:** Shift from "closed-budget" global analysis to "normalized selectivity" within individual modules.
*   **Functional Taxonomy:** Distinguish between heads based on two criteria:
    *   **Consistent Selectivity (Option A):** Heads that maintain a stable preference for specific semantic features (e.g., proximity or TTC) throughout a scenario.
    *   **Risk-Dependent Selectivity (Option B):** Heads that dynamically adjust their focus or concentration level in response to increasing collision risk ($R$).
*   **Intended Behavior:** The system should automatically identify specialized heads (e.g., "Threat-Distance Heads", "Risk-Triggered Safety Heads") by correlating attention distributions with semantic signals across multi-timestep rollouts.

## Implementation Summary
The refinement involved a deep restructuring of the XAI extraction and analysis codebase:

*   **Extraction Layer (`offline_extraction.py`):**
    *   Implemented a multi-timestep **expert rollout mode** using log-replay mechanisms to gather high-fidelity attention timeseries.
    *   Developed `_extract_within_module_distributions` to capture per-entity attention fractions for agents, traffic lights, and waypoints.
    *   Integrated scene-level collision risk estimation derived from per-agent Time-to-Collision (TTC) metrics.
*   **Analysis Layer (`head_specialization_analysis.py`):**
    *   Built a dual-stream correlation engine using **Spearman rank correlation** and **Fisher z-transform aggregation** to pool results across episodes reliably.
    *   Implemented **Risk-Conditioned Profiling**, comparing attention concentration (max fraction, entropy) between "Calm" ($R < 0.1$) and "Danger" ($R > 0.4$) driving phases.
    *   Expanded analysis to cover all input modalities, including Roadgraph, GPS Waypoints (center-of-mass shift), and SDC Recency bias.
*   **Visualization Suite:**
    *   Created automated diagnostics including **Correlation Heatmaps** with statistical significance markers, **Risk-Profile Bar Charts**, and **Multi-Panel Timeseries Plots** showing the interplay between scene risk and head focus.

## Key Decisions & Rationale

*   **Modality-Specific Normalization:** Analysis was localized to within-module distributions rather than cross-module sums.
    *   *Rationale:* Directly aligns with the Wayformer's late-fusion architecture, ensuring the metrics reflect the model's actual internal logic rather than normalization artifacts.
*   **Fisher Z-Aggregation for Correlational Meta-Analysis:** Used Fisher z-transforms for all cross-episode pooling.
    *   *Rationale:* Provides a mathematically rigorous way to average correlation coefficients, mitigating the bias inherent in directly averaging $\rho$ values and allowing for valid 95% confidence intervals.
*   **Transition to Multi-Timestep Rollouts:** Moved from single-snapshot analysis to full episode dynamics.
    *   *Rationale:* Many specialized behaviors (like risk-triggered focus) are only observable through their temporal evolution as a scenario progresses from calm to critical.
*   **Semantic Labeling in Diagnostics:** Incorporated semantic context (e.g., vehicle distance) into diagnostic plot legends.
    *   *Rationale:* Enables direct human interpretability of which specific world entities are driving the attention patterns observed in the feature-space.
