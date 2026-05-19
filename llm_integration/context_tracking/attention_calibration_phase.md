# Attention Calibration to Criticality — Development Phase Summary

**Date:** February 2026  
**Phase:** XAI Research Direction — Attention Calibration Analysis  
**Status:** Implementation Complete, Ready for Large-Scale Evaluation

---

## 1. Phase Overview

### Phase Name
**Attention Calibration to Criticality in Autonomous Driving Models**

### Primary Objective
Develop a comprehensive analytical framework to quantify and visualize how attention mechanisms in autonomous driving models correlate with driving criticality. The goal is to determine whether attention-based encoders (specifically LQ/Perceiver architecture) demonstrate safety-aware attention allocation by concentrating focus on critical (threatening) vehicles in driving scenarios.

### Functional Outcome
A complete pipeline enabling:
- Extraction of paired attention weights and criticality scores from trained models
- Multi-metric calibration analysis with statistical rigor
- Publication-ready visualizations for interpretability research
- Cross-architecture comparison capability (LQ vs Wayformer)

---

## 2. Theoretical Foundation & Design Goals

### Core Problem Statement
Attention mechanisms are widely adopted in autonomous driving perception, yet their safety-awareness remains poorly understood. Specifically:
- Do models naturally learn to attend more to dangerous vehicles?
- Is attention concentration correlated with scene criticality?
- Does calibration behavior differ across encoder architectures?

**Calibration** is defined as the degree to which a model's attention distribution correlates with the criticality (threat level) of driving situations. A well-calibrated model exhibits higher attention concentration when encountering critical scenarios.

### Key Principles & Design Paradigms

**1. Multi-Level Analysis**
- **Scene-level calibration**: Aggregate metrics across the entire driving scene
- **Vehicle-level calibration**: Granular per-entity attention allocation
- **Head-level analysis**: Per-attention-head specialization patterns

**2. Statistical Rigor**
- Correlation analysis (Pearson for linear, Spearman for monotonic relationships)
- Partial correlation to control for confounding factors (scene complexity)
- Bootstrap confidence intervals (95% CI, 1000 samples) for significance testing
- Regime breakdown analysis (low vs. high criticality)

**3. Metric Robustness**
To avoid metric-specific artifacts, three complementary concentration metrics were selected:
- **Gini Coefficient**: Measures inequality in attention distribution
- **Entropy-based Concentration**: Information-theoretic measure of uncertainty
- **Top-K Mass**: Fraction of attention on top-K attended entities

**4. Architecture Generality**
The framework was designed to support multiple encoder architectures:
- **LQ/Perceiver**: Cross-attention with learnable latent queries
- **Wayformer**: Entity-specific attention modules (planned comparison)

**Rationale for Design Choices:**
- Multi-metric approach ensures findings are not artifacts of a single concentration measure
- Scene-level and vehicle-level analysis provide complementary perspectives (global vs. local calibration)
- Statistical rigor (bootstrapping, partial correlation) addresses potential confounds and ensures reproducibility
- Architecture generality enables comparative analysis to determine if calibration is architecture-dependent

### Intended System Behavior
The system should:
1. Load trained checkpoints and extract attention weights alongside criticality scores
2. Compute multiple concentration metrics per attention head
3. Perform correlation analysis at both scene and vehicle levels
4. Generate publication-ready visualizations with statistical annotations
5. Support batch processing for large-scale evaluation (500+ scenarios)

---

## 3. Implementation Summary

### Technical Overview
The implementation consists of three main components:
1. **Offline Extraction Pipeline**: Captures attention weights and computes criticality scores
2. **Calibration Analysis Framework**: Statistical correlation analysis with multiple metrics
3. **Visualization Suite**: Publication-ready plots with professional styling

### Main Components

#### **Component 1: Attention & Criticality Extraction** (`utils/attention_extraction.py`)

**Purpose:** Extract paired attention weights and criticality scores from trained models offline.

**Key Features:**
- Dynamic architecture detection (LQ/Perceiver vs. Wayformer)
- Parameter loading with automatic key remapping for checkpoint compatibility
- Criticality computation based on TTC (Time-to-Collision), closing speed, and distance
- Multi-metric concentration computation (Gini, Entropy, Top-3 Mass) per attention head
- Token boundary tracking for extracting vehicle attention from concatenated inputs

**Output Format:**
```python
{
  'semantic_features': {'criticality': array(n_vehicles)},
  'concentration_suite': {
    'gini': array(n_heads),
    'entropy': array(n_heads),
    'top3_mass': array(n_heads)
  },
  'attention_per_vehicle': array(n_heads, n_vehicles)
}
```

**Architecture Support:**
- **LQ/Perceiver**: 16 latent queries with 2 cross-attention heads and 2 self-attention heads. Cross-attention heads attend to all scene entities (vehicles, roadgraph, traffic lights) via concatenated tokens. The extraction process identifies vehicle token boundaries and extracts the corresponding attention slices.

#### **Component 2: Calibration Analysis** (`analysis/calibration_analysis.py`)

**Purpose:** Compute calibration metrics quantifying attention-criticality correlation.

**Analysis Levels:**

1. **Scene-Level Calibration**
   - Aggregates scene criticality as max across all vehicles
   - Aggregates concentration as mean across attention heads
   - Computes Pearson (linear) and Spearman (monotonic) correlations

2. **Vehicle-Level Calibration**
   - Pairs per-vehicle criticality with normalized attention mass
   - Aggregates 750+ vehicle observations across all scenarios
   - Reveals whether the model attends more to critical vehicles

3. **Partial Correlation**
   - Controls for scene complexity (number of valid vehicles)
   - Isolates the effect of criticality on concentration

4. **Regime Breakdown**
   - Partitions scenarios into low (≤0.3), medium, and high (≥0.7) criticality
   - T-tests to determine if concentration differs significantly between regimes

**Per-Head Analysis:**
- Computes Spearman correlation for each attention head with criticality
- Identifies potential head specialization (e.g., "safety head" vs. "navigation head")

#### **Component 3: Visualization Suite** (`analysis/calibration_visualization.py`)

**Purpose:** Generate publication-ready figures for interpretability research.

**Implemented Plot Types:**

1. **Multi-Metric Scatter (3-panel)**: Scene-level concentration vs. criticality for all three metrics
2. **Scene + Vehicle Scatter (2-panel)**: Scene-level and vehicle-level calibration side-by-side
3. **Time-Series Evolution**: Criticality and concentration across scenarios sorted by criticality
4. **Regime Box Plots**: Concentration distributions across low/medium/high criticality regimes
5. **Per-Head Heatmap**: Head × Metric correlation matrix

**Styling:**
- Professional color palette (distinct colors per metric and architecture)
- Statistical annotations (Spearman ρ, significance markers: *, **, ***)
- 95% confidence interval error bars
- Publication-ready resolution (300 DPI)

### Significant Integrations

**LQ Encoder Parameter Handling:**
The LQ/Perceiver architecture uses a nested parameter structure where encoder parameters are stored within `encoder_layer` rather than at the top level. The extraction pipeline implements:
- Dynamic parameter path detection (`encoder_layer`, `lq_encoder`, or `encoder`)
- Parameter key remapping (`perceiver_attention` → `lq_attention`) for compatibility with newer model definitions
- Attention weight extraction with automatic architecture detection

**Concentration Metrics Module** (`analysis/concentration_metrics.py`):
A reusable module implementing:
- Gini coefficient for attention inequality
- Normalized entropy for uncertainty quantification
- Top-K mass for focal attention measurement
- Per-head aggregation for multi-head attention models

---

## 4. Key Decisions & Rationale

### Decision 1: Multi-Metric Concentration Analysis

**Choice:** Implement three concentration metrics (Gini, Entropy, Top-3 Mass) instead of a single metric.

**Justification:**
- **Robustness:** Ensures findings are not artifacts of metric-specific biases
- **Complementarity:** Each metric captures different aspects of attention distribution
  - Gini: Overall inequality
  - Entropy: Information-theoretic uncertainty
  - Top-3: Focal concentration on dominant entities
- **Industry Standard:** Multi-metric analysis is common in XAI research to validate findings across different mathematical formulations

### Decision 2: Vehicle-Level in Addition to Scene-Level Analysis

**Choice:** Analyze calibration at both scene-level (aggregate) and vehicle-level (granular).

**Justification:**
- **Granularity Trade-off:** Scene-level reveals global trends, vehicle-level reveals local allocation patterns
- **Complementary Insights:** A model might exhibit weak scene-level correlation (no global modulation) but strong vehicle-level correlation (spatial reallocation)
- **Statistical Power:** Vehicle-level analysis aggregates 750+ observations vs. 100 scene-level observations, improving significance detection
- **Future Application:** Vehicle-level calibration could inform attention regularization losses during training

### Decision 3: Spearman Correlation as Primary Metric

**Choice:** Use Spearman (rank-based) correlation as the primary calibration metric, alongside Pearson.

**Justification:**
- **Non-linearity:** Attention-criticality relationships may be monotonic but non-proportional (e.g., saturating attention)
- **Robustness to Outliers:** Rank-based correlation is less sensitive to extreme values
- **Interpretability:** Monotonic relationships are often more interpretable than linear assumptions in real-world scenarios
- **Best Practice:** Spearman is standard in XAI and attention analysis literature

### Decision 4: Bootstrap Confidence Intervals

**Choice:** Implement bootstrap resampling (1000 samples) for 95% confidence intervals.

**Justification:**
- **Non-parametric:** Does not assume normal distribution of correlations
- **Reproducibility:** Provides uncertainty bounds for scientific rigor
- **Comparative Analysis:** Enables statistically sound comparison across architectures
- **Industry Standard:** Bootstrap is widely accepted in ML research for small-to-medium sample sizes

### Decision 5: Criticality via DRAC (Deceleration Rate to Avoid a Crash)

**Choice:** Define criticality using DRAC: `Criticality = min(1, DRAC / DRAC_max)` where `DRAC = closing_speed² / (2 × distance)` and `DRAC_max = 8.0 m/s²`.

**Previous approach (superseded):** Weighted linear combination `0.5·TTC + 0.3·ClosingSpeed + 0.2·Distance` was mathematically redundant because TTC ≈ Distance / ClosingSpeed under constant velocity, making the three variables interdependent.

**Justification for DRAC:**
- **No redundancy:** Single physically-grounded quantity combining speed and distance without double-counting
- **Physical meaning:** DRAC represents the minimum deceleration required to avoid a collision (m/s²)
- **Interpretable thresholds:** 3.4 m/s² = comfortable braking, 6.0 = hard braking, 8.0 = emergency limit
- **Literature support:** Widely used in traffic conflict analysis (Archer, 2005; Hydén, 1987; Almqvist et al., 1991)
- **Computational efficiency:** Single division operation, no weight tuning needed

### Decision 6: Offline Extraction Over Online Logging

**Choice:** Implement offline extraction (post-training) rather than modifying training loops.

**Justification:**
- **Non-invasive:** Does not require changes to training infrastructure
- **Flexibility:** Enables analysis of any trained checkpoint without retraining
- **Reproducibility:** Extraction is deterministic and can be repeated on the same scenarios
- **Performance:** Offline extraction is faster than online logging during training (no I/O overhead)

### Decision 7: Regime Breakdown with Fixed Thresholds

**Choice:** Partition criticality into regimes using fixed thresholds (low ≤ 0.3, high ≥ 0.7).

**Justification:**
- **Interpretability:** Fixed thresholds (30th and 70th percentiles) are intuitive
- **Comparability:** Enables consistent regime definitions across different datasets
- **Statistical Testing:** Allows for t-tests to determine if concentration differs significantly between regimes
- **Future Work:** Thresholds can be adjusted based on domain expertise or empirical validation

### Decision 8: Per-Head Correlation Analysis

**Choice:** Compute Spearman correlation for each attention head separately.

**Justification:**
- **Specialization Detection:** Identifies if specific heads are more calibrated to criticality
- **Architectural Insight:** Reveals whether multi-head attention learns functional specialization (e.g., "safety head" vs. "navigation head")
- **Future Application:** Could inform head-specific regularization or pruning strategies
- **Extensibility:** Lays groundwork for attention head interpretability research

### Decision 9: Publication-Ready Visualization Suite

**Choice:** Invest in professional matplotlib styling with custom color palettes and statistical annotations.

**Justification:**
- **Research Communication:** High-quality figures are essential for thesis, papers, and presentations
- **Reproducibility:** Automated visualization pipeline ensures consistency across experiments
- **Time Efficiency:** Templates reduce time spent on figure generation for future analyses
- **Team Consistency:** Establishes a visual standard for the research group

---

## Current Status & Next Steps

### Completed Deliverables
1. ✅ Extraction pipeline supporting LQ/Perceiver architecture
2. ✅ Calibration analysis framework with multi-metric support
3. ✅ Visualization suite (5 plot types)
4. ✅ Presentation content summarizing findings
5. ✅ Initial evaluation on 100 scenarios

### Key Findings (Preliminary, 100 Scenarios)
- **Vehicle-Level Calibration:** Spearman ρ = 0.289 (p < 1e-15) — statistically significant
- **Scene-Level Calibration:** Spearman ρ = 0.041 (n.s.) — negligible correlation
- **Interpretation:** The model performs spatial reallocation of attention to critical vehicles rather than global concentration modulation

### Planned Extensions
1. **Scale to 500+ scenarios** for improved statistical power
2. **Wayformer comparison** to determine if calibration is architecture-dependent
3. **Per-head specialization analysis** using the implemented heatmap visualization
4. **Alternative criticality formulations** (e.g., multiplicative vs. additive)

---

## Technical Artifacts

### Code Modules
- `utils/attention_extraction.py` (967 lines): Extraction pipeline
- `analysis/concentration_metrics.py` (159 lines): Metric implementations
- `analysis/calibration_analysis.py` (647 lines): Statistical analysis
- `analysis/calibration_visualization.py` (800+ lines): Visualization suite

### Dataset
- 100 scenarios extracted from WOMD validation set
- Paired (attention, criticality) data with metadata
- Expandable to 500+ scenarios via batch processing

### Documentation
- `presentations/attention_calibration_presentation.md`: 12-slide research presentation
- `analysis/LQ_vs_Wayformer_Architecture.md`: Architecture comparison reference

---

## Summary

This phase successfully established a rigorous framework for evaluating attention calibration in autonomous driving models. The implementation prioritizes statistical rigor, metric robustness, and extensibility to support both current analysis and future comparative studies across architectures. The key innovation is the multi-level, multi-metric approach that reveals nuanced calibration behavior: the LQ/Perceiver model demonstrates moderate vehicle-level calibration through spatial attention reallocation, rather than global concentration modulation. This insight has implications for both model interpretability and potential training improvements (e.g., criticality-weighted losses).
