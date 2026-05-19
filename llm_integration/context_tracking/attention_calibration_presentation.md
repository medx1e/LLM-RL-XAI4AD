# Attention Calibration to Criticality — Weekly Research Presentation

---

## [SLIDE 1: RESEARCH OBJECTIVE]

**Title:** Attention Calibration to Criticality in Autonomous Driving Models

**Research Question:**
- Do attention mechanisms in autonomous driving models concentrate more on critical (threatening) vehicles?
- **Calibration** (Definition): The degree to which a model's attention distribution correlates with the criticality of driving situations
- Well-calibrated model = Higher attention concentration when situations are more dangerous

- Critical for interpretability: Does the model "look at" what matters for safety?
- Architecture comparison: Do different encoder designs (LQ/Perceiver vs Wayformer) exhibit different calibration behaviors?



## [SLIDE 2: CONCEPTS CLARIFICATION]

**Term:** Criticality Score

**Definition:** A composite metric quantifying the threat level posed by each vehicle in a driving scene. Computed per-vehicle at each timestep

**Computation:**
```
Criticality = w₁·TTC + w₂·ClosingSpeed + w₃·Distance
```

**Components:**
- **TTC (Time-to-Collision)**
- **Closing Speed**
- **Distance**

**Weights Used:** 50% TTC, 30% Closing Speed, 20% Distance 

---

**Term:** Attention Concentration

**Definition:** Measures how focused (vs. uniform) an attention distribution is across entities

**Three Metrics Implemented:**

1. **Gini Coefficient**: Measures inequality in attention distribution
   - Example: Gini = 0.85 → 85% of attention mass is concentrated on a subset of vehicles

2. **Entropy-based Concentration**: Measure of uncertainty

3. **Top-3 Mass**: Fraction of total attention on top-3 attended vehicles
   - Example: Top-3 = 0.92 → 92% of attention goes to 3 vehicles out of 40

---

## [SLIDE 3: TECHNICAL IMPLEMENTATION — EXTRACTION PIPELINE]

**Title:** Attention & Criticality Extraction

**Architecture Support:**

**LQ/Perceiver Encoder**: 
- 16 Latent Queries compress variable-length observations via cross and self-attention.
- Unified processing of all scene entities (vehicles, roadgraph, traffic lights) through global cross-attention.

**Extraction Process:**
1. Load trained checkpoint and configuration
2. Initialize encoder with `return_attention_weights=True`
3. For each scenario:
   - Extract observation from simulator state
   - Forward pass through encoder → capture attention weights
   - Compute criticality scores for all vehicles
   - Compute concentration metrics (Gini, Entropy, Top-3) per head
   - Store paired data: (criticality, concentration, attention_per_vehicle)

**Output Format:**
```python
{
  'semantic_features': {'criticality': array(n_vehicles)},
  'concentration_suite': {'gini': array(n_heads), 'entropy': array(n_heads), 'top3_mass': array(n_heads)},
  'attention_per_vehicle': array(n_heads, n_vehicles),
}
```

**Dataset Scale:** 100 scenarios extracted from WOMD validation set

---

## [SLIDE 5: TECHNICAL IMPLEMENTATION — CALIBRATION ANALYSIS]

**Title:** Correlation-Based Calibration Metrics

**Analysis Levels:**

1. **Scene-Level Calibration**
   - Does the model concentrate attention more in dangerous scenes?
   - Scatter plot: 

2. **Vehicle-Level Calibration**
   - Does the model attend more to critical vehicles?
   - Scatter plot: Vehicle criticality → Attention mass


---

## [SLIDE 6: RESULTS & OBSERVED PATTERNS]

**Title:** Key Observations from Calibration Analysis

**1. Non-Linear Calibration:**
- **Spearman ρ (0.289) >> Pearson r (0.080)**
- The model exhibits monotonic but non-linear attention-criticality mapping
- Implication: Attention increases with criticality, but not proportionally

**2. Moderate Vehicle-Level Calibration:**

- The model successfully reallocates attention mass toward critical vehicles within each scene
- Scene-level concentration remains stable (ρ = 0.041) → Model maintains consistent scanning strategy
- **Key insight:** Rather than globally "panicking" in dangerous scenes, the model performs targeted spatial reallocation of attention to threats

**3. Head Specialization Potential:**
- Head 1 (Gini = 0.80) consistently more concentrated than Head 0 (Gini = 0.73)

**4. Dataset Imbalance:**
- 95% of scenarios are low-to-medium criticality
- Limits ability to detect calibration effects in safety-critical situations

---
