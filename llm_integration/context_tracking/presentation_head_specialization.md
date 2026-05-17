# Presentation: Head Specialization Analysis & Wayformer Interpretability

## [SLIDE 1: Title Slide]
**Title:** Decoding the Wayformer: From Black-Box Attention to Functional Specialization
**Subtitle:** Weekly Progress Update - XAI & Interpretability
**Presenter:** [Your Name]

---

## [SLIDE 2: System Architecture - Wayformer in the RL Loop]
*   **Architecture:** Multi-modal Transformer-based encoder for geometric scene understanding.
*   **Input:** **Geometric Embeddings** (Sparse vectors for roads, agents) rather than pixel grids.
*   **Fusion Paradigm: Late Fusion via Latent Queries**
    *   **Concept:** Instead of mixing raw tokens immediately, each modality (e.g., Roadgraph, Agents) is processed independently by a dedicated Attention Block to extract high-level summaries.
    *   **Mechanism: The "Perceiver" Style Block (`WayformerAttention`)**
        1.  **Learned Latent Queries:** The model initializes a fixed set of learnable vectors (Latents) for each modality. These act as "questions" the model learns to ask about the scene (e.g., "Is there a car near me?").
        2.  **Cross-Attention (Aggregation):** These **Latents** (Queries) attend to the raw **Input Features** (Keys/Values). This compresses massive, variable-length inputs (e.g., 1000 road points) into a compact, fixed-size representation.
        3.  **Self-Attention (Reasoning):** The Latents then attend to *each other*. This allows the model to synthesize the information extracted (e.g., relating "Car A" to "Car B" purely in the latent space).
    *   **Final Output:** The processed latents from all modalities are concatenated to form the final State Embedding for the Policy.


---

## [SLIDE 3: Progress - The Offline Extraction Framework]
*   **New Workflow Implemented:** Decoupled Analysis Pipeline.
*   **Component 1: Offline Extractor (`offline_extraction.py`)**
    *   **Mechanism:** Loads training checkpoints and replays scenarios using the exact simulation environment (`vmax`).
    *   **Output:** Unified Data Structure (e.g., JSON/Dict) containing aligned Attention and Semantics.
    *   **Example:**
        ```json
        {
          "scenario_id": "s_102",
          "attention_weights": [0.05, 0.92, 0.03], // Head 0 focus
          "semantic_features": [
             {"id": 1, "ttc": 50.0}, // Safe
             {"id": 2, "ttc": 1.2},  // Dangerous -> Matches High Attention
             {"id": 3, "ttc": 99.9}
          ]
        }
        ```
*   **Why it matters:** Allows expensive, high-resolution analysis on thousands of scenarios without slowing down the active RL training loop.

---

## [SLIDE 5: Methodology - Head Specialization Index (HSI)]
*   **Analysis Logic:**
    1.  **Extract:** Get attention mass allocated to every vehicle in the scene.
    2.  **Compute Features:** Calculate semantic metrics for every vehicle (Time-to-Collision, Distance, Relative Velocity, Lane Position).
    3.  **Correlate:** Compute Pearson correlation ($\rho$) between "Attention Paid" and "Feature Intensity".
*   **Metric: HSI (Head Specialization Index)**
    *   **Definition:** The maximum absolute correlation a head has with *any* meaningful semantic feature.
    *   **Formula:** $HSI_h = \max_{f \in Features} |\rho(Attn_h, f)|$
*   **Threshold:** Heads with HSI > 0.3 are flagged as "Specialized".
*   **Analogy:** Like identifying departments in a company—if "Head 0" always lights up when bills come in, it's the "Finance Dept".

---

## [SLIDE 6: Experimental Results - Functional Roles discovered]
*   **Observation:** [INSERT: Overview of what the notebook revealed. e.g., "The model developed 3 distinct specialized heads."]
*   **Key Findings (Placeholders):**
    *   **Head X (Safety Head):** Strong correlation with `Time-to-Collision (TTC)`. [INSERT: HSI Score, e.g., 0.54]
    *   **Head Y (Proximity Head):** Strong correlation with `Distance-to-Ego`. [INSERT: HSI Score]
    *   **Head Z (Traffic Flow):** Correlates with `is_ahead` or `speed`. [INSERT: HSI Score]
*   **Interpretation:** The model is learning to decompose the driving task into modular components without explicit supervision.

Experimental Results - Visualizations

---


## [SLIDE 9: Proposed Next Steps & Immediate Focus]

*   **Forward Plan:**
    *   **Scale Up:** Train with **More Heads** (currently 2 -> target 8+) and **More Data** (Increase from 10GB val set).
    *   **Temporal Analysis:** Utilize model checkpoints to visualize *how* attention specialization evolves over time (e.g., "At step 5k, Head 0 learns proximity; at 20k, it learns velocity").
    *   **Methodology Expansion:** Investigate advanced analysis methods beyond Pearson correlation (e.g., Causal IO, Attention rollout).
