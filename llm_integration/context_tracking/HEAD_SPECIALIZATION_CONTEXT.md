# Context File: Head Specialization Analysis (XAI)

## 1. Phase Overview
**Phase Name:** Attention Interpretability & Head Specialization  
**Primary Objective:** To develop a robust, offline framework for discovering and quantifying the functional specialization of attention heads in multi-agent trajectory prediction models (Wayformer). The goal is to move from "black-box" attention to interpretable "Safety," "Proximity," and "Traffic Flow" functional labels.

## 2. Theoretical Foundation & Design Goals
The core challenge in Transformer-based autonomous driving agents is understanding *what* the model prioritizes during critical maneuvers. This phase is guided by the **Head Specialization Index (HSI)** paradigm:
- **Correlation-Based Interpretation:** Instead of manual inspection, we use Pearson correlation between attention distributions and grounded semantic features (e.g., Time-to-Collision, Distance-to-Ego).
- **Offline Decoupling:** Analysis is separated from training to allow for high-resolution extraction across multiple scenarios without impacting training performance or memory.
- **Unified Extraction:** Combining attention weights and semantic features into a single data structure ensures temporal and spatial alignment for accurate correlation.

## 3. Implementation Summary
A comprehensive suite of tools was built to support the XAI workflow:
- **Unified Offline Extractor (`offline_extraction.py`):** A high-performance script that samples training scenarios, initializes the simulation state, and extracts synchronized attention weights and semantic ground-truth.
- **Head Specialization Analyzer (`head_specialization_analysis.py`):** A library that implements the HSI scoring algorithm, functional labeling logic (mapping features to human-readable roles), and a JSON-based registry for model documentation.
- **Visual Analytics Suite:** Integrated tools for generating correlation heatmaps, HSI distribution bar charts, and feature-attention scatter plots for fine-grained validation.
- **Workflow Orchestration (`head_specialization_analysis.ipynb`):** A unified Jupyter interface that guides users through extraction, analysis, and discovery of head evolution during training.

## 4. Key Decisions & Rationale
- **Direct Integration with WayformerEncoder:** We chose to use the production `WayformerEncoder` for extraction rather than a proxy model. This ensures that the attention weights analyzed are identical to those used for real-time inference.
- **Environment-Aware Feature Extraction:** Semantic features (like TTC and SDC-relative coordinates) are computed by initializing the full simulation environment. This provides more accurate "ground truth" for correlation than raw observation vectors.
- **Unified Pickle Serialization:** Storing aggregated attention (per-vehicle) alongside semantic features in a single file reduces data management overhead and prevents misalignment between disparate log sources.
- **HSI Thresholding (0.3):** A standardized threshold was established to distinguish "Specialized" heads from "General Context" heads, providing a consistent metric for comparing different model versions or architectures.
- **Dynamic Token Boundary Detection:** The extractor dynamically computes entity ranges (SDC, Vehicles, Roadgraph) from the training configuration, ensuring the analysis tool is robust to changes in observation settings (e.g., increasing the number of tracked agents).
