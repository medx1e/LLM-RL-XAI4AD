# Phase Overview
This development phase, titled "Sparse Autoencoder (SAE) Interpretability Integration," focuses on establishing a robust, end-to-end pipeline for understanding the internal representations of the Wayformer model. The primary objective is to extract dense, continuous embeddings from the model's residual stream and map them into a sparse, interpretable latent space where individual features correspond to distinct, human-understandable driving concepts.

# Theoretical Foundation & Design Goals
Deep neural networks often exhibit "superposition," where the number of learned concepts exceeds the available mathematical dimensions, leading to polysemantic neurons that are difficult to interpret. To address this, we leverage Sparse Autoencoders (SAEs), which act as a decompressive lens. 

The core design paradigm involves training an auxiliary SAE on the frozen activations of the target model (Wayformer). By enforcing strict sparsity penalties, the SAE is coerced to represent the dense embeddings within a significantly wider latent space, ensuring that each active dimension (feature) represents a single, isolated concept (monosemanticity). This approach allows us to directly correlate active network features with specific simulation telemetry (e.g., time-to-collision, lead vehicle behavior) to gain granular insights into the model's decision-making process prior to the reinforcement learning policy step.

# Mathematical Formulations

### Sparse Autoencoder (SAE)
Given a dense activation vector $x \in \mathbb{R}^d$ from the model's residual stream, the SAE encodes it into a sparse latent vector $f \in \mathbb{R}^F$ (where $F > d$) using standard ReLU with an L1 penalty:
$$f = \text{ReLU}\left(W_{enc}(x - b_{pre}) + b_{enc}\right)$$
The dense vector is then reconstructed as $\hat{x}$:
$$\hat{x} = W_{dec} f + b_{dec}$$
The model is trained to minimize the reconstruction mean squared error (MSE) combined with an $L_1$ sparsity penalty:
$$\mathcal{L}(x) = ||x - \hat{x}||_2^2 + \lambda ||f||_1$$
Where $\lambda$ controls the trade-off between reconstruction fidelity and feature sparsity.

### Spearman Rank Correlation ($\rho$)
To statistically map the continuous SAE feature activations to specific driving concepts without assuming a linear relationship, we use the Spearman rank correlation. For $n$ observations of a feature activation $X$ and a telemetry variable $Y$, we convert the raw values to their corresponding ranks $R(X)$ and $R(Y)$. The correlation coefficient is computed as:
$$\rho = 1 - \frac{6 \sum_{i=1}^n d_i^2}{n(n^2 - 1)}$$
Where $d_i = R(X_i) - R(Y_i)$ is the difference in paired ranks. This metric allows us to robustly identify which telemetry signals (e.g., speed, time-to-collision) each sparse feature is most sensitive to.

# Implementation Summary
The implementation successfully establishes the complete SAE lifecycle, comprising data harvesting, model definition, training, and configuration management:

*   **Activation Harvesting (`harvester.py`):** An optimized data extraction pipeline was implemented utilizing `jax.lax.scan`. This allows for the compilation of entire simulation episodes into highly efficient XLA kernels, dramatically increasing the throughput of capturing residual stream activations and complex telemetry metrics (distances, closing speeds, etc.) across multiple scenarios simultaneously.
*   **SAE Model Architecture (`sae_model.py`):** The core Sparse Autoencoder was constructed using PyTorch, adhering to established conventions (e.g., pre-encoder bias subtraction, unit-norm decoder constraints, standard ReLU with L1 regularization).
*   **Training & Tuning Pipeline (`sae_trainer.py`, `sae_tuner.py`):** A comprehensive training loop was developed, incorporating data normalization (mean-centering and standard deviation whitening) to ensure stable convergence. Furthermore, an automated hyperparameter tuning script was integrated to systematically explore different architectural configurations and sparsity penalties via grid search.
*   **Configuration Management (`config.py`, `sae_config.yaml`):** The pipeline employs a centralized YAML configuration, loaded dynamically at runtime, establishing a single source of truth for all architectural and training parameters.
*   **Statistical Feature Annotation (`feature_annotator.py`, `feature_analysis.py`):** An advanced annotation framework utilizes continuous Spearman correlation (ρ) and z-scores to statistically map latent dimensions to telemetry variables. An analysis suite generates rich visualizations (e.g., clustered heatmaps, label distributions) to interpret feature selectivity.
*   **Causal Steering (`causal_steering.py`):** Facilitates single-step interventions by injecting temperature scalars (α) into specific SAE features to measure their direct causal effect on the policy and value FC heads (e.g., Δaccel, Δsteer).
*   **Multi-Timestep Rollout Steering (`rollout_steering.py`):** Implements paired (baseline vs. steered) episode rollouts utilizing `jax.vmap` and `jax.lax.while_loop`. By capturing aggregated V-MAX metrics (e.g., TTC, Run Red Light) across multiple scenarios, this pipeline discovers and evaluates "beneficial safety concepts".

# Key Decisions & Rationale

*   **Utilization of `jax.lax.scan` for Harvesting:** 
    *   *Rationale:* Transitioning from Python-level loops to JAX-compiled scans for scenario rollouts significantly reduces I/O bottlenecks and context-switching overhead. This architectural choice maximizes hardware utilization during the data collection phase, allowing for the rapid generation of the large datasets required for effective SAE training.
*   **Deferral of `JumpReLU` Activation:** 
    *   *Rationale:* While `JumpReLU` was initially considered to address feature shrinkage issues, its implementation was deferred. The current pipeline reliably utilizes standard ReLU activations coupled with L1 regularization, which proved sufficiently stable for generating a healthy sparsity ratio during the initial data harvesting and interpretability phases.
*   **Data Whitening (Mean & Std Normalization):** 
    *   *Rationale:* Standardizing the harvested activations before SAE encoding ensures that all latent dimensions contribute equally to the learning process. This addresses potential issues where a high-magnitude baseline or dominant dimensions might mask subtle, underlying features, leading to more robust and distinct concept discovery.
*   **Automated Grid Search Tuning:** 
    *   *Rationale:* Identifying the optimal balance between reconstruction accuracy and feature sparsity is highly empirical. Providing an automated tuner allows for the systematic evaluation of critical hyperparameters (such as the L1 coefficient and expansion factor), ensuring the final model achieves the desired feature isolation without manual trial and error.
*   **Centralized YAML Configuration:** 
    *   *Rationale:* Decoupling configuration from code execution ensures consistency across the harvesting, training, and tuning stages. It simplifies experimentation and guarantees that all pipeline components operate under identical assumptions.
*   **Spearman Correlation over Spatial Telemetry:**
    *   *Rationale:* Transitioning to continuous Spearman correlation for feature annotation (instead of relying solely on thresholded spatial data) captures graded relationships across all timesteps, yielding far more robust and accurate monosemantic labels for subtle driving concepts.
*   **JAX Vectorization (`jax.vmap`) for Rollout Steering:**
    *   *Rationale:* Running multi-timestep causal interventions requires parallel evaluation of multiple temperature (α) values. Utilizing `jax.vmap` compresses all temperature interventions into a single compiled XLA program, massively accelerating the computational throughput compared to sequential host-level loops.
