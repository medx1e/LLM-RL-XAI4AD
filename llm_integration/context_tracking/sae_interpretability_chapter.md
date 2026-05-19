# Sparse Autoencoder-Based Mechanistic Interpretability for Transformer Policies in Autonomous Driving

**Research Context Document — PhD Thesis Chapter Foundation**

> This document constitutes the theoretical and methodological knowledge base for a formal thesis chapter. It does not report experimental results. Its purpose is to articulate the scientific rationale, theoretical grounding, and methodological logic of a four-stage mechanistic interpretability pipeline applied to a transformer-based autonomous driving policy trained via reinforcement learning.

---

## 1. Introduction

### 1.1 The Opacity Problem in Deep Reinforcement Learning Policies

The deployment of deep reinforcement learning (RL) agents in safety-critical domains such as autonomous driving presents a fundamental epistemic challenge: the learned policy is encoded as a high-dimensional, distributed, and largely uninterpretable function. Transformer-based policy architectures, in particular, have demonstrated strong empirical performance across structured prediction, language modeling, and sequential decision-making tasks. Yet the mechanisms by which such architectures process environmental observations and produce control decisions remain deeply obscure. This opacity is not merely an inconvenience; it constitutes a structural barrier to safety verification, failure mode analysis, and the development of trustworthy autonomous systems.

The dominant paradigm for inspecting the internal computations of transformer models has historically been attention visualization — the examination of attention weight distributions across input tokens or scene elements. While attention analysis provides a window into the relational structure of information routing, it does not reveal the semantic content encoded in the latent representations that flow through and between attention layers. Attention weights describe *where* the model looks, not *what* the model knows or *why* it acts. More critically, attention-based interpretability methods are fundamentally correlational: they identify co-occurrence patterns between input features and attended positions, but they do not support causal claims about the functional role of any particular internal representation.

This chapter is motivated by the recognition that a complete mechanistic account of a transformer policy requires moving beyond the attention layer and into the representation space of the residual stream — the principal substrate through which information accumulates and transforms across the depth of the network. The residual stream, as conceptualized in the mechanistic interpretability literature, is the locus of all information that is integrated, revised, and ultimately projected into the policy and value heads that produce behavior. Understanding what is encoded there, and how those encodings causally determine action, is the central scientific objective of this chapter.

### 1.2 From Observational to Intervention-Based Interpretability

The field of mechanistic interpretability has advanced from the identification of circuits and components via observational correlation toward a more rigorous standard: intervention-based causal evidence. In the context of large language models, this shift has been operationalized through techniques such as activation patching, causal tracing, and representation engineering. These methods share a common logic: rather than asking which internal representations *correlate* with a behavioral output, they ask what happens to behavior when representations are *deliberately modified*. The answer to the latter question constitutes mechanistic evidence of a qualitatively different and stronger kind.

This chapter adopts an analogous philosophy in the context of transformer-based autonomous driving policies. The interpretability framework developed here comprises four interdependent stages:

1. **Representation Harvesting**: Large-scale extraction of latent activations from the transformer encoder, co-registered with rich driving telemetry across diverse scenarios.
2. **Sparse Autoencoder Training**: Decomposition of dense residual stream representations into an overcomplete sparse basis, with the goal of revealing latent monosemantic features.
3. **Semantic Feature Annotation**: Statistical alignment of learned sparse features with environment telemetry signals, producing interpretable labels for abstract latent dimensions.
4. **Causal Feature Steering and Intervention**: Direct manipulation of feature activations in the residual stream, followed by measurement of downstream behavioral change, to establish causal attribution.

Together, these stages form a pipeline that moves from passive observation of activations to active intervention on the causal structure of the policy's internal computation. This transition — from correlation to causation, from observation to intervention — is the central conceptual contribution of the interpretability framework described in this chapter.

### 1.3 Scope and Contribution

The autonomous driving setting provides a uniquely demanding test bed for mechanistic interpretability. Unlike language model tasks, where behavioral outputs are discrete tokens and evaluations can be performed on static datasets, autonomous driving involves a closed-loop sequential decision process in which interventions on internal representations propagate through time and compound across a trajectory. A steering perturbation applied at one timestep alters the agent's state, which alters its future observations, which in turn modulate its future internal representations. This temporal dynamics amplifies both the interpretability signal and the interpretability challenge.

The contribution of this chapter is a principled, end-to-end mechanistic interpretability framework that takes seriously the sequential and closed-loop nature of the autonomous driving problem, and that applies sparse dictionary learning as a principled bridge between opaque dense latent spaces and human-interpretable semantic concepts.

---

## 2. Theoretical Background

### 2.1 Transformer Latent Representations and the Residual Stream

A transformer-based policy processes environmental observations through a sequence of alternating self-attention and feedforward sublayers. Each sublayer contributes an additive update to a shared residual stream, which persists across the depth of the network. Formally, if $\mathbf{h}^{(\ell)} \in \mathbb{R}^D$ denotes the residual stream vector at layer $\ell$, the transformer dynamics can be written as:

$$
\mathbf{h}^{(\ell+1)} = \mathbf{h}^{(\ell)} + \Delta^{(\ell)}_{\text{attn}}(\mathbf{h}^{(\ell)}) + \Delta^{(\ell)}_{\text{ff}}(\mathbf{h}^{(\ell)})
$$

where $\Delta^{(\ell)}_{\text{attn}}$ and $\Delta^{(\ell)}_{\text{ff}}$ denote the additive contributions of the attention and feedforward sublayers, respectively. The final residual stream $\mathbf{h}^{(L)} \in \mathbb{R}^D$ — often referred to as the encoder output or the bottleneck representation — is subsequently projected by task-specific linear heads to produce policy logits or value estimates.

This residual stream formulation has a crucial implication for interpretability: the representation at any depth $\ell$ can be understood as a superposition of contributions from all preceding sublayers. The final encoder output therefore integrates information accumulated across the entire depth of the network. If any sublayer has learned to encode a semantically meaningful concept — such as proximity to a lead vehicle, scene density, or lateral acceleration — that concept should in principle be recoverable from the residual stream, because residual connections preserve additive structure.

The central challenge is that the dimension $D$ of the residual stream, while finite, is a compressed bottleneck that must simultaneously represent all driving-relevant information required for accurate value estimation and policy execution. This compression pressure leads to a fundamental problem: the simultaneous encoding of multiple conceptually distinct signals within a small number of latent dimensions.

### 2.2 The Superposition Hypothesis and Polysemanticity

The *superposition hypothesis*, articulated formally in the mechanistic interpretability literature, proposes that neural networks systematically encode more features than they have dimensions by placing feature directions at nearly-orthogonal angles in high-dimensional space. Under this hypothesis, a single neuron — or, more generally, a single latent dimension — participates in the encoding of multiple distinct semantic concepts simultaneously, a property known as *polysemanticity*. Polysemanticity arises as a representational efficiency strategy: because the feature directions corresponding to different concepts are statistically sparse (they rarely co-activate), their linear superposition in a shared low-dimensional subspace produces only minimal destructive interference.

Formally, consider a set of $n$ binary features $\{f_1, \ldots, f_n\}$ with $n \gg D$. Superposition encodes these features as a set of direction vectors $\{\mathbf{d}_1, \ldots, \mathbf{d}_n\} \subset \mathbb{R}^D$, where the activation of feature $f_k$ contributes an additive increment $f_k \mathbf{d}_k$ to the residual stream:

$$
\mathbf{h} = \sum_{k=1}^{n} f_k \mathbf{d}_k + \boldsymbol{\epsilon}
$$

where $\boldsymbol{\epsilon}$ represents noise from simultaneous feature activations that interfere with one another. The signal-to-noise ratio of feature recovery is controlled by the sparsity of the feature activation pattern: if at most $s$ features are simultaneously active, and the directions $\mathbf{d}_k$ are drawn from a random polytope configuration, the reconstruction error scales as $O(s/D)$ rather than $O(n/D)$, making superposition tractable when $s \ll n$.

The implication for interpretability is significant: individual neurons or linear probes applied to the residual stream will generally fail to isolate single semantic concepts because those concepts are encoded as superpositions. Recovering the true latent feature structure requires a decomposition method that explicitly models and enforces sparsity.

### 2.3 Sparse Coding and Dictionary Learning

The mathematical framework of sparse coding, developed in the signal processing and computational neuroscience literature, provides the theoretical foundation for decomposing superposed representations. In sparse coding, a signal $\mathbf{x} \in \mathbb{R}^D$ is modeled as a sparse linear combination of dictionary atoms $\{\mathbf{d}_j\}_{j=1}^{F}$ drawn from an overcomplete dictionary $\mathbf{D} \in \mathbb{R}^{D \times F}$ with $F > D$:

$$
\mathbf{x} \approx \sum_{j=1}^{F} f_j \mathbf{d}_j = \mathbf{D}\mathbf{f}, \quad \|\mathbf{f}\|_0 \ll F
$$

The pursuit of sparse representations in an overcomplete basis provides a mechanism by which the true structure of superposed activations can be disentangled. If each dictionary atom $\mathbf{d}_j$ corresponds to a single semantic concept, and if the coding coefficients $\mathbf{f}$ are genuinely sparse, then the decomposition reveals which concepts are active in any given observation by examining which coefficients are non-zero.

The classical sparse coding problem minimizes a reconstruction penalty subject to a cardinality constraint on the representation:

$$
\min_{\mathbf{f}} \|\mathbf{x} - \mathbf{D}\mathbf{f}\|_2^2 \quad \text{s.t.} \quad \|\mathbf{f}\|_0 \leq s
$$

Because the $\ell_0$ norm is combinatorially intractable, a standard convex relaxation replaces it with the $\ell_1$ norm, yielding the Lasso formulation:

$$
\min_{\mathbf{f}} \|\mathbf{x} - \mathbf{D}\mathbf{f}\|_2^2 + \lambda \|\mathbf{f}\|_1
$$

The $\ell_1$ penalty promotes sparsity by inducing many coefficients to collapse to exactly zero, while penalizing large but non-zero coefficients proportionally to their magnitude.

### 2.4 Sparse Autoencoders for Mechanistic Interpretability

Sparse autoencoders (SAEs) operationalize the sparse coding principle within a learnable end-to-end framework. An SAE consists of an encoder mapping $\mathbf{x} \mapsto \mathbf{f}$ and a decoder mapping $\mathbf{f} \mapsto \hat{\mathbf{x}}$, trained jointly to minimize a combination of reconstruction fidelity and sparsity:

$$
\mathcal{L}(\mathbf{x}) = \underbrace{\|\mathbf{x} - \hat{\mathbf{x}}\|_2^2}_{\text{reconstruction}} + \underbrace{\lambda \cdot \Omega(\mathbf{f})}_{\text{sparsity}}
$$

where $\Omega(\mathbf{f})$ is a sparsity-inducing regularizer. In the formulation adopted in this work, following the conventions established by Anthropic's interpretability research program, the encoder applies a learned affine transformation followed by a rectified linear unit:

$$
\mathbf{f} = \text{ReLU}\!\left((\mathbf{x} - \mathbf{b}_{\text{pre}}) \mathbf{W}_{\text{enc}} + \mathbf{b}_{\text{enc}}\right), \quad \mathbf{W}_{\text{enc}} \in \mathbb{R}^{D \times F}
$$

The pre-encoder bias $\mathbf{b}_{\text{pre}} \in \mathbb{R}^D$ functions as a learned centroid that re-centers the residual stream distribution before the encoding projection, enabling the encoder to operate in a zero-mean coordinate frame aligned with the principal structure of the activation manifold.

The decoder applies a learned linear transformation with no activation function:

$$
\hat{\mathbf{x}} = \mathbf{f} \mathbf{W}_{\text{dec}} + \mathbf{b}_{\text{dec}}, \quad \mathbf{W}_{\text{dec}} \in \mathbb{R}^{F \times D}
$$

A critical architectural constraint is imposed on the decoder weight matrix: each row $\mathbf{w}_j^{\text{dec}} \in \mathbb{R}^D$ is constrained to unit norm:

$$
\|\mathbf{w}_j^{\text{dec}}\|_2 = 1 \quad \forall j \in \{1, \ldots, F\}
$$

This constraint enforces that each decoder row defines a direction in residual stream space, rather than simultaneously encoding both magnitude and direction. The magnitude of a feature's contribution is thus entirely governed by the scalar activation coefficient $f_j$, while the direction is fixed as the unit vector $\mathbf{w}_j^{\text{dec}}$. This factorization is essential for interpretability: it ensures that the latent dimension $j$ is associated with a well-defined, stable feature direction in the residual stream.

The sparsity term in the training objective is the $\ell_1$ norm of the feature activation vector, averaged over the batch:

$$
\Omega(\mathbf{f}) = \frac{1}{N} \sum_{i=1}^{N} \|\mathbf{f}_i\|_1
$$

Since $\mathbf{f} \geq 0$ after ReLU, the $\ell_1$ norm reduces to $\|\mathbf{f}_i\|_1 = \sum_{j=1}^{F} f_{ij}$ — the total activation budget per observation. Critically, the sparsity penalty is computed as the per-sample $\ell_1$ norm averaged over the batch, rather than a diluted element-wise mean over the full $N \times F$ matrix. This formulation ensures that the coefficient $\lambda$ controls the total allowed activation per observation, making its interpretation invariant to the expansion factor $F/D$.

The combined training objective is therefore:

$$
\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|_2^2 + \lambda \cdot \frac{1}{N}\sum_{i=1}^{N} \|\mathbf{f}_i\|_1
$$

### 2.5 Monosemanticity and Disentangled Representations

The goal of SAE training is to induce *monosemanticity* in the latent features: a regime in which each latent dimension $j$ corresponds to a single, coherent semantic concept, and activates selectively in response to that concept rather than responding promiscuously to multiple unrelated stimuli. Monosemanticity is the representational dual of polysemanticity: where a polysemantic neuron encodes multiple concepts, a monosemantic SAE feature encodes exactly one.

The relationship between sparsity and monosemanticity is not guaranteed — it is a theoretical expectation grounded in the following argument: if features activate rarely but consistently (i.e., they are sparse yet non-random), the decoder direction $\mathbf{w}_j^{\text{dec}}$ associated with feature $j$ will be shaped during training to point in the residual stream direction most predictive of the reconstruction target when $f_j > 0$. If the residual stream direction associated with, say, high ego-speed is geometrically stable across timesteps (which it should be, given the network's weight-tying across time), then the SAE will learn to associate a dedicated feature with that direction. The $\ell_1$ penalty then acts as selection pressure against the feature co-activating with other concepts, since doing so would increase the sparsity cost.

This argument is grounded in the broader theory of *disentangled representations*, which holds that a representation is disentangled to the degree that each latent dimension is sensitive to changes in at most one generative factor of variation in the data. Sparse autoencoders provide a computationally tractable mechanism for inducing representational disentanglement without requiring explicit supervision on the identity of the generative factors.

### 2.6 The Expansion Factor and Overcomplete Dictionaries

The expansion factor $\gamma = F/D$ — the ratio of SAE latent dimensionality to residual stream dimensionality — governs the degree to which the dictionary is overcomplete. An expansion factor of unity ($\gamma = 1$) recovers a square, non-overcomplete autoencoder, while larger values allow the SAE to discover progressively more latent features than the residual stream has dimensions.

The choice of expansion factor represents a fundamental trade-off. A higher $\gamma$ increases the expressivity of the dictionary, enabling the SAE to represent a richer and more fine-grained set of semantic concepts. However, it also increases the risk of feature collapse — where multiple dictionary atoms converge to represent the same concept — and of dead features — dictionary atoms that never activate across the training dataset. A higher expansion factor also imposes greater computational demands during training and inference.

In the framework described here, the expansion factor $\gamma$ is treated as a hyperparameter and searched over a discrete grid (see Section 5.3). Empirical precedents in the SAE literature applied to large language models suggest that expansion factors of 8–32 produce a favorable balance between concept coverage and feature quality. In the autonomous driving domain, the conceptual vocabulary required to describe driving behavior — speed regimes, proximity to surrounding agents, safety-critical events, lateral dynamics — is rich but not unbounded, suggesting that expansion values of $\mathcal{O}(10)$ are a principled starting point for the search.

### 2.7 Sparsity Metrics

Two complementary sparsity metrics are employed to characterize the quality of learned representations:

**$\ell_0$ sparsity** measures the average number of active (non-zero) features per observation:

$$
L_0 = \frac{1}{N} \sum_{i=1}^{N} \|\mathbf{f}_i\|_0 = \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{F} \mathbf{1}[f_{ij} > 0]
$$

A low $L_0$ value indicates that only a small fraction of features activate for any given observation, consistent with the monosemanticity hypothesis. As a reference: in a perfectly sparse regime, $L_0 \ll F$; in a fully dense regime, $L_0 = F$.

**Dead feature percentage** measures the fraction of dictionary atoms that never activate across a representative sample of observations:

$$
\text{Dead\%} = \frac{1}{F} \sum_{j=1}^{F} \mathbf{1}\!\left[\max_{i} f_{ij} = 0\right] \times 100
$$

Dead features represent wasted representational capacity and indicate that the effective dictionary size is smaller than the nominal expansion factor would suggest. Excessive dead features are associated with suboptimal training dynamics, often arising from a mismatch between the sparsity coefficient $\lambda$ and the intrinsic dimensionality of the activation distribution.

---

## 3. Mechanistic Interpretability Methodology

The interpretability framework described in this chapter is organized as a four-stage pipeline, each stage building on the outputs of its predecessor. This section describes each stage at the level of scientific methodology, grounding each design decision in theoretical principles established in Section 2.

### 3.1 Representation Harvesting

#### 3.1.1 Purpose and Scientific Rationale

The first stage of the pipeline is the large-scale extraction of latent activations from the trained transformer policy. This harvesting procedure serves two distinct purposes: first, it provides the training corpus for the sparse autoencoder; second, it establishes a co-registered dataset of latent representations and environmental telemetry, which will later be used for statistical annotation of learned features.

The scientific importance of this stage cannot be understated. The quality of the SAE's learned features is fundamentally bounded by the distributional coverage of the harvested activation dataset. If the dataset is biased toward a narrow subset of driving scenarios — for instance, highway cruise control in low-density environments — the SAE may fail to discover features associated with rare but safety-critical events such as hard braking or dense intersection navigation. The harvesting procedure must therefore be designed to capture the full behavioral and environmental diversity of the target driving distribution.

#### 3.1.2 Trajectory-Level Collection and Temporal Continuity

Activations are harvested at the trajectory level rather than the observation level. For each scenario in the dataset, the policy is unrolled in a closed-loop replay of the recorded environment, and the encoder output — the residual stream vector $\mathbf{h}^{(L)} \in \mathbb{R}^D$ — is extracted at every timestep $t = 0, 1, \ldots, T-1$. Each extracted vector is indexed by its scenario identifier and within-episode timestep, preserving the temporal structure of the episode.

Temporal indexing is essential for two reasons. First, it enables the alignment of latent representations with time-varying telemetry signals: quantities such as time-to-collision (TTC), lateral acceleration, and scene density are naturally defined as functions of time within a trajectory, and their correlation with sparse features can only be computed if both the activations and the telemetry values are synchronized to the same temporal index. Second, temporal indexing enables the study of temporal autocorrelation in feature activations — a property relevant to understanding how driving concepts evolve over the course of an episode.

The encoder is run in a log-replay mode: the policy acts as a passive observer of the recorded trajectory, producing internal representations in response to the unfolding scenario without altering the environmental dynamics. This ensures that the harvested activations are drawn from the same behavioral distribution as the trained policy, rather than from a counterfactual distribution induced by deviations between the policy's actions and the logged expert behavior.

#### 3.1.3 Telemetry Synchronization and Driving Signal Design

At every harvested timestep, a rich vector of environmental telemetry is computed and co-registered with the corresponding latent activation. The design of this telemetry is a methodological choice of significant scientific consequence: the telemetry defines the vocabulary of interpretable concepts against which sparse features will later be evaluated. Telemetry fields that are semantically impoverished, redundant, or spuriously correlated with latent features will degrade the quality of the annotation stage.

The telemetry is organized into two broad categories:

**Continuous scalar signals** characterize the dynamical and relational state of the ego vehicle and surrounding agents:

- $v_{\text{ego}}$: ego vehicle speed (m/s) — captures longitudinal driving regime.
- $a_{\text{lat}}$: lateral acceleration (m/s²) — captures cornering intensity, derived from the product of ego speed and absolute yaw rate.
- $\dot{\psi}$: yaw rate (rad/step) — captures rate of heading change, sensitive to turning maneuvers.
- $\text{TTC}_{\min}$: minimum time-to-collision across all valid surrounding agents — a classical safety metric defined as $\text{TTC} = d / v_{\text{closing}}$ for agents with positive closing speed.
- $d_{\min}$: minimum distance to any valid agent — a spatial proximity measure sensitive to dense traffic scenarios.
- $n_{\text{agents}}$: number of valid surrounding agents — scene density indicator.
- $d_{\text{lead}}$: distance to the lead vehicle (the closest valid agent in the forward direction) — captures car-following behavior.
- $v_{\text{close,lead}}$: closing speed to the lead vehicle — captures the urgency of lead vehicle following.

**Discrete event indicators** capture safety-relevant behavioral states:

- $\mathbf{1}_{\text{TTC-crit}}$: binary flag indicating $\text{TTC}_{\min} < \tau_{\text{crit}}$, where $\tau_{\text{crit}} = 1.5\,\text{s}$ is a safety-critical threshold consistent with established automotive safety standards.
- $\mathbf{1}_{\text{hard-brake}}$: binary flag indicating that the lead vehicle has decelerated by more than $0.4g \cdot \Delta t$ in a single timestep, where $g = 9.81\,\text{m/s}^2$ and $\Delta t = 0.1\,\text{s}$ is the simulation time step.
- $\mathbf{1}_{\text{following}}$: binary flag indicating active car-following behavior (lead vehicle within 30 m).
- $\mathbf{1}_{\text{left/right}}$: binary flags indicating the presence of valid agents in the left or right lateral zones.

Notably, raw map coordinates (absolute ego position in the global frame) are excluded from the telemetry. This exclusion is scientifically motivated: absolute position is not a driving concept — it reflects the specific geographical layout of a scenario rather than a behavioral principle — and SAE features that correlate with raw position would absorb representational capacity without contributing to interpretable behavioral understanding. The exclusion prevents position-correlated noise from dominating the annotation stage.

#### 3.1.4 Per-Agent Statistics and Spatial Geometry

Beyond scalar summaries, per-agent arrays are computed for each timestep, capturing the full relational structure of the traffic scene. These include per-agent distances, TTC values, speeds, closing speeds, and spatial flags indicating whether each agent is ahead, to the left, or to the right of the ego vehicle in its local reference frame.

Agent spatial classification is performed in the ego-local coordinate frame: an agent at global position $(x_a, y_a)$ relative to ego position $(x_e, y_e)$ with ego heading $\psi$ is transformed to local coordinates:

$$
\begin{pmatrix} x_{\text{local}} \\ y_{\text{local}} \end{pmatrix} = \begin{pmatrix} \cos(-\psi) & -\sin(-\psi) \\ \sin(-\psi) & \cos(-\psi) \end{pmatrix} \begin{pmatrix} x_a - x_e \\ y_a - y_e \end{pmatrix}
$$

Agents with $x_{\text{local}} > 2.0\,\text{m}$ are classified as ahead; those with $y_{\text{local}} > 1.0\,\text{m}$ are classified as to the left; those with $y_{\text{local}} < -1.0\,\text{m}$ are to the right. This ego-centric geometric classification ensures that spatial relationships are defined in terms of the agent's own frame of reference, which is the natural coordinate system for driving decisions.

Time-to-collision is computed using the standard kinematic formula:

$$
\text{TTC}_{a} = \begin{cases} \frac{d_a}{v_{\text{close},a}} & \text{if } v_{\text{close},a} > 0.5\,\text{m/s} \\ T_{\text{horizon}} & \text{otherwise} \end{cases}
$$

where $d_a$ denotes the distance to agent $a$, $v_{\text{close},a} = -(d\mathbf{x}_a \cdot \mathbf{v}_{\text{rel}}) / d_a$ is the closing speed, and $T_{\text{horizon}}$ is a saturation constant applied in the non-closing case to prevent numerical instability. The minimum TTC across all valid agents provides a global safety margin estimate.

#### 3.1.5 Computational Efficiency and Scale

The harvesting procedure is designed to operate at scale across hundreds of scenarios. A compiled execution strategy leverages XLA-based functional scan primitives to eliminate per-timestep Python overhead: the entire episode dynamics are compiled into a single accelerator kernel, and all per-timestep outputs are accumulated in pre-allocated arrays and transferred to the host in a single bulk operation after episode completion. This design ensures that harvesting throughput scales linearly with the number of scenarios rather than being bottlenecked by host-device synchronization overhead.

The resulting dataset is stored in a hierarchical format with chunked storage for efficient random-access loading during SAE training. Dataset cardinality scales proportionally to the number of scenarios and the average episode length; at 500 scenarios with an average of approximately 60 effective timesteps, this yields on the order of 30,000 activation vectors, each of dimension $D$. This constitutes a modest but statistically adequate corpus for SAE training, given the relatively low intrinsic dimensionality of the driving feature space.

---

### 3.2 Sparse Autoencoder Training

This section provides a detailed account of the SAE training methodology, including architectural specification, normalization strategy, optimization procedure, and diagnostics.

#### 3.2.1 Activation Normalization

Prior to training, all harvested activations are normalized using empirical statistics computed over the full dataset:

$$
\tilde{\mathbf{x}} = \frac{\mathbf{x} - \boldsymbol{\mu}}{\boldsymbol{\sigma} + \epsilon}
$$

where $\boldsymbol{\mu} = \frac{1}{N}\sum_{i=1}^N \mathbf{x}_i$ is the empirical mean vector, $\boldsymbol{\sigma} = \sqrt{\frac{1}{N}\sum_{i=1}^N (\mathbf{x}_i - \boldsymbol{\mu})^2}$ is the empirical standard deviation vector (computed element-wise), and $\epsilon = 10^{-8}$ is a numerical stability constant.

This normalization serves multiple scientific purposes. First, it removes systematic offsets in the residual stream that arise from the learned policy's internal bias structure, rather than from variance in the input observations. Second, it ensures that all dimensions of the residual stream contribute approximately equally to the reconstruction objective, preventing dimensions with large variance from dominating the loss. Third, and most importantly, the normalization parameters $(\boldsymbol{\mu}, \boldsymbol{\sigma})$ must be stored alongside the trained SAE weights and applied consistently during annotation and steering — this is non-negotiable for scientific reproducibility, since any inconsistency between training-time and inference-time normalization would corrupt the feature activations.

The pre-encoder bias $\mathbf{b}_{\text{pre}}$ provides an additional, learned correction to the residual stream distribution within the SAE's own encoding function. This learned centering operates in the post-normalization space and provides a fine-grained adjustment beyond the crude mean subtraction achieved by the normalization layer. The combination of normalization and learned pre-bias provides a flexible, two-stage centering mechanism.

#### 3.2.2 Architecture Specification

The SAE architecture follows the formulation described in Section 2.4. The key architectural parameters are:

- **Input dimension**: $D$ (the Wayformer encoder's hidden dimension, fixed by the pre-trained policy architecture).
- **Latent dimension**: $F = \gamma \cdot D$, where $\gamma = 16$ is the expansion factor.
- **Encoder**: affine transformation followed by ReLU, producing non-negative feature activations.
- **Decoder**: affine transformation with unit-norm row constraint.
- **Sparsity coefficient**: $\lambda = 10^{-3}$, controlling the trade-off between reconstruction fidelity and feature sparsity.

Encoder weights $\mathbf{W}_{\text{enc}} \in \mathbb{R}^{D \times F}$ are initialized using Kaiming uniform initialization, which preserves signal variance through the ReLU nonlinearity. Decoder weights $\mathbf{W}_{\text{dec}} \in \mathbb{R}^{F \times D}$ are initialized as the transpose of the encoder weights and immediately normalized to unit row norms, establishing an approximately dual relationship between the encoder and decoder at initialization.

This initialization strategy is scientifically motivated: initializing the decoder as the transpose of the encoder ensures that, before any training, a feature's encoding direction and its decoding direction are aligned. This reduces the initial gradient variance and accelerates the emergence of coherent feature directions in the early training epochs.

#### 3.2.3 Optimization and Training Dynamics

The SAE is trained using the Adam optimizer with a cosine annealing learning rate schedule. The choice of Adam is motivated by its adaptivity to the curvature of the loss landscape: the SAE training objective exhibits heterogeneous curvature across encoder and decoder parameters, and adaptive learning rates prevent oscillation in flat directions while enabling rapid descent in steep directions.

The cosine annealing schedule implements a smooth decay from the initial learning rate $\eta_0 = 3 \times 10^{-4}$ to zero over the course of training:

$$
\eta_t = \frac{\eta_0}{2}\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)
$$

where $t$ is the current epoch and $T$ is the total number of epochs. This schedule provides a large effective learning rate during the early training phase (when features are still poorly aligned with semantic directions) and progressively reduces the step size as features approach stable configurations, reducing the risk of oscillation around feature directions late in training.

Gradient clipping is applied at each optimization step with a maximum gradient norm of 1.0:

$$
\mathbf{g} \leftarrow \mathbf{g} \cdot \min\!\left(1, \frac{1.0}{\|\mathbf{g}\|_2}\right)
$$

Gradient clipping prevents the large sparse updates that can occur when a previously inactive feature direction begins receiving gradient signal from a cluster of observations, and stabilizes the early stages of dictionary formation.

After each gradient step, the decoder weight matrix is re-normalized to enforce the unit-norm constraint. This normalization step is applied as a projected gradient update rather than as a regularization term, ensuring exact satisfaction of the constraint rather than approximate satisfaction via penalization.

#### 3.2.4 The Feature Collapse and Dead Feature Problems

Two pathological failure modes can arise during SAE training:

**Feature collapse** occurs when multiple dictionary atoms converge to represent the same semantic direction, effectively reducing the useful dictionary size below $F$. Feature collapse is a manifestation of the non-convexity of the SAE training objective: because the encoder and decoder are jointly optimized, the training dynamics can become trapped in configurations where a small subset of features monopolizes the reconstruction signal while the majority of features remain uninformative.

Feature collapse is mitigated by the unit-norm decoder constraint, which prevents individual features from growing unboundedly in magnitude and thereby dominating the reconstruction. However, the constraint alone does not prevent multiple features from converging to the same direction. Additional mitigation strategies — such as feature re-initialization when collapse is detected — are not employed in the present work, and the degree to which collapse occurs is expected to be visible in the annotations, where multiple high-activation features will share the same semantic label.

**Dead features** are dictionary atoms that fail to activate for any observation in the training corpus. A feature $j$ is considered dead if:

$$
\sum_{i=1}^{N} \mathbf{1}[f_{ij} > 0] = 0
$$

Dead features represent wasted representational capacity. Their prevalence is tracked throughout training via the dead feature percentage metric, evaluated on a fixed reference sample of up to 8,192 observations. Dead features arise when the encoder's weight initialization places a feature direction far from any activation cluster in the normalized residual stream, and the gradient of the sparsity penalty prevents the encoder from moving toward those clusters.

The sparsity coefficient $\lambda$ is the primary control for dead feature prevalence: a higher $\lambda$ increases the penalty for any activation and can push marginally active features into the dead regime. The value $\lambda = 10^{-3}$ is chosen to balance sparsity promotion against feature coverage, but the optimal value is dataset-dependent and constitutes a hyperparameter requiring empirical tuning.

#### 3.2.5 Loss Decomposition and Training Monitoring

The combined training loss $\mathcal{L}$ is decomposed into three interpretable components that are monitored independently throughout training:

**Reconstruction loss** (MSE):

$$
\mathcal{L}_{\text{recon}} = \frac{1}{N} \sum_{i=1}^{N} \|\tilde{\mathbf{x}}_i - \hat{\mathbf{x}}_i\|_2^2
$$

A declining reconstruction loss indicates that the decoder is learning to accurately reconstruct the residual stream from sparse feature activations. If the reconstruction loss plateaus early, it may indicate that the sparsity coefficient $\lambda$ is too large, preventing the encoder from activating enough features to reconstruct the input faithfully.

**$\ell_1$ sparsity** (per-element mean $\ell_1$ norm, tracked as a diagnostic scalar):

$$
\ell_1^{\text{diag}} = \frac{1}{NF} \sum_{i=1}^{N} \|\mathbf{f}_i\|_1
$$

This monitoring quantity is related to the training-objective sparsity term by $\ell_1^{\text{diag}} = \frac{1}{F}\,\Omega(\mathbf{f})$, and is therefore independent of both batch size and expansion factor, making it a scale-invariant proxy for sparsity pressure. $\ell_1^{\text{diag}}$ should decrease during training as the encoder becomes more selective, preferring sparser activations for any given observation.

**$L_0$ sparsity** (mean number of active features):

$$
L_0 = \frac{1}{N} \sum_{i=1}^{N} \|\mathbf{f}_i\|_0
$$

Unlike $\mathcal{L}_{\text{sparse}}$, $L_0$ is not differentiable and cannot be directly minimized. It serves as a diagnostic metric: a well-trained SAE should achieve $L_0 \ll F$, with typical values in the range of 5–50 active features per observation for an expansion factor of 16 applied to a $D$-dimensional residual stream.

The training trajectory of these metrics provides insight into the dynamics of dictionary formation. In a well-behaved training run, the reconstruction loss should decrease monotonically, the $\ell_1$ loss should decrease as features become more selective, and the dead feature percentage should stabilize at a value below 50%, indicating that more than half of the dictionary is genuinely utilized.

#### 3.2.6 Checkpoint Selection

Model selection is performed by tracking the combined training loss $\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda \mathcal{L}_{\text{sparse}}$ across all epochs and saving the checkpoint that achieves the minimum value. Because the combined loss reflects the balance between reconstruction fidelity and sparsity, this selection criterion rewards models that are simultaneously accurate and sparse — which is precisely the regime most conducive to monosemantic feature discovery.

The checkpoint stores not only the model parameters ($\mathbf{W}_{\text{enc}}$, $\mathbf{b}_{\text{enc}}$, $\mathbf{W}_{\text{dec}}$, $\mathbf{b}_{\text{dec}}$, $\mathbf{b}_{\text{pre}}$) but also the empirical normalization statistics ($\boldsymbol{\mu}$, $\boldsymbol{\sigma}$), enabling exact reproduction of the training-time normalization in all downstream stages.

---

### 3.3 Semantic Feature Annotation

#### 3.3.1 The Annotation Problem

Once the SAE has been trained, its $F$ latent dimensions represent learned feature directions in the residual stream. The key scientific question of the annotation stage is: *to what behavioral concept does each feature direction correspond?* This question is non-trivial because the SAE is trained in a fully unsupervised manner — no semantic labels are provided to the training procedure — and the learned features may correspond to arbitrary linear combinations of the environmental telemetry dimensions, to concepts that have no simple telemetry correlate, or to representational artifacts arising from training dynamics.

The annotation procedure operationalizes a statistical approach to interpretability: it attempts to assign each learned feature a semantic label by identifying the environmental telemetry variable that best explains the feature's activation pattern across the harvested dataset. This approach assumes that if a feature fires selectively in response to a specific driving concept, its activations will exhibit a systematic statistical relationship with the telemetry variable that measures that concept.

#### 3.3.2 Spearman Rank Correlation as a Semantics-Alignment Metric

The primary annotation method employed in this framework is Spearman rank correlation — a non-parametric measure of statistical dependence based on the ranks of the observations rather than their raw values. For a feature $j$ with activation vector $\mathbf{f}_j = (f_{1j}, \ldots, f_{Nj}) \in \mathbb{R}^N$ and a telemetry field $k$ with value vector $\mathbf{t}_k = (t_{1k}, \ldots, t_{Nk}) \in \mathbb{R}^N$, the Spearman correlation coefficient is:

$$
\rho_{jk} = 1 - \frac{6 \sum_{i=1}^N \left(\text{rank}(f_{ij}) - \text{rank}(t_{ik})\right)^2}{N(N^2 - 1)}
$$

or equivalently, it is the Pearson correlation of the rank-transformed vectors.

Spearman correlation is preferred over Pearson correlation for this task for several reasons:

1. **Non-Gaussianity of feature activations**: SAE feature activations are heavily right-skewed (most observations have zero or near-zero activation, with a long tail of high activations in semantically relevant contexts). Pearson correlation assumes Gaussian marginals; Spearman does not.
2. **Invariance to monotone nonlinearities**: Spearman $\rho$ is invariant under any monotone transformation of either variable. This means that a feature whose activation is a saturating function of ego-speed will yield the same $\rho$ with ego-speed as a feature whose activation is a linear function, provided the rank order is preserved.
3. **Robustness to outliers**: Because ranks are bounded, extreme activation values in rare scenarios cannot inflate the correlation estimate arbitrarily.
4. **Graceful handling of sparse activations**: For a very sparse feature (activating at $k \ll N$ timesteps), the $N - k$ zero activations all receive a tied rank of approximately $N/2$. The covariance between the feature's ranked activations and the telemetry's ranked activations is then driven entirely by the $k$ active timesteps. A feature that fires only 5 times but always during hard-braking events will correctly yield $\rho \approx 1.0$ with the hard-braking indicator, because the hard-braking timesteps will have high ranks in both the feature activation and the binary event vectors.

The **selectivity score** of a feature is defined as the maximum absolute Spearman correlation across all telemetry fields:

$$
s_j = \max_{k} |\rho_{jk}|
$$

and its **semantic label** is the telemetry field achieving this maximum, annotated with the sign of the correlation:

$$
\ell_j = [\text{sign}(\rho_{j,k^*})] \cdot \text{telemetry\_field}_{k^*}, \quad k^* = \arg\max_k |\rho_{jk}|
$$

A feature with $\rho_{j,k^*} > 0$ fires more strongly when the corresponding telemetry variable is high; a feature with $\rho_{j,k^*} < 0$ fires more strongly when the telemetry variable is low. Both cases are informative.

#### 3.3.3 Z-Score Analysis as a Complementary Method

An alternative annotation strategy, applicable when Spearman correlation is insufficient (e.g., for boolean telemetry fields with very rare positive events), is z-score analysis of top-activating timesteps. For feature $j$, the $K$ timesteps at which the feature activation $f_{ij}$ is largest are identified, and the telemetry value at those timesteps is compared to the global telemetry distribution:

$$
z_{jk} = \frac{\bar{t}_{k,\text{top-}K} - \mu_{t_k}}{\sigma_{t_k}}
$$

where $\bar{t}_{k,\text{top-}K}$ is the mean value of telemetry field $k$ at the top-$K$ activation timesteps, and $\mu_{t_k}$, $\sigma_{t_k}$ are the global mean and standard deviation. A large positive z-score indicates that the feature fires preferentially in high-$t_k$ environments; a large negative z-score indicates preferential firing in low-$t_k$ environments.

The z-score method has complementary strengths to Spearman correlation: it is sensitive to the absolute magnitude of the feature at its peak activations (since it explicitly selects the top-$K$ timesteps), whereas Spearman uses the full activation distribution. For binary telemetry fields such as hard braking events, which may have a prevalence of only 1–5% in the dataset, the z-score method can detect features that fire strongly during the rare events even when the global Spearman correlation is diluted by the preponderance of non-event timesteps.

In practice, both metrics are computed and stored for each feature, but the Spearman correlation is used as the primary annotation mode, with z-scores available for secondary analysis and for features that do not meet the minimum activation threshold for reliable Spearman estimation.

#### 3.3.4 Feature Activity Threshold and Dead Feature Classification

Features with fewer than a minimum number of non-zero activations (nominally $n_{\min} = 10$ across the full harvested dataset) are classified as dead and excluded from semantic annotation. This threshold is motivated by statistical considerations: for $n < 10$, the Spearman rank statistic is unreliable because the sample contains too few non-zero values to form a meaningful rank distribution. Dead features are recorded in the annotation output but carry no semantic label and are excluded from all downstream analysis.

The fraction of dead features is a key diagnostic for assessing SAE quality. A dead feature percentage above approximately 50% suggests that the expansion factor is too large relative to the complexity of the activation distribution, or that the sparsity coefficient $\lambda$ is too aggressive.

#### 3.3.5 Constant Telemetry Field Guard

A subtlety in the annotation procedure arises from the possibility that certain telemetry fields are constant or near-constant across the entire harvested dataset. This can occur for boolean event indicators (such as hard braking) in datasets dominated by routine driving scenarios with few safety-critical events. If the telemetry field has near-zero variance, Spearman correlation cannot be computed meaningfully, since rank correlation between a constant vector and any other vector is undefined (resulting in a numerical NaN). All telemetry fields with empirical standard deviation below a threshold of $10^{-9}$ are excluded from correlation computation and receive a nominal correlation of zero.

#### 3.3.6 Per-Agent Feature Reduction

Per-agent telemetry arrays (distance, TTC, speed, closing speed) are reduced to scalar summary statistics — minimum, mean, and maximum over the valid agent set — before being used in Spearman correlation. This reduction is applied using agent-validity masking: only slots corresponding to valid (present) agents contribute to the reduction, with invalid (absent) agent slots replaced by appropriate sentinel values to prevent systematic bias. Without validity masking, the zero-padding of absent agents would introduce constant offsets in the minimum and mean statistics, producing spurious correlations between features and scene density.

#### 3.3.7 Semantic Consistency and Interpretability Limits

The annotation procedure provides a systematic, statistically grounded method for assigning semantic labels to sparse features. However, it is important to recognize the fundamental limitations of correlation-based interpretability.

**Spurious alignment** can arise when a feature that genuinely encodes one concept (e.g., high closing speed to the lead vehicle) is annotated with a different but statistically associated concept (e.g., low lead vehicle distance) due to the high empirical correlation between these two quantities in typical driving datasets. The annotation assigns the label of the maximally correlated telemetry field, but this maximum may not uniquely identify the feature's true causal encoding.

**Polysemantic residual features** may still exist after SAE decomposition, particularly for low-selectivity features whose maximum $|\rho|$ is below 0.3. Such features encode multiple overlapping concepts and cannot be meaningfully annotated with a single telemetry label.

**Temporal ambiguity** arises because certain concepts are temporally correlated by the dynamics of driving scenarios: an agent that is currently performing hard braking was likely also at short distance in the previous timestep. Features that activate at hard-braking events will therefore exhibit some correlation with lead vehicle distance even if they truly encode the braking event rather than the distance. Distinguishing these cases requires temporal lag analysis beyond simple cross-sectional correlation.

Despite these limitations, the annotation stage provides a necessary and scientifically defensible first approximation to the semantic content of sparse features, sufficient to motivate the causal intervention experiments described in the following section.

---

### 3.4 Causal Feature Steering and Intervention

#### 3.4.1 The Limitation of Correlational Interpretability

The annotation stage establishes that certain sparse features are statistically associated with driving-relevant telemetry signals. A feature annotated as encoding ego speed because its activation correlates strongly ($\rho = 0.82$) with the ego vehicle's instantaneous velocity is interpretively suggestive but not causally conclusive. The correlation may arise because:

(a) The feature genuinely encodes ego speed and causally influences the policy's value estimate and action distribution.
(b) The feature encodes a confounded concept that co-varies with ego speed in the dataset (e.g., highway scenarios, which systematically co-occur with high ego speed and also with other distinct policy behaviors).
(c) The feature is an epiphenomenon — present in the residual stream but not read out by the policy or value head, and therefore causally irrelevant to behavior.

Distinguishing between these possibilities requires intervention, not observation. The causal steering stage addresses this by directly modifying feature activations in the residual stream and measuring the resulting change in the policy's output, thereby establishing a causal chain from the feature to behavior.

#### 3.4.2 The Intervention Protocol

The causal intervention protocol operates on the normalized residual stream. Given an observation $\mathbf{x}$ processed by the trained encoder to produce hidden state $\mathbf{h} \in \mathbb{R}^D$, the SAE encodes $\mathbf{h}$ to produce feature activations:

$$
\mathbf{f} = \text{ReLU}\!\left(\frac{\mathbf{h} - \boldsymbol{\mu}}{\boldsymbol{\sigma}} - \mathbf{b}_{\text{pre}}\right)\mathbf{W}_{\text{enc}} + \mathbf{b}_{\text{enc}}
$$

To intervene on feature $j$ with steering intensity $\alpha \in \mathbb{R}$, the feature activation is additively modified:

$$
\mathbf{f}'_j = f_j + \alpha, \quad \mathbf{f}'_k = f_k \; \text{for} \; k \neq j
$$

The steered feature representation is then decoded back to residual stream space:

$$
\hat{\mathbf{h}}_{\text{steered}} = \mathbf{f}' \mathbf{W}_{\text{dec}} + \mathbf{b}_{\text{dec}}
$$

The intervention induces a delta in the residual stream space:

$$
\Delta\mathbf{h} = \left(\hat{\mathbf{h}}_{\text{steered}} - \hat{\mathbf{h}}_{\text{baseline}}\right) \cdot \boldsymbol{\sigma}
$$

where the multiplication by $\boldsymbol{\sigma}$ re-scales the delta from the normalized coordinate frame back to the original residual stream coordinate frame. The modified residual stream vector is:

$$
\mathbf{h}_{\text{final}} = \mathbf{h} + \Delta\mathbf{h}
$$

This additive delta formulation has a principled interpretation: the intervention modifies the residual stream by injecting or suppressing the information associated with feature $j$, without altering the baseline information encoded by all other features. The decoder direction $\mathbf{w}_j^{\text{dec}}$ determines the precise direction in residual stream space along which the delta is applied, scaled by the magnitude $\alpha \cdot \|\Delta\mathbf{h}/\alpha\|$.

The modified residual stream $\mathbf{h}_{\text{final}}$ is then passed through the policy and value heads to compute the steered behavioral outputs:

$$
(\hat{a}_{\text{accel}}, \hat{a}_{\text{steer}}) = \pi(\mathbf{h}_{\text{final}})
$$

$$
\hat{v}_{\text{steered}} = V(\mathbf{h}_{\text{final}})
$$

The behavioral effect of the intervention is quantified as the shift from baseline:

$$
\Delta a_{\text{accel}} = \hat{a}_{\text{accel}}(\mathbf{h}_{\text{final}}) - \hat{a}_{\text{accel}}(\mathbf{h})
$$

$$
\Delta a_{\text{steer}} = \hat{a}_{\text{steer}}(\mathbf{h}_{\text{final}}) - \hat{a}_{\text{steer}}(\mathbf{h})
$$

$$
\Delta v = \hat{v}(\mathbf{h}_{\text{final}}) - \hat{v}(\mathbf{h})
$$

#### 3.4.3 Causal Validity and Intervention Strength

The causal interpretability of the steering experiment rests on several assumptions that must be carefully examined.

**Assumption 1: Linear readout of features**. The intervention protocol assumes that the policy and value heads read out information from the residual stream approximately linearly — that is, that a change of $\alpha \cdot \mathbf{w}_j^{\text{dec}}$ in the residual stream produces a change in the head output that is approximately proportional to $\alpha$. This is a reasonable assumption for the fully connected layers that typically constitute the policy and value heads, though nonlinear activations within those heads introduce higher-order terms.

**Assumption 2: Feature direction isolation**. The additive delta formulation assumes that the decoder direction $\mathbf{w}_j^{\text{dec}}$ is specific to feature $j$ and does not simultaneously activate other features' decoder directions. Because the decoder weight rows are constrained to unit norm but not to orthogonality, this assumption holds approximately, with inter-feature interference proportional to the cosine similarity between decoder rows.

**Assumption 3: Distribution shift bound**. The intervention moves the residual stream vector off the manifold of naturally occurring activations by an amount proportional to $\alpha$. For small $\alpha$, the steered $\mathbf{h}_{\text{final}}$ remains close to the natural activation manifold and the causal inference is reliable. For large $\alpha$, the steered vector may enter out-of-distribution regions of the activation space, where the behavior of the policy and value heads is undefined and the interpretation of $\Delta v$ and $\Delta a$ becomes unreliable.

**Assumption 4: No interventional confounding**. The additive intervention modifies only the feature $j$ component of the residual stream, leaving all other components unchanged. However, if feature $j$'s decoder direction has significant overlap with the effective input directions of other features, the intervention may indirectly activate or suppress those features. This constitutes a form of interventional confounding — the observed behavioral change may partially reflect indirect effects of the intervention on other features, rather than the direct causal effect of feature $j$ alone.

#### 3.4.4 Temperature Schedule and Monotonicity Analysis

The steering experiment is conducted across a calibrated range of intervention intensities (temperatures) $\alpha \in \{\alpha_1, \ldots, \alpha_K\}$, typically spanning negative and positive values of increasing magnitude, such as $\{-5, -2, -1, +1, +2, +5\}$. This bidirectional temperature schedule enables several important analyses:

**Monotonicity testing**: A feature that genuinely encodes a causal concept should produce behavioral effects that scale monotonically with $\alpha$. A feature encoding high ego speed, for instance, should produce increasing positive $\Delta a_{\text{accel}}$ as $\alpha$ increases from large negative to large positive values, reflecting the policy's tendency to decelerate when it perceives low speed and accelerate when it perceives high speed.

**Sign consistency**: A genuine causal feature should produce consistent sign of $\Delta v$ across scenarios for the same $\alpha$. The *sign flip fraction*, defined as:

$$
\text{SFF}(\alpha) = \frac{\min\!\left(|\{i : \Delta v_i > 0\}|, |\{i : \Delta v_i < 0\}|\right)}{N_{\text{scenarios}}}
$$

quantifies the degree of disagreement across scenarios. A feature with low SFF and high mean $|\Delta v|$ is a strong causal candidate; a feature with high SFF may encode a concept whose causal role reverses depending on the scenario context.

**Intervention norm consistency**: The $\ell_2$ norm of the residual stream delta $\|\Delta\mathbf{h}\|_2 = |\alpha| \cdot \|\mathbf{w}_j^{\text{dec}}\|_2 \cdot \|\boldsymbol{\sigma}\|_2$ should be non-zero and scale proportionally to $|\alpha|$. A zero delta norm would indicate that the decoder direction $\mathbf{w}_j^{\text{dec}}$ lies in the null space of the effective intervention, which would be a pathological failure of the steering protocol.

#### 3.4.5 Closed-Loop Rollout Intervention

The causal intervention modality employed throughout this study is the closed-loop rollout: the feature modification is sustained across the entire episode, with the modified action executed in the simulated environment at every timestep and the resulting state transition feeding the next observation. This design evaluates how the causal influence of a single feature *compounds through time*, producing qualitatively different episode-level outcomes compared to the unperturbed baseline.

Because the environment state diverges from the baseline trajectory after the first modified action, the causal effect of the intervention at timestep $t$ cannot be cleanly separated from the environmental consequences of modified actions at timesteps $0, 1, \ldots, t-1$. The measured difference in episode-level metrics therefore reflects the *total causal effect* of sustained feature modification — an integral of direct and indirect effects propagating through the closed-loop dynamics — rather than the direct effect of a single isolated perturbation. This is a deliberate methodological choice: the compound causal effect is the quantity of practical relevance for safety-critical autonomous driving, where any persistent internal bias in the policy would similarly propagate and amplify over a full trajectory.

#### 3.4.6 Paired Rollout Comparison and Metric Suite

The closed-loop intervention is evaluated using a paired rollout design: for each scenario and each temperature $\alpha$, a baseline rollout and a steered rollout are initialized from identical initial states and run to episode completion or a maximum step count. The behavioral divergence between the two rollouts is quantified across a comprehensive suite of driving metrics:

- **At-fault collision rate**: binary flag indicating whether the ego vehicle has caused a collision, aggregated as the maximum over the episode (an episode-level binary outcome).
- **Time-to-collision metric**: episodic mean TTC across valid agent-pairs, reflecting sustained safety margin.
- **Comfort metric**: a measure of longitudinal and lateral jerk, reflecting ride smoothness and driving style quality.
- **Speed limit compliance**: fraction of timesteps at which the ego velocity exceeds the local speed limit.
- **Lane discipline**: fraction of timesteps at which the ego vehicle straddles multiple lanes.
- **Driving direction compliance**: fraction of timesteps at which the ego vehicle travels in the correct direction relative to the road topology.
- **Progress ratio**: the fraction of the planned route completed by episode termination, reflecting efficiency.

The aggregate behavioral fingerprint of a feature is the vector of $\Delta$-metric values (steered minus baseline) across all metrics and scenarios. A feature whose steering produces large, consistent, and semantically coherent changes in this metric vector — e.g., consistently degrading TTC and increasing collision rate when amplified — provides strong causal evidence for a safety-relevant encoding.

Cross-scenario aggregation yields for each metric $m$ and temperature $\alpha$ the distribution statistics $\{\mu_m(\alpha), \sigma_m(\alpha), \text{median}_m(\alpha), \text{p5}_m(\alpha), \text{p95}_m(\alpha)\}$, providing a complete characterization of the causal effect's magnitude and uncertainty.

#### 3.4.7 Temporal Propagation and Sequential Compounding

A theoretically important property of closed-loop interventions in sequential decision systems is the *temporal compounding* of causal effects. If a feature modification at timestep $t = 0$ induces a small change $\delta$ in the acceleration action, this changes the velocity at $t = 1$, which changes the relative positions of surrounding agents, which changes the observations at $t = 1$, which changes the latent representations at $t = 1$, which may further alter the activation of the steered feature at $t = 1$, compounding the initial perturbation.

The magnitude of compounding depends on the local Lyapunov exponent of the environment dynamics under the policy's action-observation map. In benign driving scenarios (highway cruise control in low-density traffic), the dynamics are nearly linear and compounding is modest. In safety-critical scenarios (dense intersections, highway merges), small perturbations can produce qualitatively different outcomes — a slight deceleration that avoids a collision vs. a slight acceleration that causes one — illustrating how individual feature modifications can have decisive causal consequences.

The paired rollout design provides a natural way to measure compounding: the episode length divergence (number of steps in the steered rollout vs. the baseline rollout) serves as an implicit indicator of how strongly the intervention has altered the trajectory dynamics.

---

## 4. Limitations and Future Research Directions

### 4.1 Intrinsic Limitations of Sparse Autoencoder Interpretability

The mechanistic interpretability framework described in this chapter rests on a foundational assumption that must be explicitly acknowledged: **sparse autoencoders do not guarantee true semantic decomposition of residual stream representations**. The SAE training objective optimizes for reconstruction fidelity and sparsity, but it does not enforce semantic coherence, causal transparency, or conceptual uniqueness among the learned features. The emergence of monosemantic features is a hoped-for consequence of the sparsity inductive bias, not a guaranteed property of the optimization.

Several specific limitations follow from this.

**Imperfect disentanglement**: Even when individual features achieve moderate selectivity scores (e.g., $|\rho| \approx 0.5$ with a particular telemetry field), the feature activation vector may still encode a mixture of multiple concepts that happen to be correlated in the driving dataset. The SAE cannot distinguish between a genuinely monosemantic feature and a polysemantic feature that appears monosemantic due to statistical regularities in the training distribution.

**Reconstruction infidelity and information loss**: The SAE reconstruction is not lossless. The reconstruction loss at convergence quantifies the fraction of the residual stream variance that the sparse features fail to account for. If important behavioral information is encoded in the unrecovered component — e.g., in the error term $\mathbf{x} - \hat{\mathbf{x}}$ — then the feature-based analysis will miss that information entirely. Interpretability analyses based on the SAE features are therefore analyses of the SAE's model of the residual stream, not the residual stream itself.

**Feature instability under distribution shift**: SAE features are trained on a finite dataset drawn from a specific distribution of driving scenarios. Features that appear monosemantic and causally interpretable within the training distribution may fail to generalize to out-of-distribution scenarios, where the residual stream manifold may be qualitatively different.

**Ambiguity in feature-to-concept mapping**: The annotation procedure assigns each feature the telemetry concept that maximizes Spearman correlation. However, high correlation does not imply causal specificity. Two telemetry fields with high mutual information will tend to produce similar annotations even if the feature encodes only one of the two.

### 4.2 Limitations of Causal Interventions

**Interventional distribution shift**: The additive delta intervention moves the residual stream vector to a point that may lie far from the natural activation manifold. At large $|\alpha|$, the steered representations are out-of-distribution inputs to the policy and value heads, and the heads' outputs in this regime may reflect extrapolation artifacts rather than genuine causal responses. This problem is analogous to the out-of-distribution extrapolation problem in causal inference with observational data.

**Indirect effects and confounding**: The intervention on feature $j$ may activate or suppress other features through the decoder's non-orthogonal geometry. The measured behavioral change $\Delta a$ or $\Delta v$ may partially reflect these indirect effects, obscuring the direct causal contribution of feature $j$ alone.

**Closed-loop compounding as a confounder**: In rollout interventions, the compounding of temporal effects makes it difficult to attribute episode-level metric changes to the feature's direct causal role, as opposed to the downstream consequences of environmental state divergence induced by early action modifications.

**Absence of a null distribution**: Without a null distribution of behavioral effects under random (semantically unmotivated) interventions, it is difficult to assess the statistical significance of observed $\Delta v$ and $\Delta a$ values. Future work should establish baseline behavioral variability under random interventions to contextualize the observed effects.

### 4.3 Limitations of the Annotation Vocabulary

**Finite telemetry vocabulary**: The interpretability of the annotation stage is bounded by the richness of the telemetry vocabulary. The telemetry fields employed in this work capture core driving concepts (speed, TTC, scene density, lateral dynamics) but do not exhaustively represent the full conceptual space of driving behavior. Features encoding concepts for which no telemetry analogue is provided — e.g., road geometry anticipation, multi-agent intention prediction, or abstract scene topology — will be annotated with the best available telemetry proxy, producing potentially misleading labels.

**Dataset bias**: The distribution of driving scenarios in the harvested dataset shapes which features are discoverable. Rare but safety-critical events (hard braking, near-miss situations) are underrepresented in routine driving datasets, causing their associated features to have low activation counts and potentially be classified as dead.

### 4.4 Future Research Directions

#### 4.4.1 Hierarchical Sparse Representations

The current SAE architecture operates at a single representational level, decomposing the final encoder output. A natural extension is hierarchical sparse decomposition, in which multiple SAE layers are stacked: the first layer decomposes the raw residual stream into primitive features, and subsequent layers discover higher-order combinations of those primitives that correspond to compound behavioral concepts. This mirrors the hierarchical structure of concept formation in cognitive science and could reveal organizational principles beyond those discoverable at a single level of abstraction.

#### 4.4.2 Causal Abstraction and Structural Causal Models

The causal feature steering experiments described in this chapter are grounded in the interventionist tradition of causal inference, but they stop short of identifying the full causal graph structure of the policy's internal computation. A more ambitious program would seek to construct a structural causal model of the residual stream, identifying not only the causal effect of individual features on behavior but also the causal relationships among features themselves. The framework of causal abstraction — relating high-level causal models to low-level neural mechanisms via abstraction maps — provides a theoretical scaffolding for this program.

#### 4.4.3 Concept Bottleneck Integration

Concept bottleneck models (CBMs) impose interpretability at training time by requiring the policy to route all information through a layer of explicitly defined, human-interpretable concepts. Integrating the post-hoc SAE interpretability framework with concept bottleneck training could yield policies that are both high-performing and intrinsically interpretable, without the reconstruction fidelity penalty inherent to post-hoc decomposition.

#### 4.4.4 Multimodal and Temporal Mechanistic Interpretability

The present framework focuses on the scalar residual stream at a single transformer layer. Future work should extend to multi-layer, multi-head analysis — examining how individual attention heads, feedforward layers, and cross-layer information flows contribute to the final representation — and to multi-modal inputs (e.g., sensor fusion from cameras and LiDAR) where the mechanistic structure of the encoder may be qualitatively different.

#### 4.4.5 Online Feature Adaptation

The current pipeline is entirely offline: the SAE is trained on a fixed corpus and applied statically. An online adaptation framework would allow the SAE to update its feature dictionary as the driving policy is fine-tuned or as the deployment distribution shifts, maintaining an up-to-date interpretability model that reflects the current state of the policy.

#### 4.4.6 Interpretable World Models

A longer-term direction is the integration of mechanistic interpretability with world model learning. If the transformer policy maintains an implicit internal model of future world states — a world model encoded within its recurrent or attention-based architecture — the SAE framework could be applied to decompose this world model into interpretable predictive features: concepts encoding anticipated agent trajectories, expected traffic density evolution, or predicted safety margins. Interpretable world models could serve as a foundation for safety monitoring and runtime verification of autonomous driving systems.

---

## 5. Experimental Methodology

This section documents the concrete experimental instantiation of the four-stage interpretability pipeline described in Section 3. It specifies the dataset, architecture configuration, hyperparameter search strategy, and intervention protocol. Where a specific hyperparameter value was selected via the tuning procedure described in Section 5.3, the selected value is marked **INSET(parameter\_name)** where the final value is not reported here.

---

### 5.1 Environment and Dataset

#### 5.1.1 Policy Architecture and Training Context

The target policy is a Wayformer-based encoder trained via proximal policy optimization (PPO) on the Waymo Open Motion Dataset (WOMD). The Wayformer encoder processes a vectorized representation of the driving scene — encoding ego-vehicle kinematics, surrounding agent states, and road topology as sets of polyline tokens — and produces a fixed-dimensional latent encoding that feeds the policy and value heads. The residual stream bottleneck, i.e., the final encoder output, has dimensionality $D = 128$.

The policy head and value head are multi-layer perceptrons applied directly to this $D$-dimensional bottleneck. All interpretability analyses are conducted at this bottleneck layer, which integrates all scene-level information prior to action generation. This choice is motivated by the observation that the final encoder output is the unique representational locus through which all information causally relevant to the policy's decision must pass.

#### 5.1.2 Waymo Open Motion Dataset

Representation harvesting is conducted on the Waymo Open Motion Dataset (WOMD), a large-scale real-world driving dataset comprising diverse traffic scenarios recorded across urban and suburban environments at 10 Hz. WOMD provides ground-truth trajectory data for all agents in the scene, including the self-driving car (SDC), surrounding vehicles, cyclists, and pedestrians.

The harvesting corpus covers **10,000 WOMD scenarios**, each comprising up to **80 timesteps** (corresponding to 8 seconds of simulated driving at $\Delta t = 0.1\,\text{s}$ per step). Not all scenarios reach the maximum episode length, as episodes terminate upon scenario completion or policy failure. After filtering to valid timesteps only (excluding post-termination padding), the harvested dataset contains **790,936 activation rows**, each a $D = 128$-dimensional residual stream vector co-registered with a rich telemetry vector.

| Dataset property | Value |
|---|---|
| Source dataset | Waymo Open Motion Dataset (WOMD) |
| Number of scenarios | 10,000 |
| Sampling frequency | 10 Hz ($\Delta t = 0.1\,\text{s}$) |
| Maximum episode length | 80 timesteps (8 s) |
| Total harvested rows | 790,936 |
| Residual stream dimension $D$ | 128 |
| Scalar telemetry fields | 8 |
| Boolean event fields | 5 |
| Per-agent array fields | 6 |

#### 5.1.3 Telemetry Coverage and Scenario Diversity

The 10,000-scenario corpus spans a range of traffic densities, road topologies, and behavioral regimes including highway merges, urban intersections, roundabouts, and straight road segments. The TTC-critical event rate (fraction of timesteps where $\text{TTC}_{\min} < 1.5\,\text{s}$) and the hard-braking event rate provide indirect indicators of safety-critical scenario coverage within the corpus. The deliberate exclusion of ego position coordinates (discussed in Section 3.1.3) ensures that the harvested telemetry reflects behavioral concepts rather than geographical artifacts of specific map locations.

---

### 5.2 Sparse Autoencoder Configuration

The SAE is configured according to the architectural and training hyperparameters below. Values determined by the hyperparameter search are shown with their search grid; the selected value after tuning is denoted INSET.

#### 5.2.1 Architecture

| Parameter | Symbol | Value |
|---|---|---|
| Residual stream dimension | $D$ | 128 |
| Expansion factor (searched) | $\gamma$ | **INSET(sae\_expansion\_factor)** $\in \{1,\ 4,\ 16\}$ |
| Latent dimension | $F = \gamma D$ | **INSET(sae\_latent\_dim)** |
| Encoder activation | — | ReLU |
| Decoder row constraint | $\|\mathbf{w}_j^{\text{dec}}\|_2$ | 1 (unit norm, projected after every step) |
| Pre-encoder bias | $\mathbf{b}_{\text{pre}}$ | Learned, initialized to $\mathbf{0}$ |

#### 5.2.2 Training

| Parameter | Symbol | Value |
|---|---|---|
| Sparsity coefficient (searched) | $\lambda$ | **INSET(sae\_l1\_coeff)** $\in \{0.02,\ 0.05,\ 0.07\}$ |
| Learning rate (searched) | $\eta_0$ | **INSET(sae\_learning\_rate)** $\in \{3\times10^{-4},\ 10^{-3}\}$ |
| LR schedule | — | Cosine annealing to 0 over $T$ epochs |
| Training epochs (searched) | $T$ | **INSET(sae\_epochs)** $\in \{50\}$ |
| Batch size | — | 4,096 |
| Optimizer | — | Adam ($\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$) |
| Gradient clip (max $\ell_2$ norm) | — | 1.0 |
| Decoder normalization | — | Projected gradient after every optimizer step |

#### 5.2.3 Activation Normalization

All harvested activations are whitened prior to SAE training using empirical per-dimension statistics computed over the full 790,936-row corpus:

$$\tilde{\mathbf{x}}_i = \frac{\mathbf{x}_i - \boldsymbol{\mu}}{\boldsymbol{\sigma} + \epsilon}, \quad \epsilon = 10^{-8}$$

The normalization statistics $(\boldsymbol{\mu}, \boldsymbol{\sigma}) \in \mathbb{R}^D \times \mathbb{R}^D$ are stored in the model checkpoint and applied identically during annotation and steering, ensuring that the encoding and decoding operations are consistent across all pipeline stages.

#### 5.2.4 Dead-Feature Monitoring Sample

A fixed reference sample of up to 8,192 activation vectors (drawn from the training corpus) is reserved exclusively for dead-feature percentage monitoring. This sample is held constant across all epochs and configurations, enabling fair cross-configuration comparison of dead feature prevalence.

---

### 5.3 Hyperparameter Selection Strategy

#### 5.3.1 Search Space and Grid Size

The SAE hyperparameters are selected via exhaustive grid search over the Cartesian product of the following axes, as defined in the pipeline configuration:

| Parameter | Symbol | Grid |
|---|---|---|
| Sparsity coefficient | $\lambda$ | $\{0.02,\ 0.05,\ 0.07\}$ |
| Expansion factor | $\gamma$ | $\{1,\ 4,\ 16\}$ |
| Training epochs | $T$ | $\{50\}$ |
| Learning rate | $\eta_0$ | $\{3\times10^{-4},\ 10^{-3}\}$ |

The total search space contains $3 \times 3 \times 1 \times 2 = 18$ distinct configurations. Each is trained independently from scratch using the full activation corpus and identical normalization statistics, ensuring that differences in the resulting checkpoints reflect only the hyperparameter variation.

#### 5.3.2 Primary Selection Criterion

The best configuration is selected by the **combined training loss** at the optimal epoch checkpoint:

$$\mathcal{L}^* = \min_t \left[ \mathcal{L}_{\text{recon}}(t) + \lambda\,\Omega(\mathbf{f}(t)) \right]$$

where $\mathcal{L}_{\text{recon}}(t)$ and $\Omega(\mathbf{f}(t))$ are the epoch-averaged reconstruction and sparsity terms, respectively. The checkpoint minimizing $\mathcal{L}^*$ is saved for each configuration.

#### 5.3.3 Secondary Ranking Criteria

Among configurations with near-equivalent $\mathcal{L}^*$, secondary ranking is applied in the following order:

1. **Dead feature percentage**: lower is preferred, as configurations with excessive dead features waste representational capacity and reduce the effective SAE latent dimensionality.
2. **$L_0$ sparsity**: the selected configuration should achieve $L_0 \ll F$, confirming that the learned representation is genuinely sparse rather than near-dense. Configurations achieving $L_0 > F/4$ (more than 25% of features active per observation) are considered insufficiently sparse regardless of combined loss.

Reconstruction loss alone is not used as a selection criterion, because a configuration with $\lambda \to 0$ trivially minimizes reconstruction at the cost of dense, uninterpretable features.

---

### 5.5 Causal Intervention Configuration

#### 5.5.1 Steering Temperature Schedule

Causal interventions are evaluated across the following bidirectional temperature schedule:

$$\alpha \in \{-5.0,\ -2.0,\ -1.0,\ +1.0,\ +2.0,\ +5.0\}$$

This schedule spans feature suppression ($\alpha < 0$), mild perturbation ($|\alpha| = 1$), and strong amplification ($|\alpha| = 5$). The bidirectional design enables monotonicity testing and sign-flip analysis as described in Section 3.4.4. For the closed-loop setting, all temperature levels are evaluated in a single compiled pass using vectorized map operations over the $\alpha$ axis, avoiding repeated host–device synchronization.

#### 5.5.2 Single-Step Intervention Protocol

Single-step causal steering evaluates the immediate behavioral response to a feature modification at a single initial observation, without rolling out the modified action through the environment. Steering is conducted across **INSET(n\_scenarios\_single\_step)** independently drawn WOMD scenarios. For each scenario and temperature level, the behavioral shift is measured as:

$$\Delta a_{\text{accel}},\quad \Delta a_{\text{steer}},\quad \Delta v$$

relative to the unperturbed baseline. Aggregation across scenarios yields the distribution statistics $\{\mu,\,\sigma,\,\text{median},\,p_5,\,p_{95},\,\text{SFF}\}$ for each temperature level and each behavioral dimension.

#### 5.5.3 Closed-Loop Rollout Protocol

Closed-loop rollout steering sustains the feature modification at every timestep throughout the episode. Rollouts are conducted across **INSET(n\_scenarios\_rollout)** WOMD scenarios, with a maximum episode length of **80 timesteps**. For each scenario, one baseline and one steered rollout per temperature level are initialized from identical initial states. The steered rollout does not reset the environment between timesteps; state divergence from the baseline trajectory is expected and constitutes the causal signal of interest.

The behavioral effect is quantified over the following episode-level metric suite:

| Metric | Aggregation rule |
|---|---|
| At-fault collision | Max over episode (binary: any collision = 1) |
| Time-to-collision (TTC) | Mean over valid agent-pairs per episode |
| Comfort (jerk) | Mean longitudinal/lateral jerk per episode |
| Speed limit violation | Mean fraction of timesteps in violation |
| On multiple lanes | Mean fraction of timesteps multi-lane |
| Driving direction compliance | Mean fraction of timesteps compliant |
| Progress ratio | Final route completion fraction at termination |

The delta metric $\Delta m = m_{\text{steered}} - m_{\text{baseline}}$ is computed for each metric $m$, scenario, and temperature. Cross-scenario aggregation yields the full distribution of $\Delta m$ per temperature, reported as $\{\mu,\,\sigma,\,\text{median},\,p_5,\,p_{95},\,\text{SFF}\}$.

#### 5.5.4 Safety Thresholds

The following telemetry thresholds, fixed across all pipeline stages, define safety-critical event classification:

| Threshold | Symbol | Value |
|---|---|---|
| TTC critical threshold | $\tau_{\text{crit}}$ | $1.5\,\text{s}$ |
| Hard braking deceleration | $a_{\text{brake}}$ | $0.4\,g = 3.924\,\text{m/s}^2$ |
| Per-step hard braking threshold | $\Delta v_{\text{brake}} = a_{\text{brake}}\,\Delta t$ | $0.3924\,\text{m/s}$ |
| Lead vehicle following distance | $d_{\text{follow}}$ | $30\,\text{m}$ |
| Ahead classification boundary | $x_{\text{local}}$ | $> 2.0\,\text{m}$ |
| Left/right classification boundary | $|y_{\text{local}}|$ | $> 1.0\,\text{m}$ |

These thresholds are grounded in established automotive safety conventions: $1.5\,\text{s}$ TTC is a widely used surrogate for imminent collision risk; $0.4g$ deceleration corresponds to firm but non-emergency braking consistent with ISO 2631 comfort standards for passenger vehicles.

---

## Chapter Summary

This chapter has developed a four-stage mechanistic interpretability framework for transformer-based autonomous driving policies, grounded in the theoretical principles of sparse coding, superposition, and causal representation analysis. The framework moves systematically from passive harvesting of latent activations, through unsupervised sparse feature decomposition, through statistical semantic annotation, to causal intervention and behavioral measurement. Each stage is designed to advance the interpretability analysis from correlational description toward mechanistic, intervention-based understanding.

The theoretical foundations establish that the difficulty of interpreting transformer residual streams arises from the superposition of multiple concepts within a low-dimensional bottleneck, and that sparse autoencoders provide a principled mechanism for decomposing this superposition into a more interpretable overcomplete representation. The methodological design reflects careful attention to the unique challenges of the autonomous driving setting: the closed-loop sequential structure of the task, the safety-critical nature of the behavioral outputs, and the requirement for causal — not merely correlational — evidence of feature function.

The limitations acknowledged throughout this chapter define the boundaries of what the framework can and cannot claim. Interpretability remains fundamentally approximate: the SAE is a model of the residual stream, not the residual stream itself; annotation is statistical, not definitional; and causal interventions provide evidence for, not proof of, feature function. These limitations do not invalidate the framework but rather delineate the epistemic standards to which its findings should be held.
