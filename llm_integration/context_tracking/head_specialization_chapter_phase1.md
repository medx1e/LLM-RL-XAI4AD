# Head Specialization Analysis in Transformer-Based Autonomous Driving Policies

## A Study in Mechanistic Interpretability for Explainable Deep Reinforcement Learning

---

## 1. Introduction and Motivation

The deployment of deep reinforcement learning (DRL) agents in safety-critical applications such as autonomous driving demands not only high task performance but a degree of transparency that allows practitioners, regulators, and downstream safety auditors to understand *why* a policy produces a given action under a given set of environmental conditions. The opacity of modern deep neural networks constitutes a fundamental barrier to this goal, and the interpretability of transformer-based architectures—despite their remarkable empirical success across sequential decision-making domains—remains an area of active theoretical investigation.

Among the components of transformer-based DRL policies, the multi-head attention mechanism occupies a uniquely privileged position from an interpretability standpoint. Unlike dense feedforward layers, attention weights form an explicit, human-interpretable routing structure that maps the policy's information-gathering behaviour onto the geometric structure of the observation space. When a policy processes a multi-agent traffic scenario, the attention weights assigned to surrounding vehicles encode, at least implicitly, the policy's prioritisation of contextually salient entities—those whose current or anticipated states most directly govern the agent's optimal action.

The central thesis of this chapter is that the multiple heads comprising a transformer encoder do not, in a well-trained policy, converge to functionally redundant representations. Rather, through the pressure of mid-to-end reinforcement learning over diverse driving scenarios, distinct heads may be induced to specialise for different semantic abstractions of the scene: one head attending preferentially to proximate collision threats, another tracking temporal approach dynamics, another sensitive to spatial layout relative to the ego vehicle's heading. This phenomenon—which we term *head specialization*—constitutes a form of emergent functional decomposition within the policy's perceptual representation, and its characterisation provides a mechanistically grounded explanation of the policy's situational awareness.

This chapter is divided into two principal phases of analysis. The first phase, which is the subject of the present document, establishes a baseline characterisation of head specialization using a direct alignment methodology applied to a single-step snapshot of the policy's attention state. This approach quantifies the degree of functional alignment between each attention head's vehicle-level attention distribution and a curated battery of semantic features derived from the ground-truth scenario state. The second phase, to be addressed subsequently, introduces an architectural intervention in the form of specialization-inducing regularisation, seeking to explicitly encourage the functional differentiation identified in the observational analysis of Phase I.

The analyses conducted here draw on a trained proximal policy optimisation (PPO) agent whose policy network employs a late-fusion multi-modal transformer encoder operating over structured vectorised representations of the driving scene. The combination of this specific encoder architecture, the multi-agent observation structure, and the diversity of the Waymo Open Dataset provides a natural substrate for detecting and characterising emergent head specialization.

---

## 2. Theoretical Background

### 2.1 Transformer Attention in Deep Reinforcement Learning

The self-attention and cross-attention mechanisms of the transformer architecture, originally formulated for sequence-to-sequence modelling in natural language processing, have been progressively adapted to the structured, temporally grounded observation spaces of reinforcement learning. In the autonomous driving domain, the key challenge is to process observations that are inherently multi-modal—comprising agent trajectories, geometric roadgraph structure, traffic signal states, and goal-directed path representations—while maintaining permutation invariance over variable-length entity sets and respecting the temporal ordering of historical states.

The canonical scaled dot-product attention operation is defined for query matrix $\mathbf{Q} \in \mathbb{R}^{L_q \times d_k}$, key matrix $\mathbf{K} \in \mathbb{R}^{L_k \times d_k}$, and value matrix $\mathbf{V} \in \mathbb{R}^{L_k \times d_v}$ as:

$$
\mathrm{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
$$

where $d_k$ denotes the per-head dimension and the softmax is applied row-wise, producing a row-stochastic attention weight matrix $\mathbf{A} \in \mathbb{R}^{L_q \times L_k}$. In the multi-head formulation, $H$ such operations are executed in parallel with independent learned projections:

$$
\mathrm{MHA}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_H)\,\mathbf{W}^O
$$

where $\mathrm{head}_h = \mathrm{Attn}(\mathbf{Q}\mathbf{W}^Q_h,\, \mathbf{K}\mathbf{W}^K_h,\, \mathbf{V}\mathbf{W}^V_h)$, with learned projection matrices $\mathbf{W}^Q_h, \mathbf{W}^K_h \in \mathbb{R}^{d_{\mathrm{model}} \times d_k}$, $\mathbf{W}^V_h \in \mathbb{R}^{d_{\mathrm{model}} \times d_v}$, and $\mathbf{W}^O \in \mathbb{R}^{Hd_v \times d_{\mathrm{model}}}$.

The theoretical motivation for multiple heads rests on the hypothesis that different heads can learn to attend according to distinct relational patterns—syntactic vs. semantic, local vs. global, positional vs. content-based—and that the concatenation of their outputs preserves richer representational structure than any single head could provide. In the context of driving policy learning, this motivates the conjecture that heads may specialise along the dimensions of semantic salience most relevant to safe navigation: proximity, threat imminence, spatial positioning, and dynamic approach rate.

### 2.2 Head Specialization: Theory and Prior Work

The question of whether individual attention heads in large transformers acquire specialised roles has attracted substantial investigation in the natural language processing literature. Voita et al. (2019) demonstrated that a significant fraction of attention heads in trained language models can be pruned with minimal performance degradation, suggesting that many heads perform redundant or low-utility computations, while a sparse subset carries disproportionate functional importance. Complementary work by Michel et al. (2019) confirmed this sparsity hypothesis through systematic ablation studies.

More directly relevant to the present work is the line of research connecting specific heads to interpretable linguistic functions: syntactic dependency parsing (Clark et al., 2019), coreference resolution (Jain and Wallace, 2019), and positional encoding (Voita et al., 2019). These findings established the conceptual framework of *functional head specialization*: the emergence, through gradient descent, of heads whose attention patterns are systematically predictive of specific structural or semantic properties of the input.

Translating this framework to the DRL setting requires a careful reconceptualisation. In language modelling, the target properties are discrete and linguistically motivated; in autonomous driving, they are continuous, physically grounded, and safety-relevant. The semantic features most relevant to driving specialization include:

- **Time-to-Collision (TTC)**: the expected temporal horizon until collision under constant-velocity extrapolation, a primary safety-relevant signal,
- **Euclidean Distance**: the geometric proximity of a neighbouring agent to the ego vehicle,
- **Closing Speed**: the signed rate of change of inter-agent distance, encoding approach dynamics,
- **Spatial Position**: directional predicates (ahead, behind, left flank, right flank) relative to the ego vehicle's heading frame,
- **Absolute Speed**: the magnitude of a neighbouring agent's velocity vector, serving as a proxy for dynamic saliency.

The hypothesis is that distinct attention heads will exhibit systematic functional alignment with distinct subsets of these features across diverse driving scenarios, indicating that the policy has implicitly decomposed scene understanding into specialised representational channels.

### 2.3 Mechanistic Interpretability in Sequential Decision-Making

Mechanistic interpretability, as formalised by Conmy et al. (2023) and Elhage et al. (2021) among others, seeks to identify the computational circuits within neural networks that causally mediate specific input-output behaviours, going beyond post-hoc attribution methods to provide explanations grounded in the network's actual internal computations. The circuit-identification perspective distinguishes three levels of analysis: (1) the identification of which components (heads, neurons, layers) carry task-relevant information, (2) the characterisation of what information each component encodes, and (3) the explanation of how information is routed and transformed across components.

The present work operates primarily at the first and second levels. By assessing the alignment between attention weight distributions and semantic features extracted from ground-truth scenario state, we aim to identify which heads carry safety-relevant perceptual information and what specific aspects of scene geometry and dynamics each head is sensitive to. This constitutes a form of *functional probing* in the mechanistic interpretability literature: an assessment of the hypothesis that a given representational structure encodes a given property, without requiring architectural modification.

A critical epistemological consideration is the distinction between *correlation* and *causation* in this context. A strong functional alignment between a head's attention distribution and TTC does not, in isolation, establish that the head's threat-sensitive attention is causally necessary for safe driving behaviour. Causal attribution requires causal intervention—e.g., ablating or redirecting the head's attention—and establishing performance degradation on safety-critical scenarios. Phase II of the analysis addresses this concern through forced specialization, which establishes causal relevance by construction. Phase I establishes the observational baseline that motivates and guides the Phase II intervention.

---

## 3. Architectural Overview of Phase I: Simple Head Specialization Analysis

### 3.1 The Multi-Modal Observation Space

The autonomous driving agent operates in a vectorised, multi-modal observation space designed to encode the full driving scene into a structured numerical representation. The observation comprises five distinct modalities, each covering a different aspect of the scene at a given moment:

1. **Ego Trajectory**: A temporal history of the self-driving vehicle's own kinematic state over the most recent $T_{\mathrm{obs}}$ = 5 timesteps, encoded as a sequence of position, velocity, heading, and validity flags.
2. **Neighbouring Agent Trajectories**: Kinematic state histories for the $N_{\mathrm{agents}}$ = 8 most proximate surrounding agents, similarly encoded with temporal depth $T_{\mathrm{obs}}$. Each agent's trajectory is represented as a fixed-length sequence of feature vectors concatenating position offsets relative to the ego, velocity components, heading angle, and a binary validity indicator.
3. **Roadgraph Features**: A topological description of the static road geometry, encoded as a sequence of $N_{\mathrm{rg}}$ = 200 sampled roadgraph points drawn from the top-$k$ nearest road elements by lateral distance to the ego vehicle. Each point encodes position, direction, lane type classification, and validity.
4. **Traffic Light States**: The observed states of the $N_{\mathrm{tl}}$ = 5 nearest traffic control elements over the historical observation window, encoding signal phase, position, and validity.
5. **Goal Path**: A representation of the ego vehicle's designated navigation path as a sequence of $N_{\mathrm{gps}}$ = 3 waypoints, encoding relative positions along the planned route.

This modular structure reflects the semantic partitioning of the driving scene into distinct informational sources, each requiring different processing inductive biases. The agent trajectories demand permutation equivariance over a variable-length entity set; the roadgraph requires processing of a high-dimensional geometric structure; the traffic light and goal path modalities have comparatively low dimensionality but high semantic specificity. A monolithic observation vector would obscure these structural differences; the modular representation preserves them.

Formally, the total observation at timestep $t$ can be written as:

$$
\mathbf{o}_t = \bigl(\mathbf{O}^{\mathrm{sdc}}_t,\; \mathbf{O}^{\mathrm{agents}}_t,\; \mathbf{O}^{\mathrm{rg}}_t,\; \mathbf{O}^{\mathrm{tl}}_t,\; \mathbf{O}^{\mathrm{gps}}_t\bigr)
$$

where $\mathbf{O}^{\mathrm{sdc}}_t \in \mathbb{R}^{T_{\mathrm{obs}} \times d_{\mathrm{sdc}}}$, $\mathbf{O}^{\mathrm{agents}}_t \in \mathbb{R}^{N_{\mathrm{agents}} \times T_{\mathrm{obs}} \times d_{\mathrm{agent}}}$, and analogously for the remaining modalities. In practice, these structured tensors are flattened into a single observation vector prior to being passed to the encoder, which internally reconstructs the modality-specific structure through learned projection and reshaping operations.

### 3.2 The Late-Fusion Cross-Attention Encoder

The policy network employs an encoder architecture inspired by the Wayformer framework, which performs late-fusion of the five modalities through a shared set of learned latent queries attending to each modality independently via dedicated cross-attention blocks. This architecture is a specific instantiation of the Perceiver architecture class, in which a compact latent representation is built through iterative cross-attention to a high-dimensional input without requiring full pairwise self-attention over the entire token sequence.

The encoder maintains a set of $N_{\mathrm{latent}}$ = 16 learned latent query vectors $\mathbf{Z} \in \mathbb{R}^{N_{\mathrm{latent}} \times d_{\mathrm{latent}}}$. For each modality $m \in \{\mathrm{sdc}, \mathrm{agents}, \mathrm{rg}, \mathrm{tl}, \mathrm{gps}\}$, the modality's flattened token sequence $\mathbf{X}^m \in \mathbb{R}^{L_m \times d_m}$ is projected into key and value matrices via learned linear maps, and the latent queries attend to these via cross-attention:

$$
\mathbf{Z}^m = \mathrm{CrossAttn}_m(\mathbf{Z},\; \mathbf{X}^m)
$$

$$
\mathrm{CrossAttn}_m(\mathbf{Z}, \mathbf{X}^m) = \mathrm{softmax}\!\left(\frac{\mathbf{Z}\,(\mathbf{W}^K_m \mathbf{X}^{m\top})}{\sqrt{d_k}}\right) \mathbf{W}^V_m \mathbf{X}^m
$$

In the multi-head formulation, this produces raw attention weight tensors for each modality and each of the $H$ = 4 attention heads. The late-fusion paradigm means that modality interactions occur only after each modality's token sequence has been independently compressed into the latent space, rather than through a single joint self-attention over the concatenated token sequence. This architectural choice reduces computational cost while preserving the structural correspondence between attention weights and semantic entities—a key enabler of the head specialization analysis.

The per-modality latent outputs $\mathbf{Z}^m$ are aggregated into a unified latent representation that is subsequently processed by the policy and value heads.

### 3.3 Token-to-Entity Correspondence and Its Interpretability Significance

The interpretability value of this architecture lies in the structured relationship between attention weights and semantic entities. Within the agent-module cross-attention block, each attention entry $A_{l, j, h}$ represents the weight assigned by latent query $l$ to token $j$ of the agent trajectory sequence, under attention head $h$. Since the agent trajectory tokens are ordered as agent-major—the first $T_{\mathrm{obs}}$ tokens correspond to agent 0's temporal history, the next $T_{\mathrm{obs}}$ to agent 1, and so on—these attention weights have a direct semantic referent: high attention to the token block corresponding to agent $i$ indicates that the latent representation is strongly informed by agent $i$'s historical trajectory.

This token-to-entity correspondence enables a principled aggregation from raw attention weights to per-agent attention scalars, forming the foundation of the head specialization analysis. Specifically, the total attention attributed to a particular agent by a particular head can be computed by summing the relevant sub-block of the attention tensor, yielding a scalar quantity that is amenable to semantic comparison against ground-truth properties of that agent.

### 3.4 Policy and Value Architecture

The compressed latent representation produced by the encoder is passed to separate policy and value networks. The policy network maps the latent state to the parameters of an action distribution—typically a multivariate Gaussian in the continuous action space of steering angle and acceleration magnitude—while the value network produces a scalar estimate of the expected discounted cumulative reward. Both heads are implemented as shallow multilayer perceptrons operating on the concatenated encoder output.

The mid-to-end training of this architecture under the PPO objective means that the encoder's attention patterns are shaped not by an explicit supervision signal specifying what each head should attend to, but entirely by the downstream task: maximising discounted cumulative reward in the driving simulation. The head specialization observed post-training therefore constitutes an *emergent* functional organisation, arising from the geometry of the reinforcement learning problem and the inductive biases of the architecture.

---

## 4. Experimental Methodology

### 4.1 Simulation Environment

The training and evaluation environments are implemented within the Waymax simulation framework, a high-fidelity autonomous driving simulator built on the Waymo Open Dataset. Waymax provides deterministic simulation of real-world traffic scenarios extracted from logged driving data, enabling reproducible evaluation over a diverse distribution of scene types, traffic densities, and interaction complexities.

The ego vehicle dynamics are modelled using an invertible bicycle model with normalised action inputs, providing a physically plausible mapping from policy outputs to vehicle kinematics while maintaining computational tractability. Specifically, the ego vehicle's state is evolved according to:

$$
\dot{x} = v \cos(\psi), \quad \dot{y} = v \sin(\psi), \quad \dot{\psi} = \frac{v \tan(\delta)}{L}, \quad \dot{v} = a
$$

where $(x, y)$ denotes the position, $\psi$ the heading angle, $v$ the longitudinal speed, $\delta$ the steering angle, $a$ the acceleration command, and $L$ the wheelbase length. The action space $\mathcal{A} \subset \mathbb{R}^2$ consists of normalised steering and acceleration commands, clipped to $[-1, 1]^2$ and mapped to physical ranges through the dynamics model.

Surrounding agents follow their logged trajectories from the Waymo Open Dataset, providing naturalistic and diverse traffic behaviour without requiring the computational overhead of multi-agent policy learning. The simulation is initialised from scenario seeds drawn from the training split of the dataset, covering urban intersections, highway segments, residential roads, and parking scenarios.

The episode termination conditions include:

- **Collision**: any bounding box overlap between the ego vehicle and a surrounding agent,
- **Off-road departure**: the ego vehicle centre exiting the valid drivable surface,
- **Episode timeout**: the scenario reaching its maximum duration of $T_{\max}$ = 80 timesteps.

The reward function is structured to balance multiple objectives:

$$
r_t = r^{\mathrm{progress}}_t + \lambda_{\mathrm{off}} r^{\mathrm{offroad}}_t + \lambda_{\mathrm{col}} r^{\mathrm{collision}}_t + \lambda_{\mathrm{comfort}} r^{\mathrm{comfort}}_t
$$

where $r^{\mathrm{progress}}_t$ rewards progress along the goal path, $r^{\mathrm{offroad}}_t$ and $r^{\mathrm{collision}}_t$ are negative reward signals for safety violations, and $r^{\mathrm{comfort}}_t$ penalises excessive jerk to encourage smooth driving. The weighting coefficients $\lambda_{\mathrm{off}}$, $\lambda_{\mathrm{col}}$, $\lambda_{\mathrm{comfort}}$ are tuned to ensure safety constraints dominate comfort considerations while maintaining sufficient task progress incentive.

### 4.2 Observation Modalities and Feature Engineering

The observation space is constructed as a structured vectorised representation derived from the simulation state at each timestep. For the neighbouring agent modality, each of the $N_{\mathrm{agents}}$ closest agents (by current distance to the ego) is represented by a $T_{\mathrm{obs}}$-step trajectory of feature vectors encoding:

$$
\phi^{\mathrm{agent}}_{i, t} = [\Delta x_{i,t},\; \Delta y_{i,t},\; v^x_{i,t},\; v^y_{i,t},\; \psi_{i,t},\; w_i,\; \ell_i,\; \mathbb{1}[\mathrm{valid}_{i,t}]]
$$

where $(\Delta x_{i,t}, \Delta y_{i,t})$ are position offsets relative to the ego vehicle at time $t$, $(v^x_{i,t}, v^y_{i,t})$ are velocity components in the world frame, $\psi_{i,t}$ is heading angle, $(w_i, \ell_i)$ are agent width and length, and $\mathbb{1}[\mathrm{valid}_{i,t}]$ is a binary validity indicator. The validity indicator is critical for handling the variable effective number of agents per scenario: invalid tokens are masked in the attention computation to prevent the policy from attending to phantom agents.

The roadgraph representation samples the $N_{\mathrm{rg}}$ nearest road points according to a combination of lateral distance and lane relevance scoring. Each roadgraph token encodes:

$$
\phi^{\mathrm{rg}}_j = [\Delta x_j,\; \Delta y_j,\; \cos\theta_j,\; \sin\theta_j,\; \mathrm{type}_j,\; \mathrm{valid}_j]
$$

where $(\Delta x_j, \Delta y_j)$ is the position of road point $j$ relative to the ego, $(\cos\theta_j, \sin\theta_j)$ encodes the local road direction, and $\mathrm{type}_j$ is a categorical encoding of the road element classification (lane centre, road boundary, crosswalk, etc.). Traffic light features encode the state of the $N_{\mathrm{tl}}$ nearest traffic control elements over the observation window, and goal path features encode a fixed-length waypoint sequence along the designated navigation route. Together, the five modalities define a structured observation whose total flattened dimension is:

$$
d_{\mathrm{obs}} = T_{\mathrm{obs}}\,d_{\mathrm{sdc}} + N_{\mathrm{agents}}\,T_{\mathrm{obs}}\,d_{\mathrm{agent}} + N_{\mathrm{rg}}\,d_{\mathrm{rg}} + N_{\mathrm{tl}}\,T_{\mathrm{obs}}\,d_{\mathrm{tl}} + N_{\mathrm{gps}}\,d_{\mathrm{gps}}
$$

### 4.3 Training Configuration

The policy is trained using the Proximal Policy Optimisation algorithm with generalised advantage estimation (Schulman et al., 2017). The key hyperparameters of the training procedure are:

- **Discount factor**: $\gamma$ = 0.99
- **Clip ratio**: $\epsilon$ = 0.2
- **Entropy coefficient**: $c_{\mathrm{ent}}$ = 0.01
- **Value function coefficient**: $c_{\mathrm{val}}$ = 0.5
- **Learning rate**: $\alpha$ = 3e-4, with constant scheduling
- **Batch size**: $B$ = 64 transitions per update
- **Total training steps**: 20,000,000

The encoder architecture employs:

- **Number of attention heads**: $H$ = 4 per cross-attention module
- **Latent dimension**: $d_{\mathrm{latent}}$ = 64
- **Number of latent queries**: $N_{\mathrm{latent}}$ = 16
- **Per-head dimension**: $d_k = d_{\mathrm{latent}} / H$ (which is 16)
- **Feed-forward dimension**: 128 (with a 256-dimensional internal hidden layer inside the MLP)

Training is conducted on a single NVIDIA GPU (NVIDIA  H100 24GB) for a total of approximately 4 hours. Environment rollouts are collected using 16 parallel environments to ensure sufficient replay diversity.

### 4.4 Model Checkpoint Selection and Evaluation Protocol

For the head specialization analysis, the final trained checkpoint is used as the reference model, selected based on convergence of the training reward signal and stability of evaluation metrics. Convergence is assessed through a rolling window analysis of the mean episode reward over the last 2,000,000 training steps.

The evaluation protocol for the interpretability analysis differs from the standard performance evaluation. Rather than measuring task metrics such as collision rate or progress completion, the interpretability analysis aims to characterise the structural organisation of the policy's internal attention patterns. This requires a sufficiently large and diverse scenario set, distinct from the training distribution, to avoid overfitting of the observed patterns to scenario-specific idiosyncrasies.

The analysis is conducted over $N_{\mathrm{scenarios}}$ = 100 scenarios drawn from the validation split of the Waymo Open Dataset. Each scenario is processed independently, with the policy frozen at its trained weights, to extract attention weights and the corresponding ground-truth semantic features of the scene.

---

### 5.1 Motivation and Research Questions

The first phase of head specialization analysis establishes a baseline characterisation of the functional roles that individual attention heads have adopted during training. The fundamental question is whether the $H$ attention heads within the agent-module cross-attention block exhibit statistically distinguishable patterns of vehicle-level attention allocation, and whether these patterns are systematically reflective of semantically meaningful properties of the driving scene.

This question is operationalised through a set of formal research hypotheses:

**Hypothesis H1 (Specialization Existence)**: At least one attention head exhibits a systematic functional alignment between its vehicle-level attention distribution and a defined semantic feature across the evaluation scenario set.

**Hypothesis H2 (Specialization Diversity)**: The primary semantic correlates differ across at least two distinct attention heads, indicating functional non-redundancy within the encoder's representational capacity.

**Hypothesis H3 (Safety Salience)**: At least one attention head exhibits preferential attention to high-threat vehicles as characterised by low time-to-collision, indicating emergence of a threat-detection function.

**Hypothesis H4 (Proximity Sensitivity)**: At least one attention head exhibits preferential attention to proximate vehicles as characterised by small inter-vehicle distance, indicating emergence of a proximity-monitoring function.

The analysis is specifically restricted, in Phase I, to the single-observation snapshot paradigm: each scenario is processed at a single reference timestep (the episode initialisation state), and attention weights and semantic features are jointly extracted at that timestep. This restriction avoids the temporal aggregation and risk-conditioned analyses of the more sophisticated Phase II methodology, providing a methodologically tractable baseline that can be interpreted without appealing to temporal dynamics.

### 5.2 Offline Attention Extraction Protocol

The attention extraction procedure operates in an offline setting: the trained policy is frozen and applied to a large set of scenarios without any environmental interaction or policy update. For each scenario $n \in \{1, \ldots, N_{\mathrm{scenarios}}\}$, the following extraction procedure is executed:

1. **Environment initialisation**: the simulation is reset to the initial state of scenario $n$, establishing the full traffic configuration with all agents at their logged starting positions. The ego vehicle is initialised at its logged starting pose.
2. **Observation computation**: the structured observation $\mathbf{o}_0^{(n)} \in \mathbb{R}^{d_{\mathrm{obs}}}$ is computed from the initialised simulation state. The observation is assembled by extracting the $N_{\mathrm{agents}}$ closest surrounding agents, the top-$N_{\mathrm{rg}}$ roadgraph points, the $N_{\mathrm{tl}}$ nearest traffic signals, and the $N_{\mathrm{gps}}$-waypoint goal path, all transformed into the ego vehicle's coordinate frame.
3. **Encoder forward pass**: the observation is passed through the frozen encoder in inference mode. The encoder returns both the compressed latent representation and the raw attention weight tensors for each modality's cross-attention block. Specifically, for the agent module, the attention weight tensor $\mathbf{A}^{\mathrm{agents},(n)} \in \mathbb{R}^{N_{\mathrm{latent}} \times (N_{\mathrm{agents}} \cdot T_{\mathrm{obs}}) \times H}$ is extracted and retained.
4. **Semantic feature extraction**: simultaneously with the encoder forward pass, the ground-truth semantic features of the scene are computed from the simulation state, as detailed in Section 5.4.
5. **Data aggregation**: the pair $\bigl(\mathbf{A}^{\mathrm{agents},(n)},\; \mathbf{f}^{(n)}\bigr)$ for scenario $n$ is stored, where $\mathbf{f}^{(n)} \in \mathbb{R}^{N_{\mathrm{agents}} \times F}$ collects the $F$ semantic features for each of the $N_{\mathrm{agents}}$ vehicles.

This procedure yields a dataset of $\bigl\{(\mathbf{A}^{\mathrm{agents},(n)},\; \mathbf{f}^{(n)})\bigr\}_{n=1}^{N_{\mathrm{scenarios}}}$ pairs that form the basis for all subsequent analyses.

### 5.3 Attention Weight Aggregation

The raw attention weight tensor for the agent module, $\mathbf{A}^{\mathrm{agents}} \in \mathbb{R}^{N_{\mathrm{latent}} \times (N_{\mathrm{agents}} \cdot T_{\mathrm{obs}}) \times H}$, encodes the weight that each of the $N_{\mathrm{latent}}$ latent queries assigns to each of the $N_{\mathrm{agents}} \cdot T_{\mathrm{obs}}$ agent-trajectory tokens for each of the $H$ attention heads. To obtain a per-agent, per-head attention scalar that can be compared with per-agent semantic features, a principled aggregation procedure is required.

The token sequence for the agent module is structured such that tokens are indexed agent-major: $(i, t) \mapsto i \cdot T_{\mathrm{obs}} + t$ for agent index $i \in \{0, \ldots, N_{\mathrm{agents}}-1\}$ and observation timestep $t \in \{0, \ldots, T_{\mathrm{obs}}-1\}$. The attention sub-block for latent query $l$, agent $i$, and head $h$ across its temporal token block is therefore:

$$
A_{l, i, :, h} = \mathbf{A}^{\mathrm{agents}}_{l,\; i \cdot T_{\mathrm{obs}} : (i+1) \cdot T_{\mathrm{obs}},\; h} \in \mathbb{R}^{T_{\mathrm{obs}}}
$$

The aggregation from raw attention to per-agent scalars proceeds in two steps:

**Step 1 — Temporal aggregation**: sum the attention over the $T_{\mathrm{obs}}$ observation timestep tokens for each (latent, agent, head) combination:

$$
\tilde{A}_{l, i, h} = \sum_{t=0}^{T_{\mathrm{obs}}-1} A_{l, i, t, h}
$$

yielding $\tilde{\mathbf{A}} \in \mathbb{R}^{N_{\mathrm{latent}} \times N_{\mathrm{agents}} \times H}$. This summation is motivated by the interpretation that the total attention devoted to an agent's trajectory, summed across all temporal positions, reflects the head's aggregate interest in that agent's history, rather than a specific point in time. For a row-stochastic attention matrix (softmax-normalised), this sum preserves the relative distribution of attention mass across agents: the temporal aggregation collapses the within-agent structure while retaining the across-agent distribution that is central to the specialization analysis.

**Step 2 — Latent aggregation**: sum the attention over all $N_{\mathrm{latent}}$ latent queries:

$$
\hat{A}_{i, h} = \sum_{l=0}^{N_{\mathrm{latent}}-1} \tilde{A}_{l, i, h}
$$

yielding $\hat{\mathbf{A}} \in \mathbb{R}^{N_{\mathrm{agents}} \times H}$. Summation over latent queries is used rather than averaging to preserve total attention mass, which is conserved across agents by the row-stochastic property of the softmax. Specifically, for each latent query $l$ and head $h$:

$$
\sum_{i=0}^{N_{\mathrm{agents}}-1} \tilde{A}_{l, i, h} = \sum_{i=0}^{N_{\mathrm{agents}}-1}\sum_{t=0}^{T_{\mathrm{obs}}-1} A_{l, i, t, h} = 1
$$

Thus $\hat{A}_{i,h} / N_{\mathrm{latent}}$ represents the fraction of total attention devoted by head $h$ to agent $i$, averaged over latent queries. By transposing the result, we obtain the analysis-ready per-head, per-agent attention matrix:

$$
\mathbf{W}^{\mathrm{attn}} = \hat{\mathbf{A}}^\top \in \mathbb{R}^{H \times N_{\mathrm{agents}}}
$$

where $W^{\mathrm{attn}}_{h, i}$ represents the aggregated attention weight assigned by head $h$ to agent $i$ in the given scenario. This matrix constitutes the fundamental unit of analysis for the head specialization study.

### 5.4 Semantic Feature Extraction

For each scenario, the ground-truth simulation state is used to compute a battery of semantic features characterising the relationship between each surrounding agent and the ego vehicle. These features are selected to span the principal dimensions of safety-relevant perception in the driving domain:

**Time-to-Collision (TTC)**: For each agent $i$ with current distance $d_i > 0$ to the ego vehicle and closing speed $\dot{d}_i > 0$ (i.e., approaching), the TTC is defined as:

$$
\mathrm{TTC}_i = \begin{cases} d_i / \dot{d}_i & \text{if } \dot{d}_i > \delta_{\mathrm{min}} \\ T_{\mathrm{horizon}} & \text{otherwise} \end{cases}
$$

where $\delta_{\mathrm{min}}$ is a minimum closing speed threshold to avoid numerical singularities, and $T_{\mathrm{horizon}}$ = 5.0 seconds is the maximum TTC value assigned to non-approaching agents. TTC values are clipped to $[0, T_{\mathrm{horizon}}]$. This feature encodes threat imminence: a head that preferentially attends to low-TTC agents is acting as a safety-critical threat detector.

**Euclidean Distance**: The planar Euclidean distance between agent $i$ and the ego vehicle:

$$
d_i = \sqrt{(\Delta x_i)^2 + (\Delta y_i)^2}
$$

where $(\Delta x_i, \Delta y_i)$ are the relative position components in the world frame. A head that systematically weights nearby agents more heavily is encoding proximity-sensitive behaviour.

**Closing Speed**: The signed rate of change of inter-agent distance, computed as the projection of the relative velocity onto the unit vector pointing from the ego to agent $i$:

$$
\dot{d}_i = -\frac{(\Delta x_i)\,(\Delta v^x_i) + (\Delta y_i)\,(\Delta v^y_i)}{d_i + \epsilon}
$$

where $(\Delta v^x_i, \Delta v^y_i) = (v^x_i - v^x_{\mathrm{ego}},\; v^y_i - v^y_{\mathrm{ego}})$ is the relative velocity and $\epsilon$ is a regularisation constant. Positive closing speed indicates approach; negative indicates recession. A head weighted towards agents with high closing speed encodes dynamic threat assessment.

**Spatial Position Predicates**: Binary indicators of agent position relative to the ego vehicle's heading-aligned coordinate frame. Let $(x_i^{\mathrm{local}}, y_i^{\mathrm{local}})$ denote the position of agent $i$ in the ego-centric frame, obtained by rotating the world-frame displacement by $-\psi_{\mathrm{ego}}$:

$$
\begin{pmatrix} x_i^{\mathrm{local}} \\ y_i^{\mathrm{local}} \end{pmatrix} = \begin{pmatrix} \cos(-\psi_{\mathrm{ego}}) & -\sin(-\psi_{\mathrm{ego}}) \\ \sin(-\psi_{\mathrm{ego}}) & \cos(-\psi_{\mathrm{ego}}) \end{pmatrix} \begin{pmatrix} \Delta x_i \\ \Delta y_i \end{pmatrix}
$$

The four spatial predicates are then:

$$
\mathrm{IsAhead}_i = \mathbb{1}[x_i^{\mathrm{local}} > \tau_{\mathrm{fwd}}], \quad \mathrm{IsBehind}_i = \mathbb{1}[x_i^{\mathrm{local}} < -\tau_{\mathrm{fwd}}]
$$

$$
\mathrm{IsLeft}_i = \mathbb{1}[y_i^{\mathrm{local}} > \tau_{\mathrm{lat}}], \quad \mathrm{IsRight}_i = \mathbb{1}[y_i^{\mathrm{local}} < -\tau_{\mathrm{lat}}]
$$

where $\tau_{\mathrm{fwd}}$ = 2.0m and $\tau_{\mathrm{lat}}$ = 1.0m are distance thresholds that define a dead zone around the ego vehicle, filtering out agents whose spatial classification is ambiguous due to their proximity. These binary features encode the spatial layout of the scene from the ego's perspective.

**Agent Speed**: The magnitude of the velocity vector of agent $i$:

$$
s_i = \sqrt{(v^x_i)^2 + (v^y_i)^2}
$$

Speed encodes the dynamic saliency of an agent: fast-moving agents are more dynamically challenging to predict and interact with. A head systematically attending to high-speed agents regardless of spatial position or TTC encodes dynamic activity rather than proximity or threat.

These $F = 8$ features—TTC, distance, closing speed, four spatial predicates, and speed—are collected for each of the $N_{\mathrm{agents}}$ surrounding agents per scenario, forming the semantic feature matrix $\mathbf{f}^{(n)} \in \mathbb{R}^{N_{\mathrm{agents}} \times F}$ for scenario $n$.

#### Treatment of Invalid Agents

A necessary preprocessing step accounts for the variable validity structure of the observation. In many scenarios, fewer than $N_{\mathrm{agents}}$ agents are present within the observable range. Tokens corresponding to absent agents are padded with zero features and a zero validity flag. Before any cross-agent comparison, all agent indices $i$ for which the validity indicator $\mathrm{valid}_i = 0$ are excluded. Only valid agent–feature pairs $(W^{\mathrm{attn}}_{h,i},\, f_i^{(n)})$ contribute to the analysis. Scenarios with fewer than $N_{\min}$ = 3 valid agents are excluded in their entirety, as per-vehicle comparisons over very small populations are insufficiently informative.

### 5.5 The Head Specialization Index

The Head Specialization Index (HSI) is the primary scalar measure of a head's functional specificity. It quantifies the degree to which a head's attention distribution systematically reflects a particular semantic property of the scene, selecting the strongest such alignment across all features in the battery $\mathcal{F}$:

$$
\mathrm{HSI}_h = \max_{f \in \mathcal{F}} \mathcal{A}(h, f)
$$

where $\mathcal{A}(h, f)$ is a measure of the statistical alignment between head $h$'s per-agent attention weights $\{W^{\mathrm{attn}}_{h,i}\}$ and the per-agent feature values $\{f_i\}$, pooled across the evaluation scenario set. The argmax over features identifies the *primary feature* of head $h$:

$$
f^*_h = \arg\max_{f \in \mathcal{F}} \mathcal{A}(h, f)
$$

The alignment measure $\mathcal{A}(h, f)$ is operationalised as the absolute Spearman rank correlation between the attention distribution of head $h$ and feature $f$, computed across all agent-scenario pairs in the evaluation corpus:

$$
\mathcal{A}(h, f) = \left| \rho_s\left( \mathbf{a}_h, \mathbf{f} \right) \right|
$$

where $\mathbf{a}_h \in \mathbb{R}^{S \cdot N_v}$ is the vector of aggregated per-agent attention weights for head $h$ concatenated across all $S$ evaluation scenarios, and $\mathbf{f} \in \mathbb{R}^{S \cdot N_v}$ is the corresponding vector of semantic feature values. The Spearman statistic is preferred over Pearson correlation because attention distributions are non-Gaussian and often skewed, and the semantic features span heterogeneous scales; rank-based correlation provides a reliable measure of ordinal association without distributional assumptions.

A head is classified as specialised if $\mathrm{HSI}h \geq \tau{\mathrm{HSI}} = 0.3$, indicating that the most strongly aligned feature exceeds the functional specificity threshold. The threshold $\tau_{\mathrm{HSI}}$ is set to balance sensitivity against specificity: too low a value admits weakly aligned heads into the specialised taxonomy, diluting its interpretability; too high a value excludes genuinely specialised heads exhibiting moderate but consistent alignment.

Clean and self-contained — one equation defining $\mathcal{A}$ as the absolute Spearman rank correlation, with a brief justification for the choice of statistic.

A head is classified as *specialised* if $\mathrm{HSI}_h \geq \tau_{\mathrm{HSI}}$ = 0.3, indicating that the most strongly aligned feature exceeds the functional specificity threshold. The threshold $\tau_{\mathrm{HSI}}$ is set to balance sensitivity against specificity: too low a value admits weakly aligned heads into the specialised taxonomy, diluting its interpretability; too high a value excludes genuinely specialised heads exhibiting moderate but consistent alignment.

### 5.6 Head Labeling Framework

The HSI and primary feature assignments form the basis for a *head function taxonomy* that maps each attention head to an interpretable semantic role. The taxonomy is defined by the set of identified feature alignments and their directionality—that is, whether high attention is associated with high or low values of the primary feature:

| Primary Feature     | Directional Alignment                | Functional Label      |
| ------------------- | ------------------------------------ | --------------------- |
| TTC                 | High attention → low TTC            | Threat-Detection Head |
| Distance            | High attention → small distance     | Proximity Head        |
| Closing Speed       | High attention → high closing speed | Approach-Rate Head    |
| Agent Speed         | Variable                             | Dynamic Object Head   |
| Is-Ahead            | High attention → forward position   | Forward-Lane Head     |
| Is-Left or Is-Right | High attention → lateral position   | Lateral-Zone Head     |

The directional alignment for each feature is derived from first principles of driving safety. A head attending to low-TTC vehicles is performing threat detection; a head sensitive to Is-Ahead systematically monitors vehicles in the forward driving corridor; a head tracking high closing speed is encoding approach dynamics irrespective of absolute distance. The directional specification provides a semantic anchor for each label assignment, enabling a sanity-check between the observed alignment direction and the expected functional behaviour.

Heads for which $\mathrm{HSI}_h < \tau_{\mathrm{HSI}}$ are assigned the label *General Context Head*, indicating diffuse or uninterpretable attention patterns. The presence of such heads in a trained policy is expected: not all $H$ heads need be specialised, and the architecture may maintain some fraction of general-purpose attention capacity for scenarios not well-captured by the defined feature battery.

This labeling framework constitutes a *post-hoc interpretability registry*: a structured summary of the policy's inferred attentional organisation, exportable as a persistent artefact for downstream use in safety auditing, policy debugging, and documentation.

---

## 6. Head Function Registry and Interpretability Artefacts

### 6.1 The Registry Data Structure

The outputs of the Phase I analysis are consolidated into a structured *head function registry*: a persistently stored mapping from head index to functional metadata. For each head $h \in \{0, \ldots, H-1\}$, the registry records:

- **Functional label**: the inferred semantic role (e.g., "Threat-Detection Head", "Proximity Head", "General Context Head"),
- **Functional description**: a natural language description of the head's inferred behaviour,
- **Specialization status**: a binary indicator (specialised / general),
- **HSI score**: the scalar $\mathrm{HSI}_h$,
- **Primary feature**: the feature $f^*_h$ with the strongest alignment,
- **Directional alignment**: whether high attention corresponds to high or low values of $f^*_h$,
- **Number of qualifying scenarios**: the number of scenarios contributing to the analysis.

The expected-alignment-direction field serves as a sanity check on the labeling logic: a head assigned the label "Threat-Detection Head" on the basis of its TTC alignment should exhibit high attention to low-TTC vehicles. A head with the opposite directionality would attend *away* from threats, constituting a counter-intuitive finding requiring additional investigation and potentially indicating a policy failure mode.

The registry is intended to be consumed by downstream XAI systems—for instance, a natural language explanation generator that uses the head registry to construct scenario-specific explanations of the form: "The policy is currently attending primarily to the vehicle at bearing 12 o'clock due to its low time-to-collision (Head $h$, Threat-Detection Head)."

### 6.2 Implications for Policy Transparency

The existence of identifiable, semantically coherent head specializations has direct implications for policy transparency. A policy whose attention is distributed uniformly across heads, each attending to different aspects of the scene without interpretable structure, is essentially opaque from the attention-analysis perspective—the attention weights convey information but without semantic anchor points for human interpretation.

By contrast, if the Phase I analysis confirms that distinct heads have adopted distinct functional roles, the attention weights become *structurally interpretable*: a practitioner can point to a specific head and state, with empirical backing, that this head's high attention to a particular vehicle indicates that the policy's situational awareness is primarily driven by proximity considerations, or by threat imminence, or by lateral zone monitoring. This level of semantic grounding is a prerequisite for any trustworthy explainability system in safety-critical deployment.

Furthermore, the head taxonomy provides a basis for *anomaly detection*: if, in a novel scenario, the Safety Head fails to attend to an agent with low TTC while the Proximity Head does, this misalignment between functional roles and scene configuration may indicate an out-of-distribution state warranting elevated uncertainty or human oversight. The head registry thus supports not only retrospective explanation but prospective risk signalling.

---

## 7. Results and Analysis

*This section will be populated with quantitative results following the completion of experimental runs. The analysis will include: (i) per-head HSI scores across the evaluation scenario set; (ii) the inferred head function taxonomy with directional alignment characterisations; (iii) the head function registry with functional labels and descriptions; and (iv) a discussion of the observed specialization patterns in relation to the theoretical hypotheses H1–H4 and the driving safety literature.*

---

## 8. Limitations, Discussion, and Relation to Phase II

### 8.1 Limitations of the Single-Snapshot Approach

The Phase I methodology, while tractable and interpretatively direct, is subject to several limitations that motivate the more sophisticated analyses of Phase II:

**Temporal blindness**: By extracting a single observation snapshot per scenario, the Phase I analysis captures only the average specialization structure across the static initial distribution of scenario configurations. It cannot distinguish between heads that are consistently specialised throughout an episode and heads that exhibit contextual specialization—activating their specialisation only when specific scene conditions arise (e.g., high risk, imminent intersection entry). The temporal resolution of within-episode attention dynamics is entirely lost.

**Scenario-level confounding**: The initial conditions of the scenario distribution may not uniformly sample the relevant feature space. Scenarios with many vehicles at moderate distance and moderate TTC are likely over-represented relative to high-risk near-miss configurations. This sampling bias may attenuate the observed alignment for safety-critical features (TTC, closing speed) relative to their true within-episode importance, potentially leading to underestimation of the Safety Head's functional specificity.

**Softmax diffusion**: The attention softmax is row-stochastic by construction, meaning that attention to one agent necessarily comes at the expense of attention to others. In scenarios dominated by a single high-salience agent, a head attending to this agent will incidentally exhibit alignment with multiple correlated features simultaneously—for instance, the closest agent often also has the lowest TTC and the highest closing speed in congested environments. This structural confounding between distance, TTC, and closing speed makes it difficult to cleanly attribute a head's primary function based on single-snapshot analysis alone.

**Absence of causal intervention**: The observational nature of Phase I means that it cannot rule out alternative explanations for observed alignments. The structural confounds discussed above cannot be fully separated without causal intervention—ablation of specific heads and measurement of consequent behaviour changes.

### 8.2 Motivation for Phase II

The limitations of Phase I motivate a complementary analysis strategy: rather than merely observing the emergent specialization in a trained policy, Phase II introduces explicit architectural mechanisms designed to *induce* and *amplify* head specialization through specialization-promoting regularisation terms added to the training objective. This shift from observational to interventional methodology enables several advances:

1. **Causal establishment**: By construction, a head trained under explicit TTC-specialization pressure processes TTC-relevant information, establishing a causal link that observational analysis cannot provide.
2. **Controlled comparison**: The performance of a baseline policy (Phase I) and a specialization-regularised policy (Phase II) can be compared on both task metrics and interpretability metrics, quantifying the tradeoff between enforced specialization and task performance.
3. **Specialization diversity control**: Regularisation objectives can be designed to maximise the *diversity* of head specializations, reducing the functional redundancy observed in Phase I and potentially improving the breadth of the policy's semantic coverage.
4. **Robustness to evaluation distribution shift**: A policy trained with explicit specialization incentives may exhibit more consistent specialization across diverse scenario types, including out-of-distribution configurations not well-represented in the Phase I evaluation set.

The transition from Phase I to Phase II thus represents a progression from descriptive interpretability (what functional structure has emerged?) to prescriptive interpretability (how can we engineer functional structure into the policy?), with the Phase I analysis providing both the empirical motivation and the evaluation baseline for Phase II.

### 8.3 Connections to the Broader XAI Literature

The head specialization analysis framework developed here connects to several broader themes in the XAI and mechanistic interpretability literature. The functional probing methodology relates to the *concept-based explanation* paradigm (Kim et al., 2018; Ghassemi et al., 2021), which seeks to explain neural network behaviour in terms of human-interpretable high-level concepts. The semantic feature battery used in Phase I (TTC, distance, closing speed, spatial position) constitutes precisely such a concept library, and the head-concept alignment structure provides a mapping from internal policy representations to human-intelligible concepts.

The connection to *safe RL* is also significant. Interpretability analyses in safety-critical DRL have been proposed as a mechanism for identifying policy failure modes before deployment (Amodei et al., 2016; Leike et al., 2017). The head function registry produced by Phase I provides a structured audit trail for safety evaluation: the absence of a TTC-aligned head, or the presence of a head attending *away* from low-TTC agents, would constitute evidence of a potential safety oversight warranting targeted stress-testing.

The circuit-identification perspective of Conmy et al. (2023) and Elhage et al. (2021) provides the deepest theoretical framing: the head specialization analysis identifies which components of the encoder carry which safety-relevant information, constituting the first level of a three-level mechanistic explanation. The second level—characterising how the policy network *uses* the information encoded by each specialised head—and the third level—tracing the full computational circuit from observation to action—are the subjects of ongoing investigation beyond the scope of the present work.

---

## Appendix: Mathematical Notation Summary

| Symbol                           | Definition                                                                                                     |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| $H$                            | Number of attention heads per cross-attention module                                                           |
| $N_{\mathrm{agents}}$          | Number of surrounding agents in the observation                                                                |
| $T_{\mathrm{obs}}$             | Number of observation timesteps in the historical window                                                       |
| $N_{\mathrm{latent}}$          | Number of latent queries in the encoder                                                                        |
| $d_{\mathrm{latent}}$          | Dimensionality of the latent space                                                                             |
| $d_k$                          | Per-head key/query dimension ($= d_{\mathrm{latent}} / H$)                                                   |
| $\mathbf{A}^{\mathrm{agents}}$ | Raw attention weight tensor, shape$(N_{\mathrm{latent}},\; N_{\mathrm{agents}} \cdot T_{\mathrm{obs}},\; H)$ |
| $\tilde{\mathbf{A}}$           | Temporally aggregated attention, shape$(N_{\mathrm{latent}},\; N_{\mathrm{agents}},\; H)$                    |
| $\hat{\mathbf{A}}$             | Latent-aggregated attention, shape$(N_{\mathrm{agents}},\; H)$                                               |
| $\mathbf{W}^{\mathrm{attn}}$   | Analysis-ready per-head per-agent attention, shape$(H,\; N_{\mathrm{agents}})$                               |
| $\mathcal{A}(h, f)$            | Alignment measure between head$h$ and feature $f$                                                          |
| $\mathrm{HSI}_h$               | Head Specialization Index for head$h$                                                                        |
| $\tau_{\mathrm{HSI}}$          | Specialization threshold                                                                                       |
| $f^*_h$                        | Primary feature of head$h$                                                                                   |
| $\mathcal{F}$                  | Feature battery:$\{$TTC, distance, closing speed, is-ahead, is-behind, is-left, is-right, speed$\}$        |
| $N_{\mathrm{scenarios}}$       | Number of evaluation scenarios                                                                                 |
| $T_{\mathrm{horizon}}$         | Maximum TTC horizon for capping                                                                                |
| $\mathbf{f}^{(n)}$             | Semantic feature matrix for scenario$n$, shape $(N_{\mathrm{agents}},\; F)$                                |
