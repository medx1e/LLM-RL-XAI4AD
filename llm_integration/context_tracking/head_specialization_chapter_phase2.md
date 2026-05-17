# Phase II — Forced Head Specialization

## Overview and Motivation

Phase I established a passive diagnostic framework: given a trained policy, it measures the degree to which attention heads have naturally organized themselves around semantically interpretable features of the driving scene. The findings from that analysis motivate a fundamental question — if the emergent specialisation is incomplete, inconsistent, or absent for safety-critical features, can the training process itself be guided to produce more interpretable and functionally distinct head behaviour?

Phase II addresses this question through **forced specialisation**: the direct augmentation of the policy optimisation objective with auxiliary losses that explicitly encourage (i) cross-head diversity and (ii) risk-gated alignment of a designated safety head with urgency-weighted target distributions. These two terms are added to the standard Soft Actor-Critic (SAC) actor loss without modifying the architecture of the policy network, preserving the expressive capacity of the Wayformer encoder while shaping the geometry of its attention distributions during learning.

The analysis component of Phase II is correspondingly richer. Because the policy is now trained with structural priors over attention, the post-hoc interpretability analysis must move beyond single-snapshot, cross-scenario Spearman correlation (as used in Phase I) to a longitudinal, within-module distributional methodology that captures both the **temporal consistency** of specialisation across the episode horizon and its **risk-contingent modulation**. This requires multi-timestep expert rollout extraction, per-modality attention distribution tracking, and Fisher $z$-transform meta-analysis to aggregate correlation statistics in a statistically principled manner across heterogeneous episodes.

---

## 5.X The Forced Specialisation Training Objective

### 5.X.1 Augmented SAC Actor Loss

The baseline policy is trained using Soft Actor-Critic, whose actor loss minimises the expected KL divergence between the current policy and the Boltzmann distribution induced by the soft Q-function:

$$
\mathcal{L}_{\text{SAC}} = \mathbb{E}_{s_t \sim \mathcal{D},\, a_t \sim \pi} \left[ \alpha \log \pi(a_t \mid s_t) - \min_{k=1,2} Q_k(s_t, a_t) \right]
$$

where $\alpha$ is the entropy temperature coefficient, $\mathcal{D}$ is the replay buffer, and the twin Q-functions $Q_1, Q_2$ are used to mitigate overestimation bias. This objective, in isolation, places no constraint on the structure of the internal representations computed by the encoder — the attention heads are free to develop arbitrary, potentially redundant or uninterpretable allocation patterns, as long as the policy achieves high expected return.

The augmented actor loss introduces three additional terms that impose structural priors on the attention distributions produced by the Wayformer encoder:

$$
\mathcal{L}_{\text{actor}} = \mathcal{L}_{\text{SAC}} + \lambda_{\text{div}} \cdot \mathcal{L}_{\text{diversity}} + \lambda_{\text{safe}} \cdot \mathcal{L}_{\text{safety}}
$$

where $\lambda_{\text{div}}$ and $\lambda_{\text{safe}}$ are non-negative scalar coefficients controlling the relative weight of each regularisation term. The safety term is additionally **risk-gated**: it is activated only when the estimated scene-level collision risk $R$ exceeds a threshold $\tau_R = 0.2$, concentrating its gradient signal on the high-stakes subset of training transitions:

$$
\mathcal{L}_{\text{safety}} = \mathbf{1}\left[R > \tau_R\right] \cdot \mathcal{L}_{\text{safety\_raw}}
$$

This gating mechanism is motivated by the asymmetry of safety-critical behaviour in autonomous driving: the majority of timesteps involve low-risk free-flow conditions for which unconstrained attention is perfectly adequate, and applying a strong safety bias uniformly would interfere with the policy's ability to represent routine navigational reasoning. Restricting the safety loss to high-risk transitions ensures that the designated safety head develops urgency-sensitivity without distorting the encoder's general representational capacity.

The two coefficients are set to $\lambda_{\text{div}} = 0.05$ and $\lambda_{\text{safe}} = 0.05$.

---

### 5.X.2 Cross-Head Diversity Loss

The diversity loss penalises redundancy across heads by discouraging pairs of heads from assigning similar attention distributions to the same set of input tokens. Formally, for each pair of distinct heads $(i, j)$ with $i < j$, the pairwise diversity penalty is computed as the mean absolute cosine similarity between their attention weight vectors, averaged over all latent queries $q \in \{1, \ldots, Q\}$ and all elements of the training batch $b \in \{1, \ldots, B\}$:

$$
\mathcal{L}_{\text{div\_pair}}(i, j) = \frac{1}{B \cdot Q} \sum_{b=1}^{B} \sum_{q=1}^{Q} \left| \frac{\mathbf{a}_i^{(b,q)} \cdot \mathbf{a}_j^{(b,q)}}{\|\mathbf{a}_i^{(b,q)}\|_2 \cdot \|\mathbf{a}_j^{(b,q)}\|_2 + \epsilon} \right|
$$

where $\mathbf{a}_h^{(b,q)} \in \mathbb{R}^K$ is the softmax-normalised attention weight vector assigned by head $h$ for latent query $q$ in batch element $b$, over all $K$ input tokens in the relevant module. The aggregate diversity loss averages uniformly over all $\binom{H}{2}$ unique head pairs:

$$
\mathcal{L}_{\text{diversity}} = \frac{2}{H(H-1)} \sum_{i=1}^{H-1} \sum_{j=i+1}^{H} \mathcal{L}_{\text{div\_pair}}(i, j)
$$

Minimising this term drives the per-head attention distributions toward mutual orthogonality in the simplex over tokens. A value of $\mathcal{L}_{\text{diversity}} = 0$ would correspond to all head pairs exhibiting exactly zero cosine similarity — a perfectly diverse configuration in which each head focuses exclusively on a non-overlapping partition of the token space. In practice, the loss provides a soft gradient pressure toward this ideal rather than a hard constraint, allowing the policy to trade off diversity against task performance.

The absolute value in the pairwise term is important: it penalises both positive alignment (heads attending to the same tokens in the same proportion) and negative alignment (heads attending to the same tokens in systematically inverse proportions), since both imply redundant encoding of the same relational structure.

---

### 5.X.3 Risk-Gated Safety Loss

The safety loss is the most semantically targeted of the three regularisation terms. It directly supervises a designated safety head $h_{\text{safety}}$ to concentrate its attention on the most urgently threatening traffic agents, as operationalised by their time-to-collision with the ego vehicle.

**Scene-level collision risk.** The risk scalar $R \in [0, 1]$ is computed from the minimum TTC among all valid non-ego agents in the current observation:

$$
R = \text{clip}\!\left(1 - \frac{\min_{v \in \mathcal{V}_{\text{valid}}} \text{TTC}_v}{3.0},\ 0.0,\ 1.0\right)
$$

$R = 0$ when all agents have TTC $\geq 3.0$ s (safe scene); $R = 1$ when at least one agent is on an immediate collision course (TTC $\approx 0$). The safety loss is activated only when $R > \tau_R = 0.2$, corresponding to a minimum TTC below $2.4$ s.

**Exponential TTC target distribution.** For each vehicle $v$, an unnormalised target attention weight is computed using an exponential urgency function parameterised by a temperature $\tau_{\text{safety}}$:

$$
w_v = \exp\!\left(-\frac{\text{TTC}_v}{\tau_{\text{safety}}}\right)
$$

Lower TTC (higher urgency) yields exponentially higher target weight. Padding or invalid agents are assigned $w_v = 0$. Since each vehicle $v$ is represented by $N_{\text{obs}}$ temporal tokens in the agent module (one per past observation timestep), all $N_{\text{obs}}$ tokens belonging to vehicle $v$ are assigned the same scalar weight $w_v$. The full weight vector over all agent tokens is then $\ell^1$-normalised to form a valid probability distribution $P_{\text{target}}^{(b)}$.

**KL divergence penalty.** The safety loss penalises the KL divergence from the target distribution to the actual attention distribution of the safety head, averaged over all queries and batch elements:

$$
\mathcal{L}_{\text{safety\_raw}} = \frac{1}{B \cdot Q} \sum_{b=1}^{B} \sum_{q=1}^{Q} \sum_{k} P_{\text{target}}^{(b)}(k) \log \frac{P_{\text{target}}^{(b)}(k) + \epsilon}{P_{\text{actual}}^{(b,q)}(k) + \epsilon}
$$

where $P_{\text{actual}}^{(b,q)}(k)$ is the softmax attention weight assigned to token $k$ by query $q$ of head $h_{\text{safety}}$ in batch element $b$. Note that this is $\text{KL}(P_{\text{target}} \| P_{\text{actual}})$, which is minimised when the policy's safety head exactly matches the exponential TTC target. This asymmetric form is chosen deliberately: it imposes a hard requirement that $P_{\text{actual}}$ places non-negligible mass wherever $P_{\text{target}}$ does, preventing the policy from ignoring any urgency-highlighted agent.

The temperature $\tau_{\text{safety}} = 0.5$ controls the sharpness of the target distribution: lower values concentrate the target mass on the single closest-to-collision agent; higher values spread it more diffusely across all agents with below-average TTC.

---

## 5.X Within-Module Distributional Analysis

### 5.X.1 Multi-Timestep Expert Rollout Extraction

The interpretability analysis adopts a **multi-timestep expert rollout** extraction protocol designed to capture the temporal dynamics of attention behaviour within individual driving episodes. For each scenario, the environment is reset and the expert log-replay trajectory is followed for up to $T_{\max} = 80$ simulation timesteps, stepping the scene forward using the recorded ground-truth human driver actions at each step. At every timestep $t$, the encoder is applied to the current observation using the trained policy weights, and the full set of per-modality attention distributions is extracted. This yields, for each scenario $s$, a temporally aligned tuple:

$$
\left\{ \mathbf{A}^{(s,t)},\ \mathbf{f}^{(s,t)},\ R^{(s,t)} \right\}_{t=1}^{T^{(s)}}
$$

where $\mathbf{A}^{(s,t)}$ collects the per-modality attention distributions, $\mathbf{f}^{(s,t)}$ is the semantic feature vector, and $R^{(s,t)}$ is the scene-level collision risk at timestep $t$.

Crucially, the expert log-replay does **not** use the trained policy for action selection — it replays the logged human actions verbatim. The encoder is applied in forward-pass-only mode to compute attention patterns, which reflect the learned model's perceptual interpretation of the scene state rather than any closed-loop behavioural commitment. This separation isolates the representational content of the encoder from confounding effects of distributional shift introduced by the policy's own closed-loop behaviour.

---

### 5.X.2 Per-Modality Attention Distributions

The Wayformer encoder employs a late-fusion, per-modality cross-attention architecture in which each input modality maintains a dedicated attention module. Five modalities are present in the observation space, each contributing a distinct set of tokens to the encoder's cross-attention operations:

**Self-Driving Controller Trajectory ($\texttt{sdc\_traj}$).** This modality encodes the ego vehicle's own past trajectory over $N_{\text{obs}}$ observation timesteps. The per-modality attention distribution over these tokens reveals how much the encoder weighs recent versus older ego states — a quantity directly interpretable as a recency bias. The scalar summary statistic extracted at each timestep is the **centre-of-mass** of the attention distribution over the temporal axis:

$$
\text{Recency}_h^{(s,t)} = \sum_{\tau=1}^{N_{\text{obs}}} \tau \cdot a_{h,\tau}^{\text{sdc}(s,t)}
$$

where $a_{h,\tau}^{\text{sdc}(s,t)}$ is the normalised attention weight assigned by head $h$ to the $\tau$-th ego observation token, with $\tau = N_{\text{obs}}$ corresponding to the most recent timestep.

**Surrounding Agent Trajectories ($\texttt{other\_traj}$).** This modality encodes the past trajectories of the $N_v$ closest surrounding traffic agents over the same $N_{\text{obs}}$ observation window. To extract per-agent attention, the token axis is reshaped into a $(N_v, N_{\text{obs}})$ grid and summed over the temporal axis, then averaged over latent queries:

$$
a_{h,v}^{(s,t)} = \frac{1}{N_L} \sum_{l=1}^{N_L} \sum_{\tau=1}^{N_{\text{obs}}} \mathbf{A}_h^{(s,t)}\!\left[l,\ v \cdot N_{\text{obs}} + \tau\right]
$$

where $N_L$ is the number of latent queries. The resulting vector $\mathbf{a}_h^{(s,t)} = (a_{h,1}^{(s,t)}, \ldots, a_{h,N_v}^{(s,t)}) \in \mathbb{R}^{N_v}$ is normalised to sum to 1 across vehicles, yielding the fractional attention allocation per vehicle for head $h$ at timestep $t$. This is the primary quantity used in the cross-vehicle correlation analysis.

**Road Graph ($\texttt{roadgraph}$).** This modality encodes the top-$K_{\text{rg}}$ road graph elements (lane segments, road edges, crossings) nearest to the ego vehicle. Because the semantic identity of individual road graph elements is not tracked across scenarios, per-token attention is not directly interpretable as a per-entity measure. Instead, two distributional summary statistics are computed per head per timestep:

The **attention entropy**, measuring how diffusely the head distributes attention across road graph elements:

$$
H_h^{\text{rg}(s,t)} = -\sum_{k=1}^{K_{\text{rg}}} p_{h,k}^{\text{rg}(s,t)} \log p_{h,k}^{\text{rg}(s,t)}
$$

where $p_{h,k}^{\text{rg}(s,t)}$ is the normalised attention fraction for road graph token $k$.

The **attention concentration**, defined as $1$ minus the entropy normalised by its theoretical maximum:

$$
C_h^{\text{rg}(s,t)} = 1 - \frac{H_h^{\text{rg}(s,t)}}{\log K_{\text{rg}}}
$$

$C_h^{\text{rg}} = 1$ corresponds to the head focusing all attention on a single road graph element; $C_h^{\text{rg}} = 0$ corresponds to uniform diffuse attention. These two quantities together characterise whether the head has adopted a focal versus scanning strategy for processing road structure.

**Traffic Lights ($\texttt{traffic\_lights}$).** This modality encodes the $N_{\text{tl}}$ closest traffic signals over the observation window. Analogously to the agent module, the temporal axis is marginalised to obtain per-signal attention fractions:

$$
a_{h,l}^{\text{tl}(s,t)} \propto \sum_{\tau=1}^{N_{\text{obs}}} \mathbf{A}_h^{\text{tl}(s,t)}\!\left[l,\tau\right]
$$

The per-timestep summary statistic used in the analysis is the maximum signal fraction $\max_l a_{h,l}^{\text{tl}(s,t)}$, which measures whether the head concentrates on a single dominant signal or spreads its attention across all observed lights.

**GPS Waypoints ($\texttt{gps\_path}$).** This modality encodes a short sequence of $N_{\text{wp}}$ target waypoints defining the planned route. The tokens are ordered by proximity: token 1 corresponds to the nearest upcoming waypoint and token $N_{\text{wp}}$ to the most distant. The scalar summary statistic is the **attention centre-of-mass** along this ordered waypoint sequence:

$$
\text{CoM}_h^{\text{gps}(s,t)} = \sum_{k=1}^{N_{\text{wp}}} k \cdot a_{h,k}^{\text{gps}(s,t)}
$$

A low centre-of-mass indicates that the head focuses on near-term waypoints; a high value indicates attention to longer-horizon navigation targets. The modulation of this quantity under rising collision risk is of particular interpretive interest: a head that shifts its GPS attention toward nearer waypoints when danger rises can be interpreted as prioritising immediate trajectory correction over long-horizon route following — a behaviourally coherent risk-adaptive response.

---

## 5.X Within-Module HSI with Fisher Z Meta-Analysis

### 5.X.1 Cross-Vehicle Spearman Correlation Timeseries

For the agent modality, the interpretability probe is extended from the single-snapshot setting of Phase I to a longitudinal analysis. At each timestep $t$ within episode $s$, the Spearman rank correlation between the per-vehicle attention vector $\mathbf{a}_h^{(s,t)} \in \mathbb{R}^{N_v}$ and a semantic feature vector $\mathbf{f}^{(s,t)} \in \mathbb{R}^{N_v}$ is computed across the set of valid agents $\mathcal{V}_{\text{valid}}^{(s,t)}$:

$$
\rho_{h,f}^{(s,t)} = \text{Spearman}\!\left(\mathbf{a}_h^{(s,t)}\big|_{\mathcal{V}_{\text{valid}}},\ \mathbf{f}^{(s,t)}\big|_{\mathcal{V}_{\text{valid}}}\right)
$$

A timestep is included only if $|\mathcal{V}_{\text{valid}}^{(s,t)}| \geq N_{\min} = 3$ and the feature vector exhibits non-trivial variation ($\text{std}(\mathbf{f}^{(s,t)}) > 10^{-10}$). This yields a timeseries $\{\rho_{h,f}^{(s,t)}\}_{t \in \mathcal{T}_{\text{valid}}^{(s)}}$ of within-timestep cross-vehicle correlations.

---

### 5.X.2 Consistent Selectivity via Fisher Z Aggregation

The central question is: *does head $h$ consistently apply feature $f$ as an attention criterion throughout an episode, and is this consistency reproducible across episodes?*

**Within-episode aggregation.** For each episode $s$, the per-timestep correlation values $\{\rho_{h,f}^{(s,t)}\}$ are aggregated into a single episode-level summary using the Fisher $z$-transform. Direct arithmetic averaging of Spearman $\rho$ values is statistically inappropriate: the sampling distribution of $\hat{\rho}$ is asymmetric and bounded in $[-1,1]$, with variance that depends on the true population correlation. The Fisher $z$-transform maps each $\rho$ to an approximately normally distributed variable with constant variance:

$$
z_{h,f}^{(s,t)} = \text{arctanh}\!\left(\text{clip}\!\left(\rho_{h,f}^{(s,t)},\ -0.999,\ 0.999\right)\right)
$$

The within-episode mean in the $z$-domain is:

$$
\bar{z}_{h,f}^{(s)} = \frac{1}{|\mathcal{T}_{\text{valid}}^{(s)}|} \sum_{t \in \mathcal{T}_{\text{valid}}^{(s)}} z_{h,f}^{(s,t)}
$$

and the episode-level correlation is recovered by the inverse transform:

$$
\bar{\rho}_{h,f}^{(s)} = \tanh\!\left(\bar{z}_{h,f}^{(s)}\right)
$$

Episodes contributing fewer than $T_{\min} = 10$ valid timesteps are excluded from the analysis.

**Cross-episode aggregation.** The per-episode correlations $\{\bar{\rho}_{h,f}^{(s)}\}_{s \in \mathcal{S}_{h,f}}$, where $\mathcal{S}_{h,f}$ is the set of qualifying episodes for the $(h, f)$ pair, are themselves aggregated via a second application of the Fisher $z$-transform:

$$
\bar{z}_{h,f} = \frac{1}{|\mathcal{S}_{h,f}|} \sum_{s \in \mathcal{S}_{h,f}} \text{arctanh}\!\left(\text{clip}\!\left(\bar{\rho}_{h,f}^{(s)},\ -0.999,\ 0.999\right)\right)
$$

$$
\bar{\rho}_{h,f} = \tanh\!\left(\bar{z}_{h,f}\right)
$$

This two-level Fisher $z$ aggregation — first within episodes over timesteps, then across episodes — correctly treats each episode as an independent statistical unit and propagates correlation estimates through both levels of averaging without bias. A 95% confidence interval on $\bar{\rho}_{h,f}$ is computed from the standard error of the mean in the $z$-domain:

$$
\text{SE}_{h,f} = \frac{\hat{\sigma}\!\left(\left\{\text{arctanh}(\bar{\rho}_{h,f}^{(s)})\right\}\right)}{\sqrt{|\mathcal{S}_{h,f}|}}
$$

$$
\text{CI}_{95}(\bar{\rho}_{h,f}) = \left[\tanh\!\left(\bar{z}_{h,f} - 1.96 \cdot \text{SE}_{h,f}\right),\ \tanh\!\left(\bar{z}_{h,f} + 1.96 \cdot \text{SE}_{h,f}\right)\right]
$$

The HSI score for head $h$ is then:

$$
\text{HSI}_h = \max_{f \in \mathcal{F}} \left|\bar{\rho}_{h,f}\right|
$$

with the primary feature identified as the argmax. A head is designated as specialised if $\text{HSI}_h \geq \tau_{\text{HSI}} = 0.3$ and the 95% CI on its primary correlation excludes zero.

---

## 5.X Experimental Methodology

### 5.X.1 Training Environment and Dataset

The forced specialisation experiments are conducted in the Waymax simulation environment, using real-world driving scenarios sampled from the Waymo Open Dataset. The dataset provides high-fidelity reconstructions of urban and suburban driving scenes with rich agent interactions, road graph detail, and traffic control information. Scenarios are sampled from a $10$ GB subset of the Waymo Open Motion Dataset training split, comprising $5{,}319$ unique driving scenarios spanning a diverse range of interaction types including merging, yielding, intersection negotiation, and emergency avoidance.

The expert log-replay protocol used for both training stepping and post-hoc extraction preserves the ground-truth trajectory of all agents in the scene, including background traffic. The ego vehicle's actions during training are produced by the SAC policy; during post-hoc extraction, all agents follow their logged ground-truth trajectories. The observation modality is the vectorised feature representation, which encodes each entity's state history as a fixed-length feature vector rather than a rasterised image.

### 5.X.2 Network Architecture

The policy network consists of the Wayformer encoder followed by separate policy and value MLP heads. The Wayformer encoder applies per-modality cross-attention with $H = 4$ heads and a latent bottleneck of $N_L = 16$ queries. The cross-attention depth per modality is $2$ layers. The latent representations from all modalities are concatenated after their respective cross-attention modules and passed to the policy head.

### 5.X.3 Forced Specialisation Hyperparameters

The forced specialisation terms are controlled by the following hyperparameters:

| Parameter | Value |
|---|---|
| $\lambda_{\text{div}}$ | $0.05$ |
| $\lambda_{\text{safe}}$ | $0.05$ |
| $\tau_R$ (risk gate threshold) | $0.2$ |
| $\tau_{\text{safety}}$ (TTC temperature) | $0.5$ |
| Safety head index $h_{\text{safety}}$ | $0$ |

### 5.X.4 Training Configuration

The SAC agent is trained for $2 \times 10^{6}$ environment steps with a replay buffer of capacity $100{,}000$. The actor and critic networks are updated using the Adam optimiser with learning rate $3 \times 10^{-4}$. The entropy temperature $\alpha$ is automatically tuned via dual gradient descent.

### 5.X.5 Evaluation and Extraction Protocol

Following training, the post-hoc interpretability analysis is conducted on $50$ scenarios drawn from the held-out evaluation split of the dataset. For each scenario, the multi-timestep expert rollout extraction protocol (Section 5.X.1) is applied, yielding temporally aligned attention distributions, semantic features, and risk timeseries for each of the five modalities. The within-module HSI analysis is then computed over this evaluation corpus.

The minimum qualifying episode count for Fisher $z$ meta-analysis is $S_{\min} = 3$; the minimum valid timestep count per episode is $T_{\min} = 10$; the minimum valid vehicle count per timestep is $N_{\min} = 3$. The HSI specialisation threshold is $\tau_{\text{HSI}} = 0.3$.

---

## 8. Results and Analysis

*[TO BE COMPLETED — awaiting experimental results.]*
