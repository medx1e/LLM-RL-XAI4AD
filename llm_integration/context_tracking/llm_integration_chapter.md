# The LLM Integration Layer

---

## 1. Introduction

This chapter presents a principled framework for exposing the latent reasoning of such a policy to human scrutiny through the integration of a Large Language Model (LLM) as a semantic interpretability layer. The framework operates not by modifying the underlying policy — whose weights remain fixed — but by constructing, in parallel with policy execution, a structured explanatory substrate from which natural-language accounts of decision-making can be synthesized. The LLM functions neither as a planning agent nor as a model surrogate; rather, it serves as a *grounded narration module* — a system that translates formally verified computational evidence about policy behavior into coherent, causally structured natural-language explanations.

The architectural proposition is that faithfulness in autonomous driving explainability requires a strict separation between three concerns: (i) the extraction of interpretable signals from the policy's internal computation, (ii) the organization of those signals into a causal narrative structure, and (iii) the linguistic rendering of that structure as human-comprehensible explanation. Conflating these concerns — as approaches that prompt LLMs directly over raw observations tend to do — invites hallucination, unfaithfulness, and a systematic conflation of the model's actual reasoning with post-hoc rationalization.

---

## 2. Theoretical Background


### The Faithfulness Problem in LLM-Based Explanation

The integration of LLMs into XAI pipelines raises a fundamental tension. LLMs trained on large corpora of natural language encode rich commonsense and causal reasoning capabilities, but they are also prone to generating plausible-sounding text that has no grounding in the actual computational evidence available. This *hallucination* problem is especially dangerous in safety-critical domains, where an explanation that sounds convincing but misrepresents the agent's actual reasoning may generate false trust or obscure genuine failure modes.

A rigorous LLM integration architecture must therefore enforce *faithfulness constraints* at the structural level — ensuring that the language model is not queried with open-ended access to raw observations, but with a precisely delimited, formally verified evidence package that it is constrained to interpret, not supplement. This is the fundamental design principle motivating the framework developed in this chapter.

### 2.2 Counterfactual Explanation Theory

Counterfactual explanations have a well-established theoretical basis in the causal inference literature. A counterfactual explanation for a decision $a^*$ given context $s$ takes the form: *had the agent executed action $a'$ instead of $a^*$, the outcome would have been $o'$*. In the context of autonomous driving, the causal question of interest is: *which elements of the scene make the chosen action necessary?*

Formally, let $\mathcal{A}_{\text{grid}} = \{a_1, a_2, \ldots, a_K\}$ be a finite set of alternative actions, and let $\text{Rollout}(s_t, a_k, \pi_{\text{sim}}, H)$ denote the $H$-step simulated trajectory resulting from applying action $a_k$ at state $s_t$ and thereafter following a reference policy $\pi_{\text{sim}}$. An outcome function $\Omega : \mathcal{T} \to \{\text{SAFE}, \text{COLLISION}, \text{OFFROAD}\}$ evaluates each trajectory $\mathcal{T}$ for safety violations.

The *necessity score* of the actual decision is then:

$$
\eta(s_t, a^*) = \frac{|\{a_k \in \mathcal{A}_{\text{grid}} : \Omega(\text{Rollout}(s_t, a_k)) \neq \text{SAFE}\}|}{|\mathcal{A}_{\text{grid}}|}
$$

A high necessity score $\eta \approx 1$ implies that virtually all alternatives result in unsafe outcomes, providing rigorous counterfactual justification for the chosen action. Conversely, $\eta \approx 0$ characterizes routine decisions where many alternatives are equally safe, reducing the urgency of explanation.

The threat agents identified through this process — those whose presence causes alternative trajectories to fail — provide a *semantically grounded* reference set against which the policy's attention allocation can be tested.

### 2.3 Attention Grounding

Given threat agents $\mathcal{T}_{\text{threat}} = \{i : \exists a_k, \Omega(\text{Rollout}(s_t, a_k)) = \text{COLLISION} \wedge \text{agent}_i \text{ implicated}\}$, the *attention grounding score* measures the degree to which the policy's cross-attention was concentrated on these causally relevant entities at the time of the decision:

$$
G(s_t, a^*) = \sum_{i \in \mathcal{T}_{\text{threat}}} \sigma_i \cdot m_i
$$

where $m_i = \sum_h \sum_l \alpha_{h,l,\tau(i)}$ is the aggregated attention mass over heads $h$ and latent queries $l$ for the token slot $\tau(i)$ corresponding to agent $i$, and $\sigma_i = \frac{1}{1 + \text{TTC}_i}$ is a severity weight inversely proportional to the time-to-collision with threat agent $i$. The function $\tau : \mathcal{I} \to \{1, \ldots, T_{\text{obj}}\}$ maps simulator agent indices to encoder token positions via a distance-sorted ordering that replicates the feature extractor's input assembly — a non-trivial correspondence requiring explicit reconstruction at inference time.

A high grounding score $G \approx 1$ constitutes evidence that the policy's internal attention mechanism was indeed focused on the agents whose presence justified the chosen action, providing a form of *mechanistic consistency* between the counterfactual explanation and the neural computation.

---

## 4. System Architecture of the LLM Integration Layer

### 4.1 Architectural Overview

The LLM integration layer is positioned as the terminal stage of a multi-module interpretability pipeline that operates in parallel with the policy execution loop. The pipeline can be conceptualized as a directed acyclic computation graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ in which each node $v \in \mathcal{V}$ represents a computational module and each edge $e \in \mathcal{E}$ represents a structured data flow. The LLM module occupies the final node of this graph, receiving as input the aggregated output of all preceding modules and producing as output a natural-language explanation.

The pipeline proceeds through the following conceptually distinct stages:

1. **Semantic Graph Construction** — Raw simulator state is converted into a structured semantic graph $\mathcal{G}_{\text{scene}}$ encoding spatial relationships, kinematic properties, and contextual categories among agents.
2. **Adaptive Counterfactual Generation** — A context-sensitive action grid $\mathcal{A}_{\text{grid}}$ is constructed based on the detected scene context, and counterfactual rollouts are executed to evaluate the outcomes of alternative actions.
3. **Necessity Quantification** — The necessity score $\eta$ is computed from counterfactual outcomes, and threat agents are identified with their associated time-to-collision estimates.
4. **Attention Grounding** — Cross-attention weights extracted from the policy's encoder are aligned with threat agents via a distance-sorted token mapping, and the grounding score $G$ is computed.
5. **Decision Classification** — The decision is classified into one of four epistemic categories based on the joint values of $\eta$ and $G$.
6. **Structured Report Assembly** — All computed values are assembled into a formally validated, schema-compliant evidence package.
7. **Semantic Routing** — The decision classification determines which explanation template is appropriate, controlling the depth and caveat structure of the generated explanation.
8. **Prompt Construction** — Template-specific prompts are constructed from the structured report, enforcing faithfulness constraints at the syntactic level.
9. **LLM Narration** — The language model generates a natural-language explanation conditioned on the constructed prompts.

### 4.2 The Computation-Narration Interface

A key architectural feature is the strict separation between the computation layer (modules 1–6) and the narration layer (modules 7–9). The interface between these layers is a formally typed structured report $\mathcal{R}$ that functions as the sole channel of information from computation to narration. This interface enforces two critical properties:

- **Completeness**: $\mathcal{R}$ contains all information that the LLM is permitted to use in generating the explanation. Nothing from the raw policy internals or simulator state is accessible to the narration layer beyond what has been explicitly validated and included in $\mathcal{R}$.
- **Traceability**: Every claim that can appear in a generated explanation corresponds to a specific field in $\mathcal{R}$ that was computed by a specific upstream module. This enables systematic auditing of explanation faithfulness.

The structured report $\mathcal{R}$ encodes, at minimum: the ego vehicle's kinematic state, the chosen action with its continuous parameterization, the scene's active context categories, a proximity-ranked description of nearby agents with their semantic relations, the ranked list of counterfactual alternatives with outcomes, the necessity score $\eta$, the attention grounding structure $G$ with per-agent breakdown, and the decision classification label.

---

## 5. Semantic Explanation Generation

### 5.1 Semantic Graph Construction

The transformation of raw simulator state — comprising numerical arrays of agent positions, velocities, and orientations — into semantically interpretable scene descriptions is accomplished through a relational graph construction procedure. For each non-ego agent $j$ within the operational domain, a semantic edge $e_{0j}$ is constructed encoding:

- **Spatial relations**: $\{$`ahead_of`, `behind_of`, `left_of`, `right_of`, `in_ego_lane`$\}$ derived from ego-frame coordinate transformation,
- **Kinematic relations**: $\{$`approaching`, `moving_away`$\}$ derived from the signed closing speed $\dot{d}_{0j} = -\frac{\langle \mathbf{p}_j - \mathbf{p}_0, \mathbf{v}_j - \mathbf{v}_0 \rangle}{\|\mathbf{p}_j - \mathbf{p}_0\|}$,
- **Proximity buckets**: $\{$`very_close`, `close`, `medium`, `far`$\}$ based on Euclidean distance,
- **Temporal collision risk**: $\{$`ttc_imminent`, `ttc_soon`, `ttc_later`, `ttc_far`$\}$ from the time-to-collision estimate $\hat{\tau}_{0j} = \|\mathbf{p}_j - \mathbf{p}_0\| / \max(\dot{d}_{0j}, \epsilon)$ when closing.

The resulting semantic graph $\mathcal{G}_{\text{scene}}$ provides a structured, human-interpretable representation of the scene that is both compatible with downstream quantitative analysis and with natural-language generation. Context categories are inferred from the graph structure through logical rule evaluation over edge relation sets, producing a scene-level classification that drives adaptive action grid construction.

### 5.2 Context-Conditioned Action Space

A crucial insight motivating adaptive action grid construction is that the space of *relevant* counterfactual alternatives is context-dependent. In a free-flow driving scenario, braking maneuvers are rarely informative counterfactuals — the interesting perturbations concern lane changes and acceleration profiles. In a threat-approaching scenario, the relevant alternatives cluster around emergency braking and evasive maneuvers. Applying a uniform action grid across all contexts would either produce uninformative counterfactuals in free-flow conditions or miss the safety-critical alternatives in threat scenarios.

The adaptive action grid $\mathcal{A}_{\text{grid}}(c)$, conditioned on context category $c \in \mathcal{C}$, is defined as a finite enumeration of action templates $\{(a^\ell, s^\ell, \ell)\}$ where $a^\ell$ is a longitudinal acceleration, $s^\ell$ is a lateral steering command, and $\ell$ is a human-legible label. The grid is parameterized to maximize the information content of the counterfactual experiment: actions are selected to span the relevant region of the action space for the identified context, ensuring that the necessity score reflects a meaningful assessment of the decision's indispensability.

### 5.3 Explanation Template Architecture

The narration layer employs a three-tier template architecture governed by decision classification:

**Detailed Causal Explanation** (activated for $\eta \geq \eta_{\text{thresh}}$, $G \geq G_{\text{thresh}}$): The full causal narrative is generated, encompassing the ego state, scene description, the chosen action's rationale, the worst counterfactual outcome with its threat agent and time-to-collision, and a grounding confirmation citing the top-attended agents with their attention masses. This tier produces the most epistemically complete explanation, appropriate when both the decision's necessity and its attentional grounding are confirmed.

**Detailed Explanation with Transparency Caveat** (activated for $\eta \geq \eta_{\text{thresh}}$, $G < G_{\text{thresh}}$): The same causal narrative is generated, but supplemented with a mandatory transparency caveat noting that the policy's attention distribution does not confirm awareness of the identified threat agents. This classification — UNGROUNDED_CRITICAL — represents the most epistemically troubling case: a decision that was necessary (alternatives were dangerous) but whose internal attention did not focus on the entities that made it necessary. The caveat is not a failure message but a transparency signal, directing human oversight toward a decision that may have been correct for the wrong reasons.

**Brief Narration** (activated for $\eta < \eta_{\text{thresh}}$): Routine decisions are narrated concisely, acknowledging the chosen action and its context without invoking the full causal apparatus. This maintains explanation coverage without generating cognitive overhead for decisions that carry no exceptional risk profile.

---

## 6. Counterfactual Explanation Framework

### 6.1 The Necessity Score as Causal Evidence

The necessity score $\eta$ defined in Section 2.2 provides a form of causal evidence that is formally distinct from feature attribution methods such as SHAP or integrated gradients. Feature attribution methods quantify the sensitivity of the policy output to input perturbations in the input space — they answer the question "which features influenced this output?" The necessity score answers a categorically different question: "would the agent have been safe had it acted differently?"

This distinction is consequential for explanation faithfulness. A feature attribution method may assign high importance to an agent that the policy attended to for reasons unrelated to safety — for example, a distant vehicle that happens to be in the encoder's input window and whose representation was activated by training distributional properties. The necessity score, by contrast, assigns causal weight to agents only to the extent that their presence in the simulation causes alternative trajectories to result in safety violations. The threat agent set $\mathcal{T}_{\text{threat}}$ is thus constituted by causal relevance, not by attentional prominence.

### 6.2 Counterfactual Rollout Mechanics

Each alternative action $a_k \in \mathcal{A}_{\text{grid}}$ is applied at the current state $s_t$, and the resulting trajectory is evaluated over a horizon of $H$ steps using the simulator's forward dynamics. The outcome classification function $\Omega$ evaluates two distinct failure modes: collision occurrence, determined via pairwise bounding box overlap checks over the five-degree-of-freedom agent representations $(x, y, \ell, w, \psi)$, and road boundary violation, determined via lane polygon containment tests. Both checks are applied at each intermediate step of the rollout, with the first collision step recorded rather than only the terminal state — this produces a time-to-collision estimate $\hat{\tau} = (t_{\text{coll}} + 1) \cdot \Delta t$ (where $\Delta t = 0.1\text{s}$) that serves as a continuous severity measure refining the binary outcome classification.

The choice of rollout horizon $H$ involves a fundamental tradeoff. A short horizon underestimates the long-term consequences of alternative actions — a trajectory may appear safe over $H = 5$ steps but inevitably collide at step 8. A long horizon introduces simulator compounding errors and increases the probability that the reference policy's behavior diverges from the evaluation context. The operational horizon of $H = 10$ steps (corresponding to 1.0 seconds at 10 Hz) represents a pragmatic calibration that captures the most immediate collision-relevant dynamics while remaining within the regime where simulator accuracy is reliable.

### 6.3 Vectorized Parallel Rollout via Functional Transformation

A critical performance consideration in counterfactual explanation at inference time is the computational cost of executing $K = |\mathcal{A}_{\text{grid}}|$ independent forward simulations per timestep. Sequential execution over a grid of size $K = 9$ (the default $3 \times 3$ longitudinal-lateral product grid) would multiply the evaluation latency by a factor of $K$, rendering online narration impractical. The framework addresses this through a vectorized parallel rollout architecture enabled by JAX's functional transformation primitives.

The core insight is that all $K$ rollouts share the same initial state $s_t$ and diverge only in their applied actions $\{a_k\}_{k=1}^{K}$. This independence structure is precisely what the `vmap` (vectorized map) transformation exploits: a single-trajectory rollout function is written over a scalar (rank-0) state and action, and `vmap` is applied over the action batch dimension to produce a batched rollout function that executes all $K$ trajectories in a single parallelized pass through the simulator's forward dynamics. The batched state is constructed by replicating the initial state $K$ times along a leading batch dimension:

$$
s_t^{(k)} = s_t \quad \forall k \in \{1, \ldots, K\}, \quad \hat{a}^{(k)} = a_k
$$

so that the vmapped rollout kernel processes the batch $\{(s_t^{(k)}, \hat{a}^{(k)})\}_{k=1}^{K}$ in parallel.

A further performance optimization is the JIT (just-in-time) compilation of the vmapped kernel, which compiles the entire $H$-step simulation loop — including the overlap and offroad metric evaluations at each step — into a single XLA computation graph. The compiled kernel is cached after the first invocation and reused for subsequent calls with the same action grid size. Grid updates — which occur when the detected scene context changes between timestep intervals — invalidate the compiled kernel and trigger recompilation, as the vmapped batch dimension is determined by $K$ at compile time.

The rollout kernel maintains four boolean/integer state variables that accumulate across the $H$ simulation steps using JAX's functional loop semantics (unrolled for-loops over a fixed horizon):

- $c_k \in \{0, 1\}$: binary collision flag for alternative $k$,
- $\rho_k \in \{0, 1\}$: binary offroad flag for alternative $k$,
- $t_k^* \in \mathbb{Z}$: first collision step index (initialized to $H$ as a sentinel),
- $\tau_k^* \in \mathbb{Z}$: threat agent index at first collision (initialized to $-1$ as a sentinel).

The functional accumulation structure — using conditional assignments rather than early exits — is required by JAX's requirement that control flow be static or expressed via `jnp.where` rather than Python conditionals, which would prevent compilation. This constraint shapes the kernel design: the loop always runs for exactly $H$ steps, with outcome flags updated conditionally at each step.

### 6.4 Outcome Classification and Threat Agent Identification

The outcome classification at each rollout step relies on two simulator-native metric primitives: an overlap metric that computes per-agent bounding box intersection indicators, and an offroad metric that evaluates lane containment for each agent. Both metrics return per-agent values of shape $(N_{\text{agents}},)$, from which the SDC-specific values are extracted by indexing with the precomputed SDC agent index.

Collision detection proceeds as follows. At each step $t' \in \{0, \ldots, H-1\}$, the SDC's overlap indicator $\text{ovlp}_{0,t'}$ is evaluated. If a collision is detected for the first time at step $t'$ (i.e., $\text{ovlp}_{0,t'} = 1$ and $c_k = 0$ at entry to step $t'$), the threat agent is identified as:

$$
\tau_k^* = \arg\max_{j \neq 0} \, \text{ovlp}_{j,t'}
$$

where $\text{ovlp}_{j,t'}$ is the per-agent overlap value with the SDC's column zeroed out to prevent self-identification. This argmax-based threat identification yields the agent most implicated in the collision at the first occurrence, providing a per-alternative threat agent assignment that flows into the necessity scorer and attention grounder.

The scalar state normalization requirement deserves explicit discussion. JAX's vmapped functions require that the function being mapped operates on arrays of the same rank as the non-batched case. Simulator states returned by the evaluation loop carry a leading batch dimension from the evaluation harness; this must be stripped before the state is replicated for the counterfactual batch. Failure to normalize to a scalar (rank-0 in the batch dimension) state before applying `vmap` produces rank mismatches that corrupt the parallel computation silently rather than raising explicit errors in compiled code — a subtle correctness hazard that must be addressed by explicit tree-map normalization prior to batching.

### 6.5 Threat Agent Deduplication

Multiple alternative actions in $\mathcal{A}_{\text{grid}}$ may implicate the same agent — agent $i$ may appear in the threat set for both an emergency braking alternative and an evasive maneuver alternative. Naive aggregation would artificially inflate the apparent danger posed by agent $i$. The threat agent set is therefore deduplicated by retaining, for each unique agent $i$, the minimum time-to-collision across all alternatives that implicate it:

$$
\text{TTC}_i = \min_{k : \tau_k^* = i} \hat{\tau}_k
$$

This minimum-TTC representation captures the most conservative (i.e., most immediate) danger estimate associated with each threat agent, providing a well-defined severity ordering for grounding score computation. The deduplication also serves an explanatory function: presenting the same agent multiple times in different counterfactual descriptions would introduce redundancy that reduces the signal-to-noise ratio of the explanation presented to the LLM.

---

## 7. Prompting and Reasoning Strategy

### 7.1 Constrained Prompting as a Faithfulness Mechanism

The design of the prompt architecture is the primary mechanism by which the framework enforces explanation faithfulness. The system prompt issued to the LLM encodes a set of behavioral constraints that are not merely stylistic but epistemically essential:

1. **Information restriction**: The LLM is instructed to reference only information present in the structured evidence package. This constraint prevents the model from drawing on its pre-training knowledge about driving scenarios to construct plausible-sounding but ungrounded explanations.
2. **Speculation prohibition**: The LLM is explicitly prohibited from claiming knowledge of what the policy "might have been thinking." This constraint acknowledges the epistemic limits of interpretability: the framework provides evidence about the policy's external behavior and attentional patterns, not direct access to its internal intentionality.
3. **Structural coverage requirements**: The prompt specifies that every generated explanation must address three components: (a) what the vehicle did and why, (b) what the most dangerous alternative would have caused, and (c) whether attention confirms the decision. This structural mandate ensures that the generated explanation is causally complete rather than selectively emphasizing aspects that are easier to narrate.
4. **Concision constraints**: A maximum response length is imposed to prevent explanatory verbosity that exceeds the information content of the evidence package — a failure mode in which the model, constrained to a small factual basis, generates padding that reduces the signal-to-noise ratio of the explanation.

### 7.2 Template-Driven Prompt Construction

The prompt construction follows a template architecture that transforms the structured report $\mathcal{R}$ into a formatted textual evidence package. This transformation is deterministic and interpretable: every element of the user prompt corresponds to a specific field in $\mathcal{R}$, and the ordering and framing of information is controlled to guide the LLM's reasoning toward the causal structure of the explanation.

Critically, the template architecture decouples the *content* of the evidence package from the *form* of the explanation. The LLM does not receive a specification of what to say but a structured presentation of what is known. This distinction preserves the LLM's capacity for coherent prose generation while constraining the factual domain within which that generation operates.

The scene description presented to the LLM is deliberately limited to the five proximate agents ranked by Euclidean distance. This truncation is not merely a computational convenience — it reflects a principled hypothesis about the cognitive scope of explanations: an explanation that enumerates all agents in the scene is not more informative than one that focuses on the most proximate and therefore most decision-relevant entities. The proximity filter also mitigates the risk of context length saturation in LLMs with limited effective attention windows.

### 7.3 Semantic Routing Logic

The routing of reports to explanation templates is mediated by a decision classifier that maps the joint space $(\eta, G)$ to a discrete label via a threshold-based rule:

$$
\text{class}(\eta, G) = \begin{cases} \text{GROUNDED\_CRITICAL} & \text{if } \eta \geq \eta_\theta \text{ and } G \geq G_\theta \\ \text{UNGROUNDED\_CRITICAL} & \text{if } \eta \geq \eta_\theta \text{ and } G < G_\theta \\ \text{GROUNDED\_ROUTINE} & \text{if } \eta < \eta_\theta \text{ and } G \geq G_\theta \\ \text{ROUTINE} & \text{otherwise} \end{cases}
$$

where $\eta_\theta$ and $G_\theta$ are configurable thresholds defaulting to 0.5. The routing logic ensures that the depth and epistemic character of the generated explanation are calibrated to the decision's risk profile: critical decisions receive full causal narratives, while routine decisions receive concise summaries. Ungrounded critical decisions receive the full causal narrative augmented with an explicit transparency caveat — the most epistemically honest response to a situation where the computational evidence is internally inconsistent.

### 7.4 Online and Offline Narration Modes

The framework supports two operational modes that reflect different deployment contexts. In *online* mode, narration is generated synchronously during the policy evaluation loop, enabling real-time explanation generation for operator-facing interfaces. In *offline* mode, structured reports are persisted to durable storage during evaluation, and narration is generated as a batch post-processing step. The offline mode is appropriate for research contexts where evaluation throughput is the primary concern and explanation latency is not time-critical; it also enables systematic comparison of different LLM configurations against the same evidence package without re-executing the computationally expensive evaluation loop.

The decoupling of evidence generation from narration generation in offline mode has an additional epistemic benefit: it enables the generated explanations to be audited against the structured reports by human reviewers before being presented to end users, inserting a human-in-the-loop verification stage between computation and communication.

---

## 8. Human-Centered Explainability

### 8.1 Explanation Granularity and Cognitive Load

Effective human-centered explainability requires calibrating the granularity of explanations to the cognitive capacity and informational needs of the intended audience. The four-tier decision classification scheme operationalizes this calibration: routine decisions, which require minimal cognitive engagement from operators, are explained concisely; critical decisions, which demand careful human review, receive full causal narratives. This adaptive granularity strategy reduces the total cognitive load on operators over extended deployment periods while ensuring that high-stakes decisions receive appropriately detailed explanations.

The explicit distinction between GROUNDED_CRITICAL and UNGROUNDED_CRITICAL classifications is particularly significant from a human factors perspective. Both classifications involve high-necessity decisions — situations where alternative actions were dangerous — but they differ in the degree to which the policy's internal attention supports the counterfactual evidence. An operator presented with an UNGROUNDED_CRITICAL explanation is not told that the agent made an error; they are told that the computational evidence is internally inconsistent in a specific, interpretable way, enabling them to make an informed judgment about whether to trust the decision.

### 8.2 Trust Calibration Through Transparency

A persistent risk in human-AI collaboration is *automation bias* — the tendency of human operators to over-trust automated systems, especially when those systems are accompanied by fluent, authoritative-sounding explanations. The transparency caveat mechanism directly addresses this risk by introducing a formal epistemic flag that communicates the limits of the system's self-knowledge. An explanation that acknowledges its own uncertainty is epistemically more honest than one that projects uniform confidence, and it creates the conditions for *calibrated trust* — operators who trust the system appropriately in grounded decisions and apply heightened scrutiny to ungrounded ones.

The requirement that explanations always address attention grounding serves a related function. By making the attentional confirmation (or lack thereof) a mandatory component of every critical explanation, the framework ensures that operators are systematically informed about the consistency between the policy's counterfactual behavior and its internal computation — a consistency that is neither guaranteed nor verifiable from behavioral observation alone.

### 8.3 Explanation as Audit Trail

In regulatory and liability contexts, the structured reports generated by the computation layer — together with the LLM-generated narrations — constitute an explanation audit trail that can be reviewed after the fact to understand the agent's decision-making process at any point in its operational history. This audit trail has two components: the formal computational evidence (necessity scores, grounding scores, counterfactual alternatives, attention breakdowns) and the natural-language explanation (the LLM narration). The formal component is machine-readable and amenable to automated analysis; the natural-language component is human-readable and amenable to qualitative review. Together, they provide a multi-modal record of decision rationale that satisfies both technical and communicative transparency requirements.

---

## 9. LLM Evaluation Framework

### 9.1 The Evaluation Problem

Assessing the quality of LLM-generated explanations in the context of autonomous driving XAI presents a fundamentally different challenge from standard NLP evaluation. Conventional metrics such as BLEU, ROUGE, or BERTScore measure surface-level lexical or semantic similarity to a reference string — but no ground-truth reference narration exists for a given driving decision report. The adequacy of an explanation is not determined by its resemblance to a human-authored template; it is determined by its factual fidelity to the structured computational evidence, the logical coherence of its causal reasoning, and its calibration to the risk severity of the decision being explained.

This chapter adopts a reference-free evaluation paradigm using G-Eval (Liu et al., 2023), an LLM-as-judge methodology in which a capable evaluator model scores the narration against domain-specific rubric criteria. The evaluator — referred to as the *judge model* — is provided with the structured report (the input to the narrator) and the generated narration (the actual output), and scores the narration along multiple dimensions using chain-of-thought evaluation steps that operationalize each criterion. This approach aligns with emerging consensus in the evaluation of generative AI systems for high-stakes applications, where human expert annotation is the gold standard but LLM judges have been shown to correlate highly with expert assessments at a fraction of the cost.

### 9.2 Core Evaluation Metrics

The evaluation framework defines nine metrics organized into two tiers. The five core metrics assess fundamental properties of explanation quality that are intrinsic to the framework's design objectives. The four research-backed metrics extend the evaluation with methodologies adapted from the autonomous driving VQA (Visual Question Answering) and LLM reasoning evaluation literature.

**Safety Fidelity** measures whether the narration accurately reflects the safety-critical content of the structured report — specifically, whether collision outcomes, threat agent identifiers, and time-to-collision values are correctly reported without fabricating dangers absent from the evidence. A narration that invents a collision with Agent 7 when the report lists only SAFE alternatives commits a safety fidelity failure of the highest severity. Conversely, a narration that correctly identifies the most dangerous alternative and its associated TTC is rewarded.

**Perceptual Grounding** assesses whether the narration's claims about what the policy "attended to" or "was aware of" are consistent with the attention grounding data in the structured report. An agent cited as "focused on" by the narration must appear in the per-agent breakdown with non-negligible attention mass ($> 0.05$). The metric also checks that UNGROUNDED_CRITICAL cases include the mandatory transparency caveat — a structural requirement imposed by the decision classification routing.

**Causal Accuracy** evaluates the logical consistency of the narration's cause-effect reasoning. A narration may correctly name the chosen action while misrepresenting the necessity score — for example, describing a necessity of 0.75 as "a 75% probability of collision" rather than "75% of alternatives would fail." This distinction is not merely semantic: the necessity score is a population-level statistic over the action grid, not a probabilistic assessment of the chosen action's safety. Causal accuracy penalizes such misinterpretations while also checking that counterfactual claims in the narration match specific alternatives listed in the report.

**Action Justification** assesses whether the narration explains *why* the chosen action was preferred over alternatives, not merely *what* it was. A compliant narration must contrast the chosen action against at least one alternative and identify the most dangerous alternative by name. The decision classification should be reflected in the depth of justification: GROUNDED_CRITICAL decisions require substantially more causal argumentation than ROUTINE decisions.

**Conciseness and Constraint Compliance** evaluates adherence to the structural constraints imposed by the system prompt: the narration must comprise 3–5 sentences, avoid speculative language, restrict itself to information present in the report, and cover all three mandatory logical links: (a) what the vehicle did and why, (b) the most dangerous alternative and its consequence, and (c) whether attention confirms the decision.

### 9.3 Research-Backed Metrics

**Strict Context Grounding** (adapted from DriveBench, Xie et al., 2024) operationalizes a strict anti-hallucination criterion: every entity — agent identifier, numeric value, road feature — mentioned in the narration must be traceable to a specific field in the structured report JSON. This metric is stricter than safety fidelity in that it penalizes any hallucinated entity, not only safety-critical ones. It is motivated by DriveBench's finding that multimodal models frequently generate plausible-sounding text that is not grounded in the provided visual or textual context — an "answering for the right reasons" problem that persists even when the surface-level answer is correct.

**Progressive Alignment** (adapted from AutoDriDM, Yang et al., 2024) evaluates the narration across three hierarchically ordered alignment levels: *Object* (correct identification of threat agent IDs), *Scene* (accurate description of spatial relations, distances, and TTC values from the scene description), and *Decision* (correct linkage of the chosen action to the grounding score and decision class). The hierarchical structure is designed to detect a specific failure mode: a narration that arrives at a correct decision-level conclusion while failing at the object or scene level is penalized heavily, as it indicates that the model is guessing the conclusion rather than deriving it from the evidence. This progressive penalty structure instantiates a form of reasoning chain verification that goes beyond checking the final output.

**Logical Completeness** (adapted from DriveLMM-o1, Nie et al., 2024) applies a missing-step penalty: the three mandatory logical links — WHAT, DANGER, ATTENTION — must all be present for the narration to score well. A narration covering only two of the three links incurs a substantial deduction regardless of the quality of the links it does cover. This metric enforces the completeness requirement of the system prompt constraint at the scoring level, providing a quantitative instrument for detecting partial compliance.

**Risk Coherence** (adapted from DriveLMM-o1) assesses the alignment between the narration's tone and urgency and the risk level indicated by the necessity score and decision class. A necessity score $\eta \geq 0.7$ should elicit correspondingly urgent language; describing a high-necessity critical scenario with calm, routine language constitutes a coherence failure. Conversely, overstating risk for a routine scenario ($\eta < 0.3$) is penalized symmetrically — miscalibrated urgency in either direction undermines the trust-calibration function of the explanation layer. This metric operationalizes the intuition that a faithful explanation must not only be factually accurate but tonally appropriate to the severity it describes.

### 9.4 Cognitive Scorecard and Composite Scoring

The nine individual metrics are aggregated into three composite dimensions — *Situational Awareness*, *Reasoning Quality*, and *Communication Quality* — which together constitute a **Cognitive Scorecard** for a given LLM narration model.

**Situational Awareness** aggregates the perceptual and anti-hallucination metrics:

$$
\text{SitAware} = \frac{1}{3}\left(\text{PerceptualGrounding} + \text{SafetyFidelity} + \text{StrictContextGrounding}\right)
$$

This composite captures the degree to which the narration correctly perceives and reports the scene as it exists in the structured evidence, without confabulation.

**Reasoning Quality** aggregates the causal and action-reasoning metrics:

$$
\text{Reasoning} = \frac{1}{4}\left(\text{CausalAccuracy} + \text{ActionJustification} + \text{ProgressiveAlignment} + \text{RiskCoherence}\right)
$$

This composite captures the depth and correctness of the narration's causal inference — the degree to which it reasons from evidence to conclusion rather than pattern-matching to plausible-sounding explanations.

**Communication Quality** aggregates the structural compliance metrics:

$$
\text{Communication} = \frac{1}{2}\left(\text{Conciseness} + \text{LogicalCompleteness}\right)
$$

This composite captures adherence to the prompt's structural constraints and the completeness of the logical chain.

The **Overall Cognitive Score** is a weighted combination of the three composite dimensions:

$$
\text{OCS} = 0.4 \cdot \text{SitAware} + 0.4 \cdot \text{Reasoning} + 0.2 \cdot \text{Communication}
$$

The weighting scheme places equal and elevated weight on perceptual grounding and reasoning quality — reflecting the judgment that factual accuracy and causal coherence are the primary adequacy conditions for safety-critical explanation — while communication quality, though necessary for human comprehension, is given a lower weight as a secondary concern. This weighting is configurable and may be adapted for deployments where communicative requirements are heightened (e.g., operator-facing displays with strict attention constraints).

### 9.5 Multi-Model Comparison Design

A central research question motivating the evaluation framework is whether different LLM architectures and parameter scales produce qualitatively different explanation behaviors when constrained to the same structured evidence package. The framework supports multi-model comparative evaluation, in which a set of candidate narration models $\{M_1, M_2, \ldots, M_L\}$ each generate narrations for the same corpus of structured reports, and their Cognitive Scorecards are compared along all three composite dimensions.

This comparative design separates two sources of variation: variation in the evidence package (which is fixed, being derived from a single policy evaluation) and variation in the narration model (the variable of interest). By holding the evidence constant, the comparison isolates the narration model's ability to faithfully interpret and communicate the provided evidence, independent of any variation in the underlying driving decisions being explained.

The multi-model design also enables an analysis of performance stratification by decision class. A model that performs well on ROUTINE decisions but poorly on GROUNDED_CRITICAL or UNGROUNDED_CRITICAL decisions reveals a qualitatively different failure mode from a model that underperforms uniformly — the former suggests difficulty with complex causal structures while the latter suggests a general faithfulness problem. The decision class breakdown embedded in the Cognitive Scorecard provides a structured instrument for this stratified analysis.

### 9.6 Judge Model Design and Evaluation Validity

The choice of judge model is consequential for evaluation validity. A judge with insufficient contextual reasoning capability may score narrations based on surface fluency rather than factual grounding, effectively rewarding hallucinated but coherent text over faithful but awkward prose. 

Each metric is implemented as a G-Eval instance with domain-specific evaluation steps that operationalize the criterion in a chain-of-thought format. The evaluation steps are designed to direct the judge's attention to specific, verifiable claims in the narration (e.g., "list every agent ID mentioned; verify each against the report") rather than holistic impressions. This step-by-step structure reduces judge variance by constraining the evaluation procedure, trading some of the judge's generalization capacity for increased consistency across repeated evaluations of the same narration.

A pass/fail threshold of 0.7 is applied to individual metric scores. This threshold is calibrated to enforce substantive compliance while accommodating minor imprecisions in paraphrase and numerical rounding that do not constitute genuine faithfulness failures. The threshold is configurable to accommodate different deployment contexts, where the acceptable balance between strict faithfulness and communicative flexibility may differ.

## 10. Limitations and Challenges

### 10.1 Attention as Imperfect Proxy

The attention grounding analysis is predicated on the assumption that cross-attention weights provide a meaningful proxy for the policy's internal decision-relevant processing. This assumption is contested in the mechanistic interpretability literature. Jain and Wallace (2019) demonstrated that attention weights can be adversarially manipulated without changing model outputs, and Wiegreffe and Pinter (2019) showed that multiple attention configurations can produce identical predictions, undermining the claim that any particular attention pattern is the "true" explanation for a given decision.

Within the present framework, this limitation is partially mitigated by the use of attention grounding as *corroborating evidence* rather than *primary explanation*. The primary causal evidence is provided by the counterfactual analysis; attention grounding serves as an internal consistency check. When attention and counterfactuals agree (GROUNDED_CRITICAL), the explanation is epistemically strengthened. When they disagree (UNGROUNDED_CRITICAL), the explanation explicitly flags the inconsistency rather than suppressing it. This use of attention as corroboration rather than causation is more epistemically defensible than approaches that treat attention as the sole basis for explanation.

### 10.2 Counterfactual Completeness

The necessity score computed over a finite action grid $\mathcal{A}_{\text{grid}}$ is necessarily an approximation of the true necessity of the chosen action over the continuous action space $\mathcal{A}$. A finite grid cannot enumerate all possible actions, and the selection of grid elements is informed by domain knowledge about the context-relevant action space but is not guaranteed to include the globally optimal alternative. In contexts where the grid is coarse, the necessity score may underestimate the true danger of alternative actions (by missing particularly dangerous alternatives) or overestimate it (by including actions that are unrealistically extreme for the context).

The adaptive grid construction strategy partially addresses this by selecting context-specific actions that are more likely to be relevant than a uniform grid, but the fundamental incompleteness of counterfactual enumeration over a continuous space remains an irreducible limitation of the approach.

### 10.3 Hallucination Residual Risk

Despite the faithfulness constraints imposed by the prompt architecture, LLMs retain a non-zero probability of generating text that misrepresents the structured evidence. Sources of hallucination that survive the constrained prompting approach include: numerical misinterpretation (e.g., rendering a necessity score of 0.85 as "the agent was 80% confident"), relation confusion (e.g., describing an agent labeled `behind_of` as being in front of the ego vehicle), and narrative interpolation (e.g., adding causal connectives between facts that are not causally related in the evidence package).

These failure modes are qualitatively different from the hallucinations that occur when LLMs are prompted over raw observations — they are misinterpretations of provided evidence rather than fabrications from pre-training — but they remain sources of potential unfaithfulness. Mitigation strategies include post-generation verification against the structured report (checking that numerical claims in the generated text match the report fields), structured output formats that constrain the form of the generated text, and human oversight at the UNGROUNDED_CRITICAL tier where the consequences of misrepresentation are greatest.

### 10.4 Simulator Fidelity and Generalization

The counterfactual rollouts are executed within the same simulator used for policy training and evaluation. The outcomes of these rollouts are therefore only as realistic as the simulator's fidelity to real-world driving dynamics. In domains where the simulation-to-reality gap is significant — particularly in complex interaction scenarios, at the margins of the simulator's physical model, or in novel environmental conditions — the counterfactual evidence may not faithfully represent the consequences of alternative actions in deployment. This limitation is inherent to all simulation-based interpretability approaches and is not specific to the present framework, but it bears explicit acknowledgment as a constraint on the generalization of explanation-supported conclusions.

### 10.5 Token Mapping Fidelity

The attention grounding analysis depends critically on the accuracy of the mapping from encoder token positions to simulator agent indices. This mapping is derived by replicating the feature extractor's distance-sorting procedure in the interpretability pipeline. Any divergence between the pipeline's reconstruction of this mapping and the actual mapping used during policy inference — arising, for example, from floating-point precision differences, caching effects, or timestep indexing discrepancies — would corrupt the grounding score computation by assigning attention mass to the wrong agents. Maintaining bit-exact correspondence between the policy's feature extraction and the interpretability pipeline's token reconstruction is a non-trivial engineering requirement that must be validated empirically.

---

## 11. Future Research Directions

### 11.1 Mechanistic Interpretability Beyond Attention

The present framework employs attention weights as the primary window into the policy's internal processing. Future work should investigate whether mechanistic interpretability techniques — specifically, activation patching and causal tracing methods developed for language models — can be adapted to the transformer encoder of an autonomous driving policy. Activation patching enables the identification of specific attention heads and MLP layers that causally mediate particular behavioral outcomes, providing a more granular and mechanistically grounded form of explanation than aggregate attention mass.

### 11.2 Explanation Evaluation Metrics

The evaluation of explanation quality in autonomous driving contexts lacks well-established metrics. The grounding score $G$ provides a partial measure of internal consistency, but it does not directly assess whether explanations are useful to human operators in practice. Future work should develop operator studies that measure explanation utility along dimensions including decision support accuracy (do explanations help operators identify unsafe decisions?), trust calibration (do operators trust grounded decisions more than ungrounded ones?), and cognitive efficiency (what is the cognitive cost of processing explanations of different granularities?).

### 11.3 Adversarial Robustness of Explanations

A systematic investigation of the adversarial robustness of the explanation pipeline would strengthen confidence in its reliability. Of particular interest are scenarios where the policy's attention distribution is deliberately manipulated — either through adversarial perturbations of the input or through training incentives that produce attention patterns unrelated to policy semantics — in ways that produce high grounding scores for incorrect threat agents. Understanding the failure modes of the grounding mechanism under adversarial conditions is essential for deployment in safety-critical environments where manipulation resistance is required.

### 11.4 Causal Explanation Beyond Binary Outcomes

The current counterfactual framework evaluates alternatives against binary safety outcomes (SAFE, COLLISION, OFFROAD). A richer counterfactual analysis would evaluate alternatives against a continuous discomfort or risk metric — incorporating passenger comfort, traffic flow impact, and sub-threshold near-miss events — enabling necessity scores and threat agent identifications to be calibrated for routine driving situations where binary safety outcomes are uninformative. This extension would require a continuous outcome function $\Omega : \mathcal{T} \to \mathbb{R}$ replacing the discrete classification, with corresponding modifications to the necessity score formulation and explanation templates.

### 11.5 Multi-Step and Intent-Conditional Explanation

The current framework explains individual decision steps independently. Autonomous driving decisions are, however, embedded in longer-horizon behavioral intentions — a lane change begun at step $t$ is motivated by a navigation goal that will be visible at step $t+k$. Future work should investigate explanation frameworks that reason over multi-step decision sequences, attributing individual actions to trajectory-level intentions and providing explanations that are coherent across the temporal extent of a maneuver rather than atomized at the step level. This would require extending the counterfactual rollout to evaluate alternative *sequences* of actions rather than alternative *single* actions, and correspondingly extending the narrative template to express temporal causal structures.

### 11.6 Differentiable Explanation Alignment

An ambitious long-term direction is the development of training objectives that encourage alignment between the policy's attention patterns and its counterfactually identified threat agents. This would transform explanation faithfulness from an evaluation criterion applied post-hoc to a training objective optimized during policy learning — producing policies that are not merely explainable in retrospect, but whose internal computation is architecturally aligned with the explanatory structure. Such an approach would require differentiable counterfactual outcome functions and a joint training objective that balances task performance with attentional grounding, representing a significant methodological advance over current decoupled approaches.

---

## References

*The following references represent the theoretical foundations and closely related work for the framework described above. Specific citation details should be populated from the thesis bibliography.*

- Wachter, S., Mittelstadt, B., & Russell, C. (2017). Counterfactual Explanations Without Opening the Black Box: Automated Decisions and the GDPR. *Harvard Journal of Law & Technology*.
- Jain, S., & Wallace, B. C. (2019). Attention is not Explanation. *NAACL-HLT*.
- Wiegreffe, A., & Pinter, Y. (2019). Attention is not not Explanation. *EMNLP*.
- Pearl, J., & Mackenzie, D. (2018). *The Book of Why: The New Science of Cause and Effect*. Basic Books.
- Doshi-Velez, F., & Kim, B. (2017). Towards a Rigorous Science of Interpretable Machine Learning. *arXiv preprint*.
- Lipton, Z. C. (2018). The Mythos of Model Interpretability. *Queue*.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *KDD*.
- Elhage, N., et al. (2021). A Mathematical Framework for Transformer Circuits. *Transformer Circuits Thread*.
- Shao, H., et al. (2023). ReasonNet: End-to-End Driving with Temporal and Global Reasoning. *CVPR*.
- Kim, J., & Rohrbach, A. (2020). Grounded Situation Models for Robots: Where Language and Perception Meet. *AAAI*.
- European Commission. (2021). Proposal for a Regulation on Artificial Intelligence (AI Act). *Official Journal of the European Union*.
- ISO 21448:2022. Road Vehicles — Safety of the Intended Functionality (SOTIF).
