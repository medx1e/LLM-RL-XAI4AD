# Reward-Conditioned Attention — Research Overview

> Non-technical overview of what we built, why, and what we found.
> For the technical implementation details, see REWARD_CONDITIONED_TECHNICAL.md.
> For paper-ready findings, see REWARD_CONDITIONED_FINDINGS.md.

---

## 1. The Research Question

We trained autonomous driving agents using reinforcement learning (RL) with different reward functions. The reward function tells the agent what to care about — safety, navigation, comfort, etc.

The question: **does the reward function change what the agent pays attention to?**

More specifically: we can look directly inside the model's attention mechanism — a mathematical representation of "what information the model is currently focusing on at each timestep." We ask whether this attention is semantically meaningful and whether it changes predictably based on the reward the model was trained with.

---

## 2. The Models We Studied

All models share the same architecture: a **Perceiver encoder** — a neural network that takes in all available scene information (ego vehicle, nearby agents, road geometry, traffic lights, GPS route) and processes it through a cross-attention mechanism before making driving decisions.

We studied three reward configurations, each a superset of the previous:

**Basic** — the most minimal reward:
- Only penalizes actual violations: collision, going off-road, running red lights
- No incentive to follow a route or make progress

**Minimal** — adds navigation:
- Everything in basic + penalizes being off-route + rewards progression
- The agent now has a destination and is incentivized to reach it

**Complete** — adds continuous safety signal:
- Everything in minimal + penalizes driving too close to other vehicles (TTC penalty at 1.5s threshold) + speed comfort
- The agent is now penalized for *approaching* danger, not just for actual collisions

---

## 3. What Is "Attention" Here?

The Perceiver encoder has 16 learned "queries" that scan over all 280 input tokens (representations of the ego vehicle, 8 other agents, 200 road points, 5 traffic lights, 10 GPS waypoints). At each timestep, each query produces attention weights over all 280 tokens — essentially a probability distribution over "what to look at."

We aggregate these weights into 5 categories:
- **Ego (SDC)**: how much attention goes to the agent's own past trajectory
- **Other Agents**: how much attention goes to nearby vehicles
- **Road Graph**: how much attention goes to road geometry
- **Traffic Lights**: how much attention goes to traffic signals
- **GPS Path**: how much attention goes to the planned route waypoints

These 5 numbers sum to 1.0 at every timestep. They tell us the model's attentional budget allocation.

---

## 4. What We Built

A full analysis pipeline that:
1. Runs each model through driving scenarios, collecting the attention weights and scene state at every timestep
2. Computes continuous risk metrics from the scene state (collision risk based on time-to-collision, navigation risk, behavior risk)
3. Correlates risk metrics with attention weights — does attention track what the model was rewarded for?
4. Generates visualizations: scatter plots, time series, bird's-eye-view panels, and model comparison figures

The key methodological insight was that **naive pooling across scenarios is misleading**. We discovered that averaging correlations across all timesteps from all scenarios masks the true within-episode effects (a scenario with near-constant high risk contributes nothing useful). Instead, we compute correlations within each episode separately and then aggregate using Fisher z-transform — a statistically correct approach that the XAI community often skips.

---

## 5. The Most Interesting Scenario: s002

Among the 5 scenarios in our data, scenario s002 turned out to be the most analytically valuable — not because it had the most events (it had the fewest: 17), but because it has **two distinct risk cycles** within the same episode:

- Risk rises from 0.5 to 1.0 during t=0–35 (a threatening vehicle approaches)
- Risk drops to 0 during t=42–65 (the threat passes)
- Risk rises again to 0.8 during t=65–80 (a second threat emerges)

This two-cycle structure is essential: the model has to *react twice* to separate danger events within the same episode. This rules out spurious correlations and provides much stronger evidence that the attentional response is genuinely reactive rather than coincidental.

**The lesson for future experiments**: select scenarios based on risk *variability* (std > 0.3, with both calm and dangerous phases), not by event count. The event mining module currently ranks s004 first (48 events) — but s004 has near-constant maximum risk throughout, making it useless for correlation analysis.

---

## 6. The Three-Model Comparison

We ran all three models (basic, minimal, complete) on s002 and tracked their attention allocation. The results reveal a clear story about how reward design shapes learned attention.

### GPS Attention — the clearest gradient
The fraction of attention devoted to the GPS route waypoints follows a perfect ordering:

```
Minimal (route+progression reward):  0.314  ← highest GPS attention
Complete (+ TTC penalty):             0.166
Basic (no navigation reward):         0.092  ← lowest GPS attention
```

This makes intuitive sense: if you reward the model for following a route (minimal), it learns to look at the route. If you add a proximity penalty (complete), some attention budget shifts away from navigation toward safety. If you give no navigation reward at all (basic), GPS attention nearly disappears.

### Agent Attention Baseline — counterintuitive
```
Basic:    0.264  ← highest agent attention baseline
Complete: 0.173
Minimal:  0.117  ← lowest agent attention baseline
```

The basic model, despite having the simplest reward, allocates the most attention to other agents by default. Why? Because it has no GPS route to follow, no progression to pursue — agents and road geometry are the only things in its observation worth attending to. When you add navigation rewards (minimal), attention budget shifts toward GPS and away from agents.

### Road Graph — compensating for GPS absence
```
Basic:    0.476  ← highest road attention (compensates for no GPS guidance)
Complete: 0.419
Minimal:  0.402
```

Without GPS guidance, the basic model compensates by reading road geometry more intensively — trying to figure out where it can go.

### The Key Dynamic: Baseline vs. Reactivity
All three models respond to danger by increasing agent attention when risk rises. The Spearman correlations within s002 are all strongly positive (ρ > 0.76). This means **risk-reactive attention is a general property of collision-avoidance training**, not specific to the TTC reward.

What differs between models is not *whether* they react, but **from what baseline they react**, and **how strongly they respond to repeated threats**:
- Complete model: starts at 0.173, peaks at 0.27, responds clearly to both risk cycles
- Minimal model: starts at 0.117, peaks at 0.20, second cycle response is weak
- Basic model: episode terminates early (crashes at t=39) — the model can track danger but cannot navigate safely

---

## 7. The "Vigilance Prior" Finding

The most conceptually important result is visible in the two-model comparison figure (complete vs minimal on s002):

**The gap between the two agent-attention lines is present from t=0** — before any danger even appears. During the calm phase (t=42–65, risk=0), the complete model maintains 0.14–0.15 agent attention while the minimal model drops to 0.08–0.09.

This is not a reactive effect. It is a **learned prior**: the complete model, trained with continuous TTC penalties, has internalized that other agents are permanently important to monitor — even when the current situation is safe. The TTC reward fires not just at collision but at any proximity below 1.5 seconds — training the model to be continuously threat-aware, not just reactive.

**Intuition**: the TTC reward didn't change the model's reflexes. It changed its resting posture. The model with TTC training enters every timestep already more threat-aware than the one without.

---

## 8. What the Basic Model Reveals

The basic model crashes in 2 out of 3 scenarios (scenarios 1 and 2 terminate early at steps 38 and 39 respectively). This tells us that without navigation rewards, the model cannot reliably complete episodes — it has no incentive to stay on route and eventually collides.

A secondary finding from the basic model's action-conditioned attention:
- When braking: 72% of attention goes to road graph
- When steering: 59% road graph
- The basic model stares at road geometry when taking action — trying to understand the physical environment it has no GPS context for

This contrasts with the complete model during braking: road graph attention stays around 42% and agent attention is elevated. The complete model brakes *because of agents*; the basic model brakes *because of road geometry*.

---

## 9. Summary: What the Three Models Teach Us

| Question | Answer |
|----------|--------|
| Does attention track danger? | Yes, in all three models — risk-reactive agent attention is universal |
| Does reward design change the attention prior? | Yes, dramatically — GPS and agent baselines follow reward structure exactly |
| What is the unique effect of TTC reward? | Permanently elevated agent vigilance even during calm phases |
| What is the unique effect of navigation reward? | Near-doubling of GPS path attention |
| Does the basic model learn useful attention? | It tracks danger well but crashes — attention without navigation is insufficient |
| What makes a good scenario for this analysis? | Two or more distinct risk cycles with calm phases in between (s002) |
| Is pooled correlation across scenarios useful? | No — between-scenario heterogeneity confounds the estimate; use within-episode Fisher z |

---

## 10. Next Steps

1. **50-scenario overnight run** on complete model — validate s002 findings at scale
2. **50-scenario run** on minimal model — confirm the baseline gap persists across scenarios
3. **Event mining refactor** — score scenarios by risk variability, not event count
4. **Full paper draft** — the 3-way comparison figure is the centerpiece
