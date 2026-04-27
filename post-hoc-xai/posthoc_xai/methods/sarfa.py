"""SARFA: Specific and Relevant Feature Attribution.

RL-specific saliency method that considers both:

- **Relevance**: Does the feature affect the chosen action's Q-proxy?
- **Specificity**: Does the feature specifically affect *this* action vs others?

    SARFA(f, a) = Relevance(f, a) × Specificity(f, a)

Per Puri et al. (2020) Sec. 3.2, specificity is:

    Specificity = 1 − H(|ΔQ| / ‖ΔQ‖₁) / log(|A|)

where H is Shannon entropy of the normalised change distribution across all
action dimensions.  Specificity → 1 when only the target action is affected,
→ 0 when all actions change equally.

Adaptation for SAC continuous control
──────────────────────────────────────
The original SARFA targets DQN with discrete Q-values per action.  For SAC
(continuous, 2-D action: acceleration + steering) we substitute the policy's
action means for Q-values.  Using action means is the standard adaptation for
continuous-action SARFA (Puri et al. note this extension).

With only 2 action dimensions specificity is coarse (binary: either one or
both dimensions change), but the cross-action discriminability is still
meaningful for separating acceleration-relevant from steering-relevant features.

Performance
──────────────────────────────────────
``compute_raw_attribution`` (single timestep, per_category=True): 5 forward
passes — used by the standard framework __call__ pipeline.

``sarfa_batch`` (standalone function): optimised for Phase 3 scale.  Runs
exactly 6 batched forward passes (1 baseline + 5 category perturbations) over
all T timesteps in one JIT call each — no capture_intermediates overhead.
Use this for bulk experiments.

Reference:
    Puri et al., "Explain Your Move: Understanding Agent Actions Using
    Focused Feature Saliency", ICLR 2020.
"""

from typing import Literal, Optional

import jax
import jax.numpy as jnp
import numpy as np

from posthoc_xai.methods.base import AttributionMethod
from posthoc_xai.models.base import ExplainableModel


# ---------------------------------------------------------------------------
# Shared specificity kernel
# ---------------------------------------------------------------------------

def _entropy_specificity(all_changes: jnp.ndarray) -> jnp.ndarray:
    """Compute specificity via normalised Shannon entropy.

    Args:
        all_changes: |ΔQ| per action dim, shape ``(A,)``.

    Returns:
        Scalar specificity ∈ [0, 1].
    """
    n_actions = all_changes.shape[0]
    total     = jnp.sum(all_changes) + 1e-10
    dist      = all_changes / total
    entropy   = -jnp.sum(dist * jnp.log(dist + 1e-10))
    max_ent   = jnp.log(jnp.array(n_actions, dtype=jnp.float32))
    return 1.0 - entropy / (max_ent + 1e-10)


# ---------------------------------------------------------------------------
# Standalone batched SARFA — use this for Phase 3 scale experiments
# ---------------------------------------------------------------------------

def sarfa_batch(
    model,
    raw_obs: np.ndarray,
    perturbation_type: Literal["zero", "mean"] = "zero",
    target_action: int = 0,
) -> np.ndarray:
    """Fully-batched, category-level SARFA for all T timesteps at once.

    Runs exactly 6 batched forward passes (1 baseline + 5 category
    perturbations) with no ``capture_intermediates`` overhead.  This is
    10–50× faster than calling ``SARFA.__call__`` in a Python loop.

    Args:
        model: Loaded PerceiverWrapper / GenericWrapper.
        raw_obs: Shape ``(T, obs_dim)`` — all episode observations.
        perturbation_type: ``'zero'`` or ``'mean'``.
        target_action: Which action dimension is the primary target
            (0 = acceleration, 1 = steering).  Used for relevance.
            Specificity considers all action dimensions regardless.

    Returns:
        Numpy array ``(T, n_categories)`` of SARFA scores, normalised
        so each row sums to 1.  Column order matches
        ``model.observation_structure``.
    """
    params      = model._policy_params
    module      = model._policy_module
    action_size = model._action_size
    obs_struct  = model.observation_structure
    T           = raw_obs.shape[0]
    n_cats      = len(obs_struct)

    def get_action_means(obs_batch: jnp.ndarray) -> jnp.ndarray:
        """(T, obs_dim) → (T, action_size).  No capture_intermediates."""
        logits = module.apply(params, obs_batch)          # (T, 2*action_size)
        return logits[:, :action_size]                    # (T, action_size)

    obs_batch        = jnp.array(raw_obs)                 # (T, obs_dim)
    baseline_actions = get_action_means(obs_batch)        # (T, action_size)

    result = np.zeros((T, n_cats), dtype=np.float32)

    for col_idx, (_cat, (start, end)) in enumerate(obs_struct.items()):
        # Build perturbed batch — zero or mean-fill the category slice
        if perturbation_type == "zero":
            fill = jnp.zeros((T, end - start))
        else:
            fill = jnp.broadcast_to(
                obs_batch[:, start:end].mean(axis=1, keepdims=True),
                (T, end - start),
            )
        perturbed_batch     = obs_batch.at[:, start:end].set(fill)
        perturbed_actions   = get_action_means(perturbed_batch)     # (T, action_size)

        # Relevance: |Δ target action| per timestep  →  (T,)
        delta_all  = jnp.abs(baseline_actions - perturbed_actions)  # (T, action_size)
        relevance  = delta_all[:, target_action]                     # (T,)

        # Specificity: entropy-based, computed per timestep via vmap  →  (T,)
        specificity = jax.vmap(_entropy_specificity)(delta_all)      # (T,)

        result[:, col_idx] = np.array(relevance * specificity)

    # Normalise rows to sum to 1 (same convention as other methods)
    row_sums = result.sum(axis=1, keepdims=True) + 1e-10
    return result / row_sums   # (T, n_cats)


# ---------------------------------------------------------------------------
# AttributionMethod wrapper — used by the standard __call__ pipeline
# ---------------------------------------------------------------------------

class SARFA(AttributionMethod):
    """SARFA attribution (single-observation interface).

    For bulk experiments use ``sarfa_batch()`` directly — it is much faster.

    Args:
        model: ExplainableModel instance.
        perturbation_type: ``'zero'`` or ``'mean'``.
        per_category: Must be ``True``.  Per-feature SARFA (``False``) via
            ``jax.vmap`` over ``model.forward()`` is unsafe — ``forward()``
            uses ``capture_intermediates`` which does not compose with vmap.
    """

    def __init__(
        self,
        model: ExplainableModel,
        perturbation_type: Literal["zero", "mean"] = "zero",
        per_category: bool = True,
        **kwargs,
    ):
        if not per_category:
            raise ValueError(
                "per_category=False is disabled: jax.vmap over model.forward() "
                "conflicts with capture_intermediates.  Use per_category=True."
            )
        super().__init__(model, **kwargs)
        self.perturbation_type = perturbation_type

    def compute_raw_attribution(
        self,
        observation: jnp.ndarray,
        target_action: Optional[int] = None,
    ) -> jnp.ndarray:
        """Category-level SARFA for one observation.

        Performs 6 forward passes (1 baseline + 5 categories).
        For bulk use across many timesteps, call ``sarfa_batch()`` instead.
        """
        if target_action is None:
            target_action = 0

        baseline_output  = self.model.forward(observation)
        baseline_actions = baseline_output.action_mean     # (action_size,)

        flat_obs    = observation.flatten()
        attribution = jnp.zeros_like(flat_obs)

        for _cat, (start, end) in self.model.observation_structure.items():
            if self.perturbation_type == "zero":
                perturbed_val = jnp.zeros(end - start)
            else:
                perturbed_val = jnp.full(end - start, jnp.mean(flat_obs[start:end]))

            perturbed        = flat_obs.at[start:end].set(perturbed_val)
            perturbed_output = self.model.forward(perturbed.reshape(observation.shape))
            perturbed_actions = perturbed_output.action_mean

            relevance   = jnp.abs(
                baseline_actions[target_action] - perturbed_actions[target_action]
            )
            all_changes = jnp.abs(baseline_actions - perturbed_actions)
            specificity = _entropy_specificity(all_changes)
            sarfa_score = relevance * specificity

            # Distribute the category score uniformly across its features
            per_feature = sarfa_score / max(end - start, 1)
            attribution = attribution.at[start:end].set(per_feature)

        return attribution.reshape(observation.shape)

    @property
    def name(self) -> str:
        return "sarfa"
