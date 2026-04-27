"""Perturbation-based (occlusion) attribution.

Measures feature importance by observing the output change when input
features are masked or replaced:

    Importance(i) = |f(x) - f(x with feature i masked)|

Model-agnostic but slower than gradient methods for high-dimensional inputs.

Reference:
    Zeiler & Fergus, "Visualizing and Understanding CNNs" (2014)
"""

from typing import Literal, Optional

import jax
import jax.numpy as jnp

from posthoc_xai.methods.base import AttributionMethod
from posthoc_xai.models.base import ExplainableModel


class PerturbationAttribution(AttributionMethod):
    """Perturbation-based attribution.

    Args:
        model: ExplainableModel instance.
        perturbation_type: How to replace masked features (``'zero'``,
            ``'mean'``, ``'noise'``).
        per_category: If ``True``, perturb entire categories at once
            (much faster). If ``False``, perturb each feature individually.
    """

    def __init__(
        self,
        model: ExplainableModel,
        perturbation_type: Literal["zero", "mean", "noise"] = "zero",
        per_category: bool = True,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.perturbation_type = perturbation_type
        self.per_category = per_category

    def _replacement_value(self, segment: jnp.ndarray) -> jnp.ndarray:
        if self.perturbation_type == "zero":
            return jnp.zeros_like(segment)
        elif self.perturbation_type == "mean":
            return jnp.full_like(segment, jnp.mean(segment))
        elif self.perturbation_type == "noise":
            key = jax.random.PRNGKey(0)
            return jax.random.normal(key, segment.shape) * 0.01
        return jnp.zeros_like(segment)

    def compute_raw_attribution(
        self,
        observation: jnp.ndarray,
        target_action: Optional[int] = None,
    ) -> jnp.ndarray:
        baseline_val = self.model.get_action_value(observation, target_action)
        flat_obs = observation.flatten()
        attribution = jnp.zeros_like(flat_obs)

        if self.per_category:
            for _category, (start, end) in self.model.observation_structure.items():
                perturbed = flat_obs.at[start:end].set(
                    self._replacement_value(flat_obs[start:end])
                )
                perturbed_obs = perturbed.reshape(observation.shape)
                perturbed_val = self.model.get_action_value(perturbed_obs, target_action)
                importance = jnp.abs(baseline_val - perturbed_val)
                per_feature = importance / max(end - start, 1)
                attribution = attribution.at[start:end].set(per_feature)
        else:
            # Per-feature perturbation (slow but precise)
            def _single_importance(idx):
                perturbed = flat_obs.at[idx].set(
                    self._replacement_value(flat_obs[idx:idx+1])[0]
                )
                perturbed_obs = perturbed.reshape(observation.shape)
                perturbed_val = self.model.get_action_value(perturbed_obs, target_action)
                return jnp.abs(baseline_val - perturbed_val)

            indices = jnp.arange(len(flat_obs))
            attribution = jax.vmap(_single_importance)(indices)

        return attribution.reshape(observation.shape)

    @property
    def name(self) -> str:
        mode = "category" if self.per_category else "feature"
        return f"perturbation_{self.perturbation_type}_{mode}"
