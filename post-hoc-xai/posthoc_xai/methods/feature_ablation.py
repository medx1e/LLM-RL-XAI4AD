"""Feature ablation by observation category.

Measures the importance of entire feature categories (trajectory, roadgraph,
traffic lights, GPS path) by removing each one and observing the output change:

    Importance(category) = |f(x) - f(x without category)|

Provides a high-level "what type of information matters" analysis.

Implementation note:
    ``compute_raw_attribution`` divides the category importance by the number
    of features in that category so that the resulting per-feature values are
    independent of category size.  After normalization in the base-class
    ``__call__``, ``category_importance`` will then correctly reflect the
    raw ablation delta for each category rather than being inflated for
    larger categories.  This matches the behaviour of ``PerturbationAttribution``.
"""

from typing import Literal, Optional

import jax.numpy as jnp

from posthoc_xai.methods.base import AttributionMethod
from posthoc_xai.models.base import ExplainableModel


class FeatureAblation(AttributionMethod):
    """Per-category feature ablation.

    Args:
        model: ExplainableModel instance.
        replacement: How to replace ablated features (``'zero'`` or ``'mean'``).
    """

    def __init__(
        self,
        model: ExplainableModel,
        replacement: Literal["zero", "mean"] = "zero",
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.replacement = replacement

    def compute_category_importance(
        self,
        observation: jnp.ndarray,
        target_action: Optional[int] = None,
    ) -> dict[str, float]:
        """Directly compute per-category importance (more interpretable).

        Returns a dict mapping category name to the scalar output change
        caused by ablating that category.
        """
        baseline_val = self.model.get_action_value(observation, target_action)
        flat_obs = jnp.array(observation).flatten()
        importance: dict[str, float] = {}

        for category, (start, end) in self.model.observation_structure.items():
            if self.replacement == "zero":
                replacement_vals = jnp.zeros(end - start)
            else:
                replacement_vals = jnp.full(end - start, jnp.mean(flat_obs[start:end]))

            ablated = flat_obs.at[start:end].set(replacement_vals)
            ablated_obs = ablated.reshape(observation.shape)
            ablated_val = self.model.get_action_value(ablated_obs, target_action)
            importance[category] = float(jnp.abs(baseline_val - ablated_val))

        return importance

    def compute_raw_attribution(
        self,
        observation: jnp.ndarray,
        target_action: Optional[int] = None,
    ) -> jnp.ndarray:
        """Build a per-feature attribution array from category-level ablation.

        Each feature in a category receives ``category_importance / n_features``
        so that larger categories are not disproportionately weighted after
        the normalization step in the base-class ``__call__``.
        """
        cat_importance = self.compute_category_importance(observation, target_action)
        flat_obs = jnp.array(observation).flatten()
        attribution = jnp.zeros_like(flat_obs)

        for category, (start, end) in self.model.observation_structure.items():
            n_features = max(end - start, 1)
            per_feature = cat_importance[category] / n_features
            attribution = attribution.at[start:end].set(per_feature)

        return attribution.reshape(observation.shape)

    @property
    def name(self) -> str:
        return f"feature_ablation_{self.replacement}"
