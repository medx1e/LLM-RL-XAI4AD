"""Gradient x Input attribution.

Element-wise product of the input and its gradient:

    GxI(x) = x * grad f(x)

Often more meaningful than raw gradients for non-zero inputs.

Reference:
    Shrikumar et al., "Learning Important Features Through
    Propagating Activation Differences" (2017)
"""

from typing import Optional

import jax
import jax.numpy as jnp

from posthoc_xai.methods.base import AttributionMethod
from posthoc_xai.models.base import ExplainableModel


class GradientXInput(AttributionMethod):
    """Gradient x Input attribution: ``x * grad f(x)``."""

    def __init__(self, model: ExplainableModel, **kwargs):
        super().__init__(model, **kwargs)

    def compute_raw_attribution(
        self,
        observation: jnp.ndarray,
        target_action: Optional[int] = None,
    ) -> jnp.ndarray:
        def forward_fn(obs):
            return self.model.get_action_value(obs, target_action)

        gradient = jax.grad(forward_fn)(observation)
        return observation * gradient

    @property
    def name(self) -> str:
        return "gradient_x_input"
