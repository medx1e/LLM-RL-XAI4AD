"""Vanilla gradient saliency.

Computes ``∂f(x)/∂x`` where *f* maps an observation to a scalar action value.

Simple, fast, and the building block for many other gradient-based methods.

Reference:
    Simonyan et al., "Deep Inside Convolutional Networks" (2014)
"""

from typing import Optional

import jax
import jax.numpy as jnp

from posthoc_xai.methods.base import AttributionMethod
from posthoc_xai.models.base import ExplainableModel


class VanillaGradient(AttributionMethod):
    """Vanilla gradient attribution: ``∂f(x)/∂x``."""

    def __init__(self, model: ExplainableModel, **kwargs):
        super().__init__(model, **kwargs)

    def compute_raw_attribution(
        self,
        observation: jnp.ndarray,
        target_action: Optional[int] = None,
    ) -> jnp.ndarray:
        """Compute the gradient of the action value w.r.t. the input.

        Args:
            observation: Flat observation, shape ``(obs_dim,)``.
            target_action: Which action dim to differentiate (None → sum).

        Returns:
            Gradient array with same shape as *observation*.
        """

        def forward_fn(obs: jnp.ndarray) -> jnp.ndarray:
            return self.model.get_action_value(obs, target_action)

        gradient = jax.grad(forward_fn)(observation)
        return gradient

    @property
    def name(self) -> str:
        return "vanilla_gradient"
