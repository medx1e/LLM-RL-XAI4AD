"""Integrated Gradients attribution.

Computes attribution by integrating gradients along a straight-line path
from a baseline to the input:

    IG(x) = (x - x') * integral_0^1 grad f(x' + alpha*(x-x')) d_alpha

Satisfies the *completeness* and *sensitivity* axioms.

The integral is approximated with the **trapezoidal rule** over ``n_steps``
intervals (``n_steps + 1`` evaluation points), which is more accurate than a
simple Riemann sum and reduces the end-point bias present when alpha=0 or
alpha=1 produce very different gradients.

Reference:
    Sundararajan et al., "Axiomatic Attribution for Deep Networks" (2017)
"""

from typing import Optional, Union

import numpy as np
import jax
import jax.numpy as jnp

from posthoc_xai.methods.base import AttributionMethod
from posthoc_xai.models.base import ExplainableModel


class IntegratedGradients(AttributionMethod):
    """Integrated Gradients (IG).

    Args:
        model: ExplainableModel instance.
        n_steps: Number of integration intervals (more → more accurate).
            Gradients are evaluated at ``n_steps + 1`` points.
        baseline: Baseline — one of:
            ``'zero'``          all-zeros (default, semantically wrong for V-MAX)
            ``'noise'``         small Gaussian noise
            ``jnp.ndarray``     custom precomputed baseline (recommended)
            ``numpy.ndarray``   same, converted to jnp automatically

            For V-MAX, use ``posthoc_xai.utils.compute_baseline(obs_array)``
            to get a validity-zeroed mean baseline that represents an empty
            scene — semantically sound for normalized driving observations.
    """

    def __init__(
        self,
        model: ExplainableModel,
        n_steps: int = 50,
        baseline: Union[str, jnp.ndarray, np.ndarray] = "zero",
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.n_steps = n_steps
        # Pre-convert numpy arrays to jnp once at construction time
        if isinstance(baseline, np.ndarray):
            self.baseline_type = jnp.array(baseline)
        else:
            self.baseline_type = baseline

    def _get_baseline(self, observation: jnp.ndarray) -> jnp.ndarray:
        if self.baseline_type == "zero":
            return jnp.zeros_like(observation)
        elif self.baseline_type == "noise":
            key = jax.random.PRNGKey(0)
            return jax.random.normal(key, observation.shape) * 0.01
        elif isinstance(self.baseline_type, jnp.ndarray):
            return self.baseline_type
        else:
            return jnp.zeros_like(observation)

    def compute_raw_attribution(
        self,
        observation: jnp.ndarray,
        target_action: Optional[int] = None,
    ) -> jnp.ndarray:
        baseline = self._get_baseline(observation)
        # n_steps intervals → n_steps+1 evaluation points (0 and 1 inclusive)
        alphas = jnp.linspace(0.0, 1.0, self.n_steps + 1)

        def _grad_at_alpha(alpha: jnp.ndarray) -> jnp.ndarray:
            interpolated = baseline + alpha * (observation - baseline)

            def forward_fn(x):
                return self.model.get_action_value(x, target_action)

            return jax.grad(forward_fn)(interpolated)

        # Compute all gradients in parallel: shape (n_steps+1, obs_dim)
        path_gradients = jax.vmap(_grad_at_alpha)(alphas)

        # Trapezoidal rule: (g0 + 2*g1 + ... + 2*g(n-1) + gn) / (2*n_steps)
        # This has lower end-point error than a plain mean (Riemann sum).
        interior_sum = jnp.sum(path_gradients[1:-1], axis=0)
        avg_gradients = (
            path_gradients[0] + 2.0 * interior_sum + path_gradients[-1]
        ) / (2.0 * self.n_steps)

        # Multiply by (x - baseline) — this is the "completeness" scaling
        return (observation - baseline) * avg_gradients

    @property
    def name(self) -> str:
        return f"integrated_gradients_{self.n_steps}steps"
