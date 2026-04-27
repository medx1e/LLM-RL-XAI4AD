"""SmoothGrad: reducing noise in gradient-based saliency.

Averages gradients over noisy copies of the input:

    SG(x) = (1/n) * sum_i grad f(x + eps_i),   eps_i ~ N(0, sigma^2)

Reference:
    Smilkov et al., "SmoothGrad: removing noise by adding noise" (2017)
"""

from typing import Optional

import jax
import jax.numpy as jnp

from posthoc_xai.methods.base import AttributionMethod
from posthoc_xai.models.base import ExplainableModel


class SmoothGrad(AttributionMethod):
    """SmoothGrad attribution.

    Args:
        model: ExplainableModel instance.
        n_samples: Number of noisy samples.
        noise_std: Standard deviation of Gaussian noise.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        model: ExplainableModel,
        n_samples: int = 50,
        noise_std: float = 0.1,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.n_samples = n_samples
        self.noise_std = noise_std
        self.seed = seed

    def compute_raw_attribution(
        self,
        observation: jnp.ndarray,
        target_action: Optional[int] = None,
    ) -> jnp.ndarray:
        key = jax.random.PRNGKey(self.seed)

        # Generate all noise samples
        noise = jax.random.normal(key, (self.n_samples,) + observation.shape)
        noisy_inputs = observation + self.noise_std * noise

        def _compute_gradient(noisy_obs: jnp.ndarray) -> jnp.ndarray:
            def forward_fn(x):
                return self.model.get_action_value(x, target_action)

            return jax.grad(forward_fn)(noisy_obs)

        all_gradients = jax.vmap(_compute_gradient)(noisy_inputs)
        return jnp.mean(all_gradients, axis=0)

    @property
    def name(self) -> str:
        return f"smooth_grad_{self.n_samples}samples"
