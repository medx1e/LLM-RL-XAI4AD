# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Code source: https://github.com/google/brax/blob/main/brax/training/distribution.py

"""Probability distributions in JAX."""

import abc

import distrax
import jax
import jax.numpy as jnp


class ParametricDistribution(abc.ABC):
    """Abstract class for parametric (action) distribution."""

    def __init__(self, param_size, postprocessor, event_ndims, reparametrizable):
        """Abstract class for parametric (action) distribution.

        Specifies how to transform distribution parameters (i.e. policy output)
        into a distribution over actions.

        Args:
          param_size: size of the parameters for the distribution
          postprocessor: bijector which is applied after sampling (in practice, it's
            tanh or identity)
          event_ndims: rank of the distribution sample (i.e. action)
          reparametrizable: is the distribution reparametrizable

        """
        self._param_size = param_size
        self._postprocessor = postprocessor
        self._event_ndims = event_ndims  # rank of events
        self._reparametrizable = reparametrizable
        assert event_ndims in [0, 1]

    @abc.abstractmethod
    def create_dist(self, parameters):
        """Creates distribution from parameters."""

    @property
    def param_size(self):
        return self._param_size

    @property
    def reparametrizable(self):
        return self._reparametrizable

    def postprocess(self, event):
        return self._postprocessor.forward(event)

    def inverse_postprocess(self, event):
        return self._postprocessor.inverse(event)

    def sample_no_postprocessing(self, parameters, seed):
        return self.create_dist(parameters).sample(seed=seed)

    def sample(self, parameters, seed):
        """Returns a sample from the postprocessed distribution."""
        return self.postprocess(self.sample_no_postprocessing(parameters, seed))

    def mode(self, parameters):
        """Returns the mode of the postprocessed distribution."""
        return self.postprocess(self.create_dist(parameters).mode())

    def log_prob(self, parameters, actions):
        """Compute the log probability of actions."""
        dist = self.create_dist(parameters)
        log_probs = dist.log_prob(actions)
        log_probs -= self._postprocessor.forward_log_det_jacobian(actions)

        if self._event_ndims == 1:
            log_probs = jnp.sum(log_probs, axis=-1)  # sum over action dimension

        return log_probs

    def entropy(self, parameters, seed):
        """Return the entropy of the given distribution."""
        dist = self.create_dist(parameters)
        entropy = dist.entropy()
        entropy += self._postprocessor.forward_log_det_jacobian(dist.sample(seed=seed))

        if self._event_ndims == 1:
            entropy = jnp.sum(entropy, axis=-1)

        return entropy


class NormalDistribution:
    """Normal distribution."""

    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def sample(self, seed):
        return jax.random.normal(seed, shape=self.loc.shape) * self.scale + self.loc

    def mode(self):
        return self.loc

    def log_prob(self, x):
        log_unnormalized = -0.5 * jnp.square(x / self.scale - self.loc / self.scale)
        log_normalization = 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(self.scale)

        return log_unnormalized - log_normalization

    def entropy(self):
        log_normalization = 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(self.scale)
        entropy = 0.5 + log_normalization

        return entropy * jnp.ones_like(self.loc)


class TanhBijector:
    """Tanh Bijector."""

    def forward(self, x):
        return jnp.tanh(x)

    def inverse(self, y):
        return jnp.arctanh(y)

    def forward_log_det_jacobian(self, x):
        return 2.0 * (jnp.log(2.0) - x - jax.nn.softplus(-2.0 * x))


class NormalTanhDistribution(ParametricDistribution):
    """Normal distribution followed by tanh."""

    def __init__(self, event_size, min_std=0.001):
        """Initialize the distribution.

        Args:
          event_size: the size of events (i.e. actions).
          min_std: minimum std for the gaussian.

        """
        # We apply tanh to gaussian actions to bound them.
        # Normally we would use TransformedDistribution to automatically
        # apply tanh to the distribution.
        # We can't do it here because of tanh saturation
        # which would make log_prob computations impossible. Instead, most
        # of the code operate on pre-tanh actions and we take the postprocessor
        # jacobian into account in log_prob computations.
        super().__init__(param_size=2 * event_size, postprocessor=TanhBijector(), event_ndims=1, reparametrizable=True)
        self._min_std = min_std

    def create_dist(self, parameters):
        loc, scale = jnp.split(parameters, 2, axis=-1)
        scale = jax.nn.softplus(scale) + self._min_std

        return NormalDistribution(loc=loc, scale=scale)


class AffineBijector:
    """Affine bijector that map the support of the beta from [0,1] to [-1,1]."""

    affine_bijector = distrax.Lambda(lambda x: 2 * x - 1)

    def forward(self, x):
        return self.affine_bijector.forward(x)

    def inverse(self, y):
        return self.affine_bijector.inverse(y)

    def forward_log_det_jacobian(self, x):
        return self.affine_bijector.forward_log_det_jacobian(x)


class Beta:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.key = jax.random.PRNGKey(0)
        self.epsilon = 1e-6

    def generate_key(self):
        self.key, subkey = jax.random.split(self.key)

        return subkey

    def sample(self, seed):
        sample = jax.random.beta(seed, self.alpha, self.beta, shape=self.alpha.shape)

        return jnp.clip(sample, self.epsilon, 1.0 - self.epsilon)

    def mode(self):
        # default value should not be used
        default = 0.5
        # returnning a value of 1 or 0 will cause the log_prob to return -inf
        epsilon = 1e-6

        # case if alpha, beta > 1
        mask = (self.alpha > 1) & (self.beta > 1)
        mode = jnp.where(mask, (self.alpha - 1) / (self.alpha + self.beta - 2), 0.5)

        # case if alpha = beta = 1 (uniform distribution)
        mask = (self.alpha == 1) & (self.beta == 1)
        mode = jnp.where(mask, default, mode)

        # case if alpha >= 1, beta <= 1, alpha != beta
        mask = (self.alpha >= 1) & (self.beta <= 1) & (self.alpha != self.beta)
        mode = jnp.where(mask, 1.0 - epsilon, mode)

        # case if alpha <= 1, beta >= 1, alpha != beta
        mask = (self.alpha <= 1) & (self.beta >= 1) & (self.alpha != self.beta)
        mode = jnp.where(mask, epsilon, mode)

        # case if alpha, beta < 1 (u shape distribution with 2 modes 0 and 1)
        mask = (self.alpha < 1) & (self.beta < 1)
        key = self.generate_key()
        bernouilli = jax.random.bernoulli(key, 0.5, shape=self.alpha.shape)
        bernouilli = jnp.where(bernouilli == 1, 1 - epsilon, epsilon)
        mode = jnp.where(mask, bernouilli, mode)

        return mode

    def log_prob(self, x):
        return distrax.Beta(self.alpha, self.beta).log_prob(x)

    def entropy(self):
        return distrax.Beta(self.alpha, self.beta).entropy()


class BetaDistribution(ParametricDistribution):
    def __init__(self, event_size):
        """Initialize the distribution.

        Args:
          event_size: the size of events (i.e. actions).
          min_std: minimum std for the gaussian.

        """
        super().__init__(
            param_size=2 * event_size,
            postprocessor=AffineBijector(),
            event_ndims=1,
            reparametrizable=True,
        )

    def create_dist(self, parameters, is_unimodal=True):
        alpha, beta = jnp.split(parameters, 2, axis=-1)
        # Ensure unimodal distrbution with alpha, beta > 1
        alpha = jax.nn.softplus(alpha) + is_unimodal
        beta = jax.nn.softplus(beta) + is_unimodal

        return Beta(alpha, beta)
