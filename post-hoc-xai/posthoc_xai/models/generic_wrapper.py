"""Generic wrapper for any V-MAX encoder type.

This wrapper works for all encoder architectures (MTR, Wayformer, MGAIL,
MLP/None) and implements the ExplainableModel interface.  It delegates
attention extraction to the Perceiver-specific wrapper for encoders that
have attention, and returns None for those that don't.
"""

from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from posthoc_xai.models.base import ExplainableModel, ModelOutput
from posthoc_xai.models._obs_structure import (
    compute_observation_structures,
    extract_entity_validity,
)


# Encoder types that have extractable attention weights
_ATTENTION_ENCODERS = {"lq", "perceiver", "mtr", "wayformer", "lqh", "mgail"}


class GenericWrapper(ExplainableModel):
    """ExplainableModel implementation that works for any V-MAX encoder.

    Args:
        loaded: A ``LoadedVMAXModel`` returned by the loader.
    """

    def __init__(self, loaded: Any):
        self._loaded = loaded
        self._policy_params = loaded.policy_params
        self._policy_module = loaded.policy_module
        self._unflatten_fn = loaded.unflatten_fn
        self._action_size = loaded.action_size
        self._obs_size = loaded.obs_size
        self._config = loaded.config
        self._encoder_type = loaded.encoder_type

        self._obs_structure, self._detailed_structure = (
            compute_observation_structures(self._unflatten_fn, self._obs_size)
        )

    # ------------------------------------------------------------------
    # ExplainableModel interface
    # ------------------------------------------------------------------

    def forward(self, observation: jnp.ndarray) -> ModelOutput:
        needs_batch = observation.ndim == 1
        obs = observation[None, :] if needs_batch else observation

        logits = self._policy_module.apply(self._policy_params, obs)

        action_mean = logits[..., : self._action_size]
        action_log_std = logits[..., self._action_size :]
        action_std = jnp.exp(action_log_std)

        if needs_batch:
            action_mean = action_mean[0]
            action_std = action_std[0]

        return ModelOutput(
            action_mean=action_mean,
            action_std=action_std,
        )

    def get_action_value(
        self,
        observation: jnp.ndarray,
        action_idx: Optional[int] = None,
    ) -> jnp.ndarray:
        obs = observation[None, :]
        logits = self._policy_module.apply(self._policy_params, obs)
        action_mean = logits[0, : self._action_size]

        if action_idx is not None:
            return action_mean[action_idx]
        return jnp.sum(action_mean)

    def get_embedding(self, observation: jnp.ndarray) -> jnp.ndarray:
        # For the generic wrapper, return the full logits as "embedding"
        # since we can't easily isolate the encoder output without capture_intermediates
        obs = observation[None, :] if observation.ndim == 1 else observation
        logits = self._policy_module.apply(self._policy_params, obs)
        if observation.ndim == 1:
            return logits[0]
        return logits

    @property
    def observation_structure(self) -> dict[str, tuple[int, int]]:
        return self._obs_structure

    @property
    def observation_structure_detailed(self) -> dict:
        return self._detailed_structure

    @property
    def has_attention(self) -> bool:
        return self._encoder_type in _ATTENTION_ENCODERS

    @property
    def name(self) -> str:
        return f"{self._encoder_type}_{self._loaded.original_encoder_type}"

    def get_entity_validity(
        self, observation: jnp.ndarray
    ) -> dict[str, dict[str, bool]]:
        return extract_entity_validity(self._unflatten_fn, observation)
