"""Abstract base class for explainable V-MAX models.

All model wrappers must implement the ExplainableModel interface
so that XAI methods can work uniformly across encoder architectures.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import jax.numpy as jnp


class ModelOutput:
    """Standardized output from a model forward pass.

    Attributes:
        action_mean: Mean of the action distribution (action_dim,) or (batch, action_dim).
        action_std: Std of the action distribution (same shape as action_mean).
        value: V(s) if available, else None.
        embedding: Encoder output vector (for probing / downstream use).
        attention: Dict of attention weight arrays if available, else None.
            Expected keys: 'cross_attention', 'self_attention'.
    """

    __slots__ = ("action_mean", "action_std", "value", "embedding", "attention")

    def __init__(
        self,
        action_mean: jnp.ndarray,
        action_std: jnp.ndarray,
        value: Optional[jnp.ndarray] = None,
        embedding: Optional[jnp.ndarray] = None,
        attention: Optional[dict[str, jnp.ndarray]] = None,
    ):
        self.action_mean = action_mean
        self.action_std = action_std
        self.value = value
        self.embedding = embedding
        self.attention = attention


class ExplainableModel(ABC):
    """Abstract interface that every model wrapper must implement.

    The interface provides:
    - ``forward``: full forward pass returning a ``ModelOutput``
    - ``get_action_value``: scalar output suitable for ``jax.grad``
    - ``get_embedding``: encoder output only
    - ``get_attention``: attention weights (optional)
    - ``observation_structure``: maps category names to index ranges in the flat obs
    """

    # ------------------------------------------------------------------
    # Abstract methods – must be implemented by every wrapper
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(self, observation: jnp.ndarray) -> ModelOutput:
        """Full forward pass.

        Args:
            observation: Flat observation array, shape ``(batch, obs_dim)``.

        Returns:
            A ``ModelOutput`` instance.
        """

    @abstractmethod
    def get_action_value(
        self,
        observation: jnp.ndarray,
        action_idx: Optional[int] = None,
    ) -> jnp.ndarray:
        """Return a *scalar* value suitable for gradient computation.

        For continuous actions this is typically ``action_mean[action_idx]``
        (or the sum over all dims when *action_idx* is ``None``).

        This is what we differentiate w.r.t. the input for saliency.

        Args:
            observation: Flat observation array, shape ``(obs_dim,)`` (unbatched).
            action_idx: Which action dimension to target (``None`` → sum all).

        Returns:
            A scalar ``jnp.ndarray``.
        """

    @abstractmethod
    def get_embedding(self, observation: jnp.ndarray) -> jnp.ndarray:
        """Return the encoder output (latent representation).

        Args:
            observation: Flat observation array.

        Returns:
            Embedding array (e.g. 256-dim for most V-MAX encoders).
        """

    @property
    @abstractmethod
    def observation_structure(self) -> dict[str, tuple[int, int]]:
        """Map category names to ``(start_idx, end_idx)`` in the flat observation.

        Example::

            {
                'sdc_trajectory': (0, 35),
                'other_agents': (35, 315),
                'roadgraph': (315, 1115),
                'traffic_lights': (1115, 1365),
                'gps_path': (1365, 1385),
            }
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this model."""

    # ------------------------------------------------------------------
    # Optional – override when the model has attention
    # ------------------------------------------------------------------

    def get_attention(
        self, observation: jnp.ndarray
    ) -> Optional[dict[str, jnp.ndarray]]:
        """Return attention weights (if the architecture has attention).

        Returns ``None`` by default. Override in attention-based wrappers.
        """
        return None

    @property
    def has_attention(self) -> bool:
        """Whether the encoder exposes extractable attention weights."""
        return False

    # ------------------------------------------------------------------
    # Per-entity observation structure
    # ------------------------------------------------------------------

    @property
    def observation_structure_detailed(self) -> dict[str, dict]:
        """Per-entity index ranges within each observation category.

        Returns a dict like::

            {
                "other_agents": {
                    "num_entities": 8,
                    "features_per_entity": 40,
                    "entities": {
                        "agent_0": (40, 80),
                        "agent_1": (80, 120),
                        ...
                    },
                },
                ...
            }

        The default implementation derives this from ``observation_structure``
        and ``_entity_layout``.  Wrappers that pre-compute ``_detailed_structure``
        in ``__init__`` can override for speed.
        """
        if hasattr(self, "_detailed_structure"):
            return self._detailed_structure
        raise NotImplementedError(
            "Wrapper must compute _detailed_structure in __init__"
        )

    def get_entity_validity(
        self, observation: jnp.ndarray
    ) -> dict[str, dict[str, bool]]:
        """Return per-entity validity flags from a real observation.

        Args:
            observation: Flat observation, shape ``(obs_dim,)``.

        Returns:
            Dict mapping category → {entity_name: is_valid}.
        """
        raise NotImplementedError("Wrapper must implement get_entity_validity")
