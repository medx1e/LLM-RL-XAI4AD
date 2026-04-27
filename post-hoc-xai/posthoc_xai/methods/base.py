"""Base classes for all attribution / XAI methods.

Every attribution method subclasses ``AttributionMethod`` and implements
``compute_raw_attribution``.  The ``__call__`` protocol then normalizes,
aggregates by observation category, and returns an ``Attribution`` object.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import jax.numpy as jnp

from posthoc_xai.models.base import ExplainableModel


@dataclass
class Attribution:
    """Standardized attribution result returned by every XAI method.

    Attributes:
        raw: Raw attribution values (same shape as the input observation).
        normalized: Absolute-value attribution normalized to sum to 1.
        category_importance: Per-category aggregated importance scores.
        entity_importance: Per-entity importance within each category.
            Dict of ``{category: {entity_name: importance}}``.
        method_name: Identifier of the method that produced this.
        target_action: Which action dimension was targeted (None = all).
        computation_time_ms: Wall-clock time for the computation.
        extras: Optional dict for method-specific intermediate results.
    """

    raw: jnp.ndarray
    normalized: jnp.ndarray
    category_importance: dict[str, float]
    entity_importance: dict[str, dict[str, float]]
    method_name: str
    target_action: Optional[int]
    computation_time_ms: float
    extras: Optional[dict[str, Any]] = field(default=None)


class AttributionMethod(ABC):
    """Base class for all attribution methods.

    Subclasses must implement:
    - ``compute_raw_attribution``
    - ``name`` (property)

    The ``__call__`` protocol normalizes the raw result and aggregates it
    by observation category automatically.
    """

    def __init__(self, model: ExplainableModel, **kwargs: Any):
        self.model = model
        self.config = kwargs

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_raw_attribution(
        self,
        observation: jnp.ndarray,
        target_action: Optional[int] = None,
    ) -> jnp.ndarray:
        """Compute raw attribution values.

        Args:
            observation: Input observation (unbatched, shape ``(obs_dim,)``).
            target_action: Which action dimension to explain (None → all).

        Returns:
            Attribution array with same shape as *observation*.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Method identifier string (e.g. ``'vanilla_gradient'``)."""

    # ------------------------------------------------------------------
    # Default normalization & aggregation – override if needed
    # ------------------------------------------------------------------

    def normalize(self, raw_attribution: jnp.ndarray) -> jnp.ndarray:
        """Normalize to sum-of-absolute-values = 1."""
        abs_attr = jnp.abs(raw_attribution)
        total = jnp.sum(abs_attr) + 1e-10
        return abs_attr / total

    def aggregate_by_category(
        self, normalized_attribution: jnp.ndarray
    ) -> dict[str, float]:
        """Sum normalized attribution per observation category."""
        structure = self.model.observation_structure
        flat = normalized_attribution.flatten()

        category_importance: dict[str, float] = {}
        for category, (start, end) in structure.items():
            category_importance[category] = float(jnp.sum(flat[start:end]))
        return category_importance

    def aggregate_by_entity(
        self, normalized_attribution: jnp.ndarray
    ) -> dict[str, dict[str, float]]:
        """Sum normalized attribution per entity within each category."""
        detailed = self.model.observation_structure_detailed
        flat = normalized_attribution.flatten()

        entity_importance: dict[str, dict[str, float]] = {}
        for category, info in detailed.items():
            entities = info["entities"]
            cat_entities: dict[str, float] = {}
            for entity_name, (start, end) in entities.items():
                cat_entities[entity_name] = float(jnp.sum(flat[start:end]))
            entity_importance[category] = cat_entities
        return entity_importance

    # ------------------------------------------------------------------
    # __call__ protocol
    # ------------------------------------------------------------------

    def __call__(
        self,
        observation: jnp.ndarray,
        target_action: Optional[int] = None,
    ) -> Attribution:
        """Run the full attribution pipeline.

        1. Compute raw attribution.
        2. Normalize.
        3. Aggregate by category.
        4. Return an ``Attribution`` result object.
        """
        import time

        t0 = time.time()
        raw = self.compute_raw_attribution(observation, target_action)
        normalized = self.normalize(raw)
        category_importance = self.aggregate_by_category(normalized)
        entity_importance = self.aggregate_by_entity(normalized)
        elapsed_ms = (time.time() - t0) * 1000.0

        return Attribution(
            raw=raw,
            normalized=normalized,
            category_importance=category_importance,
            entity_importance=entity_importance,
            method_name=self.name,
            target_action=target_action,
            computation_time_ms=elapsed_ms,
        )
