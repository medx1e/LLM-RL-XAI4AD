"""CBM configuration.

V2 change: binary_concept_indices and continuous_concept_indices are now
auto-derived from the concept registry at access time, rather than
hardcoded.  This means adding new concepts to the registry is the ONLY
step needed — CBMConfig adapts automatically.
"""

from __future__ import annotations

import dataclasses
from typing import Literal


@dataclasses.dataclass(frozen=True)
class CBMConfig:
    """Configuration for the Concept Bottleneck Model.

    Controls bottleneck dimensions, concept loss weighting, and
    frozen-vs-joint training mode.
    """

    # Number of concept outputs (must match active registry size)
    num_concepts: int = 11

    # Hidden layer sizes for the concept head (encoder_output → concepts)
    concept_head_hidden_sizes: tuple[int, ...] = (64,)

    # Hidden layer sizes for the actor FC after bottleneck
    actor_hidden_sizes: tuple[int, ...] = (64, 32)

    # Hidden layer sizes for the critic FC after bottleneck
    critic_hidden_sizes: tuple[int, ...] = (64, 32)

    # Weight of concept supervision loss in total loss
    lambda_concept: float = 0.1

    # Training mode: "frozen" = encoder frozen, heads train;
    #                "joint" = everything trains together
    mode: Literal["frozen", "joint"] = "frozen"

    # Number of twin Q-networks for the critic
    num_critics: int = 2

    # Whether critic networks share the encoder
    shared_encoder: bool = False

    # Which concept phases to use from the registry
    concept_phases: tuple[int, ...] = (1, 2)

    # ── Auto-derived properties ──────────────────────────────────────

    @property
    def concept_names(self) -> tuple[str, ...]:
        """Return ordered concept names from the active registry phases."""
        from concepts.registry import CONCEPT_REGISTRY
        return tuple(
            name for name, (schema, _) in CONCEPT_REGISTRY.items()
            if schema.phase in self.concept_phases
        )

    @property
    def binary_concept_indices(self) -> tuple[int, ...]:
        """Indices of binary concepts in the active concept set."""
        from concepts.registry import CONCEPT_REGISTRY
        from concepts.schema import ConceptType
        active = [
            (name, schema)
            for name, (schema, _) in CONCEPT_REGISTRY.items()
            if schema.phase in self.concept_phases
        ]
        return tuple(
            i for i, (_, schema) in enumerate(active)
            if schema.concept_type == ConceptType.BINARY
        )

    @property
    def continuous_concept_indices(self) -> tuple[int, ...]:
        """Indices of continuous concepts in the active concept set."""
        from concepts.registry import CONCEPT_REGISTRY
        from concepts.schema import ConceptType
        active = [
            (name, schema)
            for name, (schema, _) in CONCEPT_REGISTRY.items()
            if schema.phase in self.concept_phases
        ]
        return tuple(
            i for i, (_, schema) in enumerate(active)
            if schema.concept_type == ConceptType.CONTINUOUS
        )
