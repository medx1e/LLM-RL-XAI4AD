"""Concept schema definitions — metadata describing each concept."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class ConceptType(Enum):
    """Whether a concept is continuous or binary."""
    CONTINUOUS = "continuous"
    BINARY = "binary"


@dataclass(frozen=True)
class ConceptSchema:
    """Immutable metadata for a single driving concept.

    Attributes:
        name: unique identifier (snake_case).
        concept_type: CONTINUOUS or BINARY.
        description: one-line human description.
        source_fields: which ConceptInput fields are read.
        formula: short textual formula.
        unit: physical unit after denormalization (e.g. "m/s", "m", "rad", "bool").
        norm_range: (lo, hi) of the *normalised* output value.
        validity_rule: how the validity mask is determined.
        phase: 1 = core set, 2 = extended set.
    """
    name: str
    concept_type: ConceptType
    description: str
    source_fields: Sequence[str]
    formula: str
    unit: str
    norm_range: tuple[float, float]
    validity_rule: str
    phase: int = 1
