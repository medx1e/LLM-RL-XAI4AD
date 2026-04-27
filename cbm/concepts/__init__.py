"""Concept extraction module for interpretable driving concepts.

Computes interpretable driving concepts from V-Max observation tensors.
Designed to match the exact V-Max observation contract so that every concept
is provably observation-faithful (no privileged simulator information).
"""

from concepts.types import ConceptInput, ConceptOutput, ObservationConfig
from concepts.registry import CONCEPT_REGISTRY, extract_all_concepts
from concepts.adapters import observation_to_concept_input

__all__ = [
    "ConceptInput",
    "ConceptOutput",
    "ObservationConfig",
    "CONCEPT_REGISTRY",
    "extract_all_concepts",
    "observation_to_concept_input",
]
