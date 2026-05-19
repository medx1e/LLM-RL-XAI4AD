# Copyright 2025 - Decision Classifier for Attention-Grounded XRL Pipeline
"""
Module 7: DecisionClassifier

Applies the 2×2 classification (necessity × grounding) to produce
a decision class label that drives downstream LLM routing.
"""

from typing import Any, Dict, Optional


# Decision class constants
GROUNDED_CRITICAL = "GROUNDED_CRITICAL"
UNGROUNDED_CRITICAL = "UNGROUNDED_CRITICAL"
GROUNDED_ROUTINE = "GROUNDED_ROUTINE"
ROUTINE = "ROUTINE"


def classify(
    necessity_score: float,
    grounding_score: Optional[float],
    config: Dict[str, Any],
) -> str:
    """
    Classify the decision into one of four categories.

    Args:
        necessity_score: Float in [0, 1] from NecessityScorer.
        grounding_score: Float in [0, 1] from AttentionGrounder, or None
            if no threat agents exist.
        config: Pipeline config with ``necessity_threshold`` and
            ``grounding_threshold`` keys.

    Returns:
        One of: ``GROUNDED_CRITICAL``, ``UNGROUNDED_CRITICAL``,
        ``GROUNDED_ROUTINE``, ``ROUTINE``.
    """
    n_thresh = config.get("necessity_threshold", 0.5)
    g_thresh = config.get("grounding_threshold", 0.5)

    # If grounding_score is None (no threats), treat as 0
    g_score = grounding_score if grounding_score is not None else 0.0

    if necessity_score >= n_thresh:
        if g_score >= g_thresh:
            return GROUNDED_CRITICAL
        else:
            return UNGROUNDED_CRITICAL
    else:
        if g_score >= g_thresh:
            return GROUNDED_ROUTINE
        else:
            return ROUTINE
