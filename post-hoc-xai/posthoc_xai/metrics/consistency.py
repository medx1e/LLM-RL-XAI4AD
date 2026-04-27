"""Consistency metrics for attribution stability.

Measures how stable explanations are across similar inputs.

- Feature-level pairwise correlation
- Category-level pairwise correlation
"""

import numpy as np
from scipy.stats import pearsonr

from posthoc_xai.methods.base import Attribution


def _safe_pearsonr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation with a guard for constant-variance inputs.

    ``scipy.stats.pearsonr`` raises ``ValueError`` (or returns NaN in newer
    versions) when one of the inputs is constant (zero variance).  This
    wrapper handles that edge case:

    - Both constant and identical → 1.0 (perfect agreement)
    - Both constant but different → 0.0 (no information)
    - One constant → 0.0 (undefined, treated as no correlation)
    """
    std_a, std_b = np.std(a), np.std(b)
    if std_a < 1e-12 and std_b < 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    if std_a < 1e-12 or std_b < 1e-12:
        return 0.0
    corr, _ = pearsonr(a, b)
    return float(corr)


def attribution_consistency(attributions: list[Attribution]) -> float:
    """Average pairwise Pearson correlation across attribution vectors.

    High consistency = method produces stable explanations.
    """
    if len(attributions) < 2:
        return 1.0

    correlations = []
    for i in range(len(attributions)):
        for j in range(i + 1, len(attributions)):
            a1 = np.array(attributions[i].normalized.flatten())
            a2 = np.array(attributions[j].normalized.flatten())
            correlations.append(_safe_pearsonr(a1, a2))

    return float(np.mean(correlations))


def category_consistency(attributions: list[Attribution]) -> float:
    """Pairwise correlation at the category level (more robust).

    Uses the ``category_importance`` dict instead of raw features.
    """
    if len(attributions) < 2:
        return 1.0

    categories = list(attributions[0].category_importance.keys())

    vectors = []
    for attr in attributions:
        vec = np.array([attr.category_importance[c] for c in categories])
        vectors.append(vec)

    correlations = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            correlations.append(_safe_pearsonr(vectors[i], vectors[j]))

    return float(np.mean(correlations))
