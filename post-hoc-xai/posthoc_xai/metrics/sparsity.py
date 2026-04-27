"""Sparsity metrics for attribution distributions.

- Gini coefficient (0 = uniform, 1 = perfectly sparse)
- Top-k concentration
- Normalized Shannon entropy
"""

import numpy as np

from posthoc_xai.methods.base import Attribution


def gini_coefficient(attribution: Attribution) -> float:
    """Gini coefficient of the normalized attribution distribution.

    0 = perfectly uniform (all features equally important).
    1 = perfectly sparse (one feature has all importance).
    """
    values = np.sort(np.array(attribution.normalized.flatten()))
    n = len(values)
    total = np.sum(values) + 1e-10
    return float(
        (2 * np.sum(np.arange(1, n + 1) * values) / (n * total)) - (n + 1) / n
    )


def top_k_concentration(attribution: Attribution, k: int = 10) -> float:
    """Fraction of total attribution in the top-k features.

    Higher = more concentrated.
    """
    values = np.array(attribution.normalized.flatten())
    sorted_vals = np.sort(values)[::-1]
    return float(np.sum(sorted_vals[:k]))


def entropy(attribution: Attribution) -> float:
    """Normalized Shannon entropy of the attribution distribution.

    0 = concentrated (low entropy).
    1 = uniform (maximum entropy).
    """
    values = np.array(attribution.normalized.flatten())
    values = values + 1e-10
    values = values / values.sum()

    ent = -np.sum(values * np.log(values))
    max_ent = np.log(len(values))
    return float(ent / max_ent)


def compute_all(attribution: Attribution) -> dict[str, float]:
    """Compute all sparsity metrics for an attribution."""
    return {
        "gini": gini_coefficient(attribution),
        "top_10_concentration": top_k_concentration(attribution, k=10),
        "top_50_concentration": top_k_concentration(attribution, k=50),
        "entropy": entropy(attribution),
    }
