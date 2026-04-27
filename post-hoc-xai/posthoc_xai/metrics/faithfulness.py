"""Faithfulness metrics for evaluating attribution quality.

- Deletion / insertion curves
- Attention-gradient correlation
- Area under curve (AUC) for both
"""

from typing import Optional

import jax.numpy as jnp
import numpy as np
from scipy.stats import pearsonr, spearmanr

from posthoc_xai.methods.base import Attribution
from posthoc_xai.models.base import ExplainableModel


def attention_gradient_correlation(
    attr_a: Attribution,
    attr_b: Attribution,
) -> dict[str, float]:
    """Compute correlation between two attribution vectors.

    Useful for comparing attention-based and gradient-based attributions.
    High correlation suggests attention is "faithful" to feature importance.

    Returns dict with pearson_r, pearson_p, spearman_rho, spearman_p.
    """
    a_flat = np.array(attr_a.normalized.flatten())
    b_flat = np.array(attr_b.normalized.flatten())

    pr, pp = pearsonr(a_flat, b_flat)
    sr, sp = spearmanr(a_flat, b_flat)

    return {
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_rho": float(sr),
        "spearman_p": float(sp),
    }


def deletion_curve(
    model: ExplainableModel,
    observation: jnp.ndarray,
    attribution: Attribution,
    n_steps: int = 20,
    target_action: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Progressively remove the most important features and track output.

    A good attribution should cause rapid output decrease.

    Returns (percentages, outputs) arrays for plotting.
    """
    flat_obs = np.array(observation.flatten())
    flat_attr = np.array(attribution.normalized.flatten())
    sorted_indices = np.argsort(flat_attr)[::-1]

    baseline_output = float(model.get_action_value(observation, target_action))
    percentages = np.linspace(0, 1, n_steps)
    outputs = [baseline_output]

    for pct in percentages[1:]:
        n_remove = int(pct * len(flat_obs))
        modified = flat_obs.copy()
        modified[sorted_indices[:n_remove]] = 0
        modified_obs = jnp.array(modified.reshape(observation.shape))
        output = float(model.get_action_value(modified_obs, target_action))
        outputs.append(output)

    return percentages, np.array(outputs)


def insertion_curve(
    model: ExplainableModel,
    observation: jnp.ndarray,
    attribution: Attribution,
    n_steps: int = 20,
    target_action: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Progressively add the most important features to a blank input.

    A good attribution should cause rapid output increase.

    Returns (percentages, outputs) arrays for plotting.
    """
    flat_obs = np.array(observation.flatten())
    flat_attr = np.array(attribution.normalized.flatten())
    sorted_indices = np.argsort(flat_attr)[::-1]

    blank_obs = jnp.zeros_like(observation)
    baseline_output = float(model.get_action_value(blank_obs, target_action))
    percentages = np.linspace(0, 1, n_steps)
    outputs = [baseline_output]

    for pct in percentages[1:]:
        n_add = int(pct * len(flat_obs))
        modified = np.zeros_like(flat_obs)
        modified[sorted_indices[:n_add]] = flat_obs[sorted_indices[:n_add]]
        modified_obs = jnp.array(modified.reshape(observation.shape))
        output = float(model.get_action_value(modified_obs, target_action))
        outputs.append(output)

    return percentages, np.array(outputs)


def area_under_deletion_curve(outputs: np.ndarray) -> float:
    """AUC for deletion curve. Lower is better.

    Integrates over the x-axis [0, 1] using the trapezoidal rule.  The
    x-spacing between *n* uniformly-spaced points is 1/(n-1), not 1/n.
    """
    x = np.linspace(0.0, 1.0, len(outputs))
    return float(np.trapz(outputs, x=x))


def area_under_insertion_curve(outputs: np.ndarray) -> float:
    """AUC for insertion curve. Higher is better.

    Same spacing fix as :func:`area_under_deletion_curve`.
    """
    x = np.linspace(0.0, 1.0, len(outputs))
    return float(np.trapz(outputs, x=x))
