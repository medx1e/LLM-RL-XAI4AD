"""Concentration metrics for attention analysis.

This module provides functions to compute various concentration metrics
for attention weights, including Gini coefficient, entropy, and top-k mass.
"""

import numpy as np
from typing import Dict


def gini_coefficient(attention: np.ndarray) -> float:
    """Compute Gini coefficient for attention distribution.
    
    The Gini coefficient measures inequality in a distribution:
    - 0: perfectly uniform distribution
    - 1: perfectly concentrated distribution (all weight on one element)
    
    Args:
        attention: Attention weights array of any shape. Will be flattened.
        
    Returns:
        Gini coefficient in [0, 1].
    """
    # Flatten and sort
    x = np.sort(attention.flatten())
    n = len(x)
    
    if n == 0:
        return 0.0
    
    # Handle edge cases
    if np.all(x == 0):
        return 0.0  # No attention at all
    
    # Normalize to sum to 1
    x = x / (np.sum(x) + 1e-10)
    
    # Compute Gini coefficient
    # Formula: G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
    # Simplified for normalized x: G = (2 * sum(i * x_i)) / n - (n + 1) / n
    index = np.arange(1, n + 1)
    gini = (2.0 * np.sum(index * x)) / n - (n + 1.0) / n
    
    return float(np.clip(gini, 0.0, 1.0))


def entropy_concentration(attention: np.ndarray) -> float:
    """Compute normalized entropy-based concentration metric.
    
    Entropy measures the uniformity of a distribution:
    - High entropy: uniform distribution
    - Low entropy: concentrated distribution
    
    This function returns 1 - normalized_entropy, so:
    - 0: uniform distribution
    - 1: concentrated distribution
    
    Args:
        attention: Attention weights array of any shape. Will be flattened.
        
    Returns:
        Concentration score in [0, 1].
    """
    # Flatten and normalize
    x = attention.flatten()
    x = x / (np.sum(x) + 1e-10)
    
    # Filter out zeros to avoid log(0)
    x = x[x > 0]
    
    if len(x) == 0:
        return 0.0
    
    # Compute entropy
    entropy = -np.sum(x * np.log(x + 1e-10))
    
    # Normalize by maximum entropy (uniform distribution)
    max_entropy = np.log(len(x))
    
    if max_entropy == 0:
        return 1.0  # Only one element
    
    normalized_entropy = entropy / max_entropy
    
    # Return concentration (1 - normalized entropy)
    concentration = 1.0 - normalized_entropy
    
    return float(np.clip(concentration, 0.0, 1.0))


def topk_mass(attention: np.ndarray, k: int = 3) -> float:
    """Compute fraction of attention mass on top-k elements.
    
    Args:
        attention: Attention weights array of any shape. Will be flattened.
        k: Number of top elements to consider.
        
    Returns:
        Fraction of total attention on top-k elements, in [0, 1].
    """
    # Flatten and normalize
    x = attention.flatten()
    total = np.sum(x)
    
    if total == 0:
        return 0.0
    
    # Get top-k values
    k = min(k, len(x))
    top_k_values = np.partition(x, -k)[-k:]
    top_k_sum = np.sum(top_k_values)
    
    return float(top_k_sum / total)


def compute_concentration_suite(attention: np.ndarray) -> Dict[str, float]:
    """Compute all concentration metrics.
    
    Args:
        attention: Attention weights array of any shape.
        
    Returns:
        Dictionary with keys: 'gini', 'entropy', 'top3_mass'.
    """
    return {
        'gini': gini_coefficient(attention),
        'entropy': entropy_concentration(attention),
        'top3_mass': topk_mass(attention, k=3),
    }


def compute_per_head_concentration(
    attention_per_vehicle: np.ndarray,
    metric: str = 'gini'
) -> np.ndarray:
    """Compute concentration metric for each attention head.
    
    Args:
        attention_per_vehicle: Array of shape (n_heads, n_vehicles) with
            aggregated attention weights per vehicle for each head.
        metric: Concentration metric to use ('gini', 'entropy', or 'top3_mass').
        
    Returns:
        Array of shape (n_heads,) with concentration scores.
    """
    n_heads = attention_per_vehicle.shape[0]
    concentrations = np.zeros(n_heads)
    
    metric_fn = {
        'gini': gini_coefficient,
        'entropy': entropy_concentration,
        'top3_mass': lambda x: topk_mass(x, k=3),
    }.get(metric, gini_coefficient)
    
    for head_idx in range(n_heads):
        concentrations[head_idx] = metric_fn(attention_per_vehicle[head_idx])
    
    return concentrations
