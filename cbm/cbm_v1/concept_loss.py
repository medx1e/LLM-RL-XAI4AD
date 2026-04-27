"""Concept supervision loss for CBM-V1.

Supports per-concept loss type (BCE for binary, Huber for continuous)
with validity masking.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from cbm_v1.config import CBMConfig


def concept_loss(
    predicted: jax.Array,
    target: jax.Array,
    valid: jax.Array,
    config: CBMConfig,
) -> jax.Array:
    """Compute masked concept supervision loss.

    Args:
        predicted: (batch, num_concepts) — concept head output (normalized [0, 1]).
        target: (batch, num_concepts) — ground-truth concept values (normalized [0, 1]).
        valid: (batch, num_concepts) — boolean validity mask.
        config: CBM configuration with concept type indices.

    Returns:
        Scalar loss averaged over valid concept entries.
    """
    valid_f = valid.astype(jnp.float32)

    # Binary concepts: BCE
    binary_idx = jnp.array(config.binary_concept_indices)
    pred_bin = jnp.take(predicted, binary_idx, axis=-1)
    tgt_bin = jnp.take(target, binary_idx, axis=-1)
    valid_bin = jnp.take(valid_f, binary_idx, axis=-1)

    # Clamp predictions for numerical stability
    eps = 1e-6
    pred_bin_safe = jnp.clip(pred_bin, eps, 1.0 - eps)
    bce = -(tgt_bin * jnp.log(pred_bin_safe) + (1 - tgt_bin) * jnp.log(1 - pred_bin_safe))
    bce_masked = (bce * valid_bin).sum()

    # Continuous concepts: Huber loss (delta=1.0)
    cont_idx = jnp.array(config.continuous_concept_indices)
    pred_cont = jnp.take(predicted, cont_idx, axis=-1)
    tgt_cont = jnp.take(target, cont_idx, axis=-1)
    valid_cont = jnp.take(valid_f, cont_idx, axis=-1)

    diff = pred_cont - tgt_cont
    abs_diff = jnp.abs(diff)
    huber = jnp.where(abs_diff <= 1.0, 0.5 * diff ** 2, abs_diff - 0.5)
    huber_masked = (huber * valid_cont).sum()

    # Average over total valid entries
    total_valid = valid_f.sum() + eps
    loss = (bce_masked + huber_masked) / total_valid

    return loss
