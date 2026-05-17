# Copyright 2025 Valeo.


"""Operations for the simulator."""

import jax
import jax.numpy as jnp
from waymax.datatypes import route


def get_index(x: jnp.ndarray, k: int = 1, squeeze: bool = True) -> jnp.ndarray:
    """Get the index of the maximum value in an array.

    Args:
        x: Input array.
        k: Number of top values to return.
        squeeze: Whether to squeeze the output.

    Returns:
        The index of the maximum value.

    """
    if k == 1:
        idx = jnp.argmax(x, keepdims=not squeeze)
    else:
        idx = jax.lax.top_k(x, k)[1]

        if squeeze:
            return idx.squeeze()

    return idx


def select_longest_sdc_path_id(sdc_paths: route.Paths) -> int:
    """Select the index of the longest SDC path that covers all trajectory points.

    Args:
        sdc_paths: Paths object with route information.

    Returns:
        Index of the longest SDC path.

    """
    # (1, num_paths, 1)
    on_route = sdc_paths.on_route
    # (1, num_paths, num_points_per_path)
    on_route = jnp.repeat(on_route, sdc_paths.num_points_per_path, axis=-1)
    # (1, num_paths, num_points_per_path)
    mask = jnp.logical_and(on_route, sdc_paths.valid)
    longest_path_idx = jnp.argmax(jnp.sum(mask, axis=-1))

    return longest_path_idx
