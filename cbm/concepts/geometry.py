"""Geometric utility functions for concept computation.

All functions are pure JAX and JIT-safe.  They operate on denormalised
(real-world metre / m-s) values unless stated otherwise.
"""

import jax
import jax.numpy as jnp


def l2_norm(xy: jax.Array, axis: int = -1, eps: float = 1e-8) -> jax.Array:
    """Euclidean norm along *axis*, with numerical floor *eps*."""
    return jnp.sqrt(jnp.sum(xy ** 2, axis=axis) + eps)


def angle_between_vectors(v1: jax.Array, v2: jax.Array) -> jax.Array:
    """Signed angle from v1 to v2 (radians, counter-clockwise positive).

    Both inputs have shape (..., 2).  Returns (...).
    """
    return jnp.arctan2(
        v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0],
        v1[..., 0] * v2[..., 0] + v1[..., 1] * v2[..., 1],
    )


def wrap_angle(angle: jax.Array) -> jax.Array:
    """Wrap angle to [-pi, pi]."""
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi


def project_onto_path(
    point: jax.Array,
    path_xy: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Find closest segment on *path_xy* to *point* and return
    (distance-to-path, progress-fraction).

    Args:
        point: (..., 2)  — query point(s).
        path_xy: (..., P, 2)  — ordered path waypoints.

    Returns:
        lateral_dist: (...)  perpendicular distance to closest segment.
        progress: (...)  fraction [0, 1] along path of projection.
    """
    # Vectors along each segment
    seg_start = path_xy[..., :-1, :]           # (..., P-1, 2)
    seg_end = path_xy[..., 1:, :]              # (..., P-1, 2)
    seg_vec = seg_end - seg_start              # (..., P-1, 2)
    seg_len_sq = jnp.sum(seg_vec ** 2, axis=-1) + 1e-8  # (..., P-1)

    # Project point onto each segment line
    point_expanded = jnp.expand_dims(point, axis=-2)  # (..., 1, 2)
    to_point = point_expanded - seg_start              # (..., P-1, 2)
    t = jnp.sum(to_point * seg_vec, axis=-1) / seg_len_sq  # (..., P-1)
    t_clamped = jnp.clip(t, 0.0, 1.0)

    # Closest point on segment
    proj = seg_start + t_clamped[..., None] * seg_vec   # (..., P-1, 2)
    diff = point_expanded - proj                        # (..., P-1, 2)
    dist = l2_norm(diff, axis=-1)                       # (..., P-1)

    # Best segment
    best_idx = jnp.argmin(dist, axis=-1)                # (...)
    lateral_dist = jnp.take_along_axis(
        dist, jnp.expand_dims(best_idx, -1), axis=-1
    ).squeeze(-1)

    # Cumulative arc length for progress
    seg_lengths = jnp.sqrt(seg_len_sq)                  # (..., P-1)
    cum_len = jnp.cumsum(seg_lengths, axis=-1)          # (..., P-1)
    total_len = cum_len[..., -1] + 1e-8                 # (...)

    # Progress = (cum length up to best segment start + t * segment length) / total
    best_t = jnp.take_along_axis(
        t_clamped, jnp.expand_dims(best_idx, -1), axis=-1
    ).squeeze(-1)
    best_seg_len = jnp.take_along_axis(
        seg_lengths, jnp.expand_dims(best_idx, -1), axis=-1
    ).squeeze(-1)
    # Cumulative up to (but not including) best segment
    cum_before = jnp.take_along_axis(
        jnp.concatenate([jnp.zeros_like(cum_len[..., :1]), cum_len[..., :-1]], axis=-1),
        jnp.expand_dims(best_idx, -1),
        axis=-1,
    ).squeeze(-1)
    progress = (cum_before + best_t * best_seg_len) / total_len

    return lateral_dist, progress


def menger_curvature(path_xy: jax.Array, eps: float = 1e-8) -> jax.Array:
    """Menger curvature at each interior point of a polyline.

    For three consecutive points P1, P2, P3:
        κ = 2 |cross(P2−P1, P3−P2)| / (|P2−P1| · |P3−P2| · |P3−P1|)

    Args:
        path_xy: (..., P, 2) — ordered polyline waypoints.
        eps: numerical floor to prevent division by zero.

    Returns:
        curvatures: (..., P-2) — curvature at each interior point.
    """
    p1 = path_xy[..., :-2, :]   # (..., P-2, 2)
    p2 = path_xy[..., 1:-1, :]  # (..., P-2, 2)
    p3 = path_xy[..., 2:, :]    # (..., P-2, 2)

    v1 = p2 - p1  # (..., P-2, 2)
    v2 = p3 - p2  # (..., P-2, 2)

    # 2D cross product magnitude: |v1_x * v2_y - v1_y * v2_x|
    cross = jnp.abs(v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0])

    len_v1 = l2_norm(v1, axis=-1, eps=eps)
    len_v2 = l2_norm(v2, axis=-1, eps=eps)
    len_v3 = l2_norm(p3 - p1, axis=-1, eps=eps)

    curvature = 2.0 * cross / (len_v1 * len_v2 * len_v3 + eps)
    return curvature

