"""Safety metric computations for event mining.

All functions operate on numpy arrays extracted by the V-Max adapter.
"""

from __future__ import annotations

import numpy as np


def compute_distances(
    ego_x: np.ndarray,
    ego_y: np.ndarray,
    other_x: np.ndarray,
    other_y: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Compute Euclidean distance from ego to each other agent per timestep.

    Args:
        ego_x: (T,) ego x positions.
        ego_y: (T,) ego y positions.
        other_x: (T, N) other agent x positions.
        other_y: (T, N) other agent y positions.
        valid: (T, N) boolean validity mask.

    Returns:
        (T, N) distance array. Invalid agents get distance = inf.
    """
    dx = other_x - ego_x[:, None]
    dy = other_y - ego_y[:, None]
    dist = np.sqrt(dx**2 + dy**2)
    dist = np.where(valid, dist, np.inf)
    return dist


def compute_ttc(
    ego_x: np.ndarray,
    ego_y: np.ndarray,
    ego_vx: np.ndarray,
    ego_vy: np.ndarray,
    other_x: np.ndarray,
    other_y: np.ndarray,
    other_vx: np.ndarray,
    other_vy: np.ndarray,
    valid: np.ndarray,
    max_ttc: float = 10.0,
) -> np.ndarray:
    """Compute time-to-collision between ego and each other agent.

    Uses a simplified point-mass linear projection:
    TTC = -dot(dp, dv) / dot(dv, dv) when closing, else max_ttc.

    Args:
        ego_x, ego_y, ego_vx, ego_vy: (T,) ego state.
        other_x, other_y, other_vx, other_vy: (T, N) other agent states.
        valid: (T, N) boolean validity mask.
        max_ttc: Cap value for non-closing or far agents.

    Returns:
        (T, N) TTC array. Invalid agents and non-closing pairs get max_ttc.
    """
    # Relative position and velocity
    dpx = other_x - ego_x[:, None]
    dpy = other_y - ego_y[:, None]
    dvx = other_vx - ego_vx[:, None]
    dvy = other_vy - ego_vy[:, None]

    # dot(dp, dv) and dot(dv, dv)
    dp_dot_dv = dpx * dvx + dpy * dvy
    dv_dot_dv = dvx**2 + dvy**2

    # TTC = -dot(dp, dv) / dot(dv, dv) for closing vehicles
    # Only valid when dp_dot_dv < 0 (closing) and dv_dot_dv > 0
    closing = dp_dot_dv < 0
    dv_nonzero = dv_dot_dv > 1e-6

    ttc = np.full_like(dpx, max_ttc)
    mask = closing & dv_nonzero & valid
    ttc[mask] = np.clip(-dp_dot_dv[mask] / dv_dot_dv[mask], 0.0, max_ttc)

    # Invalid agents get max_ttc
    ttc[~valid] = max_ttc

    return ttc


def compute_criticality(
    ttc: np.ndarray,
    min_distance: np.ndarray,
    ego_speed: np.ndarray,
    ttc_threshold: float = 5.0,
    dist_threshold: float = 20.0,
) -> np.ndarray:
    """Compute composite criticality score per timestep.

    Combines TTC, minimum distance, and ego speed into a [0, 1] score.

    Args:
        ttc: (T, N) time-to-collision array.
        min_distance: (T, N) distance array.
        ego_speed: (T,) ego speed.
        ttc_threshold: TTC below this contributes to criticality.
        dist_threshold: Distance below this contributes to criticality.

    Returns:
        (T,) criticality score in [0, 1].
    """
    # Min TTC across agents
    min_ttc = np.min(ttc, axis=1)  # (T,)
    min_dist = np.min(min_distance, axis=1)  # (T,)

    # TTC component: 1 when TTC=0, 0 when TTC >= threshold
    ttc_score = np.clip(1.0 - min_ttc / ttc_threshold, 0.0, 1.0)

    # Distance component: 1 when dist=0, 0 when dist >= threshold
    dist_score = np.clip(1.0 - min_dist / dist_threshold, 0.0, 1.0)

    # Speed component: higher speed = more critical (normalize by ~30 m/s)
    speed_score = np.clip(ego_speed / 30.0, 0.0, 1.0)

    # Weighted combination
    criticality = 0.5 * ttc_score + 0.3 * dist_score + 0.2 * speed_score
    return np.clip(criticality, 0.0, 1.0)


def compute_ego_speed(ego_vx: np.ndarray, ego_vy: np.ndarray) -> np.ndarray:
    """Compute ego speed magnitude from velocity components."""
    return np.sqrt(ego_vx**2 + ego_vy**2)
