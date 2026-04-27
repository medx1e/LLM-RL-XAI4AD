"""Individual concept extractor functions.

Each function has signature::

    def concept_fn(inp: ConceptInput) -> tuple[jax.Array, jax.Array]:
        '''Returns (raw_value, valid_mask) with shape (...).'''

All functions are pure JAX and designed to be JIT-safe.
Raw values are in real-world units (metres, m/s, radians, bool).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from concepts.types import ConceptInput
from concepts.normalize import (
    OBJ_XY, OBJ_VEL, OBJ_YAW, OBJ_LENGTH, OBJ_WIDTH,
    TL_XY, TL_STATE, TL_RED_INDICES,
    PATH_XY,
    denorm_xy, denorm_vel, denorm_size,
)
from concepts.geometry import l2_norm, wrap_angle, project_onto_path


# =====================================================================
# Phase 1 concepts
# =====================================================================

def ego_speed(inp: ConceptInput) -> tuple[jax.Array, jax.Array]:
    """Scalar ego speed at the current (last) timestep.

    Formula: ||vel_xy|| * max_speed
    Source: sdc_features[..., 0, -1, 2:4]
    Unit: m/s
    """
    cfg = inp.config
    # (…, 1, T, 2) → (…, 2)
    vel_norm = inp.sdc_features[..., 0, -1, OBJ_VEL]
    vel = denorm_vel(vel_norm, cfg)
    speed = l2_norm(vel, axis=-1)            # (…,)
    valid = inp.sdc_mask[..., 0, -1]         # (…,)
    return speed, valid


def ego_acceleration(inp: ConceptInput) -> tuple[jax.Array, jax.Array]:
    """Longitudinal acceleration estimated from speed difference.

    Formula: (speed_t - speed_{t-1}) / dt
    Requires at least 2 valid consecutive timesteps.
    Unit: m/s^2
    """
    cfg = inp.config
    vel_norm = inp.sdc_features[..., 0, :, OBJ_VEL]      # (…, T, 2)
    vel = denorm_vel(vel_norm, cfg)
    speeds = l2_norm(vel, axis=-1)                         # (…, T)
    # Finite difference between last two steps
    accel = (speeds[..., -1] - speeds[..., -2]) / cfg.dt  # (…,)
    valid = inp.sdc_mask[..., 0, -1] & inp.sdc_mask[..., 0, -2]
    return accel, valid


def dist_nearest_object(inp: ConceptInput) -> tuple[jax.Array, jax.Array]:
    """Distance (m) from SDC to the nearest valid other agent at last timestep.

    Formula: min over agents of ||agent_xy[:, -1]|| * max_meters
    Source: agent_features xy at last timestep.
    """
    cfg = inp.config
    # Agent xy at last timestep, denormalised
    agent_xy = denorm_xy(
        inp.agent_features[..., :, -1, OBJ_XY], cfg
    )  # (…, N, 2)
    dists = l2_norm(agent_xy, axis=-1)         # (…, N)
    agent_valid = inp.agent_mask[..., :, -1]   # (…, N)
    # Replace invalid with inf
    dists = jnp.where(agent_valid, dists, jnp.inf)
    min_dist = jnp.min(dists, axis=-1)         # (…,)
    # Valid if at least one agent is valid
    any_valid = jnp.any(agent_valid, axis=-1)
    return min_dist, any_valid


def num_objects_within_radius(
    inp: ConceptInput,
    radius_m: float = 10.0,
) -> tuple[jax.Array, jax.Array]:
    """Count of valid agents within *radius_m* of the SDC.

    Formula: sum(||agent_xy|| < radius AND valid)
    """
    cfg = inp.config
    agent_xy = denorm_xy(
        inp.agent_features[..., :, -1, OBJ_XY], cfg
    )  # (…, N, 2)
    dists = l2_norm(agent_xy, axis=-1)         # (…, N)
    agent_valid = inp.agent_mask[..., :, -1]   # (…, N)
    inside = (dists < radius_m) & agent_valid  # (…, N)
    count = jnp.sum(inside.astype(jnp.float32), axis=-1)  # (…,)
    # Always valid (0 is a valid count)
    valid = jnp.ones_like(count, dtype=bool)
    return count, valid


def traffic_light_red(inp: ConceptInput) -> tuple[jax.Array, jax.Array]:
    """Binary: is any *valid* traffic light currently red?

    Red states in one-hot encoding (after unknown drop):
      index 0 = ARROW_STOP, 3 = STOP, 6 = FLASHING_STOP.
    Source: tl_features[..., :, -1, 2:10] and tl_mask.
    """
    state_oh = inp.tl_features[..., :, -1, TL_STATE]   # (…, N_tl, 8)
    tl_valid = inp.tl_mask[..., :, -1]                  # (…, N_tl)

    # Sum of red columns per TL
    red_score = state_oh[..., TL_RED_INDICES].sum(axis=-1)  # (…, N_tl)
    is_red_per_tl = (red_score > 0.5) & tl_valid            # (…, N_tl)
    any_red = jnp.any(is_red_per_tl, axis=-1).astype(jnp.float32)  # (…,)
    any_tl_valid = jnp.any(tl_valid, axis=-1)
    return any_red, any_tl_valid


def dist_to_traffic_light(inp: ConceptInput) -> tuple[jax.Array, jax.Array]:
    """Distance (m) to nearest valid traffic light at last timestep.

    Formula: min over TLs of ||tl_xy[:, -1]|| * max_meters
    """
    cfg = inp.config
    tl_xy = denorm_xy(
        inp.tl_features[..., :, -1, TL_XY], cfg
    )  # (…, N_tl, 2)
    dists = l2_norm(tl_xy, axis=-1)            # (…, N_tl)
    tl_valid = inp.tl_mask[..., :, -1]         # (…, N_tl)
    dists = jnp.where(tl_valid, dists, jnp.inf)
    min_dist = jnp.min(dists, axis=-1)         # (…,)
    any_valid = jnp.any(tl_valid, axis=-1)
    return min_dist, any_valid


def heading_deviation(inp: ConceptInput) -> tuple[jax.Array, jax.Array]:
    """Signed heading deviation (rad) between ego yaw and path tangent.

    In the SDC-centric frame, the ego yaw at the last timestep should be
    near 0 when driving straight.  The path tangent is estimated from the
    first two path target points.

    Formula: wrap(ego_yaw - atan2(path_dy, path_dx))
    """
    cfg = inp.config
    ego_yaw = inp.sdc_features[..., 0, -1, OBJ_YAW.start]  # (…,)

    # Path tangent from first two path points (normalised xy)
    path_xy = denorm_xy(inp.path_features, cfg)  # (…, P, 2)
    # Use first segment as reference direction
    tangent = path_xy[..., 1, :] - path_xy[..., 0, :]  # (…, 2)
    path_angle = jnp.arctan2(tangent[..., 1], tangent[..., 0])  # (…,)

    deviation = wrap_angle(ego_yaw - path_angle)   # (…,)
    # Valid if SDC valid and path has at least 2 points (always true for P>=2)
    valid = inp.sdc_mask[..., 0, -1]
    return deviation, valid


def progress_along_route(inp: ConceptInput) -> tuple[jax.Array, jax.Array]:
    """Fraction [0, 1] indicating how far along the GPS path the SDC is.

    The SDC is at (0, 0) in its own frame.  We project (0, 0) onto the
    path polyline and report the normalised arc-length fraction.
    """
    cfg = inp.config
    path_xy = denorm_xy(inp.path_features, cfg)  # (…, P, 2)
    origin = jnp.zeros(path_xy.shape[:-2] + (2,))  # (…, 2)
    _, progress = project_onto_path(origin, path_xy)
    valid = inp.sdc_mask[..., 0, -1]
    return progress, valid


# =====================================================================
# Phase 2 concepts
# =====================================================================

def ttc_lead_vehicle(inp: ConceptInput) -> tuple[jax.Array, jax.Array]:
    """Time-to-collision (seconds) with the nearest lead vehicle.

    Lead vehicle: closest valid agent that is *ahead* of the SDC
    (positive x in SDC frame) and roughly in the same lane
    (|y| < 2.5 m).

    TTC = dist_x / (ego_speed - lead_speed_x)
    Capped at 10 s.  Negative / zero closing speed → 10 s (no collision).
    """
    cfg = inp.config
    agent_xy = denorm_xy(inp.agent_features[..., :, -1, OBJ_XY], cfg)  # (…, N, 2)
    agent_vel = denorm_vel(inp.agent_features[..., :, -1, OBJ_VEL], cfg)
    agent_valid = inp.agent_mask[..., :, -1]

    # Lead vehicle criteria: ahead (x > 0), in lane (|y| < 2.5)
    ahead = agent_xy[..., 0] > 0.0                    # (…, N)
    in_lane = jnp.abs(agent_xy[..., 1]) < 2.5         # (…, N)
    candidate = ahead & in_lane & agent_valid          # (…, N)

    dist_x = agent_xy[..., 0]                          # (…, N)
    dist_x = jnp.where(candidate, dist_x, jnp.inf)
    lead_idx = jnp.argmin(dist_x, axis=-1, keepdims=True)  # (…, 1)

    lead_dist = jnp.take_along_axis(dist_x, lead_idx, axis=-1).squeeze(-1)
    lead_vx = jnp.take_along_axis(agent_vel[..., 0], lead_idx, axis=-1).squeeze(-1)

    ego_vel = denorm_vel(inp.sdc_features[..., 0, -1, OBJ_VEL], cfg)
    ego_vx = ego_vel[..., 0]

    closing_speed = ego_vx - lead_vx
    # Only meaningful if closing speed > 0
    raw_ttc = jnp.where(closing_speed > 0.1, lead_dist / closing_speed, 10.0)
    raw_ttc = jnp.clip(raw_ttc, 0.0, 10.0)

    has_lead = jnp.any(candidate, axis=-1)
    valid = has_lead & inp.sdc_mask[..., 0, -1]
    return raw_ttc, valid


def lead_vehicle_decelerating(inp: ConceptInput) -> tuple[jax.Array, jax.Array]:
    """Binary: is the lead vehicle decelerating?

    Lead vehicle identified the same way as in ``ttc_lead_vehicle``.
    Deceleration = speed at T-1 > speed at T for the lead vehicle.
    """
    cfg = inp.config
    agent_xy = denorm_xy(inp.agent_features[..., :, -1, OBJ_XY], cfg)
    agent_valid = inp.agent_mask[..., :, -1]
    ahead = agent_xy[..., 0] > 0.0
    in_lane = jnp.abs(agent_xy[..., 1]) < 2.5
    candidate = ahead & in_lane & agent_valid

    dist_x = jnp.where(candidate, agent_xy[..., 0], jnp.inf)
    lead_idx = jnp.argmin(dist_x, axis=-1, keepdims=True)

    # Speed at last two timesteps for lead vehicle
    vel_all = denorm_vel(inp.agent_features[..., :, :, OBJ_VEL], cfg)  # (…, N, T, 2)
    speed_all = l2_norm(vel_all, axis=-1)  # (…, N, T)

    speed_last = jnp.take_along_axis(speed_all[..., -1], lead_idx, axis=-1).squeeze(-1)
    speed_prev = jnp.take_along_axis(speed_all[..., -2], lead_idx, axis=-1).squeeze(-1)

    decel = (speed_prev - speed_last > 0.5).astype(jnp.float32)  # threshold 0.5 m/s

    has_lead = jnp.any(candidate, axis=-1)
    valid_t = inp.agent_mask[..., :, -1] & inp.agent_mask[..., :, -2]
    lead_valid_t = jnp.take_along_axis(
        jnp.all(valid_t, axis=-1, keepdims=True).astype(jnp.float32),
        lead_idx, axis=-1
    ).squeeze(-1).astype(bool)
    # Actually, we need both timesteps valid for the lead agent
    # Recompute: check lead agent's mask at t and t-1
    lead_mask_last = jnp.take_along_axis(
        inp.agent_mask[..., :, -1], lead_idx, axis=-1
    ).squeeze(-1)
    lead_mask_prev = jnp.take_along_axis(
        inp.agent_mask[..., :, -2], lead_idx, axis=-1
    ).squeeze(-1)
    valid = has_lead & lead_mask_last & lead_mask_prev
    return decel, valid


def at_intersection(inp: ConceptInput) -> tuple[jax.Array, jax.Array]:
    """Binary: is the SDC likely at an intersection?

    Heuristic proxy: a valid traffic light is within 25 m.
    This is an approximation — the observation contract does not provide
    explicit intersection labels.
    """
    cfg = inp.config
    tl_xy = denorm_xy(inp.tl_features[..., :, -1, TL_XY], cfg)
    dists = l2_norm(tl_xy, axis=-1)
    tl_valid = inp.tl_mask[..., :, -1]
    dists = jnp.where(tl_valid, dists, jnp.inf)
    near_tl = jnp.any(dists < 25.0, axis=-1).astype(jnp.float32)
    any_valid = jnp.any(tl_valid, axis=-1)
    return near_tl, any_valid


# =====================================================================
# Phase 3 concepts — Path-based spatial concepts (CBM-V2)
# =====================================================================

def path_curvature_max(inp: ConceptInput) -> tuple[jax.Array, jax.Array]:
    """Maximum Menger curvature along the GPS path.

    Captures the *sharpest* curve ahead — the geometric constraint that
    most restricts safe driving speed.  Higher range than mean curvature.

    Formula: max over interior points of 2|cross|/(|a||b||c|)
    Source: path_features[xy] (denormalized)
    Unit: 1/m
    """
    from concepts.geometry import menger_curvature
    cfg = inp.config
    path_xy = denorm_xy(inp.path_features, cfg)          # (..., P, 2)
    curvatures = menger_curvature(path_xy)                # (..., P-2)
    max_curv = jnp.max(curvatures, axis=-1)               # (...,)
    # Path always present → always valid
    valid = jnp.ones(max_curv.shape, dtype=bool)
    return max_curv, valid


def path_net_heading_change(inp: ConceptInput) -> tuple[jax.Array, jax.Array]:
    """Signed net heading change from first to last path segment.

    Positive → path turns left; negative → right; ~0 → straight.
    NOT redundant with curvature (which is unsigned).

    Formula: atan2 of last segment - atan2 of first segment, wrapped to [-π, π]
    Source: path_features[xy] (denormalized)
    Unit: rad
    """
    cfg = inp.config
    path_xy = denorm_xy(inp.path_features, cfg)           # (..., P, 2)
    # First segment direction
    d_first = path_xy[..., 1, :] - path_xy[..., 0, :]    # (..., 2)
    angle_first = jnp.arctan2(d_first[..., 1], d_first[..., 0])  # (...,)
    # Last segment direction
    d_last = path_xy[..., -1, :] - path_xy[..., -2, :]   # (..., 2)
    angle_last = jnp.arctan2(d_last[..., 1], d_last[..., 0])     # (...,)

    net_change = wrap_angle(angle_last - angle_first)     # (...,)
    valid = jnp.ones(net_change.shape, dtype=bool)
    return net_change, valid


def path_straightness(inp: ConceptInput) -> tuple[jax.Array, jax.Array]:
    """Straightness ratio: chord_length / arc_length.

    1.0 = perfectly straight; approaches 0 for highly curved paths.
    More interpretable and stable than raw arc_length (which has low CoV).

    Formula: ||last - first|| / sum(||p_{i+1} - p_i||)
    Source: path_features[xy] (denormalized)
    Unit: ratio [0, 1]
    """
    cfg = inp.config
    path_xy = denorm_xy(inp.path_features, cfg)           # (..., P, 2)

    # Chord = straight-line distance from first to last point
    chord = l2_norm(path_xy[..., -1, :] - path_xy[..., 0, :], axis=-1)  # (...,)

    # Arc = sum of segment lengths
    segments = path_xy[..., 1:, :] - path_xy[..., :-1, :]  # (..., P-1, 2)
    seg_lens = l2_norm(segments, axis=-1)                    # (..., P-1)
    arc = jnp.sum(seg_lens, axis=-1)                         # (...,)

    straightness = chord / (arc + 1e-8)                      # (...,)
    straightness = jnp.clip(straightness, 0.0, 1.0)

    valid = jnp.ones(straightness.shape, dtype=bool)
    return straightness, valid


def heading_to_path_end(inp: ConceptInput) -> tuple[jax.Array, jax.Array]:
    """Angle from SDC to the last path point (route endpoint).

    Complementary to heading_deviation (which uses first-segment tangent).
    Diverges from heading_deviation on curved roads — exactly when heading
    to the goal matters most.

    Formula: atan2(end_y, end_x) in SDC frame (ego at origin)
    Source: path_features[xy] (denormalized)
    Unit: rad
    """
    cfg = inp.config
    path_xy = denorm_xy(inp.path_features, cfg)            # (..., P, 2)
    end_pt = path_xy[..., -1, :]                           # (..., 2)
    angle = jnp.arctan2(end_pt[..., 1], end_pt[..., 0])   # (...,)

    valid = jnp.ones(angle.shape, dtype=bool)
    return angle, valid

