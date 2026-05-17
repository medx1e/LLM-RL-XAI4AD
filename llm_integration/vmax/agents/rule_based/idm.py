# Copyright 2025 Valeo.

import jax
import jax.numpy as jnp
from waymax.utils import geometry

from vmax.simulator import constants


# Minimum lead distance to prevent divide-by-zero errors.
_MINIMUM_LEAD_DISTANCE = 0.1  # Units: m
_MIN_SPACING = 2.0
_SAFE_TIME_HEADWAY = 2.0
_MAX_ACCEL = 2.0
_MAX_DECEL = 4.0
_DELTA = 4.0


def select_action(observation: jax.Array):
    """Compute the action by generating a smooth trajectory and normalizing acceleration and steering."""
    unroll, new_speed, accel_action = generate_unroll(observation)

    next_x, next_y = unroll[1]

    steering_action = compute_steering(
        next_x,
        next_y,
        accel_action,
        new_speed,
    )
    # Normalize actions
    accel_action = (
        jnp.clip(
            accel_action,
            -constants.MAX_ACCEL_BICYCLE,
            constants.MAX_ACCEL_BICYCLE,
        )
        / constants.MAX_ACCEL_BICYCLE
    )
    steering_action = steering_action / constants.MAX_STEERING

    return jnp.concatenate(
        [jnp.atleast_1d(accel_action), jnp.atleast_1d(steering_action)],
        axis=0,
    )


def generate_unroll(observation: jax.Array, time_horizon: float = 4.0):
    """Generate an unroll trajectory using the target path, current speed, and sensor features."""
    (
        target_path,
        target_valid,
        desired_speed,
        sdc_features,
        curr_speed,
        other_agents_xy,
        other_agents_vel_xy,
        other_agents_dimensions,  # length, width, yaw
        other_agents_valid,
        red_light_pos,
        roadgraph_points,
    ) = observation

    target_path = correct_target_path(target_path, target_valid)

    sdc_length, sdc_width = sdc_features[0], sdc_features[1]
    sdc_vel_x, sdc_vel_y = sdc_features[2], sdc_features[3]

    red_light_xy = red_light_pos[0]
    red_light_xy = jnp.array((red_light_xy[0] - 2, red_light_xy[1]))
    red_light_status = red_light_pos[1]

    # Fix for marche arriere
    curr_speed = (sdc_vel_x > 0) * curr_speed
    sdc_vel_x = jnp.where(sdc_vel_x > 0, sdc_vel_x, 0)

    leading_vehicle_idx, is_colliding, lead_dist = identify_leading_agent(
        target_path,
        sdc_length,
        sdc_width,
        other_agents_xy,
        other_agents_dimensions,
        other_agents_valid,
    )

    # Red light handling
    # There was a major bug here for cases with red ligh but no cars

    is_red_light_dangerous = (red_light_status == 1) & ~((lead_dist < jnp.linalg.norm(red_light_xy)) & is_colliding)
    lead_dist = jnp.where(is_red_light_dangerous, jnp.linalg.norm(red_light_xy), lead_dist)
    is_colliding = is_colliding + is_red_light_dangerous

    lead_velocity = other_agents_vel_xy[leading_vehicle_idx]
    lead_speed = jnp.linalg.norm(lead_velocity) * (1 - is_red_light_dangerous)

    s_star = _MIN_SPACING + jnp.maximum(
        0,
        curr_speed * _SAFE_TIME_HEADWAY
        + curr_speed * (curr_speed - lead_speed) / (2 * jnp.sqrt(_MAX_ACCEL * _MAX_DECEL)),
    )
    s_star = s_star * is_colliding

    # Little trick to prevent division by 0
    lead_dist = lead_dist * is_colliding
    lead_dist = lead_dist + _MINIMUM_LEAD_DISTANCE * (lead_dist == 0.0)

    # Acceleration to take at time t
    accel_action = _MAX_ACCEL * (1 - (curr_speed / desired_speed) ** _DELTA - (s_star / lead_dist) ** 2)

    accel_action = jnp.clip(accel_action, -constants.MAX_ACCEL_BICYCLE, constants.MAX_ACCEL_BICYCLE)

    # Compute steering with next position x, y of the agent
    new_speed = curr_speed + accel_action * constants.TIME_DELTA
    new_speed = jnp.where(new_speed <= 0, 0, new_speed)

    # Trick to avoid going backwards
    accel_action = (new_speed - curr_speed) / constants.TIME_DELTA

    unroll = generate_smooth_trajectory(
        target_path,
        sdc_vel_x,
        sdc_vel_y,
        new_speed,
        time_horizon,
    )

    return unroll, new_speed, accel_action


def correct_target_path(target, target_valid):
    """Correct and fill the target path based on its validity."""
    first_invalid_idx = jnp.argmin(target_valid)
    first_invalid_idx = jnp.where(~jnp.any(~target_valid), target.shape[0], first_invalid_idx)

    path_valid = jnp.arange(target.shape[0]) < first_invalid_idx

    direction = target[first_invalid_idx - 1] - target[jnp.maximum(first_invalid_idx - 2, 0)]

    filled_path = jnp.where(path_valid[:, None], target, jnp.zeros_like(target))

    def prolongate(i, path):
        next_point = path[i - 1] + direction
        path = path.at[i].set(next_point)
        return path

    filled_path = jax.lax.fori_loop(first_invalid_idx, target.shape[0], prolongate, filled_path)

    return filled_path


def compute_steering(x_next, y_next, accel, new_speed):
    """Compute the steering angle using the next position, acceleration, and the computed speed."""
    # Compute desired yaw angle
    new_yaw = jnp.arctan2(y_next, x_next)
    # Compute steering angle
    steering = new_yaw / (new_speed * constants.TIME_DELTA + 0.5 * accel * constants.TIME_DELTA**2)

    # Clip steering
    steering = jnp.where(jnp.logical_and(accel < 1e-5, new_speed < 1e-5), 0, steering)
    steering = jnp.where(steering > constants.MAX_STEERING, constants.MAX_STEERING, steering)
    steering = jnp.where(steering < -constants.MAX_STEERING, -constants.MAX_STEERING, steering)
    return steering


def identify_leading_agent(
    target,
    sdc_length,
    sdc_width,
    other_agents_xy,
    other_agents_dimensions,
    other_agents_valid,
):
    """Identify the leading agent along the target path."""
    sdc_yaws = compute_yaw_along_path(target)

    target_states = jnp.hstack(
        (
            target,
            jnp.full((target.shape[0], 1), sdc_length),
            jnp.full((target.shape[0], 1), sdc_width),
            sdc_yaws[:, None],
        ),
    )

    def check_collision(target_point, obj_index):
        x, y = other_agents_xy[obj_index]
        length, width, yaw = other_agents_dimensions[obj_index]

        valid = other_agents_valid[obj_index]

        obj = jnp.array([x, y, length, width, yaw])

        return valid * geometry.has_overlap(obj, target_point)

    # Vectorize check_collision over both target points and objects
    check_collision_vmap = jax.vmap(
        jax.vmap(check_collision, in_axes=(None, 0)),  # vectorize over objects
        in_axes=(0, None),  # vectorize over target points
    )

    # Compute collision matrix
    collisions = check_collision_vmap(
        target_states,
        jnp.arange(other_agents_xy.shape[0]),
    )

    first_collision_points = jnp.argmax(collisions, axis=0)

    leading_vehicle_indices = jnp.where(
        jnp.any(collisions, axis=0),
        first_collision_points,
        jnp.inf,
    )

    leading_vehicle_idx = jnp.argmin(leading_vehicle_indices)

    is_colliding = leading_vehicle_indices[leading_vehicle_idx] != jnp.inf

    arc_lengths = jnp.linalg.norm(target[1:] - target[:-1], axis=-1)
    cumulative_arc_lenghts = jnp.concatenate([jnp.zeros(2), jnp.cumsum(arc_lengths)])

    lead_distance = cumulative_arc_lenghts[first_collision_points[leading_vehicle_idx]]

    return leading_vehicle_idx, is_colliding, lead_distance


def compute_yaw_along_path(target):
    """Compute yaw angles along the provided target path."""
    dx = jnp.diff(target[:, 0])
    dy = jnp.diff(target[:, 1])

    yaw_angles = jnp.arctan2(dy, dx)

    yaw_angles = jnp.append(yaw_angles, yaw_angles[-1])

    return yaw_angles


def generate_smooth_trajectory(
    target,
    vel_x,
    vel_y,
    new_speed,
    time_horizon=4.0,
    alignment_factor=0.5,
    curvature_factor=0.5,
    h_min=0.3,
    d_merge_base=5.0,
    k_v=0.5,
    num_path_samples: int = 50,
):
    """Generate a smooth trajectory that blends a Bezier curve with the target path."""
    # Step 1: At which point should we merge into the target path
    # Depends on the speed, velocity, and also curvature of the target

    target_directions = target[1:] - target[0]
    target_directions = jnp.where(
        target_directions == jnp.zeros_like(target_directions),
        jnp.inf * jnp.ones_like(target_directions),
        target_directions,
    )

    target_direction_idx = jnp.argmin(jnp.linalg.norm(target_directions, axis=1))
    target_direction = target_directions[target_direction_idx]

    target_direction /= jnp.linalg.norm(target_direction)
    vel_direction = jnp.hstack([vel_x, vel_y])
    vel_direction /= jnp.linalg.norm(vel_direction)

    alignment = jnp.dot(target_direction, vel_direction)
    alignment_scaling = 1 - alignment_factor * alignment

    # Max curvature of target_path
    delta_pos = target[1:] - target[:-1]

    delta_yaw = jnp.arctan2(delta_pos[:, 0], delta_pos[:, 1])

    delta_theta = delta_yaw[1:] - delta_yaw[:-1]
    delta_theta = jnp.where(delta_theta > jnp.pi, delta_theta - 2 * jnp.pi, delta_theta)
    delta_theta = jnp.where(
        delta_theta < -jnp.pi,
        delta_theta + 2 * jnp.pi,
        delta_theta,
    )
    delta_s = jnp.linalg.norm(delta_pos[1:], axis=1)

    curvature = jnp.abs(delta_theta / delta_s)

    curvature = jnp.where(jnp.isfinite(curvature), curvature, 0.0)

    max_curvature = jnp.max(curvature)

    d_merge = (d_merge_base + k_v * new_speed - curvature_factor * max_curvature) * alignment_scaling
    h_merge = d_merge / new_speed

    h_merge = jnp.clip(h_merge, h_min, time_horizon)
    d_merge = h_merge * new_speed

    # Find the blending point on the target path
    path_distances = jnp.zeros(target.shape[0])
    path_distances = path_distances.at[1:].set(
        jnp.cumsum(jnp.linalg.norm(target[1:] - target[:-1], axis=1)),
    )
    blending_index = jnp.searchsorted(path_distances, d_merge)
    blending_point = target[blending_index]

    # Step 2: Define control points for cubic Bezier curve
    p0 = jnp.array([0.0, 0.0])  # Current position
    p1 = jnp.array([vel_x, vel_y]) * h_merge / 3.0  # Velocity-scaled control
    p3 = blending_point  # Blending point
    p2 = (2 * p3 + p1) / 3.0  # Intermediate control point for smoothness

    def bezier_curve(alpha):
        """Compute cubic Bezier curve."""
        return (
            (1 - alpha) ** 3 * p0 + 3 * (1 - alpha) ** 2 * alpha * p1 + 3 * (1 - alpha) * alpha**2 * p2 + alpha**3 * p3
        )

    # Step 3: Sample the smooth path
    alpha_values = jnp.linspace(0, 1, num_path_samples)
    bezier_path = jnp.array([bezier_curve(alpha) for alpha in alpha_values])

    # Step 4: Append remaining target path after blending

    indices = jnp.arange(num_path_samples)

    remaining_path = jnp.where(
        (indices + blending_index < target.shape[0])[:, None],
        target[indices + blending_index],
        target[None, -1],
    )

    full_path = jnp.concatenate([bezier_path, remaining_path])

    # Step 5: Interpolate trajectory
    timestamps = jnp.arange(0, time_horizon + constants.TIME_DELTA, constants.TIME_DELTA)
    trajectory = interpolate_path(full_path, new_speed, timestamps)

    # Do nothing if the computation failed
    trajectory = jnp.where(
        jnp.isnan(trajectory),
        jnp.zeros_like(trajectory),
        trajectory,
    )

    return trajectory


def interpolate_path(path, speed, timestamps):
    """Interpolate a trajectory along a smooth path based on the given speed and timestamps."""
    arc_lengths = jnp.zeros(path.shape[0])
    arc_lengths = arc_lengths.at[1:].set(
        jnp.cumsum(jnp.linalg.norm(path[1:] - path[:-1], axis=1)),
    )

    desired_distances = speed * timestamps
    x_interp = jnp.interp(desired_distances, arc_lengths, path[:, 0])
    y_interp = jnp.interp(desired_distances, arc_lengths, path[:, 1])

    return jnp.stack((x_interp, y_interp), axis=1)
