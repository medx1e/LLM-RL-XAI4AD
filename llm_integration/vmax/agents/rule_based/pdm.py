# Copyright 2025 Valeo.

import functools

import jax
import jax.numpy as jnp
from waymax import datatypes
from waymax.utils import geometry

from vmax.agents.rule_based import idm
from vmax.simulator import constants


def select_action(observation: jax.Array, time_horizon: float = 8.0):
    """Select an action based on the observation and specified time horizon.

    Args:
        observation: The current observation.
        time_horizon: The time horizon for the action.

    Returns:
        The selected action.

    """
    unroll, new_speed, accel_action, _, _, _, danger, _, _ = generate_unrolls(observation, time_horizon=time_horizon)

    next_x, next_y = unroll[1]

    steering_action = idm.compute_steering(
        next_x,
        next_y,
        accel_action,
        new_speed,
    )

    accel_action = (
        jnp.clip(
            accel_action,
            -constants.MAX_ACCEL_BICYCLE,
            constants.MAX_ACCEL_BICYCLE,
        )
        / constants.MAX_ACCEL_BICYCLE
    )

    steering_action = steering_action / constants.MAX_STEERING

    accel_action = jnp.where(danger, -1.0, accel_action)
    steering_action = jnp.where(danger, 0.0, steering_action)

    curr_speed = observation[4]

    accel_action = jnp.where(
        curr_speed + accel_action * constants.TIME_DELTA * constants.MAX_ACCEL_BICYCLE < 0,
        -curr_speed / (constants.TIME_DELTA * constants.MAX_ACCEL_BICYCLE),
        accel_action,
    )

    return jnp.concatenate(
        [jnp.atleast_1d(accel_action), jnp.atleast_1d(steering_action)],
        axis=0,
    )


def generate_unrolls(observation: jax.Array, time_horizon: float = 8.0):
    """Generate multiple trajectory unrolls and select the best unroll."""
    (
        gps_target,
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

    gps_target = gps_target[::2]
    target_valid = target_valid[::2]

    sdc_vel_x, sdc_vel_y = sdc_features[2], sdc_features[3]

    def generate_trajectory(target_path: jax.Array, target_speed: float):
        obs = (
            target_path,
            target_valid,
            target_speed,
            sdc_features,
            curr_speed,
            other_agents_xy,
            other_agents_vel_xy,
            other_agents_dimensions,
            other_agents_valid,
            red_light_pos,
            roadgraph_points,
        )

        return idm.generate_unroll(obs, time_horizon=time_horizon)

    target_paths = jnp.array([gps_target, generate_offset_path(gps_target, 1), generate_offset_path(gps_target, -1)])

    target_speeds = jnp.arange(start=0.2, stop=1.2, step=0.2) * desired_speed

    num_paths = target_paths.shape[0]
    num_speeds = target_speeds.shape[0]

    batched_target_paths = jnp.repeat(target_paths, num_speeds, axis=0)
    batched_target_speeds = jnp.tile(target_speeds, num_paths)

    generate_trajectory_vmap = jax.vmap(generate_trajectory, in_axes=(0, 0))

    trajectories, new_speeds, accel_actions = generate_trajectory_vmap(
        batched_target_paths,
        batched_target_speeds,
    )

    speed_scores = (new_speeds - jnp.min(new_speeds)) / (jnp.max(new_speeds) - jnp.min(new_speeds) + 1e-6)
    speed_scores == speed_scores**2

    center_bonus = jnp.where(jnp.arange(num_speeds * num_paths) < num_speeds, 1, 0)

    simulate_bicycle = functools.partial(simulate_trajectory_bicycle, sdc_vel_x=sdc_vel_x, sdc_vel_y=sdc_vel_y)

    simulate_bicycle_vmap = jax.vmap(simulate_bicycle, in_axes=(0, 0))

    simulated_trajectories, sim_yaws = simulate_bicycle_vmap(trajectories, new_speeds)

    other_trajectories = generate_other_trajectories(
        other_agents_xy,
        other_agents_vel_xy,
        time_horizon=time_horizon,
    )

    other_agents_ttc, danger = collision_with_other_agents(
        simulated_trajectories,
        sim_yaws,
        sdc_features,
        other_trajectories,
        other_agents_dimensions,
        other_agents_valid,
    )

    overlap_ttc_scores = other_agents_ttc / time_horizon
    overlap_penalty = (other_agents_ttc < 2.0).astype(float)

    safe_speed_scores = jnp.where(other_agents_ttc <= 5.0, 0, speed_scores)

    offroad_ttc_scores = compute_offroad_score(simulated_trajectories, sim_yaws, sdc_features, roadgraph_points)
    offroad_ttc_scores = jnp.clip(offroad_ttc_scores, 0, 4.0)
    offroad_penalty = (offroad_ttc_scores < 2.0).astype(float)

    offroad_ttc_scores = offroad_ttc_scores / 4.0

    agg_score = (
        -48 * (overlap_penalty + offroad_penalty)
        + 3 * safe_speed_scores
        + 10 * (overlap_ttc_scores + offroad_ttc_scores)
        + center_bonus
    )
    agg_score /= 24

    # In some cases the target is absurd, we should raise the urgent braking manoeuver
    danger = danger + ~target_valid[0]

    best_traj = jnp.argmax(agg_score)
    return (
        trajectories[best_traj],
        new_speeds[best_traj],
        accel_actions[best_traj],
        trajectories,
        other_trajectories,
        agg_score,
        danger,
        simulated_trajectories[best_traj],
        sim_yaws[best_traj],
    )


def simulate_trajectory_bicycle(trajectory, new_speed, sdc_vel_x, sdc_vel_y):
    """Simulate a trajectory using a bicycle model with the provided speed and velocities."""
    new_traj = jnp.zeros_like(trajectory)

    def forward(x, y, vel_x, vel_y, yaw, accel, steering):
        speed = jnp.sqrt(vel_x**2 + vel_y**2)

        t = constants.TIME_DELTA

        new_x = x + vel_x * t + 0.5 * accel * jnp.cos(yaw) * t**2
        new_y = y + vel_y * t + 0.5 * accel * jnp.sin(yaw) * t**2
        delta_yaw = steering * (speed * t + 0.5 * accel * t**2)
        new_yaw = geometry.wrap_yaws(yaw + delta_yaw)
        new_vel = speed + accel * t
        new_vel_x = new_vel * jnp.cos(new_yaw)
        new_vel_y = new_vel * jnp.sin(new_yaw)

        return new_x, new_y, new_vel_x, new_vel_y, new_yaw

    def backward(x, y, vel_x, vel_y, yaw, target_x, target_y, target_speed):
        speed = jnp.sqrt(vel_x**2 + vel_y**2)
        t = constants.TIME_DELTA

        accel = (target_speed - speed) / t

        target_yaw = jnp.arctan2(target_y - y, target_x - x)
        delta_yaw = geometry.wrap_yaws(target_yaw - yaw)
        steering = delta_yaw / (speed * t + 0.5 * accel * t**2)

        accel = jnp.clip(accel, -constants.MAX_ACCEL_BICYCLE, constants.MAX_ACCEL_BICYCLE)
        steering = jnp.clip(steering, -constants.MAX_STEERING, constants.MAX_STEERING)

        return accel, steering

    def step_fn(carry, inputs):
        x, y, vel_x, vel_y, yaw = carry
        target_x, target_y, target_speed = inputs

        accel, steering = backward(x, y, vel_x, vel_y, yaw, target_x, target_y, target_speed)

        new_x, new_y, new_vel_x, new_vel_y, new_yaw = forward(x, y, vel_x, vel_y, yaw, accel, steering)

        next_carry = (new_x, new_y, new_vel_x, new_vel_y, new_yaw)
        output = next_carry

        return output, next_carry

    initial_state = (0.0, 0.0, sdc_vel_x, sdc_vel_y, 0.0)
    target_speeds = jnp.tile(new_speed, trajectory.shape[0])

    inputs = (trajectory[1:, 0], trajectory[1:, 1], target_speeds[1:])

    _, new_traj = jax.lax.scan(step_fn, initial_state, inputs)

    sim_traj = jnp.vstack([new_traj[0], new_traj[1], new_traj[4]]).T

    sim_traj = jnp.concatenate([jnp.zeros((1, 3)), sim_traj])

    return sim_traj[:, :2], sim_traj[:, 2]


def compute_offroad_score(trajectories, sdc_yaws, sdc_features, roadgraph_points):
    """Compute an offroad score for trajectories based on roadgraph information."""
    sdc_length, sdc_width = sdc_features[0], sdc_features[1]

    # Make x, y, l, w, yaw trajectories

    num_timesteps = trajectories.shape[1]
    num_trajectories = trajectories.shape[0]

    sdc_length = jnp.full((num_trajectories, num_timesteps), sdc_length)
    sdc_width = jnp.full((num_trajectories, num_timesteps), sdc_width)

    sdc_x = trajectories[:, :, 0]
    sdc_y = trajectories[:, :, 1]

    sdc_trajectories = jnp.stack(
        (sdc_x, sdc_y, sdc_length, sdc_width, sdc_yaws),
        axis=-1,
    )

    sdc_corners = jax.vmap(geometry.corners_from_bboxes)(
        sdc_trajectories,
    )  # (15, 41, 4, 2)

    # Compute distance to road_points, for now we will ignore the z-dimension

    sampled_points = roadgraph_points.xy  # (1000, 2)None

    squared_distances = jnp.sum(
        (sdc_corners[:, :, :, None, :] - sampled_points[None, None, None, :]) ** 2,
        axis=-1,
    )  # 15, 41, 4, 1000

    is_road_edge = datatypes.is_road_edge(roadgraph_points.types)

    squared_distances = jnp.where(
        roadgraph_points.valid & is_road_edge,
        squared_distances,
        jnp.inf,
    )

    nearest_indices = jnp.argmin(squared_distances, axis=-1)

    prior_indices = jnp.maximum(jnp.zeros_like(nearest_indices), nearest_indices - 1)

    nearest_xys = sampled_points[nearest_indices]

    points_to_edge = sdc_corners - nearest_xys  # 15, 41, 4, 2

    nearest_vector_xys = roadgraph_points.dir_xy[nearest_indices]
    prior_vector_xys = roadgraph_points.dir_xy[prior_indices]

    cross_product = jnp.cross(points_to_edge, nearest_vector_xys)
    cross_product_prior = jnp.cross(points_to_edge, prior_vector_xys)

    prior_point_in_same_curve = jnp.equal(
        roadgraph_points.ids[nearest_indices],
        roadgraph_points.ids[prior_indices],
    )

    offroad_sign = jnp.sign(
        jnp.where(
            jnp.logical_and(
                prior_point_in_same_curve,
                cross_product_prior < cross_product,
            ),
            cross_product_prior,
            cross_product,
        ),
    )  # 15, 41, 4

    is_offroad = jnp.any(offroad_sign > 0, axis=-1)

    first_offroad = jnp.argmax(is_offroad, axis=-1)

    ttc_offroad = first_offroad * constants.TIME_DELTA

    ttc_offroad = jnp.where(
        jnp.any(is_offroad, axis=1),
        ttc_offroad,
        (num_timesteps - 1) * constants.TIME_DELTA,
    )

    return ttc_offroad


def collision_with_other_agents(
    trajectories,
    sdc_yaws,
    sdc_features,
    other_trajectories,
    other_agents_dimensions,
    other_agents_valid,
):
    """Detect collisions between simulated trajectories and other agents."""
    sdc_length, sdc_width = sdc_features[0], sdc_features[1]

    # Make x, y, l, w, yaw trajectories

    num_timesteps = trajectories.shape[1]
    num_trajectories = trajectories.shape[0]

    sdc_length = jnp.full((num_trajectories, num_timesteps), sdc_length)
    sdc_width = jnp.full((num_trajectories, num_timesteps), sdc_width)

    sdc_x = trajectories[:, :, 0]
    sdc_y = trajectories[:, :, 1]
    sdc_trajectories = jnp.stack(
        (sdc_x, sdc_y, sdc_length, sdc_width, sdc_yaws),
        axis=-1,
    )

    other_lengths = other_agents_dimensions[:, 0]
    other_widths = other_agents_dimensions[:, 1]
    other_yaws = other_agents_dimensions[:, 2]

    other_lengths = jnp.tile(other_lengths[:, None], num_timesteps)
    other_widths = jnp.tile(other_widths[:, None], num_timesteps)
    other_yaws = jnp.tile(other_yaws[:, None], num_timesteps)

    other_x = other_trajectories[:, :, 0]
    other_y = other_trajectories[:, :, 1]

    other_agents_trajectories = jnp.stack(
        (other_x, other_y, other_lengths, other_widths, other_yaws),
        axis=-1,
    )

    check_collision = jax.vmap(
        jax.vmap(geometry.has_overlap, in_axes=(None, 0)),
        in_axes=(0, None),
    )

    all_collisions = check_collision(sdc_trajectories, other_agents_trajectories) * other_agents_valid[None, :, None]

    collisions = jnp.any(all_collisions, axis=1)

    is_colliding = jnp.any(collisions, axis=1)

    first_collision = jnp.argmax(collisions, axis=1)

    ttc = first_collision * constants.TIME_DELTA

    ttc = jnp.where(is_colliding, ttc, (num_timesteps - 1) * constants.TIME_DELTA)

    # For the danger braking, just consider vehicles in front

    front_vehicles = other_x[:, 0] > 0
    all_frontal_collisions = all_collisions * front_vehicles[None, :, None]
    frontal_collisions = jnp.any(all_frontal_collisions, axis=1)
    is_frontal_colliding = jnp.any(frontal_collisions, axis=1)

    first_frontal_collision = jnp.argmax(frontal_collisions, axis=1)

    ttc_front = first_frontal_collision * constants.TIME_DELTA

    ttc_front = jnp.where(is_frontal_colliding, ttc_front, (num_timesteps - 1) * constants.TIME_DELTA)

    danger = jnp.max(ttc_front) < 2

    return ttc, danger


def generate_other_trajectories(
    other_agents_xy,
    other_agents_vel_xy,
    time_horizon: float = 6.0,
):
    """Generate trajectories for other agents over the given time horizon."""
    timesteps = jnp.arange(0, time_horizon + constants.TIME_DELTA, constants.TIME_DELTA)
    trajs = other_agents_xy + other_agents_vel_xy * timesteps[:, None, None]

    trajs = jnp.transpose(trajs, (1, 0, 2))

    return trajs


def generate_offset_path(target_path: jax.Array, offset: float = 1.0):
    """Create an offset path from the provided target path."""
    # Generate a path that has the same direction but one lateral offset

    path_directions = jnp.diff(target_path, axis=0)
    path_directions = jnp.vstack([path_directions, path_directions[-1]])

    path_directions = jnp.where(
        path_directions == jnp.zeros_like(path_directions),
        jnp.array([0, 1]),
        path_directions,
    )

    path_directions /= jnp.linalg.norm(path_directions, axis=1)[:, None]

    rot_matrix = jnp.array([[0, 1], [-1, 0]])

    path_normals = jnp.dot(path_directions, rot_matrix)

    offset_path = target_path + offset * path_normals

    return offset_path
