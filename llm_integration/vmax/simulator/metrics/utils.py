# Copyright 2025 Valeo.


"""Utility functions for computing metrics."""

import jax
import jax.numpy as jnp
from jax.numpy.linalg import lstsq
from waymax import datatypes


LANE_WIDTH = 3.7  # meters (12 ft)
MARGIN = 0.2  # meters


def savgol_coeffs_jax(window_length, polyorder, deriv=0, delta=1.0, pos=None):
    """Compute Savitzky-Golay filter coefficients using JAX.

    Args:
        window_length: Window length for the filter.
        polyorder: Polynomial order to approximate the data.
        deriv: Order of derivative to compute.
        delta: Spacing of the samples.
        pos: Position in the window to evaluate the derivative.

    Returns:
        The computed filter coefficients.

    """
    if polyorder >= window_length:
        raise ValueError("polyorder must be less than window_length.")

    halflen, rem = divmod(window_length, 2)
    if pos is None:
        pos = halflen if rem == 1 else halflen - 0.5

    if not (0 <= pos < window_length):
        raise ValueError("pos must be nonnegative and less than window_length.")

    x = jnp.arange(-pos, window_length - pos, dtype=jnp.float32)
    x = x[::-1]  # Reverse for convolution

    order = jnp.arange(polyorder + 1).reshape(-1, 1)
    a = x**order

    y = jnp.zeros(polyorder + 1)
    y = y.at[deriv].set(jax.scipy.special.factorial(deriv) / (delta**deriv))

    coeffs, _, _, _ = lstsq(a, y)

    return coeffs


def savgol_filter_jax(x, window_length, polyorder, deriv=0, delta=1.0, mode="interp"):
    """Apply a Savitzky-Golay filter to the input array using JAX.

    Args:
        x: Input data sequence.
        window_length: Window length for the filter.
        polyorder: Polynomial order for the filter.
        deriv: Order of derivative to compute.
        delta: Spacing between samples.
        mode: Padding mode, where "interp" is implemented.

    Returns:
        The filtered array.

    """
    coeffs = savgol_coeffs_jax(window_length, polyorder, deriv=deriv, delta=delta)

    # Handle padding for the edges
    if mode == "interp":
        pad_width = window_length // 2
        x_padded = jnp.pad(x, pad_width, mode="reflect")
        y = jnp.convolve(x_padded, coeffs, mode="valid")
    else:
        raise ValueError("Currently, only mode='interp' is supported in JAX implementation.")

    # Ignore the edges for now
    y = jnp.where(jnp.arange(len(y)) < window_length // 2, y[window_length // 2], y)
    y = jnp.where(jnp.arange(len(y)) >= len(y) - window_length // 2, y[len(y) - window_length // 2 - 1], y)
    return y


def get_agent_relative_angle(ego_xy, ego_yaw, agent_xy):
    """Compute the relative angle between an agent and the ego vehicle's heading.

    Args:
        ego_xy: Ego vehicle coordinates.
        ego_yaw: Ego vehicle heading (yaw).
        agent_xy: Agent coordinates.

    Returns:
        The relative angle between the agent and the ego vehicle.

    """
    ego_direction = jnp.array([jnp.cos(ego_yaw), jnp.sin(ego_yaw)])

    agent_to_ego = agent_xy - ego_xy
    dot_product = jnp.dot(agent_to_ego, ego_direction) / jnp.linalg.norm(agent_to_ego, axis=-1)

    agent_relative_angle = jnp.arccos(dot_product)
    agent_relative_angle = jnp.where(jnp.all(agent_to_ego == 0), 0, agent_relative_angle)

    return agent_relative_angle


def is_agent_behind(ego_xy, ego_yaw, agent_xy, angle_tolerance=150):
    """Determine if an agent is behind the ego vehicle.

    Args:
        ego_xy: Ego vehicle coordinates.
        ego_yaw: Ego vehicle heading (yaw).
        agent_xy: Agent coordinates.
        angle_tolerance: Tolerance angle in degrees.

    Returns:
        Boolean indicating if the agent is behind.

    """
    return get_agent_relative_angle(ego_xy, ego_yaw, agent_xy) > jnp.deg2rad(angle_tolerance)


def is_agent_ahead(ego_xy, ego_yaw, agent_xy, angle_tolerance=30):
    """Determine if an agent is ahead of the ego vehicle.

    Args:
        ego_xy: Ego vehicle coordinates.
        ego_yaw: Ego vehicle heading (yaw).
        agent_xy: Agent coordinates.
        angle_tolerance: Tolerance angle in degrees.

    Returns:
        Boolean indicating if the agent is ahead.

    """
    return get_agent_relative_angle(ego_xy, ego_yaw, agent_xy) < jnp.deg2rad(angle_tolerance)


def get_distance_to_lane_centers(
    ego_xyz: jax.Array,
    roadgraph_points: datatypes.RoadgraphPoints,
    z_stretch: float = 2.0,
):
    """Calculate distances from the ego vehicle to lane centers with a vertical stretch factor.

    Args:
        ego_xyz: The (x, y, z) position of the ego vehicle.
        roadgraph_points: Roadgraph points information.
        z_stretch: Stretch factor for the z-coordinate.

    Returns:
        Distances between the ego vehicle and each lane center.

    """
    sampled_points = roadgraph_points.xyz

    differences = sampled_points - ego_xyz[None, :]

    # Stretch difference in altitude to avoid over/underpasses.
    z_stretched_differences = differences * jnp.array([1.0, 1.0, z_stretch])
    squared_distances = jnp.sum(z_stretched_differences**2, axis=-1)

    is_lane_center = jnp.isin(roadgraph_points.types, jnp.array([1, 2]))
    squared_distances = jnp.where(is_lane_center & roadgraph_points.valid, squared_distances, jnp.inf)

    return squared_distances


def get_closest_lane_center_idx(
    ego_xyz: jax.Array,
    roadgraph_points: datatypes.RoadgraphPoints,
    z_stretch: float = 2.0,
):
    """Retrieve the index of the closest lane center to the ego vehicle.

    Args:
        ego_xyz: The (x, y, z) position of the ego vehicle.
        roadgraph_points: Roadgraph points information.
        z_stretch: Stretch factor for z-coordinate differences.

    Returns:
        The index of the closest lane center.

    """
    squared_distances = get_distance_to_lane_centers(ego_xyz, roadgraph_points, z_stretch)
    return jnp.argmin(squared_distances)


def get_corners_distance_to_roadgraph_points(
    ego_corners: jax.Array,
    roadgraph_points: datatypes.RoadgraphPoints,
    z_stretch: float = 2.0,
):
    """Compute distances from the vehicle corners to each roadgraph point with a vertical stretch.

    Args:
        ego_corners: Corner coordinates of the ego vehicle.
        roadgraph_points: Roadgraph points information.
        z_stretch: Stretch factor for the z-coordinate.

    Returns:
        An array of distances from each corner to the roadgraph points.

    """
    sampled_points = roadgraph_points.xyz
    differences = sampled_points[None, :] - ego_corners[:, None, :]  # 4, num_roadgraph_points, 3

    z_stretched_differences = differences * jnp.array([1.0, 1.0, z_stretch])
    squared_distances = jnp.sum(z_stretched_differences**2, axis=-1)
    squared_distances = jnp.where(roadgraph_points.valid, squared_distances, jnp.inf)
    return squared_distances


def get_corners_distance_to_lane_centers(
    ego_corners: jax.Array,
    roadgraph_points: datatypes.RoadgraphPoints,
    z_stretch: float = 2.0,
):
    """Compute distances from the vehicle corners to lane centers using a vertical stretch.

    Args:
        ego_corners: Corner coordinates of the ego vehicle.
        roadgraph_points: Roadgraph points information.
        z_stretch: Stretch factor for the z-coordinate.

    Returns:
        An array of distances from each corner to the lane centers.

    """
    squared_distances = get_corners_distance_to_roadgraph_points(ego_corners, roadgraph_points, z_stretch)
    is_lane_center = jnp.isin(roadgraph_points.types, jnp.array([1, 2]))
    squared_distances = jnp.where(is_lane_center, squared_distances, jnp.inf)
    return squared_distances


def get_corners_distance_to_lane_center(
    ego_xyz: jax.Array,
    ego_corners: jax.Array,
    roadgraph_points: datatypes.RoadgraphPoints,
    lane_center_idx: int,
):
    """Compute distances from vehicle corners to a specified lane center.

    Args:
        ego_xyz: The (x, y, z) coordinates of the ego vehicle.
        ego_corners: Coordinates of the vehicle corners.
        roadgraph_points: Roadgraph points information.
        lane_center_idx: Index of the lane center.

    Returns:
        Distances from each corner to the given lane center.

    """
    lane_center = roadgraph_points.xy[lane_center_idx]
    lane_tangent_direction = roadgraph_points.dir_xy[lane_center_idx]
    lane_normal_direction = jnp.array([-lane_tangent_direction[1], lane_tangent_direction[0]])

    corners_to_lane = ego_corners - lane_center[None, :]
    corners_to_lane = jnp.sum(corners_to_lane * lane_normal_direction[None, :], axis=-1)

    return jnp.abs(corners_to_lane)


def get_corners_distance_to_closest_lane_center(
    ego_xyz: jax.Array,
    ego_corners: jax.Array,
    roadgraph_points: datatypes.RoadgraphPoints,
):
    """Compute distances from vehicle corners to the closest lane center.

    Args:
        ego_xyz: The (x, y, z) coordinates of the ego vehicle.
        ego_corners: Coordinates of the vehicle corners.
        roadgraph_points: Roadgraph points information.

    Returns:
        The distances from each corner to the closest lane center.

    """
    nearest_lane_center_idx = get_closest_lane_center_idx(ego_xyz, roadgraph_points)
    return get_corners_distance_to_lane_center(ego_xyz, ego_corners, roadgraph_points, nearest_lane_center_idx)
