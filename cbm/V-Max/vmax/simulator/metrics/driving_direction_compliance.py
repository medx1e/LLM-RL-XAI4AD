# Copyright 2025 Valeo.

"""Module for driving direction compliance metric."""

import jax.numpy as jnp
from waymax import datatypes
from waymax.metrics import abstract_metric

from vmax.simulator import constants, operations
from vmax.simulator.metrics import utils


class DrivingDirectionComplianceMetric(abstract_metric.AbstractMetric):
    """Indicates if the ego vehicle is driving into oncoming traffic, and if so, what distance was parcoured."""

    def compute(self, simulator_state: datatypes.SimulatorState) -> abstract_metric.MetricResult:
        sdc_index = operations.get_index(simulator_state.object_metadata.is_sdc)
        ego_wrongway = is_ego_driving_into_oncoming_traffic(simulator_state)
        ego_speed = simulator_state.current_sim_trajectory.speed[sdc_index].squeeze()

        value = constants.TIME_DELTA * ego_speed * ego_wrongway
        valid = jnp.ones_like(value)
        result = abstract_metric.MetricResult(value=value, valid=valid)

        return result


def is_ego_driving_into_oncoming_traffic(state: datatypes.SimulatorState):
    sdc_index = operations.get_index(state.object_metadata.is_sdc)
    ego_corners = state.current_sim_trajectory.bbox_corners[sdc_index].squeeze()
    ego_z = state.current_sim_trajectory.z[sdc_index].squeeze()
    ego_yaw = state.current_sim_trajectory.yaw[sdc_index].squeeze()

    roadgraph_points = state.roadgraph_points

    # Add z dimension to corners
    z = jnp.ones_like(ego_corners[..., 0:1]) * ego_z
    ego_corners = jnp.concatenate([ego_corners, z], axis=-1)
    squared_distances = utils.get_corners_distance_to_lane_centers(ego_corners, roadgraph_points)

    # For each corner we have the closest lane_center
    # Now we filter to extract lane centers with aligned direction

    ego_dir = jnp.array([jnp.cos(ego_yaw), jnp.sin(ego_yaw)])
    aligned_lanes = aligned_lanes = jnp.dot(roadgraph_points.dir_xy, ego_dir) >= 0
    aligned_lanes = jnp.logical_and(aligned_lanes, roadgraph_points.valid)

    aligned_distances = jnp.where(aligned_lanes, squared_distances, jnp.inf)

    # Now for each corner, we can verify if is close enough from its closest aligned center
    # We must do this to manage intersections

    closest_aligned_lane_idx = jnp.argmin(aligned_distances, axis=-1)  # 4,
    selected_lane_centers = roadgraph_points.xy[closest_aligned_lane_idx]  # 4, 2
    selected_lane_dirs = roadgraph_points.dir_xy[closest_aligned_lane_idx]  # 4, 2
    selected_lane_normals = jnp.stack([-selected_lane_dirs[..., 1], selected_lane_dirs[..., 0]], axis=-1)  # 4, 2

    corners_to_centers = ego_corners[..., :2] - selected_lane_centers
    corners_to_centers = jnp.abs(jnp.sum(corners_to_centers * selected_lane_normals, axis=-1))

    no_aligned_lanes = jnp.max(corners_to_centers) > utils.LANE_WIDTH / 2 + utils.MARGIN

    # To differentiate between driving into oncoming traffic and driving on multiple lanes
    # We also check if the vehicle is on a misaligned lane

    misaligned_lanes = jnp.logical_and(~aligned_lanes, roadgraph_points.valid)
    misaligned_distances = jnp.where(misaligned_lanes, squared_distances, jnp.inf)

    closest_misaligned_lane_idx = jnp.argmin(misaligned_distances, axis=-1)  # 4,
    selected_lane_centers = roadgraph_points.xy[closest_misaligned_lane_idx]  # 4, 2
    selected_lane_dirs = roadgraph_points.dir_xy[closest_misaligned_lane_idx]  # 4, 2
    selected_lane_normals = jnp.stack([-selected_lane_dirs[..., 1], selected_lane_dirs[..., 0]], axis=-1)  # 4, 2

    corners_to_centers = ego_corners[..., :2] - selected_lane_centers
    corners_to_centers = jnp.abs(jnp.sum(corners_to_centers * selected_lane_normals, axis=-1))

    yes_misaligned_lanes = jnp.max(corners_to_centers) <= utils.LANE_WIDTH / 2 + utils.MARGIN

    return no_aligned_lanes & yes_misaligned_lanes
