# Copyright 2025 Valeo.


import jax.numpy as jnp
from waymax import datatypes
from waymax.metrics import abstract_metric
from waymax.utils import geometry

from vmax.simulator import constants, operations
from vmax.simulator.metrics import utils


class OnMultipleLanesMetric(abstract_metric.AbstractMetric):
    """Metric to detect if the ego vehicle is on multiple lanes."""

    def compute(self, simulator_state: datatypes.SimulatorState) -> abstract_metric.MetricResult:
        """Compute the on multiple lanes metric.

        Args:
            simulator_state: Current simulator state.

        Returns:
            A MetricResult with 1.0 if the ego vehicle is on multiple lanes, otherwise 0.

        """
        sdc_index = operations.get_index(simulator_state.object_metadata.is_sdc)
        on_multiple_lanes = is_ego_on_multiple_lanes(simulator_state)
        ego_speed = simulator_state.current_sim_trajectory.speed[sdc_index].squeeze()

        value = ego_speed * on_multiple_lanes * constants.TIME_DELTA
        valid = jnp.ones_like(value, dtype=jnp.bool_)

        result = abstract_metric.MetricResult.create_and_validate(value.astype(jnp.float32), valid)

        return result


def is_ego_on_multiple_lanes(state: datatypes.SimulatorState):
    """Check if the ego vehicle is on multiple lanes.

    Args:
        state (datatypes.SimulatorState): The current state of the simulator.

    Returns:
        bool: True if the ego vehicle is on multiple lanes, False otherwise.

    """
    sdc_index = operations.get_index(state.object_metadata.is_sdc)

    traj_5dof = state.current_sim_trajectory.stack_fields(["x", "y", "length", "width", "yaw"]).squeeze()
    ego_xyz = state.current_sim_trajectory.xyz[sdc_index].squeeze()
    ego_corners = geometry.corners_from_bbox(traj_5dof[sdc_index])
    roadgraph_points = state.roadgraph_points

    corners_distance_to_lane_center = utils.get_corners_distance_to_closest_lane_center(
        ego_xyz,
        ego_corners,
        roadgraph_points,
    )

    return jnp.max(corners_distance_to_lane_center) > utils.LANE_WIDTH / 2 + utils.MARGIN
