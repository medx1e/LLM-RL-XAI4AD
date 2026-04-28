# Copyright 2025 Valeo.


import jax.numpy as jnp
from waymax import datatypes
from waymax.metrics import abstract_metric

from vmax.simulator import operations
from vmax.simulator.metrics import utils


SPEED_LIMITS_PER_LANE_TYPE = {1: 70.0, 2: 45.0}  # miles per hour


class SpeedLimitViolationMetric(abstract_metric.AbstractMetric):
    """Metric to measure ego vehicle speed limit violation in m/s."""

    def compute(self, simulator_state: datatypes.SimulatorState) -> abstract_metric.MetricResult:
        """Compute the speed limit violation metric.

        Args:
            simulator_state: Current simulator state.

        Returns:
            A MetricResult with the speed violation value and validity.

        """
        speed_limit = infer_speed_limit_from_simulator_state(simulator_state)
        sdc_index = operations.get_index(simulator_state.object_metadata.is_sdc)
        ego_speed = simulator_state.current_sim_trajectory.speed[sdc_index].squeeze()

        value = ego_speed - speed_limit
        value = jnp.where(value > 0, value, 0)
        valid = jnp.ones_like(value, dtype=jnp.bool_)

        result = abstract_metric.MetricResult.create_and_validate(jnp.float32(value), valid)

        return result


def infer_speed_limit_from_simulator_state(state: datatypes.SimulatorState):
    """Infer the speed limit from the simulator state by combining roadgraph and expert trajectory information.

    Args:
        state: Simulator state.

    Returns:
        The inferred speed limit in m/s.

    """
    speed_limit_roadgraph, lane_type = infer_speed_limit_from_roadgraph(state)
    speed_limit_expert = infer_speed_limit_from_log_trajectory(state)

    # Never go more than 45 mph on surface street or less than 70mph on freeway
    speed_limit = jnp.where(
        lane_type == 1,
        jnp.maximum(speed_limit_roadgraph, speed_limit_expert),
        jnp.minimum(speed_limit_roadgraph, speed_limit_expert),
    )

    return speed_limit


def infer_speed_limit_from_roadgraph(state: datatypes.SimulatorState):
    """Infer the speed limit based on roadgraph data.

    Args:
        state: Simulator state.

    Returns:
        A tuple containing the speed limit in m/s and the lane type.

    """
    sdc_index = operations.get_index(state.object_metadata.is_sdc)
    ego_xyz = state.current_sim_trajectory.xyz[sdc_index].squeeze()
    roadgraph_points = state.roadgraph_points

    nearest_lane_center_idx = utils.get_closest_lane_center_idx(ego_xyz, roadgraph_points)

    lane_type = roadgraph_points.types[nearest_lane_center_idx]

    # Avoid concretization error but it is ugly
    speed_limit_mph = jnp.where(lane_type == 1, SPEED_LIMITS_PER_LANE_TYPE[1], SPEED_LIMITS_PER_LANE_TYPE[2])

    return speed_limit_mph / 2.237, lane_type  # convert to meters per second


def infer_speed_limit_from_log_trajectory(state: datatypes.SimulatorState):
    """Infer the speed limit based on the expert (logged) trajectory.

    Args:
        state: Simulator state.

    Returns:
        The inferred speed limit in m/s.

    """
    sdc_index = operations.get_index(state.object_metadata.is_sdc)
    expert_speed = state.log_trajectory.speed[sdc_index]

    max_expert_speed = jnp.max(expert_speed)
    # Possible speed limits in San Francisco and Phoenix
    speed_limits_mph = jnp.array([25.0, 35.0, 45.0, 70.0])
    speed_limits_m_s = speed_limits_mph / 2.237

    speed_limit = jnp.searchsorted(speed_limits_m_s, max_expert_speed)

    return speed_limits_m_s[speed_limit]
