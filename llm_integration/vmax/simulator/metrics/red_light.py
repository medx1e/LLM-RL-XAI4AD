# Copyright 2025 Valeo.


"""Run red light metric module."""

import jax
from jax import numpy as jnp
from waymax import datatypes
from waymax.metrics import abstract_metric

from vmax.simulator import operations


class RunRedLightMetric(abstract_metric.AbstractMetric):
    """Metric to detect red light violations by the ego vehicle."""

    def compute(self, simulator_state: datatypes.SimulatorState) -> abstract_metric.MetricResult:
        """Compute the red light metric indicating if the ego vehicle has run a red light.

        Args:
            simulator_state: Current simulator state.

        Returns:
            A MetricResult with 1.0 if a red light was run, otherwise 0.

        """
        value = _has_runned_red_light(simulator_state)
        valid = jnp.ones_like(value, dtype=bool)

        return abstract_metric.MetricResult.create_and_validate(value, valid)


def _has_runned_red_light(state: datatypes.SimulatorState) -> float:
    """Check if the ego vehicle ran a red light at the current time step.

    Args:
        state: Current simulator state.

    Returns:
        1.0 if the ego vehicle ran a red light, 0.0 otherwise.

    """
    sdc_idx = operations.get_index(state.object_metadata.is_sdc)

    idx_selected_path = operations.select_longest_sdc_path_id(state.sdc_paths)
    last_rg_id, current_rg_id = _get_current_and_last_lane_id(state, sdc_idx, idx_selected_path)

    has_changed_lane = last_rg_id != current_rg_id

    closed_line = _closed_line_time_t(state)
    # current_rg_id in closed_line
    is_current_rg_on_closed_line = jnp.any(closed_line - current_rg_id == 0)

    has_runned_rl = jnp.logical_and(has_changed_lane, is_current_rg_on_closed_line).sum() > 0

    return has_runned_rl.astype(jnp.float32)


def get_id_red_for_sdc(sdc_obs: datatypes.Observation) -> jax.Array:
    """Get the identifier for the red light the ego vehicle is expected to encounter.

    Args:
        sdc_obs: Observation data for the ego vehicle.

    Returns:
        The red light identifier, or -1 if none is detected.

    """
    path_id = operations.select_longest_sdc_path_id(sdc_obs.sdc_paths)
    lane_ids = sdc_obs.sdc_paths.ids[path_id]

    traffic_lights_ids = sdc_obs.traffic_lights.lane_ids[:, 0]
    # Sort ids in distance order
    traffic_lights_ids = traffic_lights_ids[jnp.argsort(jnp.linalg.norm(sdc_obs.traffic_lights.xy[:, 0, :], axis=1))]

    mask_traffic_lights_ids = jnp.isin(traffic_lights_ids, lane_ids)
    id_ = traffic_lights_ids[jnp.argmax(mask_traffic_lights_ids)]

    tf = jnp.argwhere(sdc_obs.traffic_lights.lane_ids[:, 0] == id_, size=1).squeeze()

    return jnp.where(jnp.sum(mask_traffic_lights_ids) == 0, -1, tf)


def get_closest_rg_id_on_selected_path(
    state: datatypes.SimulatorState,
    reference_point: jax.Array,
    idx_selected_path: int,
) -> jax.Array:
    """Return the closest roadgraph point identifier on the selected path to a reference point.

    Args:
        state: Simulator state.
        reference_point: The reference (x, y) position.
        idx_selected_path: Index of the selected sdc path.

    Returns:
        The roadgraph point identifier closest to the reference point.

    """
    # (1, num_points_per_path, 2) -> (num_points_per_path, 2)
    sdc_path_xy = state.sdc_paths.xy[idx_selected_path]
    # (num_points_per_path,)
    distances = jnp.linalg.norm(reference_point[None,] - sdc_path_xy, axis=1)
    # (num_points_per_path, )
    distances = jnp.where(state.sdc_paths.valid[idx_selected_path], distances, jnp.inf)
    # top_idx shape : (1, topk=1)
    top_idx = operations.get_index(-distances)
    # (1, topk=1)
    closest_rg_id = state.sdc_paths.ids[idx_selected_path][top_idx]

    # (1,)
    return closest_rg_id.flatten()


def _closed_line_time_t(state: datatypes.SimulatorState) -> jax.Array:
    """Get IDs of lanes with a current red traffic light at the current time step.

    Args:
        state: Simulator state.

    Returns:
        An array of lane identifiers corresponding to closed (red) lights.

    """
    time_step = state.timestep
    traf_light = state.log_traffic_light
    red_ids = jnp.array([1, 4, 7])
    traf_light.lane_ids[:, time_step]
    mask_is_red_light_red = jnp.isin(traf_light.state[:, time_step], red_ids)

    # -2 is an invalid index that is not taken in the sdc path
    return jnp.where(mask_is_red_light_red, traf_light.lane_ids[:, time_step], -2)


def get_previous_lane_id_before_closed_line_idx(selected_path: jax.Array, closed_line_ids: jax.Array) -> jax.Array:
    """Fetch the lane identifier preceding a closed lane on the selected path.

    Args:
        selected_path: Array of lane identifiers forming the sdc path.
        closed_line_ids: Closed lane identifiers.

    Returns:
        The identifier of the lane before the closed lane, or -1 if none exists.

    """
    is_point_id_of_closed_line = jnp.isin(selected_path, closed_line_ids)

    first_idx = jnp.argmax(is_point_id_of_closed_line)

    idx = jax.lax.cond(
        is_point_id_of_closed_line[first_idx] != 0,
        lambda x: jnp.maximum(0, x - 1),
        lambda x: -1,
        first_idx,
    )  # (1,)
    previous_lane_id = selected_path[idx]
    return previous_lane_id


def _get_current_and_last_lane_id(
    state: datatypes.SimulatorState,
    sdc_idx: int,
    idx_selected_path: int,
) -> tuple[jax.Array, jax.Array]:
    """Retrieve the current and previous lane identifiers for the ego vehicle.

    Args:
        state: Simulator state.
        sdc_idx: Index of the ego vehicle.
        idx_selected_path: Index of the selected sdc path.

    Returns:
        A tuple (last_rg_id, current_rg_id) representing the previous and current lane identifiers.

    """
    # (2,)
    sdc_xy_t = _get_front_position_sdc(state.sim_trajectory, sdc_idx, state.timestep)
    # (2,)
    sdc_xy_t_minus_1 = _get_front_position_sdc(state.sim_trajectory, sdc_idx, state.timestep - 1)

    last_rg_id = get_closest_rg_id_on_selected_path(state, sdc_xy_t_minus_1, idx_selected_path)
    current_rg_id = get_closest_rg_id_on_selected_path(state, sdc_xy_t, idx_selected_path)

    return last_rg_id, current_rg_id


def _get_front_position_sdc(trajectory: datatypes.Trajectory, sdc_idx: int, timestep: int) -> jax.Array:
    """Get the front position of the ego vehicle.

    Args:
        trajectory: The trajectory data.
        sdc_idx: Index of the ego vehicle.

    Returns:
        The front position of the ego vehicle.

    """
    # (4, 2)
    bbox_corners = trajectory.bbox_corners[sdc_idx][timestep]
    # (2, 2)
    two_ponts_front = bbox_corners[:2]
    # (2,)
    front_position = jnp.mean(two_ponts_front, axis=0)

    return front_position
