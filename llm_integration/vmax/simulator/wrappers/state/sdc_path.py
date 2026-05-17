# Copyright 2025 Valeo.


"""Wrapper and helpers for SDC path planning."""

import chex
import jax
import jax.numpy as jnp
from waymax import datatypes

from vmax.simulator.wrappers.base import Wrapper


class SDCPathWrapper(Wrapper):
    """Wrapper to inject SDC paths in the SimulatorState."""

    def __init__(self, env: Wrapper) -> None:
        """Initialize the SDC path wrapper.

        Args:
            env: Environment to wrap.

        """
        super().__init__(env)
        self._sdc_path = True

    def reset(self, state: datatypes.SimulatorState, rng: jax.Array) -> datatypes.SimulatorState:
        """Reset the environment by adding SDC path information.

        Args:
            state: Current simulator state.
            rng: Random number generator.

        Returns:
            Updated state with SDC path integrated.

        """
        chex.assert_equal(state.shape, ())

        state = add_sdc_path_to_simulator_state(state)

        return self.env.reset(state, rng)


def _is_lane_center(roadgraph: datatypes.RoadgraphPoints) -> jax.Array:
    """Identify lane center points based on roadgraph types.

    Args:
        roadgraph: Roadgraph data.

    Returns:
        Boolean mask for lane center points.

    """
    return roadgraph.types == datatypes.MapElementIds.LANE_SURFACE_STREET


def get_sdc_lane(
    roadgraph: datatypes.RoadgraphPoints,
    sdc_xy: jax.Array,
    sdc_yaw: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Get the lane for the SDC based on position and orientation.

    Args:
        roadgraph: Roadgraph data.
        sdc_xy: SDC position.
        sdc_yaw: SDC orientation.

    Returns:
        Lane id, corresponding center point, and index interval.

    """
    mask_lane = _is_lane_center(roadgraph)

    roadgraph_xy = roadgraph.xy
    roadgraph_ids = roadgraph.ids
    roadgraph_dir = roadgraph.dir_xy
    roadgraph_yaw = jnp.arctan2(roadgraph_dir[:, 1], roadgraph_dir[:, 0])
    roadgraph_yaw = jnp.roll(roadgraph_yaw, -1)

    # Mask lane that are in the same direction as the sdc
    mask_lane = jnp.logical_and(mask_lane, jnp.abs(roadgraph_yaw - sdc_yaw) < 0.1)

    centerlane_xy = jnp.where(mask_lane[:, None], roadgraph_xy, jnp.inf)
    centerlane_dist = jnp.linalg.norm(centerlane_xy - sdc_xy, axis=-1)
    closest_idx_centerlane = jnp.argmin(centerlane_dist)

    closest_id_centerlane = roadgraph_ids[closest_idx_centerlane]

    num_points = roadgraph_ids.shape[0]

    def cond_fun(carry):
        i = carry
        in_bounds = i < num_points
        diff = jnp.linalg.norm(roadgraph_xy[i] - roadgraph_xy[i - 1])
        same_lane = jnp.logical_and(roadgraph_ids[i] == closest_id_centerlane, diff < 1.001)

        return jnp.logical_and(in_bounds, same_lane)

    def body_fun(carry):
        i = carry
        return i + 1

    init_val = closest_idx_centerlane
    final_idx = jax.lax.while_loop(cond_fun, body_fun, init_val) - 1

    return closest_id_centerlane, roadgraph_xy[final_idx], jnp.array([closest_idx_centerlane, final_idx])


def get_next_lane(
    roadgraph: datatypes.RoadgraphPoints,
    lanes_xy: jax.Array,
    previous_lane_id: int,
    previous_lane_point: jax.Array,
    sdc_trajectory: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Determine the next lane using current lane information and SDC trajectory.

    Args:
        roadgraph: Roadgraph data.
        lanes_xy: Lane center points.
        previous_lane_id: Previous lane identifier.
        previous_lane_point: Previous lane center point.
        sdc_trajectory: Trajectory history of the SDC.

    Returns:
        Updated lane centers, next lane id, next lane center, and index range.

    """
    # Mask roadgraph points that are not the same lane
    lanes_xy = jnp.where(roadgraph.ids[:, None] != previous_lane_id, lanes_xy, jnp.inf)

    # Mask roadgraph points that are not in the same direction
    mask = jnp.sum(roadgraph.dir_xy, axis=-1) != 0
    masked_xy = jnp.where(mask[:, None], lanes_xy, jnp.inf)

    # Get the index of the lane center points that are the same as the lane center
    dist = jnp.round(jnp.linalg.norm(masked_xy - previous_lane_point, axis=-1), decimals=4)
    all_idx = jnp.argwhere(dist == 0, size=3, fill_value=-1)[:, 0]

    # Add an offset of 4 to get the next point
    next_point_idx = jnp.where(all_idx == -1, -1, all_idx + 4)
    next_point_idx = jnp.where(next_point_idx >= roadgraph.ids.shape[0], -1, next_point_idx)

    # Get the closest point to the sdc trajectory in the next lane
    dist_sdc_traj = jnp.linalg.norm(masked_xy[next_point_idx] - sdc_trajectory[-1], axis=-1)
    closest_idx_sdc_traj = all_idx[jnp.argmin(dist_sdc_traj)]
    closest_id_sdc_traj = roadgraph.ids[closest_idx_sdc_traj]

    lane_id = jnp.where(closest_id_sdc_traj == -1, -2, closest_id_sdc_traj)
    lane_size = jnp.sum(lane_id == roadgraph.ids) - 1
    lane_point = roadgraph.xy[closest_idx_sdc_traj + lane_size]

    return masked_xy, lane_id, lane_point, jnp.array([closest_idx_sdc_traj, closest_idx_sdc_traj + lane_size])


def get_path_target_lane_ids(simulator_state: datatypes.SimulatorState) -> jax.Array:
    """Retrieve lane ids for computing the SDC path target.

    Args:
        simulator_state: Current simulator state.

    Returns:
        Array of lane identifiers for target computation.

    """
    roadgraph = simulator_state.roadgraph_points

    sdc_idx = jnp.argmax(simulator_state.object_metadata.is_sdc)
    sdc_xy = simulator_state.current_sim_trajectory.xy[sdc_idx]
    sdc_yaw = simulator_state.current_sim_trajectory.yaw[sdc_idx]
    sdc_trajectory = simulator_state.log_trajectory.xy[sdc_idx]

    def body_fun(carry, _x):
        lanes_xy, previous_id, previous_point = carry
        lanes_xy, next_id, next_point, tuple_idx = get_next_lane(
            roadgraph,
            lanes_xy,
            previous_id,
            previous_point,
            sdc_trajectory,
        )

        return (lanes_xy, next_id, next_point), tuple_idx

    # Mask roadgraph points that are not lane center
    lanes_xy = jnp.where(roadgraph.types[:, None] == datatypes.MapElementIds.LANE_SURFACE_STREET, roadgraph.xy, jnp.inf)

    _id, next_point, _tuple_idx = get_sdc_lane(roadgraph, sdc_xy, sdc_yaw)

    init_val = (lanes_xy, _id, next_point)

    list_idx = jax.lax.scan(body_fun, init_val, jnp.arange(10))[1]

    return jnp.vstack([_tuple_idx, list_idx])


def build_path_target(simulator_state: datatypes.SimulatorState) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Construct the SDC path target.

    Args:
        simulator_state: Current simulator state.

    Returns:
        A tuple containing target coordinates, lane ids, and validity flags.

    """

    def body_fn(carry, idx):
        start, end = idx

        def test_fn(i, val):
            j, x, y, ids, valids = val
            x = x.at[j].set(simulator_state.roadgraph_points.x[i])
            y = y.at[j].set(simulator_state.roadgraph_points.y[i])
            ids = ids.at[j].set(simulator_state.roadgraph_points.ids[i])
            valids = valids.at[j].set(simulator_state.roadgraph_points.valid[i])

            return j + 1, x, y, ids, valids

        updated = jax.lax.fori_loop(start, end, test_fn, carry)

        return updated, None

    list_idx = get_path_target_lane_ids(simulator_state)
    count = 0

    result = jax.lax.scan(body_fn, (count, jnp.zeros(300), jnp.zeros(300), jnp.zeros(300), jnp.zeros(300)), list_idx)[0]

    _, x, y, ids, valids = result

    path_target = jnp.stack([x, y], axis=-1)

    return path_target, ids, valids


def add_sdc_path_to_simulator_state(state: datatypes.SimulatorState) -> datatypes.SimulatorState:
    """Integrate SDC path planning into the simulator state.

    Args:
        state: Current simulator state.

    Returns:
        Updated simulator state with SDC path information.

    """
    path_target, ids, valids = build_path_target(state)

    # (1, 300)
    x = path_target[None, :, 0]
    # (1, 300)
    y = path_target[None, :, 1]
    # (1, 300)
    z = jnp.zeros_like(x)
    # (1, 300)
    ids = ids[None, :].astype(jnp.int32)
    # (1, 300)
    valid = valids[None, :].astype(jnp.bool_)
    # (299)
    arc_length = jnp.cumsum(jnp.linalg.norm(path_target[1:] - path_target[:-1], axis=1))
    # (1, 300)
    arc_length = jnp.concatenate([jnp.array([0]), arc_length])[None, :]
    # (1, 1)
    on_route = jnp.ones((1, 1)).astype(jnp.bool_)

    return state.replace(
        sdc_paths=datatypes.Paths(
            x=x,
            y=y,
            z=z,
            ids=ids,
            valid=valid,
            arc_length=arc_length,
            on_route=on_route,
        ),
    )
