# Copyright 2023 The Waymax Authors.
#
# Licensed under the Waymax License Agreement for Non-commercial Use
# Use (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     https://github.com/waymo-research/waymax/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Visualization functions for Waymax data structures."""

from typing import Any

import jax
import matplotlib as mpl
import numpy as np
from waymax import config as waymax_config
from waymax import datatypes
from waymax.utils import geometry
from waymax.visualization import color
from waymax.visualization import utils as waymax_utils
from waymax.visualization import viz as waymax_viz

from vmax.simulator import overrides


# The type ids for road graph elements that will be plot in visualization.
# Please refer to color.py for definition and color associcated.
_RoadGraphShown = (1, 2, 3, 15, 16, 17, 18, 19)
_RoadGraphDefaultColor = (0.9, 0.9, 0.9)


def _plot_bounding_boxes(
    ax: mpl.axes.Axes,
    traj_5dof: np.ndarray,
    time_idx: int,
    is_controlled: np.ndarray,
    obj_types: np.ndarray,
    valid: np.ndarray,
) -> None:
    """Plot bounding boxes on the axis for different time steps.

    Args:
        ax: Matplotlib axis to plot on.
        traj_5dof: Array containing bounding box data with shape (A, T, 5).
        time_idx: Current time index to highlight the bounding boxes.
        is_controlled: Boolean mask of controlled objects with shape (A,).
        obj_types: Array of object types.
        valid: Boolean array indicating valid bounding boxes at each time step (A, T).

    """
    # Plots bounding boxes (traj_5dof) with shape: (A, T)
    # is_controlled: (A,)
    # valid: (A, T)
    valid_controlled = is_controlled[:, np.newaxis] & valid
    valid_context = ~is_controlled[:, np.newaxis] & valid

    num_obj = traj_5dof.shape[0]
    time_indices = np.tile(np.arange(traj_5dof.shape[1])[np.newaxis, :], (num_obj, 1))
    # Shrinks bounding_boxes for non-current steps.
    traj_5dof[time_indices != time_idx, 2:4] /= 10
    # SDC trajectory
    overrides.plot_numpy_bounding_boxes(
        ax=ax,
        bboxes=traj_5dof[(time_indices == time_idx) & valid_controlled],  # >= for future
        color=np.array([255, 192, 203]) / 255.0,  # color.COLOR_DICT["controlled"],
        alpha=0.6,
    )

    # Past trajectories of SDC
    overrides.plot_numpy_bounding_boxes(
        ax=ax,
        bboxes=traj_5dof[(time_indices < time_idx) & (time_idx - time_indices < 20) & valid_controlled],
        color=np.array([255, 192, 203]) / 255.0,  # np.random.choice(range(256), size=3) / 255.0,
        as_center_pts=True,
        alpha=0.4,
    )

    surrounding_indices = np.unique(np.where(valid_context == True)[0])
    mask_tracked_obj = np.zeros(num_obj, dtype=bool)

    for tracked_obj in surrounding_indices:
        mask_tracked_obj[tracked_obj] = True
        valid_tracked_obj = mask_tracked_obj[:, np.newaxis] & valid

        if obj_types[tracked_obj] == 2:
            obj_color = np.array([153, 102, 204]) / 255.0  # next(colors)[:3]
        elif obj_types[tracked_obj] == 3:
            obj_color = np.array([210, 180, 140]) / 255.0
        else:
            obj_color = np.array([173, 216, 230]) / 255.0

        # Past trajectories of surrounding objects & SDC
        overrides.plot_numpy_bounding_boxes(
            ax=ax,
            bboxes=traj_5dof[(time_indices < time_idx) & (time_idx - time_indices < 20) & valid_tracked_obj],
            color=obj_color,
            as_center_pts=True,
            alpha=0.5,
        )
        # Future trajectories of surrounding objects
        overrides.plot_numpy_bounding_boxes(
            ax=ax,
            bboxes=traj_5dof[(time_indices == time_idx) & valid_tracked_obj],  # >= for future
            color=obj_color,  # color.COLOR_DICT["context"],
        )

        # False if passed current tracked object
        mask_tracked_obj[tracked_obj] = False

    # Shows current overlap
    # (A, A)
    overlap_fn = jax.jit(geometry.compute_pairwise_overlaps)
    overlap_mask_matrix = overlap_fn(traj_5dof[:, time_idx])

    # Remove overlap against invalid objects.
    overlap_mask_matrix = np.where(valid[None, :, time_idx], overlap_mask_matrix, False)
    # (A,)
    overlap_mask = np.any(overlap_mask_matrix, axis=1)

    overrides.plot_numpy_bounding_boxes(
        ax=ax,
        bboxes=traj_5dof[:, time_idx][overlap_mask & valid[:, time_idx]],
        color=color.COLOR_DICT["overlap"],
        overlap=True,
    )


def plot_trajectory(
    ax: mpl.axes.Axes,
    traj: datatypes.Trajectory,
    is_controlled: np.ndarray,
    obj_types: np.ndarray,
    time_idx: int | None = None,
    indices: np.ndarray | None = None,
) -> None:
    """Visualize a trajectory by plotting its bounding boxes.

    Args:
        ax: Matplotlib axis to draw on.
        traj: Trajectory data with shape (A, T).
        is_controlled: Boolean mask indicating controlled objects.
        obj_types: Array of object type values.
        time_idx: The time step index to highlight; use -1 for the last step.
        indices: Optional array of object IDs to annotate the plot.

    """
    if len(traj.shape) != 2:
        raise ValueError("traj should have shape (A, T)")

    traj_5dof = np.array(
        traj.stack_fields(["x", "y", "length", "width", "yaw"]),
    )  # Forces to np from jnp

    num_obj, num_steps, _ = traj_5dof.shape
    if time_idx is not None:
        if time_idx == -1:
            time_idx = num_steps - 1
        if time_idx >= num_steps:
            raise ValueError("time_idx is out of range.")

    # Adds id if needed.
    if indices is not None and time_idx is not None:
        for i in range(num_obj):
            if not traj.valid[i, time_idx]:
                continue

            ax.text(
                traj_5dof[i, time_idx, 0] - 2,
                traj_5dof[i, time_idx, 1] + 2,
                f"{indices[i]}",
                zorder=10,
            )

    _plot_bounding_boxes(
        ax,
        traj_5dof,
        time_idx,
        is_controlled,
        obj_types,
        traj.valid,
    )  # pytype: disable=wrong-arg-types  # jax-ndarray


def plot_traffic_light_signals_as_points(
    ax: mpl.axes.Axes,
    tls: datatypes.TrafficLights,
    timestep: int = 0,
    verbose: bool = False,
) -> None:
    """Plot traffic light signals as points on the given axis.

    Args:
        ax: Matplotlib axis for plotting.
        tls: Traffic lights data.
        timestep: Time step index to plot.
        verbose: If True, prints the traffic lights count.

    """
    if len(tls.shape) != 2:
        raise ValueError("Traffic light shape wrong.")

    valid = tls.valid[:, timestep]
    if valid.sum() == 0:
        return
    elif verbose:
        print(f"Traffic lights count: {valid.sum()}")

    tls_xy = tls.xy[:, timestep][valid]
    tls_state = tls.state[:, timestep][valid]

    for xy, tl_state in zip(tls_xy, tls_state, strict=False):
        tl_color = color.TRAFFIC_LIGHT_COLORS[int(tl_state)]
        ax.plot(xy[0], xy[1], marker="o", color=tl_color, ms=4)


def plot_simulator_state(
    state: datatypes.SimulatorState,
    use_log_traj: bool = True,
    viz_config: dict[str, Any] | None = None,
    batch_idx: int = -1,
    highlight_obj: waymax_config.ObjectType = waymax_config.ObjectType.SDC,
    plot_sdc_paths: bool = False,
    ax: mpl.axes.Axes | None = None,
) -> np.ndarray:
    """Plot the simulator state and return the result as an image.

    Args:
        state: Simulator state containing trajectory, road graph, and traffic light data.
        use_log_traj: Whether to use logged trajectory or simulated trajectory.
        viz_config: Optional configuration dictionary.
        batch_idx: Batch index to select from state.
        highlight_obj: Object type to highlight.
        plot_sdc_paths: If True, plots SDC paths.
        ax: Optional Matplotlib axis; if None, one is created.


    """
    if batch_idx > -1:
        if len(state.shape) != 1:
            raise ValueError(f"Expecting one batch dimension, got {len(state.shape)}")
        state = waymax_viz._index_pytree(state, batch_idx)
    if state.shape:
        raise ValueError(f"Expecting 0 batch dimension, got {len(state.shape)}")

    viz_config = waymax_utils.VizConfig() if viz_config is None else waymax_utils.VizConfig(**viz_config)

    create_fig = ax is None

    if create_fig:
        fig, ax = waymax_utils.init_fig_ax(viz_config)

    # 1. Plots trajectory.
    traj = state.log_trajectory if use_log_traj else state.sim_trajectory
    indices = np.arange(traj.num_objects) if viz_config.show_agent_id else None
    is_controlled = datatypes.get_control_mask(state.object_metadata, highlight_obj)
    obj_types = state.object_metadata.object_types

    # Check SDC indice
    plot_trajectory(
        ax,
        traj,
        is_controlled,
        obj_types,
        state.timestep,
        indices,
    )  # pytype: disable=wrong-arg-types  # jax-ndarray

    # 2. Plots road graph elements.
    waymax_viz.plot_roadgraph_points(ax, state.roadgraph_points, verbose=False)
    waymax_viz.plot_traffic_light_signals_as_points(
        ax,
        state.log_traffic_light,
        state.timestep,
        verbose=False,
    )

    # 3. Gets np img, centered on selected agent's current location.
    # [A, 2]
    current_xy = traj.xy[:, state.timestep, :]
    xy = (
        current_xy[state.object_metadata.is_sdc]
        if viz_config.center_agent_idx == -1
        else current_xy[viz_config.center_agent_idx]
    )
    origin_x, origin_y = xy[0, :2]
    ax.axis(
        (
            origin_x - viz_config.back_x,
            origin_x + viz_config.front_x,
            origin_y - viz_config.back_y,
            origin_y + viz_config.front_y,
        ),
    )

    # Plots SDC paths.
    if plot_sdc_paths:
        waymax_viz._plot_path_points(ax, state.sdc_paths)

    if create_fig:
        return waymax_utils.img_from_fig(fig)


def plot_observation(
    obs: datatypes.Observation,
    obj_idx: int = 0,
    viz_config: dict[str, Any] | None = None,
    batch_idx: int = -1,
    highlight_obj: waymax_config.ObjectType = waymax_config.ObjectType.SDC,
    plot_sdc_paths: bool = False,
    ax: mpl.axes.Axes | None = None,
) -> np.ndarray:
    """Plot an observation and return it as an image.

    Args:
        obs: Observation data.
        obj_idx: Index of the object to focus on.
        viz_config: Optional visualization configuration.
        batch_idx: Batch index for observation.
        highlight_obj: Object type to highlight.
        plot_sdc_paths: If True, plots SDC paths.
        ax: Optional Matplotlib axis.

    """
    if batch_idx > -1:
        if len(obs.shape) != 2:
            raise ValueError(f"Expecting ndim 2 for obs, got {len(obs.shape)}")
        obs = jax.tree_util.tree_map(lambda x: x[batch_idx], obs)

    # Shape: (obs_A,) -> ()
    if len(obs.shape) == 1:
        obs = jax.tree_map(lambda x: x[obj_idx], obs)
    if obs.shape:
        raise ValueError(f"Expecting shape () for obs, got {obs.shape}")

    viz_config = waymax_utils.VizConfig() if viz_config is None else waymax_utils.VizConfig(**viz_config)

    create_fig = ax is None

    if create_fig:
        fig, ax = waymax_utils.init_fig_ax(viz_config)

    # 1. Plots trajectory.
    # Shape: (num_objects, num_timesteps).
    traj = obs.trajectory
    # The current timestep index of observation is the last one in time dimension
    # as observation is toward the past.
    timestep = traj.num_timesteps - 1
    indices = np.arange(traj.num_objects) if viz_config.show_agent_id else None

    is_controlled = datatypes.get_control_mask(obs.metadata, highlight_obj)
    obj_types = obs.metadata.object_types

    plot_trajectory(
        ax,
        traj,
        is_controlled,
        obj_types,
        timestep,
        indices,
    )  # pytype: disable=wrong-arg-types  # jax-ndarray

    # 2. Plots road graph elements.
    # Shape: (num_points,)
    waymax_viz.plot_roadgraph_points(ax, obs.roadgraph_static_points, verbose=False)

    # Shape: (num_traffic_lights, num_timesteps).
    waymax_viz.plot_traffic_light_signals_as_points(
        ax,
        obs.traffic_lights,
        timestep,
        verbose=False,
    )

    if plot_sdc_paths:
        waymax_viz._plot_path_points(ax, obs.sdc_paths)

    # 3. Gets np img, centered on selected agent's current location.
    # Shape (num_objects, 2).
    current_xy = traj.xy[:, timestep, :]
    xy = (
        current_xy[obs.metadata.is_sdc]
        if viz_config.center_agent_idx == -1
        else current_xy[viz_config.center_agent_idx]
    )
    origin_x, origin_y = xy[0, :2]
    ax.axis(
        (
            origin_x - viz_config.back_x,
            origin_x + viz_config.front_x,
            origin_y - viz_config.back_y,
            origin_y + viz_config.front_y,
        ),
    )

    if create_fig:
        return waymax_utils.img_from_fig(fig)
