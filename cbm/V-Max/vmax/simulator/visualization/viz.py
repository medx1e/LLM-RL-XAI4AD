# Copyright 2025 Valeo.

"""Module for visualization of the simulator."""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import beta
from waymax import config as waymax_config
from waymax import datatypes
from waymax.visualization import utils as waymax_utils
from waymax.visualization import viz as waymax_viz

from vmax.simulator import overrides


def plot_input_agent(
    state: datatypes.SimulatorState,
    env: Any,
    viz_config: dict[str, Any] | None = None,
    batch_idx: int = -1,
    highlight_obj: waymax_config.ObjectType = waymax_config.ObjectType.SDC,
    plot_sdc_paths: bool = True,
) -> np.ndarray:
    """Plot np array image for SimulatorState.

    Args:
        state: A SimulatorState instance.
        env: The environment instance.
        viz_config: dict for optional config.
        batch_idx: optional batch index.
        highlight_obj: Represents the type of objects that will be highlighted with
            `color.COLOR_DICT['controlled']` color.
        plot_sdc_paths: Flag to plot SDC paths.

    Returns:
        np image.

    """
    if batch_idx > -1:
        if len(state.shape) != 1:
            raise ValueError(f"Expecting one batch dimension, got {len(state.shape)}")
        state = waymax_viz._index_pytree(state, batch_idx)
    if state.shape:
        raise ValueError(f"Expecting 0 batch dimension, got {len(state.shape)}")

    viz_config = waymax_utils.VizConfig() if viz_config is None else waymax_utils.VizConfig(**viz_config)

    dict_viz_config = dataclasses.asdict(viz_config)

    features_extractor = env.get_wrapper_attr("features_extractor")
    sdc_observation = features_extractor._get_sdc_observation(state)

    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(16, 8))
    fig.set_facecolor("white")

    # First ax: SimulatorState
    overrides.plot_simulator_state(
        state=state,
        use_log_traj=False,
        viz_config=dict_viz_config,
        highlight_obj=highlight_obj,
        plot_sdc_paths=plot_sdc_paths,
        ax=axs[0],
    )

    axs[0].set_aspect("equal", "box")

    # Second ax: Observation
    overrides.plot_observation(
        obs=sdc_observation,
        viz_config=dict_viz_config,
        highlight_obj=highlight_obj,
        plot_sdc_paths=plot_sdc_paths,
        ax=axs[1],
    )
    axs[1].set_aspect("equal", "box")

    # Third ax: Features
    features_extractor.plot_features(state, axs[2])
    axs[2].axis((-1, +1, -1, +1))
    axs[2].set_aspect("equal", "box")

    return waymax_utils.img_from_fig(fig)


def plot_metrics(
    state: datatypes.SimulatorState,
    env: Any,
    viz_config: dict[str, Any] | None = None,
    batch_idx: int = -1,
    highlight_obj: waymax_config.ObjectType = waymax_config.ObjectType.SDC,
    plot_sdc_paths: bool = True,
) -> np.ndarray:
    """Plot np array image showing SimulatorState and merged metric plots.

    Layout:
      - Top: SimulatorState plotted in a centered, wide axis.
      - Bottom: A single axis with merged metric bars grouped by type.

    Args:
        state: A SimulatorState instance.
        env: The environment instance.
        viz_config: dict for optional config.
        batch_idx: Optional batch index.
        highlight_obj: The type of object to highlight.
        plot_sdc_paths: Flag to plot SDC paths.

    Returns:
        np array image.

    """
    if batch_idx > -1:
        if len(state.shape) != 1:
            raise ValueError(f"Expecting one batch dimension, got {len(state.shape)}")
        state = waymax_viz._index_pytree(state, batch_idx)
    if state.shape:
        raise ValueError(f"Expecting 0 batch dimension, got {len(state.shape)}")

    viz_config = waymax_utils.VizConfig() if viz_config is None else waymax_utils.VizConfig(**viz_config)
    dict_viz_config = dataclasses.asdict(viz_config)

    # Increase the figure size for better visibility.
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1])
    ax_state = fig.add_subplot(gs[0])
    ax_metrics = fig.add_subplot(gs[1])
    fig.set_facecolor("white")
    fig.subplots_adjust(bottom=0.2)  # Extra space at bottom

    # Plot SimulatorState
    overrides.plot_simulator_state(
        state=state,
        use_log_traj=False,
        viz_config=dict_viz_config,
        batch_idx=batch_idx,
        highlight_obj=highlight_obj,
        plot_sdc_paths=plot_sdc_paths,
        ax=ax_state,
    )
    ax_state.set_aspect("equal", "box")

    # Obtain metrics from environment.
    metrics = env.metrics(state)
    reward = env.reward(state, None)

    # Remove title, place reward text on the left
    ax_state.text(
        -0.4,
        0.90,
        f"Reward: {reward:.2f}",
        transform=ax_state.transAxes,
        ha="left",
        va="top",
        fontsize=12,
    )

    xs = []
    vals = []
    labels = []
    pos = 0

    for key in metrics:
        v = metrics[key].value
        clipped_v = min(max(v, 0), 2)
        xs.append(pos)
        vals.append(clipped_v)
        labels.append(key)
        pos += 1
    pos += 0.5  # additional gap between groups

    # Plot merged metrics with colored bars and edge colors.
    ax_metrics.bar(xs, vals, align="center", color="black", edgecolor="black")
    ax_metrics.set_xticks(xs)
    ax_metrics.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax_metrics.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    return waymax_utils.img_from_fig(fig)


# Training plots (Q values, gaussian curves)
def img_from_fig2(fig) -> np.ndarray:
    """Returns a [H, W, 3] uint8 np image from fig.canvas.tostring_rgb()."""
    # Just enough margin in the figure to display xticks and yticks.
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return img


def plot_gaussian_curves(list_values, params):
    fig, ax = plt.subplots(3, 1, figsize=(6, 6))

    accel_mean = params[0][0]
    accel_std = params[0][2]
    steering_mean = params[0][1]
    steering_std = params[0][3]
    x = np.linspace(-1, 1, 1000)

    accel_y = np.exp(-0.5 * ((x - accel_mean) / accel_std) ** 2) / (accel_std * np.sqrt(2 * np.pi))
    steering_y = np.exp(-0.5 * ((x - steering_mean) / steering_std) ** 2) / (steering_std * np.sqrt(2 * np.pi))

    ax[0].plot(x, accel_y, label="Accel")
    ax[0].set_title("Accel")
    ax[1].plot(x, steering_y, label="Steering")
    ax[1].set_title("Steering")
    ax[2].plot(list_values)
    ax[2].set_title("Critics Q Values")

    fig.tight_layout()
    ax[0].legend([f"Mean: {accel_mean:.4f}\nStd: {accel_std:.4f}"])
    ax[1].plot(x, steering_y, label="Steering")
    ax[1].legend([f"Mean: {steering_mean:.4f}\nStd: {steering_std:.4f}"])

    return img_from_fig2(fig)


def plot_beta_curves(list_values, params):
    fig, ax = plt.subplots(3, 1, figsize=(6, 6))

    # Beta distribution parameters (alpha, beta)
    accel_alpha = params[0][0]
    accel_beta = params[0][2]
    steering_alpha = params[0][1]
    steering_beta = params[0][3]

    # x values must be in the range [0, 1] for the Beta distribution
    x = np.linspace(0, 1, 1000)

    # Beta distribution PDFs for accel and steering
    accel_y = beta.pdf(x, accel_alpha, accel_beta)
    steering_y = beta.pdf(x, steering_alpha, steering_beta)

    # Plotting the Beta distribution curves
    ax[0].plot(x, accel_y, label="Accel")
    ax[0].set_title("Accel")
    ax[1].plot(x, steering_y, label="Steering")
    ax[1].set_title("Steering")
    ax[2].plot(list_values)
    ax[2].set_title("Critics Q Values")

    fig.tight_layout()

    # Adding legends with alpha and beta parameters
    ax[0].legend([f"Alpha: {accel_alpha:.4f}, Beta: {accel_beta:.4f}"])
    ax[1].legend([f"Alpha: {steering_alpha:.4f}, Beta: {steering_beta:.4f}"])

    return img_from_fig2(fig)


def plot_heatmap_q_values(
    values,
    accel_values,
    steer_values,
    highlighted_points,
    grid_size=(10, 10),
):
    z = values.reshape(grid_size)
    x_labels = [f"{val:.1f}" for val in accel_values]
    y_labels = [f"{val:.1f}" for val in steer_values]

    # Create the heatmap with annotations
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        z,
        annot=True,
        fmt=".1f",
        cmap="viridis",
        cbar=True,
        xticklabels=x_labels,
        yticklabels=y_labels,
    )

    # Highlight the specified points
    for point in highlighted_points:
        x, y = point

        # Find the closest grid cell
        x_idx = (jnp.abs(accel_values - x)).argmin()
        y_idx = (jnp.abs(steer_values - y)).argmin()

        # Add a red rectangle to highlight the cell
        ax.add_patch(
            plt.Rectangle((x_idx, y_idx), 1, 1, fill=False, edgecolor="red", lw=2),
        )

    plt.xlabel("Accel")
    plt.ylabel("Steering")

    return img_from_fig2(ax.get_figure())


def plot_pdm_debug(  # noqa: C901
    state: datatypes.SimulatorState,
    obs: datatypes.Observation,
    obj_idx: int = 0,
    viz_config: dict[str, Any] | None = None,
    batch_idx: int = -1,
    highlight_obj: waymax_config.ObjectType = waymax_config.ObjectType.SDC,
    plot_sdc_paths: bool = False,
    planned_trajectory=None,
    proposals=None,
    proposal_scores=None,
    other_trajectories=None,
    target_paths=None,
    simulated_trajectory=None,
) -> np.ndarray:
    """Plot np array image for an Observation.

    Args:
      obs: An Observation instance, with shape (..., obs_A), where obs_A
        represents the number of objects that have observation view over things
        including other objects, roadgraph, and traffic lights.
      obj_idx: The object index in obs_A.
      viz_config: Dict for optional config.
      batch_idx: Optional batch index.
      highlight_obj: Represents the type of objects that will be highlighted with
        `color.COLOR_DICT['controlled']` color.

    Returns:
      np image.

    """
    if batch_idx > -1:
        if len(obs.shape) != 2:
            raise ValueError(f"Expecting ndim 2 for obs, got {len(obs.shape)}")
        obs = jax.tree_util.tree_map(lambda x: x[batch_idx], obs)

    # Shape: (obs_A,) -> ()
    obs = jax.tree_map(lambda x: x[obj_idx], obs)
    if obs.shape:
        raise ValueError(f"Expecting shape () for obs, got {obs.shape}")

    viz_config = waymax_utils.VizConfig() if viz_config is None else waymax_utils.VizConfig(**viz_config)
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

    overrides.plot_trajectory(
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

    if planned_trajectory is not None:
        nb_waypoints = planned_trajectory.shape[0]
        time_steps = np.linspace(0, 1, nb_waypoints)  # Normalize time to [0, 1] for colormap

        # Use a colormap, e.g., 'viridis', 'plasma', or 'coolwarm'
        cmap = plt.cm.viridis

        # Plot each point with a color based on time
        for i in range(nb_waypoints):  # Iterate through points
            ax.plot(
                planned_trajectory[i : i + 2, 0],  # X values
                planned_trajectory[i : i + 2, 1],  # Y values
                color=cmap(time_steps[i]),  # Color based on normalized time
                marker="o",
            )

    if proposals is not None and proposal_scores is not None:
        for i, proposal in enumerate(proposals):
            cmap = plt.cm.RdYlGn

            ax.scatter(
                proposal[:, 0],
                proposal[:, 1],
                color=cmap(proposal_scores[i]),
                marker="x",
            )

    if other_trajectories is not None:
        for other_trajectory in other_trajectories:
            nb_waypoints = other_trajectory.shape[0]
            time_steps = np.linspace(0, 1, nb_waypoints)  # Normalize time to [0, 1] for colormap

            # Use a colormap, e.g., 'viridis', 'plasma', or 'coolwarm'
            cmap = plt.cm.viridis

            # Plot each point with a color based on time
            for i in range(nb_waypoints):  # Iterate through points
                ax.plot(
                    other_trajectory[i : i + 2, 0],  # X values
                    other_trajectory[i : i + 2, 1],  # Y values
                    color=cmap(time_steps[i]),  # Color based on normalized time
                    marker="o",
                )

    if simulated_trajectory is not None:
        time_idx = simulated_trajectory.shape[0]

        cmap = plt.cm.viridis

        time_steps = np.linspace(0, 1, time_idx)  # Normalize time to [0, 1] for colormap

        # Plot each point with a color based on time
        for j in range(time_idx):
            color = np.array(cmap(time_steps[j])[:3])

            # if j ==59: color=np.array([255,0,0])/255.0

            waymax_utils.plot_numpy_bounding_boxes(
                ax=ax,
                bboxes=simulated_trajectory[j][jnp.newaxis, :],
                color=color,
                alpha=0.3,
                as_center_pts=False,
            )

    if target_paths is not None:
        for target_path in target_paths:
            ax.scatter(
                target_path[:, 0],
                target_path[:, 1],
                color="black",
                marker="x",
            )
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

    return waymax_utils.img_from_fig(fig)
