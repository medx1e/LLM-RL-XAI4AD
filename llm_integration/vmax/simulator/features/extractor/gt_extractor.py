# Copyright 2025 Valeo.


"""Feature extractor for ground truth (GT) path target features."""

from typing import Any

import jax
import jax.numpy as jnp
import matplotlib as mpl
from waymax import datatypes
from waymax.utils import geometry

from vmax.simulator import features, operations
from vmax.simulator.features import extractor


class GTFeaturesExtractor(extractor.VecFeaturesExtractor):
    """Feature extractor for ground truth (GT) path target features."""

    def __init__(
        self,
        obs_past_num_steps: int | None = None,
        objects_config: dict[str, Any] | None = None,
        roadgraphs_config: dict[str, Any] | None = None,
        traffic_lights_config: dict[str, Any] | None = None,
        path_target_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the GT feature extractor.

        Args:
            obs_past_num_steps: Number of past steps to consider.
            objects_config: Configuration for object features.
            roadgraphs_config: Configuration for roadgraph features.
            traffic_lights_config: Configuration for traffic light features.
            path_target_config: Configuration for path target features.
        """
        super().__init__(
            obs_past_num_steps=obs_past_num_steps,
            path_target_config=path_target_config,
        )

    def extract_features(self, state: datatypes.SimulatorState, key: jax.Array) -> jax.Array:
        """Extract target features from the simulator state.

        Args:
            state: The simulator state.

        Returns:
            The extracted path target features.

        """
        sdc_observation = self._get_sdc_observation(state)

        path_target_features = self._build_target_features(state, sdc_observation)

        return jnp.array(()), jnp.array(()), jnp.array(()), jnp.array(()), path_target_features.data

    def plot_features(self, state: datatypes.SimulatorState, ax: mpl.axes.Axes) -> None:
        """Plot the path target features on a given axis.

        Args:
            state: The simulator state.
            ax: The matplotlib axes object.

        """
        sdc_observation = self._get_sdc_observation(state)

        path_target_features = self._build_target_features(state, sdc_observation)

        path_target_features.plot(ax)

    def _build_target_features(self, simulator_state, sdc_obs):
        """Build the target features for the SDC based on simulation state and observation.

        Args:
            simulator_state: The current simulator state.
            sdc_obs: The SDC observation.

        Returns:
            An instance of PathTargetFeatures.

        """
        batch_dims = simulator_state.batch_dims
        len_dims = len(batch_dims)

        sdc_idx = operations.get_index(sdc_obs.is_ego)
        sdc_log_xy = jnp.take_along_axis(simulator_state.log_trajectory.xy, sdc_idx[None, None, None], axis=len_dims)

        sdc_pose2d = sdc_obs.pose2d.matrix
        sdc_log_xy = geometry.transform_points(pts=sdc_log_xy, pose_matrix=sdc_pose2d).squeeze(axis=0)

        fill_value = sdc_log_xy[-1]

        indices = jnp.arange(self._num_target_path_points) + simulator_state.timestep

        target = jnp.take_along_axis(sdc_log_xy, indices[:, None], axis=0, mode="fill", fill_value=-1)
        target = jnp.where(target == -1, fill_value, target)
        target = extractor.normalize_path(target, self._max_meters)

        return features.PathTargetFeatures(xy=target)
