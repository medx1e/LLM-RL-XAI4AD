# Copyright 2025 Valeo.

"""Feature extractor for IDM (Intelligent Driver Model)."""

from typing import Any

import jax
import jax.numpy as jnp
from waymax import datatypes

from vmax.simulator import operations
from vmax.simulator.features import extractor
from vmax.simulator.metrics import infer_speed_limit_from_simulator_state, red_light


class IDMFeaturesExtractor(extractor.VecFeaturesExtractor):
    """Feature extractor for Intelligent Driver Model (IDM)."""

    def __init__(
        self,
        obs_past_num_steps: int | None = None,
        objects_config: dict[str, Any] | None = None,
        roadgraphs_config: dict[str, Any] | None = None,
        traffic_lights_config: dict[str, Any] | None = None,
        path_target_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the IDM feature extractor.

        Args:
            obs_past_num_steps: Number of past steps to consider.
            objects_config: Configuration for object features.
            roadgraphs_config: Configuration for roadgraph features.
            traffic_lights_config: Configuration for traffic light features.
            path_target_config: Configuration for path target features.
        """
        super().__init__(
            obs_past_num_steps=obs_past_num_steps,
            objects_config=objects_config,
            roadgraphs_config=roadgraphs_config,
            traffic_lights_config=traffic_lights_config,
            path_target_config=path_target_config,
        )

    def extract_features(
        self,
        state: datatypes.SimulatorState,
        key: jax.Array = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Extract features from the simulator state for IDM.

        Args:
            state: The simulator state.
            key: Random key for noise sampling.

        Returns:
            A tuple containing:
              - SDC object features.
              - Other objects features.
              - Roadgraph features.
              - Traffic lights features.
              - Path target features.
        """
        sdc_observation = self._get_sdc_observation(state, key)

        # Extract target path
        target = sdc_observation.sdc_paths.xy[0]
        target_valid = sdc_observation.sdc_paths.valid[0]

        # Extract SDC and other agents' information
        sdc_index = operations.get_index(state.object_metadata.is_sdc, squeeze=False)
        agents_indices = operations.get_index(
            ~state.object_metadata.is_sdc,
            k=state.num_objects - 1,
            squeeze=False,
        )

        # SDC features
        sdc_features = sdc_observation.trajectory.stack_fields(
            ["length", "width", "vel_x", "vel_y"],
        )[sdc_index].squeeze()

        sdc_curr_speed = sdc_observation.trajectory.speed[sdc_index].squeeze()

        # Other agents' features
        other_agents_xy = sdc_observation.trajectory.xy[agents_indices].squeeze()
        other_agents_vel_xy = sdc_observation.trajectory.vel_xy[agents_indices].squeeze()
        other_agents_dimensions = sdc_observation.trajectory.stack_fields(
            ["length", "width", "yaw"],
        )[agents_indices].squeeze()
        other_agents_valid = sdc_observation.trajectory.valid[agents_indices].squeeze()

        # Get red light position
        red_light_pos = self._compute_red_light_position(state, sdc_observation)

        # Get desired speed
        desired_speed = infer_speed_limit_from_simulator_state(state)

        # Roadgraph features
        roadgraph_points = sdc_observation.roadgraph_static_points

        return (
            target,
            target_valid,
            desired_speed,
            sdc_features,
            sdc_curr_speed,
            other_agents_xy,
            other_agents_vel_xy,
            other_agents_dimensions,
            other_agents_valid,
            red_light_pos,
            roadgraph_points,
        )

    def _prepare_sdc_features(self, sdc_features, sdc_curr_speed, desired_speed):
        """Prepare SDC features for the observation wrapper."""
        # Format and normalize SDC features
        return jnp.expand_dims(
            jnp.concatenate([sdc_features, jnp.array([sdc_curr_speed, desired_speed])]),
            axis=0,
        )

    def _prepare_other_agents_features(self, xy, vel_xy, dimensions, valid):
        """Prepare other agents' features for the observation wrapper."""
        # Combine and format other agents' features
        num_agents = xy.shape[0]
        features_list = []

        for i in range(num_agents):
            agent_features = jnp.concatenate([xy[i], vel_xy[i], dimensions[i], jnp.expand_dims(valid[i], axis=0)])
            features_list.append(agent_features)

        return jnp.stack(features_list)

    def _prepare_roadgraph_features(self, roadgraph_points):
        """Prepare roadgraph features for the observation wrapper."""
        # Format roadgraph points data
        return roadgraph_points.xy

    def _prepare_traffic_light_features(self, red_light_pos):
        """Prepare traffic light features for the observation wrapper."""
        # Format red light position
        return jnp.expand_dims(red_light_pos[0], axis=0)

    def _prepare_path_target_features(self, target, target_valid):
        """Prepare path target features for the observation wrapper."""
        # Combine target path with validity
        return jnp.where(jnp.expand_dims(target_valid, axis=-1), target, jnp.zeros_like(target))

    def _compute_red_light_position(self, state: datatypes.SimulatorState, sdc_obs: datatypes.Observation) -> jax.Array:
        """Compute the red light position for the SDC.

        Args:
            state: The current simulator state.
            sdc_obs: The observation extracted for the SDC.

        Returns:
            A tuple with the red light position and a status flag.
        """
        red_light_for_sdc_xy = self._get_position_for_sdc_red_light(sdc_obs)

        selected_path_idx = operations.select_longest_sdc_path_id(state.sdc_paths)
        selected_path = state.sdc_paths.ids[selected_path_idx]
        closed_line_ids = self._closed_line_time_t(state)
        previous_lane_id = red_light.get_previous_lane_id_before_closed_line_idx(
            selected_path,
            closed_line_ids,
        )
        sdc_idx = operations.get_index(state.object_metadata.is_sdc)
        current_rg_id = red_light.get_closest_rg_id_on_selected_path(
            state,
            state.current_sim_trajectory.xy[sdc_idx].flatten(),
            selected_path_idx,
        )
        cond = current_rg_id == previous_lane_id
        red_light_for_sdc_xy = jnp.where(
            cond,
            red_light_for_sdc_xy,
            jnp.array([jnp.inf, jnp.inf]),
        )
        does_red_light_exist = jnp.where(
            jnp.any(red_light_for_sdc_xy == jnp.inf),
            False,
            True,
        )
        status_red_light = jnp.where(does_red_light_exist, 1, 0)
        return red_light_for_sdc_xy, status_red_light  # (inf, inf, 0) if none

    def _get_position_for_sdc_red_light(self, sdc_obs: datatypes.Observation) -> jax.Array:
        """Get the red light position for the SDC.

        Args:
            sdc_obs: The observation containing traffic light data.

        Returns:
            The position of the red light.
        """
        idx_xy_red_light = red_light.get_id_red_for_sdc(sdc_obs)
        red_light_for_sdc_xy = sdc_obs.traffic_lights.xy[idx_xy_red_light][0]
        return red_light_for_sdc_xy

    def _closed_line_time_t(self, state: datatypes.SimulatorState) -> jax.Array:
        """Determine the closed lane ids for red light assessment.

        Args:
            state: Current simulator state.

        Returns:
            An array of lane ids or -2 for invalid indices.
        """
        time_step = state.timestep
        traf_light = state.log_traffic_light
        red_ids = jnp.array([1, 2, 4, 5, 7, 8])

        mask_is_red_light_red = jnp.isin(traf_light.state[:, time_step], red_ids)

        # -2 is an invalid index that is not taken in the sdc path
        return jnp.where(mask_is_red_light_red, traf_light.lane_ids[:, time_step], -2)
