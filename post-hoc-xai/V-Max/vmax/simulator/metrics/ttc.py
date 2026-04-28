# Copyright 2025 Valeo.


"""Time to collision metric module."""

import jax
from jax import numpy as jnp
from waymax import datatypes
from waymax.metrics import abstract_metric
from waymax.utils import geometry

from vmax.simulator import constants, operations
from vmax.simulator.metrics import utils


_TIME_HORIZON = 5.0  # seconds


class TimeToCollisionMetric(abstract_metric.AbstractMetric):
    """Time-to-collision metric."""

    def compute(self, simulator_state: datatypes.SimulatorState) -> abstract_metric.MetricResult:
        """Compute the time-to-collision (TTC) metric.

        Args:
            simulator_state: Current simulator state.

        Returns:
            A MetricResult containing the minimum TTC value and its validity.

        """
        current_traj = datatypes.dynamic_slice(
            simulator_state.sim_trajectory,
            simulator_state.timestep,
            1,
            -1,
        )
        sdc_index = operations.get_index(simulator_state.object_metadata.is_sdc)

        ttc, valid = self._compute_ttc(current_traj, sdc_index, dt=constants.TIME_DELTA, time_horizon=_TIME_HORIZON)
        is_ahead = self._only_ahead_agents(current_traj, sdc_index)

        ttc = jnp.where(is_ahead, ttc, _TIME_HORIZON)

        value = jnp.min(ttc)
        valid = jnp.ones_like(value, dtype=bool)

        return abstract_metric.MetricResult.create_and_validate(value, valid)

    def _compute_ttc(
        self,
        current_traj: datatypes.Trajectory,
        sdc_index: int,
        dt: float = 0.1,
        time_horizon: float = 5.0,
    ):
        """Compute the time-to-collision for each object using constant speed and heading.

        Args:
            current_traj: Trajectory data for all objects.
            sdc_index: Index of the ego vehicle.
            dt: Time step duration.
            time_horizon: Maximum time horizon for collision prediction.

        Returns:
            A tuple (ttc, valid) where ttc is an array of time-to-collision values and valid is the validity mask.

        """
        traj_5dof = current_traj.stack_fields(["x", "y", "length", "width", "yaw"])
        velocities = current_traj.vel_xy
        timesteps = jnp.arange(0, time_horizon, dt)

        initial_positions = traj_5dof[:, :, :2]
        other_fields = traj_5dof[:, :, 2:]

        future_positions = initial_positions + velocities * timesteps[None, :, None]

        other_fields = jnp.broadcast_to(
            other_fields,
            (future_positions.shape[0], future_positions.shape[1], other_fields.shape[-1]),
        )

        # (num_objects, num_timesteps, 5)
        future_traj_5dof = jnp.concatenate([future_positions, other_fields], axis=-1)

        # vmap over the objects dimension (-2)
        check_overlap = jax.vmap(geometry.has_overlap, (0, None), -1)

        # vmap again to compute pairwise overlaps
        check_overlap = jax.vmap(check_overlap, (None, 0), -1)

        # vmap across time
        check_overlap = jax.vmap(check_overlap, 1)

        # (num_timesteps, num_objects, num_objects)
        collision_matrix = check_overlap(future_traj_5dof, future_traj_5dof)

        # Remove diagonals (self-collisions)
        self_mask = jnp.eye(traj_5dof.shape[0])
        collision_matrix = jnp.where(self_mask[None], False, collision_matrix)

        # num_objects, num_timesteps
        sdc_collisions = collision_matrix[:, sdc_index].T
        valid = current_traj.valid
        sdc_collisions = sdc_collisions * valid

        is_gonna_collide = jnp.any(sdc_collisions, axis=1)
        first_collisions = jnp.argmax(sdc_collisions, axis=1)

        ttc = dt * first_collisions.astype(jnp.float32)
        ttc = jnp.where(is_gonna_collide, ttc, time_horizon)

        return ttc, valid

    def _only_ahead_agents(self, current_traj: datatypes.Trajectory, sdc_index: int):
        """Filter agents that are ahead of the ego vehicle.

        Args:
            current_traj: Trajectory data for all objects.
            sdc_index: Index of the ego vehicle.

        Returns:
            A boolean array indicating whether each agent is ahead.

        """
        ego_yaw = current_traj.yaw[sdc_index].squeeze()
        ego_length = current_traj.length[sdc_index].squeeze()
        agents_xy = current_traj.xy.squeeze()
        ego_direction = jnp.array([jnp.cos(ego_yaw), jnp.sin(ego_yaw)])
        ego_rear_axle = agents_xy[sdc_index] - (ego_length * 0.4) * ego_direction

        return utils.is_agent_ahead(ego_rear_axle, ego_yaw, agents_xy)
