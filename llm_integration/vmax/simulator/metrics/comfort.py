# Copyright 2025 Valeo.

import jax
import jax.numpy as jnp
from waymax import datatypes
from waymax.metrics import abstract_metric

from vmax.simulator import constants, operations
from vmax.simulator.metrics import utils


class ComfortMetric(abstract_metric.AbstractMetric):
    """Comfort metric under nuPlan standards."""

    def compute(self, simulator_state: datatypes.SimulatorState) -> abstract_metric.MetricResult:
        """Compute the comfort metric for the simulation.

        Args:
            simulator_state: The current simulator state.

        Returns:
            A metric result with the computed comfort measure.

        """
        past_traj = datatypes.dynamic_slice(
            simulator_state.sim_trajectory,
            simulator_state.timestep - 9,
            10,
            -1,
        )
        sdc_index = operations.get_index(simulator_state.object_metadata.is_sdc)

        sdc_traj = jax.tree_util.tree_map(lambda x: x[sdc_index], past_traj)

        lateral_accel = _compute_lateral_acceleration(sdc_traj, constants.TIME_DELTA)
        long_accel = _compute_longitudinal_acceleration(sdc_traj, constants.TIME_DELTA)
        long_jerk = _compute_longitudinal_jerk(sdc_traj, constants.TIME_DELTA)
        yaw_rate = _compute_ego_yaw_rate(sdc_traj, constants.TIME_DELTA)
        yaw_accel = _compute_ego_yaw_acceleration(sdc_traj, constants.TIME_DELTA)

        value = (jnp.max(jnp.abs(lateral_accel)) <= 2.0).astype(jnp.float32)
        value *= jnp.min(long_accel) >= -4.05
        value *= jnp.max(long_accel) <= 2.40
        value *= jnp.max(jnp.abs(long_jerk)) <= 8.3
        value *= jnp.max(jnp.abs(yaw_accel)) <= 2.2
        value *= jnp.max(jnp.abs(yaw_rate)) <= 0.95

        valid = jnp.all(sdc_traj.valid)

        return abstract_metric.MetricResult.create_and_validate(value, valid)

    def compute_reward(self, simulator_state: datatypes.SimulatorState) -> abstract_metric.MetricResult:
        """Compute the comfort metric for the simulation.

        Args:
            simulator_state: The current simulator state.

        Returns:
            A metric result with the computed comfort measure.

        """
        past_traj = datatypes.dynamic_slice(
            simulator_state.sim_trajectory,
            simulator_state.timestep - 9,
            10,
            -1,
        )
        sdc_index = operations.get_index(simulator_state.object_metadata.is_sdc)

        sdc_traj = jax.tree_util.tree_map(lambda x: x[sdc_index], past_traj)

        lateral_accel = _compute_lateral_acceleration(sdc_traj, constants.TIME_DELTA)
        long_accel = _compute_longitudinal_acceleration(sdc_traj, constants.TIME_DELTA)
        long_jerk = _compute_longitudinal_jerk(sdc_traj, constants.TIME_DELTA)
        yaw_rate = _compute_ego_yaw_rate(sdc_traj, constants.TIME_DELTA)
        yaw_accel = _compute_ego_yaw_acceleration(sdc_traj, constants.TIME_DELTA)

        value_lateral_accel = jnp.exp(-(jnp.maximum(0, jnp.max(jnp.abs(lateral_accel)) - 2.0)))
        value_long_accel = jnp.exp(-(jnp.maximum(0, -jnp.min(long_accel) - 4.05))) * jnp.exp(
            -(jnp.maximum(0, jnp.max(long_accel) - 2.40)),
        )
        value_long_jerk = jnp.exp(-(jnp.maximum(0, jnp.max(jnp.abs(long_jerk)) - 8.3)))
        value_yaw_accel = jnp.exp(-(jnp.maximum(0, jnp.max(jnp.abs(yaw_accel)) - 2.2)))
        value_yaw_rate = jnp.exp(-(jnp.maximum(0, jnp.max(jnp.abs(yaw_rate)) - 0.95)))

        values = jnp.stack([value_lateral_accel, value_long_accel, value_long_jerk, value_yaw_accel, value_yaw_rate])
        # Compute weighted average so close to 0 values affect the result more
        weights = jnp.exp(-values)

        value = jnp.average(values, weights=weights)
        valid = jnp.all(sdc_traj.valid)

        return abstract_metric.MetricResult.create_and_validate(value, valid)


def _compute_lateral_acceleration(sdc_traj: datatypes.Trajectory, dt: float):
    """Compute lateral acceleration using the vehicle trajectory.

    Args:
        sdc_traj: The ego vehicle trajectory.
        dt: The time increment between trajectory steps.

    Returns:
        The calculated lateral acceleration.

    """
    # Extract lateral orientation
    yaw = sdc_traj.yaw
    vel = sdc_traj.stack_fields(["vel_x", "vel_y"])

    lateral_direction = jnp.stack([-jnp.sin(yaw), jnp.cos(yaw)], axis=-1)
    lateral_velocity = jnp.sum(vel * lateral_direction, axis=-1)

    lateral_acceleration = utils.savgol_filter_jax(
        lateral_velocity,
        window_length=5,
        polyorder=2,
        deriv=1,
        delta=dt,
    )

    return lateral_acceleration


def _compute_longitudinal_acceleration(sdc_traj: datatypes.Trajectory, dt: float):
    """Compute the longitudinal acceleration along the vehicle trajectory.

    Args:
        sdc_traj: The ego vehicle trajectory.
        dt: The time increment between steps.

    Returns:
        The computed longitudinal acceleration.

    """
    # Extract longitudinal orientation
    yaw = sdc_traj.yaw
    vel = sdc_traj.stack_fields(["vel_x", "vel_y"])

    longitudinal_direction = jnp.stack([jnp.cos(yaw), jnp.sin(yaw)], axis=-1)
    longitudinal_velocity = jnp.sum(vel * longitudinal_direction, axis=-1)

    longitudinal_acceleration = utils.savgol_filter_jax(
        longitudinal_velocity,
        window_length=5,
        polyorder=2,
        deriv=1,
        delta=dt,
    )

    return longitudinal_acceleration


def _compute_longitudinal_jerk(sdc_traj: datatypes.Trajectory, dt: float):
    """Compute the longitudinal jerk over the vehicle trajectory.

    Args:
        sdc_traj: The ego vehicle trajectory.
        dt: The time increment between steps.

    Returns:
        The calculated longitudinal jerk.

    """
    yaw = sdc_traj.yaw
    vel = sdc_traj.stack_fields(["vel_x", "vel_y"])

    longitudinal_direction = jnp.stack([jnp.cos(yaw), jnp.sin(yaw)], axis=-1)
    longitudinal_velocity = jnp.sum(vel * longitudinal_direction, axis=-1)

    longitudinal_jerk = utils.savgol_filter_jax(
        longitudinal_velocity,
        window_length=5,
        polyorder=3,
        deriv=2,
        delta=dt,
    )

    return longitudinal_jerk


def _compute_ego_yaw_rate(sdc_traj: datatypes.Trajectory, dt: float):
    """Compute the yaw rate of the ego vehicle.

    Args:
        sdc_traj: The ego vehicle trajectory.
        dt: The time difference between observations.

    Returns:
        The ego vehicle yaw rate.

    """
    yaw = sdc_traj.yaw
    yaw = phase_unwrap(yaw)
    yaw_rate = utils.savgol_filter_jax(yaw, window_length=5, polyorder=2, deriv=1, delta=dt)
    return yaw_rate


def _compute_ego_yaw_acceleration(sdc_traj: datatypes.Trajectory, dt: float):
    """Compute the yaw acceleration of the ego vehicle.

    Args:
        sdc_traj: The ego vehicle trajectory.
        dt: The time increment between steps.

    Returns:
        The calculated yaw acceleration.

    """
    yaw = sdc_traj.yaw
    yaw = phase_unwrap(yaw)
    yaw_accel = utils.savgol_filter_jax(yaw, window_length=5, polyorder=3, deriv=2, delta=dt)
    return yaw_accel


def phase_unwrap(headings):
    """Unwrap the heading angles to avoid discontinuities.

    Args:
        headings: A sequence of heading angles.

    Returns:
        The unwrapped heading angles.

    """
    # There are some jumps in the heading (e.g. from -np.pi to +np.pi) which causes approximation
    # of yaw to be very large.
    # We want unwrapped[j] = headings[j] - 2*pi*adjustments[j] for some integer-valued adjustments
    # making the absolute value of
    # unwrapped[j+1] - unwrapped[j] at most pi:
    # -pi <= headings[j+1] - headings[j] - 2*pi*(adjustments[j+1] - adjustments[j]) <= pi
    # -1/2 <= (headings[j+1] - headings[j])/(2*pi) - (adjustments[j+1] - adjustments[j]) <= 1/2
    # So adjustments[j+1] - adjustments[j] = round((headings[j+1] - headings[j]) / (2*pi)).

    two_pi = 2.0 * jnp.pi
    adjustments = jnp.concatenate([jnp.zeros(1, dtype=jnp.float32), jnp.cumsum(jnp.round(jnp.diff(headings) / two_pi))])
    unwrapped = headings - two_pi * adjustments
    return unwrapped
