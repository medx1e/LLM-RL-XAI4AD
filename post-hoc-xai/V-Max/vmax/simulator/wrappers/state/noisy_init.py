# Copyright 2025 Valeo.

"""Wrapper for adding noisy initial actions to the environment."""

import jax
import jax.numpy as jnp
from waymax import datatypes

from vmax.simulator import operations, wrappers


MAX_ITER_SAMPLE = 8
# Magic number to tune more? episode starts in a termination condition around 15% of time
PROB_MAKE_STRONG_NOISY_START = 0.75
FACTOR_NOISY_START = 0.9  # 90% of possible max value of accel and steering if strong noisy start

# We want to use much lower steering at high speed
BOUND_ACCEL_MAX = 0.6
BOUND_STEER_LOW_SPEED = 0.4
BOUND_STEED_HIGH_SPEED = 0.03
HIGH_SPEED = 40 / 3.6
LOW_SPEED = 0


class NoisyInitWrapper(wrappers.Wrapper):
    """Wrapper to apply noise to initial actions during reset."""

    def reset(self, state: datatypes.SimulatorState, rng: jax.Array | None = None) -> "wrappers.EnvTransition":
        """Reset the environment using noisy initial actions.

        Args:
            state: The initial state of the environment.
            rng: Random key for noise sampling.

        Returns:
            EnvTransition after applying noisy actions.

        """
        state = self.env.reset(state, rng)

        def is_noisy_state_corrupted(carry):
            _, noisy_state, _, nb_iter = carry
            return jnp.logical_and(noisy_state.done, nb_iter < MAX_ITER_SAMPLE)

        state, noisy_state, rng_key, nb_iter = self._sample_noisy_initial_action((state, state, rng, 0))

        state, noisy_state, _, _ = jax.lax.while_loop(
            is_noisy_state_corrupted,
            self._sample_noisy_initial_action,
            (state, noisy_state, rng_key, nb_iter),
        )

        return noisy_state

    def _sample_noisy_initial_action(self, carry) -> "wrappers.EnvTransition":
        """Sample a noisy action and update the transition.

        Args:
            carry: Tuple containing (state, current noisy state, random key, iteration count).

        Returns:
            Updated tuple with new state, noisy state, random key, and incremented iteration.

        """
        state, noisy_state, rng_key, nb_iter = carry

        sdc_idx = operations.get_index(state.state.object_metadata.is_sdc)
        sdc_speed = state.state.current_sim_trajectory.speed[sdc_idx][0]

        current_bound_steer = _compute_bound_steer(sdc_speed)

        arr_proba = jnp.array(
            [
                0.5 * PROB_MAKE_STRONG_NOISY_START,  # Case 1 strong and left
                0.5 * PROB_MAKE_STRONG_NOISY_START,  # Case 2 strong and right
                0.5 * (1 - PROB_MAKE_STRONG_NOISY_START),  # Case 3 normal and left
                0.5 * (1 - PROB_MAKE_STRONG_NOISY_START),  # Case 4 normal and right
            ],
        )
        arr_bounds = jnp.array(
            [
                [current_bound_steer * FACTOR_NOISY_START, current_bound_steer, BOUND_ACCEL_MAX * FACTOR_NOISY_START],
                [-current_bound_steer, -current_bound_steer * FACTOR_NOISY_START, BOUND_ACCEL_MAX * FACTOR_NOISY_START],
                [0, current_bound_steer, 0],
                [-current_bound_steer, 0, 0],
            ],
        )

        # Generate a random index based on the probability distribution
        rng_key, subkey_mode = jax.random.split(rng_key)
        current_bound_steering_accel = jax.random.choice(subkey_mode, arr_bounds, p=arr_proba)

        bound_steer_min = current_bound_steering_accel[0]
        bound_steer_max = current_bound_steering_accel[1]
        bound_accel_min = current_bound_steering_accel[2]

        rng_key, subkey_accel = jax.random.split(rng_key)
        rng_key, subkey_steer = jax.random.split(rng_key)

        # Sample accel and steer uniformly on the predefined interval
        action_accel = jax.random.uniform(key=subkey_accel, shape=(1,), minval=bound_accel_min, maxval=BOUND_ACCEL_MAX)
        action_steer = jax.random.uniform(key=subkey_steer, shape=(1,), minval=bound_steer_min, maxval=bound_steer_max)

        actions = jnp.concat([action_accel, action_steer])
        # actions = jnp.expand_dims(actions, axis=0) # TO ADD in EVAL
        actions = datatypes.Action(data=actions, valid=jnp.ones_like(actions[..., 0:1], dtype=jnp.bool_))
        actions.validate()

        history_length = 10  # TODO3 replace with historic length config? Suppose history = 10

        noisy_state = self.env.step(state, action=actions)

        for _ in range(history_length - 1):
            noisy_state = self.env.step(noisy_state, action=actions)

        return state, noisy_state, rng_key, nb_iter + 1


def _compute_bound_steer(sdc_speed: jax.Array) -> jax.Array:
    """Compute the steering bound based on the SDC's speed.

    Args:
        sdc_speed: Current speed of the SDC.

    Returns:
        Calculated steering bound.

    """
    current_bound_steer = (HIGH_SPEED - sdc_speed) / (HIGH_SPEED - LOW_SPEED) * (
        BOUND_STEER_LOW_SPEED - BOUND_STEED_HIGH_SPEED
    ) + BOUND_STEED_HIGH_SPEED

    return jnp.clip(current_bound_steer, BOUND_STEED_HIGH_SPEED, BOUND_STEER_LOW_SPEED)
