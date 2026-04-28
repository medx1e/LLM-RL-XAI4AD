# Copyright 2025 Valeo.

"""Wrappers for multi-agent operations in Brax-like Waymax environments."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
from waymax import datatypes
from waymax import env as waymax_env
from waymax.env import typedefs as types

from vmax.simulator.wrappers.base import Wrapper
from vmax.simulator.wrappers.interfaces.brax import EnvTransition


class MultiAgentBraxWrapper(Wrapper):
    """Brax-like wrapper for multi-agent environments."""

    def __init__(
        self,
        env: waymax_env.MultiAgentEnvironment,
        termination_keys: list[str] = ("offroad", "overlap", "run_red_light"),
    ) -> None:
        """Initialize the multi-agent wrapper.

        Args:
            env: The multi-agent environment to wrap.
            termination_keys: Metric keys used to determine termination.

        """
        super().__init__(env)

        self._termination_keys = termination_keys

    def termination(self, state: datatypes.SimulatorState) -> jax.Array:
        """Determine the termination condition using selected metrics.

        Args:
            state: Current simulator state.

        Returns:
            Termination flag.

        """
        metrics = self.env.metrics(state)

        termination = 0
        for key in self._termination_keys:
            termination += metrics[key].value

        return jnp.array(termination > 0, dtype=jnp.int32)

    def metrics(self, state: datatypes.SimulatorState) -> types.Metrics:
        """Extract metrics from the environment, including additional V-Max metrics.

        Args:
            state: Current simulator state.

        Returns:
            Dictionary of metric values.

        """
        metrics = self.env.metrics(state)
        metrics_dict = {key: metric.value for key, metric in metrics.items()}

        return metrics_dict

    def reset(self, state: datatypes.SimulatorState, rng: jax.Array | None = None) -> EnvTransition:
        """Reset the environment and initialize the simulation state.

        Args:
            state: An uninitialized state.
            rng: Random key for state initialization.

        Returns:
            An EnvTransition with initial state, observation, and metrics.

        """
        chex.assert_equal(state.shape, ())

        state = self.env.reset(state, rng)

        obs = self.observe(state)
        reward = jnp.zeros(state.shape + self.discount_spec().shape)
        flag = jnp.ones(state.shape + self.discount_spec().shape, dtype=jnp.bool_)
        done = jnp.zeros(state.shape + self.discount_spec().shape, dtype=jnp.bool_)
        metric_dict = self.metrics(state)

        info = {
            "steps": jnp.zeros(state.shape, dtype=jnp.int32),
            "rewards": jnp.zeros(state.shape, dtype=jnp.float32),
            "truncation": jnp.zeros(state.shape, dtype=jnp.bool_),
            "scenario_id": jnp.zeros(state.shape, dtype=jnp.int32),
        }

        return EnvTransition(
            state=state,
            observation=obs,
            reward=reward,
            done=done,
            flag=flag,
            metrics=metric_dict,
            info=info,
        )

    def step(
        self,
        env_transition: EnvTransition,
        action: datatypes.Action,
    ) -> EnvTransition:
        """Advance one simulation step.

        Args:
            env_transition: Current transition state.
            action: Action applied to the state.

        Returns:
            Updated EnvTransition after stepping.

        """
        chex.assert_equal(env_transition.state.shape, ())
        next_state = self.env.step(env_transition.state, action)

        obs = self.observe(next_state)
        reward = self.reward(next_state, action)

        # TODO - use sdc idx
        termination = jax.tree_util.tree_map(lambda x: x[0], self.termination(next_state))  # termination for SDC only
        truncation = self.truncation(next_state)
        done = jnp.logical_or(termination, truncation)
        flag = jnp.logical_not(termination)
        # TODO - get metrics for SDC only - stop_grad to prevent issues with log divergence
        # with valid obj not all the scenario
        metric_dict = jax.lax.stop_gradient(self.metrics(next_state))

        info = {
            "steps": env_transition.info["steps"] + 1,
            "rewards": env_transition.info["rewards"] + reward,
            "truncation": truncation,
            "scenario_id": env_transition.info["scenario_id"],
        }

        return EnvTransition(
            state=next_state,
            reward=reward,
            observation=obs,
            done=done,
            flag=flag,
            metrics=metric_dict,
            info=info,
        )
