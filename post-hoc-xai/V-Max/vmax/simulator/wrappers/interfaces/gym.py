# Copyright 2025 Valeo.

"""Wrapper for adapting a stateless Waymax environment to a Gym-like interface."""

import functools
from collections.abc import Iterator
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
from waymax import datatypes
from waymax.env import abstract_environment

from vmax.simulator.wrappers.base import Wrapper


class GymWrapper(Wrapper, gym.Env):
    """A stateful wrapper implementing the Gymnasium interface for the Waymax environment."""

    def __init__(
        self,
        stateless_env: abstract_environment.AbstractEnvironment,
        data_generator: Iterator[datatypes.SimulatorState],
        termination_keys: list[str] = ("offroad", "overlap", "run_red_light"),
    ) -> None:
        """Initialize the GymWrapper.

        Args:
            stateless_env: The underlying stateless environment.
            data_generator: Iterator that generates initial states.
            termination_keys: Keys used for computing termination conditions.

        """
        super().__init__(stateless_env)

        self._stateless_env = stateless_env
        self._data_generator = data_generator
        try:
            self._sample_state = next(self._data_generator)
        except StopIteration:
            raise ValueError("Data generator is empty. Cannot initialize environment.")

        self._cur_state = None
        self._initialized = False
        self._done = False
        self._jitted_stateless_env_reset = jax.jit(self._stateless_env.reset)
        self._jitted_stateless_env_step = jax.jit(self._stateless_env.step)
        self._termination_keys = termination_keys

        # Initialize spaces after we have a sample state
        self._setup_spaces()

    def _setup_spaces(self) -> None:
        """Set up the observation and action spaces based on the environment specs."""
        # Create proper observation space - this might need adjustment based on your observation structure
        obs_spec = self.stateless_env.observation_spec()
        if isinstance(obs_spec, dict):
            # For complex observation spaces with multiple components
            space_dict = {}
            for key, value in obs_spec.items():
                if hasattr(value, "shape") and hasattr(value, "dtype"):
                    space_dict[key] = gym.spaces.Box(
                        low=-float("inf"),
                        high=float("inf"),
                        shape=value.shape,
                        dtype=value.dtype,
                    )
            self._observation_space = gym.spaces.Dict(space_dict)
        else:
            # For simple flat observations
            self._observation_space = gym.spaces.Box(
                low=-float("inf"),
                high=float("inf"),
                shape=obs_spec.shape if hasattr(obs_spec, "shape") else (obs_spec,),
                dtype=jnp.float32,
            )

        # Create proper action space
        action_spec = self.stateless_env.action_spec().data
        self._action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=action_spec.shape,
            dtype=action_spec.dtype,
        )

    @property
    def simulation_state(self) -> datatypes.SimulatorState:
        """Return the current simulation state."""
        return self._cur_state

    @property
    def observation_space(self) -> gym.Space:
        """Return the observation space of the environment."""
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        """Return the action space of the environment."""
        return self._action_space

    @property
    def stateless_env(self) -> abstract_environment.AbstractEnvironment:
        """Return the underlying stateless environment."""
        return self._stateless_env

    @functools.partial(jax.jit, static_argnames=("self"))
    def observe(self, state: datatypes.SimulatorState) -> Any:
        """Convert a simulator state to an observation.

        Args:
            state: The current simulator state.

        Returns:
            Observation in the format expected by agents.
        """
        return self.stateless_env.observe(state)

    @functools.partial(jax.jit, static_argnames=("self"))
    def reward(self, state: datatypes.SimulatorState, action: datatypes.Action) -> jax.Array:
        """Calculate the reward for a given state and action.

        Args:
            state: Current simulator state.
            action: Action taken in the environment.

        Returns:
            Reward value.

        """
        return self.stateless_env.reward(state, action)

    @functools.partial(jax.jit, static_argnames=("self"))
    def termination(self, state: datatypes.SimulatorState) -> jax.Array:
        """Determine the termination condition using environment metrics.

        Args:
            state: Current simulator state.

        Returns:
            Termination flag array.

        """
        metrics = self.env.metrics(state)

        termination = 0
        for key in self._termination_keys:
            termination += metrics[key].value

        return jnp.array(termination > 0, dtype=jnp.int32)

    @functools.partial(jax.jit, static_argnames=("self"))
    def truncation(self, state: datatypes.SimulatorState) -> jax.Array:
        """Determine the truncation condition.

        Args:
            state: Current simulator state.

        Returns:
            Truncation flag array.

        """
        return self.stateless_env.truncation(state)

    @functools.partial(jax.jit, static_argnames=("self"))
    def metrics(self, state: datatypes.SimulatorState) -> dict[str, jax.Array]:
        """Extract metric values from the environment.

        Args:
            state: Current simulator state.

        Returns:
            Dictionary of metrics.

        """
        metrics = self.env.metrics(state)
        metrics_dict = {key: metric.value for key, metric in metrics.items()}

        return metrics_dict

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Any, dict]:
        """Reset the environment and return the initial observation.

        Args:
            seed: Optional seed for random number generation.
            options: Additional options for reset (not used currently).

        Returns:
            A tuple (observation, info) as per Gymnasium API.
        """
        self._cur_state = self._jitted_stateless_env_reset(next(self._data_generator))
        self._initialized = True
        self._done = False

        observation = self.observe(self._cur_state)
        info = {}

        return observation, info

    def step(self, action: jax.Array) -> tuple[Any, float, bool, bool, dict]:
        """Take a step in the environment.

        Args:
            action: Action to take in the environment.

        Returns:
            A tuple (observation, reward, terminated, truncated, info) as per Gymnasium API.
        """
        if not self._initialized:
            raise RuntimeError("Environment must be reset before calling step.")

        if self._done:
            raise RuntimeError("Environment is in terminal state. Call reset() before continuing.")

        action = jnp.asarray(action)
        # Reshape the action to what the underlying environment wants.
        action = jnp.reshape(action, self._cur_state.shape + self.stateless_env.action_spec().data.shape)
        action = datatypes.Action(data=action, valid=jnp.ones_like(action[..., 0:1], dtype=jnp.bool_))
        action.validate()
        self._cur_state = self._jitted_stateless_env_step(self._cur_state, action)

        next_obs = self.observe(self._cur_state)
        reward = self.reward(self._cur_state, action)
        termination = self.termination(self._cur_state)
        truncation = self.truncation(self._cur_state)
        done = jnp.logical_or(termination, truncation)

        # Calculate additional info
        info = self.metrics(self._cur_state)

        self._done = jnp.all(done)

        return next_obs, reward, termination, truncation, info
