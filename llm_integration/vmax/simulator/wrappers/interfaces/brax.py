# Copyright 2025 Valeo.

"""Wrappers for Brax-like interface for the Waymax environment."""

from typing import Any

import chex
import jax
import jax.numpy as jnp
from flax import struct
from waymax import datatypes
from waymax import env as waymax_env
from waymax.datatypes import operations
from waymax.env import typedefs as types

from vmax.simulator.wrappers.base import Wrapper


@chex.dataclass(frozen=True)
class EnvTransition:
    """Container class for Waymax transitions.

    Attributes:
      state: The current simulation state of shape (...).
      observation: The current observation of shape (..,).
      reward: The reward obtained in the current transition of shape (...,
        num_objects).
      done: A boolean array denoting the end of an episode of shape (...).
      flag: An array of discount values of shape (...).
      metrics: Optional dictionary of metrics.
      info: Optional dictionary of arbitrary logging information.

    """

    state: datatypes.SimulatorState
    observation: types.Observation
    reward: jax.Array
    done: jax.Array
    flag: jax.Array
    metrics: dict[str, Any] = struct.field(default_factory=dict)
    info: dict[str, Any] = struct.field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of EnvTransition."""
        return self.state.shape

    def __eq__(self, other: object) -> bool:
        return datatypes.compare_all_leaf_nodes(self, other)


class BraxWrapper(Wrapper):
    """Brax-like interface wrapper for the Waymax environment."""

    def __init__(
        self,
        env: waymax_env.PlanningAgentEnvironment,
        termination_keys: list[str] = ("offroad", "overlap"),
    ) -> None:
        """Initialize the Brax-like interface wrapper for the Waymax environment.

        Args:
            env: The environment to wrap.
            termination_keys: The keys of the metrics that define the termination condition of the environment.

        """
        super().__init__(env)

        self._termination_keys = termination_keys

    def termination(self, state: datatypes.SimulatorState) -> jax.Array:
        """Return the termination condition of the environment.

        The termination conditions are defined by selected metrics.

        Args:
            state: The current state of the environment.

        Returns:
            The termination condition of the environment.

        """
        metrics = self.env.metrics(state)

        termination = 0
        for key in self._termination_keys:
            termination += metrics[key].value

        return jnp.array(termination > 0, dtype=jnp.int32)

    def metrics(self, state: datatypes.SimulatorState) -> types.Metrics:
        """Return the metrics of the environment.

        Adding vmax metrics to the metrics of the environment.

        Args:
            state: The current state of the environment.

        Returns:
            The metrics of the environment.

        """
        metrics = self.env.metrics(state)
        metrics_dict = {key: metric.value for key, metric in metrics.items()}

        return metrics_dict

    def reset(self, state: datatypes.SimulatorState, rng: jax.Array | None = None) -> EnvTransition:
        """Reset the environment and initializes the simulation state.

        This initializer sets the initial env_transition and fills the initial simulation
        trajectory with invalid values.

        Args:
            state: An uninitialized state.
            rng: Random key for initializing the state.

        Returns:
            The initialized simulation state.

        """
        chex.assert_equal(state.shape, ())

        state = self.env.reset(state, rng)

        obs = self.observe(state)
        reward = jnp.zeros(state.shape + self.discount_spec().shape, dtype=jnp.float32)
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

    def step(self, env_transition: EnvTransition, action: datatypes.Action) -> EnvTransition:
        """Advance simulation by one timestep using the dynamics model.

        Args:
            env_transition: The env_transition containing the current state.
            action: The action to apply to the current state.

        Returns:
            The next_env_transition corresponding to the transition taken.

        """
        chex.assert_equal(env_transition.state.shape, ())

        next_state = self.env.step(env_transition.state, action)

        obs = self.observe(next_state)
        reward = self.reward(next_state, action)

        termination = self.termination(next_state)
        truncation = self.truncation(next_state)
        done = jnp.logical_or(termination, truncation)
        flag = jnp.logical_not(termination)
        metric_dict = self.metrics(next_state)

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


class VmapWrapper(Wrapper):
    """Vectorizes Brax env."""

    def reset(self, state: datatypes.SimulatorState, rng: jax.Array | None = None) -> EnvTransition:
        """Vectorized reset function for the environment.

        Args:
            state: The state to reset the environment to.
            rng: Random key for initializing the state.

        Returns:
            The next environment transition.

        """
        return jax.vmap(self.env.reset)(state, rng)

    def step(self, env_transition: EnvTransition, action: jax.Array) -> EnvTransition:
        """Vectorized step function for the environment.

        Args:
            env_transition: The current environment transition.
            action: The action to apply to the environment.

        Returns:
            The next environment transition.

        """
        return jax.vmap(self.env.step)(env_transition, action)


class AutoResetWrapper(Wrapper):
    """Automatically resets Brax envs that are done."""

    def __init__(self, env: waymax_env.PlanningAgentEnvironment) -> None:
        """Initialize the AutoResetWrapper.

        Args:
            env: The environment to wrap.

        """
        super().__init__(env)
        self._scenario_buffer = None
        self._total_scenarios = None

    def step(self, env_transition: EnvTransition, action: jax.Array) -> EnvTransition:
        """Apply the step function to the environment.

        If the environment is done, pull the next scenario from the scenario buffer, reset the environment,
        and replace the state, observation, and info in the transition with the new scenario.

        Args:
            env_transition: The current environment transition.
            action: The action to apply to the environment.

        Returns:
            The next environment transition.

        """
        env_transition.info["rewards"] = env_transition.info["rewards"] * (1 - env_transition.done)
        env_transition.info["steps"] = env_transition.info["steps"] * (1 - env_transition.done)

        next_env_transition = self.env.step(env_transition, action)

        def where_done(x, y):
            done = next_env_transition.done
            if done.shape:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))

            return jnp.where(done, x, y)

        scenario_id = self._increment_scenario_id(next_env_transition, self._total_scenarios)

        updated_env_transition = jax.tree.map(where_done, self._pull_next_scenario(scenario_id), next_env_transition)
        next_env_transition.info["scenario_id"] = scenario_id

        return next_env_transition.replace(
            state=updated_env_transition.state,
            observation=updated_env_transition.observation,
        )

    def init_and_reset(self, scenario_buffer, rng: jax.Array | None = None):
        """Initialize the scenario buffer and reset the environment.

        Args:
            scenario_buffer: The new scenario buffer.
            rng: Random key for initializing the state.

        Returns:
            The updated scenario buffer.

        """
        self._scenario_buffer = jax.vmap(self.env.reset)(scenario_buffer, rng)

        batch_dims = scenario_buffer.batch_dims
        if len(batch_dims) == 1:
            self._total_scenarios = batch_dims[0]
            self._current_scenario_id = 0
        elif len(batch_dims) == 2:
            self._total_scenarios = batch_dims[1]
            self._current_scenario_id = jnp.zeros((scenario_buffer.batch_dims[0],), dtype=jnp.int32)

        return self._pull_next_scenario(self._current_scenario_id)

    def _pull_next_scenario(self, scenario_id):
        return operations.dynamic_index(self._scenario_buffer, scenario_id, keepdims=False)

    def _increment_scenario_id(self, env_transition: EnvTransition, total_scenarios: int):
        return (env_transition.info["scenario_id"] + env_transition.done) % total_scenarios
