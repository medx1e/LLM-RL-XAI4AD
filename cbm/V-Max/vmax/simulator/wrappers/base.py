# Copyright 2025 Valeo.


"""Base class for Waymax environment wrappers."""

import jax
from dm_env import specs
from waymax import datatypes
from waymax import env as waymax_env
from waymax.env import typedefs as types


class Wrapper(waymax_env.PlanningAgentEnvironment):
    """Base wrapper class for Waymax environment."""

    def __init__(self, env: waymax_env.PlanningAgentEnvironment) -> None:
        self.env = env

    def metrics(self, state: datatypes.SimulatorState) -> types.Metrics:
        return self.env.metrics(state)

    def reset(self, state: datatypes.SimulatorState, rng: jax.Array | None = None) -> datatypes.SimulatorState:
        return self.env.reset(state, rng)

    def observe(self, state: datatypes.SimulatorState) -> types.Observation:
        return self.env.observe(state)

    def step(self, state: datatypes.SimulatorState, action: datatypes.Action):
        return self.env.step(state, action)

    def reward(self, state: datatypes.SimulatorState, action: datatypes.Action) -> jax.Array:
        return self.env.reward(state, action)

    def termination(self, state: datatypes.SimulatorState) -> jax.Array:
        return self.env.termination(state)

    def truncation(self, state: datatypes.SimulatorState) -> jax.Array:
        return self.env.truncation(state)

    def action_spec(self) -> datatypes.Action:
        return self.env.action_spec()

    def reward_spec(self) -> specs.Array:
        return self.env.reward_spec()

    def discount_spec(self) -> specs.BoundedArray:
        return self.env.discount_spec()

    def observation_spec(self) -> int:
        return self.env.observation_spec()

    @property
    def dynamics_model(self):
        return self.get_wrapper_attr("dynamics").wrapped_dynamics

    def get_wrapper_attr(self, name: str):
        if hasattr(self, name):
            return getattr(self, name)
        elif hasattr(self.env, name):
            return getattr(self.env, name)
        else:
            try:
                return self.env.get_wrapper_attr(name)
            except AttributeError as e:
                raise AttributeError(
                    f"wrapper {self.class_name()} has no attribute {name!r}",
                ) from e

    def set_wrapper_attr(self, name: str, value):
        sub_env = self.env
        attr_set = False

        while attr_set is False and isinstance(sub_env, Wrapper):
            if hasattr(sub_env, name):
                setattr(sub_env, name, value)
                attr_set = True
            else:
                sub_env = sub_env.env

        if attr_set is False:
            setattr(sub_env, name, value)

    def __str__(self):
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        return str(self)

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__
