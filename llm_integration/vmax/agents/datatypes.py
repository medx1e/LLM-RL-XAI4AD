# Copyright 2025 Valeo.

"""Datatypes for the learning module."""

from collections.abc import Callable, Mapping
from typing import Any, NamedTuple, Protocol

import flax
import jax


Params = Any
Metrics = Mapping[str, jax.Array]
ActivationFn = Callable[[jax.Array], jax.Array]
Initializer = Callable[..., Any]


class RLPartialTransition(NamedTuple):
    """Container for a partial RL transition (without next_observation)."""

    observation: jax.Array
    action: jax.Array
    reward: float
    flag: int
    done: int
    extras: tuple = ()


class RLTransition(NamedTuple):
    """Container for a complete RL transition."""

    observation: jax.Array
    action: jax.Array
    reward: float
    flag: int
    next_observation: jax.Array
    done: int
    extras: tuple = ()


@flax.struct.dataclass
class TrainingState:
    """Container for the training state."""

    params: Params
    env_steps: int


class Policy(Protocol):
    """Protocol for a policy function."""

    def __call__(self, observation: jax.Array, key: jax.Array) -> jax.Array:
        pass


class LearningFunction(Protocol):
    """Protocol for a learning function."""

    def __call__(
        self,
        carry: tuple[TrainingState, jax.Array],
        transitions: RLTransition,
    ) -> tuple[tuple[TrainingState, jax.Array], Metrics]:
        pass
