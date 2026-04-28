# Copyright 2025 Valeo.

"""Acting functions for the learning pipeline."""

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from waymax import datatypes as waymax_datatypes
from waymax.agents import expert

from vmax.agents import datatypes
from vmax.agents.rule_based import idm, pdm
from vmax.simulator import operations, wrappers


def _create_valid_action(action_data: jnp.ndarray) -> waymax_datatypes.Action:
    """Helper function to create a valid waymax Action from raw action data.

    Args:
        action_data: Raw action data.

    Returns:
        A valid waymax Action object.
    """
    valid_mask = jnp.ones_like(action_data[..., 0:1], dtype=jnp.bool_)
    action = waymax_datatypes.Action(data=action_data, valid=valid_mask)
    action.validate()
    return action


def generate_unroll(
    env_transition: wrappers.EnvTransition,
    policy_fn: Callable,
    key: jax.Array,
    unroll_length: int,
    env: wrappers.BraxWrapper,
    step_fn: Callable,
) -> tuple[wrappers.EnvTransition, dict[str, Any], datatypes.RLTransition]:
    """Generate a sequence of environment transitions using the given policy over a specified unroll length.

    Args:
        env_transition: Initial environment transition.
        policy_fn: Policy function to generate actions.
        key: Random key for generating actions.
        unroll_length: Length of the unroll.
        env: Brax environment wrapper.
        step_fn: Function to execute a single step.

    Returns:
        Tuple containing the final environment transition, metrics dictionary, and the generated RL transition.
    """

    def f(carry, _t):
        curr_state, current_key = carry
        current_key, next_key = jax.random.split(current_key)
        next_state, transition = step_fn(curr_state, env, policy_fn, current_key)

        metrics = {
            "ep_rew_mean": next_state.info["rewards"],
            "ep_len_mean": next_state.info["steps"],
            **next_state.metrics,
        }

        return (next_state, next_key), (metrics, transition)

    (env_transition, _), (metrics, data) = jax.lax.scan(
        f,
        (env_transition, key),
        (),
        length=unroll_length,
    )

    return env_transition, metrics, data


def policy_step(
    env_transition: wrappers.EnvTransition,
    env: wrappers.BraxWrapper,
    policy_fn: Callable,
    key: jax.Array | None = None,
    extra_fields: Sequence[str] = (),
    use_partial_transition: bool = False,
) -> tuple[wrappers.EnvTransition, datatypes.RLTransition | datatypes.RLPartialTransition]:
    """Perform a policy step by selecting an action and applying it to the environment.

    Args:
        env_transition: Initial environment transition.
        env: Brax environment wrapper.
        policy_fn: Policy function to generate actions.
        key: Random key for generating actions.
        extra_fields: Additional fields to include in the RL transition.
        use_partial_transition: Whether to use a partial transition.

    Returns:
        Tuple containing the next environment transition and the generated RL transition.
    """
    actions, policy_extras = policy_fn(env_transition.observation, key)
    actions = _create_valid_action(actions)

    # Apply the actions to the environment and get the resulting slice of the episode
    next_env_transition = env.step(env_transition, actions)

    state_extras = {field: env_transition.info[field] for field in extra_fields}

    if use_partial_transition:
        rl_transition = datatypes.RLPartialTransition(
            observation=env_transition.observation,
            action=actions.data,
            reward=next_env_transition.reward,
            flag=next_env_transition.flag,
            done=next_env_transition.done,
            extras={"policy_extras": policy_extras, "state_extras": state_extras},
        )
    else:
        rl_transition = datatypes.RLTransition(
            observation=env_transition.observation,
            action=actions.data,
            reward=next_env_transition.reward,
            flag=next_env_transition.flag,
            next_observation=next_env_transition.observation,
            done=next_env_transition.done,
            extras={"policy_extras": policy_extras, "state_extras": state_extras},
        )

    return next_env_transition, rl_transition


def random_step(
    env_transition: wrappers.EnvTransition,
    env: wrappers.BraxWrapper,
    policy_fn: Callable | None = None,
    key: jax.Array | None = None,
    action_shape: tuple = (2,),
    action_bounds: tuple[float, float] = (-1.0, 1.0),
) -> tuple[wrappers.EnvTransition, datatypes.RLPartialTransition]:
    """Execute a random action step within the specified bounds.

    Args:
        env_transition: Initial environment transition.
        env: Brax environment wrapper.
        policy_fn: Not used, included for API consistency.
        key: Random key for generating actions. Must be provided.
        action_shape: Shape of the action to generate.
        action_bounds: Bounds for the action values.

    Returns:
        Tuple containing the next environment transition and the generated RL transition.
    """
    if key is None:
        raise ValueError("Random key must be provided for random_step")

    # Generating actions within the specified bounds
    actions = jax.random.uniform(key=key, shape=action_shape, minval=action_bounds[0], maxval=action_bounds[1])
    actions = _create_valid_action(actions)

    next_env_transition = env.step(env_transition, actions)

    rl_transition = datatypes.RLPartialTransition(
        observation=env_transition.observation,
        action=actions.data,
        reward=next_env_transition.reward,
        flag=next_env_transition.flag,
        done=next_env_transition.done,
    )

    return next_env_transition, rl_transition


def expert_step(
    env_transition: wrappers.EnvTransition,
    env: wrappers.BraxWrapper,
    policy_fn: Callable | None = None,
    key: jax.Array | None = None,
    use_partial_transition: bool = False,
) -> tuple[wrappers.EnvTransition, datatypes.RLTransition | datatypes.RLPartialTransition]:
    """Execute a step using an expert policy based on the SDC's logged trajectory.

    Args:
        env_transition: Initial environment transition.
        env: Brax environment wrapper.
        policy_fn: Not used, included for API consistency.
        key: Not used, included for API consistency.
        use_partial_transition: Whether to use a partial transition.

    Returns:
        Tuple containing the next environment transition and the generated RL transition.
    """
    # (num_envs, num_agents, 2)
    actions = expert.infer_expert_action(env_transition.state, env.dynamics_model).data
    # (num_envs,)
    sdc_idx = operations.get_index(env_transition.state.object_metadata.is_sdc)
    # (1, 1, 1)
    sdc_idx = jnp.expand_dims(sdc_idx, axis=(0, 1, 2))
    # (num_envs, 1, 2)
    action_sdc = jnp.take_along_axis(actions, sdc_idx, axis=-2)
    # (num_envs, 2)
    action_sdc = jnp.squeeze(action_sdc, axis=-2)

    action_sdc = _create_valid_action(action_sdc)
    next_env_transition = env.step(env_transition, action_sdc)

    if use_partial_transition:
        rl_transition = datatypes.RLPartialTransition(
            observation=env_transition.observation,
            action=action_sdc.data,
            reward=next_env_transition.reward,
            flag=next_env_transition.flag,
            done=next_env_transition.done,
        )
    else:
        rl_transition = datatypes.RLTransition(
            observation=env_transition.observation,
            action=action_sdc.data,
            reward=next_env_transition.reward,
            flag=next_env_transition.flag,
            next_observation=next_env_transition.observation,
            done=next_env_transition.done,
        )

    return next_env_transition, rl_transition


def idm_step(
    env_transition: wrappers.EnvTransition,
    env: wrappers.BraxWrapper,
    policy_fn: Callable | None = None,
    key: jax.Array | None = None,
) -> tuple[wrappers.EnvTransition, datatypes.RLTransition]:
    """Execute a single step using the IDM policy.

    Args:
        env_transition: Initial environment transition.
        env: Brax environment wrapper.
        policy_fn: Not used, included for API consistency.
        key: Not used, included for API consistency.

    Returns:
        Tuple containing the next environment transition and the generated RL transition.
    """
    observation = jax.tree_map(lambda x: x[0], env_transition.observation)
    action_sdc = idm.select_action(observation)
    action_sdc = jnp.expand_dims(action_sdc, axis=0)

    action_sdc = _create_valid_action(action_sdc)
    next_env_transition = env.step(env_transition, action_sdc)

    rl_transition = datatypes.RLTransition(
        observation=env_transition.observation,
        action=action_sdc.data,
        reward=next_env_transition.reward,
        flag=next_env_transition.flag,
        next_observation=next_env_transition.observation,
        done=next_env_transition.done,
    )

    return next_env_transition, rl_transition


def pdm_step(
    env_transition: wrappers.EnvTransition,
    env: wrappers.BraxWrapper,
    policy_fn: Callable | None = None,
    key: jax.Array | None = None,
) -> tuple[wrappers.EnvTransition, datatypes.RLTransition]:
    """Execute a single step using the PDM policy.

    Args:
        env_transition: Initial environment transition.
        env: Brax environment wrapper.
        policy_fn: Not used, included for API consistency.
        key: Not used, included for API consistency.

    Returns:
        Tuple containing the next environment transition and the generated RL transition.
    """
    observation = jax.tree_map(lambda x: x[0], env_transition.observation)

    action_sdc = pdm.select_action(observation)
    action_sdc = jnp.expand_dims(action_sdc, axis=0)
    action_sdc = _create_valid_action(action_sdc)

    next_env_transition = env.step(env_transition, action_sdc)

    rl_transition = datatypes.RLTransition(
        observation=env_transition.observation,
        action=action_sdc.data,
        reward=next_env_transition.reward,
        flag=next_env_transition.flag,
        next_observation=next_env_transition.observation,
        done=next_env_transition.done,
    )

    return next_env_transition, rl_transition


def constant_step(
    env_transition: wrappers.EnvTransition,
    env: wrappers.BraxWrapper,
    policy_fn: Callable | None = None,
    key: jax.Array | None = None,
) -> tuple[wrappers.EnvTransition, datatypes.RLTransition]:
    """Execute a constant step that applies zero actions to the environment.

    Args:
        env_transition: Initial environment transition.
        env: Brax environment wrapper.
        policy_fn: Not used, included for API consistency.
        key: Not used, included for API consistency.

    Returns:
        Tuple containing the next environment transition and the generated RL transition.
    """
    # Generating actions within the specified bounds
    actions = jnp.array((0.0, 0.0))
    actions = jnp.expand_dims(actions, axis=0)
    actions = _create_valid_action(actions)

    next_env_transition = env.step(env_transition, actions)

    rl_transition = datatypes.RLTransition(
        observation=env_transition.observation,
        action=actions.data,
        reward=next_env_transition.reward,
        flag=next_env_transition.flag,
        next_observation=next_env_transition.observation,
        done=next_env_transition.done,
    )

    return next_env_transition, rl_transition
