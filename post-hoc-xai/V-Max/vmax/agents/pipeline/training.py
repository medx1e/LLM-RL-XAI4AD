# Copyright 2025 Valeo.

"""Pipeline for training and evaluation of agents."""

from __future__ import annotations

import functools
import typing

import jax
import jax.numpy as jnp

from vmax.agents import pipeline


if typing.TYPE_CHECKING:
    from waymax import datatypes as waymax_datatypes
    from waymax import env as waymax_env

    from vmax.agents import datatypes
    from vmax.agents.learning import replay_buffer


def run_training_off_policy(
    batch_scenarios: waymax_datatypes.SimulatorState,
    training_state: datatypes.TrainingState,
    buffer_state: replay_buffer.ReplayBufferState,
    key: jax.Array,
    replay_buffer: replay_buffer.ReplayBuffer,
    env: waymax_env.PlanningAgentEnvironment,
    learning_fn: datatypes.LearningFunction,
    policy_fn: datatypes.Policy,
    unroll_fn: pipeline.generate_unroll,
    scan_length: int,
    grad_updates_per_step: int,
):
    """Run the off-policy training loop.

    The training loop performs the following steps:
        - 1. Generate an unroll of transitions using the current policy.
        - 2. Insert the transitions into the replay buffer.
        - 3. Sample transitions from the replay buffer.
        - 4. Perform multiple gradient updates using the sampled transitions.

    Args:
        batch_scenarios: The batch of scenarios.
        training_state: The current training state.
        buffer_state: The replay buffer state.
        key: The random key.
        replay_buffer: The replay buffer instance.
        env: The environment.
        learning_fn: The function applying gradient updates.
        policy_fn: The policy function.
        unroll_fn: The function to generate an unroll.
        scan_length: The length of the unroll.
        grad_updates_per_step: Number of gradient updates per step.

    Returns:
        A tuple of (updated training_state, updated buffer_state, metrics).

    """

    def run_step(carry, _t):
        training_state, env_state, buffer_state, _key = carry
        _key, unroll_key, training_key = jax.random.split(_key, 3)

        policy = policy_fn(training_state.params.policy)
        next_state, rollout_metrics, data = unroll_fn(env_state, policy, unroll_key)

        training_state = training_state.replace(
            env_steps=training_state.env_steps + jnp.prod(jnp.array(data.done.shape)),
        )

        # (unroll_length, num_envs, ...) -> (num_envs * unroll_length, ...)
        data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data)
        buffer_state = replay_buffer.insert(buffer_state, data)

        buffer_state, transitions = replay_buffer.sample(buffer_state)
        # (batch_size * grad_updates_per_step, ...) -> (grad_updates_per_step, batch_size, ...)
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (grad_updates_per_step, -1) + x.shape[1:]),
            transitions,
        )

        (training_state, _), sgd_metrics = jax.lax.scan(
            learning_fn,
            (training_state, training_key),
            transitions,
        )

        return (training_state, next_state, buffer_state, _key), (rollout_metrics, sgd_metrics)

    key, subkey = jax.random.split(key, 2)
    reset_keys = jax.random.split(subkey, batch_scenarios.shape)

    env_state = env.init_and_reset(batch_scenarios, reset_keys)

    (training_state, _, buffer_state, _), (rollout_metrics, sgd_metrics) = jax.lax.scan(
        run_step,
        (training_state, env_state, buffer_state, key),
        (),
        length=scan_length,
    )

    metrics = _reshape_metrics(rollout_metrics, sgd_metrics)

    return training_state, buffer_state, metrics


def run_training_on_policy(
    batch_scenarios: waymax_datatypes.SimulatorState,
    training_state: datatypes.TrainingState,
    key: jax.Array,
    env: waymax_env.PlanningAgentEnvironment,
    learning_fn: datatypes.LearningFunction,
    policy_fn: datatypes.Policy,
    unroll_fn: pipeline.generate_unroll,
    scan_length: int,
    grad_updates_per_step: int,
):
    """Run the on-policy training loop.

    The training loop performs the following steps:
        - 1. Generate an unroll of transitions using the current policy.
        - 2. Perform multiple gradient updates using the generated transitions.

    Args:
        batch_scenarios: The batch of scenarios.
        training_state: The current training state.
        key: The random key.
        env: The environment.
        learning_fn: The function applying gradient updates.
        policy_fn: The policy function.
        unroll_fn: The function to generate an unroll.
        scan_length: The length of the unroll.
        grad_updates_per_step: Number of gradient updates per step.

    Returns:
        A tuple of (updated training_state, metrics).

    """

    def run_step(carry, _t):
        training_state, env_state, key = carry
        key, unroll_key = jax.random.split(key)

        policy = policy_fn(training_state.params.policy)
        next_state, rollout_metrics, data = unroll_fn(env_state, policy, unroll_key)

        training_state = training_state.replace(
            env_steps=training_state.env_steps + jnp.prod(jnp.array(data.done.shape)),
        )

        return (training_state, next_state, key), (data, rollout_metrics)

    key, subkey = jax.random.split(key, 2)
    reset_keys = jax.random.split(subkey, batch_scenarios.shape)

    env_state = env.init_and_reset(batch_scenarios, reset_keys)

    unroll_key, training_key = jax.random.split(key, 2)

    (training_state, _, _), (data, rollout_metrics) = jax.lax.scan(
        run_step,
        (training_state, env_state, unroll_key),
        (),
        length=scan_length,
    )

    # (batch_size, unroll_length, num_envs, ...) -> (batch_size, num_envs, unroll_length, ...)
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
    # (batch_size, num_envs, unroll_length, ...)
    data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data)

    (training_state, _), sgd_metrics = jax.lax.scan(
        functools.partial(learning_fn, transitions=data),
        (training_state, training_key),
        length=grad_updates_per_step,
    )

    metrics = _reshape_metrics(rollout_metrics, sgd_metrics)

    return training_state, metrics


def _reshape_metrics(rollout_metrics: dict, sgd_metrics: dict) -> dict:
    """Reshape and combine rollout and learning metrics.

    Args:
        rollout_metrics: Metrics from the rollout phase.
        sgd_metrics: Metrics from the gradient update phase.

    Returns:
        A dictionary of combined and reshaped metrics.

    """
    # (num_episodes, unroll_length, num_envs) -> (num_steps, num_envs)
    rollout_metrics = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), rollout_metrics)
    # (num_steps, num_envs) -> (num_envs, num_steps)
    rollout_metrics = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), rollout_metrics)
    # (scan_length, grad_updates_per_step)
    sgd_metrics = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), sgd_metrics)

    return {
        **{f"{name}": value for name, value in rollout_metrics.items()},
        **{f"train/{name}": value for name, value in sgd_metrics.items()},
    }


def run_evaluation(
    batch_scenarios: waymax_datatypes.SimulatorState,
    training_state: datatypes.TrainingState,
    env: waymax_env.PlanningAgentEnvironment,
    policy_fn: datatypes.Policy,
    step_fn: typing.Callable,
    scan_length: int,
):
    """Run the evaluation loop.

    This function evaluates the agent by running environment steps and collecting metrics.

    Args:
        batch_scenarios: The batch of scenarios.
        training_state: The current training state.
        env: The environment.
        policy_fn: The policy function.
        step_fn: The function to step the environment.
        scan_length: Number of evaluation steps.

    Returns:
        A dictionary of evaluation metrics.

    """
    policy = policy_fn(training_state.params.policy, deterministic=True)

    def run_step(carry, _t):
        env_state = carry

        env_state, _ = step_fn(env_state, env, policy)

        rollout_metrics = {
            "ep_rew_mean": env_state.info["rewards"],
            "ep_len_mean": env_state.info["steps"],
            **env_state.metrics,
        }

        return env_state, rollout_metrics

    env_state = env.init_and_reset(batch_scenarios)

    _, eval_metrics = jax.lax.scan(
        run_step,
        env_state,
        (),
        length=scan_length,
    )

    eval_metrics = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), eval_metrics)

    return eval_metrics


def prefill_replay_buffer(
    batch_scenarios: waymax_datatypes.SimulatorState,
    buffer_state: datatypes.TrainingState,
    key: jax.Array,
    env: waymax_env.PlanningAgentEnvironment,
    replay_buffer: replay_buffer.ReplayBuffer,
    action_shape: tuple[int],
    learning_start: int,
):
    """Prefill the replay buffer with random transitions.

    Args:
        batch_scenarios: The batch of scenarios.
        buffer_state: The current replay buffer state.
        key: The random key.
        env: The environment.
        replay_buffer: The replay buffer instance.
        action_shape: The shape of an action.
        learning_start: Number of steps to prefill before learning starts.

    Returns:
        The updated replay buffer state.

    """

    def run_random_step(carry, _x):
        state, buffer_state, key = carry
        key, step_key = jax.random.split(key)

        state, transition = pipeline.random_step(state, env, None, step_key, action_shape)

        buffer_state = replay_buffer.insert(buffer_state, transition)

        return (state, buffer_state, key), None

    key, subkey = jax.random.split(key, 2)
    reset_keys = jax.random.split(subkey, batch_scenarios.shape)

    env_state = env.init_and_reset(batch_scenarios, reset_keys)

    _, buffer_state, _ = jax.lax.scan(
        run_random_step,
        (env_state, buffer_state, key),
        (),
        length=learning_start,
    )[0]

    return buffer_state
