# Copyright 2025 Valeo.


"""Soft Actor-Critic (SAC) trainer."""

from __future__ import annotations

import typing
from collections.abc import Callable
from functools import partial
from time import perf_counter

import jax
import jax.numpy as jnp
from tqdm import tqdm

from vmax.agents import datatypes, pipeline
from vmax.agents.learning.reinforcement import sac
from vmax.agents.learning.replay_buffer import ReplayBuffer
from vmax.agents.pipeline import inference, pmap
from vmax.scripts.training import train_utils
from vmax.simulator import metrics as _metrics


if typing.TYPE_CHECKING:
    from waymax import datatypes as waymax_datatypes
    from waymax import env as waymax_env


def train(
    env: waymax_env.PlanningAgentEnvironment,
    data_generator: typing.Iterator[waymax_datatypes.SimulatorState],
    eval_scenario: waymax_datatypes.SimulatorState,
    num_scenario_per_eval: int,
    total_timesteps: int,
    num_envs: int,
    num_episode_per_epoch: int,
    scenario_length: int,
    log_freq: int,
    seed: int,
    learning_start: int,
    alpha: float,
    discount: float,
    tau: float,
    save_freq: int,
    eval_freq: int,
    buffer_size: int,
    batch_size: int,
    learning_rate: float,
    grad_updates_per_step: int,
    unroll_length: int,
    network_config: dict,
    progress_fn: Callable[[int, datatypes.Metrics], None] = lambda *args: None,
    checkpoint_logdir: str = "",
    disable_tqdm: bool = False,
) -> None:
    """Train a SAC agent.

    Args:
        env: An instance of the planning environment.
        data_generator: Iterator yielding simulator state samples.
        eval_scenario: The simulator state used for evaluation.
        num_scenario_per_eval: Number of evaluation scenarios.
        total_timesteps: Total training timesteps.
        num_envs: Number of parallel environments.
        num_episode_per_epoch: Episodes per epoch.
        scenario_length: Number of steps per scenario.
        log_freq: Logging frequency.
        seed: Random seed.
        learning_start: Timesteps before training starts.
        alpha: Entropy regularization coefficient.
        discount: Discount factor.
        tau: Target network update coefficient.
        save_freq: Checkpoint save frequency.
        eval_freq: Evaluation frequency.
        buffer_size: Replay buffer size.
        batch_size: Batch size.
        learning_rate: Learning rate.
        grad_updates_per_step: Gradient update iterations per step.
        unroll_length: Unroll length for generating transitions.
        network_config: Configuration dictionary for networks.
        progress_fn: Callback function for progress updates.
        checkpoint_logdir: Directory path for saving checkpoints.
        disable_tqdm: Flag to disable tqdm progress bar.

    """
    print(" SAC ".center(40, "="))

    rng = jax.random.PRNGKey(seed)
    num_devices = jax.local_device_count()

    do_save = save_freq > 1 and checkpoint_logdir is not None
    do_evaluation = eval_freq >= 1

    num_steps = num_episode_per_epoch * scenario_length
    env_steps_per_iter = num_steps * num_envs
    total_iters = (total_timesteps // env_steps_per_iter) + 1

    observation_size = env.observation_spec()
    action_size = env.action_spec().data.shape[0]

    rng, network_key = jax.random.split(rng)

    print("-> Initializing networks...")
    network, training_state, policy_fn = sac.initialize(
        action_size,
        observation_size,
        env,
        learning_rate,
        network_config,
        num_devices,
        network_key,
    )
    learning_fn = sac.make_sgd_step(network, alpha, discount, tau)
    step_fn = partial(inference.policy_step, use_partial_transition=True)

    replay_buffer = ReplayBuffer(
        buffer_size=buffer_size // num_devices,
        batch_size=batch_size * grad_updates_per_step // num_devices,
        samples_size=num_envs,
        dummy_data_sample=datatypes.RLPartialTransition(
            observation=jnp.zeros((observation_size,)),
            action=jnp.zeros((action_size,)),
            reward=0.0,
            flag=0,
            done=0,
        ),
    )
    print("-> Initializing networks... Done.")

    unroll_fn = partial(
        inference.generate_unroll,
        unroll_length=unroll_length,
        env=env,
        step_fn=step_fn,
    )

    run_training = partial(
        pipeline.run_training_off_policy,
        replay_buffer=replay_buffer,
        env=env,
        learning_fn=learning_fn,
        policy_fn=policy_fn,
        unroll_fn=unroll_fn,
        grad_updates_per_step=grad_updates_per_step,
        scan_length=num_steps // unroll_length,
    )
    run_evaluation = partial(
        pipeline.run_evaluation,
        env=env,
        policy_fn=policy_fn,
        step_fn=step_fn,
        scan_length=scenario_length * num_scenario_per_eval,
    )
    prefill_replay_buffer = partial(
        pipeline.prefill_replay_buffer,
        env=env,
        replay_buffer=replay_buffer,
        action_shape=(num_envs, action_size),
        learning_start=learning_start,
    )

    run_training = jax.pmap(run_training, axis_name="batch")
    run_evaluation = jax.pmap(run_evaluation, axis_name="batch")
    prefill_replay_buffer = jax.pmap(prefill_replay_buffer, axis_name="batch")

    print("-> Prefilling replay buffer...")
    rng, rb_key = jax.random.split(rng)
    buffer_state = jax.pmap(replay_buffer.init)(jax.random.split(rb_key, num_devices))

    rng, prefill_key = jax.random.split(rng)
    prefill_keys = jax.random.split(prefill_key, num_devices)

    buffer_state = prefill_replay_buffer(next(data_generator), buffer_state, prefill_keys)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), buffer_state)
    print("-> Prefilling replay buffer... Done.")

    time_training = perf_counter()

    current_step = 0

    print("-> Ground Control to Major Tom...")
    for iter in tqdm(range(total_iters), desc="Training", total=total_iters, dynamic_ncols=True, disable=disable_tqdm):
        rng, iter_key = jax.random.split(rng)
        iter_keys = jax.random.split(iter_key, num_devices)

        # Batch data generation
        t = perf_counter()
        batch_scenarios = next(data_generator)
        epoch_data_time = perf_counter() - t

        # Training step
        t = perf_counter()
        training_state, buffer_state, training_metrics = run_training(
            batch_scenarios,
            training_state,
            buffer_state,
            iter_keys,
        )
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), training_metrics)

        epoch_training_time = perf_counter() - t

        #  Log training metrics
        t = perf_counter()
        training_metrics = pmap.flatten_tree(training_metrics)
        training_metrics = jax.device_get(training_metrics)
        training_metrics = _metrics.collect(training_metrics, "ep_len_mean")

        current_step = int(pmap.unpmap(training_state.env_steps))

        metrics = {
            "runtime/sps": int(env_steps_per_iter / epoch_training_time),
            **{f"{name}": value for name, value in training_metrics.items()},
        }

        if do_save and not iter % save_freq:
            path = f"{checkpoint_logdir}/model_{current_step}.pkl"
            train_utils.save_params(path, pmap.unpmap(training_state.params))

        epoch_log_time = perf_counter() - t

        # Evaluation
        t = perf_counter()
        if do_evaluation and not iter % eval_freq:
            eval_metrics = run_evaluation(eval_scenario, training_state)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), eval_metrics)
            eval_metrics = pmap.flatten_tree(eval_metrics)
            eval_metrics = _metrics.collect(eval_metrics, "ep_len_mean")
            progress_fn(current_step, eval_metrics)

        epoch_eval_time = perf_counter() - t

        if not iter % log_freq:
            metrics["runtime/data_time"] = epoch_data_time
            metrics["runtime/training_time"] = epoch_training_time
            metrics["runtime/log_time"] = epoch_log_time
            metrics["runtime/eval_time"] = epoch_eval_time
            metrics["runtime/iter_time"] = epoch_data_time + epoch_training_time + epoch_log_time + epoch_eval_time
            metrics["runtime/wall_time"] = perf_counter() - time_training
            metrics["train/rl_gradient_steps"] = int(pmap.unpmap(training_state.rl_gradient_steps))
            metrics["train/env_steps"] = current_step

            progress_fn(current_step, metrics, total_timesteps)

            if disable_tqdm:
                print(f"-> Step {current_step}/{total_timesteps} - {(current_step / total_timesteps) * 100:.2f}%")

    print(f"-> Training took {perf_counter() - time_training:.2f}s")
    assert current_step >= total_timesteps

    if checkpoint_logdir:
        path = f"{checkpoint_logdir}/model_final.pkl"
        train_utils.save_params(path, pmap.unpmap(training_state.params))

    pmap.assert_is_replicated(training_state)
    pmap.synchronize_hosts()
