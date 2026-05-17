# Copyright 2025 Valeo.

"""Proximal Policy Optimization (PPO) trainer for V-Max.

Standard PPO training implementation. For XAI analysis (attention extraction,
head specialization), use offline analysis scripts after training completes.
"""

from __future__ import annotations

import os
import typing
from collections.abc import Callable
from functools import partial
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from vmax.agents import datatypes, pipeline
from vmax.agents.learning.reinforcement import ppo
from vmax.agents.pipeline import inference, pmap
from vmax.agents.networks import encoders, network_utils
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
    value_coef: float,
    entropy_coef: float,
    discount: float,
    gae_lambda: float,
    eps_clip: float,
    normalize_advantages: bool,
    save_freq: int,
    eval_freq: int,
    learning_rate: float,
    grad_updates_per_step: int,
    batch_size: int,
    unroll_length: int,
    num_minibatches: int,
    network_config: dict,
    progress_fn: Callable[[int, datatypes.Metrics], None] = lambda *args: None,
    checkpoint_logdir: str = "",
    disable_tqdm: bool = False,

) -> None:
    """Train a PPO agent using Proximal Policy Optimization.

    Args:
        env: An instance of the planning environment.
        data_generator: Iterator yielding simulator state samples.
        eval_scenario: Simulator state used for evaluation.
        num_scenario_per_eval: Number of evaluation scenarios.
        total_timesteps: Total training timesteps.
        num_envs: Number of parallel environments.
        num_episode_per_epoch: Episodes per epoch.
        scenario_length: Number of steps per scenario.
        log_freq: Frequency of logging.
        seed: Random seed.
        value_coef: Coefficient for value loss.
        entropy_coef: Coefficient for entropy loss.
        discount: Discount factor.
        gae_lambda: Lambda for Generalized Advantage Estimation.
        eps_clip: PPO clipping parameter.
        normalize_advantages: Flag for normalizing advantages.
        save_freq: Frequency to save model checkpoints.
        eval_freq: Evaluation frequency.
        learning_rate: Learning rate for optimizers.
        grad_updates_per_step: Gradient update iterations per step.
        batch_size: Batch size.
        unroll_length: Unroll length for trajectory generation.
        num_minibatches: Number of minibatches.
        network_config: Dictionary for network configurations.
        progress_fn: Callback function for reporting progress.
        checkpoint_logdir: Directory path for saving checkpoints.
        disable_tqdm: Flag to disable tqdm progress bar.
    """
    print(" PPO Training ".center(40, "="))

    rng = jax.random.PRNGKey(seed)
    num_devices = jax.local_device_count()

    do_save = save_freq > 1 and checkpoint_logdir is not None
    do_evaluation = eval_freq >= 1

    env_step_per_training_step = batch_size * unroll_length * num_minibatches
    total_iters = (total_timesteps // env_step_per_training_step) + 1

    observation_size = env.observation_spec()
    action_size = env.action_spec().data.shape[0]

    rng, network_key = jax.random.split(rng)

    print("-> Initializing networks...")
    network, training_state, policy_fn = ppo.initialize(
        action_size,
        observation_size,
        env,
        learning_rate,
        network_config,
        num_devices,
        network_key,
    )
    learning_fn = ppo.make_sgd_step(
        network,
        num_minibatches,
        gae_lambda,
        discount,
        eps_clip,
        value_coef,
        entropy_coef,
        normalize_advantages,
    )
    step_fn = partial(inference.policy_step, extra_fields=("truncation", "steps", "rewards"))
    print("-> Initializing networks... Done.")
    

    unroll_fn = partial(
        inference.generate_unroll,
        unroll_length=unroll_length,
        env=env,
        step_fn=step_fn,
    )

    run_training = partial(
        pipeline.run_training_on_policy,
        env=env,
        learning_fn=learning_fn,
        policy_fn=policy_fn,
        unroll_fn=unroll_fn,
        scan_length=batch_size * num_minibatches // num_envs,
        grad_updates_per_step=grad_updates_per_step,
    )
    run_evaluation = partial(
        pipeline.run_evaluation,
        env=env,
        policy_fn=policy_fn,
        step_fn=step_fn,
        scan_length=scenario_length * num_scenario_per_eval,
    )

    run_training = jax.pmap(run_training, axis_name="batch")
    run_evaluation = jax.pmap(run_evaluation, axis_name="batch")

    time_training = perf_counter()

    current_step = 0

    print("-> Ground Control to Major Tom...")
    for iter in tqdm(range(total_iters), desc="PPO Training", total=total_iters, dynamic_ncols=True, disable=disable_tqdm):
        rng, iter_key = jax.random.split(rng)
        iter_keys = jax.random.split(iter_key, num_devices)

        # Batch data generation
        t = perf_counter()
        batch_scenarios = next(data_generator)
        epoch_data_time = perf_counter() - t

        # Training step
        t = perf_counter()
        training_state, training_metrics = run_training(batch_scenarios, training_state, iter_keys)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), training_metrics)

        epoch_training_time = perf_counter() - t

        #  Log training metrics
        t = perf_counter()
        training_metrics = pmap.flatten_tree(training_metrics)
        training_metrics = jax.device_get(training_metrics)
        training_metrics = _metrics.collect(training_metrics, "ep_len_mean")

        current_step = int(pmap.unpmap(training_state.env_steps))

        metrics = {
            "runtime/sps": int(env_step_per_training_step / epoch_training_time),
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
