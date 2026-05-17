# Copyright 2025 Valeo.


"""Utility functions for evaluation script."""

import io
import os
import pickle
import re
import sys
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import mediapy
import numpy as np
import pandas as pd
import yaml
from etils import epath
from waymax import dynamics

from vmax.agents import pipeline
from vmax.simulator import make_env_for_evaluation, overrides, visualization
from vmax.simulator.metrics.aggregators import nuplan_aggregate_score, vmax_aggregate_score
from vmax.simulator.metrics.collector import _metrics_operands


def run_scenario_jit(scenario, rng_key: jax.Array, step_fn, reset_fn):
    # Reset environment for the new scenario.
    rng_key, reset_keys = jax.random.split(rng_key)
    if scenario.shape != ():
        reset_keys = jax.random.split(reset_keys, scenario.shape[0])
    env_transition = reset_fn(scenario, reset_keys)

    episode_metrics = {}

    for key, value in env_transition.metrics.items():
        # Initialize episode metrics.
        episode_metrics[key] = jnp.full((80,), -2.0, dtype=jnp.float32)
        # episode_metrics[key] = episode_metrics[key].at[env_transition.info["steps"][0]].set(value[0])

    @jax.jit
    def cond_fn(carry):
        env_transition, _, _ = carry
        return jnp.any(jnp.logical_not(env_transition.done))

    def body_fn(carry):
        env_transition, rng_key, episode_metrics = carry

        rng_key, step_key = jax.random.split(rng_key)
        env_transition, _ = step_fn(env_transition, key=step_key)

        # Collect episode metrics.
        for key, value in env_transition.metrics.items():
            episode_metrics[key] = episode_metrics[key].at[env_transition.info["steps"][0] - 1].set(value[0])

        return env_transition, rng_key, episode_metrics

    # Run the simulation loop
    env_transition, _, episode_metrics = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (env_transition, rng_key, episode_metrics),
    )

    return episode_metrics, env_transition.info["steps"]


def run_scenario_render(scenario, rng_key: jax.Array, env, step_fn, reset_fn, render_pov: bool = False):
    """Render a scenario in the environment and return the rendered images."""
    # Reset environment for the new scenario.
    rng_key, reset_key = jax.random.split(rng_key)
    reset_key = jax.random.split(reset_key, 1)
    env_transition = reset_fn(scenario, reset_key)

    episode_images = []

    image = plot_scene(env, env_transition, render_pov)
    episode_images.append(image)

    done = env_transition.done

    # Run the simulation loop
    while not done:
        rng_key, step_key = jax.random.split(rng_key)
        step_key = jax.random.split(step_key, 1)
        env_transition, _ = step_fn(env_transition, key=step_key)

        done = env_transition.done

        image = plot_scene(env, env_transition, render_pov)
        episode_images.append(image)

    return episode_images, env_transition.info["steps"]


def get_algorithm_modules(algorithm: str):
    """Get algorithm-specific modules with caching for performance."""
    algorithm = algorithm.lower()
    if algorithm == "sac":
        from vmax.agents.learning.reinforcement.sac.sac_factory import make_inference_fn
        from vmax.agents.learning.reinforcement.sac.sac_factory import make_networks as build_network
    elif algorithm == "bc":
        from vmax.agents.learning.imitation.bc.bc_factory import make_inference_fn
        from vmax.agents.learning.imitation.bc.bc_factory import make_networks as build_network
    elif algorithm == "bc_sac":
        from vmax.agents.learning.hybrid.bc_sac.bc_sac_factory import make_inference_fn
        from vmax.agents.learning.hybrid.bc_sac.bc_sac_factory import make_networks as build_network
    elif algorithm == "ppo":
        from vmax.agents.learning.reinforcement.ppo.ppo_factory import make_inference_fn
        from vmax.agents.learning.reinforcement.ppo.ppo_factory import make_networks as build_network
    else:
        raise ValueError(f"Invalid algorithm: {algorithm}")

    return make_inference_fn, build_network


def setup_evaluation(
    policy_type: str,
    path_model: str,
    source_dir: str,
    path_dataset: str,
    eval_name: str,
    max_num_objects: int,
    noisy_init: bool,
    sdc_paths_from_data: bool = True,
):
    """Set up the evaluation environment and parameters.

    Args:
        dummy_scenario: Scenario used for environment initialization.
        policy_type: The type of policy to evaluate.
        path_model: Identifier of the model to evaluate.
        source_dir: Source directory for evaluation results.
        path_dataset: Path to the dataset.
        eval_name: Name of the evaluation.
        max_num_objects: Maximum number of objects in the scene.
        noisy_init: Flag for noisy initializations.
        sdc_paths_from_data: Flag for SDC paths from data.

    Returns:
        The simulation environment, step function, evaluation path, and termination keys.

    """
    if policy_type == "ai":
        run_path = f"{source_dir}/{path_model}/"

        # Optimize model file detection
        model_path, model_name = get_model_path(run_path + "model/")

        # Construct eval path more efficiently
        model_name_clean = model_name.replace(".pkl", "")
        run_path_clean = run_path.replace(f"{source_dir}/", "")

        eval_path = f"{eval_name}/ai/{path_dataset}/{run_path_clean}{model_name_clean}/"

        # Load configuration efficiently
        eval_config = load_yaml_config(run_path + ".hydra/config.yaml")

        eval_config["encoder"] = eval_config["network"]["encoder"]
        eval_config["policy"] = eval_config["algorithm"]["network"]["policy"]
        eval_config["value"] = eval_config["algorithm"]["network"]["value"]
        eval_config["unflatten_config"] = eval_config["observation_config"]
        eval_config["action_distribution"] = eval_config["algorithm"]["network"]["action_distribution"]

        termination_keys = eval_config["termination_keys"]

        # Create environment.
        env = make_env_for_evaluation(
            max_num_objects=max_num_objects,
            dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
            sdc_paths_from_data=sdc_paths_from_data,
            observation_type=eval_config["observation_type"],
            observation_config=eval_config["observation_config"],
            termination_keys=termination_keys,
            noisy_init=noisy_init,
        )

        # Load model.
        policy = load_model(env, eval_config["algorithm"]["name"], eval_config, model_path)
        step_fn = make_step_fn(env, policy_type, policy)
    else:
        eval_path = f"{eval_name}/rule_based/{policy_type}/{path_dataset}/"

        eval_config = {"max_num_objects": max_num_objects}
        termination_keys = ["overlap", "offroad", "run_red_light"]

        env = make_env_for_evaluation(
            max_num_objects=max_num_objects,
            dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
            sdc_paths_from_data=sdc_paths_from_data,
            observation_type="idm",
            termination_keys=termination_keys,
            noisy_init=noisy_init,
        )
        step_fn = make_step_fn(env, policy_type)

    if noisy_init:
        eval_path += "noisy_init/"

    os.makedirs(eval_path, exist_ok=True)

    return env, step_fn, eval_path, termination_keys


def load_yaml_config(config_path: str) -> dict:
    """Load YAML configuration with optimized error handling."""
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}")
        raise


def load_model(env, algorithm, config, model_path):
    """Load and return the model for the specified learning algorithm."""
    obs_size = env.observation_spec()
    action_size = env.action_spec().data.shape[0]

    # Get algorithm modules with caching
    make_inference_fn, build_network = get_algorithm_modules(algorithm)

    # Build network with optimized approach
    unflatten_fn = env.get_wrapper_attr("features_extractor").unflatten_features

    if algorithm.lower() == "bc_sac":
        network = build_network(
            observation_size=obs_size,
            action_size=action_size,
            unflatten_fn=unflatten_fn,
            rl_learning_rate=config["algorithm"]["rl_learning_rate"],
            imitation_learning_rate=config["algorithm"]["imitation_learning_rate"],
            network_config=config,
        )
    elif algorithm.lower() in ["sac", "bc", "ppo"]:
        network = build_network(
            observation_size=obs_size,
            action_size=action_size,
            unflatten_fn=unflatten_fn,
            learning_rate=config["algorithm"]["learning_rate"],
            network_config=config,
        )

    # Create policy function.
    make_policy = make_inference_fn(network)
    params = load_params(model_path)

    return make_policy(params.policy, deterministic=True)


def make_step_fn(env, type_step: str, policy_fn=None) -> callable:
    """Return a step function based on the provided action type."""
    if type_step == "ai":
        return partial(pipeline.policy_step, env=env, policy_fn=policy_fn)
    elif type_step == "expert":
        return partial(pipeline.expert_step, env=env)
    elif type_step == "random":
        return partial(pipeline.random_step, env=env)
    elif type_step == "constant":
        return partial(pipeline.constant_step, env=env)
    elif type_step == "idm":
        return partial(pipeline.idm_step, env=env)
    elif type_step == "pdm":
        return partial(pipeline.pdm_step, env=env)
    else:
        raise ValueError(f"Invalid type step: {type_step}")


def load_params(path: str) -> Any:
    """Load serialized parameters with optimized error handling."""
    try:
        with epath.Path(path).open("rb") as fin:
            buf = fin.read()

        class ModuleCompatUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Map old module paths to new ones
                if module.startswith("vmax.learning.algorithms"):
                    new_module = module.replace("vmax.learning.algorithms.rl", "vmax.agents.learning.reinforcement")
                    print(f"Remapping: {module} -> {new_module}")
                    try:
                        # Try to import the new module path
                        __import__(new_module)
                        # Get the mapped class from the new module
                        return getattr(sys.modules[new_module], name)
                    except (ImportError, AttributeError) as e:
                        print(f"Error during remapping: {e}")

                # Default behavior for other modules
                return super().find_class(module, name)

        # Use our custom unpickler to load the parameters
        return ModuleCompatUnpickler(io.BytesIO(buf)).load()

    except Exception as e:
        print(f"Error loading parameters from {path}: {e}")
        raise


def get_model_path(model_path: str) -> tuple[str, str]:
    """Identify and return the latest model file path and its name with caching."""
    try:
        # Filter to get only files with .pkl extension
        pkl_files = [f for f in os.listdir(model_path) if f.endswith(".pkl")]

        if "model_final.pkl" in pkl_files:
            model_name = "model_final.pkl"
        else:
            # Sort files once with a more robust regex pattern
            sorted_files = sorted(
                pkl_files,
                key=lambda f: int(re.search(r"\d+", f).group()) if re.search(r"\d+", f) else 0,
            )
            model_name = sorted_files[-1] if sorted_files else None

        if model_name:
            print(f"-> Selecting latest weights: {model_name}")
            return model_path + model_name, model_name
        else:
            raise FileNotFoundError(f"No valid model files found in {model_path}")
    except Exception as e:
        print(f"Error finding model in {model_path}: {e}")
        raise


def write_video(run_path: str, episode_images: list, idx: int, is_noisy: bool = False):
    """Write and save a video with optimized memory handling."""
    if not episode_images:
        return

    # Create path dynamically
    video_path = run_path + "/mp4/" + ("noisy/" if is_noisy else "")
    os.makedirs(video_path, exist_ok=True)

    # Use more efficient video writing
    try:
        output_path = video_path + "eval_" + str(idx) + ".mp4"
        mediapy.write_video(output_path, episode_images, fps=10)
    except Exception as e:
        print(f"Error writing video: {e}")


def write_generator_result(save_path: str, num_scenarios: int, metrics: dict):
    """Save evaluation results and metrics."""
    # Create DataFrame directly from metrics
    df = pd.DataFrame(metrics)
    df.index.name = "scenario_index"

    # CSV writing
    df.to_csv(save_path + "/evaluation_episodes.csv", index=True)

    # Calculate aggregated metrics.
    metrics_mean = {}
    for key, values in metrics.items():
        if values:
            # Handle both numpy arrays and regular values.
            metrics_mean[key] = np.mean([val.item() if isinstance(val, np.ndarray) else val for val in values])

    # Add derived metrics.
    if "accuracy" in metrics_mean and "overlap" in metrics_mean and "at_fault_collision" in metrics_mean:
        metrics_mean["accuracy_only_at_fault"] = metrics_mean["accuracy"] + (
            metrics_mean["overlap"] - metrics_mean["at_fault_collision"]
        )

    # Format and write results.
    key_width = max(len(key) for key in metrics_mean)
    value_width = 10

    with open(save_path + "/evaluation_results.txt", "w") as txtfile:
        txtfile.write("=" * 50 + "\n")
        txtfile.write(f"Evaluation - {num_scenarios} scenarios :\n")
        txtfile.write("=" * 50 + "\n")
        txtfile.write(f"{'Metric':<{key_width}} {'Value':>{value_width}}\n")
        txtfile.write("-" * (key_width + value_width + 1) + "\n")

        for key, value in metrics_mean.items():
            txtfile.write(f"{key:<{key_width}} {value:>{value_width}.5f}\n")


def plot_scene(env, env_transition, sdc_pov: bool):
    """Generate and return a plot image."""
    if sdc_pov:
        return visualization.plot_input_agent(env_transition.state, env, batch_idx=0)
    else:
        return overrides.plot_simulator_state(env_transition.state, use_log_traj=False, batch_idx=0)


def _process_metric(key, value, operand, batch_index, steps, eval_metrics, batch_metrics):
    """Process a single metric and update the metrics dictionaries."""
    batch_value = value[batch_index][:steps]

    if not isinstance(operand, dict):
        # Apply operand to values directly
        new_value = operand(batch_value)
        if key not in eval_metrics:
            eval_metrics[key] = []
        eval_metrics[key].append(new_value)
        batch_metrics[key] = new_value
    else:
        # Handle nested metrics
        for sub_key, sub_operand in operand.items():
            new_value = sub_operand(batch_value)
            if sub_key not in eval_metrics:
                eval_metrics[sub_key] = []
            eval_metrics[sub_key].append(new_value)
            batch_metrics[sub_key] = new_value

    return eval_metrics, batch_metrics


def _check_episode_success(batch_metrics, term_keys):
    """Determine if an episode was successful based on termination criteria."""
    is_episode_not_finished = 0
    for key in term_keys:
        if key in batch_metrics:
            is_episode_not_finished += np.sum(batch_metrics[key])

    # Return accuracy (1 if successful, 0 otherwise)
    return 1 - (is_episode_not_finished > 0)


def append_episode_metrics(steps_done, eval_metrics, episode_metrics, term_keys, batch_size):
    """Aggregate episode metrics with support for batched data."""
    # Process each item in the batch
    for idx in range(batch_size):
        batch_metrics = {}  # Metrics for current batch item
        steps = steps_done[idx][0]

        eval_metrics["episode_length"].append(steps)
        batch_metrics["episode_length"] = steps

        # Process metric values
        for key, value in episode_metrics.items():
            operand = _metrics_operands[key]
            eval_metrics, batch_metrics = _process_metric(
                key,
                value,
                operand,
                idx,
                steps,
                eval_metrics,
                batch_metrics,
            )

        # Calculate episode success for this batch item
        accuracy = _check_episode_success(batch_metrics, term_keys)
        eval_metrics["accuracy"].append(accuracy)
        batch_metrics["accuracy"] = accuracy

        # Calculate and add aggregate scores
        if "nuplan_aggregate_score" not in eval_metrics:
            eval_metrics["nuplan_aggregate_score"] = []
        eval_metrics["nuplan_aggregate_score"].append(nuplan_aggregate_score(batch_metrics))

        if "vmax_aggregate_score" not in eval_metrics:
            eval_metrics["vmax_aggregate_score"] = []
        eval_metrics["vmax_aggregate_score"].append(vmax_aggregate_score(batch_metrics))

    return eval_metrics
