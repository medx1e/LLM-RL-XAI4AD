import os


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


import argparse
import time
from functools import partial

import jax
import numpy as np
from tqdm import tqdm

from vmax.scripts.evaluate import utils
from vmax.scripts.training.train_utils import str2bool
from vmax.simulator import datasets, make_data_generator


def parse_eval_args():
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluation script arguments")
    parser.add_argument(
        "--sdc_actor",
        "-sdc",
        type=str,
        default="expert",
        help="Actor type: 'ai' for learned policy, otherwise rule-based (default: expert)",
    )
    parser.add_argument(
        "--max_num_objects",
        "-o",
        type=int,
        default=64,
        help="Maximum number of objects in the scene (default: 64)",
    )
    parser.add_argument(
        "--scenario_indexes",
        "-si",
        nargs="*",
        type=int,
        default=None,
        help="Optional list of scenario indexes to evaluate",
    )
    parser.add_argument(
        "--render",
        "-r",
        type=str2bool,
        default=False,
        help="Render the evaluation (default: False)",
    )
    parser.add_argument(
        "--sdc_pov",
        "-pov",
        type=str2bool,
        default=False,
        help="Render from self-driving car's point of view (default: False)",
    )
    parser.add_argument(
        "--path_dataset",
        "-pd",
        type=str,
        default="local_womd_valid",
        help="Path to the dataset (default: local_womd_valid)",
    )
    parser.add_argument(
        "--path_model",
        "-pm",
        type=str,
        default="",
        help="Identifier of the model to evaluate",
    )
    parser.add_argument(
        "--eval_name",
        "-en",
        type=str,
        default="benchmark",
        help="Base evaluation directory name or identifier",
    )
    parser.add_argument(
        "--noisy_init",
        "-ni",
        type=str2bool,
        default=False,
        help="Flag to enable noisy initialization (default: False)",
    )
    parser.add_argument(
        "--src_dir",
        "-sd",
        type=str,
        default="runs",
        help="Source directory for evaluation results (default: runs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)",
    )
    parser.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        default=1,
        help="Number of scenarios to process in parallel (default: 1)",
    )
    parser.add_argument(
        "--waymo_dataset",
        "-wd",
        type=str2bool,
        default=False,
        help="Use Waymo dataset (default: False)",
    )
    parser.add_argument(
        "--plot-failures",
        "-pf",
        action="store_true",
        help="Plot failed scenarios (accuracy=0) from previously run evaluation",
    )

    args = parser.parse_args()

    # Validate the provided arguments
    if (args.render or args.sdc_pov) and args.batch_size > 1:
        raise ValueError("Batch size must be 1 when rendering or using SDC POV.")
    if args.sdc_actor == "ai" and args.path_model == "":
        raise ValueError("Model path must be provided for AI actor.")

    return args


def run_evaluation(
    env,
    data_generator,
    step_fn,
    run_path: str = "",
    scenario_indexes: list | None = None,
    termination_keys: list | None = None,
    render: bool = False,
    render_pov: bool = False,
    seed: int = 0,
    batch_size: int = 1,
    plot_failures: bool = False,
):
    """Evaluate the model over multiple episodes and store the results.

    Args:
        env: The simulation environment.
        data_generator: Generator providing evaluation scenarios.
        step_fn: Step function for simulation steps.
        run_path: Directory for saving evaluation outputs.
        scenario_indexes: Optional list of specific scenario indexes to evaluate.
        termination_keys: List of keys that determine episode termination.
        render: Whether to render the evaluation visually.
        render_pov: Whether to render from the SDC's point of view.
        seed: Random seed for reproducibility.
        batch_size: Number of scenarios to process in parallel.
        plot_failures: Whether to plot failed scenarios from previous evaluation.

    Returns:
        dict: Dictionary containing aggregated evaluation metrics.
    """
    if plot_failures and run_path:
        # Read CSV file to get failed scenarios
        csv_path = os.path.join(os.path.dirname(run_path), "evaluation_episodes.csv")
        if not os.path.exists(csv_path):
            raise ValueError(f"No evaluation CSV file found at {csv_path}")

        import pandas as pd

        df = pd.read_csv(csv_path)
        failed_scenarios = df[df["accuracy"] == 0]["scenario_index"].tolist()

        if not failed_scenarios:
            print("No failed scenarios found in previous evaluation")
            return {}

        print(f"Found {len(failed_scenarios)} failed scenarios, rendering...")
        scenario_indexes = failed_scenarios

    # JIT compile the step function and reset for speed.
    jitted_step_fn = jax.jit(step_fn)
    jitted_reset = jax.jit(env.reset)

    rng_key = jax.random.PRNGKey(seed)
    eval_metrics = {"episode_length": [], "accuracy": []}
    start_time_total = time.time()

    # Setup progress bar
    progress_bar = tqdm(desc="Evaluating scenarios", unit=" scenario")

    rendering = render or render_pov

    if rendering:
        _run_scenario = partial(
            utils.run_scenario_render,
            env=env,
            step_fn=jitted_step_fn,
            reset_fn=jitted_reset,
            render_pov=render_pov,
        )
    else:
        _run_scenario = partial(utils.run_scenario_jit, step_fn=jitted_step_fn, reset_fn=jitted_reset)
        _run_scenario = jax.vmap(_run_scenario)
        _run_scenario = jax.jit(_run_scenario)

    total_scenarios = 0

    for scenario in data_generator:
        # Skip scenarios if scenario_indexes is provided.
        if scenario_indexes is not None and total_scenarios not in scenario_indexes:
            total_scenarios += batch_size
            continue

        rng_key, scenario_key = jax.random.split(rng_key)

        if rendering:
            images, steps_done = _run_scenario(scenario, scenario_key)
            utils.write_video(run_path, images, total_scenarios)
            images.clear()
        else:
            scenario_key = jax.random.split(scenario_key, batch_size)
            episode_metrics, steps_done = _run_scenario(scenario, scenario_key)

            # Aggregate metrics for the episode.
            eval_metrics = utils.append_episode_metrics(
                steps_done,
                eval_metrics,
                episode_metrics,
                termination_keys,
                batch_size,
            )

        progress_bar.update(batch_size)
        total_scenarios += batch_size

    progress_bar.close()

    total_time = time.time() - start_time_total

    # Write aggregated evaluation results.
    if not rendering:
        utils.write_generator_result(run_path, total_scenarios, eval_metrics)

    print(
        f"-> Evaluation completed: {total_scenarios} episodes in {total_time:.2f}s "
        f"(avg {total_time / total_scenarios:.2f}s per episode)",
    )

    return None if rendering else eval_metrics


def main():
    """Main evaluation function that sets up and runs the evaluation process."""
    eval_args = parse_eval_args()

    print(f"-> Setting up evaluation for {eval_args.sdc_actor} policy on {eval_args.path_dataset} dataset...")

    if eval_args.plot_failures:
        print("-> Plotting failed scenarios from previous evaluation...")
        eval_args.render = True
        eval_args.sdc_pov = True

    batch_dims = (1,) if eval_args.render or eval_args.sdc_pov else (eval_args.batch_size, 1)
    include_sdc_paths = not eval_args.waymo_dataset

    # Create data generator for providing scenarios
    data_generator = make_data_generator(
        path=datasets.get_dataset(eval_args.path_dataset),
        max_num_objects=eval_args.max_num_objects,
        include_sdc_paths=include_sdc_paths,
        batch_dims=batch_dims,
        seed=eval_args.seed,
        repeat=1,
    )

    # Setup environment, policy, and evaluation paths
    env, step_fn, eval_path, termination_keys = utils.setup_evaluation(
        eval_args.sdc_actor,
        eval_args.path_model,
        eval_args.src_dir,
        eval_args.path_dataset,
        eval_args.eval_name,
        eval_args.max_num_objects,
        eval_args.noisy_init,
        include_sdc_paths,
    )

    print(f"-> Starting evaluation with output path: {eval_path}")

    # Run the evaluation
    eval_metrics = run_evaluation(
        env,
        data_generator,
        step_fn,
        eval_path,
        eval_args.scenario_indexes,
        termination_keys,
        eval_args.render,
        eval_args.sdc_pov,
        eval_args.seed,
        eval_args.batch_size,
        eval_args.plot_failures,
    )

    if eval_metrics is not None:
        # Print a summary of key metrics
        print("\n-> Evaluation Summary:")
        print(f"Accuracy: {np.mean(eval_metrics['accuracy']):.4f}")
        if "vmax_aggregate_score" in eval_metrics:
            print(f"V-Max Score: {np.mean(eval_metrics['vmax_aggregate_score']):.4f}")


if __name__ == "__main__":
    main()
