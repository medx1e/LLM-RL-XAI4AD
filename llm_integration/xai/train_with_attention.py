# Copyright 2025 Valeo.


"""Script to run PPO training for XAI analysis.

This script trains a PPO agent and saves checkpoints that can be used for
offline attention analysis and Head Specialization Analysis.

Usage:
    python xai/train_with_attention.py algorithm=PPO save_freq=20

Checkpoints will be saved regularly and can be analyzed offline using
the scripts in xai/attention_analysis/.
"""

import os
import sys
from functools import partial

import hydra
from omegaconf import DictConfig, OmegaConf
from waymax import dynamics

from vmax import PATH_TO_APP, simulator
from vmax.scripts.training import train_utils

# Import PPO trainer (clean version without online logging)
from xai.xai_ppo_trainer import train as ppo_train


OmegaConf.register_new_resolver("output_dir", train_utils.resolve_output_dir)


@hydra.main(version_base=None, config_name="base_config", config_path=PATH_TO_APP + "/config")
def run(cfg: DictConfig) -> None:
    """Run the training process with attention logging.

    Args:
        cfg: Configuration for the training process.

    """
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    train_utils.apply_xla_flags(config)
    train_utils.print_hyperparameters(config)
    train_utils.get_and_print_device_info()

    # Print training info
    print(" Training Configuration ".center(40, "="))
    print(f"- Save Frequency: {config.get('save_freq', 200)} iterations")
    print(f"- Checkpoints for offline XAI analysis enabled")

    env_config, run_config = train_utils.build_config_dicts(config)

    # (num_devices, num_envs, num_episode_per_epoch)
    data_generator = simulator.make_data_generator(
        path=env_config["path_dataset"],
        max_num_objects=env_config["max_num_objects"],
        include_sdc_paths=env_config["sdc_paths_from_data"],
        batch_dims=(env_config["num_envs"], env_config["num_episode_per_epoch"]),
        seed=env_config["seed"],
        distributed=True,
    )

    if config["eval_freq"] > 0:
        eval_data_generator = simulator.make_data_generator(
            path=env_config["path_dataset_eval"],
            max_num_objects=env_config["max_num_objects"],
            include_sdc_paths=env_config["sdc_paths_from_data"],
            batch_dims=(8, config["num_scenario_per_eval"] // 8),
            seed=69,
            distributed=True,
        )
        eval_scenario = next(eval_data_generator)
        del eval_data_generator
    else:
        eval_scenario = None

    env = simulator.make_env_for_training(
        max_num_objects=env_config["max_num_objects"],
        dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
        sdc_paths_from_data=env_config["sdc_paths_from_data"],
        observation_type=env_config["observation_type"],
        observation_config=env_config["observation_config"],
        reward_type=env_config["reward_type"],
        reward_config=env_config["reward_config"],
        termination_keys=env_config["termination_keys"],
    )

    absolute_run_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    relative_run_path = "/".join(absolute_run_path.split("/")[-2:])

    model_path = os.path.join(relative_run_path, "model")
    os.makedirs(model_path, exist_ok=True)

    writer = train_utils.setup_tensorboard(relative_run_path)
    progress = partial(train_utils.log_metrics, writer=writer)

    ## TRAINING (Checkpoints saved for offline XAI analysis)
    # Ensure we're using PPO
    if config["algorithm"]["name"] != "PPO":
        print(f"[WARNING] This script is designed for PPO, got {config['algorithm']['name']}")
        print("[WARNING] Falling back to standard training")
        from vmax.agents import learning
        train_fn = learning.get_train_fn(config["algorithm"]["name"])
        train_fn(
            env=env,
            data_generator=data_generator,
            eval_scenario=eval_scenario,
            **run_config,
            progress_fn=progress,
            checkpoint_logdir=model_path,
            disable_tqdm=not sys.stdout.isatty(),
        )
    else:
        # Use PPO trainer (checkpoints will be saved at save_freq intervals)
        ppo_train(
            env=env,
            data_generator=data_generator,
            eval_scenario=eval_scenario,
            **run_config,
            progress_fn=progress,
            checkpoint_logdir=model_path,
            disable_tqdm=not sys.stdout.isatty(),
        )


if __name__ == "__main__":
    run()
