# Copyright 2025 Valeo.


"""Script to run the training process."""

import os
import sys
from functools import partial

import hydra
from omegaconf import DictConfig, OmegaConf
from waymax import dynamics

from vmax import PATH_TO_APP, simulator
from vmax.agents import learning
from vmax.scripts.training import train_utils


OmegaConf.register_new_resolver("output_dir", train_utils.resolve_output_dir)


@hydra.main(version_base=None, config_name="base_config", config_path=PATH_TO_APP + "/config")
def run(cfg: DictConfig) -> None:
    """Run the training process with the provided configuration.

    Args:
        cfg: Configuration for the training process.

    """
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    train_utils.apply_xla_flags(config)
    train_utils.print_hyperparameters(config)
    train_utils.get_and_print_device_info()

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

    ## TRAINING
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


if __name__ == "__main__":
    run()
