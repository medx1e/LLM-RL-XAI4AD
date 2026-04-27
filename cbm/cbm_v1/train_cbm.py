#!/usr/bin/env python3
"""CBM-V1 training launcher.

Usage:
    /home/med1e/anaconda3/envs/vmax/bin/python cbm_v1/train_cbm.py \
        --config cbm_v1/config_womd_frozen.yaml

Override individual values:
    ... --total_timesteps 500000 --mode frozen --seed 1

Outputs are written to:
    <output_dir>/<run_name>/
        checkpoints/model_<step>.pkl
        checkpoints/model_final.pkl
        tb/                            # TensorBoard logs
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "V-Max"))

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Config loading ───────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_cli_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply --key value overrides from CLI onto the config dict."""
    i = 0
    while i < len(overrides):
        key = overrides[i].lstrip("-")
        val = overrides[i + 1] if i + 1 < len(overrides) else None
        i += 2
        if val is None:
            continue
        # Type-cast based on existing config value
        if key in config:
            orig = config[key]
            if isinstance(orig, bool):
                config[key] = val.lower() in ("1", "true", "yes")
            elif isinstance(orig, int):
                config[key] = int(val)
            elif isinstance(orig, float):
                config[key] = float(val)
            else:
                config[key] = val
        else:
            # Try numeric coercion
            try:
                config[key] = int(val)
            except ValueError:
                try:
                    config[key] = float(val)
                except ValueError:
                    config[key] = val
    return config


# ── Pretrained config loading ─────────────────────────────────────────

def load_pretrained_run_config(pretrained_dir: str) -> dict:
    """Read the hydra config from the pretrained V-Max run."""
    config_path = os.path.join(pretrained_dir, ".hydra", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Pretrained config not found at {config_path}. "
            f"Expected a V-Max run directory with .hydra/config.yaml."
        )
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Network config builder ────────────────────────────────────────────

def build_network_config(pretrained_cfg: dict, encoder_remap: dict) -> dict:
    """Build the network_config dict for CBM factory from pretrained config."""
    enc_cfg = dict(pretrained_cfg["network"]["encoder"])
    enc_type = enc_cfg.get("type", "none")
    enc_cfg["type"] = encoder_remap.get(enc_type, enc_type)

    # policy/value configs not used by CBM factory (it uses cbm_config instead)
    # but we pass them through so the factory can parse activation functions
    net_cfg = {
        "encoder": enc_cfg,
        "policy": pretrained_cfg["algorithm"]["network"]["policy"],
        "value": pretrained_cfg["algorithm"]["network"]["value"],
        "action_distribution": pretrained_cfg["algorithm"]["network"].get(
            "action_distribution", "gaussian"
        ),
        "_obs_type": pretrained_cfg.get("observation_type", "vec"),
    }
    return net_cfg


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train CBM-V1 on WOMD")
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML config file (e.g. cbm_v1/config_womd_frozen.yaml)"
    )
    args, extra = parser.parse_known_args()

    # Load and merge config
    config = load_config(args.config)
    if extra:
        config = merge_cli_overrides(config, extra)

    pretrained_dir = config.get("pretrained_dir", None)
    # Treat "none" / "" as no pretrained dir (scratch mode)
    if pretrained_dir in (None, "none", ""):
        pretrained_dir = None

    encoder_remap = config.get("encoder_remap", {"perceiver": "lq"})
    obs_type_remap = config.get("obs_type_remap", {"road": "vec", "lane": "vec"})

    if pretrained_dir is not None:
        # ── Standard path: steal configs from the pretrained V-Max run ──
        pretrained_cfg = load_pretrained_run_config(pretrained_dir)
        network_config = build_network_config(pretrained_cfg, encoder_remap)
        observation_config_dict = pretrained_cfg.get("observation_config", {})
        termination_keys = pretrained_cfg.get(
            "termination_keys", ["offroad", "overlap", "run_red_light"]
        )
        reward_type = pretrained_cfg.get("reward_type", "linear")
        raw_reward_config = pretrained_cfg.get("reward_config", {
            "offroad": -1.0, "overlap": -1.0
        })
    else:
        # ── Scratch path: all configs come from the training YAML ────────
        network_config = config["network_config"]
        observation_config_dict = config.get("observation_config", {})
        termination_keys = config.get(
            "termination_keys", ["offroad", "overlap", "run_red_light"]
        )
        reward_type = config.get("reward_type", "linear")
        raw_reward_config = config.get("reward_config", {
            "offroad": -1.0, "overlap": -1.0
        })

    # The reward wrapper expects {name: float_weight}.
    # V-Max hydra configs may have nested dicts {name: {penalty, weight, ...}}.
    # Flatten by extracting the effective weight (penalty or weight field).
    reward_config = {}
    for k, v in raw_reward_config.items():
        if isinstance(v, dict):
            # Use 'weight' if present, else 'penalty', else sum of numeric values
            if "weight" in v:
                reward_config[k] = float(v["weight"]) * float(v.get("penalty", -1.0))
            elif "penalty" in v:
                reward_config[k] = float(v["penalty"])
            else:
                reward_config[k] = float(list(v.values())[0])
        else:
            reward_config[k] = float(v)

    # Print summary
    print()
    print("=" * 60)
    print("CBM TRAINING LAUNCH")
    print("=" * 60)
    print(f"  Config          : {args.config}")
    print(f"  Pretrained      : {pretrained_dir or 'SCRATCH (no pretrained encoder)'}")
    print(f"  Data            : {config['data_path']}")
    print(f"  Output          : {config['output_dir']}/{config['run_name']}")
    print(f"  Mode            : {config['mode']}")
    print(f"  Total steps     : {config['total_timesteps']:,}")
    print(f"  Parallel envs   : {config['num_parallel_envs']}")
    print(f"  Episodes/env    : {config['num_episodes_per_env']}")
    print(f"  Scan length     : {config['scan_length']}")
    print(f"  Learning rate   : {config['learning_rate']}")
    print(f"  Lambda concept  : {config['lambda_concept']}")
    print(f"  Buffer size     : {config['buffer_size']:,}")
    print(f"  Learning start  : {config['learning_start']:,}")
    print("=" * 60)
    print()

    # Confirm data exists
    if not os.path.exists(config["data_path"]):
        raise FileNotFoundError(
            f"Dataset not found: {config['data_path']}\n"
            f"Ensure WOMD tfrecord is at the specified path."
        )

    # Import trainer after env setup
    from cbm_v1.cbm_trainer import train

    train(
        pretrained_dir=pretrained_dir,
        data_path=config["data_path"],
        output_dir=config["output_dir"],
        run_name=config["run_name"],
        mode=config["mode"],
        num_parallel_envs=config["num_parallel_envs"],
        num_episodes_per_env=config["num_episodes_per_env"],
        scan_length=config["scan_length"],
        scenario_length=config["scenario_length"],
        unroll_length=config["unroll_length"],
        total_timesteps=config["total_timesteps"],
        learning_start=config["learning_start"],
        alpha=config["alpha"],
        discount=config["discount"],
        tau=config["tau"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        grad_updates_per_step=config["grad_updates_per_step"],
        buffer_size=config["buffer_size"],
        lambda_concept=config["lambda_concept"],
        num_concepts=config.get("num_concepts", 11),
        concept_phases=tuple(config.get("concept_phases", [1, 2])),
        log_freq=config["log_freq"],
        save_freq=config["save_freq"],
        seed=config["seed"],
        observation_config_dict=observation_config_dict,
        network_config=network_config,
        encoder_remap=encoder_remap,
        obs_type_remap=obs_type_remap,
        termination_keys=termination_keys,
        reward_type=reward_type,
        reward_config=reward_config,
        disable_tqdm=config.get("disable_tqdm", False),
    )


if __name__ == "__main__":
    main()
