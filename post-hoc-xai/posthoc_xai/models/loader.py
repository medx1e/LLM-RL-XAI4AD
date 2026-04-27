"""V-MAX model loader with all compatibility fixes.

Handles the 5 known compatibility issues between the pretrained weights
(from runs_rlc/) and the current V-MAX codebase:

1. Pickle module path mismatch → use ``load_params()`` + tensorboardX
2. Encoder type aliases → ``perceiver`` → ``lq``, ``mgail`` → ``lqh``
3. Observation type aliases → ``road`` / ``lane`` → ``vec``
4. Parameter key mismatch → ``perceiver_attention`` → ``lq_attention``, etc.
5. ``speed_limit`` feature → sac_seed* models are broken
"""

import os
import sys
from pathlib import Path
from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
import yaml

# ---------------------------------------------------------------------------
# Remapping tables
# ---------------------------------------------------------------------------

ENCODER_REMAP: dict[str, str] = {"perceiver": "lq", "mgail": "lqh"}
OBS_TYPE_REMAP: dict[str, str] = {"road": "vec", "lane": "vec"}
PARAM_KEY_REMAP: dict[str, str] = {
    "perceiver_attention": "lq_attention",
    "mgail_attention": "lq_attention",
}

# Map (possibly aliased) encoder names → wrapper class names
# The actual import happens lazily in ``load_model`` to avoid circular imports.
ENCODER_TO_WRAPPER_NAME: dict[str, str] = {
    "perceiver": "PerceiverWrapper",
    "lq": "PerceiverWrapper",
    "mtr": "GenericWrapper",
    "wayformer": "GenericWrapper",
    "mgail": "GenericWrapper",
    "lqh": "GenericWrapper",
    "none": "GenericWrapper",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_vmax_importable(vmax_repo: Optional[str] = None) -> None:
    """Add the V-Max repo to ``sys.path`` if it is not already importable."""
    try:
        import vmax  # noqa: F401
        return
    except ImportError:
        pass

    if vmax_repo is None:
        # Default: look for V-Max/ next to the project root
        project_root = Path(__file__).resolve().parents[2]
        vmax_repo = str(project_root / "V-Max")

    if vmax_repo not in sys.path:
        sys.path.insert(0, vmax_repo)


def _remap_param_keys(params: Any, old_name: str, new_name: str) -> Any:
    """Recursively rename dict keys in a (possibly nested) param tree."""
    if isinstance(params, dict):
        return {
            (new_name if k == old_name else k): _remap_param_keys(v, old_name, new_name)
            for k, v in params.items()
        }
    return params


def _fix_param_keys(policy_params: Any) -> Any:
    """Apply all parameter-key remappings (Issue 4)."""
    for old_key, new_key in PARAM_KEY_REMAP.items():
        needs_remap = False
        for path, _ in jax.tree_util.tree_leaves_with_path(policy_params):
            if any(old_key in str(p) for p in path):
                needs_remap = True
                break
        if needs_remap:
            policy_params = _remap_param_keys(policy_params, old_key, new_key)
    return policy_params


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class LoadedVMAXModel:
    """Container for everything needed to use a loaded V-MAX model.

    Attributes:
        config: Raw YAML config from ``.hydra/config.yaml``.
        original_encoder_type: Encoder type string *before* alias remapping.
        encoder_type: Encoder type string *after* alias remapping (``lq``, ``mtr``, …).
        obs_type: Observation type after remapping (``vec``).
        policy_params: Policy parameters (with keys already fixed).
        policy_fn: Callable ``(obs, rng) -> (action, info)`` for inference.
        policy_module: The Flax ``PolicyNetwork`` nn.Module (for ``apply`` / ``capture_intermediates``).
        unflatten_fn: Converts flat obs → ``(features_tuple, masks_tuple)``.
        env: The Waymax evaluation environment.
        network: The ``SACNetworks`` container.
        obs_size: Integer observation size.
        action_size: Integer action size.
        model_dir: Path to the model directory.
        data_gen: Data generator (if data_path was provided).
    """

    def __init__(self, **kwargs: Any):
        for k, v in kwargs.items():
            setattr(self, k, v)


def load_vmax_model(
    model_dir: Union[str, Path],
    data_path: Optional[Union[str, Path]] = None,
    max_num_objects: int = 64,
    vmax_repo: Optional[str] = None,
) -> LoadedVMAXModel:
    """Load a V-MAX pretrained model with all compatibility fixes.

    Args:
        model_dir: Path to the model directory (e.g.
            ``runs_rlc/womd_sac_road_perceiver_minimal_42``).
        data_path: Path to a ``.tfrecord`` file.  If ``None`` the data
            generator is not created (you'll need to supply observations
            manually).
        max_num_objects: Maximum number of objects in a scenario.
        vmax_repo: Explicit path to the V-Max repo root (if not on sys.path).

    Returns:
        A ``LoadedVMAXModel`` with everything needed for inference and XAI.
    """
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    _ensure_vmax_importable(vmax_repo)

    from waymax import dynamics
    from vmax.simulator import make_env_for_evaluation, make_data_generator
    from vmax.agents.learning.reinforcement.sac.sac_factory import (
        make_inference_fn,
        make_networks,
    )
    from vmax.scripts.evaluate.utils import load_params
    from vmax.agents.networks import encoders, network_utils, decoders
    from vmax.agents.networks.network_factory import PolicyNetwork

    model_dir = Path(model_dir)

    # 1. Load config ---------------------------------------------------------
    config_path = model_dir / ".hydra" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    original_encoder_type = config["network"]["encoder"]["type"]

    # 2. Fix encoder alias (Issue 2) ----------------------------------------
    encoder_type = ENCODER_REMAP.get(original_encoder_type, original_encoder_type)
    config["network"]["encoder"]["type"] = encoder_type

    # 3. Fix observation type alias (Issue 3) --------------------------------
    obs_type = OBS_TYPE_REMAP.get(config["observation_type"], config["observation_type"])

    # 4. Check speed_limit (Issue 5) ----------------------------------------
    rg_features = (
        config.get("observation_config", {})
        .get("roadgraphs", {})
        .get("features", [])
    )
    if "speed_limit" in rg_features:
        raise RuntimeError(
            f"Model '{model_dir}' uses the 'speed_limit' roadgraph feature "
            f"which is not supported by Waymax. Use a womd_* model instead."
        )

    # 5. Build eval_config ---------------------------------------------------
    eval_config = dict(config)
    eval_config["encoder"] = config["network"]["encoder"]
    eval_config["policy"] = config["algorithm"]["network"]["policy"]
    eval_config["value"] = config["algorithm"]["network"]["value"]
    eval_config["unflatten_config"] = config["observation_config"]
    eval_config["action_distribution"] = config["algorithm"]["network"]["action_distribution"]

    # 6. Create environment --------------------------------------------------
    env = make_env_for_evaluation(
        max_num_objects=config.get("max_num_objects", max_num_objects),
        dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
        sdc_paths_from_data=True,
        observation_type=obs_type,
        observation_config=config["observation_config"],
        termination_keys=config["termination_keys"],
        noisy_init=False,
    )

    obs_size = env.observation_spec()
    action_size = env.action_spec().data.shape[0]
    unflatten_fn = env.get_wrapper_attr("features_extractor").unflatten_features

    # 7. Build network -------------------------------------------------------
    network = make_networks(
        observation_size=obs_size,
        action_size=action_size,
        unflatten_fn=unflatten_fn,
        learning_rate=eval_config["algorithm"]["learning_rate"],
        network_config=eval_config,
    )
    make_policy = make_inference_fn(network)

    # 8. Load params & fix keys (Issues 1 + 4) ------------------------------
    model_path = str(model_dir / "model" / "model_final.pkl")
    training_state = load_params(model_path)
    policy_params = _fix_param_keys(training_state.policy)

    # 9. Create policy function ----------------------------------------------
    policy_fn = make_policy(policy_params, deterministic=True)

    # 10. Rebuild the Flax module for capture_intermediates ------------------
    _config = network_utils.convert_to_dict_with_activation_fn(eval_config)
    enc_cfg = dict(_config["encoder"])
    enc_type = enc_cfg.pop("type")

    if enc_type == "none":
        encoder_layer = None
    else:
        encoder_layer = encoders.get_encoder(enc_type)(unflatten_fn, **enc_cfg)

    pol_cfg = _config["policy"]
    fc_cfg = {
        k: v
        for k, v in pol_cfg.items()
        if k not in ("type", "final_activation", "num_networks", "shared_encoder")
    }
    fc_layer = decoders.get_fully_connected(pol_cfg["type"])(**fc_cfg)

    output_size = network.parametric_action_distribution.param_size
    policy_module = PolicyNetwork(
        encoder_layer=encoder_layer,
        fully_connected_layer=fc_layer,
        output_size=output_size,
        final_activation=pol_cfg["final_activation"],
    )

    # 11. Optionally create data generator -----------------------------------
    data_gen = None
    if data_path is not None:
        data_gen = make_data_generator(
            path=str(data_path),
            max_num_objects=config.get("max_num_objects", max_num_objects),
            include_sdc_paths=True,
            batch_dims=(1,),
            seed=42,
            repeat=1,
        )

    return LoadedVMAXModel(
        config=config,
        original_encoder_type=original_encoder_type,
        encoder_type=encoder_type,
        obs_type=obs_type,
        policy_params=policy_params,
        policy_fn=policy_fn,
        policy_module=policy_module,
        unflatten_fn=unflatten_fn,
        env=env,
        network=network,
        obs_size=obs_size,
        action_size=action_size,
        model_dir=model_dir,
        data_gen=data_gen,
    )
