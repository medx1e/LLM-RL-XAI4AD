"""
Rollout Engine.

Runs a closed-loop episode with any registered model and returns a
ScenarioData dataclass containing pure-numpy arrays ready for rendering.

Public API
----------
run_rollout(model_key, data_path, scenario_idx, num_steps=80) -> ScenarioData
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ── Path setup ───────────────────────────────────────────────────────────────
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "V-Max"))

import jax
import jax.numpy as jnp
import yaml
from waymax import datatypes as wdatatypes

from vmax.simulator import make_env_for_evaluation, make_data_generator
from vmax.scripts.evaluate.utils import load_params

from bev_visualizer.model_registry import MODEL_REGISTRY

# ── Make Waymax metric registration idempotent ────────────────────────────────
# Streamlit (and any multi-call context) will call make_env_for_evaluation
# more than once. Waymax raises on duplicate metric registration, so we
# patch register_metric to silently skip already-registered metrics.
import waymax.metrics as _wm_metrics
_original_register = _wm_metrics.register_metric

def _safe_register_metric(name, metric_class):
    try:
        _original_register(name, metric_class)
    except Exception:
        pass  # already registered — safe to skip

_wm_metrics.register_metric = _safe_register_metric


# ── Output dataclass ─────────────────────────────────────────────────────────

@dataclasses.dataclass
class ScenarioData:
    """Pure-numpy data extracted from one closed-loop rollout.

    Attributes
    ----------
    ego_xy     : (T, 2)  — model-driven ego (x, y) at each timestep.
    ego_yaw    : (T,)    — ego heading in radians.
    agents_xy  : (T, A, 2) — all other agents (x, y).
    agents_valid: (T, A)   — boolean validity mask per agent per step.
    agents_types: (A,)    — object type ids.
    frame_states: list[SimulatorState] — one Waymax state per step (for rendering).
    rewards    : (T,)   — per-step reward received by the ego.
    dones      : (T,)   — episode termination flag.
    model_key  : str    — which model produced this rollout.
    scenario_idx: int   — which scenario from the dataset.
    """
    ego_xy: np.ndarray
    ego_yaw: np.ndarray
    agents_xy: np.ndarray
    agents_valid: np.ndarray
    agents_types: np.ndarray
    frame_states: list
    rewards: np.ndarray
    dones: np.ndarray
    model_key: str
    scenario_idx: int


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_hydra_config(run_dir: str) -> dict:
    """Load .hydra/config.yaml from a V-Max run directory."""
    cfg_path = Path(run_dir) / ".hydra" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _build_network_config(pretrained_cfg: dict) -> dict:
    """Extract network config from a pretrained run's hydra config."""
    enc_cfg = dict(pretrained_cfg["network"]["encoder"])
    enc_type = enc_cfg.get("type", "none")
    enc_cfg["type"] = {"perceiver": "lq"}.get(enc_type, enc_type)
    obs_type = {"road": "vec", "lane": "vec"}.get(
        pretrained_cfg.get("observation_type", "vec"), "vec"
    )
    return {
        "encoder": enc_cfg,
        "policy": pretrained_cfg["algorithm"]["network"]["policy"],
        "value": pretrained_cfg["algorithm"]["network"]["value"],
        "action_distribution": pretrained_cfg["algorithm"]["network"].get(
            "action_distribution", "gaussian"
        ),
        "_obs_type": obs_type,
    }


# Param key remapping (Issue 4 from GUIDE2LOAD_MODELS.md)
_PARAM_KEY_REMAP = {
    "perceiver_attention": "lq_attention",
    "mgail_attention": "lq_attention",
}


def _remap_dict_keys(d, old_name, new_name):
    """Recursively rename keys in a nested dict."""
    if isinstance(d, dict):
        return {
            (new_name if k == old_name else k): _remap_dict_keys(v, old_name, new_name)
            for k, v in d.items()
        }
    return d


def _remap_param_keys(params):
    """Apply all known param key remappings."""
    for old_key, new_key in _PARAM_KEY_REMAP.items():
        needs_remap = False
        for path, _ in jax.tree_util.tree_leaves_with_path(params):
            if any(old_key in str(p) for p in path):
                needs_remap = True
                break
        if needs_remap:
            print(f"  [FIX] Param key: {old_key} → {new_key}")
            params = _remap_dict_keys(params, old_key, new_key)
    return params


def _build_sac_policy(run_dir: str):
    """Build a deterministic SAC policy from a V-Max run directory."""
    from vmax.agents.learning.reinforcement.sac import sac_factory
    from waymax import dynamics

    cfg = _load_hydra_config(run_dir)
    network_config = _build_network_config(cfg)
    obs_cfg = cfg.get("observation_config", {})
    termination_keys = cfg.get("termination_keys", ["offroad", "overlap"])
    obs_type = network_config.get("_obs_type", "vec")
    # reward_type="linear" — same as training. reward_config uses env default
    # (do NOT read reward_config from hydra; its nested format differs from env API)

    # Build env to get observation + action sizes (mirrors eval_cbm.py exactly)
    env = make_env_for_evaluation(
        max_num_objects=64,
        dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
        sdc_paths_from_data=True,
        observation_type=obs_type,
        observation_config=obs_cfg,
        reward_type="linear",
        termination_keys=termination_keys,
    )
    obs_size = env.observation_spec()      # returns int directly
    action_size = env.action_spec().data.shape[0]
    unflatten_fn = env.get_wrapper_attr("features_extractor").unflatten_features

    # Build network
    network = sac_factory.make_networks(
        observation_size=obs_size,
        action_size=action_size,
        unflatten_fn=unflatten_fn,
        learning_rate=1e-4,  # not used for inference
        network_config=network_config,
    )

    # Load params + fix param key mismatch (Issue 4 from GUIDE2LOAD_MODELS)
    model_path = Path(run_dir) / "model" / "model_final.pkl"
    params = load_params(str(model_path))
    policy_params = _remap_param_keys(params.policy)

    # Deterministic policy
    make_policy = sac_factory.make_inference_fn(network)
    policy = make_policy(policy_params, deterministic=True)

    return env, policy, termination_keys


def _build_cbm_policy(cfg: dict):
    """Build a deterministic CBM policy from model_registry config."""
    import cbm_v1.cbm_sac_factory as cbm_factory
    from cbm_v1.config import CBMConfig
    from waymax import dynamics

    pretrained_dir = cfg["pretrained_dir"]
    pretrained_cfg = _load_hydra_config(pretrained_dir)
    network_config = _build_network_config(pretrained_cfg)
    obs_cfg = pretrained_cfg.get("observation_config", {})
    termination_keys = pretrained_cfg.get("termination_keys", ["offroad", "overlap"])
    obs_type = network_config.get("_obs_type", "vec")

    env = make_env_for_evaluation(
        max_num_objects=64,
        dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
        sdc_paths_from_data=True,
        observation_type=obs_type,
        observation_config=obs_cfg,
        reward_type="linear",
        termination_keys=termination_keys,
    )
    obs_size = env.observation_spec()      # int
    action_size = env.action_spec().data.shape[0]
    unflatten_fn = env.get_wrapper_attr("features_extractor").unflatten_features

    cbm_config = CBMConfig(
        mode="frozen",
        num_concepts=cfg.get("num_concepts", 11),
        concept_phases=tuple(cfg.get("concept_phases", [1, 2])),
    )

    network = cbm_factory.make_networks(
        observation_size=obs_size,
        action_size=action_size,
        unflatten_fn=unflatten_fn,
        learning_rate=1e-4,
        network_config=network_config,
        cbm_config=cbm_config,
    )

    params = load_params(cfg["checkpoint"])
    make_policy = cbm_factory.make_inference_fn(network)
    policy = make_policy(params.policy, deterministic=True)

    return env, policy, termination_keys


# ── Public API ────────────────────────────────────────────────────────────────

def run_rollout(
    model_key: str,
    data_path: str,
    scenario_idx: int = 0,
    num_steps: int = 80,
) -> ScenarioData:
    """Run a single closed-loop episode and return structured numpy data.

    Parameters
    ----------
    model_key    : Key from MODEL_REGISTRY (e.g. "SAC Baseline — WOMD seed 42").
    data_path    : Path to the WOMD TFRecord file.
    scenario_idx : Which scenario to use from the dataset (0-indexed).
    num_steps    : Number of simulation steps (default 80 = one Waymax episode).

    Returns
    -------
    ScenarioData with ego trajectory, agent positions, and per-frame states.
    """
    if model_key not in MODEL_REGISTRY:
        raise KeyError(
            f"Model '{model_key}' not in registry. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    cfg = MODEL_REGISTRY[model_key]
    model_type = cfg["type"]

    print(f"[rollout_engine] Building policy for: {model_key}")
    if model_type == "sac":
        env, policy, termination_keys = _build_sac_policy(cfg["run_dir"])
    elif model_type == "cbm":
        env, policy, termination_keys = _build_cbm_policy(cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # ── Load exactly one scenario using O(1) memory ──────────────────────
    print(f"[rollout_engine] Loading scenario {scenario_idx}…")
    data_gen = make_data_generator(
        path=data_path,
        max_num_objects=64,
        include_sdc_paths=True,
        batch_dims=(1,),  # Load one at a time to avoid O(N) memory explosion
        seed=0,
        repeat=True,
    )
    
    # Fast-forward the iterator
    for _ in range(scenario_idx):
        next(data_gen)
        
    scenario = next(data_gen)

    # ── cuSolver warmup (prevents crash on first jit) ────────────────────
    _dummy = jnp.linalg.solve(jnp.eye(4, dtype=jnp.float32), jnp.ones(4, dtype=jnp.float32))
    jax.block_until_ready(_dummy)

    # ── Reset environment ───────────────────────────────────────────────────
    rng = jax.random.PRNGKey(0)
    rng, rk = jax.random.split(rng)
    reset_keys = jax.random.split(rk, 1)
    env_state = jax.jit(env.reset)(scenario, reset_keys)

    # ── JAX Lax Scan Loop (Massive Performance Improvement) ──────────────────
    def step_fn(state, _):
        obs = state.observation
        raw_action, _ = policy(obs, None)
        action = wdatatypes.Action(
            data=raw_action,
            valid=jnp.ones((*raw_action.shape[:-1], 1), dtype=jnp.bool_),
        )
        next_state = env.step(state, action)
        return next_state, next_state

    print(f"[rollout_engine] Running {num_steps}-step closed-loop episode via jax.lax.scan…")
    
    fast_step = jax.jit(lambda state: jax.lax.scan(step_fn, state, None, length=num_steps))
    final_env_state, stacked_env_states = fast_step(env_state)
    jax.block_until_ready(final_env_state.observation)

    print("[rollout_engine] Copying states back to CPU...")
    # Extract batch dim 0 across the entire stacked sequence (T, ...)
    waymax_states = jax.tree_util.tree_map(lambda x: x[:, 0], stacked_env_states.state)
    waymax_states_cpu = jax.device_get(waymax_states)
    
    rewards = np.array(jax.device_get(stacked_env_states.reward)).squeeze()  # → (T,)
    dones = np.array(jax.device_get(stacked_env_states.done)).squeeze()      # → (T,)

    # ── Extract to pure numpy / lists ────────────────────────────────────────
    frame_states = []
    ego_xy_list = []
    ego_yaw_list = []
    agents_xy_list = []
    agents_valid_list = []

    traj = waymax_states_cpu.sim_trajectory
    is_sdc = np.array(waymax_states_cpu.object_metadata.is_sdc) # shape (T, A)
    t_idx = np.array(waymax_states_cpu.timestep)                # shape (T,)

    for step in range(num_steps):
        # Reconstruct the SimulatorState object for this frame
        frame_state = jax.tree_util.tree_map(lambda x: x[step], waymax_states_cpu)
        frame_states.append(frame_state)
        
        curr_t = int(t_idx[step])
        # SDC mask for this frame (it should be constant, but indexing by step is safe)
        sdc_mask = is_sdc[step]
        
        # SDC features at current timestep
        sdc_x = float(np.array(traj.x)[step, sdc_mask, curr_t][0])
        sdc_y = float(np.array(traj.y)[step, sdc_mask, curr_t][0])
        sdc_yaw = float(np.array(traj.yaw)[step, sdc_mask, curr_t][0])
        ego_xy_list.append([sdc_x, sdc_y])
        ego_yaw_list.append(sdc_yaw)
        
        # All agents features at current timestep
        all_x = np.array(traj.x)[step, :, curr_t]
        all_y = np.array(traj.y)[step, :, curr_t]
        all_valid = np.array(traj.valid)[step, :, curr_t]
        agents_xy_list.append(np.stack([all_x, all_y], axis=-1))
        agents_valid_list.append(all_valid)

    agents_types = np.array(frame_states[0].object_metadata.object_types)

    print(f"[rollout_engine] Done. {len(frame_states)} frames collected.")
    return ScenarioData(
        ego_xy=np.array(ego_xy_list),
        ego_yaw=np.array(ego_yaw_list),
        agents_xy=np.array(agents_xy_list),
        agents_valid=np.array(agents_valid_list),
        agents_types=agents_types,
        frame_states=frame_states,
        rewards=rewards,
        dones=dones,
        model_key=model_key,
        scenario_idx=scenario_idx,
    )
