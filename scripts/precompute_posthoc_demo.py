#!/usr/bin/env python3
"""Offline precompute script for the Post-hoc XAI demo.

Produces (for each model × scenario pair):
  platform_cache/{slug}/scenario_{idx:04d}_artifact.pkl      PlatformScenarioArtifact
  platform_cache/{slug}/scenario_{idx:04d}_attr_{method}.pkl list[Attribution]  (length T)
  platform_cache/{slug}/scenario_{idx:04d}_attention.pkl     list[dict]         (length T)
  platform_cache/{slug}/scenario_{idx:04d}_frames.pkl        list[np.ndarray]   (length T, BEV frames)

Usage
-----
    python scripts/precompute_posthoc_demo.py
    python scripts/precompute_posthoc_demo.py --overwrite
    python scripts/precompute_posthoc_demo.py --no-attention
    python scripts/precompute_posthoc_demo.py --scenarios 0 1 2
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

# ── Path bootstrap (same order as platform/__init__.py) ──────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_CBM_ROOT = _PROJECT_ROOT / "cbm"
_POSTHOC_ROOT = _PROJECT_ROOT / "post-hoc-xai"

for _p in [str(_PROJECT_ROOT), str(_CBM_ROOT), str(_POSTHOC_ROOT), str(_CBM_ROOT / "V-Max")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Override stdlib `platform` in sys.modules with our package
import importlib.util as _ilu
_pkg_init = _PROJECT_ROOT / "platform" / "__init__.py"
_spec = _ilu.spec_from_file_location(
    "platform", str(_pkg_init),
    submodule_search_locations=[str(_PROJECT_ROOT / "platform")],
)
_mod = _ilu.module_from_spec(_spec)
sys.modules["platform"] = _mod
_spec.loader.exec_module(_mod)

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# ── Imports (heavy; JAX/Flax/Waymax load here) ────────────────────────────────
import numpy as np
import jax
import jax.numpy as jnp
from waymax import datatypes as wdatatypes

import platform  # triggers platform path setup
from platform.shared.contracts import PlatformScenarioArtifact
from platform.shared.model_catalog import PLATFORM_MODELS
from platform.shared.scenario_store import save_artifact
from platform.posthoc.adapter import (
    precompute_attribution_series,
    precompute_attention_series,
    load_explainable_model,
)

# ── Demo constants ─────────────────────────────────────────────────────────────

DATA_PATH = str(_CBM_ROOT / "data" / "training.tfrecord")

# Primary Perceiver models for the curated demo
DEMO_MODEL_KEYS = [
    "SAC Perceiver — WOMD seed 42",
    "SAC Perceiver Complete — WOMD seed 42",
]

DEMO_SCENARIOS = [1, 2, 3]

DEMO_METHODS = [
    "integrated_gradients",
    "feature_ablation",
    "gradient_x_input",
]

# ── Internal rollout helpers (mirror rollout_engine, but capture observations) ─

def _load_hydra_config(run_dir: str) -> dict:
    import yaml
    cfg_path = Path(run_dir) / ".hydra" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _build_network_config(pretrained_cfg: dict) -> dict:
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


_PARAM_KEY_REMAP = {
    "perceiver_attention": "lq_attention",
    "mgail_attention": "lq_attention",
}


def _remap_dict_keys(d, old_name, new_name):
    if isinstance(d, dict):
        return {
            (new_name if k == old_name else k): _remap_dict_keys(v, old_name, new_name)
            for k, v in d.items()
        }
    return d


def _remap_param_keys(params):
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


def _build_env_and_policy(run_dir: str):
    """Return (env, policy_fn, termination_keys) for a SAC run directory."""
    from vmax.agents.learning.reinforcement.sac import sac_factory
    from vmax.scripts.evaluate.utils import load_params
    from vmax.simulator import make_env_for_evaluation
    from waymax import dynamics

    cfg = _load_hydra_config(run_dir)
    network_config = _build_network_config(cfg)
    obs_cfg = cfg.get("observation_config", {})
    termination_keys = cfg.get("termination_keys", ["offroad", "overlap"])
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
    obs_size = env.observation_spec()
    action_size = env.action_spec().data.shape[0]
    unflatten_fn = env.get_wrapper_attr("features_extractor").unflatten_features

    network = sac_factory.make_networks(
        observation_size=obs_size,
        action_size=action_size,
        unflatten_fn=unflatten_fn,
        learning_rate=1e-4,
        network_config=network_config,
    )
    model_path = Path(run_dir) / "model" / "model_final.pkl"
    params = load_params(str(model_path))
    policy_params = _remap_param_keys(params.policy)
    make_policy = sac_factory.make_inference_fn(network)
    policy_fn = make_policy(policy_params, deterministic=True)

    return env, policy_fn, termination_keys


def _run_episode_and_capture_obs(
    env,
    policy_fn,
    data_path: str,
    scenario_idx: int,
    num_steps: int = 80,
) -> tuple[object, np.ndarray]:
    """Run closed-loop episode and return (ScenarioData, raw_observations).

    raw_observations has shape (T, obs_size) — flat observation fed to the
    policy at each step.

    Returns
    -------
    (ScenarioData, raw_observations)
    """
    from vmax.simulator import make_data_generator
    from bev_visualizer.rollout_engine import ScenarioData

    # Patch Waymax metric registration to be idempotent
    import waymax.metrics as _wm_metrics
    _original_register = _wm_metrics.register_metric
    def _safe_register(name, cls):
        try:
            _original_register(name, cls)
        except Exception:
            pass
    _wm_metrics.register_metric = _safe_register

    print(f"  Loading scenario {scenario_idx}…")
    data_gen = make_data_generator(
        path=data_path,
        max_num_objects=64,
        include_sdc_paths=True,
        batch_dims=(1,),
        seed=0,
        repeat=True,
    )
    for _ in range(scenario_idx):
        next(data_gen)
    scenario = next(data_gen)

    # Warmup
    _dummy = jnp.linalg.solve(jnp.eye(4, dtype=jnp.float32), jnp.ones(4, dtype=jnp.float32))
    jax.block_until_ready(_dummy)

    rng = jax.random.PRNGKey(0)
    rng, rk = jax.random.split(rng)
    reset_keys = jax.random.split(rk, 1)
    env_state = jax.jit(env.reset)(scenario, reset_keys)

    # Modified scan: stacks (env_state, observation) at each step
    def step_fn(state, _):
        obs = state.observation                      # capture before action
        raw_action, _ = policy_fn(obs, None)
        action = wdatatypes.Action(
            data=raw_action,
            valid=jnp.ones((*raw_action.shape[:-1], 1), dtype=jnp.bool_),
        )
        next_state = env.step(state, action)
        return next_state, (next_state, obs)          # ← obs captured here

    print(f"  Running {num_steps}-step scan…")
    fast_step = jax.jit(
        lambda s: jax.lax.scan(step_fn, s, None, length=num_steps)
    )
    _final_state, (stacked_env_states, stacked_obs) = fast_step(env_state)
    jax.block_until_ready(_final_state.observation)

    # raw_observations: drop batch dim → (T, obs_size)
    raw_obs_np = np.array(jax.device_get(stacked_obs))  # (T, 1, obs_size)
    if raw_obs_np.ndim == 3:
        raw_obs_np = raw_obs_np[:, 0, :]  # (T, obs_size)

    # Extract ScenarioData the same way as rollout_engine
    waymax_states_cpu = jax.device_get(
        jax.tree_util.tree_map(lambda x: x[:, 0], stacked_env_states.state)
    )
    rewards = np.array(jax.device_get(stacked_env_states.reward)).squeeze()
    dones = np.array(jax.device_get(stacked_env_states.done)).squeeze()

    frame_states = []
    ego_xy_list, ego_yaw_list = [], []
    agents_xy_list, agents_valid_list = [], []

    traj = waymax_states_cpu.sim_trajectory
    is_sdc = np.array(waymax_states_cpu.object_metadata.is_sdc)
    t_idx = np.array(waymax_states_cpu.timestep)

    for t in range(num_steps):
        frame_state = jax.tree_util.tree_map(lambda x: x[t], waymax_states_cpu)
        frame_states.append(frame_state)
        curr_t = int(t_idx[t])
        sdc_mask = is_sdc[t]
        ego_xy_list.append([
            float(np.array(traj.x)[t, sdc_mask, curr_t][0]),
            float(np.array(traj.y)[t, sdc_mask, curr_t][0]),
        ])
        ego_yaw_list.append(float(np.array(traj.yaw)[t, sdc_mask, curr_t][0]))
        all_x = np.array(traj.x)[t, :, curr_t]
        all_y = np.array(traj.y)[t, :, curr_t]
        agents_xy_list.append(np.stack([all_x, all_y], axis=-1))
        agents_valid_list.append(np.array(traj.valid)[t, :, curr_t])

    scenario_data = ScenarioData(
        ego_xy=np.array(ego_xy_list),
        ego_yaw=np.array(ego_yaw_list),
        agents_xy=np.array(agents_xy_list),
        agents_valid=np.array(agents_valid_list),
        agents_types=np.array(frame_states[0].object_metadata.object_types),
        frame_states=frame_states,
        rewards=rewards,
        dones=dones,
        model_key="",  # filled by caller
        scenario_idx=scenario_idx,
    )
    return scenario_data, raw_obs_np


# ── BEV frame pre-rendering ────────────────────────────────────────────────────

def _prerender_bev_frames(
    artifact: PlatformScenarioArtifact,
    out_path: Path,
    overwrite: bool = False,
) -> None:
    if out_path.exists() and not overwrite:
        print(f"  BEV frames already cached: {out_path.name} — skip")
        return

    from bev_visualizer.bev_renderer import render_frame

    states = artifact.scenario_data.frame_states
    total = len(states)
    frames = []
    for step, state in enumerate(states):
        frames.append(render_frame(state, overlay_fn=None, step=step))
        if (step + 1) % 20 == 0:
            print(f"  BEV render: {step + 1}/{total}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(frames, fh)
    print(f"  Saved {total} BEV frames → {out_path}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    scenarios: list[int] = args.scenarios
    overwrite: bool = args.overwrite
    skip_attention: bool = args.no_attention

    print(f"Demo models   : {DEMO_MODEL_KEYS}")
    print(f"Scenarios     : {scenarios}")
    print(f"Methods       : {DEMO_METHODS}")
    print(f"Overwrite     : {overwrite}")
    print(f"Attention     : {not skip_attention}")
    print()

    for model_key in DEMO_MODEL_KEYS:
        if model_key not in PLATFORM_MODELS:
            print(f"[SKIP] {model_key!r} not in PLATFORM_MODELS catalog — skipping.")
            continue

        entry = PLATFORM_MODELS[model_key]
        if not entry.exists_on_disk:
            print(f"[SKIP] {model_key!r} model directory not found on disk — skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Model: {model_key}")
        print(f"{'='*60}")

        # Load XAI model (once per model — reused for all scenarios/methods)
        print("Loading ExplainableModel (JAX/Flax)…")
        xai_model = load_explainable_model(entry)
        print("  OK")

        # Grab env+policy from xai_model internals to avoid a second heavy load
        # xai_model._loaded is a LoadedVMAXModel
        loaded = xai_model._loaded
        env = loaded.env
        policy_fn = loaded.policy_fn

        cache_dir = _PROJECT_ROOT / "platform_cache" / entry.cache_slug
        cache_dir.mkdir(parents=True, exist_ok=True)

        for scenario_idx in scenarios:
            print(f"\n  --- Scenario {scenario_idx} ---")
            artifact_path = cache_dir / f"scenario_{scenario_idx:04d}_artifact.pkl"

            if artifact_path.exists() and not overwrite:
                print(f"  Artifact already exists — loading from cache.")
                with open(artifact_path, "rb") as fh:
                    artifact = pickle.load(fh)
            else:
                scenario_data, raw_obs = _run_episode_and_capture_obs(
                    env, policy_fn, DATA_PATH, scenario_idx
                )
                scenario_data.model_key = model_key

                artifact = PlatformScenarioArtifact(
                    scenario_data=scenario_data,
                    model_key=model_key,
                    scenario_idx=scenario_idx,
                    raw_observations=raw_obs,
                    interesting_timesteps=None,
                    notes=f"{entry.description} — scenario {scenario_idx}",
                )
                save_artifact(artifact)
                print(f"  Saved artifact: {artifact_path.name}")

            # Attributions
            for method in DEMO_METHODS:
                attr_path = cache_dir / f"scenario_{scenario_idx:04d}_attr_{method}.pkl"
                if attr_path.exists() and not overwrite:
                    print(f"  Attribution '{method}' cached — skip")
                    continue
                print(f"  Computing attribution series: {method}…")
                try:
                    out = precompute_attribution_series(
                        artifact, method, xai_model, overwrite=overwrite
                    )
                    print(f"  Saved → {out.name}")
                except Exception as exc:
                    print(f"  [ERROR] {method}: {exc}")

            # Attention
            if not skip_attention and entry.has_attention:
                attn_path = cache_dir / f"scenario_{scenario_idx:04d}_attention.pkl"
                if attn_path.exists() and not overwrite:
                    print(f"  Attention cached — skip")
                else:
                    print(f"  Computing attention series…")
                    try:
                        out = precompute_attention_series(
                            artifact, xai_model, overwrite=overwrite
                        )
                        if out:
                            print(f"  Saved → {out.name}")
                    except Exception as exc:
                        print(f"  [ERROR] attention: {exc}")

            # BEV frames
            frames_path = cache_dir / f"scenario_{scenario_idx:04d}_frames.pkl"
            _prerender_bev_frames(artifact, frames_path, overwrite=overwrite)

    print("\n\nPrecompute complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute Post-hoc XAI demo cache.")
    parser.add_argument(
        "--scenarios",
        type=int,
        nargs="+",
        default=DEMO_SCENARIOS,
        metavar="IDX",
        help=f"Scenario indices to process (default: {DEMO_SCENARIOS})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-compute and overwrite existing cache files.",
    )
    parser.add_argument(
        "--no-attention",
        action="store_true",
        help="Skip attention series computation.",
    )
    args = parser.parse_args()
    main(args)
