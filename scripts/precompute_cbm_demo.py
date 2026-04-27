#!/usr/bin/env python3
"""Precompute CBM demo cache for the platform.

For each curated scenario (from cbm/curated_scenarios.json) runs the CBM
rollout, captures BEV frame states, concept predictions and ground-truth
concept values, then saves everything as PlatformScenarioArtifact pickles.

Outputs (per scenario):
    platform_cache/CBM_Scratch_V2/scenario_{idx:04d}_artifact.pkl
    platform_cache/CBM_Scratch_V2/scenario_{idx:04d}_frames.pkl

Usage:
    # from /home/med1e/platform_fyp
    conda activate vmax
    python scripts/precompute_cbm_demo.py --data cbm/data/training.tfrecord
    python scripts/precompute_cbm_demo.py --data /path/to/validation.tfrecord
    python scripts/precompute_cbm_demo.py --data cbm/data/training.tfrecord --top_k 3
    python scripts/precompute_cbm_demo.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

# ── Path bootstrap ────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_CBM_ROOT     = _PROJECT_ROOT / "cbm"

for _p in [
    str(_PROJECT_ROOT),
    str(_CBM_ROOT),
    str(_CBM_ROOT / "V-Max"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Override stdlib `platform` with our package
import importlib.util as _ilu
_pkg_init = _PROJECT_ROOT / "platform" / "__init__.py"
_spec = _ilu.spec_from_file_location(
    "platform", str(_pkg_init),
    submodule_search_locations=[str(_PROJECT_ROOT / "platform")],
)
_mod = _ilu.module_from_spec(_spec)
sys.modules["platform"] = _mod
_spec.loader.exec_module(_mod)

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# ── Heavy imports ─────────────────────────────────────────────────────
import numpy as np
import jax
import jax.numpy as jnp
import yaml
from waymax import datatypes as wdatatypes, dynamics
from waymax import metrics as _wm

# Patch Waymax metric registration to be idempotent
_orig_register = _wm.register_metric
def _safe_register(name, cls):
    try: _orig_register(name, cls)
    except Exception: pass
_wm.register_metric = _safe_register

from vmax.simulator import make_env_for_evaluation, make_data_generator
from vmax.scripts.evaluate.utils import load_params

from concepts.types import ObservationConfig
from concepts.adapters import observation_to_concept_input
from concepts.registry import extract_all_concepts

from cbm_v1.config import CBMConfig
import cbm_v1.cbm_sac_factory as cbm_factory

import platform  # our package
from platform.shared.contracts import PlatformScenarioArtifact
from platform.shared.scenario_store import save_artifact
from bev_visualizer.rollout_engine import ScenarioData

# ── Constants ─────────────────────────────────────────────────────────
CBM_MODEL_KEY  = "CBM Scratch V2 — λ=0.5"
CBM_CHECKPOINT = str(_CBM_ROOT / "cbm_scratch_v2_lambda05" / "checkpoints" / "model_final.pkl")
CBM_YAML       = str(_CBM_ROOT / "cbm_v1" / "config_womd_scratch.yaml")
NUM_CONCEPTS   = 15
CONCEPT_PHASES = (1, 2, 3)
NUM_STEPS      = 80

_PLATFORM_CACHE = _PROJECT_ROOT / "platform_cache" / "CBM_Scratch_V2"


# ── Config loading (scratch model — no hydra) ─────────────────────────

def _load_scratch_config(yaml_path: str):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    obs_cfg  = cfg.get("observation_config", {})
    term     = cfg.get("termination_keys", ["offroad", "overlap", "run_red_light"])
    net_cfg  = cfg["network_config"]
    obs_type = {"road": "vec", "lane": "vec"}.get(net_cfg.get("_obs_type", "vec"), "vec")
    return net_cfg, obs_cfg, term, obs_type


# ── Build environment + CBM policy ───────────────────────────────────

def _build(yaml_path: str, checkpoint: str):
    net_cfg, obs_cfg, term, obs_type = _load_scratch_config(yaml_path)

    env = make_env_for_evaluation(
        max_num_objects=64,
        dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
        sdc_paths_from_data=True,
        observation_type=obs_type,
        observation_config=obs_cfg,
        termination_keys=term,
        noisy_init=False,
    )
    obs_size   = env.observation_spec()
    action_size = env.action_spec().data.shape[0]
    unflatten_fn = env.get_wrapper_attr("features_extractor").unflatten_features

    cbm_config = CBMConfig(
        mode="scratch",
        num_concepts=NUM_CONCEPTS,
        concept_phases=CONCEPT_PHASES,
    )
    network = cbm_factory.make_networks(
        observation_size=obs_size,
        action_size=action_size,
        unflatten_fn=unflatten_fn,
        learning_rate=1e-4,
        network_config=net_cfg,
        cbm_config=cbm_config,
    )
    params = load_params(checkpoint)
    policy_fn = cbm_factory.make_inference_fn(network)(params.policy, deterministic=True)

    concept_config = ObservationConfig(
        obs_past_num_steps=obs_cfg.get("obs_past_num_steps", 5),
        num_closest_objects=obs_cfg.get("objects", {}).get("num_closest_objects", 8),
        roadgraph_top_k=obs_cfg.get("roadgraphs", {}).get("roadgraph_top_k", 200),
        num_closest_traffic_lights=obs_cfg.get("traffic_lights", {}).get("num_closest_traffic_lights", 5),
        num_target_path_points=obs_cfg.get("path_target", {}).get("num_points", 10),
        max_meters=obs_cfg.get("roadgraphs", {}).get("max_meters", 70),
    )

    return env, policy_fn, cbm_config, concept_config, unflatten_fn


# ── Run rollout and capture obs ───────────────────────────────────────

def _run_rollout(env, policy_fn, data_path: str, local_idx: int):
    """Run rollout for scenario at position local_idx in data_path.

    Returns (ScenarioData, raw_obs_np) where raw_obs_np is (T, obs_size).
    """
    print(f"    Loading scenario at position {local_idx}…")
    data_gen = make_data_generator(
        path=data_path,
        max_num_objects=64,
        include_sdc_paths=True,
        batch_dims=(1,),
        seed=0,
        repeat=True,
    )
    for _ in range(local_idx):
        next(data_gen)
    scenario = next(data_gen)

    rng = jax.random.PRNGKey(0)
    rng, rk = jax.random.split(rng)
    env_state = jax.jit(env.reset)(scenario, jax.random.split(rk, 1))

    def step_fn(state, _):
        obs = state.observation
        raw_action, _ = policy_fn(obs, None)
        action = wdatatypes.Action(
            data=raw_action,
            valid=jnp.ones((*raw_action.shape[:-1], 1), dtype=jnp.bool_),
        )
        next_state = env.step(state, action)
        return next_state, (next_state, obs)

    print(f"    Running {NUM_STEPS}-step scan…")
    _final, (stacked_states, stacked_obs) = jax.jit(
        lambda s: jax.lax.scan(step_fn, s, None, length=NUM_STEPS)
    )(env_state)
    jax.block_until_ready(_final.observation)

    # raw_obs: (T, 1, obs_size) → (T, obs_size)
    raw_obs_np = np.array(jax.device_get(stacked_obs))
    if raw_obs_np.ndim == 3:
        raw_obs_np = raw_obs_np[:, 0, :]

    # Build ScenarioData
    wstates  = jax.device_get(
        jax.tree_util.tree_map(lambda x: x[:, 0], stacked_states.state)
    )
    rewards = np.array(jax.device_get(stacked_states.reward)).squeeze()
    dones   = np.array(jax.device_get(stacked_states.done)).squeeze()

    traj   = wstates.sim_trajectory
    is_sdc = np.array(wstates.object_metadata.is_sdc)
    t_idx  = np.array(wstates.timestep)

    frame_states, ego_xy, ego_yaw, agents_xy, agents_valid = [], [], [], [], []
    for t in range(NUM_STEPS):
        fs = jax.tree_util.tree_map(lambda x: x[t], wstates)
        frame_states.append(fs)
        curr_t = int(t_idx[t])
        sm     = is_sdc[t]
        ego_xy.append([float(np.array(traj.x)[t, sm, curr_t][0]),
                        float(np.array(traj.y)[t, sm, curr_t][0])])
        ego_yaw.append(float(np.array(traj.yaw)[t, sm, curr_t][0]))
        agents_xy.append(np.stack([np.array(traj.x)[t, :, curr_t],
                                    np.array(traj.y)[t, :, curr_t]], axis=-1))
        agents_valid.append(np.array(traj.valid)[t, :, curr_t])

    scenario_data = ScenarioData(
        ego_xy=np.array(ego_xy),
        ego_yaw=np.array(ego_yaw),
        agents_xy=np.array(agents_xy),
        agents_valid=np.array(agents_valid),
        agents_types=np.array(frame_states[0].object_metadata.object_types),
        frame_states=frame_states,
        rewards=rewards,
        dones=dones,
        model_key=CBM_MODEL_KEY,
        scenario_idx=local_idx,
    )
    return scenario_data, raw_obs_np


# ── Compute interesting timesteps per archetype ───────────────────────

def _interesting_timesteps(arch: str, pred_c, true_c, valid_c, cnames) -> list[int]:
    """Return timestep indices that are 'interesting' for this archetype.

    These are highlighted in the BEV and on the concept timeline.
    """
    def ci(name):
        return cnames.index(name) if name in cnames else None

    steps = np.arange(80)

    if arch == "red_light_stop":
        idx = ci("traffic_light_red")
        if idx is None:
            return []
        active = (true_c[:, idx] > 0.8) & valid_c[:, idx]
        return steps[active].tolist()

    elif arch == "ttc_success":
        idx = ci("ttc_lead_vehicle")
        if idx is None:
            return []
        danger = (true_c[:, idx] < 0.35) & valid_c[:, idx]
        return steps[danger].tolist()

    elif arch == "curvature_nav":
        idx = ci("path_curvature_max")
        if idx is None:
            return []
        high = true_c[:, idx] > 0.5
        return steps[high].tolist()

    elif arch == "concept_failure":
        # Steps where prediction error is highest across key concepts
        key = [n for n in ["traffic_light_red", "dist_nearest_object", "ttc_lead_vehicle"]
               if ci(n) is not None]
        if not key:
            return []
        errors = np.max(
            np.abs(pred_c[:, [ci(n) for n in key]] - true_c[:, [ci(n) for n in key]]),
            axis=1,
        )
        # Top-10 error steps
        top = np.argsort(errors)[::-1][:10]
        return sorted(top.tolist())

    return []


# ── Compute concept predictions and ground truth ──────────────────────

@jax.jit
def _pred_concepts(obs):
    _, concepts = cbm_factory._cbm_policy_module.apply(
        _POLICY_PARAMS, obs,
        method=cbm_factory._cbm_policy_module.encode_and_predict_concepts,
    )
    return concepts

_POLICY_PARAMS = None   # set in main after model load
_UNFLATTEN_FN  = None
_CONCEPT_CFG   = None
_CBM_CFG       = None


def _compute_concepts(raw_obs_np):
    """Batch-compute pred and true concepts for all T timesteps.

    Returns pred (T,15), true (T,15), valid (T,15).
    """
    T = raw_obs_np.shape[0]
    preds, trues, valids = [], [], []
    BATCH = 16
    for start in range(0, T, BATCH):
        obs_b = jnp.array(raw_obs_np[start:start + BATCH])
        preds.append(np.array(_pred_concepts(obs_b)))
        inp = observation_to_concept_input(obs_b, _UNFLATTEN_FN, _CONCEPT_CFG)
        out = extract_all_concepts(inp, phases=CONCEPT_PHASES)
        trues.append(np.array(out.normalized))
        valids.append(np.array(out.valid))
    return (np.concatenate(preds,  axis=0),
            np.concatenate(trues,  axis=0),
            np.concatenate(valids, axis=0))


# ── BEV frame pre-rendering ───────────────────────────────────────────

def _render_bev_frames(artifact, frames_path: Path, overwrite: bool):
    if frames_path.exists() and not overwrite:
        print(f"    BEV frames cached — skip")
        return
    from bev_visualizer.bev_renderer import render_frame

    interesting = set(artifact.interesting_timesteps or [])
    arch  = (artifact.metadata or {}).get("archetype", "")
    color = {
        "red_light_stop":  "#E53935",
        "ttc_success":     "#FB8C00",
        "curvature_nav":   "#43A047",
        "concept_failure": "#8E24AA",
    }.get(arch, "#1565C0")

    def make_overlay(t):
        if t not in interesting:
            return None
        def overlay_fn(ax, step):
            ax.text(
                0.02, 0.97, "★ Key event",
                transform=ax.transAxes,
                fontsize=9, fontweight="bold", color="white",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc=color, alpha=0.85, ec="none"),
                zorder=10,
            )
        return overlay_fn

    frames = []
    T = len(artifact.scenario_data.frame_states)
    for t, state in enumerate(artifact.scenario_data.frame_states):
        frames.append(render_frame(state, overlay_fn=make_overlay(t), step=t))
        if (t + 1) % 20 == 0:
            print(f"    BEV: {t+1}/{T}")
    frames_path.parent.mkdir(parents=True, exist_ok=True)
    with open(frames_path, "wb") as fh:
        pickle.dump(frames, fh)
    print(f"    Saved {T} BEV frames → {frames_path.name}")


# ── Main ──────────────────────────────────────────────────────────────

def main(args):
    global _POLICY_PARAMS, _UNFLATTEN_FN, _CONCEPT_CFG, _CBM_CFG

    with open(_CBM_ROOT / "curated_scenarios.json") as f:
        index = json.load(f)

    # Collect scenarios to process (top_k per archetype, deduplicated)
    to_process = {}   # local_idx → {archetype, rank, score, scenario_idx, no_at_fault, progress}
    for arch_name, entries in index["archetypes"].items():
        for e in entries[:args.top_k]:
            li = e["local_idx"]
            if li not in to_process:
                to_process[li] = {
                    "local_idx":    li,
                    "scenario_idx": e["scenario_idx"],
                    "archetype":    arch_name,
                    "rank":         e["rank"],
                    "score":        e["score"],
                    "progress":     e["progress"],
                    "no_at_fault":  e["no_at_fault"],
                }

    print(f"Scenarios to precompute: {len(to_process)} unique")
    print(f"Data path: {args.data}")
    print(f"Checkpoint: {CBM_CHECKPOINT}")
    print()

    if not Path(CBM_CHECKPOINT).exists():
        print(f"ERROR: checkpoint not found: {CBM_CHECKPOINT}")
        sys.exit(1)

    # ── Build model ───────────────────────────────────────────────────
    print("Building CBM model…")
    env, policy_fn, cbm_config, concept_config, unflatten_fn = _build(
        CBM_YAML, CBM_CHECKPOINT
    )
    _POLICY_PARAMS = load_params(CBM_CHECKPOINT).policy
    _UNFLATTEN_FN  = unflatten_fn
    _CONCEPT_CFG   = concept_config
    _CBM_CFG       = cbm_config

    cnames = list(cbm_config.concept_names)
    print(f"  Concepts: {NUM_CONCEPTS} — {cnames}")

    # cuSolver warmup
    _d = jnp.linalg.solve(jnp.eye(4, dtype=jnp.float32), jnp.ones(4, dtype=jnp.float32))
    jax.block_until_ready(_d)
    print("  Model ready.\n")

    _PLATFORM_CACHE.mkdir(parents=True, exist_ok=True)

    # ── Process each scenario ─────────────────────────────────────────
    for i, entry in enumerate(to_process.values()):
        local_idx = entry["local_idx"]
        arch      = entry["archetype"]

        print(f"[{i+1}/{len(to_process)}] local_idx={local_idx}  archetype={arch}  "
              f"score={entry['score']:.3f}")

        artifact_path = _PLATFORM_CACHE / f"scenario_{local_idx:04d}_artifact.pkl"
        frames_path   = _PLATFORM_CACHE / f"scenario_{local_idx:04d}_frames.pkl"

        if artifact_path.exists() and not args.overwrite:
            print(f"  Artifact cached — skip (use --overwrite to redo)")
        else:
            scenario_data, raw_obs = _run_rollout(
                env, policy_fn, args.data, local_idx
            )
            print(f"  Computing concept predictions…")
            pred_c, true_c, valid_c = _compute_concepts(raw_obs)

            interesting = _interesting_timesteps(arch, pred_c, true_c, valid_c, cnames)
            print(f"  Interesting timesteps ({arch}): {interesting[:8]}{'…' if len(interesting) > 8 else ''}")

            artifact = PlatformScenarioArtifact(
                scenario_data=scenario_data,
                model_key=CBM_MODEL_KEY,
                scenario_idx=local_idx,
                raw_observations=raw_obs,
                interesting_timesteps=interesting if interesting else None,
                notes=(
                    f"{arch.replace('_', ' ').title()} — "
                    f"rank {entry['rank']} — score {entry['score']:.3f} — "
                    f"progress {entry['progress']:.3f}"
                ),
                metadata={
                    "pred_concepts":  pred_c,     # (80, 15) float32
                    "true_concepts":  true_c,
                    "valid_mask":     valid_c,
                    "concept_names":  cnames,
                    "archetype":      arch,
                    "archetype_rank": entry["rank"],
                    "archetype_score":entry["score"],
                    "scenario_idx":   entry["scenario_idx"],
                    "no_at_fault":    entry["no_at_fault"],
                    "progress":       entry["progress"],
                },
            )

            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            with open(artifact_path, "wb") as fh:
                pickle.dump(artifact, fh)
            print(f"  Saved artifact → {artifact_path.name}")

        # BEV frames
        if not artifact_path.exists():
            print(f"  Skipping BEV frames (no artifact)")
        else:
            if not frames_path.exists() or args.overwrite:
                with open(artifact_path, "rb") as fh:
                    artifact = pickle.load(fh)
                _render_bev_frames(artifact, frames_path, args.overwrite)
            else:
                print(f"  BEV frames cached — skip")

        print()

    print("=" * 50)
    print("Precompute complete.")
    print(f"Cache: {_PLATFORM_CACHE}")
    print(f"Files: {len(list(_PLATFORM_CACHE.glob('*.pkl')))} pkl files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute CBM demo cache.")
    parser.add_argument(
        "--data",
        default=str(_CBM_ROOT / "data" / "training.tfrecord"),
        help="Path to WOMD TFRecord (training or validation)",
    )
    parser.add_argument(
        "--top_k", type=int, default=3,
        help="Top K scenarios per archetype to process (default: 3)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-compute and overwrite existing cache files",
    )
    main(parser.parse_args())
