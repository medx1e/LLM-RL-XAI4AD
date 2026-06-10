#!/usr/bin/env python3
"""Precompute BEV artifacts for the LLM Narration tab.

This script is the first step of the narration-tab precompute pipeline.
It must be run BEFORE precompute_llm_narration.py (which generates JSON
reports and LLM narrations) because the narration tab's BEV player needs:

  1. PlatformScenarioArtifact  — contains the scenario rollout data and is
     used by the tab to determine num_steps (total timesteps T), which drives
     the slider range and the forward-fill of narration lists.

  2. Pre-rendered BEV frames  — the bird's-eye-view animation frames, one
     numpy array per timestep. Stored separately to make Streamlit loading
     instantaneous (no on-demand rendering at view time).

Both files are written to the shared platform cache that ALL tabs read:
  platform_cache/{model_slug}/scenario_{idx:04d}_artifact.pkl
  platform_cache/{model_slug}/scenario_{idx:04d}_frames.pkl

The script processes the exact (model, scenario) pairs declared in
platform/llm_narration.py: DRIVING_MODELS and SCENARIO_CATALOG.  There is
no overlap with precompute_posthoc_demo.py's scope (attributions/attention),
so both scripts can be run independently and their outputs coexist.

Workflow
--------
  Step 1 — run THIS script:
    python scripts/precompute_narration_artifacts.py

  Step 2 — run XAI reports and narrations:
    python scripts/precompute_llm_narration.py --phase reports
    python scripts/precompute_llm_narration.py --phase narrations

  Step 3 — launch the viewer:
    streamlit run app.py  →  "LLM Narration" tab

Usage
-----
    python scripts/precompute_narration_artifacts.py
    python scripts/precompute_narration_artifacts.py --scenarios 0 1 2
    python scripts/precompute_narration_artifacts.py --overwrite
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import sys
from pathlib import Path

# ── Path bootstrap (mirrors platform/__init__.py + precompute_posthoc_demo.py) ─
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_CBM_ROOT     = _PROJECT_ROOT / "cbm"
_POSTHOC_ROOT = _PROJECT_ROOT / "post-hoc-xai"

_paths = [str(_PROJECT_ROOT), str(_CBM_ROOT), str(_POSTHOC_ROOT)]
try:
    import vmax as _vmax
except ImportError:
    _paths.append(str(_CBM_ROOT / "V-Max"))

for _p in _paths:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Override stdlib `platform` with the project package (same trick as all
# other scripts in this directory).
_pkg_init = _PROJECT_ROOT / "platform" / "__init__.py"
_spec = importlib.util.spec_from_file_location(
    "platform", str(_pkg_init),
    submodule_search_locations=[str(_PROJECT_ROOT / "platform")],
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["platform"] = _mod
_spec.loader.exec_module(_mod)

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# ── Imports (JAX/Flax/Waymax initialise here) ────────────────────────────────
import jax                                       # noqa: E402
import jax.numpy as jnp                          # noqa: E402
import numpy as np                               # noqa: E402
from waymax import datatypes as wdatatypes        # noqa: E402

import platform                                  # noqa: E402  triggers setup_paths()
from platform.shared.contracts import PlatformScenarioArtifact   # noqa: E402
from platform.shared.model_catalog import PLATFORM_MODELS         # noqa: E402
from platform.shared.scenario_store import save_artifact          # noqa: E402
from platform.posthoc.adapter import load_explainable_model       # noqa: E402

# Narration-tab config: which models and scenarios to target.
from platform.llm_narration import DRIVING_MODELS, SCENARIO_CATALOG  # noqa: E402

# Default data path (same as precompute_posthoc_demo.py).
_DEFAULT_DATA_PATH = str(_CBM_ROOT / "data" / "training.tfrecord")

# Cache root where artifacts land (same tree other tabs read).
_PLATFORM_CACHE_ROOT = _PROJECT_ROOT / "platform_cache"


# =============================================================================
# EPISODE ROLLOUT
# =============================================================================

def _run_episode(
    env,
    policy_fn,
    data_path: str,
    scenario_idx: int,
    num_steps: int = 80,
) -> tuple[object, np.ndarray]:
    """Run a closed-loop episode and return (ScenarioData, raw_observations).

    Mirrors the same logic in precompute_posthoc_demo._run_episode_and_capture_obs
    so both scripts produce identical artifacts for the same (model, scenario).
    """
    from vmax.simulator import make_data_generator
    from bev_visualizer.rollout_engine import ScenarioData

    # Make Waymax metric registration idempotent (can be called multiple times).
    import waymax.metrics as _wm
    _orig = _wm.register_metric
    def _safe(name, cls):
        try: _orig(name, cls)
        except Exception: pass
    _wm.register_metric = _safe

    print(f"    Loading scenario {scenario_idx}…")
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

    # JAX warmup (avoids tracing overhead on the first scan).
    jax.block_until_ready(
        jnp.linalg.solve(jnp.eye(4, dtype=jnp.float32), jnp.ones(4, dtype=jnp.float32))
    )

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

    print(f"    Running {num_steps}-step scan…")
    final_state, (stacked_states, stacked_obs) = jax.jit(
        lambda s: jax.lax.scan(step_fn, s, None, length=num_steps)
    )(env_state)
    jax.block_until_ready(final_state.observation)

    raw_obs_np = np.array(jax.device_get(stacked_obs))
    if raw_obs_np.ndim == 3:
        raw_obs_np = raw_obs_np[:, 0, :]   # (T, obs_size)

    states_cpu = jax.device_get(
        jax.tree_util.tree_map(lambda x: x[:, 0], stacked_states.state)
    )
    rewards = np.array(jax.device_get(stacked_states.reward)).squeeze()
    dones   = np.array(jax.device_get(stacked_states.done)).squeeze()

    traj    = states_cpu.sim_trajectory
    is_sdc  = np.array(states_cpu.object_metadata.is_sdc)
    t_idx   = np.array(states_cpu.timestep)

    ego_xy_list, ego_yaw_list, agents_xy_list, agents_valid_list, frame_states = \
        [], [], [], [], []

    for t in range(num_steps):
        frame_state = jax.tree_util.tree_map(lambda x: x[t], states_cpu)
        frame_states.append(frame_state)
        ct = int(t_idx[t])
        mask = is_sdc[t]
        ego_xy_list.append([
            float(np.array(traj.x)[t, mask, ct][0]),
            float(np.array(traj.y)[t, mask, ct][0]),
        ])
        ego_yaw_list.append(float(np.array(traj.yaw)[t, mask, ct][0]))
        agents_xy_list.append(
            np.stack([np.array(traj.x)[t, :, ct], np.array(traj.y)[t, :, ct]], axis=-1)
        )
        agents_valid_list.append(np.array(traj.valid)[t, :, ct])

    scenario_data = ScenarioData(
        ego_xy=np.array(ego_xy_list),
        ego_yaw=np.array(ego_yaw_list),
        agents_xy=np.array(agents_xy_list),
        agents_valid=np.array(agents_valid_list),
        agents_types=np.array(frame_states[0].object_metadata.object_types),
        frame_states=frame_states,
        rewards=rewards,
        dones=dones,
        model_key="",      # filled by the caller
        scenario_idx=scenario_idx,
    )
    return scenario_data, raw_obs_np


# =============================================================================
# BEV FRAME PRE-RENDERING
# =============================================================================

def _prerender_bev_frames(
    artifact: PlatformScenarioArtifact,
    frames_path: Path,
    overwrite: bool = False,
) -> None:
    """Render all BEV frames to numpy arrays and pickle them to disk."""
    if frames_path.exists() and not overwrite:
        print(f"    BEV frames already cached — skip ({frames_path.name})")
        return

    from bev_visualizer.bev_renderer import render_frame

    states = artifact.scenario_data.frame_states
    total  = len(states)
    frames = []
    for step, state in enumerate(states):
        frames.append(render_frame(state, overlay_fn=None, step=step))
        if (step + 1) % 20 == 0:
            print(f"    BEV render: {step + 1}/{total}")

    frames_path.parent.mkdir(parents=True, exist_ok=True)
    with open(frames_path, "wb") as fh:
        pickle.dump(frames, fh)
    print(f"    Saved {total} BEV frames → {frames_path.name}")


# =============================================================================
# MAIN LOOP
# =============================================================================

def main(args: argparse.Namespace) -> None:
    scenarios: list[int] = args.scenarios
    overwrite: bool      = args.overwrite
    data_path: str       = args.data

    # Build the (display_label, platform_model_key) pairs to process.
    model_pairs = [
        (display, DRIVING_MODELS[display])
        for display in args.models
        if display in DRIVING_MODELS
    ]
    if not model_pairs:
        print("No valid driving-model labels given. Check --models argument.")
        print(f"Available: {list(DRIVING_MODELS.keys())}")
        return

    print("=" * 62)
    print("  Narration artifact precompute")
    print("=" * 62)
    print(f"  Models    : {[d for d, _ in model_pairs]}")
    print(f"  Scenarios : {scenarios}")
    print(f"  Data      : {data_path}")
    print(f"  Overwrite : {overwrite}")
    print()

    for display_label, model_key in model_pairs:
        if model_key not in PLATFORM_MODELS:
            print(f"[SKIP] '{model_key}' not in PLATFORM_MODELS catalog.")
            continue

        entry = PLATFORM_MODELS[model_key]
        if not entry.exists_on_disk:
            print(f"[SKIP] '{model_key}' model directory not found on disk:")
            print(f"       {entry.model_dir}")
            continue

        print(f"\n{'─'*62}")
        print(f"  {display_label}  ({entry.cache_slug})")
        print(f"{'─'*62}")

        # Load the policy once per model — reused across all scenarios.
        print("  Loading model (JAX/Flax)…")
        xai_model  = load_explainable_model(entry)
        loaded     = xai_model._loaded
        env        = loaded.env
        policy_fn  = loaded.policy_fn
        print("  OK\n")

        cache_dir = _PLATFORM_CACHE_ROOT / entry.cache_slug
        cache_dir.mkdir(parents=True, exist_ok=True)

        for scenario_idx in scenarios:
            print(f"  Scenario {scenario_idx}")
            artifact_path = cache_dir / f"scenario_{scenario_idx:04d}_artifact.pkl"
            frames_path   = cache_dir / f"scenario_{scenario_idx:04d}_frames.pkl"

            # ── Artifact ──────────────────────────────────────────────────────
            if artifact_path.exists() and not overwrite:
                print(f"    Artifact cached — loading ({artifact_path.name})")
                with open(artifact_path, "rb") as fh:
                    artifact = pickle.load(fh)
            else:
                scenario_data, raw_obs = _run_episode(
                    env, policy_fn, data_path, scenario_idx
                )
                scenario_data.model_key = model_key

                artifact = PlatformScenarioArtifact(
                    scenario_data=scenario_data,
                    model_key=model_key,
                    scenario_idx=scenario_idx,
                    raw_observations=raw_obs,
                    notes=f"{entry.description} — scenario {scenario_idx}",
                )
                save_artifact(artifact)
                print(f"    Saved artifact → {artifact_path.name}")

            # ── BEV frames ────────────────────────────────────────────────────
            _prerender_bev_frames(artifact, frames_path, overwrite=overwrite)

    print("\n\nPrecompute complete.")
    print("You can now run: python scripts/precompute_llm_narration.py")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    _default_scenarios = [idx for idx, _ in SCENARIO_CATALOG]
    _default_models    = list(DRIVING_MODELS.keys())

    parser = argparse.ArgumentParser(
        description=(
            "Precompute BEV artifacts (scenario rollout + bird's-eye-view frames) "
            "for the LLM Narration tab. Run this before precompute_llm_narration.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenarios", type=int, nargs="+", default=_default_scenarios,
        metavar="IDX",
        help="Scenario indices to process.",
    )
    parser.add_argument(
        "--models", nargs="+", default=_default_models,
        metavar="LABEL",
        help=(
            f"Driving model display labels. Choose from: {_default_models}."
        ),
    )
    parser.add_argument(
        "--data", default=_DEFAULT_DATA_PATH,
        metavar="PATH",
        help="Path to the .tfrecord dataset file.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-compute and overwrite existing cache files.",
    )
    main(parser.parse_args())
