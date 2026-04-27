"""Scenario scanning, episode rollout, and selection."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from posthoc_xai.experiments.config import ExperimentConfig
from posthoc_xai.models.base import ExplainableModel


@dataclass
class ScenarioInfo:
    """Summary of a single scenario after episode rollout."""

    index: int
    num_valid_agents: int = 0
    num_traffic_lights: int = 0
    has_traffic_lights: bool = False

    # Episode results
    total_steps: int = 0
    collision: bool = False
    offroad: bool = False
    comfort_violation: bool = False
    ran_red_light: bool = False
    route_completion: float = 0.0

    # Derived
    tags: list[str] = field(default_factory=list)
    key_timesteps: list[int] = field(default_factory=list)

    # Path to saved observations (.npz)
    saved_obs_path: str | None = None


def _compute_key_timesteps(
    total_steps: int,
    strategy: str,
    interval: int = 20,
    failure_step: int | None = None,
) -> list[int]:
    """Determine which timesteps to analyze for a scenario."""
    if total_steps == 0:
        return [0]

    if strategy == "all":
        return list(range(total_steps))

    if strategy == "fixed_interval":
        steps = list(range(0, total_steps, interval))
        if (total_steps - 1) not in steps:
            steps.append(total_steps - 1)
        return sorted(steps)

    # key_moments: first, mid, last, plus pre-failure if applicable
    steps = set()
    steps.add(0)
    steps.add(total_steps // 2)
    steps.add(total_steps - 1)

    if failure_step is not None and failure_step > 1:
        steps.add(max(0, failure_step - 1))

    return sorted(steps)


def _tag_scenario(info: ScenarioInfo) -> list[str]:
    """Assign tags based on scenario properties."""
    tags = []
    if info.collision:
        tags.append("collision")
    if info.offroad:
        tags.append("offroad")
    if info.comfort_violation:
        tags.append("comfort_violation")
    if info.ran_red_light:
        tags.append("ran_red_light")
    if info.has_traffic_lights:
        tags.append("has_lights")
    if info.num_valid_agents >= 5:
        tags.append("crowded")
    if info.route_completion >= 1.0:
        tags.append("full_completion")
    if not tags:
        tags.append("normal")
    return tags


def scan_scenarios(
    model: ExplainableModel,
    config: ExperimentConfig,
    progress_fn: Optional[Callable[[int, int, ScenarioInfo], None]] = None,
) -> list[ScenarioInfo]:
    """Scan scenarios by running episodes and collecting metadata.

    Creates a fresh data generator for reproducibility (seed=42), runs each
    episode to completion, and saves observations at key timesteps.

    Args:
        model: Loaded ExplainableModel (must have ``_loaded`` attribute).
        config: Experiment configuration.
        progress_fn: Optional callback ``(current, total, info)`` for progress.

    Returns:
        List of ScenarioInfo for all scanned scenarios.
    """
    from vmax.simulator import make_data_generator
    from vmax.agents import pipeline

    loaded = model._loaded

    # Create a fresh data generator for reproducibility
    data_gen = make_data_generator(
        path=config.data_path,
        max_num_objects=loaded.config.get("max_num_objects", 64),
        include_sdc_paths=True,
        batch_dims=(1,),
        seed=42,
        repeat=1,
    )

    step_fn = partial(pipeline.policy_step, env=loaded.env, policy_fn=loaded.policy_fn)
    jit_reset = jax.jit(loaded.env.reset)

    # Directory for saved observations
    obs_dir = os.path.join(config.results_dir, "observations")
    os.makedirs(obs_dir, exist_ok=True)

    catalog: list[ScenarioInfo] = []
    data_iter = iter(data_gen)

    for scenario_idx in range(config.n_scenarios):
        try:
            scenario = next(data_iter)
        except StopIteration:
            print(f"  Data exhausted after {scenario_idx} scenarios")
            break

        # Run episode
        rng_key = jax.random.PRNGKey(scenario_idx)
        rng_key, reset_key = jax.random.split(rng_key)
        env_transition = jit_reset(scenario, jax.random.split(reset_key, 1))

        # Determine key timesteps after a first pass to know total_steps.
        # We collect observations at every step, then save only the key ones.
        obs_buffer: dict[int, np.ndarray] = {}
        step = 0

        while not bool(env_transition.done):
            # Collect observation for this step
            obs_flat = np.array(env_transition.observation).flatten()
            obs_buffer[step] = obs_flat

            rng_key, step_key = jax.random.split(rng_key)
            env_transition, _transition = step_fn(
                env_transition, key=jax.random.split(step_key, 1)
            )
            step += 1

        # Collect final observation
        obs_buffer[step] = np.array(env_transition.observation).flatten()
        total_steps = step

        # Extract metrics
        metrics = {k: float(v[0]) for k, v in env_transition.metrics.items()}

        collision = metrics.get("collision", 0.0) > 0
        offroad_val = metrics.get("offroad", 0.0) > 0
        comfort = metrics.get("comfort", 1.0) < 1.0
        red_light = metrics.get("ran_red_light", 0.0) > 0
        route_comp = metrics.get("route_completion", 0.0)

        # Failure step detection (for key_moments strategy)
        failure_step = None
        if collision or offroad_val:
            failure_step = total_steps

        # Count valid agents and traffic lights from the first observation
        try:
            validity = model.get_entity_validity(
                jnp.array(obs_buffer[0])
            )
            num_valid_agents = sum(
                1 for v in validity.get("other_agents", {}).values() if v
            )
            num_tl = sum(
                1 for v in validity.get("traffic_lights", {}).values() if v
            )
        except Exception:
            num_valid_agents = 0
            num_tl = 0

        # Compute key timesteps
        key_ts = _compute_key_timesteps(
            total_steps, config.timestep_strategy, config.timestep_interval, failure_step
        )

        info = ScenarioInfo(
            index=scenario_idx,
            num_valid_agents=num_valid_agents,
            num_traffic_lights=num_tl,
            has_traffic_lights=num_tl > 0,
            total_steps=total_steps,
            collision=collision,
            offroad=offroad_val,
            comfort_violation=comfort,
            ran_red_light=red_light,
            route_completion=route_comp,
            key_timesteps=key_ts,
        )
        info.tags = _tag_scenario(info)

        # Save observations at key timesteps
        obs_path = os.path.join(obs_dir, f"s{scenario_idx:03d}.npz")
        obs_to_save = {}
        for ts in key_ts:
            if ts in obs_buffer:
                obs_to_save[f"step_{ts}"] = obs_buffer[ts]
            elif total_steps in obs_buffer:
                obs_to_save[f"step_{ts}"] = obs_buffer[total_steps]
        np.savez_compressed(obs_path, **obs_to_save)
        info.saved_obs_path = obs_path

        catalog.append(info)

        if progress_fn:
            progress_fn(scenario_idx + 1, config.n_scenarios, info)
        else:
            tag_str = ", ".join(info.tags)
            print(
                f"  [{scenario_idx + 1}/{config.n_scenarios}] "
                f"scenario_{scenario_idx:03d}: {total_steps} steps, "
                f"collision={collision}, agents={num_valid_agents}, "
                f"tags=[{tag_str}]"
            )

    return catalog


def select_scenarios(
    catalog: list[ScenarioInfo],
    config: ExperimentConfig,
) -> list[ScenarioInfo]:
    """Filter and rank scenarios for analysis.

    Applies config filters, always includes failure scenarios, and ranks
    remaining by "interestingness" (more agents + traffic lights).

    Returns:
        Selected subset, capped at config.max_selected.
    """
    failures = []
    candidates = []

    for info in catalog:
        is_failure = info.collision or info.offroad

        # Check filters
        if config.min_valid_agents > 0 and info.num_valid_agents < config.min_valid_agents:
            if not (config.include_failures and is_failure):
                continue
        if config.require_traffic_lights and not info.has_traffic_lights:
            if not (config.include_failures and is_failure):
                continue

        if is_failure and config.include_failures:
            failures.append(info)
        else:
            candidates.append(info)

    # Sort candidates by interestingness (agents + traffic lights)
    candidates.sort(
        key=lambda s: (s.num_valid_agents + s.num_traffic_lights, s.total_steps),
        reverse=True,
    )

    # Combine: failures first, then most interesting
    selected = failures.copy()
    for c in candidates:
        if c not in selected:
            selected.append(c)
        if len(selected) >= config.max_selected:
            break

    print(
        f"\nSelected {len(selected)}/{len(catalog)} scenarios "
        f"({len(failures)} failures, {len(selected) - len(failures)} others)"
    )
    return selected


# ------------------------------------------------------------------
# Catalog serialization
# ------------------------------------------------------------------

def save_catalog(catalog: list[ScenarioInfo], path: str) -> None:
    """Save scenario catalog to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = [asdict(info) for info in catalog]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Catalog saved to {path}")


def load_catalog(path: str) -> list[ScenarioInfo]:
    """Load scenario catalog from JSON."""
    with open(path) as f:
        data = json.load(f)
    catalog = []
    for d in data:
        info = ScenarioInfo(index=d["index"])
        for k, v in d.items():
            if k != "index":
                setattr(info, k, v)
        catalog.append(info)
    print(f"  Catalog loaded from {path} ({len(catalog)} scenarios)")
    return catalog
