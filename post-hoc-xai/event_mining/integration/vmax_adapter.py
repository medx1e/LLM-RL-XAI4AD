"""Extract ScenarioData from V-Max episode rollouts.

Reuses the scanner loop pattern from ``posthoc_xai.experiments.scanner``
but collects per-timestep trajectory data instead of just episode-level flags.

Key adaptation: V-Max doesn't expose raw sim_state directly. We extract
entity positions/velocities from **unflattened observations** at each step:
    features, masks = unflatten_fn(obs)
    sdc_feat, other_feat, rg_feat, tl_feat, gps_feat = features
    # other_feat shape: (1, 8, 5, 7) -> 8 agents, 5 timesteps, 7 features
    # features per agent: [x, y, vx, vy, yaw, length, width]
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from event_mining.events.base import ScenarioData
from event_mining.metrics import (
    compute_distances,
    compute_ttc,
    compute_criticality,
    compute_ego_speed,
)


class VMaxAdapter:
    """Extracts ScenarioData from V-Max model rollouts."""

    def __init__(self, store_raw_obs: bool = True):
        """
        Args:
            store_raw_obs: Whether to keep raw flat observations in ScenarioData
                (needed for XAI methods later).
        """
        self.store_raw_obs = store_raw_obs

    def prepare(self, model):
        """Pre-compile JAX functions for reuse across scenarios.

        Call once before ``extract_scenario_data`` to avoid re-compilation.

        Args:
            model: An ExplainableModel with ``_loaded`` attribute.

        Returns:
            self (for chaining).
        """
        from vmax.agents import pipeline

        loaded = model._loaded
        self._env = loaded.env
        self._unflatten_fn = loaded.unflatten_fn
        self._step_fn = partial(
            pipeline.policy_step, env=loaded.env, policy_fn=loaded.policy_fn
        )
        self._jit_reset = jax.jit(loaded.env.reset)
        self._prepared = True
        return self

    def extract_scenario_data(
        self,
        model,
        scenario,
        scenario_id: str,
        rng_seed: int = 0,
    ) -> ScenarioData:
        """Run a full episode and extract per-timestep data.

        Args:
            model: An ExplainableModel (from ``posthoc_xai.load_model()``).
                Must have ``_loaded`` attribute with env, policy_fn, unflatten_fn.
            scenario: A raw scenario from the data generator.
            scenario_id: Identifier string for this scenario.
            rng_seed: Random seed for the episode.

        Returns:
            Populated ScenarioData with trajectories, actions, metrics.
        """
        # Use pre-compiled functions if available
        if not getattr(self, "_prepared", False):
            self.prepare(model)

        env = self._env
        unflatten_fn = self._unflatten_fn
        step_fn = self._step_fn
        jit_reset = self._jit_reset

        # Reset environment
        rng_key = jax.random.PRNGKey(rng_seed)
        rng_key, reset_key = jax.random.split(rng_key)
        env_transition = jit_reset(scenario, jax.random.split(reset_key, 1))

        # Collect per-step data
        ego_data = []  # list of (x, y, vx, vy, yaw, length, width)
        action_data = []  # list of (accel, steering)
        other_data = []  # list of (N_agents, 7)
        other_valid_data = []  # list of (N_agents,) bool
        raw_obs_list = []
        step_metrics = []  # list of metric dicts

        # Static features (captured once)
        rg_feat_static = None
        rg_mask_static = None
        tl_data = []
        tl_valid_data = []
        gps_path_static = None

        step = 0

        while not bool(env_transition.done):
            obs = env_transition.observation  # (1, obs_dim)

            if self.store_raw_obs:
                raw_obs_list.append(np.array(obs).flatten())

            # Unflatten observation to get per-category features
            features, masks = unflatten_fn(obs)
            sdc_feat, other_feat, rg_feat, tl_feat, gps_feat = features
            sdc_mask, other_mask, rg_mask, tl_mask = masks

            # Extract ego data from sdc_feat
            # sdc_feat shape: (batch, [1_ego,] n_timesteps, 7)
            # Squeeze to (n_timesteps, 7) regardless of extra dims
            sdc_arr = np.array(sdc_feat[0])  # drop batch dim
            while sdc_arr.ndim > 2:
                sdc_arr = sdc_arr[0]  # squeeze leading singleton dims
            # sdc_arr is now (n_timesteps, 7) — take most recent
            ego_current = sdc_arr[-1]  # (7,)
            ego_data.append(ego_current[:7] if len(ego_current) >= 7 else np.pad(
                ego_current, (0, 7 - len(ego_current))
            ))

            # Extract other agent data
            # other_feat shape: (batch, N_agents, n_timesteps, 7)
            other_arr = np.array(other_feat[0])  # (N_agents, n_timesteps, 7)
            while other_arr.ndim > 3:
                other_arr = other_arr[0]  # squeeze unexpected leading dims
            other_current = other_arr[:, -1, :]  # (N_agents, 7) — most recent timestep
            n_agents = other_current.shape[0]
            if other_current.shape[1] >= 7:
                other_data.append(other_current[:, :7])
            else:
                other_data.append(np.pad(
                    other_current, ((0, 0), (0, 7 - other_current.shape[1]))
                ))

            # Other agent validity: use any-valid-across-time per agent
            # other_mask shape: (batch, N_agents, n_timesteps)
            other_mask_arr = np.array(other_mask[0])  # (N_agents, n_timesteps) or (N_agents,)
            while other_mask_arr.ndim > 2:
                other_mask_arr = other_mask_arr[0]
            if other_mask_arr.ndim == 2:
                # Any timestep valid → agent is valid
                other_valid_data.append(other_mask_arr.any(axis=-1).astype(bool))
            else:
                other_valid_data.append(other_mask_arr.astype(bool))

            # Static features — capture on first step
            if step == 0:
                rg_feat_static = np.array(rg_feat[0])
                rg_mask_static = np.array(rg_mask[0]).astype(bool)
                gps_arr = np.array(gps_feat[0])
                if gps_arr.ndim >= 2:
                    gps_path_static = gps_arr[:, :2] if gps_arr.shape[-1] >= 2 else gps_arr
                else:
                    gps_path_static = gps_arr

            # Traffic lights (may change per step)
            # tl_feat shape: (batch, N_lights, n_timesteps, features)
            tl_arr = np.array(tl_feat[0])  # (N_lights, n_timesteps, features)
            tl_data.append(tl_arr)
            tl_mask_arr = np.array(tl_mask[0])  # (N_lights, n_timesteps)
            while tl_mask_arr.ndim > 2:
                tl_mask_arr = tl_mask_arr[0]
            if tl_mask_arr.ndim == 2:
                tl_valid_data.append(tl_mask_arr.any(axis=-1).astype(bool))
            else:
                tl_valid_data.append(tl_mask_arr.astype(bool))

            # Take step
            rng_key, step_key = jax.random.split(rng_key)
            env_transition, transition = step_fn(
                env_transition, key=jax.random.split(step_key, 1)
            )

            # Extract action
            action = np.array(transition.action).flatten()
            if len(action) >= 2:
                action_data.append((float(action[0]), float(action[1])))
            else:
                action_data.append((float(action[0]) if len(action) > 0 else 0.0, 0.0))

            # Per-step metrics
            m = {k: float(v[0]) for k, v in env_transition.metrics.items()}
            step_metrics.append(m)

            step += 1

        total_steps = step
        if total_steps == 0:
            # Edge case: episode was already done at reset
            return self._empty_scenario_data(scenario_id)

        # Stack collected arrays
        ego_arr = np.array(ego_data)  # (T, 7)
        other_arr = np.stack(other_data)  # (T, N_agents, 7)
        other_valid_arr = np.stack(other_valid_data)  # (T, N_agents)
        actions = np.array(action_data)  # (T, 2)

        # Extract per-step collision/offroad from metrics
        step_collision = np.array([
            m.get("collision", 0.0) > 0 for m in step_metrics
        ])
        step_offroad = np.array([
            m.get("offroad", 0.0) > 0 for m in step_metrics
        ])

        # Episode-level outcomes (from final metrics)
        final_metrics = step_metrics[-1] if step_metrics else {}
        has_collision = final_metrics.get("collision", 0.0) > 0
        has_offroad = final_metrics.get("offroad", 0.0) > 0
        route_completion = final_metrics.get("route_completion", 0.0)

        # Find first occurrence of collision/offroad
        collision_time = None
        if has_collision:
            coll_steps = np.where(step_collision)[0]
            collision_time = int(coll_steps[0]) if len(coll_steps) > 0 else total_steps - 1

        offroad_time = None
        if has_offroad:
            offroad_steps = np.where(step_offroad)[0]
            offroad_time = int(offroad_steps[0]) if len(offroad_steps) > 0 else total_steps - 1

        # Compute derived safety metrics
        ego_x = ego_arr[:, 0]
        ego_y = ego_arr[:, 1]
        ego_vx = ego_arr[:, 2]
        ego_vy = ego_arr[:, 3]
        other_x = other_arr[:, :, 0]
        other_y = other_arr[:, :, 1]
        other_vx = other_arr[:, :, 2]
        other_vy = other_arr[:, :, 3]

        distances = compute_distances(ego_x, ego_y, other_x, other_y, other_valid_arr)
        ttc = compute_ttc(
            ego_x, ego_y, ego_vx, ego_vy,
            other_x, other_y, other_vx, other_vy,
            other_valid_arr,
        )
        ego_speed = compute_ego_speed(ego_vx, ego_vy)
        criticality = compute_criticality(ttc, distances, ego_speed)
        nearest_agent = np.argmin(distances, axis=1)

        # Stack traffic lights
        tl_stacked = np.stack(tl_data) if tl_data else None
        tl_valid_stacked = np.stack(tl_valid_data) if tl_valid_data else None

        # Raw observations
        raw_obs = np.stack(raw_obs_list) if raw_obs_list else None

        return ScenarioData(
            scenario_id=scenario_id,
            total_steps=total_steps,
            ego_x=ego_x,
            ego_y=ego_y,
            ego_vx=ego_vx,
            ego_vy=ego_vy,
            ego_yaw=ego_arr[:, 4],
            ego_length=ego_arr[:, 5],
            ego_width=ego_arr[:, 6],
            ego_accel=actions[:, 0],
            ego_steering=actions[:, 1],
            other_agents=other_arr,
            other_agents_valid=other_valid_arr,
            road_graph=rg_feat_static,
            road_graph_valid=rg_mask_static,
            traffic_lights=tl_stacked,
            traffic_lights_valid=tl_valid_stacked,
            gps_path=gps_path_static,
            ttc=ttc,
            min_distance=distances,
            nearest_agent_id=nearest_agent,
            criticality=criticality,
            step_collision=step_collision,
            step_offroad=step_offroad,
            has_collision=has_collision,
            collision_time=collision_time,
            has_offroad=has_offroad,
            offroad_time=offroad_time,
            route_completion=route_completion,
            raw_observations=raw_obs,
        )

    def _empty_scenario_data(self, scenario_id: str) -> ScenarioData:
        """Return an empty ScenarioData for edge cases."""
        empty = np.array([])
        empty2d = np.empty((0, 1))
        return ScenarioData(
            scenario_id=scenario_id,
            total_steps=0,
            ego_x=empty, ego_y=empty, ego_vx=empty, ego_vy=empty,
            ego_yaw=empty, ego_length=empty, ego_width=empty,
            ego_accel=empty, ego_steering=empty,
            other_agents=np.empty((0, 1, 7)),
            other_agents_valid=np.empty((0, 1), dtype=bool),
        )
