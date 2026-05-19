# Copyright 2025 Valeo.

"""Activation harvester for SAE training data collection.

Extends OfflineExtractor to capture residual stream vectors and rich telemetry
in HDF5 format. The harvested data is the input to sae_trainer.py.

Usage:
    python -m sae_interpretability.harvester \\
        --run_dir ../../runs/PPO_VEC_WAYFORMER \\
        --dataset ../../training.tfrecord \\
        --n_scenarios 500 --output harvest.h5
"""

from __future__ import annotations
from sae_interpretability.config import SAEConfig
from xai.attention_analysis.offline_extraction import OfflineExtractor
import jax.numpy as jnp
import jax
import h5py

import argparse
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

import numpy as np

project_root = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Scalar telemetry fields written as float32 [N]
# NOTE: ego_x / ego_y intentionally excluded — raw map coordinates are not
# driving concepts and absorb SAE features with spurious position variance.
_SCALAR_KEYS = [
    'ego_speed',
    'ego_lat_accel',         # lateral acceleration (m/s²) — cornering intensity
    'ego_yaw_rate',          # heading change rate  (rad/step) — turning signal
    'min_ttc',
    'min_agent_dist',
    'n_valid_agents',        # count of valid nearby agents (scene density)
    'lead_vehicle_dist',     # distance to lead vehicle (inf if none ahead)
    'lead_vehicle_closing_speed',  # closing speed to lead vehicle
]

# Boolean telemetry fields written as bool [N]
_BOOL_KEYS = [
    'is_ttc_critical',
    'is_lead_vehicle_hard_braking',
    'is_following_lead',     # lead vehicle within 30 m and ahead
    'has_agent_left',        # any valid agent to the left
    'has_agent_right',       # any valid agent to the right
]

# Per-agent fields written as float32 [N, n_agents]
_AGENT_FLOAT_KEYS = ['agent_dists', 'agent_ttcs', 'agent_speeds',
                     'agent_is_ahead', 'agent_is_left', 'agent_closing_speed']


class ActivationHarvester(OfflineExtractor):
    """Harvests residual stream activations and extended telemetry for SAE training.

    Extends OfflineExtractor to:
    1. Capture the encoder output (residual stream) at every rollout timestep.
    2. Compute two boolean event flags that are easy to correlate with SAE features:
       - is_ttc_critical: True when min TTC across valid agents < 1.5 s.
       - is_lead_vehicle_hard_braking: True when the closest ahead-agent
         decelerates > 0.4g in a single 0.1 s step.
    3. Write everything to a resizable HDF5 file suitable for PyTorch DataLoader.
    """

    G = 9.81  # m/s²

    def __init__(
        self,
        run_dir: str,
        dataset_path: str,
        cfg: SAEConfig,
        checkpoint_name: str = "model_final.pkl",
    ):
        super().__init__(run_dir, dataset_path, checkpoint_name)
        self.sae_cfg = cfg

    # ------------------------------------------------------------------
    # Telemetry helpers
    # ------------------------------------------------------------------

    def _compute_lead_vehicle_hard_braking(
        self,
        curr_sem: Dict[str, np.ndarray],
        prev_sem: Optional[Dict[str, np.ndarray]],
    ) -> bool:
        """True if the closest ahead-agent drops speed by > 0.4g * dt in one step."""
        if prev_sem is None:
            return False

        is_ahead = np.asarray(curr_sem.get('is_ahead', []), dtype=bool)
        valid = np.asarray(curr_sem.get('valid', []), dtype=bool)
        distances = np.asarray(curr_sem.get('distance_to_ego', []))
        curr_speeds = np.asarray(curr_sem.get('agent_speeds', []))
        prev_speeds = np.asarray(prev_sem.get(
            'agent_speeds', np.zeros_like(curr_speeds)))

        n = min(len(is_ahead), len(valid), len(distances),
                len(curr_speeds), len(prev_speeds))
        if n == 0:
            return False

        is_ahead = is_ahead[:n]
        valid = valid[:n]
        distances = distances[:n]
        curr_speeds = curr_speeds[:n]
        prev_speeds = prev_speeds[:n]

        ahead_and_valid = is_ahead & valid
        if not ahead_and_valid.any():
            return False

        ahead_dists = np.where(ahead_and_valid, distances, np.inf)
        lead_idx = int(np.argmin(ahead_dists))

        threshold = self.sae_cfg.hard_braking_g_threshold * \
            self.G * self.sae_cfg.harvest_dt
        speed_drop = float(prev_speeds[lead_idx]) - \
            float(curr_speeds[lead_idx])
        return bool(speed_drop > threshold)

    def _build_telemetry_row(
        self,
        semantic: Dict[str, np.ndarray],
        prev_semantic: Optional[Dict[str, np.ndarray]],
    ) -> Dict[str, Any]:
        """Flatten a semantic features dict into a single telemetry row."""
        row: Dict[str, Any] = {}

        sdc_idx = int(semantic.get('sdc_index', 0))

        # --- Ego state ---------------------------------------------------
        ego_speed = float(semantic.get('ego_speed', 0.0))
        row['ego_speed'] = ego_speed

        # Lateral acceleration & yaw rate (derived from prev step)
        if prev_semantic is not None:
            prev_speed = float(prev_semantic.get('ego_speed', 0.0))
            # Ego yaw at current and previous timestep
            ego_yaw_curr = float(semantic.get('ego_yaw', 0.0)) if 'ego_yaw' in semantic else 0.0
            ego_yaw_prev = float(prev_semantic.get('ego_yaw', 0.0)) if 'ego_yaw' in prev_semantic else 0.0
            yaw_rate = ego_yaw_curr - ego_yaw_prev
            avg_speed = 0.5 * (ego_speed + prev_speed)
            lat_accel = avg_speed * abs(yaw_rate) / self.sae_cfg.harvest_dt if self.sae_cfg.harvest_dt > 0 else 0.0
        else:
            yaw_rate = 0.0
            lat_accel = 0.0

        row['ego_lat_accel'] = float(lat_accel)
        row['ego_yaw_rate'] = float(yaw_rate)

        # --- Per-agent arrays --------------------------------------------
        distances = np.asarray(semantic.get('distance_to_ego', []))
        ttc = np.asarray(semantic.get('ttc', []))
        speeds = np.asarray(semantic.get('agent_speeds', []))
        valid = np.asarray(semantic.get('valid', []), dtype=bool)
        closing = np.asarray(semantic.get('closing_speed', []))
        is_ahead = np.asarray(semantic.get('is_ahead', []))
        is_left = np.asarray(semantic.get('is_left', []))
        is_right = np.asarray(semantic.get('is_right', []))

        row['agent_dists'] = distances
        row['agent_ttcs'] = ttc
        row['agent_speeds'] = speeds
        row['agent_valid'] = valid
        row['agent_closing_speed'] = closing
        row['agent_is_ahead'] = is_ahead
        row['agent_is_left'] = is_left

        # --- Scalar summaries --------------------------------------------
        valid_ttc = ttc[valid] if (
            len(ttc) > 0 and len(valid) > 0) else np.array([])
        min_ttc = float(np.min(valid_ttc)) if len(valid_ttc) > 0 else 5.0
        min_dist = float(np.min(distances[valid])) if valid.any() and len(
            distances) > 0 else 999.0

        row['min_ttc'] = min_ttc
        row['min_agent_dist'] = min_dist
        row['n_valid_agents'] = float(valid.sum())

        # Lead vehicle (closest ahead + valid)
        ahead_and_valid = is_ahead.astype(bool) & valid
        if ahead_and_valid.any():
            ahead_dists = np.where(ahead_and_valid, distances, np.inf)
            lead_idx = int(np.argmin(ahead_dists))
            row['lead_vehicle_dist'] = float(distances[lead_idx])
            row['lead_vehicle_closing_speed'] = float(closing[lead_idx])
            row['is_following_lead'] = bool(distances[lead_idx] < 30.0)
        else:
            row['lead_vehicle_dist'] = 999.0
            row['lead_vehicle_closing_speed'] = 0.0
            row['is_following_lead'] = False

        # Spatial presence flags
        row['has_agent_left'] = bool((is_left.astype(bool) & valid).any())
        row['has_agent_right'] = bool((is_right.astype(bool) & valid).any()) if len(is_right) > 0 else False

        # --- Boolean event flags -----------------------------------------
        row['is_ttc_critical'] = bool(
            min_ttc < self.sae_cfg.ttc_critical_threshold)
        row['is_lead_vehicle_hard_braking'] = self._compute_lead_vehicle_hard_braking(
            semantic, prev_semantic
        )

        return row

    # ------------------------------------------------------------------
    # HDF5 helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resize_and_write(ds: h5py.Dataset, start: int, data: np.ndarray):
        new_size = start + len(data)
        ds.resize(new_size, axis=0)
        ds[start:new_size] = data

    # ------------------------------------------------------------------
    # Main harvesting entry point
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Fast path: jax.lax.scan over timesteps
    # ------------------------------------------------------------------

    def _compute_telemetry_vectorized(
        self,
        raw: Dict[str, np.ndarray],
        valid_steps: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute all telemetry for T steps in one vectorized numpy pass.

        Args:
            raw: Dict of stacked arrays from jax.device_get after scan.
                 Keys: x_t, y_t, vx_t, vy_t, yaw_t, valid_t, is_sdc.
                 Shapes: [T, n_objects] each.
            valid_steps: bool [T] mask — False for post-termination steps.

        Returns:
            Dict matching the keys expected by the HDF5 writer.
        """
        TTC_HORIZON = self.TTC_HORIZON
        n_v = self._n_vehicles

        x   = raw['x_t']          # [T, n_objects]
        y   = raw['y_t']
        vx  = raw['vx_t']
        vy  = raw['vy_t']
        yaw = raw['yaw_t']
        valid_agents = raw['valid_t'].astype(bool)  # [T, n_objects]
        is_sdc = raw['is_sdc']                       # [T, n_objects]

        sdc_idx = int(np.argmax(is_sdc[0]))

        # ---- Ego state ------------------------------------------------
        ego_x   = x[:, sdc_idx]          # [T]
        ego_y   = y[:, sdc_idx]
        ego_vx  = vx[:, sdc_idx]
        ego_vy  = vy[:, sdc_idx]
        ego_yaw = yaw[:, sdc_idx]
        ego_speed = np.sqrt(ego_vx**2 + ego_vy**2)   # [T]

        # ---- Per-agent (first n_v agents) -----------------------------
        x_v  = x[:, :n_v]          # [T, n_v]
        y_v  = y[:, :n_v]
        vx_v = vx[:, :n_v]
        vy_v = vy[:, :n_v]
        valid_v = valid_agents[:, :n_v].copy()  # [T, n_v]
        # Exclude the ego vehicle from all agent-level statistics.
        # sdc_idx is always valid and always at distance 0 from itself,
        # which would make it win every .min() call (e.g. min_dist = 0).
        if sdc_idx < n_v:
            valid_v[:, sdc_idx] = False

        dx = x_v - ego_x[:, None]       # [T, n_v]
        dy = y_v - ego_y[:, None]
        dist = np.sqrt(dx**2 + dy**2)   # [T, n_v]

        rel_vx = vx_v - ego_vx[:, None]
        rel_vy = vy_v - ego_vy[:, None]
        dist_safe = dist + 1e-6
        closing = -(dx * rel_vx + dy * rel_vy) / dist_safe  # [T, n_v]

        ttc = np.where(
            closing > 0.5,
            dist / (closing + 1e-6),
            TTC_HORIZON,
        )
        ttc = np.clip(ttc, 0.0, TTC_HORIZON)   # [T, n_v]

        # Spatial (ego-local frame)
        cos_y = np.cos(-ego_yaw)[:, None]       # [T, 1]
        sin_y = np.sin(-ego_yaw)[:, None]
        x_local =  dx * cos_y - dy * sin_y
        y_local =  dx * sin_y + dy * cos_y

        is_ahead = (x_local >  2.0).astype(np.float32)   # [T, n_v]
        is_left  = (y_local >  1.0).astype(np.float32)

        agent_speeds = np.sqrt(vx_v**2 + vy_v**2)   # [T, n_v]

        # ---- Scalar summaries -----------------------------------------
        # min TTC over valid non-ego agents
        inf_val = np.full_like(ttc, TTC_HORIZON)
        ttc_valid = np.where(valid_v, ttc, inf_val)  # [T, n_v]
        min_ttc = ttc_valid.min(axis=1)              # [T]

        dist_valid = np.where(valid_v, dist, np.full_like(dist, 999.0))
        min_dist = dist_valid.min(axis=1)  # [T]  — ego excluded via valid_v

        # ---- Boolean events -------------------------------------------
        is_ttc_critical = min_ttc < self.sae_cfg.ttc_critical_threshold  # [T]

        # Hard braking: speed drop of lead ahead agent > threshold in 1 step
        threshold = self.sae_cfg.hard_braking_g_threshold * self.G * self.sae_cfg.harvest_dt
        ahead_and_valid = is_ahead.astype(bool) & valid_v  # [T, n_v]
        # Lead vehicle index per step: argmin distance among ahead+valid
        dist_lead = np.where(ahead_and_valid, dist, np.full_like(dist, np.inf))
        lead_idx = dist_lead.argmin(axis=1)          # [T]
        lead_speed = agent_speeds[np.arange(len(lead_idx)), lead_idx]  # [T]
        # Hard braking at step t: lead speed drops from t-1 → t
        speed_drop = np.concatenate([[0.0], lead_speed[:-1] - lead_speed[1:]])
        is_hard_braking = (speed_drop > threshold) & ahead_and_valid[
            np.arange(len(lead_idx)), lead_idx]      # [T]

        # ---- Ego derived dynamics (yaw rate, lateral accel) -------------
        ego_yaw_rate = np.concatenate(
            [[0.0], np.diff(ego_yaw)])                  # [T]  rad/step
        avg_speed = 0.5 * (ego_speed + np.concatenate(
            [[ego_speed[0]], ego_speed[:-1]]))
        dt = self.sae_cfg.harvest_dt
        ego_lat_accel = avg_speed * np.abs(ego_yaw_rate) / dt if dt > 0 else np.zeros_like(ego_speed)

        # ---- Scene density -----------------------------------------------
        n_valid_agents = valid_v.sum(axis=1).astype(np.float32)   # [T]

        # ---- Lead vehicle features ---------------------------------------
        lead_dist_arr  = dist_lead.min(axis=1)          # [T]
        lead_closing   = closing[np.arange(len(lead_idx)), lead_idx]  # [T]
        # Clamp when no vehicle ahead (dist_lead was inf)
        lead_dist_arr  = np.where(np.isinf(lead_dist_arr), 999.0, lead_dist_arr)
        lead_closing   = np.where(np.isinf(dist_lead.min(axis=1)), 0.0, lead_closing)
        is_following   = (lead_dist_arr < 30.0) & ahead_and_valid[
            np.arange(len(lead_idx)), lead_idx]        # [T]

        # ---- Spatial presence flags --------------------------------------
        is_right = (y_local < -1.0).astype(np.float32)  # [T, n_v]
        has_agent_left  = (is_left.astype(bool) & valid_v).any(axis=1)   # [T]
        has_agent_right = (is_right.astype(bool) & valid_v).any(axis=1)  # [T]

        return {
            # scalars [T]
            'ego_speed':                    ego_speed.astype(np.float32),
            'ego_lat_accel':                ego_lat_accel.astype(np.float32),
            'ego_yaw_rate':                 ego_yaw_rate.astype(np.float32),
            'min_ttc':                      min_ttc.astype(np.float32),
            'min_agent_dist':               min_dist.astype(np.float32),
            'n_valid_agents':               n_valid_agents,
            'lead_vehicle_dist':            lead_dist_arr.astype(np.float32),
            'lead_vehicle_closing_speed':   lead_closing.astype(np.float32),
            # booleans [T]
            'is_ttc_critical':              is_ttc_critical,
            'is_lead_vehicle_hard_braking': is_hard_braking,
            'is_following_lead':            is_following,
            'has_agent_left':               has_agent_left,
            'has_agent_right':              has_agent_right,
            # per-agent [T, n_v]
            'agent_dists':          dist.astype(np.float32),
            'agent_ttcs':           ttc.astype(np.float32),
            'agent_speeds':         agent_speeds.astype(np.float32),
            'agent_is_ahead':       is_ahead,
            'agent_is_left':        is_left,
            'agent_closing_speed':  closing.astype(np.float32),
            'agent_valid':          valid_v,
        }

    def run_harvest_fast(self, n_scenarios: int, output_path: str) -> None:
        """Fast harvest using jax.lax.scan — eliminates the Python timestep loop.

        Design mirrors evaluate.py's jax.lax.while_loop approach:
          1. jax.lax.scan compiles the full T-step episode into one XLA kernel.
          2. Raw trajectory arrays are stacked automatically by scan.
          3. A single jax.device_get per scenario moves all data to the host.
          4. Telemetry is computed in one vectorized numpy pass.

        Args:
            n_scenarios: Number of scenarios to process.
            output_path: Destination HDF5 file path.
        """
        from functools import partial as ft_partial
        from vmax.simulator import make_data_generator, datasets
        from vmax.agents import pipeline

        if self.encoder is None:
            self.setup()

        data_gen = make_data_generator(
            path=datasets.get_dataset(self.dataset_path),
            max_num_objects=self.config["max_num_objects"],
            include_sdc_paths=not self.config.get("waymo_dataset", False),
            batch_dims=(1,),   # must stay 1 — telemetry is per-scenario
            seed=42,
            repeat=1,
        )

        T = self.sae_cfg.harvest_max_timesteps
        D = self.sae_cfg.wayformer_hidden_dim
        encoder        = self.encoder
        encoder_params = self.encoder_params
        env            = self.env

        # JAX-traceable expert step (same function evaluate.py uses)
        _expert_step = jax.jit(ft_partial(pipeline.expert_step, env=env))
        _reset       = jax.jit(env.reset)

        # ------------------------------------------------------------------
        # scan body — MUST be a pure JAX function (no Python control flow
        # on traced values).  Returns (carry, per_step_outputs).
        # ------------------------------------------------------------------
        def _scan_body(env_transition, _):
            obs = env_transition.observation          # [1, obs_dim]

            # Encoder forward pass
            latent, _ = encoder.apply({'params': encoder_params}, obs)
            latent = latent[0]                        # squeeze batch → [D]

            # Raw trajectory snapshot at the *current* sim timestep
            state = env_transition.state
            traj  = state.sim_trajectory
            ts    = jnp.squeeze(state.timestep).astype(jnp.int32)

            # traj.x shape: [1, n_objects, total_T]  → slice [n_objects]
            x_t    = traj.x[0, :, ts]
            y_t    = traj.y[0, :, ts]
            vx_t   = traj.vel_x[0, :, ts]
            vy_t   = traj.vel_y[0, :, ts]
            yaw_t  = traj.yaw[0, :, ts]
            valid_t = traj.valid[0, :, ts]
            is_sdc  = state.object_metadata.is_sdc[0]  # [n_objects]

            done = env_transition.done  # [1] bool — True if episode ended

            # Step the environment (expert log-replay, deterministic)
            env_transition, _ = _expert_step(
                env_transition, key=jax.random.PRNGKey(0))

            return env_transition, {
                'latent':  latent,    # [D]
                'x_t':     x_t,       # [n_objects]
                'y_t':     y_t,
                'vx_t':    vx_t,
                'vy_t':    vy_t,
                'yaw_t':   yaw_t,
                'valid_t': valid_t,
                'is_sdc':  is_sdc,
                'done':    done,      # [1]
            }

        # JIT-compile the full episode scan once
        @jax.jit
        def _run_episode(init_transition):
            _, outputs = jax.lax.scan(
                _scan_body, init_transition, None, length=T)
            return outputs

        # ------------------------------------------------------------------
        # HDF5 setup
        # ------------------------------------------------------------------
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
        total_rows = 0
        n_v = self._n_vehicles

        with h5py.File(output_path, 'w') as hf:
            act_ds = hf.create_dataset('activations', shape=(0, D),
                                       maxshape=(None, D), dtype='float32',
                                       chunks=(4096, D))
            sid_ds = hf.create_dataset('scenario_ids', shape=(0,),
                                       maxshape=(None,), dtype='int32',
                                       chunks=(4096,))
            ts_ds  = hf.create_dataset('timesteps', shape=(0,),
                                       maxshape=(None,), dtype='int32',
                                       chunks=(4096,))
            scalar_ds = {
                k: hf.create_dataset(f'telemetry/{k}', shape=(0,),
                                     maxshape=(None,), dtype='float32',
                                     chunks=(4096,))
                for k in _SCALAR_KEYS
            }
            bool_ds = {
                k: hf.create_dataset(f'telemetry/{k}', shape=(0,),
                                     maxshape=(None,), dtype='bool',
                                     chunks=(4096,))
                for k in _BOOL_KEYS
            }
            agent_float_ds: Dict[str, h5py.Dataset] = {}
            agent_valid_ds: Optional[h5py.Dataset]  = None

            print(f"[Harvester-fast] Processing {n_scenarios} scenarios → {output_path}")

            for scenario_idx, scenario_batch in enumerate(data_gen):
                if scenario_idx >= n_scenarios:
                    break

                print(f"  Scenario {scenario_idx + 1}/{n_scenarios}",
                      end="", flush=True)

                try:
                    init_transition = _reset(scenario_batch)

                    # ---- One compiled call for the entire episode ----
                    raw = _run_episode(init_transition)

                    # ---- Single device_get — all T steps at once -----
                    raw_np: Dict[str, np.ndarray] = jax.device_get(raw)

                    # ---- Build valid-step mask ------------------------
                    # done[t]=True means the episode ENDED at step t.
                    # We keep steps where done was still False on entry.
                    done_np = raw_np['done'].squeeze(-1).astype(bool)  # [T]
                    # valid until (and including) the first done step
                    first_done = int(np.argmax(done_np)) if done_np.any() else T
                    valid_steps = np.zeros(T, dtype=bool)
                    valid_steps[:first_done] = True

                    n_new = int(valid_steps.sum())
                    if n_new == 0:
                        print(" ✗ empty")
                        continue

                    # ---- Vectorized telemetry in numpy ---------------
                    tel = self._compute_telemetry_vectorized(raw_np, valid_steps)

                    # Slice to valid timesteps only
                    latents = raw_np['latent'][valid_steps]   # [n_new, D]
                    ts_arr  = np.where(valid_steps)[0].astype(np.int32)

                    # ---- Bulk HDF5 write (once per scenario) ---------
                    self._resize_and_write(act_ds, total_rows, latents)
                    self._resize_and_write(sid_ds, total_rows,
                                           np.full(n_new, scenario_idx, dtype='int32'))
                    self._resize_and_write(ts_ds,  total_rows, ts_arr)

                    for k in _SCALAR_KEYS:
                        self._resize_and_write(scalar_ds[k], total_rows,
                                               tel[k][valid_steps])
                    for k in _BOOL_KEYS:
                        self._resize_and_write(bool_ds[k], total_rows,
                                               tel[k][valid_steps])

                    for k in _AGENT_FLOAT_KEYS:
                        if k not in agent_float_ds:
                            agent_float_ds[k] = hf.create_dataset(
                                f'telemetry/{k}', shape=(0, n_v),
                                maxshape=(None, n_v), dtype='float32',
                                chunks=(4096, n_v))
                        self._resize_and_write(
                            agent_float_ds[k], total_rows,
                            tel[k][valid_steps].astype(np.float32))

                    if agent_valid_ds is None:
                        agent_valid_ds = hf.create_dataset(
                            'telemetry/agent_valid', shape=(0, n_v),
                            maxshape=(None, n_v), dtype='bool',
                            chunks=(4096, n_v))
                    self._resize_and_write(
                        agent_valid_ds, total_rows,
                        tel['agent_valid'][valid_steps])

                    total_rows += n_new
                    print(f" ✓ ({n_new} steps, total={total_rows})")

                except Exception as e:
                    print(f" ✗ {e}")
                    traceback.print_exc()

            meta = hf.create_group('metadata')
            meta.attrs['checkpoint']            = self.checkpoint_name
            meta.attrs['run_dir']               = self.run_dir
            meta.attrs['n_scenarios_requested'] = n_scenarios
            meta.attrs['n_rows']                = total_rows
            meta.attrs['hidden_dim']            = D
            meta.attrs['sae_expansion_factor']  = self.sae_cfg.sae_expansion_factor
            meta.attrs['ttc_critical_threshold'] = self.sae_cfg.ttc_critical_threshold
            meta.attrs['hard_braking_g_threshold'] = self.sae_cfg.hard_braking_g_threshold
            meta.attrs['harvester_mode']        = 'fast_scan'

        print(f"\n[Harvester-fast] Done. {total_rows} rows → {output_path}")
        print(f"  File size: {os.path.getsize(output_path) / 1e6:.1f} MB")

    def run_harvest(self, n_scenarios: int, output_path: str) -> None:
        """Process n_scenarios and write activations + telemetry to HDF5.

        Args:
            n_scenarios: Number of scenarios to process.
            output_path: Destination HDF5 file path.
        """
        from vmax.simulator import make_data_generator, datasets

        if self.encoder is None:
            self.setup()

        data_gen = make_data_generator(
            path=datasets.get_dataset(self.dataset_path),
            max_num_objects=self.config["max_num_objects"],
            include_sdc_paths=not self.config.get("waymo_dataset", False),
            batch_dims=(1,),
            seed=42,
            repeat=1,
        )

        @jax.jit
        def _forward_latent(e_params, obs):
            latent, _ = self.encoder.apply({'params': e_params}, obs)
            return latent

        os.makedirs(os.path.dirname(os.path.abspath(output_path))
                    or '.', exist_ok=True)
        D = self.sae_cfg.wayformer_hidden_dim
        total_rows = 0

        with h5py.File(output_path, 'w') as hf:
            # Resizable activation + index datasets
            act_ds = hf.create_dataset('activations', shape=(0, D), maxshape=(None, D),
                                       dtype='float32', chunks=(4096, D))
            sid_ds = hf.create_dataset('scenario_ids', shape=(0,), maxshape=(None,),
                                       dtype='int32', chunks=(4096,))
            ts_ds = hf.create_dataset('timesteps', shape=(0,), maxshape=(None,),
                                      dtype='int32', chunks=(4096,))

            # Scalar / bool telemetry datasets (pre-created)
            scalar_ds = {
                k: hf.create_dataset(f'telemetry/{k}', shape=(0,), maxshape=(None,),
                                     dtype='float32', chunks=(4096,))
                for k in _SCALAR_KEYS
            }
            bool_ds = {
                k: hf.create_dataset(f'telemetry/{k}', shape=(0,), maxshape=(None,),
                                     dtype='bool', chunks=(4096,))
                for k in _BOOL_KEYS
            }
            # Per-agent datasets created lazily on first batch (need n_agents)
            agent_float_ds: Dict[str, h5py.Dataset] = {}
            agent_valid_ds: Optional[h5py.Dataset] = None

            print(
                f"[Harvester] Processing {n_scenarios} scenarios → {output_path}")

            for scenario_idx, scenario_batch in enumerate(data_gen):
                if scenario_idx >= n_scenarios:
                    break

                print(
                    f"  Scenario {scenario_idx + 1}/{n_scenarios}", end="", flush=True)

                try:
                    env_transition = self.env.reset(scenario_batch)
                    batch_latents: List[np.ndarray] = []
                    batch_tel: List[Dict] = []
                    batch_ts: List[int] = []
                    prev_semantic = None

                    for t in range(self.sae_cfg.harvest_max_timesteps):
                        obs = env_transition.observation
                        latent = _forward_latent(self.encoder_params, obs)
                        latent_np = np.array(jax.device_get(latent))
                        if latent_np.ndim > 1:
                            latent_np = latent_np[0]  # squeeze batch dim → [D]

                        squeezed_state = jax.tree_util.tree_map(
                            lambda x: x.squeeze(0) if hasattr(
                                x, 'squeeze') and x.ndim > 0 else x,
                            env_transition.state,
                        )
                        semantic = self.extract_semantic_features(
                            squeezed_state)
                        tel_row = self._build_telemetry_row(
                            semantic, prev_semantic)

                        batch_latents.append(latent_np)
                        batch_tel.append(tel_row)
                        batch_ts.append(t)
                        prev_semantic = semantic

                        if t > 0 and bool(jax.device_get(env_transition.done)):
                            break

                        try:
                            env_transition = self._expert_step_single(
                                env_transition)
                        except Exception:
                            break

                    if not batch_latents:
                        print(" ✗ empty")
                        continue

                    n_new = len(batch_latents)
                    act_arr = np.stack(batch_latents, axis=0)  # [T, D]

                    self._resize_and_write(act_ds, total_rows, act_arr)
                    self._resize_and_write(
                        sid_ds, total_rows,
                        np.full(n_new, scenario_idx, dtype='int32')
                    )
                    self._resize_and_write(
                        ts_ds, total_rows,
                        np.array(batch_ts, dtype='int32')
                    )

                    for k in _SCALAR_KEYS:
                        self._resize_and_write(
                            scalar_ds[k], total_rows,
                            np.array([r[k]
                                     for r in batch_tel], dtype='float32')
                        )
                    for k in _BOOL_KEYS:
                        self._resize_and_write(
                            bool_ds[k], total_rows,
                            np.array([r[k] for r in batch_tel], dtype=bool)
                        )

                    # Per-agent datasets — create on first batch
                    sample = batch_tel[0]
                    n_agents = len(np.asarray(sample.get('agent_dists', [])))

                    for k in _AGENT_FLOAT_KEYS:
                        if k not in agent_float_ds:
                            agent_float_ds[k] = hf.create_dataset(
                                f'telemetry/{k}', shape=(0, n_agents),
                                maxshape=(None, n_agents), dtype='float32',
                                chunks=(4096, n_agents)
                            )
                        mat = np.array([
                            np.asarray(r[k], dtype='float32')[:n_agents]
                            for r in batch_tel
                        ])
                        self._resize_and_write(
                            agent_float_ds[k], total_rows, mat)

                    if agent_valid_ds is None:
                        agent_valid_ds = hf.create_dataset(
                            'telemetry/agent_valid', shape=(0, n_agents),
                            maxshape=(None, n_agents), dtype='bool',
                            chunks=(4096, n_agents)
                        )
                    mat_valid = np.array([
                        np.asarray(r['agent_valid'], dtype=bool)[:n_agents]
                        for r in batch_tel
                    ])
                    self._resize_and_write(
                        agent_valid_ds, total_rows, mat_valid)

                    total_rows += n_new
                    print(f" ✓ ({n_new} steps, total={total_rows})")

                except Exception as e:
                    print(f" ✗ {e}")
                    traceback.print_exc()

            # Metadata
            meta = hf.create_group('metadata')
            meta.attrs['checkpoint'] = self.checkpoint_name
            meta.attrs['run_dir'] = self.run_dir
            meta.attrs['n_scenarios_requested'] = n_scenarios
            meta.attrs['n_rows'] = total_rows
            meta.attrs['hidden_dim'] = D
            meta.attrs['sae_expansion_factor'] = self.sae_cfg.sae_expansion_factor
            meta.attrs['ttc_critical_threshold'] = self.sae_cfg.ttc_critical_threshold
            meta.attrs['hard_braking_g_threshold'] = self.sae_cfg.hard_braking_g_threshold

        print(f"\n[Harvester] Done. {total_rows} rows → {output_path}")
        print(f"  File size: {os.path.getsize(output_path) / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Harvest Wayformer residual stream activations for SAE training."
    )
    parser.add_argument("--run_dir", required=True,
                        help="Path to training run directory")
    parser.add_argument("--dataset", required=True,
                        help="Dataset path or name")
    parser.add_argument("--n_scenarios", type=int, default=500)
    parser.add_argument("--output", default="harvest.h5")
    parser.add_argument("--checkpoint", default="model_final.pkl")
    parser.add_argument("--expansion_factor", type=int, default=16)
    parser.add_argument(
        "--fast", action="store_true",
        help="Use jax.lax.scan episode loop (much faster, identical output)")
    args = parser.parse_args()

    cfg = SAEConfig()
    harvester = ActivationHarvester(
        args.run_dir, args.dataset, cfg, args.checkpoint)
    harvester.setup()
    if args.fast:
        harvester.run_harvest_fast(args.n_scenarios, args.output)
    else:
        harvester.run_harvest(args.n_scenarios, args.output)


if __name__ == "__main__":
    main()
