"""Multi-timestep Causal Steering with Rollout Comparison.

For each scenario and steering temperature α, two full episode rollouts are run
from the same initial state:

  • Baseline rollout  — agent acts normally (no SAE intervention).
  • Steered rollout   — at every timestep the target SAE feature is clamped
                        before the policy head computes the action.

Both rollouts are evaluated with the full vmax metrics suite.  The difference
in aggregate scores (Δscore = steered − baseline) is the causal fingerprint of
the feature: a feature that reliably improves safety when activated but was
previously weakly expressed is a candidate for a "beneficial safety concept".

Key implementation note
    The environment state diverges after the first steered action.  That's
    expected and desired — the whole point is to see how a single feature's
    influence compounds through time.  Do NOT reset the environment between
    timesteps.

Usage:
    python -m sae_interpretability.rollout_steering \\
        --run_dir ../../runs/PPO_VEC_WAYFORMER \\
        --dataset ../../training.tfrecord \\
        --sae_checkpoint sae_model.pt \\
        --feature_idx 42 \\
        --temperatures -5.0 -2.0 -1.0 1.0 2.0 5.0 \\
        --n_scenarios 5 \\
        --max_steps 80 \\
        --output data/sae_interpretability/rollout_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sae_interpretability.utils import (
    apply_phd_style,
    save_figure,
    save_incremental_results,
    _json_default,
    PHD_BLUE, PHD_ORANGE, PHD_GREEN, PHD_PURPLE, PHD_GRAY, PHD_TEAL,
)

apply_phd_style()
from waymax import datatypes as waymax_datatypes

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from sae_interpretability.causal_steering import CausalSteerer
from sae_interpretability.config import SAEConfig
from vmax.simulator.metrics import (
    _VMAX_METRICS_REGISTRY,
    AtFaultCollisionMetric,
    ComfortMetric,
    DrivingDirectionComplianceMetric,
    OnMultipleLanesMetric,
    ProgressRatioMetric,
    RunRedLightMetric,
    SpeedLimitViolationMetric,
    TimeToCollisionMetric,
)


# ---------------------------------------------------------------------------
# Metrics collected per rollout step
# ---------------------------------------------------------------------------

_STEP_METRICS: Dict[str, Any] = {
    "run_red_light":               RunRedLightMetric(),
    "ttc":                         TimeToCollisionMetric(),
    "at_fault_collision":          AtFaultCollisionMetric(),
    "comfort":                     ComfortMetric(),
    "speed_limit":                 SpeedLimitViolationMetric(),
    "on_multiple_lanes":           OnMultipleLanesMetric(),
    "driving_direction_compliance": DrivingDirectionComplianceMetric(),
    "progress_ratio_nuplan":       ProgressRatioMetric(),
}


# Metric aggregation semantics — mirrors vmax collector
_MAX_METRICS   = {"run_red_light", "at_fault_collision", "overlap", "offroad"}
_FINAL_METRICS = {"progress_ratio_nuplan", "sdc_progression"}


def _aggregate_buffers(
    metric_bufs: Dict[str, np.ndarray],
    n_steps: int,
) -> Dict[str, float]:
    """Aggregate fixed-size metric buffers into episode-level scalars.

    Buffers are pre-allocated to max_steps and filled with -2.0 as sentinel.
    Only the first n_steps entries are meaningful.

    Args:
        metric_bufs: Dict of (max_steps,) arrays, sentinel=-2.0.
        n_steps: Actual episode length.

    Returns:
        Episode-level scalar for each metric.
    """
    agg: Dict[str, float] = {}
    for k, buf in metric_bufs.items():
        vals = buf[:n_steps]
        valid = vals[vals >= -1.0]   # exclude -2.0 sentinels
        if len(valid) == 0:
            agg[k] = float('nan')
        elif k in _MAX_METRICS:
            agg[k] = float(np.max(valid))
        elif k in _FINAL_METRICS:
            agg[k] = float(valid[-1])
        else:
            agg[k] = float(np.mean(valid))
    return agg


# ---------------------------------------------------------------------------
# RolloutSteerer — JAX-optimized paired rollouts
# ---------------------------------------------------------------------------

class RolloutSteerer(CausalSteerer):
    """Run paired baseline / steered full-episode rollouts and compare metrics.

    JAX optimizations applied
    ─────────────────────────
    1. JIT  – The combined (encode → steer → policy → step) body function is
              compiled once via jax.jit, eliminating Python overhead per step.
    2. while_loop – Episodes run inside jax.lax.while_loop: the "done" flag
              is checked on-device via jnp.any(~done), never transferred to
              the host, so there is no device→host sync per timestep.
    3. Fixed buffers – Metric values are stored in pre-allocated (max_steps,)
              JAX arrays using .at[t].set(v) instead of Python list appends,
              keeping the entire carry on the accelerator.
    4. vmap  – All steering temperatures are vectorised with jax.vmap over
              alpha, running every temperature in a single compiled XLA
              program. Baseline runs once and is shared.
    """

    def setup(self) -> "RolloutSteerer":
        super().setup()
        self._build_jit_fns()
        return self

    # ------------------------------------------------------------------
    # Opt 1 – JIT-compiled per-step computation
    # ------------------------------------------------------------------

    def _build_jit_fns(self) -> None:
        """Build and JIT-compile the forward-pass functions once at setup.

        Captures model weights in closures so every call avoids re-tracing.
        """
        encoder        = self.encoder
        enc_params     = self.encoder_params
        sae_w          = self._sae_w
        act_mean       = self._act_mean_jnp
        act_std        = self._act_std_jnp
        policy_fc      = self._policy_fc_params
        policy_dense   = self._policy_dense_params

        @jax.jit
        def encode_obs(obs: jnp.ndarray) -> jnp.ndarray:
            """obs [1, D] → h [D]."""
            h, _ = encoder.apply({'params': enc_params}, obs)
            return h[0]

        @jax.jit
        def apply_steer(h: jnp.ndarray, feature_idx: int, alpha: jnp.ndarray) -> jnp.ndarray:
            """Causal intervention: h [D] → h_mod [D]."""
            h_norm = (h - act_mean) / act_std
            pre_bias = sae_w.get('pre_bias', jnp.zeros_like(h_norm))
            f = jnp.maximum((h_norm - pre_bias) @ sae_w['W_enc'] + sae_w['b_enc'], 0.0)
            h_rec = f @ sae_w['W_dec'] + sae_w['b_dec']
            f_s   = f.at[feature_idx].add(alpha)
            h_s   = f_s @ sae_w['W_dec'] + sae_w['b_dec']
            return h + (h_s - h_rec) * act_std

        @jax.jit
        def policy_head(h: jnp.ndarray) -> jnp.ndarray:
            """h [D] → action [1, action_dim]."""
            x = h[None]  # [1, D]
            for key in sorted(policy_fc.keys()):
                layer = policy_fc[key]
                x = jnp.maximum(x @ layer['kernel'] + layer['bias'], 0.0)
            return x @ policy_dense['kernel'] + policy_dense['bias']

        self._jit_encode  = encode_obs
        self._jit_steer   = apply_steer
        self._jit_policy  = policy_head

    # ------------------------------------------------------------------
    # Opt 2 & 3 – while_loop + fixed-size buffers
    # ------------------------------------------------------------------

    def _run_episode_while(
        self,
        init_transition,
        step_fn,          # (env_transition, action) → env_transition
        forward_fn,       # env_transition → (action_data [1,2], policy_out [1,D])
        max_steps: int,
        metric_keys: List[str],
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, int]:
        """Run one episode with jax.lax.while_loop and fixed-size metric buffers.

        All state stays on-device until after the loop exits.

        Args:
            init_transition: Initial EnvTransition (from env.reset).
            step_fn:   Wrapped env.step (JIT-compiled).
            forward_fn: Maps env_transition → (action_data, policy_vec).
            max_steps: Episode length cap.
            metric_keys: Metric names to collect from env_transition.metrics.

        Returns:
            (metric_bufs, accel_buf, steer_buf, n_steps)
        """
        # Pre-allocate fixed-size buffers (Opt 3)
        accel_buf = jnp.full((max_steps,), -2.0)
        steer_buf = jnp.full((max_steps,), -2.0)
        metric_bufs = {k: jnp.full((max_steps,), -2.0) for k in metric_keys}

        def cond_fn(carry):
            env_trans, t, _accel, _steer, _mbufs = carry
            # On-device done check — no host sync (Opt 2)
            return jnp.logical_and(
                t < max_steps,
                jnp.logical_not(jnp.any(env_trans.done)),
            )

        def body_fn(carry):
            env_trans, t, accel_b, steer_b, mbufs = carry

            # Forward pass (JIT-compiled closure, Opt 1)
            action_data, policy_vec = forward_fn(env_trans)

            # Store actions into fixed buffers (Opt 3)
            accel_b = accel_b.at[t].set(action_data[0, 0])
            steer_b = steer_b.at[t].set(action_data[0, 1])

            # Step the environment
            action = waymax_datatypes.Action(
                data=action_data,
                valid=jnp.ones((1, 1), dtype=jnp.bool_),
            )
            env_trans_new = step_fn(env_trans, action)

            # Read metrics from wrapper's pre-computed dict (Opt 3)
            for k in mbufs:
                if k in env_trans_new.metrics:
                    mbufs[k] = mbufs[k].at[t].set(env_trans_new.metrics[k][0])

            return env_trans_new, t + 1, accel_b, steer_b, mbufs

        init_carry = (init_transition, jnp.int32(0), accel_buf, steer_buf, metric_bufs)
        final_trans, n_steps, accel_buf, steer_buf, metric_bufs = jax.lax.while_loop(
            cond_fn, body_fn, init_carry
        )

        return (
            {k: np.array(jax.device_get(v)) for k, v in metric_bufs.items()},
            np.array(jax.device_get(accel_buf)),
            np.array(jax.device_get(steer_buf)),
            int(jax.device_get(n_steps)),
        )

    # ------------------------------------------------------------------
    # Single paired rollout  (baseline once + steered vmapped over α)
    # ------------------------------------------------------------------

    def run_paired_rollout(
        self,
        scenario_batch,
        feature_idx: int,
        temperatures: List[float],
        max_steps: int = 80,
    ) -> Dict[str, Any]:
        """Run one baseline and N steered rollouts from the same initial state.

        The baseline runs once with while_loop.  All steered temperatures run
        via jax.vmap over alpha (Opt 4) — one compiled XLA program covers every α.

        Key invariant: NO env.reset between timesteps in either arm.  State
        diverges naturally after the first steered action.

        Args:
            scenario_batch: Batched scenario from the data generator.
            feature_idx:    SAE feature index to steer.
            temperatures:   All α values to test in parallel.
            max_steps:      Maximum episode length.

        Returns:
            Dict with per-α results and aggregated metrics for both arms.
        """
        # JIT-compile env.step once
        jit_step = jax.jit(self.env.step)

        metric_keys = list(self.env.reset(scenario_batch).metrics.keys())

        # ── Baseline rollout (once per scenario) ─────────────────────────
        init_trans = self.env.reset(scenario_batch)

        def baseline_forward(env_trans):
            h = self._jit_encode(env_trans.observation)
            policy_out = self._jit_policy(h)
            return policy_out[:, :2], policy_out

        b_mbufs, b_accel, b_steer, b_steps = self._run_episode_while(
            init_trans, jit_step, baseline_forward, max_steps, metric_keys
        )
        baseline_metrics = _aggregate_buffers(b_mbufs, b_steps)

        # ── Steered rollouts — Opt 4: vmap over temperatures ─────────────
        # Build a function that takes a scalar alpha and runs one steered
        # episode using scan (fixed max_steps required by vmap).
        feature_idx_j = jnp.int32(feature_idx)
        alphas_j = jnp.array(temperatures, dtype=jnp.float32)  # [n_temps]

        def steered_scan_step(carry, _):
            """One step of a steered rollout (scan-compatible, α in closure)."""
            env_trans, done, alpha = carry

            h     = self._jit_encode(env_trans.observation)
            h_mod = self._jit_steer(h, feature_idx_j, alpha)
            policy_out = self._jit_policy(h_mod)
            action_data = policy_out[:, :2]

            action = waymax_datatypes.Action(
                data=action_data,
                valid=jnp.ones((1, 1), dtype=jnp.bool_),
            )
            env_trans_new = jit_step(env_trans, action)
            new_done = jnp.logical_or(done, jnp.any(env_trans_new.done))

            # Freeze state after done (mask update so scan can run fixed iters)
            env_trans_next = jax.lax.cond(
                done,
                lambda: env_trans,
                lambda: env_trans_new,
            )

            step_metrics = {
                k: jax.lax.cond(
                    done,
                    lambda: jnp.float32(-2.0),
                    lambda: env_trans_new.metrics[k][0] if k in env_trans_new.metrics
                            else jnp.float32(-2.0),
                )
                for k in metric_keys
            }

            step_out = (
                action_data[0, 0],   # accel scalar
                action_data[0, 1],   # steer scalar
                step_metrics,        # dict of scalars
                done,                # was done BEFORE this step (mask)
            )
            return (env_trans_next, new_done, alpha), step_out

        def run_steered_for_alpha(alpha: jnp.ndarray):
            """Full steered episode for one alpha value."""
            init_carry = (init_trans, jnp.bool_(False), alpha)
            _, outputs = jax.lax.scan(
                steered_scan_step, init_carry, None, length=max_steps
            )
            return outputs  # (accel[T], steer[T], {metric: [T]}, done_mask[T])

        # vmap over the alpha axis (Opt 4) — shapes become [n_temps, max_steps, ...]
        vmapped_steered = jax.jit(jax.vmap(run_steered_for_alpha))
        s_accel_all, s_steer_all, s_mbufs_all, s_done_all = vmapped_steered(alphas_j)

        # Transfer to host once
        s_accel_all = np.array(jax.device_get(s_accel_all))   # [n_temps, max_steps]
        s_steer_all = np.array(jax.device_get(s_steer_all))
        s_done_all  = np.array(jax.device_get(s_done_all))    # [n_temps, max_steps] bool
        s_mbufs_all = {k: np.array(jax.device_get(v)) for k, v in s_mbufs_all.items()}

        # ── Build per-temperature result dicts ────────────────────────────
        per_alpha_results: List[Dict] = []
        for ti, alpha in enumerate(temperatures):
            # Effective episode length = first step where done was True
            done_steps = np.where(s_done_all[ti])[0]
            n_steps_s  = int(done_steps[0]) if len(done_steps) else max_steps

            mbufs_ti = {k: s_mbufs_all[k][ti] for k in metric_keys}
            steered_metrics = _aggregate_buffers(mbufs_ti, n_steps_s)

            delta_metrics = {
                k: steered_metrics[k] - baseline_metrics.get(k, float('nan'))
                if not (np.isnan(steered_metrics[k]) or np.isnan(baseline_metrics.get(k, float('nan'))))
                else float('nan')
                for k in steered_metrics
            }

            per_alpha_results.append({
                'alpha':            float(alpha),
                'feature_idx':      feature_idx,
                'n_steps_baseline': b_steps,
                'n_steps_steered':  n_steps_s,
                'baseline_metrics': baseline_metrics,
                'steered_metrics':  steered_metrics,
                'delta_metrics':    delta_metrics,
                'per_step_baseline': {
                    'accel': b_accel[:b_steps].tolist(),
                    'steer': b_steer[:b_steps].tolist(),
                },
                'per_step_steered': {
                    'accel': s_accel_all[ti, :n_steps_s].tolist(),
                    'steer': s_steer_all[ti, :n_steps_s].tolist(),
                },
            })

        return per_alpha_results

    # ------------------------------------------------------------------
    # Outer experiment loop
    # ------------------------------------------------------------------

    def run_rollout_experiment(
        self,
        features: Dict[str, int],
        temperatures: List[float],
        n_scenarios: int = 5,
        max_steps: int = 80,
        output_path: str = "data/sae_interpretability/rollout_results.json",
    ) -> Dict[str, Any]:
        """Run paired rollouts for multiple named features across multiple scenarios.

        For each feature and scenario, baseline runs once and all temperatures
        are evaluated in parallel via vmap.  Results are keyed by feature name
        and saved together in a single JSON.

        Args:
            features:     Dict mapping human-readable name → SAE feature index.
                          Example: {"speed_limit": 42, "lane_change": 17}
            temperatures: List of α values (vmapped in parallel per feature).
            n_scenarios:  Number of independent scenarios per feature.
            max_steps:    Maximum episode length.
            output_path:  Output JSON path.

        Returns:
            Full results dict keyed by feature name.
        """
        from vmax.simulator import make_data_generator, datasets

        out_dir = Path(output_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\n[RolloutSteerer] {len(features)} feature(s)  "
            f"α={temperatures}  {n_scenarios} scenarios  max_steps={max_steps}\n"
            f"  Optimizations: JIT + while_loop + fixed_buffers + vmap({len(temperatures)} temps)"
        )

        all_feature_results: Dict[str, Any] = {}

        for feature_name, feature_idx in features.items():
            print(f"\n{'='*60}")
            print(f"  Feature: '{feature_name}'  (idx={feature_idx})")
            print(f"{'='*60}")

            data_gen = make_data_generator(
                path=datasets.get_dataset(self.dataset_path),
                max_num_objects=self.config["max_num_objects"],
                include_sdc_paths=not self.config.get("waymo_dataset", False),
                batch_dims=(1,),
                seed=42,
                repeat=1,
            )

            all_scenarios: List[Dict] = []

            for scen_idx, scenario_batch in enumerate(data_gen):
                if scen_idx >= n_scenarios:
                    break

                print(f"\n  ── Scenario {scen_idx + 1}/{n_scenarios} ──")
                per_alpha = self.run_paired_rollout(
                    scenario_batch, feature_idx, temperatures, max_steps
                )

                for res in per_alpha:
                    res['scenario_id'] = scen_idx

                all_scenarios.append({
                    'scenario_id': scen_idx,
                    'temperatures': per_alpha,
                })

            # ── Per-feature summary ────────────────────────────────────────
            summary = _summarise_rollout_results(all_scenarios, temperatures, feature_idx)

            print(f"\n  Summary for '{feature_name}' (idx={feature_idx}):")
            _print_summary_table(summary, temperatures, all_scenarios)

            plot_rollout_metrics(
                summary, temperatures, feature_idx, out_dir,
                feature_name=feature_name,
                all_scenarios=all_scenarios,
            )
            plot_action_trajectories(
                all_scenarios, temperatures, feature_idx, out_dir,
                feature_name=feature_name,
            )

            feature_data = {
                'feature_name': feature_name,
                'feature_idx':  feature_idx,
                'temperatures': temperatures,
                'n_scenarios':  len(all_scenarios),
                'max_steps':    max_steps,
                'summary':      summary,
                'scenarios':    all_scenarios,
            }
            all_feature_results[feature_name] = feature_data

            save_incremental_results(
                output_path=output_path,
                feature_name=feature_name,
                feature_data=feature_data,
                all_features_meta={name: idx for name, idx in features.items()},
                temperatures=temperatures,
                n_scenarios=n_scenarios,
                max_steps=max_steps,
            )
            print(f"\n  [Incremental Save] Feature '{feature_name}' appended to {output_path}")

        print(f"\n[RolloutSteerer] All {len(features)} features completed. Results → {output_path}")
        return {
            'features':     {name: idx for name, idx in features.items()},
            'temperatures': temperatures,
            'n_scenarios':  n_scenarios,
            'max_steps':    max_steps,
            'results':      all_feature_results,
        }




# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _summarise_rollout_results(
    all_scenarios: List[Dict],
    temperatures: List[float],
    feature_idx: int,
) -> Dict:
    """Build cross-scenario aggregated statistics keyed by temperature."""
    summary: Dict[str, Any] = {}

    for alpha in temperatures:
        key = f"alpha_{alpha:+.2f}"
        delta_lists: Dict[str, List[float]] = {}

        for scen in all_scenarios:
            for temp_result in scen['temperatures']:
                if temp_result['alpha'] != alpha:
                    continue
                for metric_name, dv in temp_result['delta_metrics'].items():
                    delta_lists.setdefault(metric_name, []).append(dv)

        agg: Dict[str, Any] = {}
        for metric_name, vals in delta_lists.items():
            arr = np.array([v for v in vals if not np.isnan(v)])
            if len(arr) == 0:
                continue
            agg[metric_name] = {
                'mean':   float(arr.mean()),
                'std':    float(arr.std()),
                'median': float(np.median(arr)),
                'p5':     float(np.percentile(arr, 5)),
                'p95':    float(np.percentile(arr, 95)),
                'sign_flip_frac': float(
                    min((arr > 0).sum(), (arr < 0).sum()) / len(arr)
                ),
            }
        summary[key] = {'alpha': alpha, 'metrics': agg}

    return summary


def _print_summary_table(
    summary: Dict,
    temperatures: List[float],
    all_scenarios: List[Dict],
) -> None:
    """Print a compact ASCII table of baseline absolute value and mean Δmetric per temperature."""
    metric_names: List[str] = []
    for alpha in temperatures:
        key = f"alpha_{alpha:+.2f}"
        if key in summary:
            metric_names = list(summary[key]['metrics'].keys())
            break

    if not metric_names:
        return

    # Collect baseline values (mean across scenarios for each metric)
    baseline_vals: Dict[str, float] = {}
    for mn in metric_names:
        vals = []
        for scen in all_scenarios:
            for res in scen.get('temperatures', []):
                bm = res.get('baseline_metrics', {})
                if mn in bm and not np.isnan(bm[mn]):
                    vals.append(bm[mn])
                    break  # one baseline per scenario is enough
        baseline_vals[mn] = float(np.mean(vals)) if vals else float('nan')

    col_w = 10
    # Header: baseline column + one column per alpha
    header = (
        f"{'metric':<30}"
        f"{'baseline':>{col_w}}"
        + "".join(f"{a:>{col_w}.2f}" for a in temperatures)
    )
    print("\n" + header)
    print("-" * len(header))

    for mn in metric_names:
        bv = baseline_vals.get(mn, float('nan'))
        row = f"{mn:<30}{bv:>{col_w}.4f}"
        for alpha in temperatures:
            key = f"alpha_{alpha:+.2f}"
            val = summary.get(key, {}).get('metrics', {}).get(mn, {}).get('mean', float('nan'))
            row += f"{val:>{col_w}.4f}"
        print(row)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_rollout_metrics(
    summary: Dict,
    temperatures: List[float],
    feature_idx: int,
    out_dir: Path,
    feature_name: str = "",
    all_scenarios: Optional[List[Dict]] = None,
) -> None:
    """Per-scenario + aggregate bar charts of Δmetric per steering temperature.

    One figure is produced **per scenario** (showing every metric as a
    subplot column, one bar per temperature).  An additional *aggregate*
    figure shows the cross-scenario mean ± std.  All figures are saved in
    both PNG (300 dpi) and PDF (vector) formats.

    Args:
        summary:        Cross-scenario summary from ``_summarise_rollout_results``.
        temperatures:   List of steering temperatures α.
        feature_idx:    SAE feature index being steered.
        out_dir:        Output directory (PNG + PDF written here).
        feature_name:   Human-readable feature label.
        all_scenarios:  Raw per-scenario results for individual figures.
    """
    label     = feature_name if feature_name else str(feature_idx)
    safe_name = label.replace(' ', '_')
    out_dir   = Path(out_dir)

    metric_names: List[str] = []
    for alpha in temperatures:
        key = f"alpha_{alpha:+.2f}"
        if key in summary:
            metric_names = list(summary[key]['metrics'].keys())
            break

    if not metric_names:
        return

    n_metrics = len(metric_names)
    n_temps   = len(temperatures)

    # ── Sub-directory for per-scenario figures ────────────────────────────
    scen_dir = out_dir / f'f{feature_idx}_{safe_name}' / 'metrics_per_scenario'
    scen_dir.mkdir(parents=True, exist_ok=True)

    # ── 1.  Per-scenario figures ──────────────────────────────────────────
    if all_scenarios:
        for scen in all_scenarios:
            scen_id = scen.get('scenario_id', 0)

            # Collect per-scenario delta values
            scen_means: Dict[str, List[float]] = {mn: [] for mn in metric_names}
            for temp_result in scen.get('temperatures', []):
                dm = temp_result.get('delta_metrics', {})
                for mn in metric_names:
                    scen_means[mn].append(dm.get(mn, float('nan')))

            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(4.0 * n_cols, 4.2 * n_rows),
                sharey=False,
                squeeze=False,
            )
            axes_flat = axes.flatten()
            # Remove any unused/excess axes in the grid
            for idx in range(n_metrics, len(axes_flat)):
                fig.delaxes(axes_flat[idx])

            for ax, mn in zip(axes_flat[:n_metrics], metric_names):
                vals      = np.array(scen_means[mn])
                bar_cols  = [PHD_TEAL if v >= 0 else PHD_ORANGE for v in vals]
                x         = np.arange(n_temps)

                ax.bar(x, vals, color=bar_cols, edgecolor='white',
                       alpha=0.88, width=0.65)
                ax.axhline(0, color='#888', linewidth=1.0, linestyle='--')
                ax.set_xticks(x)
                ax.set_xticklabels(
                    [f"{a:+.1f}" for a in temperatures],
                    fontsize=8, rotation=45, ha='right',
                )
                ax.set_title(mn.replace('_', ' '), fontsize=9, fontweight='bold')
                ax.set_xlabel('α (steering temperature)', fontsize=8)
                ax.set_ylabel('Δ (steered − baseline)', fontsize=8)
                ax.tick_params(labelsize=7)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            if n_rows > 1:
                fig.suptitle(
                    f"'{label}' (f{feature_idx}) — Scenario {scen_id}: Δmetric per α",
                    fontsize=12, fontweight='bold', y=0.98,
                )
                plt.tight_layout(rect=[0, 0, 1, 0.95])
            else:
                fig.suptitle(
                    f"'{label}' (f{feature_idx}) — Scenario {scen_id}: Δmetric per α",
                    fontsize=12, fontweight='bold', y=1.02,
                )
                plt.tight_layout()
            stem = f'scenario_{scen_id:02d}_metrics'
            save_figure(fig, scen_dir, stem)
            plt.close(fig)

    # ── 2. Aggregate figure (mean ± std across all scenarios) ─────────────
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 4.5 * n_rows),
        sharey=False,
        squeeze=False,
    )
    axes_flat = axes.flatten()
    # Remove any unused/excess axes in the grid
    for idx in range(n_metrics, len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    for ax, mn in zip(axes_flat[:n_metrics], metric_names):
        means_arr = np.array([
            summary.get(f"alpha_{a:+.2f}", {}).get('metrics', {}).get(mn, {}).get('mean', float('nan'))
            for a in temperatures
        ])
        stds_arr  = np.array([
            summary.get(f"alpha_{a:+.2f}", {}).get('metrics', {}).get(mn, {}).get('std', 0.0)
            for a in temperatures
        ])
        bar_cols  = [PHD_TEAL if m >= 0 else PHD_ORANGE for m in means_arr]
        x         = np.arange(n_temps)

        ax.bar(x, means_arr, yerr=stds_arr, color=bar_cols,
               edgecolor='white', alpha=0.88, capsize=5, width=0.65,
               error_kw=dict(elinewidth=1.2, ecolor=PHD_GRAY))
        ax.axhline(0, color='#888', linewidth=1.0, linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{a:+.1f}" for a in temperatures],
            fontsize=8, rotation=45, ha='right',
        )
        ax.set_title(mn.replace('_', ' '), fontsize=9, fontweight='bold')
        ax.set_xlabel('α (steering temperature)', fontsize=8)
        ax.set_ylabel('Δ (steered − baseline)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if n_rows > 1:
        fig.suptitle(
            f"'{label}' (f{feature_idx}) — Aggregate Δmetric per α "
            f"(mean ± std over scenarios)",
            fontsize=12, fontweight='bold', y=0.98,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.suptitle(
            f"'{label}' (f{feature_idx}) — Aggregate Δmetric per α "
            f"(mean ± std over scenarios)",
            fontsize=12, fontweight='bold', y=1.02,
        )
        plt.tight_layout()
    agg_dir = out_dir / f'f{feature_idx}_{safe_name}'
    agg_dir.mkdir(parents=True, exist_ok=True)
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        save_figure(fig, agg_dir, 'aggregate_metrics')
    plt.close(fig)
    print(f"[RolloutSteerer] Metrics plots → {agg_dir}/")


def plot_action_trajectories(
    all_scenarios: List[Dict],
    temperatures: List[float],
    feature_idx: int,
    out_dir: Path,
    feature_name: str = "",
) -> None:
    """One figure per scenario per temperature: baseline vs steered trajectories.

    Each figure contains 2 rows (acceleration, steering angle) and 2 columns
    (baseline | steered), with a shared x-axis (timestep).
    All figures are saved in both PNG (300 dpi) and PDF (vector) formats.

    Output structure::

        <out_dir>/f<idx>_<name>/trajectories_per_scenario/
            scenario_00_alpha_p1_0.png
            scenario_00_alpha_p1_0.pdf
            ...
    """
    label     = feature_name if feature_name else str(feature_idx)
    safe_name = label.replace(' ', '_')
    out_dir   = Path(out_dir)
    n_scen    = len(all_scenarios)
    if n_scen == 0:
        return

    traj_dir = out_dir / f'f{feature_idx}_{safe_name}' / 'trajectories_per_scenario'
    traj_dir.mkdir(parents=True, exist_ok=True)

    for alpha in temperatures:
        alpha_str = f"{alpha:+.1f}".replace('+', 'p').replace('-', 'n').replace('.', '_')

        for scen in all_scenarios:
            scen_id     = scen.get('scenario_id', 0)
            temp_result = next(
                (r for r in scen['temperatures'] if r['alpha'] == alpha), None
            )
            if temp_result is None:
                continue

            # ── 2-row × 1-col layout: Acceleration and Steering Angle comparative overlay ──
            fig, axes = plt.subplots(
                2, 1,
                figsize=(8, 6.5),
                sharex=True,
                gridspec_kw={'hspace': 0.3},
            )

            row_labels  = ['Acceleration', 'Steering Angle']
            dim_keys    = ['accel', 'steer']
            base_data   = temp_result.get('per_step_baseline', {})
            steer_data  = temp_result.get('per_step_steered',  {})

            for row_idx, (dim_name, row_label) in enumerate(zip(dim_keys, row_labels)):
                ax = axes[row_idx]

                vals_base  = base_data.get(dim_name, [])
                vals_steer = steer_data.get(dim_name, [])

                t_base  = np.arange(len(vals_base))
                t_steer = np.arange(len(vals_steer))

                # Plot Baseline (Teal/Blue)
                ax.plot(t_base, vals_base, label='Baseline', color=PHD_BLUE, linewidth=1.8, alpha=0.8)
                ax.fill_between(t_base, 0, vals_base, color=PHD_BLUE, alpha=0.06)

                # Plot Steered (Orange/Red)
                ax.plot(t_steer, vals_steer, label=f'Steered (α={alpha:+.1f})', color=PHD_ORANGE, linewidth=1.8, alpha=0.9)
                ax.fill_between(t_steer, 0, vals_steer, color=PHD_ORANGE, alpha=0.12)

                ax.axhline(0, color='#aaa', linewidth=0.7, linestyle=':')
                ax.set_ylabel(row_label, fontsize=10, fontweight='bold')
                ax.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=8, loc='upper right')

                ax.tick_params(labelsize=8)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='y', linewidth=0.5, alpha=0.4)

                # X-label (bottom row only)
                if row_idx == 1:
                    ax.set_xlabel('Timestep', fontsize=9, fontweight='bold')

            fig.suptitle(
                f"'{label}' (f{feature_idx}) — Scenario {scen_id}  α={alpha:+.1f}: "
                f"Action Trajectories",
                fontsize=12, fontweight='bold', y=1.01,
            )

            stem = f'scenario_{scen_id:02d}_alpha_{alpha_str}'
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                save_figure(fig, traj_dir, stem)
            plt.close(fig)

    global_dir = out_dir / f'f{feature_idx}_{safe_name}'
    print(f"[RolloutSteerer] Trajectory plots → {global_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_features(raw: List[str]) -> Dict[str, int]:
    """Parse 'name:idx' strings into a {name: idx} dict.

    Example input: ["speed_limit:42", "lane_change:17"]
    """
    features: Dict[str, int] = {}
    for token in raw:
        if ':' not in token:
            raise argparse.ArgumentTypeError(
                f"Feature '{token}' must be in 'name:idx' format, e.g. 'speed:42'"
            )
        name, idx_str = token.rsplit(':', 1)
        features[name.strip()] = int(idx_str.strip())
    return features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-timestep causal steering with vmax rollout metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Feature specification (choose one):
              --features name:idx [name:idx ...]   inline pairs
              --features_file path/to/features.json  JSON dict {"name": idx, ...}

            Example:
              python -m sae_interpretability.rollout_steering \\\\
                --run_dir runs/PPO \\\\
                --dataset training.tfrecord \\\\
                --sae_checkpoint sae.pt \\\\
                --features speed_limit:42 lane_change:17 braking:93 \\\\
                --temperatures -5.0 -1.0 1.0 5.0
        """),
    )
    parser.add_argument("--run_dir",        required=True)
    parser.add_argument("--dataset",        required=True)
    parser.add_argument("--sae_checkpoint", required=True)

    feat_group = parser.add_mutually_exclusive_group(required=True)
    feat_group.add_argument(
        "--features", nargs="+", metavar="NAME:IDX",
        help="One or more features as 'name:idx' pairs, e.g. speed_limit:42",
    )
    feat_group.add_argument(
        "--features_file", metavar="PATH",
        help="JSON file mapping feature names to indices: {\"speed_limit\": 42, ...}",
    )

    parser.add_argument(
        "--temperatures", type=float, nargs="+",
        default=[-5.0, -2.0, -1.0, 1.0, 2.0, 5.0],
    )
    parser.add_argument("--n_scenarios", type=int, default=5)
    parser.add_argument("--max_steps",   type=int, default=80)
    parser.add_argument(
        "--output",
        default="data/sae_interpretability/rollout_results.json",
    )
    parser.add_argument("--checkpoint", default="model_final.pkl")
    args = parser.parse_args()

    # Resolve features dict
    if args.features:
        features = _parse_features(args.features)
    else:
        with open(args.features_file) as f:
            features = {str(k): int(v) for k, v in json.load(f).items()}

    print(f"[RolloutSteerer] Features to steer: {features}")

    cfg = SAEConfig()
    steerer = RolloutSteerer(
        args.run_dir, args.dataset, args.sae_checkpoint, cfg, args.checkpoint
    )
    steerer.setup()
    steerer.run_rollout_experiment(
        features=features,
        temperatures=args.temperatures,
        n_scenarios=args.n_scenarios,
        max_steps=args.max_steps,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
