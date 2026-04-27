"""CBM-V1 training loop — production version.

Mirrors V-Max's sac_trainer.py in structure but uses CBM-SAC networks
and exposes concept_loss as a first-class logged metric.

Usage (called from train_cbm.py):
    cbm_trainer.train(config, output_dir)
"""

from __future__ import annotations

import logging
import os
import pickle
import sys
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from etils import epath
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "V-Max"))

from waymax import dynamics

from vmax.simulator import make_env_for_training, make_data_generator
from vmax.agents import datatypes, pipeline
from vmax.agents.pipeline import inference, pmap as pmap_utils
from vmax.agents.learning.replay_buffer import ReplayBuffer
from concepts.types import ObservationConfig
from concepts.adapters import observation_to_concept_input
from concepts.registry import extract_all_concepts

from cbm_v1.config import CBMConfig
from cbm_v1.networks import CBMPolicyNetwork
import cbm_v1.cbm_sac_factory as cbm_factory

logger = logging.getLogger(__name__)


# ── Checkpoint utilities ──────────────────────────────────────────────

def save_params(path: str, params: Any) -> None:
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


def load_pretrained_params(pretrained_dir: str) -> Any:
    """Load params from a V-Max checkpoint directory.

    V-Max saves checkpoints under <run_dir>/model/model_final.pkl
    or <run_dir>/model/model_<step>.pkl.
    """
    import glob

    # V-Max places checkpoints in a 'model' subdirectory
    search_dirs = [
        pretrained_dir,
        os.path.join(pretrained_dir, "model"),
    ]

    candidates = []
    for d in search_dirs:
        final = os.path.join(d, "model_final.pkl")
        if os.path.exists(final):
            candidates.append(final)
        numbered = sorted(
            glob.glob(os.path.join(d, "model_*.pkl")),
            key=lambda p: int(p.split("model_")[-1].replace(".pkl", ""))
            if p.split("model_")[-1].replace(".pkl", "").isdigit() else -1,
        )
        candidates += numbered[::-1]  # highest step first

    for path in candidates:
        if os.path.exists(path):
            logger.info(f"Loading pretrained params from {path}")
            from vmax.scripts.evaluate.utils import load_params
            return load_params(path)

    raise FileNotFoundError(
        f"No .pkl checkpoint found in {pretrained_dir} or {pretrained_dir}/model/. "
        f"Expected model_final.pkl or model_<step>.pkl."
    )


# ── Main training function ────────────────────────────────────────────

def train(
    pretrained_dir: str,
    data_path: str,
    output_dir: str,
    run_name: str,
    # Architecture
    mode: str,
    num_parallel_envs: int,
    num_episodes_per_env: int,
    # Schedule
    scan_length: int,
    scenario_length: int,
    unroll_length: int,
    total_timesteps: int,
    learning_start: int,
    # SAC
    alpha: float,
    discount: float,
    tau: float,
    learning_rate: float,
    batch_size: int,
    grad_updates_per_step: int,
    buffer_size: int,
    # CBM
    lambda_concept: float,
    num_concepts: int,
    concept_phases: tuple[int, ...],
    # Logging
    log_freq: int,
    save_freq: int,
    seed: int,
    # Observation config (from pretrained run)
    observation_config_dict: dict,
    network_config: dict,
    encoder_remap: dict,
    obs_type_remap: dict,
    termination_keys: list,
    reward_type: str,
    reward_config: dict,
    disable_tqdm: bool = False,
) -> None:
    """Train CBM-V1 on WOMD.

    Args:
        pretrained_dir:     Path to V-Max checkpoint directory (for encoder init).
        data_path:          Path to WOMD tfrecord file.
        output_dir:         Root directory for run outputs.
        run_name:           Subdirectory name for this run.
        mode:               "frozen" (encoder fixed) or "joint" (end-to-end).
        num_parallel_envs:  Parallel env streams per device (outer vmap).
        num_episodes_per_env: Episode pool per stream (inner vmap).
        scan_length:        jax.lax.scan steps per training iteration.
        scenario_length:    Waymax scenario length in steps.
        unroll_length:      Steps per unroll (usually 1).
        total_timesteps:    Total environment steps to train for.
        learning_start:     Random steps before gradient updates begin.
        alpha:              SAC entropy coefficient.
        discount:           RL discount factor.
        tau:                Soft target update rate.
        learning_rate:      Adam learning rate.
        batch_size:         Transitions per gradient update.
        grad_updates_per_step: Gradient updates per scan step.
        buffer_size:        Replay buffer capacity.
        lambda_concept:     Weight for concept supervision loss.
        num_concepts:       Number of concept dimensions.
        log_freq:           Log every N iterations.
        save_freq:          Checkpoint every N iterations.
        seed:               RNG seed.
        observation_config_dict: Observation config dict (from pretrained hydra config).
        network_config:     Network architecture config dict.
        encoder_remap:      Map pretrained encoder type names to CBM names.
        obs_type_remap:     Map pretrained obs type names to CBM names.
        termination_keys:   Episode termination conditions.
        reward_type:        Reward function type.
        reward_config:      Reward config dict.
        disable_tqdm:       Suppress tqdm progress bar.
    """
    print(" CBM Training ".center(50, "="))
    print(f"  Mode      : {mode}")
    print(f"  Timesteps : {total_timesteps:,}")
    print(f"  Encoder   : {('pretrained (' + pretrained_dir + ')') if pretrained_dir and mode != 'scratch' else 'SCRATCH (random init)'}")
    print(f"  Concepts  : {num_concepts} (phases {concept_phases})")
    print(f"  Data      : {data_path}")

    rng = jax.random.PRNGKey(seed)
    num_devices = jax.local_device_count()
    print(f"  Devices   : {num_devices}")

    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── TensorBoard ──────────────────────────────────────────────────
    try:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))
        print(f"  TensorBoard: {os.path.join(run_dir, 'tb')}")
    except ImportError:
        writer = None
        print("  TensorBoard: not available (tensorboardX not installed)")

    # ── Environment ──────────────────────────────────────────────────
    print("\n-> Building environment...")
    obs_type = obs_type_remap.get(network_config.get("_obs_type", "vec"), "vec")

    env = make_env_for_training(
        max_num_objects=64,
        dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
        sdc_paths_from_data=True,
        observation_type=obs_type,
        observation_config=observation_config_dict,
        termination_keys=termination_keys,
        reward_type=reward_type,
        reward_config=reward_config,
    )
    observation_size = env.observation_spec()
    action_size = env.action_spec().data.shape[0]
    print(f"   obs_size={observation_size}, action_size={action_size}")

    # ── Data generators ──────────────────────────────────────────────
    print("-> Building data generators...")
    # batch_dims = (num_devices, num_parallel_envs, num_episodes_per_env)
    # pmap strips num_devices; inside pmap: (num_parallel_envs, num_episodes_per_env)
    # AutoResetWrapper outer-vmaps over num_parallel_envs,
    # VmapWrapper inner-vmaps over num_episodes_per_env
    data_gen = make_data_generator(
        path=data_path,
        max_num_objects=64,
        include_sdc_paths=True,
        batch_dims=(num_devices, num_parallel_envs, num_episodes_per_env),
        seed=seed,
        repeat=None,
    )

    # ── Concept targets function ─────────────────────────────────────
    obs_cfg = observation_config_dict
    concept_config = ObservationConfig(
        obs_past_num_steps=obs_cfg.get("obs_past_num_steps", 5),
        num_closest_objects=obs_cfg.get("objects", {}).get("num_closest_objects", 8),
        roadgraph_top_k=obs_cfg.get("roadgraphs", {}).get("roadgraph_top_k", 200),
        num_closest_traffic_lights=obs_cfg.get("traffic_lights", {}).get(
            "num_closest_traffic_lights", 5),
        num_target_path_points=obs_cfg.get("path_target", {}).get("num_points", 10),
        max_meters=obs_cfg.get("roadgraphs", {}).get("max_meters", 70),
    )
    unflatten_fn = env.get_wrapper_attr("features_extractor").unflatten_features

    def concept_targets_fn(observations):
        inp = observation_to_concept_input(observations, unflatten_fn, concept_config)
        out = extract_all_concepts(inp, phases=concept_phases)
        return out.normalized, out.valid

    # ── CBM config ───────────────────────────────────────────────────
    cbm_config = CBMConfig(
        num_concepts=num_concepts,
        lambda_concept=lambda_concept,
        mode=mode,
        concept_phases=concept_phases,
    )

    # ── Networks ─────────────────────────────────────────────────────
    print("-> Initializing CBM networks...")
    if mode == "scratch" or pretrained_dir in (None, "none", ""):
        pretrained_params = None
        print("   Scratch mode: encoder initialized with random weights.")
    else:
        pretrained_params = load_pretrained_params(pretrained_dir)

    rng, net_key = jax.random.split(rng)
    cbm_network, training_state, policy_fn = cbm_factory.initialize(
        action_size=action_size,
        observation_size=observation_size,
        env=env,
        learning_rate=learning_rate,
        network_config=network_config,
        cbm_config=cbm_config,
        num_devices=num_devices,
        key=net_key,
        pretrained_params=pretrained_params,
    )

    # Policy module is auto-registered inside make_networks → set_cbm_policy_module.
    # Nothing to do here; verify it is set.
    if cbm_factory._cbm_policy_module is None:
        raise RuntimeError("CBM policy module not registered. "
                           "Ensure make_networks was called first.")

    sgd_step = cbm_factory.make_sgd_step(
        cbm_network, alpha, discount, tau,
        concept_targets_fn=concept_targets_fn,
        cbm_config=cbm_config,
    )
    print("   Done.")

    # ── Replay buffer ────────────────────────────────────────────────
    print("-> Building replay buffer...")
    replay_buffer = ReplayBuffer(
        buffer_size=buffer_size // num_devices,
        batch_size=batch_size * grad_updates_per_step // num_devices,
        samples_size=num_parallel_envs,
        dummy_data_sample=datatypes.RLPartialTransition(
            observation=jnp.zeros((observation_size,)),
            action=jnp.zeros((action_size,)),
            reward=0.0,
            flag=0,
            done=0,
        ),
    )

    rng, rb_key = jax.random.split(rng)
    buffer_state = jax.pmap(replay_buffer.init)(
        jax.random.split(rb_key, num_devices)
    )

    # ── cuSolver warm-up ─────────────────────────────────────────────
    # InvertibleBicycleModel calls a JAX linear solver during env.reset.
    # If the cuSolver handle has not been initialised yet, JAX throws
    # "gpusolverDnCreate failed: cuSolver internal error" — especially
    # when VRAM is already partially occupied by the replay buffer.
    # Running a trivial linalg op here forces the handle to be created
    # in a controlled context BEFORE prefill floods the GPU.
    # This is a no-op for training logic and is safe in both frozen and
    # joint modes.
    print("--> Warming up cuSolver handle...")
    _dummy = jnp.linalg.solve(jnp.eye(4, dtype=jnp.float32),
                               jnp.ones(4, dtype=jnp.float32))
    jax.block_until_ready(_dummy)
    del _dummy
    print("    Done.")

    # ── Prefill ──────────────────────────────────────────────────────
    print(f"-> Prefilling replay buffer ({learning_start:,} steps)...")
    prefill_fn = jax.pmap(
        partial(
            pipeline.prefill_replay_buffer,
            env=env,
            replay_buffer=replay_buffer,
            action_shape=(num_parallel_envs, action_size),
            learning_start=learning_start,
        ),
        axis_name="batch",
    )
    rng, prefill_key = jax.random.split(rng)
    prefill_keys = jax.random.split(prefill_key, num_devices)

    t0 = perf_counter()
    buffer_state = prefill_fn(next(data_gen), buffer_state, prefill_keys)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), buffer_state)
    print(f"   Prefill done in {perf_counter()-t0:.1f}s "
          f"(sample_pos={int(jax.device_get(buffer_state.sample_position[0]))})") 

    # ── Training step setup ──────────────────────────────────────────
    step_fn = partial(inference.policy_step, use_partial_transition=True)
    unroll_fn = partial(
        inference.generate_unroll,
        unroll_length=unroll_length,
        env=env,
        step_fn=step_fn,
    )
    run_training = partial(
        pipeline.run_training_off_policy,
        replay_buffer=replay_buffer,
        env=env,
        learning_fn=sgd_step,
        policy_fn=policy_fn,
        unroll_fn=unroll_fn,
        grad_updates_per_step=grad_updates_per_step,
        scan_length=scan_length,
    )
    run_training_pmap = jax.pmap(run_training, axis_name="batch")

    # ── Training schedule ────────────────────────────────────────────
    env_steps_per_iter = scan_length * num_parallel_envs
    total_iters = (total_timesteps // env_steps_per_iter) + 1

    print(f"\n-> Starting training loop")
    print(f"   scan_length={scan_length}, num_parallel_envs={num_parallel_envs}")
    print(f"   env_steps_per_iter={env_steps_per_iter}, total_iters={total_iters:,}")
    print(f"   Checkpointing every {save_freq} iters → {checkpoint_dir}")
    print()

    time_training = perf_counter()
    current_step = 0

    for it in tqdm(range(total_iters), desc="CBM Training",
                   total=total_iters, dynamic_ncols=True, disable=disable_tqdm):
        rng, iter_key = jax.random.split(rng)
        iter_keys = jax.random.split(iter_key, num_devices)

        t_data = perf_counter()
        batch_scenarios = next(data_gen)
        dt_data = perf_counter() - t_data

        t_train = perf_counter()
        training_state, buffer_state, training_metrics = run_training_pmap(
            batch_scenarios, training_state, buffer_state, iter_keys,
        )
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), training_metrics)
        dt_train = perf_counter() - t_train

        # ── Metrics ─────────────────────────────────────────────────
        t_log = perf_counter()
        flat_metrics = pmap_utils.flatten_tree(training_metrics)
        flat_metrics = jax.device_get(flat_metrics)
        # Reduce each metric to a scalar (mean over all scan/grad dims).
        # We don't use _metrics.collect because it requires complete
        # episodes (ep_len_mean episode boundaries) which may not be
        # present in every scan window.
        flat_metrics = {k: float(np.nanmean(v)) for k, v in flat_metrics.items()
                        if isinstance(v, np.ndarray)}

        current_step = int(pmap_utils.unpmap(training_state.env_steps))

        if not it % log_freq:
            dt_log = perf_counter() - t_log
            wall = perf_counter() - time_training
            sps = int(env_steps_per_iter / dt_train) if dt_train > 0 else 0

            # Compute VMAX score: 1.0 - at_fault_collision rate.
            # This matches the leaderboard formula in runs_rlc/runs_accuracy.txt.
            # Logged as metrics/vmax_score so it appears in TensorBoard alongside
            # the per-component driving metrics.
            _at_fault = flat_metrics.get("metrics/at_fault_collision",
                                         flat_metrics.get("at_fault_collision", None))
            _vmax_score = (1.0 - float(_at_fault)) if _at_fault is not None else float("nan")

            metrics = {
                **flat_metrics,
                "metrics/vmax_score":     _vmax_score,
                "runtime/sps":           sps,
                "runtime/data_time":     dt_data,
                "runtime/training_time": dt_train,
                "runtime/log_time":      dt_log,
                "runtime/wall_time":     wall,
                "train/env_steps":       current_step,
                "train/rl_gradient_steps": int(pmap_utils.unpmap(
                    training_state.rl_gradient_steps)),
            }

            _log_metrics(it, current_step, total_timesteps, metrics, writer,
                         disable_tqdm)

        # ── Checkpoint ──────────────────────────────────────────────
        if save_freq > 0 and not it % save_freq and it > 0:
            ckpt_path = os.path.join(checkpoint_dir, f"model_{current_step}.pkl")
            save_params(ckpt_path, pmap_utils.unpmap(training_state.params))
            logger.info(f"Saved checkpoint: {ckpt_path}")

    dt_total = perf_counter() - time_training
    print(f"\n-> Training done in {dt_total/3600:.2f}h  ({dt_total:.0f}s)")

    # Final checkpoint
    final_path = os.path.join(checkpoint_dir, "model_final.pkl")
    save_params(final_path, pmap_utils.unpmap(training_state.params))
    print(f"-> Final checkpoint saved: {final_path}")

    if writer:
        writer.close()


# ── Helpers ───────────────────────────────────────────────────────────

def _log_metrics(
    it: int,
    step: int,
    total_timesteps: int,
    metrics: dict,
    writer,
    disable_tqdm: bool,
) -> None:
    pct = 100.0 * step / total_timesteps

    # Pull the key training metrics for the console line
    # run_training_off_policy prefixes SGD metrics with "train/"
    pol = metrics.get("train/policy_loss", float("nan"))
    val = metrics.get("train/value_loss", float("nan"))
    con = metrics.get("train/concept_loss", float("nan"))
    rew = metrics.get("ep_rew_mean", float("nan"))
    sps = metrics.get("runtime/sps", 0)

    if disable_tqdm:
        print(
            f"[iter {it:6d}] step={step:>9,} ({pct:5.1f}%) | "
            f"pol={pol:+.4f} val={val:.4f} con={con:.4f} "
            f"rew={rew:.3f} sps={sps}"
        )
    else:
        logger.info(
            f"step={step:,} ({pct:.1f}%) | "
            f"policy={pol:+.4f} value={val:.4f} concept={con:.4f} "
            f"reward={rew:.3f} sps={sps}"
        )

    if writer is not None:
        for key, value in metrics.items():
            try:
                prefix = "metrics/" if "/" not in key else ""
                if "ep_len_mean" in key or "ep_rew_mean" in key:
                    prefix = "rollout/"
                writer.add_scalar(f"{prefix}{key}", float(value), step)
            except (TypeError, ValueError):
                pass
