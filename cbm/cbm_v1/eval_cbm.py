#!/usr/bin/env python3
"""CBM-V1 Phase 1 Evaluation Script.

Reports two things:
  1. Per-concept accuracy  — how well the concept head predicts each concept
  2. Task performance      — driving metrics vs the pretrained SAC baseline

Usage:
    /home/med1e/anaconda3/envs/vmax/bin/python cbm_v1/eval_cbm.py \
        --checkpoint runs_cbm/cbm_v1_frozen_womd_42/checkpoints/model_final.pkl \
        --pretrained_dir runs_rlc/womd_sac_road_perceiver_minimal_42 \
        --data data/training.tfrecord \
        --num_scenarios 64

Output:
    Console table + runs_cbm/<run>/eval_<step>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "V-Max"))

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from functools import partial

from waymax import dynamics, datatypes as wdatatypes
from vmax.simulator import make_env_for_evaluation, make_data_generator
from vmax.agents.pipeline import inference
from vmax.scripts.evaluate.utils import load_params

from concepts.types import ObservationConfig
from concepts.adapters import observation_to_concept_input
from concepts.registry import extract_all_concepts, CONCEPT_REGISTRY
from concepts.schema import ConceptType

from cbm_v1.config import CBMConfig
import cbm_v1.cbm_sac_factory as cbm_factory


# ── Concept accuracy metrics ──────────────────────────────────────────

def binary_accuracy(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    """Fraction of correctly classified binary concepts where valid=True."""
    mask = valid.astype(bool)
    if mask.sum() == 0:
        return float("nan")
    pred_binary = (pred[mask] >= 0.5).astype(float)
    tgt_binary = (target[mask] >= 0.5).astype(float)
    return float(np.mean(pred_binary == tgt_binary))


def mae(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    """Mean absolute error on continuous concepts where valid=True."""
    mask = valid.astype(bool)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - target[mask])))


def r2_score(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    """R² (coefficient of determination) for continuous concepts."""
    mask = valid.astype(bool)
    if mask.sum() < 2:
        return float("nan")
    p, t = pred[mask], target[mask]
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CBM-V1 Phase 1 Evaluation")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to CBM checkpoint .pkl file")
    parser.add_argument("--pretrained_dir", required=True,
                        help="Pretrained V-Max run directory (for config + encoder remap)")
    parser.add_argument("--data", required=True,
                        help="Path to WOMD TFRecord file")
    parser.add_argument("--num_scenarios", type=int, default=64,
                        help="Number of scenarios to evaluate (default: 64)")
    parser.add_argument("--mode", default="frozen", choices=["frozen", "joint", "scratch"],
                        help="CBM mode the checkpoint was trained with")
    parser.add_argument("--num_concepts", type=int, default=11,
                        help="Number of concepts (default: 11 for V1, 15 for V2)")
    parser.add_argument("--concept_phases", nargs="+", type=int, default=[1, 2],
                        help="Concept phases to extract (default: 1 2)")
    parser.add_argument("--output_dir", default=None,
                        help="Directory to write eval_results.json (default: checkpoint dir)")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("CBM-V1  PHASE 1 EVALUATION")
    print("=" * 60)
    print(f"  Checkpoint     : {args.checkpoint}")
    print(f"  Pretrained dir : {args.pretrained_dir}")
    print(f"  Data           : {args.data}")
    print(f"  Scenarios      : {args.num_scenarios}")
    print(f"  Mode           : {args.mode}")
    print()

    # ── Load pretrained run config ───────────────────────────────────
    hydra_cfg_path = os.path.join(args.pretrained_dir, ".hydra", "config.yaml")
    with open(hydra_cfg_path) as f:
        pretrained_cfg = yaml.safe_load(f)

    obs_cfg_dict = pretrained_cfg.get("observation_config", {})
    termination_keys = pretrained_cfg.get("termination_keys", ["offroad", "overlap"])

    enc_cfg = dict(pretrained_cfg["network"]["encoder"])
    enc_type = enc_cfg.get("type", "none")
    enc_cfg["type"] = {"perceiver": "lq"}.get(enc_type, enc_type)
    obs_type = {"road": "vec", "lane": "vec"}.get(
        pretrained_cfg.get("observation_type", "vec"), "vec")

    network_config = {
        "encoder": enc_cfg,
        "policy": pretrained_cfg["algorithm"]["network"]["policy"],
        "value": pretrained_cfg["algorithm"]["network"]["value"],
        "action_distribution": pretrained_cfg["algorithm"]["network"].get(
            "action_distribution", "gaussian"),
        "_obs_type": obs_type,
    }

    # ── Build environment ────────────────────────────────────────────
    print("-> Building evaluation environment...")
    env = make_env_for_evaluation(
        max_num_objects=64,
        dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
        sdc_paths_from_data=True,
        observation_type=obs_type,
        observation_config=obs_cfg_dict,
        termination_keys=termination_keys,
        noisy_init=False,
    )
    observation_size = env.observation_spec()
    action_size = env.action_spec().data.shape[0]
    print(f"   obs_size={observation_size}, action_size={action_size}")

    # ── Concept config ───────────────────────────────────────────────
    concept_config = ObservationConfig(
        obs_past_num_steps=obs_cfg_dict.get("obs_past_num_steps", 5),
        num_closest_objects=obs_cfg_dict.get("objects", {}).get("num_closest_objects", 8),
        roadgraph_top_k=obs_cfg_dict.get("roadgraphs", {}).get("roadgraph_top_k", 200),
        num_closest_traffic_lights=obs_cfg_dict.get("traffic_lights", {}).get(
            "num_closest_traffic_lights", 5),
        num_target_path_points=obs_cfg_dict.get("path_target", {}).get("num_points", 10),
        max_meters=obs_cfg_dict.get("roadgraphs", {}).get("max_meters", 70),
    )
    unflatten_fn = env.get_wrapper_attr("features_extractor").unflatten_features

    # ── Load CBM checkpoint ──────────────────────────────────────────
    print("-> Loading CBM checkpoint...")
    cbm_params = load_params(args.checkpoint)
    print("   Done.")

    # ── Build CBM networks ───────────────────────────────────────────
    print("-> Building CBM networks...")
    cbm_config = CBMConfig(
        mode=args.mode,
        num_concepts=args.num_concepts,
        concept_phases=tuple(args.concept_phases)
    )
    concept_names = cbm_config.concept_names
    concept_types = {
        name: schema.concept_type
        for name, (schema, _) in CONCEPT_REGISTRY.items()
        if name in concept_names
    }

    cbm_network = cbm_factory.make_networks(
        observation_size=observation_size,
        action_size=action_size,
        unflatten_fn=unflatten_fn,
        learning_rate=1e-4,  # not used during eval
        network_config=network_config,
        cbm_config=cbm_config,
    )
    print("   Done.")

    # ── Data generator ───────────────────────────────────────────────
    # Eval uses make_env_for_evaluation (VmapWrapper only, no AutoResetWrapper)
    # so batch_dims = (num_scenarios,) — no pmap, no scan wrapper
    print(f"-> Loading {args.num_scenarios} scenarios...")
    data_gen = make_data_generator(
        path=args.data,
        max_num_objects=64,
        include_sdc_paths=True,
        batch_dims=(args.num_scenarios,),
        seed=0,
        repeat=True,
    )
    scenarios = next(data_gen)
    print("   Done.")

    # ── Deterministic policy ─────────────────────────────────────────
    policy_fn = cbm_factory.make_inference_fn(cbm_network)
    policy = policy_fn(cbm_params.policy, deterministic=True)

    # ── JIT-compiled eval helpers ────────────────────────────────────

    @jax.jit
    def get_concept_predictions(obs):
        """Run encoder + concept head on a batch of observations."""
        z, concepts = cbm_factory._cbm_policy_module.apply(
            cbm_params.policy, obs,
            method=cbm_factory._cbm_policy_module.encode_and_predict_concepts,
        )
        return concepts

    @jax.jit
    def get_concept_targets(obs):
        """Extract ground-truth concept values from observation."""
        inp = observation_to_concept_input(obs, unflatten_fn, concept_config)
        out = extract_all_concepts(inp, phases=cbm_config.concept_phases)
        return out.normalized, out.valid

    @jax.jit
    def eval_step(env_transition, _):
        """Single environment step with the CBM policy."""
        obs = env_transition.observation
        raw_action, _ = policy(obs, None)
        # Waymax's PlanningAgentEnvironment expects a datatypes.Action pytree
        # (with .data and .valid), not a raw JAX array.  VmapWrapper will vmap
        # over the leading batch dimension of each leaf.
        # action_spec says valid has shape (1,) per env, not ()
        action = wdatatypes.Action(
            data=raw_action,
            valid=jnp.ones((*raw_action.shape[:-1], 1), dtype=jnp.bool_),
        )
        next_transition = env.step(env_transition, action)
        metrics = {
            "reward": next_transition.reward,
            "done": next_transition.done,
            **{k: v for k, v in next_transition.metrics.items()},
        }
        return next_transition, (obs, metrics)

    # ── CUDA / cuSolver warm-up ──────────────────────────────────────
    # InvertibleBicycleModel calls a JAX linear solver internally during
    # env.reset. If the cuSolver handle hasn't been initialised yet, JAX
    # throws "gpusolverDnCreate failed: cuSolver internal error".
    # Running a trivial linalg op before the first env.reset forces the
    # handle to be created in a controlled context.
    print("-> Warming up cuSolver handle...")
    _dummy = jnp.linalg.solve(jnp.eye(4, dtype=jnp.float32),
                               jnp.ones(4, dtype=jnp.float32))
    jax.block_until_ready(_dummy)
    print("   Done.")

    # ── Run rollouts ─────────────────────────────────────────────────
    print(f"\n-> Running rollouts ({args.num_scenarios} scenarios × 80 steps)...")
    rng = jax.random.PRNGKey(0)
    rng, rk = jax.random.split(rng)
    
    t0 = perf_counter()
    
    # Batch rollouts to prevent OOM
    CHUNK = 10
    all_obs_list = []
    
    # metrics dict of lists
    all_metrics_lists = {}
    
    import math
    num_chunks = math.ceil(args.num_scenarios / CHUNK)
    
    for i in range(num_chunks):
        start = i * CHUNK
        end = min((i + 1) * CHUNK, args.num_scenarios)
        size = end - start
        
        print(f"   Running chunk {i+1}/{num_chunks} ({size} scenarios)...")
        # slice scenario
        scenario_chunk = jax.tree_util.tree_map(lambda x: x[start:end], scenarios)
        rk, chunk_rk = jax.random.split(rk)
        reset_keys_chunk = jax.random.split(chunk_rk, size)
        
        env_state_chunk = jax.jit(env.reset)(scenario_chunk, reset_keys_chunk)
        
        # Collect observations and metrics over full episode
        final_state, (chunk_obs, chunk_metrics) = jax.lax.scan(
            eval_step, env_state_chunk, None, length=80
        )
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), chunk_obs)
        all_obs_list.append(chunk_obs)
        
        for k, v in chunk_metrics.items():
            if k not in all_metrics_lists:
                all_metrics_lists[k] = []
            all_metrics_lists[k].append(v)

    dt = perf_counter() - t0
    print(f"   Rollout done in {dt:.1f}s")

    # Combine observations: chunk_obs is (T, size, D)
    all_obs = jnp.concatenate(all_obs_list, axis=1)
    
    # Combine metrics: each is list of (T, size) arrays
    all_metrics = {k: jnp.concatenate(v_list, axis=1) for k, v_list in all_metrics_lists.items()}

    # Flatten time × scenario for concept evaluation
    T, N, D = all_obs.shape
    obs_flat = all_obs.reshape(T * N, D)

    # ── Concept accuracy ─────────────────────────────────────────────
    print("\n-> Computing concept accuracy...")

    # Batch the concept computation to avoid OOM
    CHUNK = 256
    pred_chunks, tgt_chunks, valid_chunks = [], [], []
    for start in range(0, T * N, CHUNK):
        obs_chunk = obs_flat[start:start + CHUNK]
        pred_chunks.append(np.array(get_concept_predictions(obs_chunk)))
        tgt, val = get_concept_targets(obs_chunk)
        tgt_chunks.append(np.array(tgt))
        valid_chunks.append(np.array(val))

    pred_all = np.concatenate(pred_chunks, axis=0)   # (T*N, 11)
    tgt_all = np.concatenate(tgt_chunks, axis=0)     # (T*N, 11)
    valid_all = np.concatenate(valid_chunks, axis=0)  # (T*N, 11)

    concept_metrics = {}
    print()
    print(f"  {'Concept':<28}  {'Type':>10}  {'Valid%':>7}  {'Metric':>8}  {'Value':>8}")
    print("  " + "-" * 68)

    for i, name in enumerate(concept_names):
        p = pred_all[:, i]
        t = tgt_all[:, i]
        v = valid_all[:, i]
        valid_pct = 100.0 * v.mean()
        ctype = concept_types[name]

        if ctype == ConceptType.BINARY:
            acc = binary_accuracy(p, t, v)
            concept_metrics[name] = {"type": "binary", "accuracy": acc,
                                     "valid_pct": valid_pct}
            metric_name, metric_val = "accuracy", acc
            print(f"  {name:<28}  {'binary':>10}  {valid_pct:>6.1f}%  {metric_name:>8}  {metric_val:>7.3f}")
        else:
            m = mae(p, t, v)
            r2 = r2_score(p, t, v)
            concept_metrics[name] = {"type": "continuous", "mae": m, "r2": r2,
                                     "valid_pct": valid_pct}
            print(f"  {name:<28}  {'continuous':>10}  {valid_pct:>6.1f}%  {'MAE':>8}  {m:>7.4f}  R²={r2:.3f}")

    # ── Task metrics ─────────────────────────────────────────────────
    print("\n-> Computing task metrics...")

    # all_metrics values: (80, num_scenarios) each
    task_results = {}

    # Episode return
    rewards = np.array(all_metrics["reward"])          # (80, N)
    dones   = np.array(all_metrics["done"])            # (80, N)
    ep_returns = rewards.sum(axis=0)                   # (N,)
    task_results["ep_return_mean"] = float(ep_returns.mean())
    task_results["ep_return_std"]  = float(ep_returns.std())

    # Accuracy: episode completed without any termination key triggering
    # done=1 means the scenario ended (either by termination or time limit)
    # We consider a scenario "successful" if it reached the time limit (step 80)
    # without early termination — i.e., done was never True before the last step
    early_done = (dones[:-1] > 0.5).any(axis=0)       # (N,) — done before last step
    accuracy = float((~early_done).mean())
    task_results["accuracy"] = accuracy

    # Per driving metric (offroad, overlap, etc.)
    skip_keys = {"reward", "done"}
    for key, vals in all_metrics.items():
        if key in skip_keys:
            continue
        arr = np.array(vals)   # (80, N)
        if arr.ndim == 2:
            # Mean over episodes; max per scenario then mean across scenarios
            task_results[key] = float(arr.max(axis=0).mean())

    print()
    print(f"  {'Metric':<30}  {'Value':>10}")
    print("  " + "-" * 44)
    print(f"  {'accuracy':<30}  {task_results['accuracy']:>10.4f}")
    print(f"  {'ep_return_mean':<30}  {task_results['ep_return_mean']:>10.4f}")
    print(f"  {'ep_return_std':<30}  {task_results['ep_return_std']:>10.4f}")
    for key in sorted(task_results):
        if key in {"accuracy", "ep_return_mean", "ep_return_std"}:
            continue
        print(f"  {key:<30}  {task_results[key]:>10.4f}")

    # ── Baseline comparison ──────────────────────────────────────────
    BASELINE_ACCURACY = {
        "womd_sac_road_perceiver_minimal_42": 0.97467,
        "womd_sac_road_perceiver_minimal_69": 0.97435,
        "womd_sac_road_perceiver_minimal_99": 0.96866,
    }
    baseline_name = os.path.basename(args.pretrained_dir)
    baseline_acc  = BASELINE_ACCURACY.get(baseline_name, None)

    print()
    print("  Baseline comparison:")
    print(f"  {'CBM-V1 accuracy':<30}  {accuracy:>10.4f}")
    if baseline_acc is not None:
        delta = accuracy - baseline_acc
        sign = "+" if delta >= 0 else ""
        print(f"  {'Baseline accuracy':<30}  {baseline_acc:>10.4f}  ({baseline_name})")
        print(f"  {'Delta':<30}  {sign}{delta:>9.4f}")
    else:
        print(f"  (no baseline entry for '{baseline_name}')")

    # ── Summary print ────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("CONCEPT ACCURACY SUMMARY")
    print("=" * 60)
    binary_accs = [v["accuracy"] for v in concept_metrics.values()
                   if v["type"] == "binary" and not np.isnan(v["accuracy"])]
    cont_maes   = [v["mae"] for v in concept_metrics.values()
                   if v["type"] == "continuous" and not np.isnan(v["mae"])]
    cont_r2s    = [v["r2"] for v in concept_metrics.values()
                   if v["type"] == "continuous" and not np.isnan(v["r2"])]

    if binary_accs:
        print(f"  Binary concepts  — mean accuracy : {np.mean(binary_accs):.3f}")
    if cont_maes:
        print(f"  Continuous concepts — mean MAE   : {np.mean(cont_maes):.4f}")
        print(f"  Continuous concepts — mean R²    : {np.mean(cont_r2s):.3f}")
    print(f"  Task accuracy (no early term.)   : {accuracy:.4f}")
    if baseline_acc is not None:
        print(f"  vs. baseline                     : {sign}{delta:.4f}")
    print("=" * 60)

    # ── Save results ─────────────────────────────────────────────────
    output_dir = args.output_dir or os.path.dirname(args.checkpoint)
    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out_path = os.path.join(output_dir, f"eval_{ckpt_name}.json")

    results = {
        "checkpoint": args.checkpoint,
        "pretrained_dir": args.pretrained_dir,
        "num_scenarios": args.num_scenarios,
        "mode": args.mode,
        "concept_metrics": concept_metrics,
        "task_metrics": task_results,
        "baseline_accuracy": baseline_acc,
        "baseline_name": baseline_name,
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n-> Results saved: {out_path}")


if __name__ == "__main__":
    main()
