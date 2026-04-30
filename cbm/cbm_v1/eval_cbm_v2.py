#!/usr/bin/env python3
"""CBM Evaluation Script — V2.

Fixes over eval_cbm.py:
  - Scratch model support: accepts --config (training YAML) instead of
    requiring --pretrained_dir with a .hydra/config.yaml
  - Explicitly reports progress_ratio_nuplan and all key driving metrics
  - Per-scenario breakdown saved to JSON (not just mean/std)
  - Rollout cache (.npz): saves pred_concepts, true_concepts, ego_actions,
    rewards, dones, driving metrics per step — enables curation + visualization
    without re-running inference
  - Correct metric aggregation per metric type

Usage — frozen/joint model (has pretrained_dir):
    python cbm_v1/eval_cbm_v2.py \\
        --checkpoint cbm_model/checkpoints/model_final.pkl \\
        --pretrained_dir runs_rlc/womd_sac_road_perceiver_minimal_42 \\
        --data data/validation.tfrecord \\
        --num_scenarios 200 --mode frozen --num_concepts 11 --concept_phases 1 2

Usage — scratch model (no pretrained_dir):
    python cbm_v1/eval_cbm_v2.py \\
        --checkpoint cbm_scratch_v2_lambda05/checkpoints/model_final.pkl \\
        --config cbm_v1/config_womd_scratch.yaml \\
        --data data/validation.tfrecord \\
        --num_scenarios 200 --mode scratch --num_concepts 15 --concept_phases 1 2 3
"""

from __future__ import annotations

import argparse
import json
import math
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

from waymax import dynamics, datatypes as wdatatypes
from vmax.simulator import make_env_for_evaluation, make_data_generator
from vmax.scripts.evaluate.utils import load_params

from concepts.types import ObservationConfig
from concepts.adapters import observation_to_concept_input
from concepts.registry import extract_all_concepts, CONCEPT_REGISTRY
from concepts.schema import ConceptType

from cbm_v1.config import CBMConfig
import cbm_v1.cbm_sac_factory as cbm_factory


# ── Concept accuracy metrics ──────────────────────────────────────────

def binary_accuracy(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    mask = valid.astype(bool)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean((pred[mask] >= 0.5) == (target[mask] >= 0.5)))


def mae(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    mask = valid.astype(bool)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - target[mask])))


def r2_score(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    mask = valid.astype(bool)
    if mask.sum() < 2:
        return float("nan")
    p, t = pred[mask], target[mask]
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    return float("nan") if ss_tot == 0 else float(1.0 - ss_res / ss_tot)


# ── Config loading ────────────────────────────────────────────────────

def load_config_from_pretrained(pretrained_dir: str) -> tuple[dict, dict, list, str]:
    """Load network_config, obs_cfg_dict, termination_keys, obs_type from hydra."""
    cfg_path = os.path.join(pretrained_dir, ".hydra", "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"No .hydra/config.yaml found in {pretrained_dir}.\n"
            f"If this is a scratch model, use --config instead of --pretrained_dir."
        )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    obs_cfg_dict = cfg.get("observation_config", {})
    termination_keys = cfg.get("termination_keys", ["offroad", "overlap", "run_red_light"])

    enc_cfg = dict(cfg["network"]["encoder"])
    enc_cfg["type"] = {"perceiver": "lq"}.get(enc_cfg.get("type", "none"), enc_cfg.get("type", "none"))
    obs_type = {"road": "vec", "lane": "vec"}.get(cfg.get("observation_type", "vec"), "vec")

    network_config = {
        "encoder": enc_cfg,
        "policy": cfg["algorithm"]["network"]["policy"],
        "value": cfg["algorithm"]["network"]["value"],
        "action_distribution": cfg["algorithm"]["network"].get("action_distribution", "gaussian"),
        "_obs_type": obs_type,
    }
    return network_config, obs_cfg_dict, termination_keys, obs_type


def load_config_from_yaml(yaml_path: str) -> tuple[dict, dict, list, str]:
    """Load network_config, obs_cfg_dict, termination_keys, obs_type from training YAML."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    if "network_config" not in cfg:
        raise ValueError(
            f"No 'network_config' key found in {yaml_path}.\n"
            f"This YAML must be a scratch training config with inline network_config."
        )

    obs_cfg_dict = cfg.get("observation_config", {})
    termination_keys = cfg.get("termination_keys", ["offroad", "overlap", "run_red_light"])
    obs_type = {"road": "vec", "lane": "vec"}.get(
        cfg["network_config"].get("_obs_type", "vec"), "vec"
    )
    network_config = cfg["network_config"]
    return network_config, obs_cfg_dict, termination_keys, obs_type


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CBM Evaluation V2")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to CBM checkpoint .pkl")
    # Config source — one of the two must be provided
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--pretrained_dir",
                     help="V-Max pretrained run dir (frozen/joint models)")
    grp.add_argument("--config",
                     help="Training YAML path (scratch models without pretrained_dir)")
    # Data
    parser.add_argument("--data", required=True,
                        help="Path to WOMD TFRecord (use validation split for eval)")
    parser.add_argument("--num_scenarios", type=int, default=200,
                        help="Number of scenarios to evaluate (default: 200)")
    # Model spec
    parser.add_argument("--mode", default="frozen",
                        choices=["frozen", "joint", "scratch"])
    parser.add_argument("--num_concepts", type=int, default=11)
    parser.add_argument("--concept_phases", nargs="+", type=int, default=[1, 2])
    # Output
    parser.add_argument("--output_dir", default=None,
                        help="Where to write results (default: same dir as checkpoint)")
    parser.add_argument("--no_cache", action="store_true",
                        help="Skip saving the rollout cache .npz (saves disk space)")
    parser.add_argument("--chunk_size", type=int, default=10,
                        help="Scenarios per rollout chunk (reduce if OOM, default: 10)")
    args = parser.parse_args()

    concept_phases = tuple(args.concept_phases)

    print()
    print("=" * 65)
    print("CBM EVALUATION V2")
    print("=" * 65)
    print(f"  Checkpoint   : {args.checkpoint}")
    print(f"  Config src   : {args.pretrained_dir or args.config}")
    print(f"  Data         : {args.data}")
    print(f"  Scenarios    : {args.num_scenarios}")
    print(f"  Mode         : {args.mode}")
    print(f"  Concepts     : {args.num_concepts} (phases {concept_phases})")
    print(f"  Save cache   : {not args.no_cache}")
    print()

    # ── Load network + observation config ────────────────────────────
    print("-> Loading config...")
    if args.pretrained_dir:
        network_config, obs_cfg_dict, termination_keys, obs_type = \
            load_config_from_pretrained(args.pretrained_dir)
    else:
        network_config, obs_cfg_dict, termination_keys, obs_type = \
            load_config_from_yaml(args.config)
    print("   Done.")

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

    # ── Load checkpoint ──────────────────────────────────────────────
    print("-> Loading checkpoint...")
    cbm_params = load_params(args.checkpoint)
    print("   Done.")

    # ── Build CBM networks ───────────────────────────────────────────
    print("-> Building CBM networks...")
    cbm_config = CBMConfig(
        mode=args.mode,
        num_concepts=args.num_concepts,
        concept_phases=concept_phases,
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
        learning_rate=1e-4,
        network_config=network_config,
        cbm_config=cbm_config,
    )
    policy_fn = cbm_factory.make_inference_fn(cbm_network)
    policy = policy_fn(cbm_params.policy, deterministic=True)
    print("   Done.")

    # ── Data generator ───────────────────────────────────────────────
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

    # ── JIT helpers ──────────────────────────────────────────────────

    @jax.jit
    def get_concept_predictions(obs):
        _, concepts = cbm_factory._cbm_policy_module.apply(
            cbm_params.policy, obs,
            method=cbm_factory._cbm_policy_module.encode_and_predict_concepts,
        )
        return concepts

    @jax.jit
    def get_concept_targets(obs):
        inp = observation_to_concept_input(obs, unflatten_fn, concept_config)
        out = extract_all_concepts(inp, phases=concept_phases)
        return out.normalized, out.valid

    @jax.jit
    def eval_step(env_transition, _):
        obs = env_transition.observation
        raw_action, _ = policy(obs, None)
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
        return next_transition, (obs, raw_action, metrics)

    # ── cuSolver warm-up ─────────────────────────────────────────────
    print("-> Warming up cuSolver...")
    _d = jnp.linalg.solve(jnp.eye(4, dtype=jnp.float32), jnp.ones(4, dtype=jnp.float32))
    jax.block_until_ready(_d)
    print("   Done.")

    # ── Rollout ──────────────────────────────────────────────────────
    CHUNK = args.chunk_size
    num_chunks = math.ceil(args.num_scenarios / CHUNK)
    print(f"\n-> Running rollouts ({args.num_scenarios} scenarios, {num_chunks} chunks of {CHUNK})...")

    all_obs_list, all_actions_list, all_metrics_lists = [], [], {}
    rng = jax.random.PRNGKey(0)

    t0 = perf_counter()
    for i in range(num_chunks):
        start = i * CHUNK
        end = min((i + 1) * CHUNK, args.num_scenarios)
        size = end - start
        print(f"   chunk {i+1}/{num_chunks} (scenarios {start}–{end-1})...")

        chunk = jax.tree_util.tree_map(lambda x: x[start:end], scenarios)
        rng, rk = jax.random.split(rng)
        reset_keys = jax.random.split(rk, size)

        env_state = jax.jit(env.reset)(chunk, reset_keys)
        _, (chunk_obs, chunk_actions, chunk_metrics) = jax.lax.scan(
            eval_step, env_state, None, length=80
        )
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), chunk_obs)

        all_obs_list.append(chunk_obs)
        all_actions_list.append(chunk_actions)
        for k, v in chunk_metrics.items():
            all_metrics_lists.setdefault(k, []).append(v)

    dt = perf_counter() - t0
    print(f"   Rollout done in {dt:.1f}s ({args.num_scenarios / dt:.1f} scenarios/s)")

    # Concatenate across chunks: (T=80, N, ...)
    all_obs     = jnp.concatenate(all_obs_list, axis=1)        # (80, N, D)
    all_actions = jnp.concatenate(all_actions_list, axis=1)    # (80, N, 2)
    all_metrics = {k: jnp.concatenate(v, axis=1)
                   for k, v in all_metrics_lists.items()}       # (80, N) each

    T, N, D = all_obs.shape

    # ── Concept accuracy ─────────────────────────────────────────────
    print("\n-> Computing concept accuracy...")
    obs_flat = all_obs.reshape(T * N, D)

    CBATCH = 8192
    pred_chunks, tgt_chunks, valid_chunks = [], [], []
    for s in range(0, T * N, CBATCH):
        obs_b = obs_flat[s:s + CBATCH]
        pred_chunks.append(np.array(get_concept_predictions(obs_b)))
        tgt_b, val_b = get_concept_targets(obs_b)
        tgt_chunks.append(np.array(tgt_b))
        valid_chunks.append(np.array(val_b))

    pred_all  = np.concatenate(pred_chunks,  axis=0)   # (T*N, C)
    tgt_all   = np.concatenate(tgt_chunks,   axis=0)
    valid_all = np.concatenate(valid_chunks, axis=0)

    # Reshape back for cache: (T, N, C)
    pred_TNC  = pred_all.reshape(T, N, -1)
    tgt_TNC   = tgt_all.reshape(T, N, -1)
    valid_TNC = valid_all.reshape(T, N, -1)

    concept_metrics = {}
    print()
    print(f"  {'Concept':<28}  {'Type':>10}  {'Valid%':>7}  {'Metric':>10}  {'Value':>8}")
    print("  " + "-" * 70)

    for i, name in enumerate(concept_names):
        p, t, v = pred_all[:, i], tgt_all[:, i], valid_all[:, i]
        valid_pct = 100.0 * v.mean()
        ctype = concept_types[name]
        phase = "(P3)" if name in [
            "path_curvature_max","path_net_heading_change",
            "path_straightness","heading_to_path_end"] else "    "

        if ctype == ConceptType.BINARY:
            acc = binary_accuracy(p, t, v)
            concept_metrics[name] = {"type": "binary", "accuracy": acc, "valid_pct": valid_pct}
            print(f"  {phase} {name:<24}  {'binary':>10}  {valid_pct:>6.1f}%  {'accuracy':>10}  {acc:>7.4f}")
        else:
            m, r2 = mae(p, t, v), r2_score(p, t, v)
            concept_metrics[name] = {"type": "continuous", "mae": m, "r2": r2, "valid_pct": valid_pct}
            print(f"  {phase} {name:<24}  {'continuous':>10}  {valid_pct:>6.1f}%  {'MAE':>10}  {m:>7.4f}  R²={r2:.3f}")

    # ── Task metrics ─────────────────────────────────────────────────
    print("\n-> Computing task metrics...")

    rewards = np.array(all_metrics["reward"])    # (80, N)
    dones   = np.array(all_metrics["done"])      # (80, N)

    # Episode completion: no early termination before final step
    early_done   = (dones[:-1] > 0.5).any(axis=0)     # (N,)
    ep_returns   = rewards.sum(axis=0)                  # (N,)
    accuracy     = float((~early_done).mean())

    task_results = {
        "accuracy":           accuracy,
        "ep_return_mean":     float(ep_returns.mean()),
        "ep_return_std":      float(ep_returns.std()),
        "n_scenarios":        N,
    }

    # Per-driving-metric — smart aggregation per metric type
    FINAL_STEP_METRICS = {"progress_ratio_nuplan", "sdc_progression", "log_divergence"}
    EVER_HAPPENED      = {"at_fault_collision", "offroad", "overlap", "run_red_light",
                          "sdc_off_route", "sdc_wrongway", "on_multiple_lanes"}

    skip = {"reward", "done"}
    per_scenario = {"early_done": early_done.tolist(), "ep_return": ep_returns.tolist()}

    for key, arr_jnp in all_metrics.items():
        if key in skip:
            continue
        arr = np.array(arr_jnp)   # (80, N)
        if arr.ndim != 2:
            continue
        if key in FINAL_STEP_METRICS:
            # Value at final step
            val_per_scenario = arr[-1]
        elif key in EVER_HAPPENED:
            # Did it happen at all during episode
            val_per_scenario = (arr > 0.5).any(axis=0).astype(float)
        else:
            # Default: max over time
            val_per_scenario = arr.max(axis=0)

        task_results[key] = float(val_per_scenario.mean())
        per_scenario[key] = val_per_scenario.tolist()

    # Print key metrics explicitly
    KEY_METRICS = [
        "accuracy", "ep_return_mean", "progress_ratio_nuplan",
        "at_fault_collision", "offroad", "run_red_light",
        "sdc_off_route", "vmax_score", "log_divergence",
    ]
    print()
    print(f"  {'Metric':<32}  {'Value':>10}")
    print("  " + "-" * 46)
    for key in KEY_METRICS:
        if key in task_results:
            print(f"  {key:<32}  {task_results[key]:>10.4f}")
    print()
    print("  Other metrics:")
    for key in sorted(task_results):
        if key in KEY_METRICS or key in {"ep_return_std", "n_scenarios"}:
            continue
        print(f"  {key:<32}  {task_results[key]:>10.4f}")

    # ── Summary ──────────────────────────────────────────────────────
    binary_accs = [v["accuracy"] for v in concept_metrics.values()
                   if v["type"] == "binary" and not np.isnan(v["accuracy"])]
    cont_maes   = [v["mae"]      for v in concept_metrics.values()
                   if v["type"] == "continuous" and not np.isnan(v["mae"])]
    cont_r2s    = [v["r2"]       for v in concept_metrics.values()
                   if v["type"] == "continuous" and not np.isnan(v["r2"])]

    print()
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    if binary_accs:
        print(f"  Binary concept mean accuracy  : {np.mean(binary_accs):.4f}")
    if cont_maes:
        print(f"  Continuous concept mean MAE   : {np.mean(cont_maes):.4f}")
        print(f"  Continuous concept mean R²    : {np.mean(cont_r2s):.4f}")
    prog = task_results.get("progress_ratio_nuplan", float("nan"))
    coll = task_results.get("at_fault_collision",    float("nan"))
    print(f"  Route progress (nuplan)       : {prog:.4f}")
    print(f"  At-fault collision rate       : {coll:.4f}")
    print(f"  Episode completion (no crash) : {accuracy:.4f}")
    print("=" * 65)

    # ── Save JSON results ────────────────────────────────────────────
    output_dir = args.output_dir or os.path.dirname(args.checkpoint)
    os.makedirs(output_dir, exist_ok=True)
    ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]

    results = {
        "checkpoint":     args.checkpoint,
        "config_source":  args.pretrained_dir or args.config,
        "data":           args.data,
        "num_scenarios":  N,
        "mode":           args.mode,
        "num_concepts":   args.num_concepts,
        "concept_phases": list(concept_phases),
        "concept_metrics": concept_metrics,
        "task_metrics":    task_results,
        "per_scenario":    per_scenario,
    }
    json_path = os.path.join(output_dir, f"eval_{ckpt_stem}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n-> Results saved: {json_path}")

    # ── Save rollout cache ───────────────────────────────────────────
    if not args.no_cache:
        cache_path = os.path.join(output_dir, f"eval_{ckpt_stem}_cache.npz")
        print(f"-> Saving rollout cache: {cache_path}")

        # Build driving metrics array (T, N, M)
        drv_keys = [k for k in all_metrics if k not in {"reward", "done"}]
        drv_arr = np.stack([np.array(all_metrics[k]) for k in drv_keys], axis=-1)  # (T,N,M)

        np.savez_compressed(
            cache_path,
            # Concepts
            pred_concepts  = pred_TNC.astype(np.float32),    # (T, N, C)
            true_concepts  = tgt_TNC.astype(np.float32),
            valid_mask     = valid_TNC.astype(bool),
            concept_names  = np.array(concept_names),
            # Actions & rewards
            ego_actions    = np.array(all_actions).astype(np.float32),  # (T, N, 2)
            rewards        = rewards.astype(np.float32),                 # (T, N)
            dones          = dones.astype(np.float32),
            # Driving metrics
            driving_metrics     = drv_arr.astype(np.float32),
            driving_metric_keys = np.array(drv_keys),
            # Scenario index
            scenario_indices    = np.arange(N),
        )
        size_mb = os.path.getsize(cache_path) / 1e6
        print(f"   Cache saved ({size_mb:.1f} MB)")
        print()
        print("   Cache fields:")
        print("     pred_concepts  (T, N, C) — model concept predictions per step")
        print("     true_concepts  (T, N, C) — ground-truth concepts per step")
        print("     valid_mask     (T, N, C) — per-concept validity mask")
        print("     ego_actions    (T, N, 2) — [acceleration, steering] per step")
        print("     rewards        (T, N)    — reward per step")
        print("     dones          (T, N)    — done flag per step")
        print("     driving_metrics(T, N, M) — all Waymax driving metrics per step")


if __name__ == "__main__":
    main()
