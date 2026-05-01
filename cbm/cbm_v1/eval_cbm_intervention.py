#!/usr/bin/env python3
"""CBM Concept Intervention Sweep.

Evaluates the causal effect of each concept by artificially injecting 
MAX (1.0) and MIN (0.0) values across ALL steps of ALL scenarios. 
This is the "counterfactual shock" test.

It measures two things:
1. Action Sensitivity Score (ASS): The mean L2 distance between the natural 
   action and the intervened action.
2. Task Metric Deltas: How forcing the concept to an extreme changes 
   collision rates, route progress, etc.

Performance note:
  Uses jax.lax.scan for the 80-step episode loop so the full episode
  compiles into a single XLA kernel per chunk (not 80 separate dispatches).

Usage:
    python cbm_v1/eval_cbm_intervention.py \\
        --checkpoint cbm_scratch_v2_lambda05/checkpoints/model_final.pkl \\
        --config cbm_v1/config_womd_scratch.yaml \\
        --data data/validation.tfrecord \\
        --num_scenarios 1024 --num_concepts 15 --concept_phases 1 2 3
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
from cbm_v1.config import CBMConfig
import cbm_v1.cbm_sac_factory as cbm_factory


# ── Config loading (reuse from eval_cbm_v2) ──────────────────────────

def load_config_from_pretrained(pretrained_dir: str):
    cfg_path = os.path.join(pretrained_dir, ".hydra", "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"No .hydra/config.yaml found in {pretrained_dir}.")
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


def load_config_from_yaml(yaml_path: str):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    if "network_config" not in cfg:
        raise ValueError(f"No 'network_config' key found in {yaml_path}.")
    obs_cfg_dict = cfg.get("observation_config", {})
    termination_keys = cfg.get("termination_keys", ["offroad", "overlap", "run_red_light"])
    obs_type = {"road": "vec", "lane": "vec"}.get(
        cfg["network_config"].get("_obs_type", "vec"), "vec"
    )
    return cfg["network_config"], obs_cfg_dict, termination_keys, obs_type


# ── Task metric aggregation ──────────────────────────────────────────

FINAL_STEP_METRICS = {"progress_ratio_nuplan", "sdc_progression", "log_divergence"}
EVER_HAPPENED      = {"at_fault_collision", "offroad", "overlap", "run_red_light",
                      "sdc_off_route", "sdc_wrongway", "on_multiple_lanes"}


def aggregate_task_metrics(all_metrics: dict) -> dict:
    """Aggregate raw (T, N) metric arrays into per-scenario scalars → means."""
    rewards = np.array(all_metrics["reward"])
    dones   = np.array(all_metrics["done"])
    early_done = (dones[:-1] > 0.5).any(axis=0)
    ep_returns = rewards.sum(axis=0)
    accuracy   = float((~early_done).mean())

    results = {
        "accuracy":       accuracy,
        "ep_return_mean": float(ep_returns.mean()),
        "ep_return_std":  float(ep_returns.std()),
    }
    for key, arr_jnp in all_metrics.items():
        if key in {"reward", "done", "concepts", "actions"}:
            continue
        arr = np.array(arr_jnp)
        if arr.ndim != 2:
            continue
        if key in FINAL_STEP_METRICS:
            val = arr[-1]
        elif key in EVER_HAPPENED:
            val = (arr > 0.5).any(axis=0).astype(float)
        else:
            val = arr.max(axis=0)
        results[key] = float(val.mean())

    return results


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CBM Concept Intervention")
    parser.add_argument("--checkpoint", required=True, help="Path to CBM checkpoint .pkl")

    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--pretrained_dir", help="V-Max pretrained run dir (frozen/joint models)")
    grp.add_argument("--config",         help="Training YAML path (scratch models)")

    parser.add_argument("--data",          required=True,      help="Path to TFRecord")
    parser.add_argument("--num_scenarios", type=int, default=1024)
    parser.add_argument("--mode",          default="scratch",   choices=["frozen", "joint", "scratch"])
    parser.add_argument("--num_concepts",  type=int, default=15)
    parser.add_argument("--concept_phases", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--output_dir",    default=None)
    parser.add_argument("--chunk_size",    type=int, default=10)
    parser.add_argument(
        "--concepts", nargs="+", type=int, default=None,
        help="Subset of concept indices to intervene on (default: all). "
             "E.g. --concepts 4 9 10"
    )
    args = parser.parse_args()

    concept_phases = tuple(args.concept_phases)

    print()
    print("=" * 65)
    print("CBM CONCEPT INTERVENTION SWEEP")
    print("=" * 65)
    print(f"  Checkpoint     : {args.checkpoint}")
    print(f"  Config src     : {args.pretrained_dir or args.config}")
    print(f"  Data           : {args.data}")
    print(f"  Scenarios      : {args.num_scenarios}")
    print(f"  Mode           : {args.mode}")
    print(f"  Concepts       : {args.num_concepts} (phases {concept_phases})")
    sweep_desc = f"all {args.num_concepts} × 2 = {args.num_concepts * 2} runs" \
                 if args.concepts is None else f"subset {args.concepts} × 2"
    print(f"  Sweep          : {sweep_desc}")
    print()

    # ── Load config ──────────────────────────────────────────────────
    if args.pretrained_dir:
        network_config, obs_cfg_dict, termination_keys, obs_type = \
            load_config_from_pretrained(args.pretrained_dir)
    else:
        network_config, obs_cfg_dict, termination_keys, obs_type = \
            load_config_from_yaml(args.config)

    # ── Build environment ────────────────────────────────────────────
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
    action_size      = env.action_spec().data.shape[0]

    # ── CBM setup ────────────────────────────────────────────────────
    cbm_config = CBMConfig(
        mode=args.mode,
        num_concepts=args.num_concepts,
        concept_phases=concept_phases,
    )
    concept_names = cbm_config.concept_names
    target_indices = args.concepts if args.concepts is not None else list(range(args.num_concepts))

    unflatten_fn = env.get_wrapper_attr("features_extractor").unflatten_features
    cbm_network  = cbm_factory.make_networks(
        observation_size=observation_size,
        action_size=action_size,
        unflatten_fn=unflatten_fn,
        learning_rate=1e-4,
        network_config=network_config,
        cbm_config=cbm_config,
    )
    cbm_params = load_params(args.checkpoint)
    dist        = cbm_network.parametric_action_distribution

    # ── JIT helpers ──────────────────────────────────────────────────
    policy_module = cbm_factory._cbm_policy_module

    @jax.jit
    def get_concepts(obs):
        _, concepts = policy_module.apply(
            cbm_params.policy, obs,
            method=policy_module.encode_and_predict_concepts,
        )
        return concepts

    @jax.jit
    def act_from_concepts(concepts):
        logits = policy_module.apply(
            cbm_params.policy, concepts,
            method=policy_module.act_from_concepts,
        )
        return dist.mode(logits)

    # ── Load scenarios ───────────────────────────────────────────────
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

    # cuSolver warm-up
    _d = jnp.linalg.solve(jnp.eye(4, dtype=jnp.float32), jnp.ones(4, dtype=jnp.float32))
    jax.block_until_ready(_d)

    CHUNK      = args.chunk_size
    num_chunks = math.ceil(args.num_scenarios / CHUNK)
    N          = args.num_scenarios

    # ── JIT-compiled episode runner using lax.scan ────────────────────
    no_override_mask = jnp.zeros(args.num_concepts, dtype=jnp.bool_)
    jit_reset = jax.jit(env.reset)
    jit_step  = jax.jit(env.step)

    @jax.jit
    def run_chunk(chunk_scenarios, reset_keys, override_mask, override_value):
        env_state = jit_reset(chunk_scenarios, reset_keys)

        def step_fn(env_state, _):
            c_natural  = get_concepts(env_state.observation)
            c_act      = jnp.where(override_mask, override_value, c_natural)
            raw_action = act_from_concepts(c_act)
            action = wdatatypes.Action(
                data=raw_action,
                valid=jnp.ones((*raw_action.shape[:-1], 1), dtype=jnp.bool_),
            )
            new_state = jit_step(env_state, action)
            outs = {
                "reward":   new_state.reward,
                "done":     new_state.done,
                "concepts": c_natural,  # (chunk_size, C)
                "actions":  raw_action, # (chunk_size, 2)
                **{k: v for k, v in new_state.metrics.items()},
            }
            return new_state, outs

        _, chunk_metrics = jax.lax.scan(step_fn, env_state, None, length=80)
        return chunk_metrics

    def run_rollout_scan(override_index: int | None, override_value: float | None):
        """Run all scenarios. Returns (metrics, actions, concepts) arrays."""
        override_mask = (
            jnp.arange(args.num_concepts) == override_index
            if override_index is not None
            else no_override_mask
        )
        override_val = jnp.array(float(override_value) if override_value is not None else 0.0)

        chunk_metrics_all = {}
        rng = jax.random.PRNGKey(0)
        for ci in range(num_chunks):
            start = ci * CHUNK
            end   = min((ci + 1) * CHUNK, N)
            chunk      = jax.tree_util.tree_map(lambda x: x[start:end], scenarios)
            rng, rk    = jax.random.split(rng)
            reset_keys = jax.random.split(rk, end - start)
            
            chunk_out = run_chunk(chunk, reset_keys, override_mask, override_val)
            jax.block_until_ready(chunk_out)
            
            for k, v in chunk_out.items():
                chunk_metrics_all.setdefault(k, []).append(np.array(v))

        # Concatenate along scenario axis
        all_metrics = {k: np.concatenate(v, axis=1) for k, v in chunk_metrics_all.items()}
        all_actions = all_metrics.pop("actions")   # (80, N, 2)
        all_concepts = all_metrics.pop("concepts") # (80, N, C)

        return all_metrics, all_actions, all_concepts

    # ── Step 1: Baseline rollout ─────────────────────────────────────
    print("\n-> [1/N] Running BASELINE rollout (no intervention)...")
    print("   (First run compiles XLA kernel — may take a few minutes)")
    t0 = perf_counter()
    baseline_raw, baseline_actions, baseline_concepts = run_rollout_scan(None, None)
    baseline_task = aggregate_task_metrics(baseline_raw)
    print(f"   Done in {perf_counter() - t0:.1f}s")

    print("\n   Baseline results:")
    for k in ["accuracy", "progress_ratio_nuplan", "at_fault_collision",
              "run_red_light", "offroad"]:
        if k in baseline_task:
            print(f"     {k:<32}  {baseline_task[k]:.4f}")

    # Baseline concept population means
    baseline_concept_means = baseline_concepts.mean(axis=(0, 1)).tolist()  # (C,)

    # ── Step 2: Intervention sweep ───────────────────────────────────
    total_runs = len(target_indices) * 2
    run_idx    = 1
    intervention_results = []

    for concept_idx in target_indices:
        concept_name = concept_names[concept_idx]

        for override_val in [0.0, 1.0]:
            run_idx += 1
            label = f"force {concept_name} = {override_val}"
            print(f"\n-> [{run_idx}/{total_runs + 1}]  Intervening: {label}")
            t0 = perf_counter()

            intv_raw, intv_actions, _ = run_rollout_scan(concept_idx, override_val)
            intv_task = aggregate_task_metrics(intv_raw)

            dt = perf_counter() - t0
            print(f"   Done in {dt:.1f}s")

            # ── Action Sensitivity Score (ASS) ────────────────────
            action_delta = intv_actions - baseline_actions   # (80, N, 2)
            ass = float(np.linalg.norm(action_delta, axis=-1).mean())

            mean_accel_change = float(np.abs(action_delta[..., 0]).mean())
            mean_steer_change = float(np.abs(action_delta[..., 1]).mean())

            deltas = {
                k: intv_task.get(k, float("nan")) - baseline_task.get(k, float("nan"))
                for k in baseline_task
            }

            entry = {
                "concept":            concept_name,
                "index":              concept_idx,
                "override_value":     override_val,
                "description": (
                    f"Force {concept_name}={'MAX (1.0)' if override_val == 1.0 else 'MIN (0.0)'}"
                ),
                "action_sensitivity_score": ass,
                "mean_accel_change":  mean_accel_change,
                "mean_steer_change":  mean_steer_change,
                "metrics":            intv_task,
                "delta":              deltas,
            }
            intervention_results.append(entry)

            print(f"     Action Sensitivity Score (ASS) : {ass:.4f}")
            for k in ["progress_ratio_nuplan", "at_fault_collision", "run_red_light"]:
                delta = deltas.get(k, float("nan"))
                flag  = "⚠️ " if abs(delta) > 0.05 else "  "
                print(f"     {flag}{k:<32}  Δ={delta:+.4f}")

    # ── Summary table ────────────────────────────────────────────────
    print()
    print("=" * 90)
    print("INTERVENTION SUMMARY")
    print("=" * 90)
    print(f"  {'Concept':<28}  {'Value':>6}  {'ASS':>8}  {'ΔProgress':>10}  {'ΔCollision':>11}  {'ΔRedLight':>10}")
    print("  " + "-" * 78)
    for r in intervention_results:
        d = r["delta"]
        print(
            f"  {r['concept']:<28}"
            f"  {r['override_value']:>6.1f}"
            f"  {r['action_sensitivity_score']:>8.4f}"
            f"  {d.get('progress_ratio_nuplan', float('nan')):>+10.4f}"
            f"  {d.get('at_fault_collision', float('nan')):>+11.4f}"
            f"  {d.get('run_red_light', float('nan')):>+10.4f}"
        )
    print("=" * 90)

    # Rank concepts by ASS
    print("\n  Top concepts by Action Sensitivity Score:")
    ranked = sorted(intervention_results, key=lambda x: x["action_sensitivity_score"], reverse=True)
    for i, r in enumerate(ranked[:10]):
        print(f"    {i+1:2d}. {r['concept']:<28} @ {r['override_value']:.1f}  ASS={r['action_sensitivity_score']:.4f}")

    # ── Save results ─────────────────────────────────────────────────
    output_dir = args.output_dir or os.path.dirname(args.checkpoint)
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "experiment":           "concept_intervention",
        "checkpoint":           args.checkpoint,
        "config_source":        args.pretrained_dir or args.config,
        "data":                 args.data,
        "num_scenarios":        N,
        "mode":                 args.mode,
        "num_concepts":         args.num_concepts,
        "concept_phases":       list(concept_phases),
        "concept_names":        list(concept_names),
        "baseline_concept_means": baseline_concept_means,
        "baseline":             baseline_task,
        "interventions":        intervention_results,
    }

    json_path = os.path.join(output_dir, "eval_intervention.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n-> Results saved: {json_path}")


if __name__ == "__main__":
    main()
