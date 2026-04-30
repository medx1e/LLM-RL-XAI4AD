#!/usr/bin/env python3
"""Evaluate a V-Max LQ/Perceiver model with all compatibility fixes applied.

Usage:
    PYTHONPATH=V-Max python eval_lq_baseline.py \
        --model_dir runs_rlc/womd_sac_road_perceiver_minimal_42 \
        --data /path/to/validation.tfrecord \
        --num_scenarios 1024 \
        --output eval_results/lq_womd_150gb.csv
"""
import argparse, math, os, sys, csv
from pathlib import Path
from time import perf_counter

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
sys.path.insert(0, str(Path(__file__).parent / "V-Max"))

import jax, jax.numpy as jnp, numpy as np, yaml
from functools import partial
from waymax import dynamics, datatypes as wdatatypes
from vmax.simulator import make_env_for_evaluation, make_data_generator
from vmax.scripts.evaluate.utils import load_params
from vmax.agents.learning.reinforcement.sac import sac_factory
from vmax.agents.networks import network_factory


def remap_keys(p, old, new):
    if isinstance(p, dict):
        return {(new if k == old else k): remap_keys(v, old, new) for k, v in p.items()}
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--num_scenarios", type=int, default=1024)
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--output", default="eval_results/lq_baseline.csv")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────
    with open(f"{args.model_dir}/.hydra/config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Fix obs type
    obs_type = {"road": "vec", "lane": "vec"}.get(cfg["observation_type"], cfg["observation_type"])

    # Fix encoder type
    enc_cfg = dict(cfg["network"]["encoder"])
    enc_cfg["type"] = {"perceiver": "lq", "mgail": "lqh"}.get(enc_cfg["type"], enc_cfg["type"])

    network_config = {
        "encoder": enc_cfg,
        "policy": cfg["algorithm"]["network"]["policy"],
        "value": cfg["algorithm"]["network"]["value"],
        "action_distribution": cfg["algorithm"]["network"].get("action_distribution", "gaussian"),
        "_obs_type": obs_type,
    }
    obs_cfg = cfg.get("observation_config", {})
    termination_keys = cfg.get("termination_keys", ["offroad", "overlap", "run_red_light"])

    # ── Build env ─────────────────────────────────────────────────────
    env = make_env_for_evaluation(
        max_num_objects=64,
        dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
        sdc_paths_from_data=True,
        observation_type=obs_type,
        observation_config=obs_cfg,
        termination_keys=termination_keys,
        noisy_init=False,
    )
    obs_size = env.observation_spec()
    action_size = env.action_spec().data.shape[0]
    unflatten_fn = env.get_wrapper_attr("features_extractor").unflatten_features

    # ── Build network ─────────────────────────────────────────────────
    network = sac_factory.make_networks(
        observation_size=obs_size,
        action_size=action_size,
        unflatten_fn=unflatten_fn,
        learning_rate=cfg["algorithm"]["learning_rate"],
        network_config=network_config,
    )
    make_policy = sac_factory.make_inference_fn(network)

    # ── Load + fix params ─────────────────────────────────────────────
    params = load_params(f"{args.model_dir}/model/model_final.pkl")
    policy_params = params.policy
    policy_params = remap_keys(policy_params, "perceiver_attention", "lq_attention")
    policy_params = remap_keys(policy_params, "mgail_attention", "lq_attention")

    policy = make_policy(policy_params, deterministic=True)

    # ── Data ──────────────────────────────────────────────────────────
    data_gen = make_data_generator(
        path=args.data,
        max_num_objects=64,
        include_sdc_paths=True,
        batch_dims=(args.num_scenarios,),
        seed=0,
        repeat=True,
    )
    scenarios = next(data_gen)

    # ── Rollout ───────────────────────────────────────────────────────
    @jax.jit
    def eval_step(state, _):
        obs = state.observation
        raw_action, _ = policy(obs, None)
        action = wdatatypes.Action(
            data=raw_action,
            valid=jnp.ones((*raw_action.shape[:-1], 1), dtype=jnp.bool_),
        )
        next_state = env.step(state, action)
        metrics = {"reward": next_state.reward, "done": next_state.done,
                   **{k: v for k, v in next_state.metrics.items()}}
        return next_state, metrics

    CHUNK = args.chunk_size
    num_chunks = math.ceil(args.num_scenarios / CHUNK)
    all_metrics = {}
    rng = jax.random.PRNGKey(0)

    t0 = perf_counter()
    for i in range(num_chunks):
        s, e = i * CHUNK, min((i + 1) * CHUNK, args.num_scenarios)
        chunk = jax.tree_util.tree_map(lambda x: x[s:e], scenarios)
        rng, rk = jax.random.split(rng)
        state = jax.jit(env.reset)(chunk, jax.random.split(rk, e - s))
        _, chunk_metrics = jax.lax.scan(eval_step, state, None, length=80)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), chunk_metrics)
        for k, v in chunk_metrics.items():
            all_metrics.setdefault(k, []).append(v)
        print(f"  chunk {i+1}/{num_chunks} done")

    dt = perf_counter() - t0
    print(f"Rollout done in {dt:.1f}s")

    all_metrics = {k: jnp.concatenate(v, axis=1) for k, v in all_metrics.items()}

    # ── Aggregate ─────────────────────────────────────────────────────
    rewards = np.array(all_metrics["reward"])   # (80, N)
    dones   = np.array(all_metrics["done"])

    early_done = (dones[:-1] > 0.5).any(axis=0)
    accuracy   = float((~early_done).mean())

    FINAL = {"progress_ratio_nuplan", "sdc_progression", "log_divergence"}
    EVER  = {"at_fault_collision", "offroad", "overlap", "run_red_light",
             "sdc_off_route", "sdc_wrongway", "on_multiple_lanes"}

    results = {"accuracy": accuracy, "ep_return_mean": float(rewards.sum(axis=0).mean())}
    for key, arr in all_metrics.items():
        if key in {"reward", "done"}: continue
        arr = np.array(arr)
        if arr.ndim != 2: continue
        if key in FINAL:        val = arr[-1]
        elif key in EVER:       val = (arr > 0.5).any(axis=0).astype(float)
        else:                   val = arr.max(axis=0)
        results[key] = float(val.mean())

    # ── Print ─────────────────────────────────────────────────────────
    print("\n=== LQ BASELINE RESULTS ===")
    for k, v in sorted(results.items()):
        print(f"  {k:<35} {v:.4f}")

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results.keys()))
        w.writeheader(); w.writerow(results)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
