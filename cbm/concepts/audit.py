#!/usr/bin/env python3
"""Concept audit script — inspect TFRecord data and report concept statistics.

Usage (from ~/cbm):
    conda activate vmax
    python -m concepts.audit --data data/training.tfrecord \
                             --model_dir runs_rlc/womd_sac_road_perceiver_minimal_42 \
                             --num_scenarios 50

Reports:
  1. Observation feature groups and shapes
  2. Which concepts are computable
  3. Per-concept: valid rate, min/max/mean/std (continuous), positive rate (binary)
  4. Clipping / masking statistics
  5. Example dumps for debugging
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "V-Max"))

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from concepts.types import ConceptInput, ObservationConfig
from concepts.schema import ConceptType
from concepts.registry import extract_all_concepts, CONCEPT_REGISTRY
from concepts.adapters import observation_to_concept_input


# ---- Remapping tables (from GUIDE2LOAD_MODELS) -----------------------
ENCODER_REMAP = {"perceiver": "lq", "mgail": "lqh"}
OBS_TYPE_REMAP = {"road": "vec", "lane": "vec"}
PARAM_KEY_REMAP = {"perceiver_attention": "lq_attention", "mgail_attention": "lq_attention"}


def remap_param_keys(params, old_name, new_name):
    if isinstance(params, dict):
        return {
            (new_name if k == old_name else k): remap_param_keys(v, old_name, new_name)
            for k, v in params.items()
        }
    return params


def load_env_and_data(model_dir: str, data_path: str, max_num_objects: int = 64):
    """Load V-Max environment and data generator with all compatibility fixes."""
    from waymax import dynamics
    from vmax.simulator import make_env_for_evaluation, make_data_generator
    from vmax.agents.learning.reinforcement.sac.sac_factory import make_inference_fn, make_networks
    from vmax.scripts.evaluate.utils import load_params

    with open(f"{model_dir}/.hydra/config.yaml") as f:
        config = yaml.safe_load(f)

    # Fix encoder type
    encoder_type = config["network"]["encoder"]["type"]
    if encoder_type in ENCODER_REMAP:
        config["network"]["encoder"]["type"] = ENCODER_REMAP[encoder_type]

    # Fix observation type
    obs_type = config["observation_type"]
    obs_type = OBS_TYPE_REMAP.get(obs_type, obs_type)

    # Check speed_limit
    rg_features = config.get("observation_config", {}).get("roadgraphs", {}).get("features", [])
    if "speed_limit" in rg_features:
        raise RuntimeError(f"Model uses speed_limit feature (unsupported)")

    # Build environment
    env = make_env_for_evaluation(
        max_num_objects=max_num_objects,
        dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
        sdc_paths_from_data=True,
        observation_type=obs_type,
        observation_config=config["observation_config"],
        termination_keys=config["termination_keys"],
        noisy_init=False,
    )

    data_gen = make_data_generator(
        path=data_path,
        max_num_objects=max_num_objects,
        include_sdc_paths=True,
        batch_dims=(1,),
        seed=42,
        repeat=1,
    )

    unflatten_fn = env.get_wrapper_attr("features_extractor").unflatten_features

    obs_cfg = config["observation_config"]
    concept_config = ObservationConfig(
        obs_past_num_steps=obs_cfg.get("obs_past_num_steps", 5),
        num_closest_objects=obs_cfg.get("objects", {}).get("num_closest_objects", 8),
        roadgraph_top_k=obs_cfg.get("roadgraphs", {}).get("roadgraph_top_k", 200),
        num_closest_traffic_lights=obs_cfg.get("traffic_lights", {}).get("num_closest_traffic_lights", 5),
        num_target_path_points=obs_cfg.get("path_target", {}).get("num_points", 10),
        max_meters=obs_cfg.get("roadgraphs", {}).get("max_meters", 70),
    )

    return env, data_gen, unflatten_fn, concept_config, config


def report_observation_shapes(unflatten_fn, obs):
    """Print observation feature group shapes."""
    features, masks = unflatten_fn(obs)
    names = ["SDC trajectory", "Other agents", "Roadgraph", "Traffic lights", "GPS path"]
    mask_names = ["SDC mask", "Agent mask", "RG mask", "TL mask"]

    print("\n" + "=" * 70)
    print("OBSERVATION FEATURE GROUPS AND SHAPES")
    print("=" * 70)
    for name, feat in zip(names, features):
        print(f"  {name:20s}: {feat.shape}")
    for name, mask in zip(mask_names, masks):
        print(f"  {name:20s}: {mask.shape}")
    print()


def report_concept_registry():
    """Print which concepts are registered and computable."""
    print("=" * 70)
    print("REGISTERED CONCEPTS")
    print("=" * 70)
    for name, (schema, _) in CONCEPT_REGISTRY.items():
        status = "ACTIVE"
        print(f"  Phase {schema.phase} | {schema.concept_type.value:10s} | {name:30s} | {status}")
    print()


def collect_concept_stats(
    env, data_gen, unflatten_fn, concept_config, num_scenarios, num_steps_per_scenario=1,
):
    """Run concept extraction on multiple scenarios and collect statistics."""
    all_raws = []
    all_valids = []
    concept_names = None

    reset_fn = jax.jit(env.reset)
    rng = jax.random.PRNGKey(0)

    count = 0
    for scenario in data_gen:
        if count >= num_scenarios:
            break

        rng, reset_key = jax.random.split(rng)
        reset_key = jax.random.split(reset_key, 1)

        try:
            env_transition = reset_fn(scenario, reset_key)
        except Exception as e:
            print(f"  [WARN] Scenario {count} reset failed: {e}")
            count += 1
            continue

        obs = env_transition.observation
        inp = observation_to_concept_input(obs, unflatten_fn, concept_config)
        out = extract_all_concepts(inp)

        if concept_names is None:
            concept_names = out.names

        # Remove batch dim (1,)
        all_raws.append(np.array(out.raw[0]))
        all_valids.append(np.array(out.valid[0]))

        count += 1
        if count % 10 == 0:
            print(f"  Processed {count}/{num_scenarios} scenarios...")

    raws = np.stack(all_raws, axis=0)   # (S, C)
    valids = np.stack(all_valids, axis=0)  # (S, C)
    return concept_names, raws, valids


def print_stats(concept_names, raws, valids):
    """Print per-concept statistics."""
    print("=" * 70)
    print("PER-CONCEPT STATISTICS")
    print("=" * 70)

    S = raws.shape[0]

    for i, name in enumerate(concept_names):
        schema, _ = CONCEPT_REGISTRY[name]
        col_raw = raws[:, i]
        col_valid = valids[:, i].astype(bool)
        valid_count = col_valid.sum()
        valid_rate = valid_count / S * 100

        print(f"\n  --- {name} ({schema.concept_type.value}, phase {schema.phase}) ---")
        print(f"    Valid rate:  {valid_rate:.1f}% ({valid_count}/{S})")

        if valid_count == 0:
            print(f"    [NO VALID SAMPLES]")
            continue

        valid_vals = col_raw[col_valid]

        if schema.concept_type == ConceptType.BINARY:
            pos_rate = valid_vals.mean() * 100
            print(f"    Positive rate: {pos_rate:.1f}%")
        else:
            print(f"    Min:   {valid_vals.min():.4f}")
            print(f"    Max:   {valid_vals.max():.4f}")
            print(f"    Mean:  {valid_vals.mean():.4f}")
            print(f"    Std:   {valid_vals.std():.4f}")

            # Clipping stats (values at boundary)
            if name in ("ego_speed",):
                clipped_hi = (valid_vals >= 29.9).sum()
                print(f"    Clipped at max: {clipped_hi}/{valid_count}")
            elif name in ("dist_nearest_object", "dist_to_traffic_light"):
                at_max = (valid_vals >= 69.5).sum()
                print(f"    At max distance: {at_max}/{valid_count}")

    # Masking summary
    print(f"\n{'=' * 70}")
    print("MASKING SUMMARY")
    print("=" * 70)
    for i, name in enumerate(concept_names):
        valid_rate = valids[:, i].mean() * 100
        print(f"  {name:35s}: {valid_rate:5.1f}% valid")


def print_examples(concept_names, raws, valids, n=3):
    """Print a few example observations for debugging."""
    print(f"\n{'=' * 70}")
    print(f"EXAMPLE DUMPS (first {n} scenarios)")
    print("=" * 70)

    for s in range(min(n, raws.shape[0])):
        print(f"\n  --- Scenario {s} ---")
        for i, name in enumerate(concept_names):
            v = "VALID" if valids[s, i] else "INVALID"
            print(f"    {name:35s}: {raws[s, i]:10.4f}  [{v}]")


def main():
    parser = argparse.ArgumentParser(description="Concept extraction audit")
    parser.add_argument("--data", type=str, default="data/training.tfrecord")
    parser.add_argument("--model_dir", type=str,
                        default="runs_rlc/womd_sac_road_perceiver_minimal_42")
    parser.add_argument("--num_scenarios", type=int, default=50)
    parser.add_argument("--max_num_objects", type=int, default=64)
    args = parser.parse_args()

    print("Loading environment and data...")
    env, data_gen, unflatten_fn, concept_config, raw_config = load_env_and_data(
        args.model_dir, args.data, args.max_num_objects
    )

    # Report observation shapes from first scenario
    print("Getting first observation...")
    scenario = next(iter(data_gen))
    rng = jax.random.PRNGKey(0)
    _, reset_key = jax.random.split(rng)
    reset_key = jax.random.split(reset_key, 1)
    env_transition = jax.jit(env.reset)(scenario, reset_key)
    obs = env_transition.observation

    report_observation_shapes(unflatten_fn, obs)
    report_concept_registry()

    # Re-create data generator (consumed one scenario)
    data_gen = __import__("vmax.simulator", fromlist=["make_data_generator"]).make_data_generator(
        path=args.data,
        max_num_objects=args.max_num_objects,
        include_sdc_paths=True,
        batch_dims=(1,),
        seed=42,
        repeat=1,
    )

    print(f"Collecting concept stats over {args.num_scenarios} scenarios...")
    concept_names, raws, valids = collect_concept_stats(
        env, data_gen, unflatten_fn, concept_config, args.num_scenarios
    )

    print_stats(concept_names, raws, valids)
    print_examples(concept_names, raws, valids, n=3)

    print(f"\n{'=' * 70}")
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
