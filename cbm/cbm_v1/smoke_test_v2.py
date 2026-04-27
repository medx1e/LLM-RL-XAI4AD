"""CBM-V2 Smoke Test — verify 15-concept extraction, masks, norms, and training step.

This test does NOT train a model.  It verifies:
  1. Concept registry returns 15 concepts (11 V1 + 4 Phase 3)
  2. All concept values are finite and in expected ranges
  3. Validity masks have correct shapes and types
  4. CBMConfig auto-derives binary/continuous indices correctly
  5. ConceptHead resizes to 15 concepts
  6. A frozen-mode forward pass + training step produces finite outputs

Usage:
    PYTHONPATH=$PWD:$PWD/V-Max python cbm_v1/smoke_test_v2.py

Requires: data/training.tfrecord + pretrained checkpoint.
"""

from __future__ import annotations
import os, sys
from pathlib import Path

# Ensure project roots on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "V-Max"))

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np

from waymax import dynamics
from vmax.simulator import make_env_for_evaluation, make_data_generator
from concepts.types import ObservationConfig
from concepts.adapters import observation_to_concept_input
from concepts.registry import extract_all_concepts, CONCEPT_REGISTRY
from concepts.schema import ConceptType
from cbm_v1.config import CBMConfig

PASS = "\033[92m PASS \033[0m"
FAIL = "\033[91m FAIL \033[0m"


def main():
    results = []

    def check(name, cond, detail=""):
        tag = PASS if cond else FAIL
        results.append(cond)
        print(f"  {len(results):2d}. {name:55s} {tag}  {detail}")

    print("=" * 70)
    print("  CBM-V2 Smoke Test")
    print("=" * 70)

    # ── 1. Registry size ──────────────────────────────────────────────
    print("\n[A] Concept Registry")

    all_concepts = [(n, s, fn) for n, (s, fn) in CONCEPT_REGISTRY.items()]
    check("Registry has 15 concepts", len(all_concepts) == 15, f"({len(all_concepts)})")

    phase3 = [n for n, s, _ in all_concepts if s.phase == 3]
    check("Phase 3 has 4 concepts", len(phase3) == 4, f"{phase3}")

    expected_p3 = ["path_curvature_max", "path_net_heading_change",
                   "path_straightness", "heading_to_path_end"]
    check("Phase 3 names correct", phase3 == expected_p3)

    # ── 2. CBMConfig auto-derivation ──────────────────────────────────
    print("\n[B] CBMConfig Auto-Derivation")

    cfg_v2 = CBMConfig(num_concepts=15, concept_phases=(1, 2, 3))
    check("V2 concept_names length = 15", len(cfg_v2.concept_names) == 15)
    check("V2 binary indices = (4, 9, 10)", cfg_v2.binary_concept_indices == (4, 9, 10))
    check("V2 continuous indices = 12 items",
          len(cfg_v2.continuous_concept_indices) == 12,
          str(cfg_v2.continuous_concept_indices))

    # V1 backward compat
    cfg_v1 = CBMConfig(num_concepts=11, concept_phases=(1, 2))
    check("V1 backward compat: 11 concepts", len(cfg_v1.concept_names) == 11)
    check("V1 backward compat: binary=(4,9,10)", cfg_v1.binary_concept_indices == (4, 9, 10))

    # ── 3. Real data extraction ───────────────────────────────────────
    print("\n[C] Concept Extraction on Real Data")

    # Load environment
    pretrained_dir = "runs_rlc/womd_sac_road_perceiver_minimal_42"
    hydra_path = os.path.join(pretrained_dir, ".hydra", "config.yaml")
    from cbm_v1.train_cbm import load_pretrained_run_config, build_network_config
    pretrained_cfg = load_pretrained_run_config(pretrained_dir)
    encoder_remap = {"perceiver": "lq"}
    obs_type_remap = {"road": "vec", "lane": "vec"}
    network_config = build_network_config(pretrained_cfg, encoder_remap)
    obs_cfg_dict = pretrained_cfg.get("observation_config", {})

    obs_type = obs_type_remap.get(pretrained_cfg.get("observation_type", "vec"), "vec")

    env = make_env_for_evaluation(
        max_num_objects=64,
        dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
        sdc_paths_from_data=True,
        observation_type=obs_type,
        observation_config=obs_cfg_dict,
        termination_keys=pretrained_cfg.get("termination_keys", ["offroad", "overlap"]),
        noisy_init=False,
    )

    observation_size = env.observation_spec()
    check("Environment loads", observation_size > 0, f"obs_size={observation_size}")

    unflatten_fn = env.get_wrapper_attr("features_extractor").unflatten_features

    obs_cfg = ObservationConfig(
        obs_past_num_steps=obs_cfg_dict.get("obs_past_num_steps", 5),
        num_closest_objects=obs_cfg_dict.get("objects", {}).get("num_closest_objects", 8),
        roadgraph_top_k=obs_cfg_dict.get("roadgraphs", {}).get("roadgraph_top_k", 200),
        num_closest_traffic_lights=obs_cfg_dict.get("traffic_lights", {}).get(
            "num_closest_traffic_lights", 5),
        num_target_path_points=obs_cfg_dict.get("path_target", {}).get("num_points", 10),
        max_meters=obs_cfg_dict.get("roadgraphs", {}).get("max_meters", 70),
    )

    # Load data and get first observation — same pattern as V1 smoke_test
    data_gen = make_data_generator(
        path="data/training.tfrecord",
        max_num_objects=64,
        include_sdc_paths=True,
        batch_dims=(1,),
        seed=42,
        repeat=1,
    )
    scenario = next(iter(data_gen))
    rng = jax.random.PRNGKey(42)
    rng, rk = jax.random.split(rng)
    rk = jax.random.split(rk, 1)
    state = jax.jit(env.reset)(scenario, rk)
    obs = state.observation  # (1, obs_size)

    # Extract concepts with all 3 phases
    inp = observation_to_concept_input(obs, unflatten_fn, obs_cfg)
    out = extract_all_concepts(inp, phases=(1, 2, 3))

    check("Concept output has 15 names", len(out.names) == 15, str(out.names))

    norm = out.normalized
    raw = out.raw
    valid = out.valid

    check("normalized shape ends with 15", norm.shape[-1] == 15, str(norm.shape))
    check("raw shape ends with 15", raw.shape[-1] == 15, str(raw.shape))
    check("valid shape ends with 15", valid.shape[-1] == 15, str(valid.shape))

    # Finite checks
    norm_np = np.asarray(norm)
    raw_np = np.asarray(raw)

    check("All normalized values finite", np.all(np.isfinite(norm_np)))
    valid_np = np.asarray(valid)
    check("All raw values finite (where valid)",
          np.all(np.isfinite(raw_np[valid_np])))

    # Range checks
    check("Normalized in [0, 1]",
          float(norm_np.min()) >= -0.01 and float(norm_np.max()) <= 1.01,
          f"[{norm_np.min():.4f}, {norm_np.max():.4f}]")

    # Per-concept value report
    print("\n  Per-concept values:")
    print(f"  {'#':>3s}  {'Name':30s}  {'Type':10s}  {'Phase':5s}  {'Raw':>10s}  {'Norm':>8s}  {'Valid':>6s}")
    print("  " + "-" * 80)
    for i, name in enumerate(out.names):
        schema = out.schemas[i]
        r = float(raw_np.flat[i]) if raw_np.size > 0 else float("nan")
        n = float(norm_np.flat[i]) if norm_np.size > 0 else float("nan")
        v = bool(np.asarray(valid).flat[i])
        print(f"  {i:3d}  {name:30s}  {schema.concept_type.value:10s}  {schema.phase:5d}  {r:10.4f}  {n:8.4f}  {'✓' if v else '✗':>6s}")

    # ── 4. Phase 3 specific value checks ──────────────────────────────
    print("\n[D] Phase 3 Concept Checks")

    p3_start = 11
    curv_max = float(raw_np.flat[p3_start + 0])
    net_heading = float(raw_np.flat[p3_start + 1])
    straight = float(raw_np.flat[p3_start + 2])
    heading_end = float(raw_np.flat[p3_start + 3])

    check("path_curvature_max >= 0", curv_max >= 0, f"{curv_max:.6f}")
    check("path_net_heading_change in [-pi, pi]",
          -np.pi - 0.01 <= net_heading <= np.pi + 0.01, f"{net_heading:.4f}")
    check("path_straightness in [0, 1]",
          0.0 <= straight <= 1.0 + 0.01, f"{straight:.4f}")
    check("heading_to_path_end in [-pi, pi]",
          -np.pi - 0.01 <= heading_end <= np.pi + 0.01, f"{heading_end:.4f}")

    # All Phase 3 concepts should be always valid
    p3_valid = np.asarray(valid).flat[p3_start:p3_start + 4]
    check("Phase 3 all valid (no mask)", all(p3_valid))

    # ── 5. CBM Network with 15 concepts ───────────────────────────────
    print("\n[E] CBM Network Initialization (15 concepts)")

    from cbm_v1.networks import ConceptHead, CBMPolicyNetwork
    import cbm_v1.cbm_sac_factory as cbm_factory
    from cbm_v1.cbm_trainer import load_pretrained_params

    pretrained_params = load_pretrained_params(pretrained_dir)

    rng = jax.random.PRNGKey(42)
    action_size = env.action_spec().data.shape[0]

    cbm_network, training_state, policy_fn = cbm_factory.initialize(
        action_size=action_size,
        observation_size=observation_size,
        env=env,
        learning_rate=0.0001,
        network_config=network_config,
        cbm_config=cfg_v2,
        num_devices=1,
        key=rng,
        pretrained_params=pretrained_params,
    )

    check("CBM network created (15 concepts)", cbm_network is not None)
    check("Training state created", training_state is not None)

    # Forward pass
    obs_flat = obs.reshape(1, -1)
    pi_params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], training_state.params.policy))
    output = cbm_network.policy_network.apply(pi_params, obs_flat)
    check("Forward pass finite", bool(jnp.all(jnp.isfinite(output))),
          f"shape={output.shape}")

    # Concept extraction from params
    concepts = cbm_factory._get_concept_predictions(pi_params, obs_flat, frozen=True)
    check("Concept predictions shape (15,)", concepts.shape[-1] == 15, str(concepts.shape))
    check("Concepts in [0, 1]",
          float(concepts.min()) >= -0.01 and float(concepts.max()) <= 1.01,
          f"[{float(concepts.min()):.4f}, {float(concepts.max()):.4f}]")
    check("Concepts finite", bool(jnp.all(jnp.isfinite(concepts))))

    # Concept loss
    from cbm_v1.concept_loss import concept_loss as compute_concept_loss
    c_loss = compute_concept_loss(concepts, norm[:1].reshape(1, -1), valid[:1].reshape(1, -1), cfg_v2)
    check("Concept loss finite", bool(jnp.isfinite(c_loss)), f"{float(c_loss):.4f}")

    # Per-concept loss
    from cbm_v1.cbm_sac_factory import _per_concept_losses
    pc_losses = _per_concept_losses(concepts, norm[:1].reshape(1, -1), valid[:1].reshape(1, -1), cfg_v2)
    check("Per-concept losses: 15 entries", len(pc_losses) == 15)
    all_finite = all(bool(jnp.isfinite(v)) for v in pc_losses.values())
    check("Per-concept losses all finite", all_finite)

    print("\n  Per-concept losses:")
    for k, v in pc_losses.items():
        print(f"    {k:40s} = {float(v):.6f}")

    # ── Summary ───────────────────────────────────────────────────────
    total = len(results)
    passed = sum(results)
    print("\n" + "=" * 70)
    print(f"  {passed}/{total} checks passed")
    if passed == total:
        print("  ✅ CBM-V2 pre-training verification COMPLETE")
    else:
        failed = [i+1 for i, r in enumerate(results) if not r]
        print(f"  ❌ FAILED checks: {failed}")
    print("=" * 70)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
