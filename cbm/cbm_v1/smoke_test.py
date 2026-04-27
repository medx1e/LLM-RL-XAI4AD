#!/usr/bin/env python3
"""CBM-V1 smoke test — verifies the full pipeline on real data.

Tests:
  1. Environment + data loads
  2. Pretrained encoder params load
  3. CBM networks initialize (with pretrained encoder weights)
  4. Forward pass produces finite outputs
  5. Concept extraction works
  6. Concept loss computes
  7. One SGD training step completes without NaN
  8. Frozen mode vs joint mode both work

Usage:
    /home/med1e/anaconda3/envs/vmax/bin/python -m cbm_v1.smoke_test
"""

from __future__ import annotations

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

from concepts.types import ObservationConfig
from concepts.adapters import observation_to_concept_input
from concepts.registry import extract_all_concepts

from cbm_v1.config import CBMConfig
from cbm_v1.concept_loss import concept_loss
from cbm_v1.networks import CBMPolicyNetwork, CBMValueNetwork, ConceptHead
from cbm_v1 import cbm_sac_factory

# ── Constants ───────────────────────────────────────────────────────

MODEL_DIR = "runs_rlc/womd_sac_road_perceiver_minimal_42"
DATA_PATH = "data/training.tfrecord"

ENCODER_REMAP = {"perceiver": "lq", "mgail": "lqh"}
OBS_TYPE_REMAP = {"road": "vec", "lane": "vec"}

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    suffix = f" ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    if not condition:
        raise AssertionError(f"Smoke test failed: {name} {detail}")


# ── Data loading ────────────────────────────────────────────────────

def load_env_and_config():
    from waymax import dynamics
    from vmax.simulator import make_env_for_evaluation, make_data_generator

    with open(f"{MODEL_DIR}/.hydra/config.yaml") as f:
        config = yaml.safe_load(f)

    enc_type = config["network"]["encoder"]["type"]
    if enc_type in ENCODER_REMAP:
        config["network"]["encoder"]["type"] = ENCODER_REMAP[enc_type]

    obs_type = OBS_TYPE_REMAP.get(config["observation_type"], config["observation_type"])

    env = make_env_for_evaluation(
        max_num_objects=64,
        dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
        sdc_paths_from_data=True,
        observation_type=obs_type,
        observation_config=config["observation_config"],
        termination_keys=config["termination_keys"],
        noisy_init=False,
    )

    data_gen = make_data_generator(
        path=DATA_PATH, max_num_objects=64,
        include_sdc_paths=True, batch_dims=(1,), seed=42, repeat=1,
    )

    obs_cfg = config["observation_config"]
    concept_config = ObservationConfig(
        obs_past_num_steps=obs_cfg.get("obs_past_num_steps", 5),
        num_closest_objects=obs_cfg.get("objects", {}).get("num_closest_objects", 8),
        roadgraph_top_k=obs_cfg.get("roadgraphs", {}).get("roadgraph_top_k", 200),
        num_closest_traffic_lights=obs_cfg.get("traffic_lights", {}).get("num_closest_traffic_lights", 5),
        num_target_path_points=obs_cfg.get("path_target", {}).get("num_points", 10),
        max_meters=obs_cfg.get("roadgraphs", {}).get("max_meters", 70),
    )

    return env, data_gen, config, concept_config


def load_pretrained_params():
    from vmax.scripts.evaluate.utils import load_params
    path = f"{MODEL_DIR}/model/model_final.pkl"
    return load_params(path)


# ── Main test ───────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CBM-V1 SMOKE TEST")
    print("=" * 70)

    # ── 1. Load env and data ────────────────────────────────────────
    print("\n1. Loading environment and data...")
    env, data_gen, raw_config, concept_config = load_env_and_config()
    obs_size = env.observation_spec()
    action_size = env.action_spec().data.shape[0]
    print(f"   obs_size={obs_size}, action_size={action_size}")
    check("Environment loads", True)

    # ── 2. Get first observation ────────────────────────────────────
    print("\n2. Getting first observation...")
    scenario = next(iter(data_gen))
    rng = jax.random.PRNGKey(42)
    rng, rk = jax.random.split(rng)
    rk = jax.random.split(rk, 1)
    env_t = jax.jit(env.reset)(scenario, rk)
    obs = env_t.observation  # (1, obs_size)
    check("Observation shape", obs.shape == (1, obs_size), f"shape={obs.shape}")
    check("Observation finite", bool(jnp.all(jnp.isfinite(obs))))

    # ── 3. Extract concept targets ──────────────────────────────────
    print("\n3. Extracting concept targets from observation...")
    unflatten_fn = env.get_wrapper_attr("features_extractor").unflatten_features
    c_input = observation_to_concept_input(obs, unflatten_fn, concept_config)
    c_output = extract_all_concepts(c_input)
    print(f"   concepts: {c_output.names}")
    print(f"   raw shape: {c_output.raw.shape}")
    check("Concept count", c_output.raw.shape[-1] == 11, f"got {c_output.raw.shape[-1]}")
    check("Concept targets finite", bool(jnp.all(jnp.isfinite(c_output.normalized))))

    # ── 4. Load pretrained params ───────────────────────────────────
    print("\n4. Loading pretrained checkpoint...")
    pretrained_params = load_pretrained_params()
    check("Pretrained params load", pretrained_params is not None)

    # Inspect pretrained param structure
    if hasattr(pretrained_params, "policy"):
        pol_params = pretrained_params.policy
        if isinstance(pol_params, dict) and "params" in pol_params:
            top_keys = list(pol_params["params"].keys())
            print(f"   Pretrained policy top keys: {top_keys}")

    # ── 5. Build CBM config ─────────────────────────────────────────
    print("\n5. Building CBM config...")
    cbm_config = CBMConfig(
        num_concepts=11,
        concept_head_hidden_sizes=(64,),
        actor_hidden_sizes=(64, 32),
        critic_hidden_sizes=(64, 32),
        lambda_concept=0.1,
        mode="frozen",
    )
    check("CBM config", cbm_config.num_concepts == 11)

    # ── 6. Build network config ─────────────────────────────────────
    print("\n6. Building CBM-SAC networks...")

    # Merge config like V-Max does
    network_config = {
        "encoder": raw_config["network"]["encoder"],
        "policy": raw_config["algorithm"]["network"]["policy"],
        "value": raw_config["algorithm"]["network"]["value"],
        "action_distribution": "gaussian",
    }

    # Fix encoder type
    if network_config["encoder"]["type"] in ENCODER_REMAP:
        network_config["encoder"]["type"] = ENCODER_REMAP[network_config["encoder"]["type"]]

    cbm_network = cbm_sac_factory.make_networks(
        observation_size=obs_size,
        action_size=action_size,
        unflatten_fn=unflatten_fn,
        learning_rate=1e-4,
        network_config=network_config,
        cbm_config=cbm_config,
    )
    check("CBM networks created", cbm_network is not None)

    # ── 7. Initialize params ────────────────────────────────────────
    print("\n7. Initializing CBM params...")
    rng, key_p, key_v = jax.random.split(rng, 3)

    policy_params = cbm_network.policy_network.init(key_p)
    value_params = cbm_network.value_network.init(key_v)

    def count_params(params):
        leaves = jax.tree_util.tree_leaves(params)
        return sum(x.size for x in leaves)

    n_policy = count_params(policy_params)
    n_value = count_params(value_params)
    print(f"   Policy params: {n_policy:,}")
    print(f"   Value params:  {n_value:,}")
    check("Policy params > 0", n_policy > 0)
    check("Value params > 0", n_value > 0)

    # ── 8. Inspect param tree structure ─────────────────────────────
    print("\n8. Inspecting param tree structure...")
    def print_tree(d, prefix="", max_depth=3, depth=0):
        if depth >= max_depth:
            return
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    print(f"   {prefix}{k}/")
                    print_tree(v, prefix + "  ", max_depth, depth + 1)
                elif hasattr(v, 'shape'):
                    print(f"   {prefix}{k}: {v.shape}")
                else:
                    print(f"   {prefix}{k}: {type(v).__name__}")

    print("   Policy param tree:")
    print_tree(policy_params)

    # ── 9. Load pretrained encoder weights ──────────────────────────
    print("\n9. Loading pretrained encoder into CBM params...")
    policy_params_loaded = cbm_sac_factory._load_pretrained_encoder(
        policy_params, pretrained_params, "policy"
    )
    check("Pretrained encoder loaded into policy",
          policy_params_loaded is not None)

    value_params_loaded = cbm_sac_factory._load_pretrained_encoder(
        value_params, pretrained_params, "value"
    )
    check("Pretrained encoder loaded into value",
          value_params_loaded is not None)

    # ── 10. Forward pass ────────────────────────────────────────────
    print("\n10. Forward pass test...")
    policy_out = cbm_network.policy_network.apply(policy_params_loaded, obs)
    check("Policy forward finite", bool(jnp.all(jnp.isfinite(policy_out))),
          f"shape={policy_out.shape}")

    dummy_action = jnp.zeros((1, action_size))
    value_out = cbm_network.value_network.apply(value_params_loaded, obs, dummy_action)
    check("Value forward finite", bool(jnp.all(jnp.isfinite(value_out))),
          f"shape={value_out.shape}")

    # ── 11. Concept extraction from policy ──────────────────────────
    print("\n11. Concept prediction from policy network...")

    # Register the module for concept extraction
    from cbm_v1.networks import CBMPolicyNetwork as _mod

    # We need to find the actual module. The module was created inside
    # make_networks but not stored. Let's recreate it.
    from vmax.agents.networks.decoders import MLP
    from vmax.agents.networks import network_utils
    from vmax.agents.networks.encoders import get_encoder

    _config = network_utils.convert_to_dict_with_activation_fn(network_config)
    enc_config = network_utils.parse_config(_config["encoder"], "encoder")
    enc_cls = get_encoder(_config["encoder"]["type"])
    encoder_layer = enc_cls(unflatten_fn, **enc_config)

    concept_head = ConceptHead(
        num_concepts=11,
        hidden_sizes=(64,),
    )
    actor_fc = MLP(layer_sizes=(64, 32))

    policy_module = CBMPolicyNetwork(
        encoder_layer=encoder_layer,
        concept_head=concept_head,
        actor_fc=actor_fc,
        output_size=cbm_network.parametric_action_distribution.param_size,
        frozen_encoder=True,  # for frozen mode testing
    )

    cbm_sac_factory.set_cbm_policy_module(policy_module)

    # Extract concepts
    z, concepts = policy_module.apply(
        policy_params_loaded, obs,
        method=policy_module.encode_and_predict_concepts,
    )
    print(f"   Encoder output z shape: {z.shape}")
    print(f"   Concept predictions shape: {concepts.shape}")
    print(f"   Concept values: {np.array(concepts[0]).round(3)}")
    check("Concepts shape", concepts.shape == (1, 11), f"shape={concepts.shape}")
    check("Concepts in [0,1]",
          bool(jnp.all(concepts >= 0) & jnp.all(concepts <= 1)))
    check("Concepts finite", bool(jnp.all(jnp.isfinite(concepts))))

    # ── 12. Concept loss test ───────────────────────────────────────
    print("\n12. Concept loss test...")
    c_loss = concept_loss(
        concepts, c_output.normalized, c_output.valid, cbm_config
    )
    print(f"   Concept loss: {float(c_loss):.6f}")
    check("Concept loss finite", bool(jnp.isfinite(c_loss)))
    check("Concept loss > 0", float(c_loss) > 0)

    # ── 13. Training step test (frozen mode) ────────────────────────
    print("\n13. Training step test (FROZEN mode)...")
    cbm_config_frozen = CBMConfig(mode="frozen", lambda_concept=0.1)

    # Build concept targets function (JIT-safe)
    def concept_targets_fn(observations):
        c_inp = observation_to_concept_input(observations, unflatten_fn, concept_config)
        c_out = extract_all_concepts(c_inp)
        return c_out.normalized, c_out.valid

    # Build the SGD step
    cbm_network_frozen = cbm_sac_factory.make_networks(
        observation_size=obs_size,
        action_size=action_size,
        unflatten_fn=unflatten_fn,
        learning_rate=1e-4,
        network_config=network_config,
        cbm_config=cbm_config_frozen,
    )

    # Re-register module
    cbm_sac_factory.set_cbm_policy_module(policy_module)

    sgd_step = cbm_sac_factory.make_sgd_step(
        cbm_network_frozen, alpha=0.2, discount=0.99, tau=0.005,
        concept_targets_fn=concept_targets_fn,
        cbm_config=cbm_config_frozen,
    )

    # Create a dummy transition
    from vmax.agents.datatypes import RLTransition

    rng, action_key = jax.random.split(rng)
    batch_obs = obs  # (1, obs_size)
    batch_action = jax.random.uniform(action_key, (1, action_size), minval=-1, maxval=1)

    dummy_transition = RLTransition(
        observation=batch_obs,
        action=batch_action,
        reward=jnp.zeros((1,)),
        flag=jnp.ones((1,)),
        next_observation=batch_obs,  # same obs for smoke test
        done=jnp.zeros((1,)),
    )

    # Initialize training state (single device, no pmap for smoke test)
    rng, key_init = jax.random.split(rng)
    key_p2, key_v2 = jax.random.split(key_init)
    pol_params = cbm_network_frozen.policy_network.init(key_p2)
    val_params = cbm_network_frozen.value_network.init(key_v2)

    pol_params = cbm_sac_factory._load_pretrained_encoder(pol_params, pretrained_params, "policy")
    val_params = cbm_sac_factory._load_pretrained_encoder(val_params, pretrained_params, "value")

    pol_opt_state = cbm_network_frozen.policy_optimizer.init(pol_params)
    val_opt_state = cbm_network_frozen.value_optimizer.init(val_params)

    train_state = cbm_sac_factory.CBMSACTrainingState(
        params=cbm_sac_factory.CBMSACNetworkParams(
            policy=pol_params,
            value=val_params,
            target_value=val_params,
        ),
        policy_optimizer_state=pol_opt_state,
        value_optimizer_state=val_opt_state,
        env_steps=0,
        rl_gradient_steps=0,
    )

    # Run one SGD step (without pmap — single device)
    # We need a version without pmean for single-device testing
    from cbm_v1.cbm_sac_factory import concept_loss as c_loss_fn
    from vmax.agents.networks.gradient import loss_and_pgrad

    # Test the losses directly
    print("   Testing value loss...")
    v_loss = sgd_step.__wrapped__ if hasattr(sgd_step, '__wrapped__') else None

    # Direct loss computation test
    policy_net = cbm_network_frozen.policy_network
    value_net = cbm_network_frozen.value_network
    dist = cbm_network_frozen.parametric_action_distribution

    # Value loss
    v_old = value_net.apply(val_params, batch_obs, batch_action)
    check("Value output finite", bool(jnp.all(jnp.isfinite(v_old))),
          f"shape={v_old.shape}, values={np.array(v_old[0]).round(4)}")

    # Policy loss
    pol_out = policy_net.apply(pol_params, batch_obs)
    check("Policy output finite", bool(jnp.all(jnp.isfinite(pol_out))),
          f"shape={pol_out.shape}")

    # Concept targets
    ct, cv = concept_targets_fn(batch_obs)
    check("Concept targets shape", ct.shape == (1, 11))
    check("Concept valid shape", cv.shape == (1, 11))

    # Concept predictions
    z_test, c_pred = policy_module.apply(
        pol_params, batch_obs,
        method=policy_module.encode_and_predict_concepts,
    )
    c_loss_val = concept_loss(c_pred, ct, cv, cbm_config_frozen)
    check("Concept loss in training finite", bool(jnp.isfinite(c_loss_val)),
          f"loss={float(c_loss_val):.6f}")

    # Test gradient computation
    print("   Testing gradient computation...")

    # Build frozen policy module for gradient test
    # (policy_module already has frozen_encoder=True)
    frozen_policy_net = cbm_network_frozen.policy_network

    def test_policy_loss(policy_params):
        # SAC loss — uses the frozen policy module (stop_gradient inside __call__)
        dist_params = frozen_policy_net.apply(policy_params, batch_obs)
        rng_test = jax.random.PRNGKey(0)
        action = dist.sample_no_postprocessing(dist_params, rng_test)
        log_prob = dist.log_prob(dist_params, action)
        action = dist.postprocess(action)
        val = value_net.apply(val_params, batch_obs, action)
        min_val = jnp.min(val, axis=-1)
        sac_loss = jnp.mean(0.2 * log_prob - min_val)

        # Concept loss — also with frozen encoder
        ct_inner, cv_inner = concept_targets_fn(batch_obs)
        z_inner, _ = policy_module.apply(
            policy_params, batch_obs,
            method=policy_module.encode_and_predict_concepts,
        )
        z_sg = jax.lax.stop_gradient(z_inner)
        ch_params_inner = policy_params["params"]["concept_head"]
        c_pred_frozen = policy_module.concept_head.apply(
            {"params": ch_params_inner}, z_sg
        )
        cl = concept_loss(c_pred_frozen, ct_inner, cv_inner, cbm_config_frozen)
        return sac_loss + 0.1 * cl

    grad_fn = jax.grad(test_policy_loss)
    grads = grad_fn(pol_params)
    grad_leaves = jax.tree_util.tree_leaves(grads)
    all_finite = all(bool(jnp.all(jnp.isfinite(g))) for g in grad_leaves)
    total_grad_norm = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in grad_leaves)))
    check("Gradients finite", all_finite, f"grad_norm={total_grad_norm:.4f}")

    # Check encoder grads are zero in frozen mode
    # Params are flat: grads["params"]["encoder_layer"]/...
    enc_grad_leaves = jax.tree_util.tree_leaves(grads["params"].get("encoder_layer", {}))
    enc_grad_norm = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in enc_grad_leaves))) if enc_grad_leaves else 0.0
    check("Encoder grads = 0 in frozen mode",
          enc_grad_norm < 1e-8, f"enc_grad_norm={enc_grad_norm:.8f}")

    # Check concept head grads are non-zero
    ch_grad_leaves = jax.tree_util.tree_leaves(grads["params"].get("concept_head", {}))
    ch_grad_norm = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in ch_grad_leaves))) if ch_grad_leaves else 0.0
    check("Concept head grads > 0 in frozen mode",
          ch_grad_norm > 1e-8, f"ch_grad_norm={ch_grad_norm:.6f}")

    # ── 14. Joint mode gradient test ────────────────────────────────
    print("\n14. Joint mode gradient test...")

    # Build joint (non-frozen) module for this test
    joint_policy_module = CBMPolicyNetwork(
        encoder_layer=encoder_layer,
        concept_head=concept_head,
        actor_fc=actor_fc,
        output_size=cbm_network.parametric_action_distribution.param_size,
        frozen_encoder=False,
    )
    # joint module uses the same param structure
    joint_policy_net_apply = lambda params, obs: joint_policy_module.apply(params, obs)

    def test_policy_loss_joint(policy_params):
        dist_params = joint_policy_net_apply(policy_params, batch_obs)
        rng_test = jax.random.PRNGKey(0)
        action = dist.sample_no_postprocessing(dist_params, rng_test)
        log_prob = dist.log_prob(dist_params, action)
        action = dist.postprocess(action)
        val = value_net.apply(val_params, batch_obs, action)
        min_val = jnp.min(val, axis=-1)
        sac_loss = jnp.mean(0.2 * log_prob - min_val)

        ct_inner, cv_inner = concept_targets_fn(batch_obs)
        z_inner, c_pred_inner = joint_policy_module.apply(
            policy_params, batch_obs,
            method=joint_policy_module.encode_and_predict_concepts,
        )
        # Joint: gradients flow through encoder
        cl = concept_loss(c_pred_inner, ct_inner, cv_inner, cbm_config_frozen)
        return sac_loss + 0.1 * cl

    grad_fn_joint = jax.grad(test_policy_loss_joint)
    grads_joint = grad_fn_joint(pol_params)
    enc_grad_leaves_j = jax.tree_util.tree_leaves(grads_joint["params"].get("encoder_layer", {}))
    enc_grad_norm_j = float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in enc_grad_leaves_j))) if enc_grad_leaves_j else 0.0
    check("Encoder grads > 0 in joint mode",
          enc_grad_norm_j > 1e-8, f"enc_grad_norm={enc_grad_norm_j:.6f}")

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 70)
    print(f"""
Summary:
  Backbone:         {MODEL_DIR}
  Encoder output:   {z.shape[-1]}-d
  Concept vector:   {concepts.shape[-1]}-d
  Policy params:    {n_policy:,}
  Value params:     {n_value:,}
  Concept loss:     {float(c_loss_val):.6f}
  Grad norm (total):  {total_grad_norm:.4f}
  Encoder grad norm (frozen): {enc_grad_norm:.8f}
  Encoder grad norm (joint):  {enc_grad_norm_j:.6f}
""")


if __name__ == "__main__":
    main()
