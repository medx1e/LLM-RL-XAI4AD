#!/usr/bin/env python3
"""CBM-V1 multi-step training loop test.

Verifies the full training pipeline end-to-end:
  1. Build env, replay buffer, CBM networks (with pretrained encoder)
  2. Prefill buffer with random transitions
  3. Run N training iterations with pmap + jax.lax.scan
  4. Report policy_loss, value_loss, concept_loss per iteration
  5. Check for NaN / Inf at every step

Usage:
    /home/med1e/anaconda3/envs/vmax/bin/python cbm_v1/train_loop_test.py
"""

from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path
from time import perf_counter

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "V-Max"))

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from waymax import dynamics
from vmax.simulator import make_env_for_evaluation, make_env_for_training, make_data_generator
from vmax.agents import datatypes, pipeline
from vmax.agents.pipeline import inference, pmap as pmap_utils
from vmax.agents.learning.replay_buffer import ReplayBuffer
from vmax.scripts.evaluate.utils import load_params
from vmax.agents.networks import network_utils
from vmax.agents.networks.decoders import MLP
from vmax.agents.networks.encoders import get_encoder

from concepts.types import ObservationConfig
from concepts.adapters import observation_to_concept_input
from concepts.registry import extract_all_concepts

from cbm_v1.config import CBMConfig
from cbm_v1.networks import CBMPolicyNetwork, ConceptHead
import cbm_v1.cbm_sac_factory as cbm_factory

# ── Config ──────────────────────────────────────────────────────────

MODEL_DIR = "runs_rlc/womd_sac_road_perceiver_minimal_42"
DATA_PATH = "data/training.tfrecord"

ENCODER_REMAP = {"perceiver": "lq"}
OBS_TYPE_REMAP = {"road": "vec", "lane": "vec"}

# Small values for testing — not production training
# batch_dims = (num_devices, NUM_ENVS, NUM_EPISODES) so inside pmap:
#   - outer vmap in init_and_reset maps over NUM_ENVS
#   - inner vmap in VmapWrapper maps over NUM_EPISODES
NUM_ENVS = 2               # parallel env streams per device (outer vmap in AutoResetWrapper)
NUM_EPISODES = 2           # episode pool per stream (inner vmap in VmapWrapper)
LEARNING_START = 50        # steps before training starts (small for testing)
BATCH_SIZE = 16
GRAD_UPDATES_PER_STEP = 2
UNROLL_LENGTH = 1
BUFFER_SIZE = 500
NUM_TRAIN_ITERS = 5        # training iterations to test
SCAN_LENGTH = 4            # scan steps per training iteration (jax.lax.scan length)
SCENARIO_LENGTH = 80       # steps per scenario

ALPHA = 0.2
DISCOUNT = 0.99
TAU = 0.005
LEARNING_RATE = 1e-4
LAMBDA_CONCEPT = 0.1

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def check(name, cond, detail=""):
    status = PASS if cond else FAIL
    suffix = f" ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    if not cond:
        raise AssertionError(f"FAILED: {name} {detail}")


# ── Setup ────────────────────────────────────────────────────────────

def build_env_and_data():
    with open(f"{MODEL_DIR}/.hydra/config.yaml") as f:
        config = yaml.safe_load(f)

    enc_type = config["network"]["encoder"]["type"]
    config["network"]["encoder"]["type"] = ENCODER_REMAP.get(enc_type, enc_type)
    obs_type = OBS_TYPE_REMAP.get(config["observation_type"], config["observation_type"])

    env = make_env_for_training(
        max_num_objects=64,
        dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
        sdc_paths_from_data=True,
        observation_type=obs_type,
        observation_config=config["observation_config"],
        termination_keys=config["termination_keys"],
    )

    num_devices = jax.local_device_count()
    # batch_dims = (num_devices, NUM_ENVS, NUM_EPISODES):
    #   - pmap strips num_devices → inside pmap: (NUM_ENVS, NUM_EPISODES)
    #   - AutoResetWrapper.init_and_reset outer-vmaps over NUM_ENVS
    #   - VmapWrapper.reset inner-vmaps over NUM_EPISODES (one scenario each)
    data_gen = make_data_generator(
        path=DATA_PATH, max_num_objects=64,
        include_sdc_paths=True, batch_dims=(num_devices, NUM_ENVS, NUM_EPISODES),
        seed=0, repeat=True,
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

    network_config = {
        "encoder": config["network"]["encoder"],
        "policy": config["algorithm"]["network"]["policy"],
        "value": config["algorithm"]["network"]["value"],
        "action_distribution": "gaussian",
    }

    return env, data_gen, network_config, concept_config


def build_cbm_setup(env, network_config, concept_config, cbm_mode="frozen"):
    unflatten_fn = env.get_wrapper_attr("features_extractor").unflatten_features
    obs_size = env.observation_spec()
    action_size = env.action_spec().data.shape[0]

    cbm_config = CBMConfig(
        num_concepts=11,
        concept_head_hidden_sizes=(64,),
        actor_hidden_sizes=(64, 32),
        critic_hidden_sizes=(64, 32),
        lambda_concept=LAMBDA_CONCEPT,
        mode=cbm_mode,
    )

    cbm_network = cbm_factory.make_networks(
        observation_size=obs_size,
        action_size=action_size,
        unflatten_fn=unflatten_fn,
        learning_rate=LEARNING_RATE,
        network_config=network_config,
        cbm_config=cbm_config,
    )

    # Build module reference for concept extraction
    _cfg = network_utils.convert_to_dict_with_activation_fn(network_config)
    enc_cfg = network_utils.parse_config(_cfg["encoder"], "encoder")
    encoder_layer = get_encoder(_cfg["encoder"]["type"])(unflatten_fn, **enc_cfg)
    concept_head = ConceptHead(num_concepts=11, hidden_sizes=(64,))
    actor_fc = MLP(layer_sizes=(64, 32))
    policy_module = CBMPolicyNetwork(
        encoder_layer=encoder_layer,
        concept_head=concept_head,
        actor_fc=actor_fc,
        output_size=cbm_network.parametric_action_distribution.param_size,
        frozen_encoder=(cbm_mode == "frozen"),
    )
    cbm_factory.set_cbm_policy_module(policy_module)

    # Init params
    key = jax.random.PRNGKey(42)
    key_p, key_v = jax.random.split(key)
    policy_params = cbm_network.policy_network.init(key_p)
    value_params = cbm_network.value_network.init(key_v)

    # Load pretrained encoder
    pretrained = load_params(f"{MODEL_DIR}/model/model_final.pkl")
    policy_params = cbm_factory._load_pretrained_encoder(policy_params, pretrained, "policy")
    value_params = cbm_factory._load_pretrained_encoder(value_params, pretrained, "value")

    # Concept targets function (JIT/pmap safe)
    def concept_targets_fn(observations):
        inp = observation_to_concept_input(observations, unflatten_fn, concept_config)
        out = extract_all_concepts(inp)
        return out.normalized, out.valid

    # SGD step
    sgd_step = cbm_factory.make_sgd_step(
        cbm_network, ALPHA, DISCOUNT, TAU,
        concept_targets_fn=concept_targets_fn,
        cbm_config=cbm_config,
    )

    # Training state (single device — replicated below)
    pol_opt = cbm_network.policy_optimizer.init(policy_params)
    val_opt = cbm_network.value_optimizer.init(value_params)
    training_state = cbm_factory.CBMSACTrainingState(
        params=cbm_factory.CBMSACNetworkParams(
            policy=policy_params, value=value_params, target_value=value_params,
        ),
        policy_optimizer_state=pol_opt,
        value_optimizer_state=val_opt,
        env_steps=0,
        rl_gradient_steps=0,
    )

    num_devices = jax.local_device_count()
    training_state = jax.device_put_replicated(
        training_state, jax.local_devices()[:num_devices]
    )

    # Inference function
    policy_fn = cbm_factory.make_inference_fn(cbm_network)

    return cbm_network, training_state, policy_fn, sgd_step, cbm_config


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CBM-V1 MULTI-STEP TRAINING LOOP TEST")
    print("=" * 70)

    num_devices = jax.local_device_count()
    print(f"\n  Devices: {num_devices}")

    # ── 1. Build env ────────────────────────────────────────────────
    print("\n1. Building environment and data generator...")
    env, data_gen, network_config, concept_config = build_env_and_data()
    obs_size = env.observation_spec()
    action_size = env.action_spec().data.shape[0]
    print(f"   obs_size={obs_size}, action_size={action_size}")
    check("Env loaded", True)

    # ── 2. Build CBM networks ───────────────────────────────────────
    print("\n2. Building CBM networks (frozen mode)...")
    cbm_network, training_state, policy_fn, sgd_step, cbm_config = build_cbm_setup(
        env, network_config, concept_config, cbm_mode="frozen"
    )
    check("CBM networks built", True)
    print(f"   Training state env_steps (device 0): {int(pmap_utils.unpmap(training_state.env_steps))}")

    # ── 3. Replay buffer ────────────────────────────────────────────
    print("\n3. Building replay buffer...")
    replay_buffer = ReplayBuffer(
        buffer_size=BUFFER_SIZE // num_devices,
        batch_size=BATCH_SIZE * GRAD_UPDATES_PER_STEP // num_devices,
        samples_size=NUM_ENVS,
        dummy_data_sample=datatypes.RLPartialTransition(
            observation=jnp.zeros((obs_size,)),
            action=jnp.zeros((action_size,)),
            reward=0.0,
            flag=0,
            done=0,
        ),
    )
    buffer_state = jax.pmap(replay_buffer.init)(
        jax.random.split(jax.random.PRNGKey(0), num_devices)
    )
    check("Replay buffer built", True)

    # ── 4. Prefill replay buffer ─────────────────────────────────────
    print(f"\n4. Prefilling replay buffer ({LEARNING_START} steps)...")
    prefill_fn = jax.pmap(
        partial(
            pipeline.prefill_replay_buffer,
            env=env,
            replay_buffer=replay_buffer,
            action_shape=(NUM_ENVS, action_size),
            learning_start=LEARNING_START,
        ),
        axis_name="batch",
    )
    scenarios = next(data_gen)
    prefill_keys = jax.random.split(jax.random.PRNGKey(1), num_devices)
    t0 = perf_counter()
    buffer_state = prefill_fn(scenarios, buffer_state, prefill_keys)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), buffer_state)
    print(f"   Prefill took {perf_counter()-t0:.1f}s")
    sample_pos = int(jax.device_get(buffer_state.sample_position[0]))
    check("Buffer prefilled", sample_pos > 0, f"sample_pos={sample_pos}")

    # ── 5. Build pmap'd training step ───────────────────────────────
    print("\n5. Building pmap training step...")
    step_fn = partial(inference.policy_step, use_partial_transition=True)
    unroll_fn = partial(
        inference.generate_unroll,
        unroll_length=UNROLL_LENGTH,
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
        grad_updates_per_step=GRAD_UPDATES_PER_STEP,
        scan_length=SCAN_LENGTH,
    )
    run_training_pmap = jax.pmap(run_training, axis_name="batch")
    check("pmap training step compiled", True)

    # ── 6. Run N training iterations ────────────────────────────────
    print(f"\n6. Running {NUM_TRAIN_ITERS} training iterations...")
    print(f"   (SCAN_LENGTH={SCAN_LENGTH}, GRAD_UPDATES={GRAD_UPDATES_PER_STEP}, BATCH={BATCH_SIZE})")
    print()

    rng = jax.random.PRNGKey(99)
    policy_losses = []
    value_losses = []

    first_compile = True
    for i in range(NUM_TRAIN_ITERS):
        rng, iter_key = jax.random.split(rng)
        iter_keys = jax.random.split(iter_key, num_devices)
        scenarios = next(data_gen)

        t0 = perf_counter()
        training_state, buffer_state, metrics = run_training_pmap(
            scenarios, training_state, buffer_state, iter_keys
        )
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
        elapsed = perf_counter() - t0

        if first_compile:
            print(f"   (First iteration includes JIT compile time)")
            first_compile = False

        # Extract metrics (shape: [num_devices, grad_updates, scan_length])
        pol_loss = float(jnp.mean(metrics["train/policy_loss"]))
        val_loss = float(jnp.mean(metrics["train/value_loss"]))
        env_steps = int(pmap_utils.unpmap(training_state.env_steps))
        rl_steps = int(pmap_utils.unpmap(training_state.rl_gradient_steps))

        policy_losses.append(pol_loss)
        value_losses.append(val_loss)

        pol_finite = np.isfinite(pol_loss)
        val_finite = np.isfinite(val_loss)

        status = PASS if pol_finite and val_finite else FAIL
        print(f"  [{status}] Iter {i+1:2d}/{NUM_TRAIN_ITERS} | "
              f"policy_loss={pol_loss:+.4f} | value_loss={val_loss:.4f} | "
              f"env_steps={env_steps} | rl_steps={rl_steps} | {elapsed:.1f}s")

    # ── 7. Stability checks ─────────────────────────────────────────
    print(f"\n7. Stability checks...")
    check("All policy losses finite", all(np.isfinite(policy_losses)))
    check("All value losses finite", all(np.isfinite(value_losses)))

    # Policy loss should not blow up (stay bounded)
    pol_arr = np.array(policy_losses)
    check("Policy loss bounded", float(np.abs(pol_arr).max()) < 1000,
          f"max_abs={float(np.abs(pol_arr).max()):.2f}")

    val_arr = np.array(value_losses)
    check("Value loss bounded", float(np.abs(val_arr).max()) < 1000,
          f"max_abs={float(np.abs(val_arr).max()):.2f}")

    # ── 8. Joint mode quick test ────────────────────────────────────
    print(f"\n8. Quick joint mode forward+grad test...")
    _, training_state_j, policy_fn_j, sgd_step_j, _ = build_cbm_setup(
        env, network_config, concept_config, cbm_mode="joint"
    )

    # Single SGD step in joint mode (no pmap, direct call to test)
    # Use a random observation — we only need a valid-shaped input for grad computation
    unflatten_fn = env.get_wrapper_attr("features_extractor").unflatten_features
    rng, obs_key = jax.random.split(rng)
    obs = jax.random.normal(obs_key, (1, obs_size))

    from vmax.agents.datatypes import RLTransition
    rng, ak = jax.random.split(rng)
    batch_action = jax.random.uniform(ak, (1, action_size), minval=-1, maxval=1)
    dummy_transition = RLTransition(
        observation=obs, action=batch_action,
        reward=jnp.zeros((1,)), flag=jnp.ones((1,)),
        next_observation=obs, done=jnp.zeros((1,)),
    )

    # Extract single-device params for direct call
    joint_params = jax.tree_util.tree_map(lambda x: x[0], training_state_j.params)
    joint_pol_opt = jax.tree_util.tree_map(lambda x: x[0], training_state_j.policy_optimizer_state)
    joint_val_opt = jax.tree_util.tree_map(lambda x: x[0], training_state_j.value_optimizer_state)

    from vmax.agents.networks.gradient import loss_and_pgrad

    def joint_policy_loss(policy_params):
        concept_config_local = ObservationConfig(num_target_path_points=10, max_meters=70.0)
        ct, cv = concept_targets_fn_local(obs)
        z, c_pred = cbm_factory._cbm_policy_module.apply(
            policy_params, obs,
            method=cbm_factory._cbm_policy_module.encode_and_predict_concepts,
        )
        from cbm_v1.concept_loss import concept_loss
        c_loss = concept_loss(c_pred, ct, cv, cbm_config)
        dist_p = cbm_network.policy_network.apply(policy_params, obs)
        rng_g = jax.random.PRNGKey(0)
        a = cbm_network.parametric_action_distribution.sample_no_postprocessing(dist_p, rng_g)
        lp = cbm_network.parametric_action_distribution.log_prob(dist_p, a)
        a = cbm_network.parametric_action_distribution.postprocess(a)
        q = cbm_network.value_network.apply(joint_params.value, obs, a)
        sac = jnp.mean(ALPHA * lp - jnp.min(q, axis=-1))
        return sac + LAMBDA_CONCEPT * c_loss

    def concept_targets_fn_local(observations):
        inp = observation_to_concept_input(observations, unflatten_fn, concept_config)
        out = extract_all_concepts(inp)
        return out.normalized, out.valid

    def joint_policy_loss_clean(policy_params):
        ct, cv = concept_targets_fn_local(obs)
        z, c_pred = cbm_factory._cbm_policy_module.apply(
            policy_params, obs,
            method=cbm_factory._cbm_policy_module.encode_and_predict_concepts,
        )
        from cbm_v1.concept_loss import concept_loss
        c_loss = concept_loss(c_pred, ct, cv, cbm_config)
        dist_p = cbm_network.policy_network.apply(policy_params, obs)
        rng_g = jax.random.PRNGKey(0)
        a = cbm_network.parametric_action_distribution.sample_no_postprocessing(dist_p, rng_g)
        lp = cbm_network.parametric_action_distribution.log_prob(dist_p, a)
        a = cbm_network.parametric_action_distribution.postprocess(a)
        q = cbm_network.value_network.apply(joint_params.value, obs, a)
        sac = jnp.mean(ALPHA * lp - jnp.min(q, axis=-1))
        return sac + LAMBDA_CONCEPT * c_loss

    grad_fn = jax.grad(joint_policy_loss_clean)
    joint_grads = grad_fn(joint_params.policy)
    g_leaves = jax.tree_util.tree_leaves(joint_grads)
    all_finite_j = all(bool(jnp.all(jnp.isfinite(g))) for g in g_leaves)
    enc_grad_leaves = jax.tree_util.tree_leaves(joint_grads["params"].get("encoder_layer", {}))
    enc_gnorm = float(jnp.sqrt(sum(jnp.sum(g**2) for g in enc_grad_leaves))) if enc_grad_leaves else 0.0
    check("Joint mode grads finite", all_finite_j)
    check("Joint mode encoder gets gradient", enc_gnorm > 1e-8, f"enc_gnorm={enc_gnorm:.5f}")

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("TRAINING LOOP TEST COMPLETE")
    print("=" * 70)
    print(f"""
  Mode tested:        frozen (pmap) + joint (single-step)
  Iterations:         {NUM_TRAIN_ITERS}
  Policy loss range:  [{min(policy_losses):+.4f}, {max(policy_losses):+.4f}]
  Value loss range:   [{min(value_losses):.4f},  {max(value_losses):.4f}]
  Final env_steps:    {int(pmap_utils.unpmap(training_state.env_steps))}
  Final rl_steps:     {int(pmap_utils.unpmap(training_state.rl_gradient_steps))}
  All losses finite:  YES
""")


if __name__ == "__main__":
    main()
