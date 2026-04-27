"""
V-MAX Model Exploration Script
Load a model, inspect params, run a scenario, extract attention weights.
Run from: /home/med1e/rl-il/RL-IL/vmax/
"""
import os, sys
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Ensure vmax package is importable
VMAX_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vmax")
if VMAX_REPO not in sys.path:
    sys.path.insert(0, VMAX_REPO)

import yaml
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from waymax import dynamics

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs_rlc"
DATA_DIR = BASE_DIR / "data"

MODEL_NAME = "womd_sac_road_perceiver_minimal_42"  # Perceiver encoder, #2 ranked
MODEL_DIR = RUNS_DIR / MODEL_NAME

# ── 1. Load model weights (raw pickle) and inspect parameter tree ──────────

print("=" * 60)
print("STEP 1: Loading raw model weights and inspecting structure")
print("=" * 60)

# Load config
config_path = MODEL_DIR / ".hydra" / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

print(f"\nModel: {MODEL_NAME}")
print(f"Encoder type (config): {config['network']['encoder']['type']}")
print(f"Observation type: {config['observation_type']}")
print(f"Algorithm: {config['algorithm']['name']}")

# Load params using the official loader
model_path = str(MODEL_DIR / "model" / "model_final.pkl")

from vmax.scripts.evaluate.utils import load_params
training_state = load_params(model_path)

print(f"\nTraining state type: {type(training_state).__name__}")
print(f"Training state fields: {list(training_state.__dataclass_fields__.keys())}")

# Inspect the policy params tree
policy_params = training_state.policy

# Flatten to see everything
flat_params = jax.tree_util.tree_leaves_with_path(policy_params)
print(f"\nTotal parameter arrays: {len(flat_params)}")
total_params = sum(np.prod(leaf.shape) for _, leaf in flat_params)
print(f"Total parameters: {total_params:,}")

print("\n── All parameter paths and shapes ──")
for path, leaf in flat_params:
    path_str = "/".join(str(p) for p in path)
    print(f"  {path_str}: {leaf.shape}")

# ── 2. Set up environment, load data, build the policy ─────────────────────

print("\n" + "=" * 60)
print("STEP 2: Setting up environment and loading data")
print("=" * 60)

# Fix the perceiver→lq alias issue
eval_config = dict(config)
eval_config["encoder"] = dict(config["network"]["encoder"])
if eval_config["encoder"]["type"] == "perceiver":
    print("\n[FIX] Remapping encoder type 'perceiver' -> 'lq'")
    eval_config["encoder"]["type"] = "lq"
eval_config["network"] = dict(config["network"])
eval_config["network"]["encoder"] = dict(config["network"]["encoder"])
eval_config["network"]["encoder"]["type"] = eval_config["encoder"]["type"]
eval_config["policy"] = config["algorithm"]["network"]["policy"]
eval_config["value"] = config["algorithm"]["network"]["value"]
eval_config["unflatten_config"] = config["observation_config"]
eval_config["action_distribution"] = config["algorithm"]["network"]["action_distribution"]

from vmax.simulator import make_env_for_evaluation, make_data_generator

# Fix observation type alias: 'road' and 'lane' were renamed to 'vec'
obs_type = config["observation_type"]
if obs_type in ("road", "lane"):
    print(f"[FIX] Remapping observation type '{obs_type}' -> 'vec'")
    obs_type = "vec"

# Create environment
env = make_env_for_evaluation(
    max_num_objects=config.get("max_num_objects", 64),
    dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
    sdc_paths_from_data=True,
    observation_type=obs_type,
    observation_config=config["observation_config"],
    termination_keys=config["termination_keys"],
    noisy_init=False,
)
print(f"Environment created")
print(f"  obs_spec: {env.observation_spec()}")
print(f"  action_spec shape: {env.action_spec().data.shape}")

# Build the model manually to handle param key remapping
from vmax.agents.learning.reinforcement.sac.sac_factory import make_inference_fn, make_networks
from vmax.scripts.evaluate.utils import load_params as _load_params

obs_size = env.observation_spec()
action_size = env.action_spec().data.shape[0]
unflatten_fn = env.get_wrapper_attr("features_extractor").unflatten_features

network = make_networks(
    observation_size=obs_size,
    action_size=action_size,
    unflatten_fn=unflatten_fn,
    learning_rate=eval_config["algorithm"]["learning_rate"],
    network_config=eval_config,
)
make_policy = make_inference_fn(network)

# Load params and remap 'perceiver_attention' -> 'lq_attention' if needed
loaded_state = _load_params(model_path)
raw_policy_params = loaded_state.policy


def remap_param_keys(params, old_name, new_name):
    """Recursively rename keys in a nested param dict."""
    if isinstance(params, dict):
        return {
            (new_name if k == old_name else k): remap_param_keys(v, old_name, new_name)
            for k, v in params.items()
        }
    return params


# Check if we need remapping
needs_remap = False
flat_check = jax.tree_util.tree_leaves_with_path(raw_policy_params)
for path, _ in flat_check:
    if any("perceiver_attention" in str(p) for p in path):
        needs_remap = True
        break

if needs_remap:
    print("[FIX] Remapping param key 'perceiver_attention' -> 'lq_attention'")
    policy_params = remap_param_keys(raw_policy_params, "perceiver_attention", "lq_attention")
    # Also need to wrap in the FrozenDict structure Flax expects
    import flax
    policy_params = flax.core.freeze(policy_params) if isinstance(policy_params, dict) else policy_params
else:
    policy_params = raw_policy_params

policy_fn = make_policy(policy_params, deterministic=True)
print("Policy function loaded successfully!")

# Load data
DATA_PATH = str(DATA_DIR / "training.tfrecord")
print(f"\nLoading data from: {DATA_PATH}")
data_gen = make_data_generator(
    path=DATA_PATH,
    max_num_objects=config.get("max_num_objects", 64),
    include_sdc_paths=True,
    batch_dims=(1,),
    seed=42,
    repeat=1,
)

# Get first scenario
scenario = next(iter(data_gen))
print(f"Scenario loaded!")

# ── 3. Run a single scenario ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3: Running inference on a single scenario")
print("=" * 60)

from functools import partial
from vmax.agents import pipeline

# Reset environment
rng_key = jax.random.PRNGKey(0)
rng_key, reset_key = jax.random.split(rng_key)
reset_key = jax.random.split(reset_key, 1)
env_transition = jax.jit(env.reset)(scenario, reset_key)

print(f"Environment reset.")
print(f"  Done: {env_transition.done}")
print(f"  Observation shape: {env_transition.observation.shape}")
print(f"  Metrics keys: {list(env_transition.metrics.keys())}")

# Take one step
step_fn = partial(pipeline.policy_step, env=env, policy_fn=policy_fn)
rng_key, step_key = jax.random.split(rng_key)
step_key = jax.random.split(step_key, 1)
env_transition, transition = step_fn(env_transition, key=step_key)

print(f"\nAfter 1 step:")
print(f"  Done: {env_transition.done}")
print(f"  Reward: {env_transition.reward}")
print(f"  Action shape: {transition.action.shape}")
print(f"  Action values: {transition.action}")

# Run full episode
print("\nRunning full episode...")
step_count = 1
while not bool(env_transition.done):
    rng_key, step_key = jax.random.split(rng_key)
    step_key = jax.random.split(step_key, 1)
    env_transition, transition = step_fn(env_transition, key=step_key)
    step_count += 1

print(f"  Episode done in {step_count} steps")
print(f"  Final metrics:")
for k, v in env_transition.metrics.items():
    print(f"    {k}: {float(v[0]):.4f}")

# ── 4. Inspect attention weights from parameters ─────────────────────────

print("\n" + "=" * 60)
print("STEP 4: Extracting attention-related parameters")
print("=" * 60)

print("\nSearching for attention-related parameters...")
attention_params = {}
for path, leaf in flat_params:
    path_str = "/".join(str(p) for p in path)
    if any(kw in path_str.lower() for kw in ["attn", "cross", "self_attn", "latent", "rezero"]):
        attention_params[path_str] = leaf

print(f"Found {len(attention_params)} attention-related parameter arrays:")
for name, param in sorted(attention_params.items()):
    print(f"  {name}: {param.shape}")

# Extract learned latent vectors
for path, leaf in flat_params:
    path_str = "/".join(str(p) for p in path)
    if "latents" in path_str.lower() and leaf.ndim >= 2:
        print(f"\nLearned latent vectors: {path_str}")
        print(f"  Shape: {leaf.shape} (num_latents x latent_dim)")
        print(f"  Mean: {float(leaf.mean()):.4f}, Std: {float(leaf.std()):.4f}")
        print(f"  Min: {float(leaf.min()):.4f}, Max: {float(leaf.max()):.4f}")

# ── 5. Forward pass with intermediate extraction ─────────────────────────

print("\n" + "=" * 60)
print("STEP 5: Forward pass with capture_intermediates")
print("=" * 60)

from vmax.agents.networks import encoders, network_utils, decoders
from vmax.agents.networks.network_factory import PolicyNetwork

# Rebuild the policy network module
_config = network_utils.convert_to_dict_with_activation_fn(eval_config)
unflatten_fn = env.get_wrapper_attr("features_extractor").unflatten_features

encoder_config = dict(_config["encoder"])
encoder_type = encoder_config.pop("type")
encoder_layer = encoders.get_encoder(encoder_type)(unflatten_fn, **encoder_config)

policy_config = _config.get("policy")
fc_config = {k: v for k, v in policy_config.items()
             if k not in ("type", "final_activation", "num_networks", "shared_encoder")}
fc_layer = decoders.get_fully_connected(policy_config["type"])(**fc_config)

action_size = env.action_spec().data.shape[0]
output_size = action_size * 2  # mean + log_std for NormalTanh

policy_module = PolicyNetwork(
    encoder_layer=encoder_layer,
    fully_connected_layer=fc_layer,
    output_size=output_size,
    final_activation=policy_config["final_activation"],
)

# Reset and get a fresh observation
rng_key = jax.random.PRNGKey(42)
rng_key, reset_key = jax.random.split(rng_key)
reset_key = jax.random.split(reset_key, 1)
env_transition = jax.jit(env.reset)(scenario, reset_key)
obs = env_transition.observation  # shape (1, obs_dim)
print(f"Observation for forward pass: shape={obs.shape}")

# Apply with capture_intermediates - capture everything
output, state = policy_module.apply(
    policy_params,
    obs,
    capture_intermediates=True,
    mutable=["intermediates"],
)

print(f"Forward pass output shape: {output.shape}")
print(f"Output (first 6 action logits): {output[0, :6]}")

# Inspect captured intermediates
intermediates = state.get("intermediates", {})


def print_intermediates(d, prefix="", depth=0):
    if depth > 6:
        return
    if isinstance(d, dict):
        for k, v in d.items():
            print_intermediates(v, f"{prefix}/{k}", depth + 1)
    elif isinstance(d, (list, tuple)):
        for i, item in enumerate(d):
            if hasattr(item, "shape"):
                print(f"  {prefix}[{i}]: shape={item.shape}")
            else:
                print_intermediates(item, f"{prefix}[{i}]", depth + 1)
    elif hasattr(d, "shape"):
        print(f"  {prefix}: shape={d.shape}")
    else:
        print(f"  {prefix}: {type(d).__name__}")


print(f"\nCaptured intermediates:")
print_intermediates(intermediates)

# ── 6. Compute attention maps manually ───────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6: Computing attention maps from the encoder")
print("=" * 60)

# The LQEncoder:
# 1. Unflattens obs -> features (sdc_traj, other_traj, roadgraph, traffic_lights, gps_path) + masks
# 2. Embeds each feature type through MLPs
# 3. Adds positional encodings
# 4. Concatenates all embeddings
# 5. Passes through LQAttention (cross-attn from learned latents to input, then self-attn)

# We can manually compute the attention by:
# - Running the encoder forward pass up to the attention computation
# - Then computing softmax(Q*K^T / sqrt(dk)) ourselves

# First, let's unflatten the observation to see the input structure
features, masks = unflatten_fn(obs)
sdc_traj_feat, other_traj_feat, rg_feat, tl_feat, gps_path_feat = features
sdc_mask, other_mask, rg_mask, tl_mask = masks

print("Input feature shapes:")
print(f"  SDC trajectory:    {sdsc_traj_feat.shape}, mask: {sdc_mask.shape}")
print(f"  Other agents:      {other_traj_feat.shape}, mask: {other_mask.shape}")
print(f"  Roadgraph:         {rg_feat.shape}, mask: {rg_mask.shape}")
print(f"  Traffic lights:    {tl_feat.shape}, mask: {tl_mask.shape}")
print(f"  GPS path:          {gps_path_feat.shape}")

# Count valid elements per category
print(f"\nValid elements (first scenario):")
print(f"  SDC traj steps:    {int(sdc_mask[0].sum())}/{sdc_mask[0].size}")
print(f"  Other agents:      {int(other_mask[0].sum())}/{other_mask[0].size}")
print(f"  Roadgraph points:  {int(rg_mask[0].sum())}/{rg_mask[0].size}")
print(f"  Traffic lights:    {int(tl_mask[0].sum())}/{tl_mask[0].size}")

print("\n" + "=" * 60)
print("EXPLORATION COMPLETE!")
print("=" * 60)
print("""
What we confirmed:
1. Model loading works (with perceiver->lq alias fix)
2. Environment setup + data loading works
3. Inference runs - the policy produces actions and drives the car
4. Full parameter tree is accessible (attention Q/K/V weights, latents, etc.)
5. capture_intermediates reveals internal activations
6. Observations can be unflattened to see per-category features

For attention visualization, the key approach is:
- Use capture_intermediates on the AttentionLayer to get post-softmax attention scores
- OR manually compute attention from the Q/K/V projection weights
- Map attention weights back to input categories (sdc, agents, roads, TLs, GPS)
""")
