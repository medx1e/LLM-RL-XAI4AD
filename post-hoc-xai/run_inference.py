"""Quick inference: load a model, run one scenario, print metrics."""
import os, sys
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

VMAX_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vmax")
if VMAX_REPO not in sys.path:
    sys.path.insert(0, VMAX_REPO)

import yaml, jax, flax
from functools import partial
from waymax import dynamics
from vmax.simulator import make_env_for_evaluation, make_data_generator
from vmax.agents.learning.reinforcement.sac.sac_factory import make_inference_fn, make_networks
from vmax.scripts.evaluate.utils import load_params
from vmax.agents import pipeline

# === Config ===
MODEL_DIR = "runs_rlc/womd_sac_road_wayformer_minimal_42"
DATA_PATH = "data/training.tfrecord"

# === Remapping tables ===
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


# 1. Load & fix config
with open(f"{MODEL_DIR}/.hydra/config.yaml") as f:
    config = yaml.safe_load(f)

enc_type = config["network"]["encoder"]["type"]
if enc_type in ENCODER_REMAP:
    print(f"[FIX] Encoder: {enc_type} -> {ENCODER_REMAP[enc_type]}")
    config["network"]["encoder"]["type"] = ENCODER_REMAP[enc_type]

obs_type = OBS_TYPE_REMAP.get(config["observation_type"], config["observation_type"])
if obs_type != config["observation_type"]:
    print(f"[FIX] Obs type: {config['observation_type']} -> {obs_type}")

# 2. Build eval config
eval_config = dict(config)
eval_config["encoder"] = config["network"]["encoder"]
eval_config["policy"] = config["algorithm"]["network"]["policy"]
eval_config["value"] = config["algorithm"]["network"]["value"]
eval_config["unflatten_config"] = config["observation_config"]
eval_config["action_distribution"] = config["algorithm"]["network"]["action_distribution"]

# 3. Create environment
env = make_env_for_evaluation(
    max_num_objects=config.get("max_num_objects", 64),
    dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
    sdc_paths_from_data=True,
    observation_type=obs_type,
    observation_config=config["observation_config"],
    termination_keys=config["termination_keys"],
    noisy_init=False,
)

# 4. Build network
network = make_networks(
    observation_size=env.observation_spec(),
    action_size=env.action_spec().data.shape[0],
    unflatten_fn=env.get_wrapper_attr("features_extractor").unflatten_features,
    learning_rate=eval_config["algorithm"]["learning_rate"],
    network_config=eval_config,
)
make_policy = make_inference_fn(network)

# 5. Load params & fix keys
training_state = load_params(f"{MODEL_DIR}/model/model_final.pkl")
policy_params = training_state.policy

for old_key, new_key in PARAM_KEY_REMAP.items():
    for path, _ in jax.tree_util.tree_leaves_with_path(policy_params):
        if any(old_key in str(p) for p in path):
            print(f"[FIX] Param key: {old_key} -> {new_key}")
            policy_params = remap_param_keys(policy_params, old_key, new_key)
            break

policy_fn = make_policy(policy_params, deterministic=True)

# 6. Load data & run episode
data_gen = make_data_generator(
    path=DATA_PATH,
    max_num_objects=config.get("max_num_objects", 64),
    include_sdc_paths=True,
    batch_dims=(1,),
    seed=42,
    repeat=1,
)
scenario = next(iter(data_gen))

rng_key = jax.random.PRNGKey(0)
rng_key, reset_key = jax.random.split(rng_key)
env_transition = jax.jit(env.reset)(scenario, jax.random.split(reset_key, 1))

step_fn = partial(pipeline.policy_step, env=env, policy_fn=policy_fn)
steps = 0
while not bool(env_transition.done):
    rng_key, step_key = jax.random.split(rng_key)
    env_transition, transition = step_fn(env_transition, key=jax.random.split(step_key, 1))
    steps += 1

print(f"\nEpisode done in {steps} steps")
print("Metrics:")
for k, v in env_transition.metrics.items():
    print(f"  {k}: {float(v[0]):.4f}")
