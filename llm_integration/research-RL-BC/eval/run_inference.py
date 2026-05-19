"""Quick inference: load a model, run one scenario, print metrics."""
import os, sys
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Correct the Python path to point to the repository root
# Script is in research-RL-BC/eval/, so we go up 3 levels
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import yaml, jax, flax
import numpy as np
from tqdm import tqdm
from functools import partial
from waymax import dynamics
from vmax.simulator import make_env_for_evaluation, make_data_generator
from vmax.simulator.metrics.aggregators import vmax_aggregate_score, nuplan_aggregate_score
from vmax.scripts.evaluate import utils
from vmax.agents import pipeline

import argparse

# === Config ===
parser = argparse.ArgumentParser(description="Quick inference: load a model, run scenarios, print metrics.")
parser.add_argument("--model", type=str, default="runs_rlc/womd_sac_road_perceiver_minimal_42", help="Path to the model directory")
parser.add_argument("--dataset", type=str, default="data/training.tfrecord", help="Path to the dataset")
parser.add_argument("--num_scenarios", "-n", type=int, default=1, help="Number of scenarios to run")
args = parser.parse_args()

MODEL_DIR = args.model
DATA_PATH = args.dataset

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

# === Print Model Info ===
print("\n" + "="*50)
print("      MODEL CONFIGURATION")
print("="*50)
algo_name = config.get("algorithm", {}).get("name", "N/A")
print(f"Algorithm:       {algo_name}")
print(f"Learning Rate:   {config.get('algorithm', {}).get('learning_rate', 'N/A')}")
print(f"Observation:     {config.get('observation_type', 'N/A')}")
num_closest_objects = config.get("observation_config", {}).get("objects", {}).get("num_closest_objects", "N/A")
print(f"Num Closest Objects:     {num_closest_objects}")
policy_layers = config.get("algorithm", {}).get("network", {}).get("policy", {}).get("layer_sizes", "N/A")
print(f"Policy Layers:   {policy_layers}")

encoder_cfg = config.get("network", {}).get("encoder", {})
print("-" * 50)
print("      ENCODER INFO")
print(f"Type:            {encoder_cfg.get('type', 'N/A')}")
if encoder_cfg.get("type") in ["lq", "perceiver"]:
    print(f"Depth:           {encoder_cfg.get('encoder_depth', 'N/A')}")
    print(f"Num Latents:     {encoder_cfg.get('num_latents', 'N/A')}")
    print(f"Latent Heads:    {encoder_cfg.get('latent_num_heads', 'N/A')}")
    print(f"Dk:              {encoder_cfg.get('dk', 'N/A')}")
elif encoder_cfg.get("type") == "mlp":
    print(f"Layer Sizes:     {encoder_cfg.get('layer_sizes', 'N/A')}")
print("="*50 + "\n")

# 2. Build eval config
eval_config = dict(config)
eval_config["encoder"] = config.get("network", {}).get("encoder", {})
eval_config["policy"] = config.get("algorithm", {}).get("network", {}).get("policy", {})
eval_config["value"] = config.get("algorithm", {}).get("network", {}).get("value", {})
eval_config["unflatten_config"] = config.get("observation_config", {})
eval_config["action_distribution"] = config.get("algorithm", {}).get("network", {}).get("action_distribution", "gaussian")

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
make_inference_fn, build_network = utils.get_algorithm_modules(algo_name)

# For bc_sac, it needs two learning rates, others need one.
extra_kwargs = {}
if algo_name.lower() == "bc_sac":
    extra_kwargs["rl_learning_rate"] = eval_config["algorithm"]["rl_learning_rate"]
    extra_kwargs["imitation_learning_rate"] = eval_config["algorithm"]["imitation_learning_rate"]
else:
    extra_kwargs["learning_rate"] = eval_config["algorithm"]["learning_rate"]

network = build_network(
    observation_size=env.observation_spec(),
    action_size=env.action_spec().data.shape[0],
    unflatten_fn=env.get_wrapper_attr("features_extractor").unflatten_features,
    network_config=eval_config,
    **extra_kwargs
)
make_policy = make_inference_fn(network)

# 5. Load params & fix keys
training_state = utils.load_params(f"{MODEL_DIR}/model/model_final.pkl")

# Extract policy params based on algorithm structure
if hasattr(training_state, "params") and hasattr(training_state.params, "policy"):
    policy_params = training_state.params.policy
elif hasattr(training_state, "policy"):
    policy_params = training_state.policy
else:
    # Fallback/Older versions
    policy_params = training_state

for old_key, new_key in PARAM_KEY_REMAP.items():
    for path, _ in jax.tree_util.tree_leaves_with_path(policy_params):
        if any(old_key in str(p) for p in path):
            print(f"[FIX] Param key: {old_key} -> {new_key}")
            policy_params = remap_param_keys(policy_params, old_key, new_key)
            break

policy_fn = make_policy(policy_params, deterministic=True)

# 6. Load data & run episodes
data_gen = make_data_generator(
    path=DATA_PATH,
    max_num_objects=config.get("max_num_objects", 64),
    include_sdc_paths=True,
    batch_dims=(1,),
    seed=42,
    repeat=1,
)

rng_key = jax.random.PRNGKey(0)
step_fn = partial(pipeline.policy_step, env=env, policy_fn=policy_fn)
jitted_step_fn = jax.jit(step_fn)
jitted_reset = jax.jit(env.reset)

# Dictionary to accumulate all metrics across scenarios
eval_metrics = {"episode_length": [], "accuracy": []}
termination_keys = config["termination_keys"]

print(f"Running inference on {args.num_scenarios} scenarios...")
for i, scenario in enumerate(tqdm(data_gen, total=args.num_scenarios)):
    if i >= args.num_scenarios:
        break
        
    rng_key, scenario_key = jax.random.split(rng_key)
    
    # Run scenario using utility function (JIT loop)
    # This collects metrics for every step into an array
    episode_metrics, steps_done = utils.run_scenario_jit(
        scenario, 
        scenario_key, 
        step_fn=jitted_step_fn, 
        reset_fn=jitted_reset
    )
    
    # Process and aggregate metrics for this scenario
    # Ensure steps_done and episode_metrics have batch dimensions (batch, ...)
    if steps_done.ndim == 1:
        steps_done = steps_done[:, np.newaxis]
    
    for k in episode_metrics:
        if episode_metrics[k].ndim == 1:
            episode_metrics[k] = episode_metrics[k][np.newaxis, :]
        
    eval_metrics = utils.append_episode_metrics(
        steps_done,
        eval_metrics,
        episode_metrics,
        termination_keys,
        batch_size=1
    )

# 7. Print aggregate results
print("\n" + "="*50)
print("            EVALUATION SUMMARY")
print("="*50)
print(f"Scenarios:       {len(eval_metrics['accuracy'])}")

# Calculate and print means for all metrics
for k, v in eval_metrics.items():
    if k in ['accuracy', 'vmax_aggregate_score', 'nuplan_aggregate_score']:
        continue
    print(f"Mean {k:25}: {np.mean(v):.4f}")

print("-" * 50)
print(f"MEAN ACCURACY:             {np.mean(eval_metrics['accuracy']):.4f}")
print(f"V-MAX AGGREGATE SCORE:     {np.mean(eval_metrics['vmax_aggregate_score']):.4f}")
print(f"NUPLAN AGGREGATE SCORE:    {np.mean(eval_metrics['nuplan_aggregate_score']):.4f}")
print("="*50 + "\n")
