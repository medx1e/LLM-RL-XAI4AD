# How to Load and Use V-MAX Pretrained Models

This guide documents all the compatibility issues, required fixes, and working commands to load the V-MAX pretrained model weights from `runs_rlc/` using the current vmax codebase.

---

## Prerequisites

```bash
conda activate vmax

# Required: install missing dependency (pickle deserialization needs it)
pip install tensorboardX
```

Ensure `vmax/` repo is importable. Either:
- Run scripts from inside the `vmax/` directory, OR
- Add it to your Python path: `sys.path.insert(0, "/path/to/RL-IL/vmax")`

---

## Known Compatibility Issues

The pretrained weights were saved with an older version of the codebase. There are **5 breaking issues** you must handle:

### Issue 1: Pickle Module Path Mismatch

**Symptom:** `ModuleNotFoundError: No module named 'vmax.learning'`

**Cause:** The `.pkl` files reference `vmax.learning.algorithms.rl.sac.sac_factory.SACNetworkParams` but the current code moved this to `vmax.agents.learning.reinforcement.sac.sac_factory`.

**Fix:** The official `load_params()` in `vmax/vmax/scripts/evaluate/utils.py` already handles this with `ModuleCompatUnpickler`. However, this only works after installing `tensorboardX` (see Prerequisites). Always use this function:

```python
from vmax.scripts.evaluate.utils import load_params
training_state = load_params("runs_rlc/<model>/model/model_final.pkl")
# training_state has fields: policy, value, target_value
```

### Issue 2: Encoder Type Aliases

**Symptom:** `ValueError: Unknown encoder: perceiver` or `ValueError: Unknown encoder: mgail`

**Cause:** Two encoder types were renamed:

| Config says | Registry has | Fix |
|-------------|-------------|-----|
| `perceiver` | `lq` | Remap to `lq` |
| `mgail` | `lqh` | Remap to `lqh` |

**Affected models:**
- **perceiver -> lq**: All `*_perceiver_*` models (21 models) + `womd_sac_lane_*` models
- **mgail -> lqh**: All `*_mgail_*` models (3 models)

**Fix:** After loading the config, remap the encoder type:

```python
import yaml

with open(f"runs_rlc/{model_name}/.hydra/config.yaml") as f:
    config = yaml.safe_load(f)

encoder_type = config["network"]["encoder"]["type"]

ENCODER_REMAP = {"perceiver": "lq", "mgail": "lqh"}
if encoder_type in ENCODER_REMAP:
    config["network"]["encoder"]["type"] = ENCODER_REMAP[encoder_type]
```

### Issue 3: Observation Type Alias

**Symptom:** `ValueError: Unknown feature extractor: road` (or `lane`)

**Cause:** The observation types `road` and `lane` were renamed to `vec`.

| Config says | Code expects |
|-------------|-------------|
| `road` | `vec` |
| `lane` | `vec` |
| `vec` | `vec` (OK) |

**Affected models:** All models except `sac_seed0/42/69` (which already use `vec`).

**Fix:**
```python
obs_type = config["observation_type"]
OBS_TYPE_REMAP = {"road": "vec", "lane": "vec"}
obs_type = OBS_TYPE_REMAP.get(obs_type, obs_type)
```

### Issue 4: Parameter Key Mismatch (Attention Module Names)

**Symptom:** `flax.errors.ScopeParamNotFoundError: Could not find parameter named "latents" in scope "/encoder_layer/lq_attention"`

**Cause:** The saved weights use old Flax module names that don't match the current code:

| Saved param key | Code expects | Affected models |
|-----------------|-------------|-----------------|
| `perceiver_attention` | `lq_attention` | All `*_perceiver_*` models |
| `mgail_attention` | `lq_attention` | All `*_mgail_*` models |

**Fix:** Recursively rename the keys in the loaded parameter dict:

```python
def remap_param_keys(params, old_name, new_name):
    """Recursively rename keys in a nested param dict."""
    if isinstance(params, dict):
        return {
            (new_name if k == old_name else k): remap_param_keys(v, old_name, new_name)
            for k, v in params.items()
        }
    return params

PARAM_KEY_REMAP = {
    "perceiver_attention": "lq_attention",
    "mgail_attention": "lq_attention",
}

training_state = load_params(model_path)
policy_params = training_state.policy

# Check which remapping is needed
import jax
for path, _ in jax.tree_util.tree_leaves_with_path(policy_params):
    path_str = "/".join(str(p) for p in path)
    for old_key, new_key in PARAM_KEY_REMAP.items():
        if old_key in path_str:
            policy_params = remap_param_keys(policy_params, old_key, new_key)
            break
```

### Issue 5: `speed_limit` Feature (sac_seed* models ONLY)

**Symptom:** `AttributeError: 'RoadgraphPoints' object has no attribute 'speed_limit'`

**Cause:** The `sac_seed0`, `sac_seed42`, `sac_seed69` models were trained with `speed_limit` as a roadgraph feature, but Waymax's `RoadgraphPoints` doesn't have this attribute. The `FEATURE_MAP` in the current codebase also doesn't include it.

**Affected models:** ONLY `sac_seed0`, `sac_seed42`, `sac_seed69` (the 3 best-performing LQ models).

**Impact:** These models expect **5 features per roadgraph point** (waypoints=2, direction=2, speed_limit=1) but the current code only provides **4** (waypoints=2, direction=2). The encoder's `rg_enc_layer_0` kernel has shape `(5, 256)` instead of `(4, 256)`, so you **cannot** simply remove `speed_limit` from the config — the weight dimensions won't match.

**Workaround options:**
1. **Add `speed_limit` to the feature extractor** by computing it from lane type (the codebase already has `infer_speed_limit_from_roadgraph()` in `vmax/simulator/metrics/speed_limit.py`). Add to `FEATURE_MAP`:
   ```python
   # In vec_extractor.py FEATURE_MAP:
   "speed_limit": ("speed_limit",)
   ```
   Then add the feature extraction logic in `_build_roadgraph_features`.

2. **Use these models for parameter inspection only** (not live inference). You can still analyze the weights, attention patterns, and do static weight-level XAI.

3. **Use the other 30+ models** that don't have this issue. The `womd_sac_road_perceiver_minimal_42` model (97.47% accuracy, #2 ranked) works perfectly.

---

## Model Compatibility Matrix

| Model Group | Count | obs_type fix | encoder fix | param key fix | speed_limit |
|---|---|---|---|---|---|
| `sac_seed*` | 3 | OK (vec) | OK (lq) | none needed | **BROKEN** |
| `*_perceiver_*` | 21 | road -> vec | perceiver -> lq | perceiver_attention -> lq_attention | OK |
| `*_lane_perceiver_*` | 3 | lane -> vec | perceiver -> lq | perceiver_attention -> lq_attention | OK |
| `*_mgail_*` | 3 | road -> vec | mgail -> lqh | mgail_attention -> lq_attention | OK |
| `*_mtr_*` | 3 | road -> vec | OK (mtr) | none needed | OK |
| `*_wayformer_*` | 3 | road -> vec | OK (wayformer) | none needed | OK |
| `*_none_*` | 2 | road -> vec | OK (none) | none needed | OK |

**Recommended models for getting started** (fewest issues, best accuracy):
1. `womd_sac_road_perceiver_minimal_42` — 97.47% accuracy, Perceiver/LQ encoder
2. `womd_sac_road_perceiver_minimal_69` — 97.44% accuracy
3. `womd_sac_road_mtr_minimal_42` — MTR encoder, no param key fix needed
4. `womd_sac_road_wayformer_minimal_42` — Wayformer encoder, no param key fix needed

---

## Complete Loading Recipe

Here is a copy-paste-ready function that handles all issues:

```python
import os, sys, io, pickle, yaml
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Ensure vmax is importable
sys.path.insert(0, "/path/to/RL-IL/vmax")  # adjust this path

import jax
import jax.numpy as jnp
import flax
from waymax import dynamics
from vmax.simulator import make_env_for_evaluation, make_data_generator
from vmax.agents.learning.reinforcement.sac.sac_factory import make_inference_fn, make_networks
from vmax.scripts.evaluate.utils import load_params


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


def load_vmax_model(model_dir, data_path, max_num_objects=64):
    """
    Load a V-MAX pretrained model with all compatibility fixes applied.

    Args:
        model_dir: Path to model directory (e.g. "runs_rlc/womd_sac_road_perceiver_minimal_42")
        data_path: Path to tfrecord data file (e.g. "data/training.tfrecord")
        max_num_objects: Max objects in scene (default 64)

    Returns:
        dict with keys: policy_fn, env, data_gen, config, policy_params
    """
    # 1. Load config
    with open(f"{model_dir}/.hydra/config.yaml") as f:
        config = yaml.safe_load(f)

    original_encoder_type = config["network"]["encoder"]["type"]

    # 2. Fix encoder type alias
    if original_encoder_type in ENCODER_REMAP:
        print(f"[FIX] Encoder type: {original_encoder_type} -> {ENCODER_REMAP[original_encoder_type]}")
        config["network"]["encoder"]["type"] = ENCODER_REMAP[original_encoder_type]

    # 3. Fix observation type alias
    obs_type = config["observation_type"]
    if obs_type in OBS_TYPE_REMAP:
        print(f"[FIX] Observation type: {obs_type} -> {OBS_TYPE_REMAP[obs_type]}")
        obs_type = OBS_TYPE_REMAP[obs_type]

    # 4. Check for speed_limit issue
    rg_features = config.get("observation_config", {}).get("roadgraphs", {}).get("features", [])
    if "speed_limit" in rg_features:
        raise RuntimeError(
            f"Model '{model_dir}' uses 'speed_limit' roadgraph feature which is not "
            f"supported by Waymax. Use a different model (see GUIDE2LOAD_MODELS.md Issue 5)."
        )

    # 5. Build eval config
    eval_config = dict(config)
    eval_config["encoder"] = config["network"]["encoder"]
    eval_config["policy"] = config["algorithm"]["network"]["policy"]
    eval_config["value"] = config["algorithm"]["network"]["value"]
    eval_config["unflatten_config"] = config["observation_config"]
    eval_config["action_distribution"] = config["algorithm"]["network"]["action_distribution"]

    # 6. Create environment
    env = make_env_for_evaluation(
        max_num_objects=max_num_objects,
        dynamics_model=dynamics.InvertibleBicycleModel(normalize_actions=True),
        sdc_paths_from_data=True,
        observation_type=obs_type,
        observation_config=config["observation_config"],
        termination_keys=config["termination_keys"],
        noisy_init=False,
    )

    # 7. Build network
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

    # 8. Load and fix parameter keys
    model_path = f"{model_dir}/model/model_final.pkl"
    training_state = load_params(model_path)
    policy_params = training_state.policy

    for old_key, new_key in PARAM_KEY_REMAP.items():
        needs_remap = False
        for path, _ in jax.tree_util.tree_leaves_with_path(policy_params):
            if any(old_key in str(p) for p in path):
                needs_remap = True
                break
        if needs_remap:
            print(f"[FIX] Param key: {old_key} -> {new_key}")
            policy_params = remap_param_keys(policy_params, old_key, new_key)

    # 9. Create policy function
    policy_fn = make_policy(policy_params, deterministic=True)

    # 10. Create data generator
    data_gen = make_data_generator(
        path=data_path,
        max_num_objects=max_num_objects,
        include_sdc_paths=True,
        batch_dims=(1,),
        seed=42,
        repeat=1,
    )

    print(f"Model loaded successfully! obs_size={obs_size}, action_size={action_size}")

    return {
        "policy_fn": policy_fn,
        "env": env,
        "data_gen": data_gen,
        "config": config,
        "policy_params": policy_params,
        "unflatten_fn": unflatten_fn,
    }
```

---

## Usage Examples

### Example 1: Load a model and run one scenario

```python
result = load_vmax_model(
    model_dir="runs_rlc/womd_sac_road_perceiver_minimal_42",
    data_path="data/training.tfrecord",
)

env = result["env"]
policy_fn = result["policy_fn"]
data_gen = result["data_gen"]

# Get a scenario
scenario = next(iter(data_gen))

# Reset environment
rng_key = jax.random.PRNGKey(0)
rng_key, reset_key = jax.random.split(rng_key)
reset_key = jax.random.split(reset_key, 1)
env_transition = jax.jit(env.reset)(scenario, reset_key)

# Step through the episode
from functools import partial
from vmax.agents import pipeline

step_fn = partial(pipeline.policy_step, env=env, policy_fn=policy_fn)

steps = 0
while not bool(env_transition.done):
    rng_key, step_key = jax.random.split(rng_key)
    step_key = jax.random.split(step_key, 1)
    env_transition, transition = step_fn(env_transition, key=step_key)
    steps += 1

print(f"Episode done in {steps} steps")
for k, v in env_transition.metrics.items():
    print(f"  {k}: {float(v[0]):.4f}")
```

### Example 2: Inspect model parameters

```python
from vmax.scripts.evaluate.utils import load_params
import jax, numpy as np

state = load_params("runs_rlc/womd_sac_road_perceiver_minimal_42/model/model_final.pkl")
flat = jax.tree_util.tree_leaves_with_path(state.policy)

print(f"Total params: {sum(np.prod(l.shape) for _, l in flat):,}")
for path, leaf in flat:
    print(f"  {'/'.join(str(p) for p in path)}: {leaf.shape}")
```

### Example 3: Extract attention intermediates (for XAI)

```python
from vmax.agents.networks import encoders, network_utils, decoders
from vmax.agents.networks.network_factory import PolicyNetwork

result = load_vmax_model(
    model_dir="runs_rlc/womd_sac_road_perceiver_minimal_42",
    data_path="data/training.tfrecord",
)

# Rebuild the Flax module for capture_intermediates
eval_config = result["config"]
eval_config["network"]["encoder"]["type"] = "lq"  # already remapped in load_vmax_model
_config = network_utils.convert_to_dict_with_activation_fn({
    "encoder": eval_config["network"]["encoder"],
    "policy": eval_config["algorithm"]["network"]["policy"],
})

enc_cfg = dict(_config["encoder"])
enc_type = enc_cfg.pop("type")
encoder_layer = encoders.get_encoder(enc_type)(result["unflatten_fn"], **enc_cfg)

pol_cfg = _config["policy"]
fc_cfg = {k: v for k, v in pol_cfg.items() if k not in ("type", "final_activation", "num_networks", "shared_encoder")}
fc_layer = decoders.get_fully_connected(pol_cfg["type"])(**fc_cfg)

env = result["env"]
action_size = env.action_spec().data.shape[0]

policy_module = PolicyNetwork(
    encoder_layer=encoder_layer,
    fully_connected_layer=fc_layer,
    output_size=action_size * 2,
    final_activation=pol_cfg["final_activation"],
)

# Run forward pass with intermediates
scenario = next(iter(result["data_gen"]))
rng_key = jax.random.PRNGKey(0)
_, reset_key = jax.random.split(rng_key)
reset_key = jax.random.split(reset_key, 1)
env_transition = jax.jit(env.reset)(scenario, reset_key)
obs = env_transition.observation

output, state = policy_module.apply(
    result["policy_params"],
    obs,
    capture_intermediates=True,
    mutable=["intermediates"],
)
# state["intermediates"] contains all layer activations
# including cross_attn Q/K/V projections for attention visualization
```

### Example 4: Unflatten observations to see per-category features

```python
result = load_vmax_model(
    model_dir="runs_rlc/womd_sac_road_perceiver_minimal_42",
    data_path="data/training.tfrecord",
)
unflatten_fn = result["unflatten_fn"]

# Get an observation
scenario = next(iter(result["data_gen"]))
rng_key = jax.random.PRNGKey(0)
_, reset_key = jax.random.split(rng_key)
reset_key = jax.random.split(reset_key, 1)
env_transition = jax.jit(result["env"].reset)(scenario, reset_key)

features, masks = unflatten_fn(env_transition.observation)
sdc_traj, other_agents, roadgraph, traffic_lights, gps_path = features
sdc_mask, agent_mask, rg_mask, tl_mask = masks

print(f"SDC trajectory:  {sdc_traj.shape}")    # (1, 1, 5, 7) - 5 timesteps, 7 features
print(f"Other agents:    {other_agents.shape}") # (1, 8, 5, 7) - 8 agents x 5 timesteps
print(f"Roadgraph:       {roadgraph.shape}")    # (1, 200, 4)  - 200 points, 4 features
print(f"Traffic lights:  {traffic_lights.shape}")# (1, 5, 5, 10) - 5 lights x 5 timesteps
print(f"GPS path:        {gps_path.shape}")     # (1, 10, 2) - 10 waypoints (x,y)
```

---

## CLI Usage (evaluate.py)

The evaluate script **does not** apply the fixes above automatically. It will fail on most models. If you want to use it, you must either:

1. **Patch the codebase** (recommended — see "Patching the Codebase" below), OR
2. **Use models that don't need fixes** (`womd_sac_road_mtr_*`, `womd_sac_road_wayformer_*`, `womd_sac_road_none_*` — but these still need the `road -> vec` fix)

### CLI syntax (after patching)

```bash
cd /path/to/RL-IL/vmax

# Evaluate perceiver model (after codebase patches applied)
python -m vmax.scripts.evaluate.evaluate \
    --sdc_actor ai \
    --path_model womd_sac_road_perceiver_minimal_42 \
    --path_dataset /path/to/RL-IL/data/training.tfrecord \
    --src_dir /path/to/RL-IL/runs_rlc \
    --batch_size 8

# Render a scenario as video
python -m vmax.scripts.evaluate.evaluate \
    --sdc_actor ai \
    --path_model womd_sac_road_perceiver_minimal_42 \
    --path_dataset /path/to/RL-IL/data/training.tfrecord \
    --src_dir /path/to/RL-IL/runs_rlc \
    --render true \
    --sdc_pov true
```

**Critical flags:**
- `--src_dir`: Must point to `runs_rlc/`, NOT the default `runs/`
- `--path_dataset`: Direct path to your `.tfrecord` file
- `--batch_size 1` is required when using `--render` or `--sdc_pov`

### Your friend's broken command and why it failed

```bash
# THIS WILL FAIL:
python -m vmax.scripts.evaluate.evaluate --sdc_actor ai --path_model sac_seed0 --path_dataset training.tfrecord --batch_size 8
```

**Failures:**
1. `--src_dir` defaults to `runs` — should be `runs_rlc`
2. `sac_seed0` uses `speed_limit` in roadgraph features (Issue 5)
3. `--path_dataset training.tfrecord` — needs full path

---

## Patching the Codebase (Optional)

If you want the CLI and the original `load_model()` to work without the manual Python fixes, apply these patches:

### Patch 1: Add encoder aliases

In `vmax/vmax/agents/networks/encoders/__init__.py`, add aliases to the registry:

```python
encoders = {
    "mlp": MLPEncoder,
    "lq": LQEncoder,
    "perceiver": LQEncoder,   # ← ADD THIS
    "wayformer": WayformerEncoder,
    "mtr": MTREncoder,
    "lqh": LQHEncoder,
    "mgail": LQHEncoder,      # ← ADD THIS
}
```

### Patch 2: Add observation type alias

In `vmax/vmax/simulator/features/extractor/__init__.py`, add aliases:

```python
mapping = {
    "vec": VecFeaturesExtractor,
    "road": VecFeaturesExtractor,  # ← ADD THIS
    "lane": VecFeaturesExtractor,  # ← ADD THIS
    "gt": GTFeaturesExtractor,
    "idm": IDMFeaturesExtractor,
}
```

### Patch 3: Handle param key remapping in load_params

In `vmax/vmax/scripts/evaluate/utils.py`, after loading params in `load_model()`, add renaming logic before `make_policy()`.

---

## Architecture Quick Reference

| Encoder | Param count | Attention type | Module name in params |
|---------|------------|----------------|----------------------|
| LQ (perceiver) | ~652K | Cross + Self attention, 4 layers, tied weights | `perceiver_attention` or `lq_attention` |
| LQH (mgail) | ~830K | Cross attention per feature type, no self-attn | `mgail_attention` or `lq_attention` |
| MTR | ~560K | Local k-NN attention | `mtr_attention` |
| Wayformer | ~640K | Late fusion, per-modality attention | `*_attention` per feature type |
| None (MLP) | ~430K | No encoder, flat MLP only | N/A |

### Observation structure (all models except sac_seed*)

| Component | Shape | Features |
|-----------|-------|----------|
| SDC trajectory | (1, 5, 7) | waypoints(2) + velocity(2) + yaw(1) + size(2) |
| Other agents | (8, 5, 7) | Same as SDC, 8 closest agents |
| Roadgraph | (200, 4) | waypoints(2) + direction(2) |
| Traffic lights | (5, 5, 10) | waypoints(2) + state(7) + valid(1), 5 closest |
| GPS path | (10, 2) | waypoints(2), 10 target points |

The `valid` feature in each category is used as a **mask** (not passed to encoder).

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'vmax.learning'` | Missing tensorboardX | `pip install tensorboardX` |
| `ModuleNotFoundError: No module named 'vmax.scripts'` | vmax not on sys.path | `sys.path.insert(0, "path/to/RL-IL/vmax")` |
| `ValueError: Unknown encoder: perceiver` | Encoder rename | Remap to `lq` (see Issue 2) |
| `ValueError: Unknown encoder: mgail` | Encoder rename | Remap to `lqh` (see Issue 2) |
| `ValueError: Unknown feature extractor: road` | Obs type rename | Remap to `vec` (see Issue 3) |
| `ScopeParamNotFoundError: "latents" in lq_attention` | Param key mismatch | Remap keys (see Issue 4) |
| `AttributeError: 'RoadgraphPoints' has no 'speed_limit'` | sac_seed* models only | Use different model (see Issue 5) |
| `FileNotFoundError` with `--src_dir` | Wrong base dir | Use `--src_dir runs_rlc` |
