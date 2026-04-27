# V-MAX Model Weights - Exploration & XAI Guide

This document provides a comprehensive overview of the V-MAX model weights obtained from the paper authors, intended for **Explainable AI (XAI)** experiments on autonomous driving policies.

---

## Table of Contents

1. [Directory Structure](#directory-structure)
2. [Model Naming Convention](#model-naming-convention)
3. [Encoder Architectures](#encoder-architectures)
4. [Datasets](#datasets)
5. [Reward Configurations](#reward-configurations)
6. [Model Performance Summary](#model-performance-summary)
7. [Configuration Files Reference](#configuration-files-reference)
8. [Loading Models for XAI Analysis](#loading-models-for-xai-analysis)
9. [XAI Analysis Recommendations](#xai-analysis-recommendations)

---

## Directory Structure

```
runs_rlc/
├── runs_accuracy.txt                          # Performance rankings for all models
├── VMAX_MODELS_GUIDE.md                       # This guide
│
├── sac_seed{0,42,69}/                         # Best-performing models (LQ encoder)
│   ├── .hydra/
│   │   ├── config.yaml                        # Full training configuration
│   │   ├── hydra.yaml                         # Hydra framework settings
│   │   └── overrides.yaml                     # Command-line overrides used
│   ├── model/
│   │   └── model_final.pkl                    # Trained model weights
│   ├── train.log                              # Training logs
│   └── events.out.tfevents.*                  # TensorBoard logs
│
├── {dataset}_{algo}_{obs}_{encoder}_{reward}_{seed}/
│   ├── .hydra/                                # Configuration files
│   ├── model/
│   │   └── model_final.pkl                    # Model checkpoint (~5-11 MB)
│   └── train.log                              # Training output
│
└── ... (36 model directories total)
```

---

## Model Naming Convention

Each model directory follows this naming pattern:

```
{dataset}_{algorithm}_{observation}_{encoder}_{reward}_{seed}
```

| Component | Options | Description |
|-----------|---------|-------------|
| **dataset** | `womd`, `nuplan`, `mix` | Training data source |
| **algorithm** | `sac` | Soft Actor-Critic (all models) |
| **observation** | `road`, `lane` | Observation representation type |
| **encoder** | `perceiver`, `mgail`, `mtr`, `wayformer`, `none` | Neural encoder architecture |
| **reward** | `minimal`, `basic`, `complete` | Reward function complexity |
| **seed** | `42`, `69`, `99` | Random seed for reproducibility |

### Examples

| Directory Name | Interpretation |
|----------------|----------------|
| `womd_sac_road_perceiver_minimal_42` | WOMD dataset, SAC algorithm, road observation, Perceiver encoder, minimal reward, seed 42 |
| `nuplan_sac_road_perceiver_minimal_69` | nuPlan dataset, Perceiver encoder, seed 69 |
| `womd_sac_lane_perceiver_minimal_42` | Lane-based observation instead of road-based |
| `womd_sac_road_none_minimal_42` | No encoder (MLP baseline) |

### Special Cases: `sac_seed{0,42,69}`

These are the **best-performing models** using:
- **LQ (Latent Query) encoder** - a variant of Perceiver
- **Vector observation type** (`vec`)
- Extended reward configuration with comfort and overspeed penalties
- 30M training timesteps (vs 25M for others)

---

## Encoder Architectures

### 1. Perceiver (`type: perceiver`)

Cross-attention based architecture inspired by DeepMind's Perceiver.

```yaml
encoder:
  type: perceiver
  embedding_layer_sizes: [256, 256]    # Input embedding MLP
  embedding_activation: relu
  encoder_depth: 4                      # Number of cross-attention layers
  dk: 64                                # Key/query dimension
  num_latents: 16                       # Number of latent vectors
  latent_num_heads: 2                   # Self-attention heads in latent space
  latent_head_features: 16              # Features per head
  cross_num_heads: 2                    # Cross-attention heads
  cross_head_features: 16
  ff_mult: 2                            # Feed-forward expansion factor
  attn_dropout: 0.0
  ff_dropout: 0.0
  tie_layer_weights: true               # Share weights across layers
```

**XAI Potential**: Cross-attention weights reveal which input elements (objects, road points, traffic lights) the model attends to.

### 2. LQ - Latent Query (`type: lq`)

Similar to Perceiver, used in the best-performing `sac_seed*` models.

```yaml
encoder:
  type: lq
  # Same architecture as Perceiver
  encoder_depth: 4
  dk: 64
  num_latents: 16
  # ...
```

### 3. MTR - Motion Transformer (`type: mtr`)

Based on Waymo's Motion Transformer architecture.

```yaml
encoder:
  type: mtr
  encoder_depth: 4
  dk: 64
  num_latents: 16
  ff_mult: 4                            # Larger feed-forward network
  k: 8                                  # k-nearest neighbor attention
```

**XAI Potential**: Local attention patterns (k=8 neighbors) show spatial reasoning.

### 4. Wayformer (`type: wayformer`)

Waymo's factorized attention architecture.

```yaml
encoder:
  type: wayformer
  attention_depth: 2                    # Fewer layers
  fusion_type: late                     # Late fusion of modalities
  # ...
```

**XAI Potential**: Late fusion allows analyzing per-modality contributions separately.

### 5. MGAIL (`type: mgail`)

Encoder from Multi-agent Generative Adversarial Imitation Learning.

```yaml
encoder:
  type: mgail
  use_self_attention: false             # No self-attention variant
  # ...
```

**Model Size**: Largest at ~11MB due to additional components.

### 6. None (`type: none`)

MLP baseline with no encoder - flattened observations fed directly to policy.

```yaml
encoder:
  type: none
```

**XAI Potential**: Baseline for comparing what attention-based encoders learn.

---

## Datasets

### WOMD (Waymo Open Motion Dataset)

- **Prefix**: `womd_*`
- **Source**: Waymo's autonomous driving scenarios
- **Path**: `gs://valeo-cp2137-datasets/v1.2/womd/train/training.tfrecord@1000`
- **Most models trained on this dataset**

### nuPlan

- **Prefix**: `nuplan_*`
- **Source**: nuPlan autonomous driving benchmark
- **3 models available** (perceiver_minimal, seeds 42/69/99)

### Mix

- **Prefix**: `mix_*`
- **Source**: Combined WOMD + nuPlan
- **3 models available** (perceiver_minimal, seeds 42/69/99)

---

## Reward Configurations

### Minimal Reward

Used by most models. Focuses on safety and basic navigation.

```yaml
reward_config:
  overlap:                    # Collision penalty
    bonus: 0.0
    penalty: -1.0
    weight: 1.0
  offroad:                    # Going off road
    bonus: 0.0
    penalty: -1.0
    weight: 1.0
  red_light:                  # Running red lights
    penalty: -1.0
    weight: 1.0
  off_route:                  # Deviating from planned route
    penalty: -1.0
    weight: 0.6
  progression:                # Making progress toward goal
    bonus: 1.0
    penalty: 0.0
    weight: 0.2
```

### Basic Reward

Simpler configuration without route/progression rewards.

```yaml
reward_config:
  overlap: {penalty: -1.0, weight: 1.0}
  offroad: {penalty: -1.0, weight: 1.0}
  red_light: {penalty: -1.0, weight: 1.0}
  # No off_route or progression
```

### Complete Reward

Extended configuration with comfort and safety metrics.

```yaml
reward_config:
  # ... minimal rewards plus:
  speed:                      # Speed limit compliance
    bonus: 1.0
    penalty: -1.0
    weight: 0.3
  ttc:                        # Time-to-collision safety
    threshold: 1.5
    penalty: -1.0
    weight: 0.3
  comfort:                    # Smooth driving
    weight: 0.3
```

### sac_seed* Extended Reward

```yaml
reward_config:
  overlap: -1
  offroad: -1
  red_light: -1
  off_route: -0.2
  progression: 0.2
  comfort: 0.2                # Acceleration smoothness
  overspeed: -0.1             # Exceeding speed limits
```

---

## Model Performance Summary

From `runs_accuracy.txt` - sorted by overall accuracy:

### Top 10 Models

| Rank | Model | Accuracy | At-Fault Accuracy |
|------|-------|----------|-------------------|
| 1 | `sac_seed0` | 97.86% | 98.25% |
| 2 | `womd_sac_road_perceiver_minimal_42` | 97.47% | 97.87% |
| 3 | `womd_sac_road_perceiver_minimal_69` | 97.44% | 98.00% |
| 4 | `sac_seed42` | 97.27% | 97.84% |
| 5 | `womd_sac_road_perceiver_basic_42` | 97.20% | 97.79% |
| 6 | `sac_seed69` | 97.19% | 97.70% |
| 7 | `womd_sac_road_perceiver_basic_69` | 97.06% | 98.01% |
| 8 | `womd_sac_road_perceiver_minimal_99` | 96.87% | 97.54% |
| 9 | `mix_sac_road_perceiver_minimal_69` | 96.80% | 97.39% |
| 10 | `womd_sac_road_mgail_minimal_42` | 96.66% | 97.39% |

### Encoder Performance Comparison (WOMD, minimal reward, averaged across seeds)

| Encoder | Avg Accuracy | Notes |
|---------|--------------|-------|
| **LQ** (sac_seed*) | 97.44% | Best overall |
| **Perceiver** | 97.26% | Strong attention-based |
| **Perceiver Basic** | 96.69% | Simpler reward hurts |
| **MGAIL** | 96.27% | Competitive |
| **MTR** | 96.07% | Good spatial reasoning |
| **Wayformer** | 96.04% | Late fusion |
| **Perceiver Complete** | 96.18% | Complex reward not better |
| **None** | 69.95% | MLP baseline (much worse) |

### Key Insight

The **None encoder baseline** (69.95%) vs attention-based encoders (96%+) shows that the encoder architecture is critical. This gap is valuable for XAI - the attention mechanisms capture essential driving-relevant features.

---

## Configuration Files Reference

### Reading Model Configuration

Each model's full configuration is stored in `.hydra/config.yaml`:

```python
import yaml

def load_model_config(model_dir):
    """Load the training configuration for a model."""
    config_path = f"{model_dir}/.hydra/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Example usage
config = load_model_config("womd_sac_road_perceiver_minimal_42")
print(f"Encoder type: {config['network']['encoder']['type']}")
print(f"Dataset: {config['path_dataset']}")
print(f"Seed: {config['seed']}")
```

### Key Configuration Fields

```yaml
# Training parameters
total_timesteps: 25000000          # Training duration
num_envs: 16                       # Parallel environments
seed: 42                           # Random seed

# Observation configuration
observation_type: road             # 'road', 'lane', or 'vec'
observation_config:
  obs_past_num_steps: 5            # History length
  objects:
    num_closest_objects: 8         # Number of tracked agents
  roadgraphs:
    max_meters: 70                 # Road graph range
    max_num_lanes: 10
  traffic_lights:
    num_closest_traffic_lights: 5
  path_target:
    num_points: 10                 # Route waypoints

# Network architecture
network:
  encoder:
    type: perceiver                # Encoder type
    # ... encoder-specific params
  policy:
    layer_sizes: [256, 64, 32]     # Policy MLP
  value:
    layer_sizes: [256, 64, 32]     # Value MLP
    num_networks: 2                # Twin critics

# SAC algorithm
algorithm:
  name: SAC
  learning_rate: 0.0001
  discount: 0.99
  tau: 0.005                       # Target network update
  alpha: 0.2                       # Entropy coefficient
  batch_size: 64
  buffer_size: 1000000
```

---

## Loading Models for XAI Analysis

### Prerequisites

```bash
pip install jax jaxlib flax pickle5 pyyaml
```

### Basic Model Loading

```python
import pickle
import yaml
from pathlib import Path

def load_vmax_model(model_dir):
    """
    Load a V-MAX model and its configuration.

    Args:
        model_dir: Path to the model directory

    Returns:
        tuple: (model_params, config)
    """
    model_dir = Path(model_dir)

    # Load model weights
    model_path = model_dir / "model" / "model_final.pkl"
    with open(model_path, 'rb') as f:
        model_params = pickle.load(f)

    # Load configuration
    config_path = model_dir / ".hydra" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return model_params, config

# Example
params, config = load_vmax_model("womd_sac_road_perceiver_minimal_42")
```

### Inspecting Model Structure

```python
import jax.numpy as jnp

def inspect_model_params(params, prefix=""):
    """Recursively print model parameter shapes."""
    if isinstance(params, dict):
        for key, value in params.items():
            inspect_model_params(value, f"{prefix}/{key}")
    elif hasattr(params, 'shape'):
        print(f"{prefix}: {params.shape}")
    else:
        print(f"{prefix}: {type(params)}")

# Example output for Perceiver encoder:
# /encoder/embedding/Dense_0/kernel: (input_dim, 256)
# /encoder/embedding/Dense_1/kernel: (256, 256)
# /encoder/cross_attention/query: (16, 64)
# /encoder/cross_attention/key: (256, 64)
# /encoder/cross_attention/value: (256, 64)
# /encoder/latent_self_attention/...
# /policy/Dense_0/kernel: (encoder_out, 256)
# ...
```

### Extracting Attention Weights for XAI

```python
def extract_attention_weights(model_params, config):
    """
    Extract attention-related parameters for XAI analysis.

    Returns dict with:
    - cross_attention: Input-to-latent attention weights
    - self_attention: Latent self-attention weights
    - embedding: Input embedding weights
    """
    encoder_type = config['network']['encoder']['type']

    if encoder_type in ['perceiver', 'lq']:
        return {
            'cross_attention': {
                'query': model_params['encoder']['cross_attention']['query'],
                'key': model_params['encoder']['cross_attention']['key'],
                'value': model_params['encoder']['cross_attention']['value'],
            },
            'self_attention': model_params['encoder'].get('self_attention', {}),
            'latents': model_params['encoder'].get('latents', None),
            'embedding': model_params['encoder']['embedding'],
        }
    elif encoder_type == 'mtr':
        return {
            'local_attention': model_params['encoder']['local_attention'],
            'k_neighbors': config['network']['encoder']['k'],
        }
    elif encoder_type == 'wayformer':
        return {
            'modality_encoders': model_params['encoder']['modality_encoders'],
            'fusion': model_params['encoder']['fusion'],
            'fusion_type': config['network']['encoder']['fusion_type'],
        }
    else:
        return None
```

### Computing Attention Maps During Inference

```python
import jax
import jax.numpy as jnp

def compute_cross_attention(query, key, value, mask=None):
    """
    Compute cross-attention weights.

    Args:
        query: (num_latents, dk)
        key: (seq_len, dk)
        value: (seq_len, dv)
        mask: Optional attention mask

    Returns:
        attention_weights: (num_latents, seq_len)
        output: (num_latents, dv)
    """
    dk = query.shape[-1]
    scores = jnp.matmul(query, key.T) / jnp.sqrt(dk)

    if mask is not None:
        scores = jnp.where(mask, scores, -1e9)

    attention_weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.matmul(attention_weights, value)

    return attention_weights, output

# The attention_weights matrix shows which input elements
# each latent vector attends to - crucial for XAI
```

### Batch Loading Multiple Models

```python
def load_model_family(base_dir, pattern):
    """
    Load all models matching a pattern for comparative analysis.

    Example: load_model_family(".", "womd_sac_road_perceiver_minimal_*")
    """
    from glob import glob

    models = {}
    for model_dir in glob(f"{base_dir}/{pattern}"):
        name = Path(model_dir).name
        try:
            params, config = load_vmax_model(model_dir)
            models[name] = {'params': params, 'config': config}
        except Exception as e:
            print(f"Failed to load {name}: {e}")

    return models

# Load all Perceiver models with different seeds
perceiver_models = load_model_family(".", "womd_sac_road_perceiver_minimal_*")
```

---

## XAI Analysis Recommendations

### 1. Attention Visualization

**Goal**: Understand what the model focuses on when making decisions.

```python
# For each input scenario:
# 1. Extract attention weights from cross-attention layers
# 2. Map weights back to input elements (objects, road points, traffic lights)
# 3. Visualize as heatmaps overlaid on the driving scene

def visualize_attention(attention_weights, scenario):
    """
    Overlay attention weights on driving scenario visualization.

    attention_weights: (num_latents, num_objects + num_road_pts + num_tl)
    scenario: Contains positions of all elements
    """
    # Split attention by input modality
    obj_end = scenario['num_objects']
    road_end = obj_end + scenario['num_road_points']

    obj_attention = attention_weights[:, :obj_end].mean(axis=0)
    road_attention = attention_weights[:, obj_end:road_end].mean(axis=0)
    tl_attention = attention_weights[:, road_end:].mean(axis=0)

    # Visualize with matplotlib/plotly
    # ...
```

### 2. Encoder Ablation Study

**Goal**: Quantify what each encoder component contributes.

| Comparison | What It Reveals |
|------------|-----------------|
| Perceiver vs None | Value of attention mechanisms |
| Perceiver vs MTR | Global vs local attention |
| Perceiver vs Wayformer | Early vs late fusion |
| Different seeds | Model uncertainty/variance |

### 3. Latent Space Analysis

**Goal**: Understand the learned representations.

```python
# 1. Collect latent representations for many scenarios
# 2. Apply dimensionality reduction (PCA, t-SNE, UMAP)
# 3. Color by scenario properties (speed, traffic density, etc.)
# 4. Identify clusters and their meanings

from sklearn.manifold import TSNE

def analyze_latent_space(model, scenarios):
    latents = [get_latent_representation(model, s) for s in scenarios]
    latents = np.stack(latents)

    tsne = TSNE(n_components=2)
    embedded = tsne.fit_transform(latents)

    # Plot with scenario labels
    # ...
```

### 4. Feature Attribution

**Goal**: Which input features most influence actions.

```python
# Using JAX's autodiff for gradient-based attribution
def compute_saliency(model, params, observation, action_idx):
    """Compute input saliency via gradients."""
    def forward(obs):
        return model.apply(params, obs)[action_idx]

    grad_fn = jax.grad(forward)
    saliency = grad_fn(observation)
    return jnp.abs(saliency)
```

### 5. Counterfactual Analysis

**Goal**: What input changes would alter the decision?

```python
# 1. Take a baseline scenario
# 2. Systematically modify inputs (remove objects, change positions)
# 3. Observe action changes
# 4. Identify critical elements

def counterfactual_analysis(model, params, scenario):
    baseline_action = model.apply(params, scenario)

    results = []
    for i in range(scenario['num_objects']):
        modified = remove_object(scenario, i)
        new_action = model.apply(params, modified)
        results.append({
            'removed_object': i,
            'action_change': jnp.linalg.norm(new_action - baseline_action)
        })

    return sorted(results, key=lambda x: x['action_change'], reverse=True)
```

### 6. Cross-Dataset Generalization

**Goal**: How well do models transfer between datasets?

| Train Dataset | Eval on WOMD | Eval on nuPlan |
|---------------|--------------|----------------|
| WOMD | In-distribution | Transfer |
| nuPlan | Transfer | In-distribution |
| Mix | Should generalize | Should generalize |

### Recommended Model Pairs for Comparison

| Comparison | Models | Research Question |
|------------|--------|-------------------|
| Best vs Baseline | `sac_seed0` vs `womd_sac_road_none_minimal_42` | What does attention learn? |
| Encoder types | All `womd_*_minimal_42` | Which architecture is most interpretable? |
| Reward complexity | `perceiver_minimal` vs `perceiver_complete` | Does reward shape representations? |
| Dataset transfer | `womd_*` vs `nuplan_*` vs `mix_*` | Domain adaptation in attention |

---

## Quick Reference

### File Locations

| What | Where |
|------|-------|
| Model weights | `{model_dir}/model/model_final.pkl` |
| Full config | `{model_dir}/.hydra/config.yaml` |
| Overrides used | `{model_dir}/.hydra/overrides.yaml` |
| Training logs | `{model_dir}/train.log` |
| TensorBoard | `{model_dir}/events.out.tfevents.*` |
| Performance | `runs_accuracy.txt` |

### Model Sizes

| Encoder | Approximate Size |
|---------|------------------|
| None | 5.1 MB |
| MTR | 7.0 MB |
| Perceiver | 7.5 MB |
| LQ | 7.6 MB |
| MGAIL | 11 MB |

### Observation Dimensions

| Component | Dimension |
|-----------|-----------|
| Objects | 8 closest, 5 timesteps, features: waypoints, velocity, yaw, size, valid |
| Roadgraph | Up to 200 points, 70m range, features: waypoints, direction, valid |
| Traffic lights | 5 closest, features: waypoints, state, valid |
| Path target | 10 points, 5-step gap |

---

## Contact & Citation

These model weights are from the V-MAX paper. Please cite appropriately when using for research.

For issues with this guide or the XAI analysis code, refer to the original V-MAX repository or paper authors.
