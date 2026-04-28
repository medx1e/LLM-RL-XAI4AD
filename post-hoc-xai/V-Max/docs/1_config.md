# V-Max Configuration Guide

This guide explains how to use and understand the YAML configuration files for V-Max training and evaluation. It covers the structure, main parameters, and how to customize settings for your experiments.

---

## 1. Configuration Structure

V-Max uses [Hydra](https://hydra.cc/) for flexible configuration management. The main configuration files are organized as follows:

- **`base_config.yaml`**: Top-level settings for training, environment, rewards, and observations.
- **`algorithm/`**: Algorithm-specific configs (e.g., SAC, PPO, BC, BC-SAC).
- **`network/`**: Network architecture and encoder configs.
- **`network/encoder/`**: Encoder-specific configs (e.g., mlp, mtr, lq, lqh, wayformer).

You can override any parameter from the command line when launching training.

---

## 2. Main Configuration Files

### `base_config.yaml`
Defines global training, environment, reward, and observation settings.

**Key sections:**
- `total_timesteps`: Total number of environment steps for training.
- `num_envs`: Number of parallel environments.
- `path_dataset`: Path to the training dataset.
- `reward_type` & `reward_config`: Reward shaping.
- `observation_type` & `observation_config`: Observation features and structure.

**Example:**
```yaml
total_timesteps: 20_000_000
num_envs: 16
path_dataset: /path/to/dataset
reward_type: linear
observation_type: vec
```

### `algorithm/$alg_name.yaml`
Algorithm-specific parameters (e.g., learning rate, batch size, network structure).

**Example:**
```yaml
name: SAC
learning_rate: 3e-4
batch_size: 64
network:
  policy:
    layer_sizes: [256, 256]
    activation: relu
```

### `network/base.yaml` & `network/encoder/$enc_name.yaml`
Defines the neural network architecture and encoder details.

**Example:**
```yaml
policy:
  type: mlp
  layer_sizes: [256, 256]
encoder:
  type: mlp
  embedding_layer_sizes: [256, 256]
```

---

## 3. Customizing Your Training

You can override any parameter directly from the command line. For example:

```bash
python vmax/scripts/train.py total_timesteps=1000000 algorithm=sac network/encoder=mlp
```

---

## 4. Key Parameters Explained

| Parameter                | Description                                 | Example Value         |
|--------------------------|---------------------------------------------|----------------------|
| `total_timesteps`        | Number of training steps                    | `20_000_000`         |
| `num_envs`               | Parallel environments                       | `16`                 |
| `path_dataset`           | Path to training data                       | `/data/train.tfrecord`|
| `reward_type`            | Reward function type                        | `linear`             |
| `observation_type`       | Type of observation function                | `vec`                |
| `algorithm`              | Algorithm to use                            | `sac`, `ppo`, `bc`   |
| `network/encoder`        | Encoder architecture                        | `mlp`, `mtr`, `lq`   |

---

## 5. Tips
- Use the provided YAML files as templates and only change what you need.
- For advanced users, create new YAML files for custom algorithms or encoders.
- Check comments in each YAML file for more details on possible values.

---

For more details, see the comments in each config file or refer to the [Hydra documentation](https://hydra.cc/docs/intro/).
