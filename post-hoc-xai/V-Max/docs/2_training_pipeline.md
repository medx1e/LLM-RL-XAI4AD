# Training Pipeline Documentation

This document provides an overview of the training pipeline in the V-Max project, with a focus on the Soft Actor-Critic (SAC) reinforcement learning algorithm as an example. The goal is to help new contributors understand how the training process works, the main components involved, and how data flows through the system.

## Overview

The training pipeline is designed to train reinforcement learning agents in a simulated environment. It is modular, supporting different algorithms and environments, and is built to scale across multiple devices using JAX.

The main steps in the training pipeline are:

1. **Configuration and Initialization**
2. **Data Generation**
3. **Replay Buffer Management**
4. **Training Loop**
5. **Evaluation and Logging**
6. **Checkpointing**

Below, we describe each step using the SAC algorithm as an example.

---

## 1. Configuration and Initialization

- The training process is launched via a script (e.g., `scripts/training/train.py`) that loads configuration files using Hydra and OmegaConf.
- The configuration specifies hyperparameters, environment settings, network architectures, and paths for saving outputs.
- The environment and data generators are created based on the configuration.
- The SAC networks (policy and value networks) and optimizers are initialized. The networks are distributed across available devices using JAX's `pmap`.

## 2. Data Generation

- The simulator provides batches of scenarios (environments) for training and evaluation.
- For each training iteration, a batch of scenarios is sampled and reset.
- The agent interacts with the environment using its current policy to generate transitions (state, action, reward, next state, done).

## 3. Replay Buffer Management

- An off-policy replay buffer stores transitions collected from the environment.
- The buffer is pre-filled with random actions before training starts to ensure diverse initial data.
- During training, new transitions are inserted into the buffer, and batches are sampled for learning.

## 4. Training Loop

- The main training loop iterates over the total number of timesteps.
- In each iteration:
  - The agent generates an unroll (sequence) of transitions using its policy.
  - These transitions are added to the replay buffer.
  - A batch of transitions is sampled from the buffer.
  - Multiple gradient updates are performed using the sampled data (SGD steps).
  - The policy and value networks are updated.
- The training loop is parallelized across devices for efficiency.

## 5. Evaluation and Logging

- At regular intervals, the agent is evaluated on a separate set of scenarios.
- Evaluation uses a deterministic version of the policy.
- Metrics such as episode reward, episode length, and custom environment metrics are logged.
- Progress is reported via TensorBoard and console outputs.

## 6. Checkpointing

- Model parameters are periodically saved to disk for later analysis or resuming training.
- The final model is saved at the end of training.

---

## Key Components and Functions

- **`train.py`**: Entry point for training. Handles configuration, environment setup, and launching the training loop.
- **`sac_trainer.py`**: Implements the SAC training loop, including data collection, replay buffer management, and evaluation.
- **`sac_factory.py`**: Contains SAC-specific network and optimizer initialization, and loss functions.
- **`pipeline/training.py`**: Provides generic training and evaluation loops, supporting both off-policy and on-policy algorithms.
- **`pipeline/inference.py`**: Defines how the agent interacts with the environment (policy steps, random steps, etc.).
- **Replay Buffer**: Stores and samples transitions for off-policy learning.

---

## Data Flow Diagram (SAC Example)

```
[Simulator Scenarios] --(reset/init)--> [Environment]
      |                                      |
      v                                      v
[Agent Policy] <---(observations)--- [Environment State]
      |                                      |
      v                                      v
[Actions] --(step)--> [Environment] --(transitions)--> [Replay Buffer]
      |                                      |
      v                                      v
[Sampled Batch] <--- [Replay Buffer] <---(insert)
      |
      v
[SGD Updates] --(update)--> [Policy/Value Networks]
```

---

## Tips for New Contributors

- Start by reading `train.py` to see how the pipeline is launched.
- Follow the flow: configuration → environment/data generation → training loop → evaluation.
- Use the SAC implementation as a template for adding new algorithms.
- Check the replay buffer and pipeline modules for data handling logic.
- Use logging and TensorBoard to monitor training progress.

---

For more details, refer to the code and docstrings in each module. If you have questions, ask a team member or consult the codebase for examples.
