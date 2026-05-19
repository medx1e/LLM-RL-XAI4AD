# V-Max Reward Function Documentation

## Overview

In V-Max, the reward function is a key component that guides the behavior of autonomous agents during simulation. Rewards are used to evaluate how well an agent is performing with respect to safety, comfort, rule compliance, and task completion. By designing appropriate reward functions, you can encourage agents to drive safely, follow traffic rules, and achieve their objectives efficiently.

## How Rewards Work in V-Max

V-Max uses a flexible reward system based on wrappers. The main reward wrapper is `RewardLinearWrapper`, which computes the total reward as a weighted sum of several individual reward functions. Each function measures a specific aspect of driving behavior (e.g., staying on the road, avoiding collisions, obeying speed limits).

You can configure which reward functions to use and their relative importance by providing a `reward_config` dictionary, where keys are reward function names and values are their weights.

## Configuring Rewards

To use the reward system, wrap your environment with `RewardLinearWrapper` and provide a configuration, for example:

```python
reward_config = {
    "overlap": -10.0,           # Penalize collisions
    "offroad": -5.0,            # Penalize driving off the road
    "progression": 1.0,         # Reward making progress
    "comfort": 0.5,             # Reward smooth driving
    # ... add more as needed
}

env = RewardLinearWrapper(env, reward_config)
```

Each time the agent takes an action, the wrapper computes the total reward as:

```
reward = sum(weight * reward_fn(state) for each reward_fn in config)
```

## Available Reward Functions

Here are the main reward functions you can use in your configuration:

| Name                | Description                                                      |
|---------------------|------------------------------------------------------------------|
| `overlap`           | Penalizes collisions with other objects                           |
| `offroad`           | Penalizes driving off the road                                   |
| `off_route`         | Penalizes deviating from the planned route                       |
| `below_ttc`         | Penalizes unsafe time-to-collision situations                    |
| `red_light`         | Penalizes running red lights                                     |
| `overspeed`         | Penalizes exceeding the speed limit                              |
| `driving_direction` | Penalizes driving in the wrong direction                         |
| `lane_deviation`    | Penalizes deviating from the intended lane                       |
| `log_div_clip`      | Penalizes large deviations from the expected trajectory           |
| `log_div`           | Measures log divergence from the expected trajectory              |
| `progression`       | Rewards making forward progress along the route                  |
| `comfort`           | Rewards smooth and comfortable driving                           |

- **Penalty-based rewards** (e.g., `overlap`, `offroad`) usually return `True` (1.0) when a violation occurs, so you should assign them negative weights.
- **Reward-based rewards** (e.g., `progression`, `comfort`) return positive values when the agent behaves well, so you should assign them positive weights.

## Custom Rewards

If you need a custom reward function, you can subclass `RewardCustomWrapper` and implement your own logic in the `reward` method.

```python
class MyCustomRewardWrapper(RewardCustomWrapper):
    def reward(self, state, action):
        # Your custom reward logic here
        return jnp.array(...)
```

## Extending the Reward System

To add a new reward function:
1. Implement a new function (e.g., `_compute_my_new_reward(state)`) in `reward.py`.
2. Add it to the `_get_reward_fn` dictionary.
3. Use its name in your `reward_config`.

## Tips
- Tune the weights in `reward_config` to balance between safety, efficiency, and comfort.
- Use negative weights for penalties and positive weights for desirable behaviors.
- Test your configuration to ensure the agent learns the intended behavior.

## Summary
- The reward system in V-Max is modular and configurable.
- Use `RewardLinearWrapper` with a `reward_config` dictionary to combine multiple reward functions.
- Choose and tune reward functions to match your simulation goals.
- Extend or customize as needed for your use case.

For more details, see the source code in `vmax/simulator/wrappers/reward.py`.
