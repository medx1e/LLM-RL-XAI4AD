# Copyright 2025 Valeo.

"""Factory functions for the Proximal Policy Optimization (PPO) algorithm."""

from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from vmax.agents import datatypes, networks


@flax.struct.dataclass
class PPONetworkParams:
    """Parameters for PPO network."""

    policy: datatypes.Params
    value: datatypes.Params


@flax.struct.dataclass
class PPONetworks:
    """PPO networks."""

    policy_network: Any
    value_network: Any
    parametric_action_distribution: Any
    optimizer: Any


@flax.struct.dataclass
class PPOTrainingState(datatypes.TrainingState):
    """Training state for PPO algorithm."""

    params: PPONetworkParams
    optimizer_state: optax.OptState
    rl_gradient_steps: int


def initialize(
    action_size: int,
    observation_size: int,
    env: Any,
    learning_rate: float,
    network_config: dict,
    num_devices: int,
    key: jax.Array,
) -> tuple[PPONetworks, PPOTrainingState, datatypes.Policy]:
    """Initialize PPO components.

    Args:
        action_size: Size of the action space.
        observation_size: Size of the observation space.
        env: Environment instance with a features extractor.
        learning_rate: Learning rate for the optimizer.
        network_config: Network configuration dictionary.
        num_devices: Number of devices to use.
        key: Random key for initialization.

    Returns:
        A tuple of (networks, training state, policy function).

    """
    network = make_networks(
        observation_size=observation_size,
        action_size=action_size,
        unflatten_fn=env.get_wrapper_attr("features_extractor").unflatten_features,
        learning_rate=learning_rate,
        network_config=network_config,
    )

    policy_function = make_inference_fn(network)

    key_policy, key_value = jax.random.split(key)

    init_params = PPONetworkParams(
        policy=network.policy_network.init(key_policy),
        value=network.value_network.init(key_value),
    )
    optimizer_state = network.optimizer.init(init_params)

    training_state = PPOTrainingState(
        optimizer_state=optimizer_state,
        params=init_params,
        env_steps=0,
        rl_gradient_steps=0,
    )

    training_state = jax.device_put_replicated(training_state, jax.local_devices()[:num_devices])

    return network, training_state, policy_function


def make_inference_fn(ppo_networks: PPONetworks) -> datatypes.Policy:
    """Create the policy inference function for PPO.

    Args:
        ppo_networks: Instance of PPONetworks.

    Returns:
        A callable policy function.

    """

    def make_policy(params: datatypes.Params, deterministic: bool = False) -> datatypes.Policy:
        policy_network = ppo_networks.policy_network
        parametric_action_distribution = ppo_networks.parametric_action_distribution

        def policy(observations: jax.Array, key_sample: jax.Array = None) -> tuple[jax.Array, dict]:
            logits = policy_network.apply(params, observations)

            if deterministic:
                return parametric_action_distribution.mode(logits), {}

            raw_actions = parametric_action_distribution.sample_no_postprocessing(logits, key_sample)

            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(raw_actions)

            return postprocessed_actions, {"log_prob": log_prob, "raw_action": raw_actions}

        return policy

    return make_policy


def make_networks(
    observation_size: int,
    action_size: int,
    unflatten_fn: callable,
    learning_rate: int,
    network_config: dict,
) -> PPONetworks:
    """Construct PPO networks.

    Args:
        observation_size: Size of the observation space.
        action_size: Size of the action space.
        unflatten_fn: Function to unflatten network outputs.
        learning_rate: Learning rate used for the optimizer.
        network_config: Network configuration dictionary.

    Returns:
        An instance of PPONetworks.

    """
    if "gaussian" in network_config["action_distribution"]:
        parametric_action_distribution = networks.NormalTanhDistribution(event_size=action_size)
    elif "beta" in network_config["action_distribution"]:
        parametric_action_distribution = networks.BetaDistribution(event_size=action_size)

    output_size = parametric_action_distribution.param_size

    policy_network = networks.make_policy_network(network_config, observation_size, output_size, unflatten_fn)
    value_network = networks.make_value_network(
        network_config,
        observation_size,
        action_size,
        unflatten_fn,
        concat_obs_action=False,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate),
    )

    return PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
        optimizer=optimizer,
    )


def make_sgd_step(
    ppo_network: PPONetworks,
    num_minibatches: int,
    gae_lambda: float,
    discount: float,
    eps_clip: float,
    value_coef: float,
    entropy_coef: float,
    normalize_advantages: bool,
) -> datatypes.LearningFunction:
    """Create the SGD step function for PPO.

    Args:
        ppo_network: The PPO networks.
        num_minibatches: Number of minibatches.
        gae_lambda: Lambda for Generalized Advantage Estimation.
        discount: Discount factor.
        eps_clip: PPO clipping parameter.
        value_coef: Coefficient for value loss.
        entropy_coef: Coefficient for entropy term.
        normalize_advantages: Flag to normalize advantages.

    Returns:
        A function that executes an SGD step.

    """
    ppo_loss = _make_loss_fn(
        ppo_network,
        gae_lambda,
        discount,
        eps_clip,
        value_coef,
        entropy_coef,
        normalize_advantages,
    )
    ppo_update = networks.gradient_update_fn(ppo_loss, ppo_network.optimizer, pmap_axis_name="batch", has_aux=True)

    def sgd_step(
        carry: tuple[PPOTrainingState, jax.Array],
        _t,
        transitions: datatypes.RLTransition,
    ) -> tuple[tuple[PPOTrainingState, jax.Array], datatypes.Metrics]:
        training_state, key = carry

        key, key_permunation, key_grad = jax.random.split(key, 3)

        def convert_data(x: jax.Array):
            x = jax.random.permutation(key_permunation, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])

            return x

        shuffled_transitions = jax.tree_util.tree_map(convert_data, transitions)
        (training_state, _), sgd_metrics = jax.lax.scan(
            minibatch_step,
            (training_state, key_grad),
            shuffled_transitions,
            length=num_minibatches,
        )

        return (training_state, key), sgd_metrics

    def minibatch_step(
        carry: tuple[PPOTrainingState, jax.Array],
        data: datatypes.RLTransition,
    ) -> tuple[tuple[PPOTrainingState, jax.Array], datatypes.Metrics]:
        training_state, key = carry
        key, key_loss = jax.random.split(key)

        (_, sgd_metrics), params, optimizer_state = ppo_update(
            training_state.params,
            data,
            key_loss,
            optimizer_state=training_state.optimizer_state,
        )

        training_state = training_state.replace(
            params=params,
            optimizer_state=optimizer_state,
            rl_gradient_steps=training_state.rl_gradient_steps + 1,
        )

        return (training_state, key), sgd_metrics

    return sgd_step


def _make_loss_fn(
    ppo_network: PPONetworks,
    gae_lambda: float,
    discount: float,
    eps_clip: float,
    value_coef: float,
    entropy_coef: float,
    normalize_advantages: bool,
) -> tuple[jax.Array, datatypes.Metrics]:
    """Define PPO loss and associated metrics.

    Args:
        ppo_network: The PPO networks.
        gae_lambda: Lambda for Generalized Advantage Estimation.
        discount: Discount factor.
        eps_clip: Clipping parameter.
        value_coef: Coefficient for value loss.
        entropy_coef: Coefficient for entropy loss.
        normalize_advantages: Whether or not to normalize the advantages.

    Returns:
        The computed loss and a dictionary of metrics.

    """
    policy_apply = ppo_network.policy_network.apply
    value_apply = ppo_network.value_network.apply
    parametric_action_distribution = ppo_network.parametric_action_distribution

    def compute_loss(
        params,
        data: datatypes.RLTransition,
        key: jax.Array,
    ) -> tuple[jax.Array, datatypes.Metrics]:
        # (T, B, ...) -> (B, T, ...)
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

        # Flatten the batch dimension
        obs = jnp.reshape(data.observation, (-1,) + data.observation.shape[2:])

        policy_logits = policy_apply(params.policy, obs)
        baseline = value_apply(params.value, obs).squeeze(-1)

        # Unflatten the batch dimension
        policy_logits = jnp.reshape(policy_logits, (data.observation.shape[0], -1) + policy_logits.shape[1:])
        baseline = jnp.reshape(baseline, (data.observation.shape[0], -1))

        bootstrap_value = value_apply(params.value, data.next_observation[-1]).squeeze(-1)

        rewards = data.reward
        truncation = data.extras["state_extras"]["truncation"]
        termination = (1 - data.flag) * (1 - truncation)

        target_log_probs = parametric_action_distribution.log_prob(
            policy_logits,
            data.extras["policy_extras"]["raw_action"],
        )
        log_probs = data.extras["policy_extras"]["log_prob"]

        vs, advantages = _compute_gae(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            values=baseline,
            bootstrap_value=bootstrap_value,
            lambda_=gae_lambda,
            discount=discount,
        )

        if normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        rho_s = jnp.exp(target_log_probs - log_probs)
        surrogate_loss1 = advantages * rho_s
        surrogate_loss2 = advantages * jnp.clip(rho_s, 1 - eps_clip, 1 + eps_clip)

        policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

        # Value function loss
        v_error = vs - baseline
        value_loss = jnp.mean(v_error * v_error) * value_coef

        # Entropy reward
        entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, key))
        entropy_loss = entropy_coef * -entropy

        total_loss = policy_loss + value_loss + entropy_loss

        return total_loss, {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "total_loss": total_loss,
        }

    return compute_loss


def _compute_gae(
    truncation: jax.Array,
    termination: jax.Array,
    rewards: jax.Array,
    values: jax.Array,
    bootstrap_value: jax.Array,
    lambda_: float,
    discount: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute Generalized Advantage Estimation (GAE).

    Args:
        truncation: Truncation mask.
        termination: Termination mask.
        rewards: Reward values.
        values: Value predictions.
        bootstrap_value: Value estimate for the last state.
        lambda_: GAE lambda parameter.
        discount: Discount factor.

    Returns:
        A tuple with the target values and computed advantages.

    """
    truncation_mask = 1 - truncation

    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = jnp.concatenate([values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
    deltas = rewards + discount * (1 - termination) * values_t_plus_1 - values
    deltas *= truncation_mask

    adv = jnp.zeros_like(bootstrap_value)
    vs_minus_v_xs = []

    def compute_vs_minus_v_xs(carry, target_t):
        lambda_, adv = carry
        truncation_mask, delta, termination = target_t
        adv = delta + discount * (1 - termination) * truncation_mask * lambda_ * adv
        return (lambda_, adv), (adv)

    (_, _), (vs_minus_v_xs) = jax.lax.scan(
        compute_vs_minus_v_xs,
        (lambda_, adv),
        (truncation_mask, deltas, termination),
        length=int(truncation_mask.shape[0]),
        reverse=True,
    )
    # Add V(x_s) to get v_s.
    td_target = jnp.add(vs_minus_v_xs, values)

    vs_t_plus_1 = jnp.concatenate([td_target[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
    advantages = (rewards + discount * (1 - termination) * vs_t_plus_1 - values) * truncation_mask

    return jax.lax.stop_gradient(td_target), jax.lax.stop_gradient(advantages)
