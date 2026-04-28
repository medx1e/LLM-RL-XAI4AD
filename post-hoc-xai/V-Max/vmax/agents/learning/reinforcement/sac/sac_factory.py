# Copyright 2025 Valeo.

"""Factory functions for the Soft Actor-Critic (SAC) algorithm."""

from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from vmax.agents import datatypes, networks


@flax.struct.dataclass
class SACNetworkParams:
    """Parameters for SAC network."""

    policy: datatypes.Params
    value: datatypes.Params
    target_value: datatypes.Params


@flax.struct.dataclass
class SACNetworks:
    """SAC networks."""

    policy_network: Any
    value_network: Any
    parametric_action_distribution: Any
    policy_optimizer: Any
    value_optimizer: Any


@flax.struct.dataclass
class SACTrainingState(datatypes.TrainingState):
    """Training state for SAC algorithm."""

    params: SACNetworkParams
    policy_optimizer_state: optax.OptState
    value_optimizer_state: optax.OptState
    rl_gradient_steps: int


def initialize(
    action_size: int,
    observation_size: int,
    env: Any,
    learning_rate: float,
    network_config: dict,
    num_devices: int,
    key: jax.Array,
) -> tuple[SACNetworks, SACTrainingState, datatypes.Policy]:
    """Initialize SAC components.

    Args:
        action_size: Size of the action space.
        observation_size: Size of the observation space.
        env: Environment instance with a features extractor.
        learning_rate: Learning rate for the optimizers.
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

    policy_params = network.policy_network.init(key_policy)
    policy_optimizer_state = network.policy_optimizer.init(policy_params)
    value_params = network.value_network.init(key_value)
    value_optimizer_state = network.value_optimizer.init(value_params)

    init_params = SACNetworkParams(
        policy=policy_params,
        value=value_params,
        target_value=value_params,
    )

    training_state = SACTrainingState(
        params=init_params,
        policy_optimizer_state=policy_optimizer_state,
        value_optimizer_state=value_optimizer_state,
        env_steps=0,
        rl_gradient_steps=0,
    )

    training_state = jax.device_put_replicated(training_state, jax.local_devices()[:num_devices])

    return network, training_state, policy_function


def make_inference_fn(sac_network: SACNetworks) -> datatypes.Policy:
    """Create the policy inference function for SAC.

    Args:
        sac_network: Instance of SACNetworks.

    Returns:
        A callable policy function.

    """

    def make_policy(params: datatypes.Params, deterministic: bool = False) -> datatypes.Policy:
        policy_network = sac_network.policy_network
        parametric_action_distribution = sac_network.parametric_action_distribution

        def policy(observations: jax.Array, key_sample: jax.Array = None) -> tuple[jax.Array, dict]:
            logits = policy_network.apply(params, observations)

            if deterministic:
                return parametric_action_distribution.mode(logits), {}

            return parametric_action_distribution.sample(logits, key_sample), {}

        return policy

    return make_policy


def make_networks(
    observation_size: int,
    action_size: int,
    unflatten_fn: callable,
    learning_rate: int,
    network_config: dict,
) -> SACNetworks:
    """Construct SAC networks.

    Args:
        observation_size: Size of the observation space.
        action_size: Size of the action space.
        unflatten_fn: Function to unflatten network inputs.
        learning_rate: Learning rate used for the optimizers.
        network_config: Network configuration dictionary.

    Returns:
        An instance of SACNetworks.

    """
    if "gaussian" in network_config["action_distribution"]:
        parametric_action_distribution = networks.NormalTanhDistribution(event_size=action_size)
    elif "beta" in network_config["action_distribution"]:
        parametric_action_distribution = networks.BetaDistribution(event_size=action_size)

    output_size = parametric_action_distribution.param_size

    policy_network = networks.make_policy_network(network_config, observation_size, output_size, unflatten_fn)
    value_network = networks.make_value_network(network_config, observation_size, action_size, unflatten_fn)

    policy_optimizer = optax.adam(learning_rate)
    value_optimizer = optax.adam(learning_rate)

    return SACNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
    )


def make_sgd_step(sac_network: SACNetworks, alpha: float, discount: float, tau: float) -> datatypes.LearningFunction:
    """Create the SGD step function for SAC.

    Args:
        sac_network: The SAC networks.
        alpha: Entropy regularization coefficient.
        discount: Discount factor.
        tau: Coefficient for target network updates.

    Returns:
        A function that executes an SGD step.

    """
    value_loss, policy_loss = _make_loss_fn(sac_network=sac_network, alpha=alpha, discount=discount)

    policy_update = networks.gradient_update_fn(policy_loss, sac_network.policy_optimizer, pmap_axis_name="batch")
    value_update = networks.gradient_update_fn(value_loss, sac_network.value_optimizer, pmap_axis_name="batch")

    def sgd_step(
        carry: tuple[SACTrainingState, jax.Array],
        transitions: datatypes.RLTransition,
    ) -> tuple[tuple[SACTrainingState, jax.Array], datatypes.Metrics]:
        training_state, key = carry

        key, key_value, key_policy = jax.random.split(key, 3)

        value_loss, value_params, value_optimizer_state = value_update(
            training_state.params.value,
            training_state.params.policy,
            training_state.params.target_value,
            transitions,
            key_value,
            optimizer_state=training_state.value_optimizer_state,
        )
        policy_loss, policy_params, policy_optimizer_state = policy_update(
            training_state.params.policy,
            training_state.params.value,
            transitions,
            key_policy,
            optimizer_state=training_state.policy_optimizer_state,
        )

        new_target_value_params = jax.tree_util.tree_map(
            lambda x, y: x * (1 - tau) + y * tau,
            training_state.params.target_value,
            value_params,
        )

        sgd_metrics = {"policy_loss": policy_loss, "value_loss": value_loss}

        params = SACNetworkParams(
            policy=policy_params,
            value=value_params,
            target_value=new_target_value_params,
        )

        training_state = training_state.replace(
            params=params,
            policy_optimizer_state=policy_optimizer_state,
            value_optimizer_state=value_optimizer_state,
            rl_gradient_steps=training_state.rl_gradient_steps + 1,
        )

        return (training_state, key), sgd_metrics

    return sgd_step


def _make_loss_fn(sac_network: SACNetworks, alpha: float, discount: float) -> tuple[callable, callable]:
    """Define the loss functions for SAC.

    Args:
        sac_network: The SAC networks.
        alpha: Entropy regularization coefficient.
        discount: Discount factor.

    Returns:
        A tuple containing the value loss and policy loss functions.

    """
    policy_network = sac_network.policy_network
    value_network = sac_network.value_network
    parametric_action_distribution = sac_network.parametric_action_distribution

    def compute_value_loss(
        value_params: datatypes.Params,
        policy_params: datatypes.Params,
        target_value_params: datatypes.Params,
        transitions: datatypes.RLTransition,
        key: jax.Array,
    ) -> jax.Array:
        value_old_action = value_network.apply(value_params, transitions.observation, transitions.action)
        next_dist_params = policy_network.apply(policy_params, transitions.next_observation)

        next_action = parametric_action_distribution.sample_no_postprocessing(next_dist_params, key)
        next_log_prob = parametric_action_distribution.log_prob(next_dist_params, next_action)
        next_action = parametric_action_distribution.postprocess(next_action)

        next_value = value_network.apply(target_value_params, transitions.next_observation, next_action)
        next_v = jnp.min(next_value, axis=-1) - alpha * next_log_prob

        target_value = jax.lax.stop_gradient(transitions.reward + transitions.flag * discount * next_v)
        value_error = value_old_action - jnp.expand_dims(target_value, -1)
        value_loss = 0.5 * jnp.mean(jnp.square(value_error))

        return value_loss

    def compute_policy_loss(
        policy_params: datatypes.Params,
        value_params: datatypes.Params,
        transitions: datatypes.RLTransition,
        key: jax.Array,
    ) -> jax.Array:
        dist_params = policy_network.apply(policy_params, transitions.observation)

        action = parametric_action_distribution.sample_no_postprocessing(dist_params, key)
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        action = parametric_action_distribution.postprocess(action)

        value_action = value_network.apply(value_params, transitions.observation, action)
        min_value = jnp.min(value_action, axis=-1)
        policy_loss = alpha * log_prob - min_value

        return jnp.mean(policy_loss)

    return compute_value_loss, compute_policy_loss
