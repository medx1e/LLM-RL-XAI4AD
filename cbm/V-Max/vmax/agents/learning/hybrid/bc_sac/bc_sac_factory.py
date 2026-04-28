# Copyright 2025 Valeo.


"""Factory functions for the BC-SAC algorithm."""

from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from vmax.agents import datatypes, networks


@flax.struct.dataclass
class BCSACNetworkParams:
    """Parameters for BC-SAC network."""

    policy: datatypes.Params
    value: datatypes.Params
    target_value: datatypes.Params


@flax.struct.dataclass
class BCSACNetworks:
    """BC-SAC network."""

    policy_network: Any
    value_network: Any
    parametric_action_distribution: Any
    rl_policy_optimizer: Any
    imitation_policy_optimizer: Any
    value_optimizer: Any


@flax.struct.dataclass
class BCSACTrainingState(datatypes.TrainingState):
    """Training state for BC-SAC algorithm."""

    params: BCSACNetworkParams
    rl_policy_optimizer_state: optax.OptState
    imitation_policy_optimizer_state: optax.OptState
    value_optimizer_state: optax.OptState
    il_gradient_steps: int = 0
    rl_gradient_steps: int = 0


def initialize(
    action_size: int,
    observation_size: int,
    env: Any,
    rl_learning_rate: float,
    imitation_learning_rate: float,
    network_config: dict,
    num_devices: int,
    key: jax.Array,
) -> tuple[BCSACNetworks, BCSACTrainingState, datatypes.Policy]:
    """Initialize the BC-SAC components for training.

    Args:
        action_size: Size of the action space.
        observation_size: Size of the observation space.
        env: Environment object providing necessary wrappers.
        rl_learning_rate: Learning rate for reinforcement learning.
        imitation_learning_rate: Learning rate for imitation learning.
        network_config: Dictionary with network configuration.
        num_devices: Number of devices for computation.
        key: Random key used for initialization.

    Returns:
        A tuple containing the networks, the replicated training state, and the policy function.

    """
    network = make_networks(
        observation_size=observation_size,
        action_size=action_size,
        unflatten_fn=env.get_wrapper_attr("features_extractor").unflatten_features,
        rl_learning_rate=rl_learning_rate,
        imitation_learning_rate=imitation_learning_rate,
        network_config=network_config,
    )

    policy_function = make_inference_fn(network)

    key_policy, key_value = jax.random.split(key)

    policy_params = network.policy_network.init(key_policy)
    rl_policy_optimizer_state = network.rl_policy_optimizer.init(policy_params)
    imitation_policy_optimizer_state = network.imitation_policy_optimizer.init(policy_params)
    value_params = network.value_network.init(key_value)
    value_optimizer_state = network.value_optimizer.init(value_params)

    init_params = BCSACNetworkParams(
        policy=policy_params,
        value=value_params,
        target_value=value_params,
    )

    training_state = BCSACTrainingState(
        params=init_params,
        rl_policy_optimizer_state=rl_policy_optimizer_state,
        imitation_policy_optimizer_state=imitation_policy_optimizer_state,
        value_optimizer_state=value_optimizer_state,
        env_steps=0,
        il_gradient_steps=0,
        rl_gradient_steps=0,
    )

    training_state = jax.device_put_replicated(training_state, jax.local_devices()[:num_devices])

    return network, training_state, policy_function


def make_inference_fn(sac_network: BCSACNetworks) -> datatypes.Policy:
    """Create the inference function for the BC-SAC network.

    Args:
        sac_network: The BC-SAC network instance.

    Returns:
        A function that computes the policy output given network parameters.

    """

    def make_policy(params: datatypes.Params, deterministic: bool = False) -> datatypes.Policy:
        def policy(observations: jax.Array, key_sample: jax.Array = None) -> jax.Array:
            logits = sac_network.policy_network.apply(params, observations)

            if deterministic:
                return sac_network.parametric_action_distribution.mode(logits), {}

            return sac_network.parametric_action_distribution.sample(logits, key_sample), {}

        return policy

    return make_policy


def make_networks(
    observation_size: int,
    action_size: int,
    unflatten_fn: callable,
    rl_learning_rate,
    imitation_learning_rate,
    network_config: dict,
) -> BCSACNetworks:
    """Create the networks required for BC-SAC training.

    Args:
        observation_size: Size of the observation space.
        action_size: Size of the action space.
        unflatten_fn: Function to unflatten network outputs.
        rl_learning_rate: Learning rate for RL updates.
        imitation_learning_rate: Learning rate for imitation updates.
        network_config: Network configuration parameters.

    Returns:
        An instance of BCSACNetworks with initialized sub-networks and optimizers.

    """
    if "gaussian" in network_config["action_distribution"]:
        parametric_action_distribution = networks.NormalTanhDistribution(event_size=action_size)
    elif "beta" in network_config["action_distribution"]:
        parametric_action_distribution = networks.BetaDistribution(event_size=action_size)

    output_size = parametric_action_distribution.param_size

    policy_network = networks.make_policy_network(network_config, observation_size, output_size, unflatten_fn)
    value_network = networks.make_value_network(network_config, observation_size, action_size, unflatten_fn)

    rl_policy_optimizer = optax.adam(rl_learning_rate)
    value_optimizer = optax.adam(rl_learning_rate)

    imitation_policy_optimizer = optax.adam(imitation_learning_rate)

    return BCSACNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
        rl_policy_optimizer=rl_policy_optimizer,
        imitation_policy_optimizer=imitation_policy_optimizer,
        value_optimizer=value_optimizer,
    )


def make_rl_sgd_step(bc_sac_network: BCSACNetworks, alpha: float, discount: float, tau: float) -> callable:
    """Create the SGD step function for RL training.

    Args:
        bc_sac_network: Instance of BC-SAC networks.
        alpha: Entropy coefficient.
        discount: Discount factor.
        tau: Target network update coefficient.

    Returns:
        A function that performs one SGD step for reinforcement learning.

    """
    value_loss, policy_loss = _make_rl_losses(bc_sac_network=bc_sac_network, alpha=alpha, discount=discount)

    policy_update = networks.gradient_update_fn(policy_loss, bc_sac_network.rl_policy_optimizer, pmap_axis_name="batch")
    value_update = networks.gradient_update_fn(value_loss, bc_sac_network.value_optimizer, pmap_axis_name="batch")

    def sgd_step(
        carry: tuple[BCSACTrainingState, jax.Array],
        transitions: datatypes.RLTransition,
    ) -> tuple[tuple[BCSACTrainingState, jax.Array], datatypes.Metrics]:
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
        policy_loss, policy_params, rl_policy_optimizer_state = policy_update(
            training_state.params.policy,
            training_state.params.value,
            transitions,
            key_policy,
            optimizer_state=training_state.rl_policy_optimizer_state,
        )

        new_target_value_params = jax.tree_util.tree_map(
            lambda x, y: x * (1 - tau) + y * tau,
            training_state.params.target_value,
            value_params,
        )

        metrics = {"policy_loss": policy_loss, "value_loss": value_loss}

        new_params = BCSACNetworkParams(
            policy=policy_params,
            value=value_params,
            target_value=new_target_value_params,
        )

        training_state = training_state.replace(
            params=new_params,
            rl_policy_optimizer_state=rl_policy_optimizer_state,
            imitation_policy_optimizer_state=training_state.imitation_policy_optimizer_state,
            value_optimizer_state=value_optimizer_state,
            rl_gradient_steps=training_state.rl_gradient_steps + 1,
        )

        return (training_state, key), metrics

    return sgd_step


def make_imitation_sgd_step(bc_sac_network: BCSACNetworks, loss_type: str) -> callable:
    """Create the SGD step function for imitation training.

    Args:
        bc_sac_network: Instance of BC-SAC networks.
        loss_type: Identifier for the loss function to use.

    Returns:
        A function that performs one SGD step for imitation learning.

    """
    policy_loss = _make_imitation_losses(bc_sac_network=bc_sac_network, loss_type=loss_type)
    policy_update = networks.gradient_update_fn(
        policy_loss,
        bc_sac_network.imitation_policy_optimizer,
        pmap_axis_name="batch",
    )

    def sgd_step(
        carry: tuple[BCSACTrainingState, jax.Array],
        transitions: datatypes.RLTransition,
    ) -> tuple[tuple[BCSACTrainingState, jax.Array], datatypes.Metrics]:
        training_state, key = carry
        key, key_policy = jax.random.split(key, 2)

        policy_loss, policy_params, imitation_policy_optimizer_state = policy_update(
            training_state.params.policy,
            transitions,
            key_policy,
            optimizer_state=training_state.imitation_policy_optimizer_state,
        )

        metrics = {"imitation_loss": policy_loss}

        training_state = training_state.replace(
            params=training_state.params.replace(policy=policy_params),
            rl_policy_optimizer_state=training_state.rl_policy_optimizer_state,
            imitation_policy_optimizer_state=imitation_policy_optimizer_state,
            value_optimizer_state=training_state.value_optimizer_state,
            il_gradient_steps=training_state.il_gradient_steps + 1,
        )
        return (training_state, key), metrics

    return sgd_step


def _make_rl_losses(bc_sac_network: BCSACNetworks, alpha: float, discount: float) -> tuple[callable, callable]:
    """Generate the loss functions for RL training.

    Args:
        bc_sac_network: The BC-SAC networks.
        alpha: Entropy coefficient.
        discount: Discount factor.

    Returns:
        A tuple containing the value loss function and the policy loss function.

    """
    policy_network = bc_sac_network.policy_network
    value_network = bc_sac_network.value_network
    parametric_action_distribution = bc_sac_network.parametric_action_distribution

    def value_loss(
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

    def policy_loss(
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

    return value_loss, policy_loss


def _make_imitation_losses(bc_sac_network: BCSACNetworks, loss_type: str) -> callable:
    """Generate the loss function for imitation training.

    Args:
        bc_sac_network: The BC-SAC networks.
        loss_type: Identifier for the loss type.

    Returns:
        A function that computes the imitation loss.

    """
    policy_network = bc_sac_network.policy_network
    parametric_action_distribution = bc_sac_network.parametric_action_distribution

    def policy_loss(
        policy_params: datatypes.Params,
        transitions: datatypes.RLTransition,
        key: jax.Array,
    ) -> jax.Array:
        dist_params = policy_network.apply(policy_params, transitions.observation)
        action = parametric_action_distribution.sample_no_postprocessing(dist_params, key)
        action = parametric_action_distribution.postprocess(action)

        if loss_type == "mse":
            total_loss = ((action - transitions.action) ** 2).mean()
        elif loss_type == "mae":
            total_loss = (abs(action - transitions.action)).mean()
        else:
            raise ValueError(f"Loss type {loss_type} not supported")

        return jnp.mean(total_loss)

    return policy_loss
