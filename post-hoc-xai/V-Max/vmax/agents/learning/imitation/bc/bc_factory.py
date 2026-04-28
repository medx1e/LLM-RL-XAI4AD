# Copyright 2025 Valeo.

"""Factory functions for the Behavioral Cloning (BC) algorithm."""

from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from vmax.agents import datatypes, networks


@flax.struct.dataclass
class BCNetworkParams:
    """Parameters for BC network."""

    policy: datatypes.Params


@flax.struct.dataclass
class BCNetwork:
    """BC network."""

    policy_network: Any
    optimizer: Any


@flax.struct.dataclass
class BCTrainingState(datatypes.TrainingState):
    """Training state for BC algorithm."""

    params: BCNetworkParams
    optimizer_state: optax.OptState
    il_gradient_steps: int


def initialize(
    action_size: int,
    observation_size: int,
    env: Any,
    learning_rate: float,
    network_config: dict,
    num_devices: int,
    key: jax.Array,
) -> tuple[BCNetwork, BCTrainingState, datatypes.Policy]:
    """Initialize BC components.

    Args:
        action_size: Size of the action space.
        observation_size: Size of the observation space.
        env: Environment instance with a features extractor.
        learning_rate: Learning rate for the optimizer.
        network_config: Network configuration dictionary.
        num_devices: Number of devices to use.
        key: Random key for initialization.

    Returns:
        A tuple of (network, training_state, policy_function).

    """
    network = make_networks(
        observation_size=observation_size,
        action_size=action_size,
        unflatten_fn=env.get_wrapper_attr("features_extractor").unflatten_features,
        learning_rate=learning_rate,
        network_config=network_config,
    )

    policy_function = make_inference_fn(network)

    policy_params = network.policy_network.init(key)
    init_params = BCNetworkParams(policy=policy_params)
    optimizer_state = network.optimizer.init(init_params.policy)

    training_state = BCTrainingState(
        params=init_params,
        optimizer_state=optimizer_state,
        env_steps=0,
        il_gradient_steps=0,
    )

    training_state = jax.device_put_replicated(training_state, jax.local_devices()[:num_devices])

    return network, training_state, policy_function


def make_inference_fn(bc_network: BCNetwork) -> datatypes.Policy:
    """Create the inference function for the BC network.

    Args:
        bc_network: Instance of BCNetwork.

    Returns:
        A callable policy function.

    """

    def make_policy(params: datatypes.Params, deterministic: bool = False) -> datatypes.Policy:
        policy_network = bc_network.policy_network

        def policy(observations: jax.Array, key_sample: jax.Array = None) -> jax.Array:
            logits = policy_network.apply(params, observations)

            return logits, {}

        return policy

    return make_policy


def make_networks(
    observation_size: int,
    action_size: int,
    unflatten_fn: callable,
    learning_rate: int,
    network_config: dict,
) -> BCNetwork:
    """Construct the BC network.

    Args:
        observation_size: Size of the observation space.
        action_size: Size of the action space.
        unflatten_fn: Function to unflatten network inputs.
        learning_rate: Learning rate used for the optimizer.
        network_config: Network configuration dictionary.

    Returns:
        An instance of BCNetwork.

    """
    policy_network = networks.make_policy_network(network_config, observation_size, action_size, unflatten_fn)
    optimizer = optax.adam(learning_rate=learning_rate)

    return BCNetwork(policy_network=policy_network, optimizer=optimizer)


def make_sgd_step(bc_network: BCNetwork, loss_type: str) -> datatypes.LearningFunction:
    """Create the SGD step function for BC training.

    Args:
        bc_network: Instance of BCNetwork.
        loss_type: Identifier for the loss type.

    Returns:
        A function that executes an SGD step.

    """
    policy_loss = _make_loss_fn(bc_network, loss_type)
    policy_update = networks.gradient_update_fn(policy_loss, bc_network.optimizer, pmap_axis_name="batch")

    def sgd_step(
        carry: tuple[BCTrainingState, jax.Array],
        transitions: datatypes.RLTransition,
    ) -> tuple[tuple[BCTrainingState, jax.Array], datatypes.Metrics]:
        training_state, _key = carry

        policy_loss, policy_params, optimizer_state = policy_update(
            training_state.params.policy,
            transitions,
            optimizer_state=training_state.optimizer_state,
        )

        sgd_metrics = {"imitation_loss": policy_loss}

        params = BCNetworkParams(policy=policy_params)

        training_state = training_state.replace(
            params=params,
            optimizer_state=optimizer_state,
            il_gradient_steps=training_state.il_gradient_steps + 1,
        )

        return (training_state, _key), sgd_metrics

    return sgd_step


def _make_loss_fn(bc_network: BCNetwork, loss_type: str):
    """Generate the loss function for BC training.

    Args:
        bc_network: Instance of BCNetwork.
        loss_type: Identifier for the loss type.

    Returns:
        A function that computes the loss given network parameters and transitions.

    """
    policy_network = bc_network.policy_network

    def compute_policy_loss(policy_params: datatypes.Params, transitions: datatypes.RLTransition) -> jax.Array:
        action = policy_network.apply(policy_params, transitions.observation)

        if loss_type == "mse":
            policy_loss = ((action - transitions.action) ** 2).mean()
        elif loss_type == "mae":
            policy_loss = (abs(action - transitions.action)).mean()
        else:
            raise ValueError(f"Loss type {loss_type} not supported.")

        return jnp.mean(policy_loss)

    return compute_policy_loss
