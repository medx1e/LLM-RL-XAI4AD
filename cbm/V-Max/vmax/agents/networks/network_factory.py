# Copyright 2025 Valeo.


"""Factory functions for creating policy and value networks."""

import dataclasses
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from vmax.agents.networks import decoders, encoders, network_utils


@dataclasses.dataclass
class Network:
    """Container for network initialization and application functions."""

    init: Callable[..., Any]
    apply: Callable[..., Any]


class PolicyNetwork(nn.Module):
    """Policy network module that builds the forward propagation path."""

    encoder_layer: encoders.Encoder | None = None
    fully_connected_layer: decoders.FullyConnected | None = None
    final_activation: Callable | None = None

    output_size: int = 1

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        """Compute the forward pass for the policy network.

        Args:
            obs: The observation tensor.

        Returns:
            The network's output tensor.

        """
        x = self.encoder_layer(obs) if self.encoder_layer is not None else obs
        x = self.fully_connected_layer(x)
        x = nn.Dense(self.output_size)(x)

        if self.final_activation:
            x = self.final_activation(x)

        return x


class ValueNetwork(nn.Module):
    """Value network module supporting multiple parallel networks."""

    encoder_layer: encoders.Encoder | None = None
    fully_connected_layer: decoders.FullyConnected | None = None
    final_activation: Callable | None = None

    output_size: int = 1
    num_networks: int = 1
    shared_encoder: bool = False

    @nn.compact
    def __call__(self, obs: jax.Array, actions: jax.Array | None = None) -> jax.Array:
        """Run the forward pass over one or multiple networks.

        Args:
            obs: The observation tensor.
            actions: An optional action tensor to concatenate.

        Returns:
            A tensor with concatenated outputs from each network.

        """
        shared_encoder = self.shared_encoder and self.encoder_layer is not None

        if self.shared_encoder and self.encoder_layer is not None:
            obs = self.encoder_layer(obs)

        out = []
        for _ in range(self.num_networks):
            x = obs
            if not shared_encoder and self.encoder_layer is not None:
                x = self.encoder_layer(x)

            x = jnp.concatenate([x, actions], axis=-1) if actions is not None else x
            x = self.fully_connected_layer(x)
            x = nn.Dense(self.output_size)(x)

            if self.final_activation:
                x = self.final_activation(x)

            out.append(x)

        return jnp.concatenate(out, axis=-1)


def _build_encoder_layer(encoder_config: dict, unflatten_fn) -> encoders.Encoder | None:
    """Build the encoder layer from its configuration.

    Args:
        encoder_config: A dictionary containing the encoder settings.
        unflatten_fn: The function used to unflatten inputs.

    Returns:
        An encoder layer or None if the type is set to 'none'.

    """
    encoder_type = encoder_config["type"]

    if encoder_type == "none":
        return None

    encoder_config = network_utils.parse_config(encoder_config, "encoder")
    encoder = encoders.get_encoder(encoder_type)

    return encoder(unflatten_fn, **encoder_config)


def _build_fc_layer(
    config: dict,
    keys_to_remove: list[str] = ("final_activation", "num_networks", "shared_encoder"),
) -> decoders.FullyConnected:
    """Construct the fully connected layer using the provided configuration.

    Args:
        config: Configuration settings for the fully connected layer.
        keys_to_remove: Specific keys to exclude from the configuration.

    Returns:
        A fully connected layer instance.

    """
    value_type = config["type"]
    value_config = network_utils.parse_config(config, keys_to_remove=keys_to_remove)

    return decoders.get_fully_connected(value_type)(**value_config)


def _assemble_policy_network(
    encoder_layer: encoders.Encoder,
    policy_fc_layer: decoders.FullyConnected,
    final_activation: Callable | None,
    obs_size: int,
    output_size: int,
) -> Network:
    """Combine encoder and FC layers into a policy network.

    Args:
        encoder_layer: The encoder layer.
        policy_fc_layer: The fully connected layer for policy.
        final_activation: The activation function applied at the final layer.
        obs_size: The dimension of the observation tensor.
        output_size: The dimension of the output tensor.

    Returns:
        A Network containing initialization and application functions for the policy.

    """
    policy_net = PolicyNetwork(
        encoder_layer=encoder_layer,
        fully_connected_layer=policy_fc_layer,
        output_size=output_size,
        final_activation=final_activation,
    )

    def apply(policy_params, obs):
        return policy_net.apply(policy_params, obs)

    dummy_obs = jnp.zeros((1, obs_size))

    return Network(init=lambda key: policy_net.init(key, dummy_obs), apply=apply)


def _assemble_value_network(
    encoder_layer: encoders.Encoder,
    value_fc_layer: decoders.FullyConnected,
    final_activation: Callable | None,
    obs_size: int,
    output_size: int,
    num_networks: int,
    shared_encoder: bool,
    concat_obs_action: bool,
) -> Network:
    """Combine encoder and FC layers into a value network.

    Args:
        encoder_layer: The encoder layer.
        value_fc_layer: The fully connected layer for value computation.
        final_activation: The activation function after the final layer.
        obs_size: The observation dimension.
        output_size: The action/output dimension.
        num_networks: The number of networks to instantiate.
        shared_encoder: Whether to share the encoder across networks.
        concat_obs_action: Whether to concatenate observation and action.

    Returns:
        A Network with initialization and application functions for the value network.

    """
    value_net = ValueNetwork(
        encoder_layer=encoder_layer,
        fully_connected_layer=value_fc_layer,
        final_activation=final_activation,
        num_networks=num_networks,
        shared_encoder=shared_encoder,
    )

    def apply(value_params, obs, actions=None):
        return value_net.apply(value_params, obs, actions)

    dummy_obs = jnp.zeros((1, obs_size))
    dummy_action = jnp.zeros((1, output_size)) if concat_obs_action else None

    return Network(init=lambda key: value_net.init(key, dummy_obs, dummy_action), apply=apply)


def make_policy_network(config: dict, obs_size: int, output_size: int, unflatten_fn) -> tuple[Network, Network | None]:
    """Create a policy network based on the provided configuration.

    Args:
        config: A dictionary with configuration parameters.
        obs_size: The size of the observation.
        output_size: The size of the output.
        unflatten_fn: Function to handle input unflattening.

    Returns:
        A Network instance representing the policy.

    """
    _config = network_utils.convert_to_dict_with_activation_fn(config)
    policy_config = _config.get("policy")

    encoder_layer = _build_encoder_layer(_config["encoder"], unflatten_fn)
    policy_fc_layer = _build_fc_layer(policy_config)

    policy_network = _assemble_policy_network(
        encoder_layer,
        policy_fc_layer,
        policy_config["final_activation"],
        obs_size,
        output_size,
    )

    return policy_network


def make_value_network(
    config: dict,
    obs_size: int,
    output_size: int,
    unflatten_fn: Callable,
    concat_obs_action: bool = True,
) -> tuple[Network, Network | None]:
    """Create a value network based on the provided configuration.

    Args:
        config: A dictionary with configuration parameters.
        obs_size: The size of the observation.
        output_size: The size of the output.
        unflatten_fn: Function to handle input unflattening.
        concat_obs_action: Whether to concatenate observation and action inputs.

    Returns:
        A Network instance representing the value network.

    """
    _config = network_utils.convert_to_dict_with_activation_fn(config)
    value_config = _config.get("value")

    encoder_layer = _build_encoder_layer(_config["encoder"], unflatten_fn)
    value_fc_layer = _build_fc_layer(value_config)

    value_network = _assemble_value_network(
        encoder_layer,
        value_fc_layer,
        value_config["final_activation"],
        obs_size,
        output_size,
        value_config["num_networks"],
        value_config["shared_encoder"],
        concat_obs_action=concat_obs_action,
    )

    return value_network
