# Copyright 2025 Valeo.

"""Utility functions for network modules."""

from typing import Any

from flax import linen as nn


def parse_config(config: dict[str, Any], keys_to_remove: list[str] | None = None) -> dict[str, Any]:
    """Parse a configuration dictionary.

    Args:
        config: The configuration dictionary.
        keys_to_remove: A list of keys to remove from the dictionary.

    Returns:
        A new dictionary with the unwanted keys removed.

    """
    return {k: v for k, v in config.items() if k != "type" and k not in keys_to_remove}


def convert_to_dict_with_activation_fn(config: dict[str, Any]) -> dict[str, Any]:
    """Convert configuration entries to include activation functions if applicable.

    Args:
        config: The original configuration dictionary.

    Returns:
        A new dictionary where any key with 'activation' applies the corresponding function.

    """

    def convert_value(value: Any) -> Any:
        if isinstance(value, dict):
            return convert_to_dict_with_activation_fn(value)
        return value

    return {
        key: get_activation_fn(value) if "activation" in key else convert_value(value) for key, value in config.items()
    }


def get_activation_fn(activation_type: str | None) -> nn.activation:
    """Retrieve the activation function corresponding to the given activation type.

    Args:
        activation_type: A string that specifies which activation function to use.

    Returns:
        The activation function if found; otherwise, None.

    """
    activation_map = {
        "relu": nn.relu,
        "tanh": nn.tanh,
        "sigmoid": nn.sigmoid,
        "leaky_relu": nn.leaky_relu,
        "softplus": nn.softplus,
        "softmax": nn.softmax,
    }

    return activation_map.get(activation_type)
