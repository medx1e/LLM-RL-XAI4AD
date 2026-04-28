# Copyright 2025 Valeo.


"""Fully connected layer module."""

from .mlp import MLP


FullyConnected = MLP


def get_fully_connected(fc_name: str) -> FullyConnected:
    """Retrieve the fully connected layer class by its name.

    Args:
        fc_name: The identifier name for the fully connected layer.

    Returns:
        A callable class for the specified fully connected layer.

    Raises:
        ValueError: If the provided layer name is unknown.

    """
    encoders = {
        "mlp": MLP,
    }

    try:
        return encoders[fc_name.lower()]
    except KeyError:
        raise ValueError(f"Unknown fully connecter layer: {fc_name}") from None
