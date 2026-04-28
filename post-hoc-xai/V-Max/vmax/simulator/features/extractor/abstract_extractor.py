# Copyright 2025 Valeo.


"""Abstract base module for feature extractors."""

from abc import ABC, abstractmethod

import jax
import matplotlib as mpl
from waymax import datatypes


class AbstractFeaturesExtractor(ABC):
    """Abstract interface for feature extractors."""

    @abstractmethod
    def extract_features(self, state: datatypes.SimulatorState) -> tuple[jax.Array, ...]:
        """Extract features from a simulator state.

        Args:
            state: The current simulator state.

        Returns:
            A tuple containing the extracted features.

        """

    @abstractmethod
    def unflatten_features(self, vectorized_obs: jax.Array) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
        """Convert a flat observation array back into feature tensors and masks.

        Args:
            vectorized_obs: The vectorized observation.

        Returns:
            A tuple of feature tensors and their corresponding masks.

        """

    @abstractmethod
    def plot_features(self, state: datatypes.SimulatorState, ax: mpl.axes.Axes) -> None:
        """Plot extracted features on a matplotlib Axes.

        Args:
            state: The simulator state.
            ax: The matplotlib axes object.

        """
