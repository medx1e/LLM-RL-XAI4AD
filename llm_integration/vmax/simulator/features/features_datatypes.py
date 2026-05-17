# Copyright 2025 Valeo.


"""Module defining feature datatypes."""

from collections.abc import Sequence
from dataclasses import field

import chex
import jax
import jax.numpy as jnp
from matplotlib.patches import Rectangle


@chex.dataclass
class ObjectFeatures:
    """Dataclass representing features of dynamic objects in the environment."""

    field_names: Sequence[str]  # Field names
    xy: jax.Array = field(default_factory=lambda: jnp.array(()))  # Position coordinates over time
    vel_xy: jax.Array = field(default_factory=lambda: jnp.array(()))  # Velocity components over time
    yaw: jax.Array = field(default_factory=lambda: jnp.array(()))  # Yaw angles over time
    length: jax.Array = field(default_factory=lambda: jnp.array(()))  # Object lengths
    width: jax.Array = field(default_factory=lambda: jnp.array(()))  # Object widths
    object_types: jax.Array = field(default_factory=lambda: jnp.array(()))  # Object types
    valid: jax.Array = field(default_factory=lambda: jnp.array(()))  # Validity mask over time

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the data array."""
        return self.stack_fields().shape

    @property
    def batch_dims(self) -> tuple[int, ...]:
        """Return the dimensions of the batch, excluding the last dimension."""
        return self.shape[:-1]

    @property
    def num_objects(self) -> int:
        """Return the number of objects."""
        return self.shape[0]

    @property
    def num_past_observation(self) -> int:
        """Return the number of past observations."""
        return self.shape[1]

    def stack_fields(self) -> jax.Array:
        """Return a concatenated version of a set of field names for Trajectory."""
        if len(self.field_names) == 0:
            return jnp.array(())

        return jnp.concatenate([getattr(self, field_name) for field_name in self.field_names], axis=-1)

    def plot(self, ax) -> None:
        """Plot the object features."""
        # 1. Plot objects trajectories and bbox
        plot_bbox = "length" in self.field_names and "width" in self.field_names

        for i in range(self.num_objects):
            valid = self.valid[i, -1, 0]

            if not valid:
                continue

            trajectory = self.xy[i]
            valid = self.valid[i][:, 0]
            trajectory = trajectory[valid]

            if trajectory.shape[0] == 0:
                continue

            color = "lightpink" if i == 0 else "lightskyblue"

            # Plot trajectory
            ax.plot(trajectory[:, 0], trajectory[:, 1], color=color)

            # Plot bbox
            if plot_bbox:
                length = self.length[i][-1][0]
                width = self.width[i][-1][0]
                angle = self.yaw[i][-1][0] * 180 / jnp.pi
                xy = trajectory[-1] - jnp.array([length / 2, width / 2])
                ax.add_patch(
                    Rectangle(
                        xy,
                        length,
                        width,
                        angle=angle,
                        edgecolor=color,
                        facecolor="none",
                        rotation_point="center",
                    ),
                )


@chex.dataclass
class RoadgraphFeatures:
    """Dataclass representing features of the road graph."""

    field_names: Sequence[str]  # Field names
    xy: jax.Array = field(default_factory=lambda: jnp.array(()))  # Position coordinates
    dir_xy: jax.Array = field(default_factory=lambda: jnp.array(()))  # Direction vectors
    types: jax.Array = field(default_factory=lambda: jnp.array(()))  # Types
    ids: jax.Array = field(default_factory=lambda: jnp.array(()))  # Unique identifiers
    valid: jax.Array = field(default_factory=lambda: jnp.array(()))  # Validity mask

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the data array."""
        return self.stack_fields().shape

    def stack_fields(self) -> jax.Array:
        """Return a concatenated version of a set of field names for Trajectory."""
        if len(self.field_names) == 0:
            return jnp.array(())

        return jnp.concatenate([getattr(self, field_name) for field_name in self.field_names], axis=-1)

    def plot(self, ax) -> None:
        """Plot the roadgraph features."""
        if len(self.field_names) == 0:
            return

        roadgraph_xy = self.xy[self.valid[..., 0]]
        roadgraph_xy = roadgraph_xy.reshape(-1, 2)
        ax.plot(roadgraph_xy[:, 0], roadgraph_xy[:, 1], ".", color="grey", ms=2)


@chex.dataclass
class TrafficLightFeatures:
    """Dataclass representing features of traffic lights."""

    field_names: Sequence[str]  # Field names
    xy: jax.Array = field(default_factory=lambda: jnp.array(()))  # Position coordinates
    state: jax.Array = field(default_factory=lambda: jnp.array(()))  # Traffic light state
    ids: jax.Array = field(default_factory=lambda: jnp.array(()))  # Unique identifiers
    valid: jax.Array = field(default_factory=lambda: jnp.array(()))  # Validity mask

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the data array."""
        return self.stack_fields().shape

    @property
    def num_traffic_lights(self) -> int:
        """Return the number of traffic lights."""
        return self.shape[0]

    def stack_fields(self) -> jax.Array:
        """Return a concatenated version of a set of field names for Trajectory."""
        if len(self.field_names) == 0:
            return jnp.array(())

        return jnp.concatenate([getattr(self, field_name) for field_name in self.field_names], axis=-1)

    def plot(self, ax) -> None:
        """Plot the traffic light features."""
        for i in range(self.num_traffic_lights):
            if not jnp.all(self.valid[i]):
                continue

            xy = self.xy[i][-1]
            # NOTE: Unknow state is 0, maybe bug in plot
            tl_state = jnp.argmax(self.state[i][-1]) + 1

            if tl_state in [1, 4, 7]:
                ax.plot(xy[0], xy[1], "o", ms=3, color="red")
            elif tl_state in [2, 5, 8]:
                ax.plot(xy[0], xy[1], "o", ms=3, color="yellow")
            elif tl_state in [3, 6]:
                ax.plot(xy[0], xy[1], "o", ms=3, color="green")


@chex.dataclass
class PathTargetFeatures:
    """Dataclass representing features of path targets."""

    xy: jax.Array = field(default_factory=lambda: jnp.array(()))  # Position coordinates
    valid: jax.Array = field(default_factory=lambda: jnp.array(()))  # Validity mask NOTE: Not used

    @property
    def data(self) -> jax.Array:
        """Return the position coordinates."""
        return self.xy

    def plot(self, ax) -> None:
        """Plot the path target features."""
        ax.plot(self.xy[:, 0], self.xy[:, 1], "o", ms=3, color="salmon")
