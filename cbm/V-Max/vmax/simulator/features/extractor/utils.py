# Copyright 2025 Valeo.


"""Utility functions and constants for feature extractors."""

import jax
import jax.numpy as jnp


MAX_METERS = 50  # m
MAX_SPEED = 30  # m/s

# https://waymo.com/open/data/motiontfexample.
# LANE_UNDEFINED = 0 -> 0
# LANE_FREEWAY = 1 -> 1
# LANE_SURFACE_STREET = 2 -> 2
# LANE_BIKE_LANE = 3 -> 3
# # Original definition skips 4.
# ROAD_LINE_UNKNOWN = 5 -> 0
# ROAD_LINE_BROKEN_SINGLE_WHITE = 6 -> 4
# ROAD_LINE_SOLID_SINGLE_WHITE = 7 -> 4
# ROAD_LINE_SOLID_DOUBLE_WHITE = 8 -> 4
# ROAD_LINE_BROKEN_SINGLE_YELLOW = 9 -> 4
# ROAD_LINE_BROKEN_DOUBLE_YELLOW = 10 -> 4
# ROAD_LINE_SOLID_SINGLE_YELLOW = 11 -> 4
# ROAD_LINE_SOLID_DOUBLE_YELLOW = 12 -> 4
# ROAD_LINE_PASSING_DOUBLE_YELLOW = 13 -> 4
# ROAD_EDGE_UNKNOWN = 14 -> 0
# ROAD_EDGE_BOUNDARY = 15 -> 5
# ROAD_EDGE_MEDIAN = 16 -> 5
# STOP_SIGN = 17 -> 6
# CROSSWALK = 18 -> 7
# SPEED_BUMP = 19 -> 8
# UNKNOWN = -1 -> 0
RG_MAPPING = (0, 1, 2, 3, 0, 4, 4, 4, 4, 4, 4, 4, 4, 0, 5, 5, 6, 7, 8, 0)

# UNKNOWN = 0
# ARROW_STOP = 1
# ARROW_CAUTION = 2
# ARROW_GO = 3
# STOP = 4
# CAUTION = 5
# GO = 6
# FLASHING_STOP = 7
# FLASHING_CAUTION = 8
TL_MAPPING = tuple(range(9))

# UNSET = 0
# VEHICLE = 1
# PEDESTRIAN = 2
# CYCLIST = 3
# OTHER = 4
OBJECT_MAPPING = tuple(range(5))


def get_feature_size(feature_key: str, dict_mapping: dict) -> int:
    """Get the size of the feature.

    Args:
        feature_key: the feature key
        dict_mapping: the dictionary mapping
    Returns:
        the size of the feature

    """
    if feature_key in ["xy", "vel_xy", "dir_xy"]:
        return 2
    elif feature_key in ["speed", "length", "width", "height", "valid", "yaw", "arc_length"]:
        return 1
    elif feature_key == "state":
        return max(dict_mapping["state"])
    elif feature_key == "types":
        return max(dict_mapping["types"])
    elif feature_key == "object_types":
        return max(dict_mapping["object_types"])
    else:
        raise ValueError(f"Feature {feature_key} not supported")


def normalize_path(x: jax.Array, meters: int) -> jax.Array:
    """Normalize the path by the meters.

    Args:
        x: the path to normalize.
        meters: the meters.

    Returns:
        Normalized path.

    """
    x = jnp.clip(x, min=-meters, max=meters)
    x = x / meters

    return x


def normalize_by_feature(data: jax.Array, feature_key: str, meters: int, dict_mapping: dict) -> jax.Array:
    """Normalize the data by the feature key.

    Args:
        data: the data to normalize
        feature_key: the feature key
        meters: the meters
        dict_mapping: the dictionary mapping
    Returns:
        normalized data

    """
    if feature_key == "xy":
        data = normalize_path(data, meters)
    elif feature_key == "state":  # trafficlight
        data = onehot_encoder(data, dict_mapping["state"])
    elif feature_key == "types":  # roadgraph
        data = onehot_encoder(data, dict_mapping["types"])
    elif feature_key == "object_types":  # objects
        data = onehot_encoder(data, dict_mapping["object_types"])
    elif feature_key == "vel_xy":
        data = jnp.clip(data, min=0, max=MAX_SPEED)
        data = data / MAX_SPEED  # m/s
    elif feature_key in ["length", "width", "height"]:
        data = data / meters  # m
    elif feature_key in ["valid", "yaw", "arc_length", "dir_xy", "ids"]:
        pass
    else:
        raise ValueError(f"Feature {feature_key} not supported")

    return data


def onehot_encoder(types: jax.Array, mapping: tuple[int]) -> jax.Array:
    """One-hot encoder for the type of objects.

    Args:
        types: the type of objects
        mapping: the mapping of the types
    Returns:
        one-hot encoded type

    """
    # Map the provided types using the given mapping.
    mapped = jnp.take(jnp.array(mapping), types, axis=-1)
    # Optionally, if 0 is reserved for an "unknown" type, then we build one more slot and drop it afterwards.
    onehot = jax.nn.one_hot(mapped, max(mapping) + 1, axis=-1)
    # Drop the first "unknown" column. The result will have a size of max_val
    return onehot[..., 1:]
