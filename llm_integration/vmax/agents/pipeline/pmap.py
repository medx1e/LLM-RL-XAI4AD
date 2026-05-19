# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parallel map utility functions."""

import functools
from typing import Any

import jax
import jax.numpy as jnp


def unpmap(v):
    """Unreplicate a pytree by extracting the first element of each device replica.

    Args:
        v: A pytree of arrays with device replicas.

    Returns:
        A pytree where each leaf is the first replica.

    """
    return jax.tree_util.tree_map(lambda x: x[0], v)


def flatten_tree(v):
    """Flatten each array in a pytree.

    Args:
        v: A pytree of arrays.

    Returns:
        A pytree with each array flattened.

    """
    return jax.tree_util.tree_map(lambda x: x.flatten(), v)


def bcast_local_devices(value, local_devices_to_use=1):
    """Broadcast a value to a subset of local devices.

    Args:
        value: The value to broadcast.
        local_devices_to_use: Number of local devices to use (default is 1).

    Returns:
        The value replicated across the selected local devices.

    """
    devices = jax.local_devices()[:local_devices_to_use]
    return jax.device_put_replicated(value, devices)


def synchronize_hosts() -> None:
    """Synchronize computation across hosts."""
    if jax.process_count() == 1:
        return

    # Make sure all processes stay up until the end of main.
    x = jnp.ones([jax.local_device_count()])
    x = jax.device_get(jax.pmap(lambda x: jax.lax.psum(x, "i"), "i")(x))

    assert x[0] == jax.device_count()


def _fingerprint(x: Any) -> float:
    """Compute a fingerprint by summing the values in a pytree.

    Args:
        x: A pytree of arrays.

    Returns:
        The computed fingerprint as a float.

    """
    sums = jax.tree_util.tree_map(jnp.sum, x)
    return jax.tree_util.tree_reduce(lambda x, y: x + y, sums)


def is_replicated(x: Any, axis_name: str) -> jax.Array:
    """Check if a pytree is replicated across devices.

    Should be called inside a function decorated with pmap.

    Args:
        x: A pytree to check.
        axis_name: The pmap axis name.

    Returns:
        A boolean array indicating if x is replicated.

    """
    fp = _fingerprint(x)
    return jax.lax.pmin(fp, axis_name=axis_name) == jax.lax.pmax(fp, axis_name=axis_name)


def assert_is_replicated(x: Any, debug: Any = None):
    """Assert that a pytree is replicated across devices.

    This function should be called from non-jitted code.

    Args:
        x: A pytree to check.
        debug: Optional debug message for failure.

    """
    f = functools.partial(is_replicated, axis_name="i")
    assert jax.pmap(f, axis_name="i")(x)[0], debug
