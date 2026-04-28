# Copyright 2023 The Waymax Authors.
#
# Licensed under the Waymax License Agreement for Non-commercial Use
# Use (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     https://github.com/waymo-research/waymax/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Customized route paths data structure for Waymax datatypes."""

import chex
import jax
from jax import numpy as jnp
from waymax import datatypes


def filter_sdc_paths(sdc_paths: datatypes.Paths, reference_points: jax.Array, num_points: int) -> datatypes.Paths:
    """Returns the `num_points` ahead of the reference point in the SDC paths.

    Args:
        sdc_paths: Paths data structure.
        reference_points: A tensor of shape (..., 2) - the reference point used to
            measure distance.
        num_points: The number of points to return.

    Returns:
        A Paths data structure with the `num_points` ahead of the reference point.

    """
    chex.assert_equal_shape_prefix([sdc_paths, reference_points], reference_points.ndim - 1)
    chex.assert_equal(len(sdc_paths.shape), reference_points.ndim)
    chex.assert_equal(reference_points.shape[-1], 2)

    if num_points > sdc_paths.num_points_per_path:
        raise NotImplementedError("Not enough points in roadgraph.")
    elif num_points < sdc_paths.num_points_per_path:
        distances = jnp.linalg.norm(reference_points[..., jnp.newaxis, :] - sdc_paths.xy, axis=-1)
        sdc_idx = jnp.argmin(distances, axis=-1)
        indices = jnp.arange(num_points) + sdc_idx + 1

        stacked = jnp.stack(
            [
                sdc_paths.x,
                sdc_paths.y,
                sdc_paths.z,
                sdc_paths.ids,
                sdc_paths.valid,
                sdc_paths.arc_length,
            ],
            axis=-1,
            dtype=jnp.float32,
        )
        filtered = jnp.take_along_axis(stacked, indices[None, ..., None], axis=-2, fill_value=-1.0)

        return datatypes.Paths(
            x=filtered[..., 0],
            y=filtered[..., 1],
            z=filtered[..., 2],
            ids=filtered[..., 3],
            valid=filtered[..., 4],
            arc_length=filtered[..., 5],
            on_route=sdc_paths.on_route,
        )
    else:
        return sdc_paths
