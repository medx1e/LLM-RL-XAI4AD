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

"""Roadgraph based datastructures for Waymax based on WOMD."""

import chex
import jax
from jax import numpy as jnp
from waymax.datatypes.roadgraph import RoadgraphPoints


def rotate_rectangle(rectangle: jnp.ndarray, yaw: jnp.ndarray) -> jnp.ndarray:
    """Rotate a rectangle by a specified yaw angle.

    Args:
        rectangle: Array representing rectangle coordinates.
        yaw: Angle of rotation.

    Returns:
        Array of rotated rectangle coordinates.

    """
    rot_matrix = jnp.array(
        [
            [jnp.cos(yaw), -jnp.sin(yaw)],
            [jnp.sin(yaw), jnp.cos(yaw)],
        ],
    )
    return jnp.dot(rectangle, rot_matrix)


def points_in_rectangle(points: jnp.ndarray, rectangle: jnp.ndarray) -> jnp.ndarray:
    """Determine which points lie inside a rectangle.

    Args:
        points: Array of points with shape (N, 2).
        rectangle: Array of rectangle corners in counter-clockwise order (4, 2).

    Returns:
        Boolean array indicating whether each point is inside the rectangle.

    """
    # Get rectangle edges as vectors
    edge1 = rectangle[1] - rectangle[0]  # Top edge
    edge2 = rectangle[3] - rectangle[0]  # Right edge

    # Create basis vectors from edges
    edge1_normalized = edge1 / jnp.linalg.norm(edge1)
    edge2_normalized = edge2 / jnp.linalg.norm(edge2)

    # Convert points to local rectangle coordinates
    points_local = points - rectangle[0]  # Translate to origin

    # Project onto rectangle edges
    proj1 = jnp.dot(points_local, edge1_normalized)
    proj2 = jnp.dot(points_local, edge2_normalized)

    # Check if projections are within rectangle bounds
    in_bounds = jnp.logical_and(
        jnp.logical_and(proj1 >= 0, proj1 <= jnp.linalg.norm(edge1)),
        jnp.logical_and(proj2 >= 0, proj2 <= jnp.linalg.norm(edge2)),
    )

    return in_bounds


def filter_box_roadgraph_points(
    roadgraph: RoadgraphPoints,
    reference_points: jax.Array,
    reference_yaw: jax.Array,
    meters_box: list[int],
    topk: int,
) -> RoadgraphPoints:
    """Filter roadgraph points within a given bounding box and return the top-k closest.

    The function rotates and translates a bounding box around a reference point,
    then selects and returns the closest roadgraph points inside the box.

    Args:
        roadgraph: RoadgraphPoints data structure.
        reference_points: Array with the reference coordinates.
        reference_yaw: Yaw angle for rotation.
        meters_box: Dictionary specifying box dimensions.
        topk: Number of top closest points to return.

    Returns:
        A filtered RoadgraphPoints structure.

    """
    chex.assert_equal_shape_prefix([roadgraph, reference_points], reference_points.ndim - 1)
    chex.assert_equal(len(roadgraph.shape), reference_points.ndim)
    chex.assert_equal(reference_points.shape[-1], 2)

    reference_box = jnp.array(
        [
            [-meters_box["back"], -meters_box["right"]],
            [meters_box["front"], -meters_box["right"]],
            [meters_box["front"], meters_box["left"]],
            [-meters_box["back"], meters_box["left"]],
        ],
    )

    rotated_box = rotate_rectangle(reference_box, reference_yaw).squeeze()
    translated_box = rotated_box + reference_points

    distances = jnp.linalg.norm(reference_points[..., jnp.newaxis, :] - roadgraph.xy, axis=-1)

    roadgraph.valid = points_in_rectangle(roadgraph.xy, translated_box)
    valid_distances = jnp.where(roadgraph.valid, distances, float("inf"))
    _, top_idx = jax.lax.top_k(-valid_distances, topk)

    # Rearrange the idx to respect the original order # NOTE: Review this
    _idx = jnp.argsort(top_idx, axis=-1)
    top_idx = jnp.take_along_axis(top_idx, _idx, axis=-1)

    stacked = jnp.stack(
        [
            roadgraph.x,
            roadgraph.y,
            roadgraph.z,
            roadgraph.dir_x,
            roadgraph.dir_y,
            roadgraph.dir_z,
            roadgraph.types,
            roadgraph.ids,
            roadgraph.valid,
        ],
        axis=-1,
        dtype=jnp.float32,
    )
    filtered = jnp.take_along_axis(stacked, top_idx[..., None], axis=-2)

    return RoadgraphPoints(
        x=filtered[..., 0],
        y=filtered[..., 1],
        z=filtered[..., 2],
        dir_x=filtered[..., 3],
        dir_y=filtered[..., 4],
        dir_z=filtered[..., 5],
        types=filtered[..., 6].astype(jnp.int32),
        ids=filtered[..., 7].astype(jnp.int32),
        valid=filtered[..., 8].astype(jnp.bool_),
    )


def filter_topk_roadgraph_points(roadgraph: RoadgraphPoints, reference_points: jax.Array, topk: int) -> RoadgraphPoints:
    """Return the top-k closest roadgraph points to a reference point.

    Args:
        roadgraph: RoadgraphPoints data structure.
        reference_points: Array with reference coordinates.
        topk: Number of points to return.

    Returns:
        A RoadgraphPoints structure containing the top-k closest points.

    """
    chex.assert_equal_shape_prefix([roadgraph, reference_points], reference_points.ndim - 1)
    chex.assert_equal(len(roadgraph.shape), reference_points.ndim)
    chex.assert_equal(reference_points.shape[-1], 2)

    if topk > roadgraph.num_points:
        raise NotImplementedError("Not enough points in roadgraph.")
    elif topk < roadgraph.num_points:
        distances = jnp.linalg.norm(reference_points[..., jnp.newaxis, :] - roadgraph.xy, axis=-1)
        valid_distances = jnp.where(roadgraph.valid, distances, float("inf"))
        _, top_idx = jax.lax.top_k(-valid_distances, topk)

        # Rearrange the idx to respect the original order # NOTE: Review this
        _idx = jnp.argsort(top_idx, axis=-1)
        top_idx = jnp.take_along_axis(top_idx, _idx, axis=-1)

        stacked = jnp.stack(
            [
                roadgraph.x,
                roadgraph.y,
                roadgraph.z,
                roadgraph.dir_x,
                roadgraph.dir_y,
                roadgraph.dir_z,
                roadgraph.types,
                roadgraph.ids,
                roadgraph.valid,
            ],
            axis=-1,
            dtype=jnp.float32,
        )
        filtered = jnp.take_along_axis(stacked, top_idx[..., None], axis=-2)

        return RoadgraphPoints(
            x=filtered[..., 0],
            y=filtered[..., 1],
            z=filtered[..., 2],
            dir_x=filtered[..., 3],
            dir_y=filtered[..., 4],
            dir_z=filtered[..., 5],
            types=filtered[..., 6].astype(jnp.int32),
            ids=filtered[..., 7].astype(jnp.int32),
            valid=filtered[..., 8].astype(jnp.bool_),
        )
    else:
        return roadgraph
