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

"""Metrics relating to route."""

from jax import numpy as jnp
from waymax import datatypes
from waymax.metrics import abstract_metric

from vmax.simulator.metrics import utils


class OffRouteMetric(abstract_metric.AbstractMetric):
    """Off-route metric for the SDC.

    The SDC is considered off-route either if 1) it is farther than
    MAX_DISTANCE_TO_ROUTE_PATH from the closest on-route path, or 2) it is farther
    from the closest on-route path than the closest off-route path by
    MAX_RELATIVE_DISTANCE_TO_OFF_ROUTE_PATH.

    If the SDC is on-route, the SDC trajectory is completely invalid, or there are
    no valid on-route paths, it returns zero.

    If the SDC is off-route, this metric returns the distance to the closest valid
    on-route path. If there are no valid on-route paths, it returns the distance
    to the closest valid off-route path.
    """

    MAX_DISTANCE_TO_ROUTE_PATH = utils.LANE_WIDTH / 2 + utils.MARGIN  # Meters.
    MAX_RELATIVE_DISTANCE_TO_OFF_ROUTE_PATH = MAX_DISTANCE_TO_ROUTE_PATH / 2 + utils.MARGIN  # Meters.

    def compute(self, simulator_state: datatypes.SimulatorState) -> abstract_metric.MetricResult:
        """Computes the off route metric.

        Args:
          simulator_state: The current simulator state of shape (....).

        Returns:
          A (...) MetricResult containing the metric result described above.

        Raises:
          ValueError: If `simulator_state.sdc_paths` is undefined.
        """
        sdc_paths = simulator_state.sdc_paths
        if sdc_paths is None:
            raise ValueError("SimulatorState.sdc_paths required to compute the off-route metric.")

        # Shape: (..., num_objects, num_timesteps=1, 2)
        obj_xy = datatypes.dynamic_slice(
            simulator_state.sim_trajectory.xy,
            simulator_state.timestep,
            1,
            axis=-2,
        )
        # Shape: (..., 2)
        sdc_xy = datatypes.select_by_onehot(
            obj_xy[..., 0, :],
            simulator_state.object_metadata.is_sdc,
            keepdims=False,
        )
        # Shape: (..., num_objects, num_timesteps=1)
        obj_valid = datatypes.dynamic_slice(
            simulator_state.sim_trajectory.valid,
            simulator_state.timestep,
            1,
            axis=-1,
        )
        # Shape: (...)
        sdc_valid = datatypes.select_by_onehot(
            obj_valid[..., 0],
            simulator_state.object_metadata.is_sdc,
            keepdims=False,
        )

        # Shape: (..., num_paths, num_points_per_path)
        sdc_dist_to_paths = jnp.linalg.norm(
            sdc_xy[..., jnp.newaxis, jnp.newaxis, :] - sdc_paths.xy,
            axis=-1,
            keepdims=False,
        )
        # Set distances to invalid paths to inf.
        sdc_dist_to_valid_paths = jnp.where(sdc_paths.valid, sdc_dist_to_paths, jnp.inf)
        # Set distances to invalid SDC states to inf.
        sdc_dist_to_valid_paths = jnp.where(jnp.expand_dims(sdc_valid, (-1, -2)), sdc_dist_to_valid_paths, jnp.inf)
        sdc_dist_to_valid_on_route_paths = jnp.where(sdc_paths.on_route, sdc_dist_to_valid_paths, jnp.inf)
        sdc_dist_to_valid_off_route_paths = jnp.where(~sdc_paths.on_route, sdc_dist_to_valid_paths, jnp.inf)

        # Shape: (...)
        min_sdc_dist_to_valid_on_route_paths = jnp.min(sdc_dist_to_valid_on_route_paths, axis=(-1, -2))
        min_sdc_dist_to_valid_off_route_paths = jnp.min(sdc_dist_to_valid_off_route_paths, axis=(-1, -2))

        sdc_off_route = (min_sdc_dist_to_valid_on_route_paths > self.MAX_DISTANCE_TO_ROUTE_PATH) | (
            min_sdc_dist_to_valid_on_route_paths - min_sdc_dist_to_valid_off_route_paths
            > self.MAX_RELATIVE_DISTANCE_TO_OFF_ROUTE_PATH
        )

        off_route = jnp.where(sdc_off_route, min_sdc_dist_to_valid_on_route_paths, 0.0)
        valid = jnp.isfinite(off_route)
        off_route = jnp.where(valid, off_route, 0.0)
        return abstract_metric.MetricResult.create_and_validate(off_route, valid)
