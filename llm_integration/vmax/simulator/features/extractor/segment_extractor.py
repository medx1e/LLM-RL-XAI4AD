# Copyright 2025 Valeo.

"""Feature extractor for segment observation."""

import jax
import jax.numpy as jnp
from waymax import datatypes

from vmax.simulator import features, metrics, operations
from vmax.simulator.features import extractor


class SegmentFeaturesExtractor(extractor.VecFeaturesExtractor):
    """Feature extractor for segment observation."""

    def __init__(
        self,
        obs_past_num_steps: int = 1,
        objects_config: dict | None = None,
        roadgraphs_config: dict | None = None,
        traffic_lights_config: dict | None = None,
        path_target_config: dict | None = None,
    ) -> None:
        """Initialize the segment features extractor.

        Args:
            obs_past_num_steps: Number of past steps to consider.
            objects_config: Configuration dictionary for object features.
            roadgraphs_config: Configuration dictionary for roadgraph features.
            traffic_lights_config: Configuration dictionary for traffic light features.
            path_target_config: Configuration dictionary for path target features.

        """
        super().__init__(
            obs_past_num_steps,
            objects_config,
            roadgraphs_config,
            traffic_lights_config,
            path_target_config,
        )

        self._max_num_lanes = roadgraphs_config["max_num_lanes"]
        self._max_num_points_per_lane = roadgraphs_config["max_num_points_per_lane"]

    def unflatten_features(self, vectorized_obs: jax.Array) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
        """Unflatten a vectorized observation into features and masks.

        Args:
            vectorized_obs: The vectorized observation.

        Returns:
            A tuple containing the features and masks.

        """
        batch_dims = vectorized_obs.shape[-3:-1]
        flatten_size = vectorized_obs.shape[-1]
        unflatten_size = 0

        object_features_size = self.get_features_size(self._object_features_key)
        traffic_lights_features_size = self.get_features_size(self._traffic_lights_features_key)
        path_target_feature_size = self.get_features_size(self._path_target_features_key)

        sdc_object_size = 1 * self._obs_past_num_steps * object_features_size
        sdc_object_features = vectorized_obs[..., unflatten_size : unflatten_size + sdc_object_size]
        sdc_object_features = sdc_object_features.reshape(
            *batch_dims,
            1,
            self._obs_past_num_steps,
            object_features_size,
        )
        unflatten_size += sdc_object_size

        other_objects_size = self._num_closest_objects * self._obs_past_num_steps * object_features_size
        other_objects_features = vectorized_obs[..., unflatten_size : unflatten_size + other_objects_size]
        other_objects_features = other_objects_features.reshape(
            *batch_dims,
            self._num_closest_objects,
            self._obs_past_num_steps,
            object_features_size,
        )
        unflatten_size += other_objects_size

        num_points = self._max_num_lanes * self._max_num_points_per_lane
        roadgraph_size = (
            self._max_num_lanes * extractor.get_feature_size("types", self._dict_mapping)
            if "types" in self._roadgraph_features_key
            else 0
        )
        roadgraph_size += num_points * 2 if "xy" in self._roadgraph_features_key else 0
        roadgraph_size += num_points * 2 if "dir_xy" in self._roadgraph_features_key else 0
        roadgraph_size += self._max_num_lanes if "valid" in self._roadgraph_features_key else 0

        roadgraphs_features = vectorized_obs[..., unflatten_size : unflatten_size + roadgraph_size]
        roadgraphs_features = roadgraphs_features.reshape(
            *batch_dims,
            self._max_num_lanes,
            roadgraph_size // self._max_num_lanes,
        )
        unflatten_size += roadgraph_size

        traffic_lights_features = vectorized_obs[..., unflatten_size : unflatten_size + traffic_lights_features_size]
        traffic_lights_features = traffic_lights_features.reshape(
            *batch_dims,
            1,
            1,
            traffic_lights_features_size,
        )
        unflatten_size += traffic_lights_features_size

        path_target_size = self._num_target_path_points * path_target_feature_size
        path_target_features = vectorized_obs[..., unflatten_size : unflatten_size + path_target_size]
        path_target_features = path_target_features.reshape(
            *batch_dims,
            self._num_target_path_points,
            path_target_feature_size,
        )
        unflatten_size += path_target_size

        assert flatten_size == unflatten_size, f"Unflatten size {unflatten_size} does not match {flatten_size}"

        features = (
            sdc_object_features[..., :-1],
            other_objects_features[..., :-1],
            roadgraphs_features[..., :-1],
            traffic_lights_features[..., :-1],
            path_target_features,
        )
        masks = (
            sdc_object_features[..., -1].astype(bool),
            other_objects_features[..., -1].astype(bool),
            roadgraphs_features[..., -1].astype(bool),
            traffic_lights_features[..., -1].astype(bool),
        )

        return features, masks

    def _build_roadgraph_features(
        self,
        sdc_obs: datatypes.Observation,
        slice_size: int = 100,
    ) -> features.RoadgraphFeatures:
        """Construct roadgraph features from the observation.

        Args:
            sdc_obs: The observation for the SDC.
            slice_size: The slice length used for sampling points.

        Returns:
            An instance of RoadgraphFeatures.

        """
        # (self._max_num_lanes,)
        closest_lane_ids = _find_closest_unique_lane_ids(sdc_obs, self._max_num_lanes)
        v_map_sub_sample_lane = jax.vmap(_n_sample_from_lane, in_axes=(None, 0, None, None))
        # (self._max_num_lanes, self._max_num_points_per_lane, 2)
        samples, sdc_frame_samples = v_map_sub_sample_lane(
            sdc_obs,
            closest_lane_ids,
            slice_size,
            self._max_num_points_per_lane,
        )
        # (self._max_num_lanes, self._max_num_points_per_lane * 2)
        sdc_frame_samples = jnp.reshape(sdc_frame_samples, (sdc_frame_samples.shape[0], -1))
        # (self._max_num_lanes, self._max_num_points_per_lane * 2)
        sdc_frame_samples = extractor.normalize_by_feature(
            sdc_frame_samples,
            "xy",
            self._max_meters,
            self._dict_mapping,
        )

        roadgraph_features = features.RoadgraphFeatures(field_names=self._roadgraph_features_key)
        roadgraph_features.xy = sdc_frame_samples

        if "types" in self._roadgraph_features_key:
            v_map__extract_lane_type_from_id = jax.vmap(_extract_lane_type_from_id, in_axes=(None, 0))
            # (self._max_num_lanes,)
            rg_types = v_map__extract_lane_type_from_id(sdc_obs, closest_lane_ids)
            # (self._max_num_lanes, 8)
            rg_types = extractor.normalize_by_feature(rg_types, "types", self._max_meters, self._dict_mapping)
            # (self._max_num_lanes, self._max_num_points_per_lane * 2 + 8)
            roadgraph_features.types = rg_types

        if "dir_xy" in self._roadgraph_features_key:
            v_map_get_direction_vector = jax.vmap(_get_direction_vector_from_sample, in_axes=(None, 0, 0))
            # (self._max_num_lanes, self._max_num_points_per_lane, 2)
            directions = v_map_get_direction_vector(sdc_obs, samples, closest_lane_ids)
            # (self._max_num_lanes, self._max_num_points_per_lane * 2)
            directions = jnp.reshape(directions, (self._max_num_lanes, -1))
            # (self._max_num_lanes, self._max_num_points_per_lane * 4 + 8)
            roadgraph_features.dir_xy = extractor.normalize_by_feature(
                directions,
                "dir_xy",
                self._max_meters,
                self._dict_mapping,
            )

        if "valid" in self._roadgraph_features_key:
            # (self._max_num_lanes, 1)
            valid_mask = jnp.ones((self._max_num_lanes,), dtype=jnp.bool_)[..., None]
            # (self._max_num_lanes, self._max_num_points_per_lane * 4 + 8 + 1)
            roadgraph_features.valid = valid_mask

        return roadgraph_features

    def _build_traffic_lights_features(self, sdc_obs: datatypes.Observation) -> features.TrafficLightFeatures:
        """Build traffic light features using the current lane's observation.

        Args:
            sdc_obs: The SDC observation.

        Returns:
            An instance of TrafficLightFeatures.

        """
        if len(self._traffic_lights_features_key) == 0:
            return jnp.array(())

        red_light_id = metrics.get_id_red_for_sdc(sdc_obs)

        traffic_light_features = features.TrafficLightFeatures(field_names=self._traffic_lights_features_key)
        for key in self._traffic_lights_features_key:
            feature = getattr(sdc_obs.traffic_lights, key)[red_light_id]
            feature = jnp.expand_dims(feature, axis=0)
            feature = extractor.normalize_by_feature(feature, key, self._max_meters, self._dict_mapping)

            if feature.ndim == 2:
                feature = jnp.expand_dims(feature, axis=-1)

            setattr(traffic_light_features, key, feature[:, -1:, :])

        return traffic_light_features


def _get_direction_vector_from_sample(sdc_obs: datatypes.Observation, sample: jax.Array, lane_id: int) -> jax.Array:
    """Compute the direction vector for a sample point on a lane.

    Args:
        sdc_obs: The SDC observation containing roadgraph points.
        sample: The sample points.
        lane_id: The lane identifier.

    Returns:
        The direction vector derived from the sample.

    """
    # (n_rg_points)
    lane_mask = sdc_obs.roadgraph_static_points.ids == lane_id
    # (n_rg_points, n_sample)
    distances = jnp.linalg.norm(sdc_obs.roadgraph_static_points.xy[:, None, :] - sample, axis=-1)
    distances = jnp.where(lane_mask[:, None], distances, jnp.inf)
    # (n_sample)
    closest_idxs = jnp.argmin(distances, axis=0)
    dirs = sdc_obs.roadgraph_static_points.dir_xy[closest_idxs]

    return dirs


def _extract_lane_type_from_id(sdc_obs: datatypes.Observation, lane_id: int) -> int:
    """Extract the lane type based on the lane identifier.

    Args:
        sdc_obs: The SDC observation containing roadgraph points.
        lane_id: The lane ID from which to extract the lane type.

    Returns:
        The lane type as an integer.

    """
    mask = sdc_obs.roadgraph_static_points.ids == lane_id
    first_true_idx = jnp.argmax(mask)

    return sdc_obs.roadgraph_static_points.types[first_true_idx]


def _n_sample_from_lane(
    sdc_obs: datatypes.Observation,
    lane_id: jax.Array,
    slice_size: int = 200,
    n_sample: int = 8,
) -> tuple[jax.Array, jax.Array]:
    """Linearly sample points from a given lane.

    Args:
        sdc_obs: The SDC observation with roadgraph data.
        lane_id: The lane identifier to sample from.
        slice_size: The size of the slice to extract.
        n_sample: Number of points to sample.

    Returns:
        A tuple containing sampled points and the SDC frame sample.

    """
    lane_mask = sdc_obs.roadgraph_static_points.ids == lane_id

    rg_xy = jnp.where(
        lane_mask[:, None],
        sdc_obs.roadgraph_static_points.xy,
        jnp.inf,
    )
    distances = jnp.linalg.norm(rg_xy, axis=1)
    closest_rg_idx_from_sdc = operations.get_index(-distances)

    lane_rg_points = jnp.where(
        lane_mask[:, None],
        sdc_obs.roadgraph_static_points.xy,
        0,
    )

    lane_rg_points = jax.lax.dynamic_slice(
        lane_rg_points,
        (closest_rg_idx_from_sdc - slice_size // 2, 0),
        (slice_size, 2),
    )

    # 0 for lane with other id, 1 for points of lane with current id
    # pad mask shape [[num_envs, num_ep, max_num_lanes, slice_size]
    is_point_from_lane_id = jnp.any(lane_rg_points != jnp.array((0, 0)), axis=1)
    is_point_from_lane_id_flipped = jnp.flip(is_point_from_lane_id, axis=0)
    # Normally is_point_from_lane_id is like 0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0
    # (only one change from 0 to 1 and from 1 to 0)
    # indice_other_lane_to_current_lane ==> should return indice 6, change from other to current lane
    # indice_current_lane_to_other_lane ==> should return indice 13, change from current to other

    # argmax give the first occurance of 1 (so indice from 0 to 1)
    indice_other_lane_to_current_lane = jnp.argmax(is_point_from_lane_id)

    # argmax of flipped array give the last occurance of 1 (so indice from 1 to 0)
    indice_current_lane_to_other_lane = slice_size - 1 - jnp.argmax(is_point_from_lane_id_flipped)

    # Hack to fill the x,y value from points of other lane with "border" values of current lane
    # (this is a hack to keep the shape=slice_size and not mess up the linear interpolation
    lane_rg_points = jnp.where(
        (jnp.arange(slice_size) < indice_other_lane_to_current_lane)[:, None],
        lane_rg_points[indice_other_lane_to_current_lane][None, :],
        lane_rg_points,
    )
    lane_rg_points = jnp.squeeze(lane_rg_points)
    lane_rg_points = jnp.where(
        (jnp.arange(slice_size) > indice_current_lane_to_other_lane)[:, None],
        lane_rg_points[indice_current_lane_to_other_lane][None, :],
        lane_rg_points,
    )
    lane_rg_points = jnp.squeeze(lane_rg_points)

    lane_rg_points_x = lane_rg_points[..., 0]
    lane_rg_points_y = lane_rg_points[..., 1]

    distances = jnp.cumsum(jnp.sqrt(jnp.diff(lane_rg_points_x) ** 2 + jnp.diff(lane_rg_points_y) ** 2))
    new_distances = jnp.linspace(0, distances[-1], n_sample)

    def _linear_interp(new_distances: jax.Array, distances: jax.Array, values: jax.Array) -> jax.Array:
        idx = jnp.searchsorted(distances, new_distances) - 1
        idx = jnp.clip(idx, 0, len(distances) - 2)

        d0 = distances[idx]
        d1 = distances[idx + 1] + 0.0001
        v0 = values[idx]
        v1 = values[idx + 1]

        fraction = (new_distances - d0) / (d1 - d0)

        return v0 + fraction * (v1 - v0)

    new_x = _linear_interp(new_distances, distances, lane_rg_points_x)
    new_y = _linear_interp(new_distances, distances, lane_rg_points_y)

    sample = jnp.concatenate([new_x[:, None], new_y[:, None]], axis=1)
    sdc_frame_sample = jnp.squeeze(sample)

    return sample, sdc_frame_sample


def _find_closest_unique_lane_ids(sdc_obs: datatypes.Observation, topk: int = 10) -> jax.Array:
    """Identify closest unique lane IDs to the SDC.

    Args:
        sdc_obs: The SDC observation with roadgraph data.
        topk: The number of unique lane IDs to retrieve.

    Returns:
        An array of the closest unique lane IDs.

    """
    rg_xy = sdc_obs.roadgraph_static_points.xy

    def get_next_closest_lane_id(distances, x):
        idx = operations.get_index(-distances)
        lane_id = sdc_obs.roadgraph_static_points.ids[idx]
        distances = jnp.where(sdc_obs.roadgraph_static_points.ids == lane_id, jnp.inf, distances)
        return distances, lane_id

    init_distances = jnp.linalg.norm(rg_xy, axis=1)
    init_distances = jnp.where(sdc_obs.roadgraph_static_points.ids == -1, jnp.inf, init_distances)
    lane_ids = jax.lax.scan(get_next_closest_lane_id, init_distances, length=topk)[1]

    return jnp.squeeze(lane_ids)
