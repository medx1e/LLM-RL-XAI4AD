import chex
import jax
import jax.numpy as jnp
from waymax import datatypes


def apply_random_masking(sdc_observation: datatypes.Observation, key: jax.Array) -> datatypes.Observation:
    """Apply a mask to an SDC observation. Only applied to objects and traffic lights.

    Masking is not persistant through time. The probability of an object or traffic light being observed is relative
    to the distance to the SDC. The closer the object or traffic light, the higher the probability of being observed.

    Args:
        sdc_observation: The SDC observation.
        key: The mask to apply.

    Returns:
        The SDC observation with the mask applied.

    """
    objects = sdc_observation.trajectory
    traffic_lights = sdc_observation.traffic_lights

    # Split key for objects and traffic lights
    key_obj, key_tl = jax.random.split(key)

    # Define scale factor (adjustable threshold)
    scale = 200.0  # Using 200 as the scaling factor

    # Update mask for objects: closer objects have higher probability to remain valid
    obj_distance = jnp.linalg.norm(objects.xy[:, -1, :], axis=-1)
    obj_prob = jnp.exp(-obj_distance / scale)
    obj_random = jax.random.uniform(key_obj, shape=obj_prob.shape)
    new_obj_valid = objects.valid.at[:, -1].set(objects.valid[:, -1] & (obj_random < obj_prob))

    # Update mask for traffic lights: closer lights have higher probability to remain valid
    tl_distance = jnp.linalg.norm(traffic_lights.xy[:, -1], axis=-1)
    tl_prob = jnp.exp(-tl_distance / scale)
    tl_random = jax.random.uniform(key_tl, shape=tl_prob.shape)
    new_tl_valid = traffic_lights.valid.at[:, -1].set(traffic_lights.valid[:, -1] & (tl_random < tl_prob))

    # Assign updated valid fields back to the observation
    objects = objects.replace(valid=new_obj_valid)
    traffic_lights = traffic_lights.replace(valid=new_tl_valid)

    return sdc_observation.replace(trajectory=objects, traffic_lights=traffic_lights)


def apply_gaussian_noise(sdc_observation: datatypes.Observation, key: jax.Array) -> datatypes.Observation:
    """Apply Gaussian noise to the observation. Each data point is perturbed by a random value sampled from a Gaussian
    distribution. The standard deviation of the distribution is 0.1 times the range of the data point.

    Args:
        sdc_observation: The SDC observation.
        key: The key to generate the noise.

    Returns:
        The SDC observation with the noise applied.

    """
    objects = sdc_observation.trajectory
    traffic_lights = sdc_observation.traffic_lights
    roadgraph = sdc_observation.roadgraph_static_points

    # Split key for objects and traffic lights
    key_obj, key_tl, key_rg = jax.random.split(key, 3)

    # Define noise scale factor
    noise_scale = 0.1

    # Generate noise for objects
    obj_noise = jax.random.normal(key_obj, shape=objects.xy.shape) * noise_scale
    new_obj_xy = objects.xy + obj_noise

    # Generate noise for traffic lights
    tl_noise = jax.random.normal(key_tl, shape=traffic_lights.xy.shape) * noise_scale
    new_tl_xy = traffic_lights.xy + tl_noise

    # Generate noise for roadgraph
    roadgraph_noise = jax.random.normal(key_rg, shape=roadgraph.xy.shape) * noise_scale
    new_rg_xy = roadgraph.xy + roadgraph_noise

    # Assign updated xy fields back to the observation
    objects = objects.replace(x=new_obj_xy[:, :, 0], y=new_obj_xy[:, :, 1])
    traffic_lights = traffic_lights.replace(x=new_tl_xy[:, :, 0], y=new_tl_xy[:, :, 1])
    roadgraph = roadgraph.replace(x=new_rg_xy[:, 0], y=new_rg_xy[:, 1])

    return sdc_observation.replace(trajectory=objects, traffic_lights=traffic_lights, roadgraph_static_points=roadgraph)


def apply_obstruction(sdc_observation: datatypes.Observation) -> datatypes.Trajectory:
    """Compute the obstruction of the trajectory.

    For each agent, check if there is an obstruction in the direct line of sight with the SDC. SDC is at pos [0, 0]

    Args:
        trajectory: The trajectory.

    Returns:
        The trajectory with the obstruction.

    """
    trajectory = sdc_observation.trajectory
    object_types = sdc_observation.metadata.object_types

    mask = trajectory.valid & (object_types == 1)[..., jnp.newaxis]
    _, object_idx = jax.lax.top_k(~sdc_observation.is_ego, k=trajectory.num_objects - 1)

    bbox_corners = jnp.take_along_axis(
        trajectory.bbox_corners,
        object_idx[..., jnp.newaxis, jnp.newaxis, jnp.newaxis],
        axis=0,
    )
    mask = jnp.take_along_axis(mask, object_idx[..., jnp.newaxis], axis=0)
    bbox_corners = jnp.where(mask[..., jnp.newaxis, jnp.newaxis], bbox_corners, jnp.inf)
    bbox_corners = jnp.swapaxes(bbox_corners, 0, 1)
    vmap_on_objects = jax.vmap(_is_vector_intersect_polygon, in_axes=(0, None))
    vmap_on_history = jax.vmap(vmap_on_objects, in_axes=(0, 0))

    def compute_line_of_sight(idx):
        # (obs_past_num_steps, 2)
        vector = trajectory.xy[idx]
        # (obs_past_num_steps, num_objects - 1)
        intersects = vmap_on_history(bbox_corners, vector)
        # (num_objects - 1, obs_past_num_steps)
        intersects = jnp.swapaxes(intersects, 0, 1)
        # Set False at the object index
        intersects = intersects.at[idx - 1].set(False)

        return ~jnp.any(intersects, axis=0)

    valid_line_of_sight = jax.vmap(compute_line_of_sight)(object_idx)

    sdc_valid = trajectory.valid[0][None, ...]
    others_valid = jnp.logical_and(trajectory.valid[1:], valid_line_of_sight)
    valid = jnp.concatenate([sdc_valid, others_valid], axis=0)

    trajectory = trajectory.replace(valid=valid)

    return sdc_observation.replace(trajectory=trajectory)


def _is_vector_intersect_polygon(bbox_corner: jax.Array, vector: jax.Array) -> bool:
    """Check if the vector intersects with the bbox_corner defined by 4 points.

    Args:
        bbox_corner: The bbox corner points.
        vector: The vector to check for intersection.

    Returns:
        True if an intersection is found, False otherwise.
    """
    chex.assert_shape(bbox_corner, (4, 2))
    chex.assert_shape(vector, (2,))

    eps = 1e-8
    num_points = bbox_corner.shape[0]
    # Compute indices for the next vertex (wrap around using modulo)
    idx = jnp.arange(num_points)
    idx_next = jnp.mod(idx + 1, num_points)

    # Starting points and edge vectors for each bbox_corner edge.
    q = bbox_corner  # shape (N, 2)
    s = bbox_corner[idx_next] - bbox_corner  # shape (N, 2)

    # Compute cross product of r and each s vector.
    r_cross_s = vector[0] * s[:, 1] - vector[1] * s[:, 0]  # shape (N,)

    # Avoid division by zero by flagging nearly zero values.
    valid = jnp.abs(r_cross_s) > eps

    # Compute t and u using vectorized operations.
    t = jnp.where(valid, (q[:, 0] * s[:, 1] - q[:, 1] * s[:, 0]) / r_cross_s, -1.0)
    u = jnp.where(valid, (q[:, 0] * vector[1] - q[:, 1] * vector[0]) / r_cross_s, -1.0)

    # Check if intersection lies on both segments.
    cond = valid & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)

    return jnp.any(cond)
