"""Shared helpers for computing observation structure (category + entity level).

Used by both PerceiverWrapper and GenericWrapper to avoid code duplication.
"""

import jax.numpy as jnp
import numpy as np

# Category names in the order the flat observation is laid out.
CATEGORY_NAMES = [
    "sdc_trajectory",
    "other_agents",
    "roadgraph",
    "traffic_lights",
    "gps_path",
]

# Entity name prefixes for each category.
_ENTITY_PREFIXES = {
    "sdc_trajectory": "sdc",
    "other_agents": "agent",
    "roadgraph": "road_pt",
    "traffic_lights": "light",
    "gps_path": "waypoint",
}


def compute_observation_structures(
    unflatten_fn, obs_size: int
) -> tuple[dict[str, tuple[int, int]], dict[str, dict]]:
    """Compute both category-level and entity-level observation structure.

    Args:
        unflatten_fn: V-MAX unflatten_features function.
        obs_size: Total flat observation size.

    Returns:
        Tuple of (category_structure, detailed_structure).
    """
    dummy_obs = jnp.zeros((1, 1, obs_size))
    features, masks = unflatten_fn(dummy_obs)

    # features = (sdc_feat, other_feat, rg_feat, tl_feat, gps_feat)
    # masks    = (sdc_mask, other_mask, rg_mask, tl_mask)  — no mask for gps
    feat_list = list(features)
    mask_list = list(masks) + [None]  # pad so zip works (gps has no mask)

    category_structure: dict[str, tuple[int, int]] = {}
    detailed_structure: dict[str, dict] = {}
    idx = 0

    for cat_name, feat, mask in zip(CATEGORY_NAMES, feat_list, mask_list):
        # Strip leading batch dims (1, 1, ...) → real shape
        real_feat_shape = feat.shape[2:]  # e.g. (8, 5, 7) for other_agents
        num_entities = real_feat_shape[0]
        feat_per_entity = int(np.prod(real_feat_shape[1:])) if len(real_feat_shape) > 1 else 1

        if mask is not None:
            real_mask_shape = mask.shape[2:]  # e.g. (8, 5) for other_agents
            mask_per_entity = int(np.prod(real_mask_shape[1:])) if len(real_mask_shape) > 1 else 1
        else:
            mask_per_entity = 0

        raw_per_entity = feat_per_entity + mask_per_entity
        total_flat = num_entities * raw_per_entity

        # Category-level range
        category_structure[cat_name] = (idx, idx + total_flat)

        # Entity-level ranges
        prefix = _ENTITY_PREFIXES[cat_name]
        entities: dict[str, tuple[int, int]] = {}
        for i in range(num_entities):
            entity_start = idx + i * raw_per_entity
            entity_end = entity_start + raw_per_entity
            entities[f"{prefix}_{i}"] = (entity_start, entity_end)

        detailed_structure[cat_name] = {
            "num_entities": num_entities,
            "features_per_entity": raw_per_entity,
            "entities": entities,
        }

        idx += total_flat

    assert idx == obs_size, (
        f"Observation structure total ({idx}) != obs_size ({obs_size})"
    )
    return category_structure, detailed_structure


def extract_entity_validity(
    unflatten_fn, observation: jnp.ndarray
) -> dict[str, dict[str, bool]]:
    """Extract per-entity validity flags from a real observation.

    Args:
        unflatten_fn: V-MAX unflatten_features function.
        observation: Flat observation, shape ``(obs_dim,)``.

    Returns:
        Dict mapping category → {entity_name: is_valid}.
    """
    features, masks = unflatten_fn(observation)
    mask_list = list(masks) + [None]  # gps has no mask

    validity: dict[str, dict[str, bool]] = {}

    for cat_name, mask in zip(CATEGORY_NAMES, mask_list):
        prefix = _ENTITY_PREFIXES[cat_name]
        entity_valid: dict[str, bool] = {}

        if mask is not None:
            mask_np = np.array(mask)
            num_entities = mask_np.shape[0]
            for i in range(num_entities):
                # Entity is valid if any of its mask values > 0
                entity_valid[f"{prefix}_{i}"] = bool(mask_np[i].any())
        else:
            # No mask (gps_path) — assume all valid
            feat = features[CATEGORY_NAMES.index(cat_name)]
            num_entities = feat.shape[0] if feat.ndim > 1 else 1
            for i in range(num_entities):
                entity_valid[f"{prefix}_{i}"] = True

        validity[cat_name] = entity_valid

    return validity
