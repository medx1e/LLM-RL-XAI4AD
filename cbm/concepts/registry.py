"""Concept registry: maps concept names to extractor functions + schemas.

Call ``extract_all_concepts(inp)`` to run every registered concept and
get a single ``ConceptOutput``.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import partial

import jax
import jax.numpy as jnp

from concepts.schema import ConceptSchema, ConceptType
from concepts.types import ConceptInput, ConceptOutput
from concepts import extractors as E


# =====================================================================
# Schema + extractor pairs
# =====================================================================

_REGISTRY: OrderedDict[str, tuple[ConceptSchema, callable]] = OrderedDict()


def _reg(schema: ConceptSchema, fn: callable) -> None:
    _REGISTRY[schema.name] = (schema, fn)


# ---- Phase 1 --------------------------------------------------------

_reg(
    ConceptSchema(
        name="ego_speed",
        concept_type=ConceptType.CONTINUOUS,
        description="Ego vehicle scalar speed at current timestep",
        source_fields=["sdc_features (vel_xy)", "sdc_mask"],
        formula="||vel_xy[-1]|| * max_speed",
        unit="m/s",
        norm_range=(0.0, 1.0),
        validity_rule="sdc_mask[0, -1]",
        phase=1,
    ),
    E.ego_speed,
)

_reg(
    ConceptSchema(
        name="ego_acceleration",
        concept_type=ConceptType.CONTINUOUS,
        description="Longitudinal acceleration from consecutive speed estimates",
        source_fields=["sdc_features (vel_xy)", "sdc_mask"],
        formula="(speed[-1] - speed[-2]) / dt",
        unit="m/s^2",
        norm_range=(-1.0, 1.0),
        validity_rule="sdc_mask[0, -1] AND sdc_mask[0, -2]",
        phase=1,
    ),
    E.ego_acceleration,
)

_reg(
    ConceptSchema(
        name="dist_nearest_object",
        concept_type=ConceptType.CONTINUOUS,
        description="Distance to nearest valid other agent",
        source_fields=["agent_features (xy)", "agent_mask"],
        formula="min_n ||agent_xy[n, -1]|| * max_meters",
        unit="m",
        norm_range=(0.0, 1.0),
        validity_rule="any(agent_mask[:, -1])",
        phase=1,
    ),
    E.dist_nearest_object,
)

_reg(
    ConceptSchema(
        name="num_objects_within_10m",
        concept_type=ConceptType.CONTINUOUS,
        description="Count of valid agents within 10 m of SDC",
        source_fields=["agent_features (xy)", "agent_mask"],
        formula="sum(||agent_xy|| < 10 AND valid)",
        unit="count",
        norm_range=(0.0, 1.0),
        validity_rule="always valid (count can be 0)",
        phase=1,
    ),
    partial(E.num_objects_within_radius, radius_m=10.0),
)

_reg(
    ConceptSchema(
        name="traffic_light_red",
        concept_type=ConceptType.BINARY,
        description="Whether any visible traffic light is currently red",
        source_fields=["tl_features (state_onehot)", "tl_mask"],
        formula="any(onehot[{0,3,6}] > 0.5 AND tl_valid)",
        unit="bool",
        norm_range=(0.0, 1.0),
        validity_rule="any(tl_mask[:, -1])",
        phase=1,
    ),
    E.traffic_light_red,
)

_reg(
    ConceptSchema(
        name="dist_to_traffic_light",
        concept_type=ConceptType.CONTINUOUS,
        description="Distance to nearest valid traffic light",
        source_fields=["tl_features (xy)", "tl_mask"],
        formula="min_n ||tl_xy[n, -1]|| * max_meters",
        unit="m",
        norm_range=(0.0, 1.0),
        validity_rule="any(tl_mask[:, -1])",
        phase=1,
    ),
    E.dist_to_traffic_light,
)

_reg(
    ConceptSchema(
        name="heading_deviation",
        concept_type=ConceptType.CONTINUOUS,
        description="Signed heading deviation from path tangent",
        source_fields=["sdc_features (yaw)", "path_features (xy)", "sdc_mask"],
        formula="wrap(ego_yaw - atan2(path_dy, path_dx))",
        unit="rad",
        norm_range=(-1.0, 1.0),
        validity_rule="sdc_mask[0, -1]",
        phase=1,
    ),
    E.heading_deviation,
)

_reg(
    ConceptSchema(
        name="progress_along_route",
        concept_type=ConceptType.CONTINUOUS,
        description="Fraction of GPS path traversed (0=start, 1=end)",
        source_fields=["path_features (xy)", "sdc_mask"],
        formula="project_onto_path((0,0), path_xy).arc_fraction",
        unit="fraction",
        norm_range=(0.0, 1.0),
        validity_rule="sdc_mask[0, -1]",
        phase=1,
    ),
    E.progress_along_route,
)

# ---- Phase 2 --------------------------------------------------------

_reg(
    ConceptSchema(
        name="ttc_lead_vehicle",
        concept_type=ConceptType.CONTINUOUS,
        description="Time-to-collision with lead vehicle (capped 10 s)",
        source_fields=["agent_features (xy, vel_xy)", "agent_mask", "sdc_features (vel_xy)"],
        formula="lead_dist_x / max(ego_vx - lead_vx, eps); capped 10s",
        unit="s",
        norm_range=(0.0, 1.0),
        validity_rule="exists lead vehicle (ahead, in lane, valid)",
        phase=2,
    ),
    E.ttc_lead_vehicle,
)

_reg(
    ConceptSchema(
        name="lead_vehicle_decelerating",
        concept_type=ConceptType.BINARY,
        description="Whether the lead vehicle is decelerating",
        source_fields=["agent_features (xy, vel_xy)", "agent_mask"],
        formula="lead_speed[-2] - lead_speed[-1] > 0.5 m/s",
        unit="bool",
        norm_range=(0.0, 1.0),
        validity_rule="lead vehicle exists and valid at t and t-1",
        phase=2,
    ),
    E.lead_vehicle_decelerating,
)

_reg(
    ConceptSchema(
        name="at_intersection",
        concept_type=ConceptType.BINARY,
        description="Heuristic: any valid traffic light within 25 m",
        source_fields=["tl_features (xy)", "tl_mask"],
        formula="any(||tl_xy|| < 25 AND tl_valid)",
        unit="bool",
        norm_range=(0.0, 1.0),
        validity_rule="any(tl_mask[:, -1])",
        phase=2,
    ),
    E.at_intersection,
)

# ---- Phase 3 — Path-based spatial concepts (CBM-V2) -----------------

_reg(
    ConceptSchema(
        name="path_curvature_max",
        concept_type=ConceptType.CONTINUOUS,
        description="Maximum Menger curvature along the GPS path",
        source_fields=["path_features (xy)"],
        formula="max_k Menger_curvature(path_xy, k)",
        unit="1/m",
        norm_range=(0.0, 1.0),
        validity_rule="always valid (path always present)",
        phase=3,
    ),
    E.path_curvature_max,
)

_reg(
    ConceptSchema(
        name="path_net_heading_change",
        concept_type=ConceptType.CONTINUOUS,
        description="Signed net heading change first→last path segment",
        source_fields=["path_features (xy)"],
        formula="wrap(atan2(last_seg) - atan2(first_seg))",
        unit="rad",
        norm_range=(-1.0, 1.0),
        validity_rule="always valid",
        phase=3,
    ),
    E.path_net_heading_change,
)

_reg(
    ConceptSchema(
        name="path_straightness",
        concept_type=ConceptType.CONTINUOUS,
        description="Chord/arc ratio — 1.0 = straight, 0 = maximally curved",
        source_fields=["path_features (xy)"],
        formula="||last - first|| / sum(||seg_i||)",
        unit="ratio",
        norm_range=(0.0, 1.0),
        validity_rule="always valid",
        phase=3,
    ),
    E.path_straightness,
)

_reg(
    ConceptSchema(
        name="heading_to_path_end",
        concept_type=ConceptType.CONTINUOUS,
        description="Angle from ego to route endpoint in SDC frame",
        source_fields=["path_features (xy)"],
        formula="atan2(end_y, end_x)",
        unit="rad",
        norm_range=(-1.0, 1.0),
        validity_rule="always valid",
        phase=3,
    ),
    E.heading_to_path_end,
)

# =====================================================================
# Public API
# =====================================================================

CONCEPT_REGISTRY = _REGISTRY


def _normalize_concept(raw: jax.Array, schema: ConceptSchema) -> jax.Array:
    """Map raw concept to [0, 1] range."""
    if schema.concept_type == ConceptType.BINARY:
        return raw  # already 0/1

    # Heuristic normalization per concept
    name = schema.name
    if name == "ego_speed":
        return jnp.clip(raw / 30.0, 0.0, 1.0)
    elif name == "ego_acceleration":
        return jnp.clip((raw + 6.0) / 12.0, 0.0, 1.0)  # [-6, 6] → [0, 1]
    elif name in ("dist_nearest_object", "dist_to_traffic_light"):
        return jnp.clip(raw / 70.0, 0.0, 1.0)
    elif name == "num_objects_within_10m":
        return jnp.clip(raw / 8.0, 0.0, 1.0)
    elif name == "heading_deviation":
        return jnp.clip((raw + jnp.pi) / (2 * jnp.pi), 0.0, 1.0)
    elif name == "progress_along_route":
        return jnp.clip(raw, 0.0, 1.0)
    elif name == "ttc_lead_vehicle":
        return jnp.clip(raw / 10.0, 0.0, 1.0)
    # ---- Phase 3 normalizations ----
    elif name == "path_curvature_max":
        return jnp.clip(raw / 0.25, 0.0, 1.0)   # 0.25 1/m ≈ 4m radius turn
    elif name in ("path_net_heading_change", "heading_to_path_end"):
        return jnp.clip((raw + jnp.pi) / (2 * jnp.pi), 0.0, 1.0)
    elif name == "path_straightness":
        return jnp.clip(raw, 0.0, 1.0)           # already [0, 1]
    else:
        return raw


def extract_all_concepts(
    inp: ConceptInput,
    phases: tuple[int, ...] = (1, 2, 3),
) -> ConceptOutput:
    """Run all registered concept extractors and collate results.

    Args:
        inp: structured observation input.
        phases: which concept phases to include (default: all).

    Returns:
        A ``ConceptOutput`` with one column per concept.
    """
    names = []
    raws = []
    norms = []
    valids = []
    schemas = []

    for name, (schema, fn) in _REGISTRY.items():
        if schema.phase not in phases:
            continue
        raw, valid = fn(inp)
        norm = _normalize_concept(raw, schema)
        names.append(name)
        raws.append(raw)
        norms.append(norm)
        valids.append(valid)
        schemas.append(schema)

    return ConceptOutput(
        names=tuple(names),
        raw=jnp.stack(raws, axis=-1),
        normalized=jnp.stack(norms, axis=-1),
        valid=jnp.stack(valids, axis=-1),
        schemas=tuple(schemas),
    )
