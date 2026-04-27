"""Core data types for concept extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import jax
import jax.numpy as jnp

from concepts.schema import ConceptSchema


# ---------------------------------------------------------------------------
# Observation configuration (mirrors the V-Max observation_config contract)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObservationConfig:
    """Parameters that define the V-Max observation contract.

    These must match the model's training observation_config exactly,
    because the concept extractor needs them to un-normalise and slice
    the observation tensors correctly.
    """
    obs_past_num_steps: int = 5
    num_closest_objects: int = 8
    roadgraph_top_k: int = 200
    num_closest_traffic_lights: int = 5
    num_target_path_points: int = 10
    max_meters: float = 70.0
    max_speed: float = 30.0
    # Per-timestep feature counts (after mask removal by unflatten)
    object_feature_dim: int = 7   # xy(2)+vel_xy(2)+yaw(1)+length(1)+width(1)
    roadgraph_feature_dim: int = 4  # xy(2)+dir_xy(2)
    tl_feature_dim: int = 10       # xy(2)+state_onehot(8)
    path_feature_dim: int = 2      # xy(2)
    dt: float = 0.1               # simulation timestep (seconds)


# ---------------------------------------------------------------------------
# Concept I/O
# ---------------------------------------------------------------------------

@dataclass
class ConceptInput:
    """Structured representation of observation tensors for concept extraction.

    All arrays are JAX arrays.  Shapes use ``B`` for arbitrary leading batch
    dims (may be empty for a single sample).

    This is the **only** input a concept extractor sees.  It must contain
    exactly the information visible in the V-Max observation—nothing more.
    """
    # SDC ego trajectory:  (..., 1, T, F_obj)
    sdc_features: jax.Array
    sdc_mask: jax.Array           # (..., 1, T)

    # Other agents:        (..., N_agents, T, F_obj)
    agent_features: jax.Array
    agent_mask: jax.Array         # (..., N_agents, T)

    # Roadgraph:           (..., N_rg, F_rg)
    roadgraph_features: jax.Array
    roadgraph_mask: jax.Array     # (..., N_rg)

    # Traffic lights:      (..., N_tl, T, F_tl)
    tl_features: jax.Array
    tl_mask: jax.Array            # (..., N_tl, T)

    # Path target:         (..., N_path, 2)
    path_features: jax.Array

    # Config used for denormalization
    config: ObservationConfig = field(default_factory=ObservationConfig)


@dataclass
class ConceptOutput:
    """Output of concept extraction for one observation (or batch).

    All value arrays have shape ``(..., C)`` where ``C = len(names)``.
    """
    names: Sequence[str]              # concept names in column order
    raw: jax.Array                    # (..., C) raw (denormalised) values
    normalized: jax.Array             # (..., C) values in [0, 1] or {0, 1}
    valid: jax.Array                  # (..., C) boolean mask
    schemas: Sequence[ConceptSchema]  # metadata for each concept
