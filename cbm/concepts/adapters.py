"""Adapters: convert V-Max observation tensors into ConceptInput.

This is the ONLY place that knows how V-Max structures its flat observation
vector.  The rest of the concept module operates solely on ConceptInput.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from concepts.types import ConceptInput, ObservationConfig


def observation_to_concept_input(
    flat_obs: jax.Array,
    unflatten_fn: Callable,
    config: ObservationConfig | None = None,
) -> ConceptInput:
    """Convert a flat V-Max observation vector into a ConceptInput.

    Args:
        flat_obs: the flat observation tensor as produced by the
            ObservationWrapper.  Shape ``(..., obs_size)``.
        unflatten_fn: the ``features_extractor.unflatten_features`` method
            that reconstructs structured feature tensors + masks.
        config: observation configuration; defaults to the standard
            ``womd_sac_road_perceiver_minimal_42`` contract.

    Returns:
        A ``ConceptInput`` ready for concept extraction.
    """
    if config is None:
        config = ObservationConfig()

    features, masks = unflatten_fn(flat_obs)
    sdc_feat, agent_feat, rg_feat, tl_feat, path_feat = features
    sdc_mask, agent_mask, rg_mask, tl_mask = masks

    return ConceptInput(
        sdc_features=sdc_feat,
        sdc_mask=sdc_mask,
        agent_features=agent_feat,
        agent_mask=agent_mask,
        roadgraph_features=rg_feat,
        roadgraph_mask=rg_mask,
        tl_features=tl_feat,
        tl_mask=tl_mask,
        path_features=path_feat,
        config=config,
    )


def structured_to_concept_input(
    sdc_features: jax.Array,
    sdc_mask: jax.Array,
    agent_features: jax.Array,
    agent_mask: jax.Array,
    roadgraph_features: jax.Array,
    roadgraph_mask: jax.Array,
    tl_features: jax.Array,
    tl_mask: jax.Array,
    path_features: jax.Array,
    config: ObservationConfig | None = None,
) -> ConceptInput:
    """Build a ConceptInput directly from already-unflattened tensors.

    Useful for the online RL path where the environment already provides
    structured observations, or for unit tests.
    """
    if config is None:
        config = ObservationConfig()

    return ConceptInput(
        sdc_features=sdc_features,
        sdc_mask=sdc_mask,
        agent_features=agent_features,
        agent_mask=agent_mask,
        roadgraph_features=roadgraph_features,
        roadgraph_mask=roadgraph_mask,
        tl_features=tl_features,
        tl_mask=tl_mask,
        path_features=path_features,
        config=config,
    )
