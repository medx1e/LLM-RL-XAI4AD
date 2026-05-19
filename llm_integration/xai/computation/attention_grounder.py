# Copyright 2025 - Attention Grounder for Attention-Grounded XRL Pipeline
"""
Module 6: AttentionGrounder

Given raw cross-attention weights from the LQ/Perceiver encoder and the
threat agents identified by the NecessityScorer, compute the Attention
Grounding Score — a measure of whether the model was actually attending
to the agents that would have caused failures.

IMPORTANT: The encoder's token sequence for "other objects" is ordered by
distance to the SDC (closest first), NOT by original simulator agent ID.
Therefore a ``token_agent_ids`` mapping must be supplied to translate
between token positions and the original agent IDs used by the
NecessityScorer / CounterfactualExplainer.
"""

from typing import Any, Dict, List, Optional

import numpy as np


def compute(
    attention_weights: Dict[str, Any],
    encoder_layout: Dict[str, Any],
    threat_agents: List[Dict[str, Any]],
    config: Dict[str, Any],
    token_agent_ids: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute the attention grounding score.

    Args:
        attention_weights: Dict from LQEncoder. Expected key (from config):
            e.g. ``cross_attn_0`` → array of shape ``[B, N_heads, N_latents, N_tokens]``
            or ``[N_heads, N_latents, N_tokens]`` (single sample).
        encoder_layout: Dict with ``n_sdc_timesteps``, ``num_objects``,
            ``timestep_agent``, ``roadgraph_top_k``, ``num_traffic_lights``,
            ``tl_timesteps``, ``gps_path_len``.
        threat_agents: List of ``{agent_id, min_ttc}`` dicts from NecessityScorer.
            ``agent_id`` is the *original simulator index*.
        config: Pipeline config dict (needs ``attention_layer_key``).
        token_agent_ids: 1-D integer array of length ``num_objects`` where
            ``token_agent_ids[j]`` is the original simulator agent ID that
            occupies token-position *j* in the encoder's "other objects"
            region.  Obtained by sorting agents by distance to SDC
            (closest first) — the same order the feature extractor uses.
            If ``None``, falls back to the identity mapping (position == ID),
            which is only correct if the feature extractor does not re-order
            agents.

    Returns:
        Dictionary with:
            - ``grounding_score``: float or None (if no threat agents)
            - ``per_agent_breakdown``: list of ``{agent_id, severity, attention_mass}``
    """
    if not threat_agents:
        return {"grounding_score": None, "per_agent_breakdown": []}

    # 1. Get the cross-attention weights for the target layer
    layer_key = config.get("attention_layer_key", "cross_attn_0")
    attn = attention_weights.get(layer_key)
    if attn is None:
        return {"grounding_score": None, "per_agent_breakdown": []}

    attn = np.asarray(attn)

    # Handle batch dimension: use first sample if batched
    if attn.ndim == 4:
        attn = attn[0]  # → 3D

    # Normalize axis order to [N_heads, N_latents, N_tokens].
    # LQEncoder returns [N_latents, N_tokens, N_heads] where N_tokens is
    # the largest dimension (input token count).  Detect and transpose.
    if attn.ndim == 3 and attn.shape[2] < attn.shape[1]:
        # Current: [N_latents, N_tokens, N_heads] → target: [N_heads, N_latents, N_tokens]
        attn = attn.transpose(2, 0, 1)

    # 2. Aggregate across heads and latents → per-token attention mass
    # Shape after transpose: [N_heads, N_latents, N_tokens]
    # Sum over heads (axis 0) and latents (axis 1) → [N_tokens]
    per_token_mass = attn.sum(axis=(0, 1))

    # 3. Compute token boundaries for the vehicle region
    n_sdc = encoder_layout.get("n_sdc_timesteps", 11)  # SDC tokens
    num_objects = encoder_layout.get("num_objects", 64)
    timestep_agent = encoder_layout.get("timestep_agent", 11)

    vehicles_start = n_sdc

    # 4. Compute per-slot attention mass
    #    Slot j (token position j in the other-objects region) occupies
    #    tokens [vehicles_start + j*T .. vehicles_start + (j+1)*T)
    per_slot_mass = np.zeros(num_objects, dtype=np.float64)
    for j in range(num_objects):
        start = vehicles_start + j * timestep_agent
        end = start + timestep_agent
        if end <= len(per_token_mass):
            per_slot_mass[j] = per_token_mass[start:end].sum()

    # 5. Normalize to sum to 1 across all slots
    total = per_slot_mass.sum()
    if total > 0:
        per_slot_mass = per_slot_mass / total

    # 6. Build reverse mapping: original_agent_id → token slot index
    #    token_agent_ids[slot] = original_agent_id
    #    We need: original_agent_id → slot
    if token_agent_ids is not None:
        token_agent_ids = np.asarray(token_agent_ids).ravel()
        # Build dict: agent_id → slot_index
        aid_to_slot = {}
        for slot_idx, orig_id in enumerate(token_agent_ids[:num_objects]):
            orig_id_int = int(orig_id)
            if orig_id_int not in aid_to_slot:
                aid_to_slot[orig_id_int] = slot_idx
    else:
        # Fallback: identity mapping (slot index == agent ID)
        aid_to_slot = {j: j for j in range(num_objects)}

    # 7. Compute severity and grounding score
    breakdown = []
    grounding_score = 0.0

    for ta in threat_agents:
        aid = ta["agent_id"]
        min_ttc = ta["min_ttc"]
        severity = 1.0 / (1.0 + min_ttc)

        slot = aid_to_slot.get(aid)
        if slot is not None and 0 <= slot < num_objects:
            a_mass = float(per_slot_mass[slot])
        else:
            # Agent was not among the closest objects fed to the encoder
            a_mass = 0.0

        grounding_score += a_mass * severity
        breakdown.append(
            {
                "agent_id": aid,
                "severity": round(severity, 4),
                "attention_mass": round(a_mass, 6),
            }
        )

    return {
        "grounding_score": round(float(grounding_score), 4),
        "per_agent_breakdown": breakdown,
    }


def reconstruct_token_agent_ids(
    state,
    ego_idx: int,
    num_closest_objects: int,
) -> np.ndarray:
    """Reconstruct the distance-sorted agent IDs that the feature extractor
    used when assembling the encoder's token sequence.

    This replicates the logic from
    ``VecFeaturesExtractor._build_objects_features``:

    1. Compute Euclidean distance from the SDC to every agent at the
       *current* timestep.
    2. Sort by distance ascending, select the ``num_closest_objects + 1``
       closest (including the SDC itself).
    3. Drop the SDC (index 0 after sort) — the remaining IDs, in order,
       are the original simulator agent IDs assigned to each token slot.

    Args:
        state: JAX SimulatorState (possibly batched with leading dim 1).
        ego_idx: Index of the SDC agent in the object dimension.
        num_closest_objects: Number of other (non-SDC) objects the feature
            extractor keeps — corresponds to
            ``objects_config["num_closest_objects"]``.

    Returns:
        1-D int array of length ``num_closest_objects``, where entry *j* is
        the original simulator agent ID occupying token-slot *j*.
    """
    import jax as _jax

    # ── Extract current-timestep positions ──
    traj = state.sim_trajectory
    timestep_raw = _jax.device_get(state.timestep)
    timestep_arr = np.array(timestep_raw)
    idx = int(timestep_arr[0]) if timestep_arr.ndim >= 1 else int(timestep_arr)

    def _get_field(field):
        val = np.array(_jax.device_get(field))
        if val.ndim == 3:          # [B, N_agents, T]
            return val[0, :, idx]
        elif val.ndim == 2:        # [N_agents, T]
            return val[:, idx]
        return val

    cur_x = _get_field(traj.x)
    cur_y = _get_field(traj.y)
    cur_valid = _get_field(traj.valid).astype(bool)

    n_agents = len(cur_x)

    # ── Compute distance to SDC (ego) ──
    ego_x, ego_y = float(cur_x[ego_idx]), float(cur_y[ego_idx])
    dx = cur_x - ego_x
    dy = cur_y - ego_y
    distances = np.sqrt(dx ** 2 + dy ** 2)

    # Invalid agents get infinite distance so they sort to the end
    distances = np.where(cur_valid, distances, np.inf)

    # ── Sort ascending by distance, pick top-(num_closest_objects + 1) ──
    # top_k maximizes, so negate for ascending order
    k = min(num_closest_objects + 1, n_agents)
    closest_idxs = np.argsort(distances)[:k]

    # ── Drop the SDC ──
    # The SDC should be index 0 (distance 0), but let's be defensive
    # and remove it wherever it appears in the sorted list.
    closest_idxs = closest_idxs[closest_idxs != ego_idx]

    # Ensure exactly num_closest_objects entries (pad with -1 if needed)
    if len(closest_idxs) < num_closest_objects:
        pad = np.full(num_closest_objects - len(closest_idxs), -1, dtype=closest_idxs.dtype)
        closest_idxs = np.concatenate([closest_idxs, pad])
    else:
        closest_idxs = closest_idxs[:num_closest_objects]

    return closest_idxs
