"""Configuration for reward-conditioned attention analysis.

Token structure (280 total, LQ encoder):
    sdc          tokens [0:5]    — 1 ego   × 5 timesteps
    other_agents tokens [5:45]   — 8 agents × 5 timesteps
    roadgraph    tokens [45:245] — 200 road points × 1
    traffic_lights tokens [245:270] — 5 lights × 5 timesteps
    gps_path     tokens [270:280] — 10 waypoints × 1

Verified by probe.py:  Dense_1 (K projection) shape = (1, 280, 32)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Token ranges (start inclusive, end exclusive)
# ---------------------------------------------------------------------------

TOKEN_RANGES: dict[str, tuple[int, int]] = {
    "sdc":             (0,   5),
    "other_agents":    (5,   45),
    "roadgraph":       (45,  245),
    "traffic_lights":  (245, 270),
    "gps_path":        (270, 280),
}

N_TOKENS = 280
N_QUERIES = 16  # num_latents (learned latent queries)
N_HEADS = 2     # cross_num_heads
HEAD_DIM = 16   # cross_head_features
N_AGENTS = 8    # other agents per step
N_LIGHTS = 5
N_GPS = 10

# Tokens per entity in each category
TOKENS_PER_ENTITY: dict[str, int] = {
    "sdc":             5,   # 1 agent × 5 timesteps
    "other_agents":    5,   # each agent has 5 timesteps
    "roadgraph":       1,   # each road point is one token
    "traffic_lights":  5,   # each light has 5 timesteps
    "gps_path":        1,   # each waypoint is one token
}

# ---------------------------------------------------------------------------
# Risk thresholds
# ---------------------------------------------------------------------------

# TTC in normalized coordinate space (not real-world meters)
# The VMaxAdapter provides normalized TTC values; 3.0 maps to full risk
TTC_THRESHOLD: float = 3.0

# Acceleration threshold for behavior_risk (normalized coordinates)
# Empirically calibrated: |accel| > ~0.3 in normalized space → high risk
ACCEL_THRESHOLD: float = 0.3

# ---------------------------------------------------------------------------
# AnalysisConfig
# ---------------------------------------------------------------------------


@dataclass
class AnalysisConfig:
    """Configuration for one experiment run."""

    model_path: str = "runs_rlc/womd_sac_road_perceiver_complete_42"
    data_path: str = "data/training.tfrecord"
    n_scenarios: int = 50
    output_dir: str = "results/reward_attention/womd_sac_road_perceiver_complete_42"

    # Risk thresholds
    ttc_threshold: float = TTC_THRESHOLD
    accel_threshold: float = ACCEL_THRESHOLD

    # Attention aggregation: which encoder layer to use
    # 'avg' → average all 4 layers; 0-3 → specific layer
    attention_layer: str = "avg"

    # Subgroup analysis thresholds
    high_risk_threshold: float = 0.5
    braking_threshold: float = 0.15   # |accel| in normalized space
    steering_threshold: float = 0.1   # |steering| in normalized space

    # Random seed for scenario shuffling
    seed: int = 42


# ---------------------------------------------------------------------------
# TimestepRecord
# ---------------------------------------------------------------------------


@dataclass
class TimestepRecord:
    """One timestep of attention + risk data.

    Attention values (attn_*) are in [0, 1] and represent the fraction of
    total attention directed at each token category (averaged over queries,
    then summed over category tokens, normalized so categories sum to 1.0).

    Risk values (risk_*) are in [0, 1].
    """

    # Identifiers
    scenario_id: int = 0
    timestep: int = 0

    # --- Attention per category (normalized fractions, sum ≈ 1.0) ---
    attn_sdc: float = 0.0
    attn_agents: float = 0.0
    attn_roadgraph: float = 0.0
    attn_lights: float = 0.0
    attn_gps: float = 0.0

    # --- Per-agent attention (8 floats, not normalized to category sum) ---
    # attn_per_agent[i] = fraction of total attention on agent i (unnormalized)
    attn_per_agent: list[float] = field(default_factory=lambda: [0.0] * N_AGENTS)

    # --- Attention to specific agents ---
    attn_to_nearest: float = 0.0   # attention on the nearest valid agent
    attn_to_threat: float = 0.0    # attention on the agent with lowest TTC

    # --- Risk metrics [0, 1] ---
    collision_risk: float = 0.0
    safety_risk: float = 0.0
    navigation_risk: float = 0.0
    behavior_risk: float = 0.0

    # --- Raw physical quantities (normalized coordinate space) ---
    min_ttc: float = float("inf")
    accel: float = 0.0
    steering: float = 0.0
    ego_speed: float = 0.0

    # --- Derived features ---
    num_valid_agents: int = 0
    nearest_agent_id: int = -1   # index 0-7 in other_agents
    threat_agent_id: int = -1    # agent with minimum TTC

    # --- Event flags ---
    is_collision_step: bool = False
    is_offroad_step: bool = False


# ---------------------------------------------------------------------------
# Helper: get agent token indices
# ---------------------------------------------------------------------------


def get_agent_token_range(agent_idx: int) -> tuple[int, int]:
    """Return (start, end) token indices for agent i (0-indexed, 0-7)."""
    base = TOKEN_RANGES["other_agents"][0]
    start = base + agent_idx * TOKENS_PER_ENTITY["other_agents"]
    end = start + TOKENS_PER_ENTITY["other_agents"]
    return (start, end)
