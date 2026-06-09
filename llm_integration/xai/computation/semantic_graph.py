# Copyright 2025 - Semantic Graph Builder for Attention-Grounded XRL Pipeline
"""
Module 2: SemanticGraphBuilder

Converts raw simulator state into a structured scene description with
semantic edges and context categories for downstream modules.

Outputs:
    - agent_graph: nodes (agents) + edges (semantic relations)
    - context_categories: list of active labels (following, threat_approaching, etc.)
    - ego_state_summary: speed, action state, heading
"""

import math
from typing import Any, Dict, List, Tuple


# =============================================================================
# CONTEXT CATEGORY CONSTANTS
# =============================================================================

CONTEXT_FOLLOWING = "following"
CONTEXT_THREAT_APPROACHING = "threat_approaching"
CONTEXT_FREE_FLOW = "free_flow"
CONTEXT_NEAR_BOUNDARY = "near_boundary"


class SemanticGraphBuilder:
    """Build a semantic graph from simulator state and classify the scene context."""

    def __init__(self, lane_width_m: float = 2.0):
        self.lane_width_m = lane_width_m

    # --------------------------------------------------------------------- #
    # Coordinate helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def ego_frame(dx: float, dy: float, ego_yaw: float) -> Tuple[float, float]:
        """Convert world delta (dx,dy) into ego-local coords: x_forward, y_left."""
        c = math.cos(ego_yaw)
        s = math.sin(ego_yaw)
        x_local =  c * dx + s * dy
        y_local = -s * dx + c * dy
        return x_local, y_local

    @staticmethod
    def distance_bucket(dist: float) -> str:
        if dist < 5.0:
            return "very_close"
        if dist < 15.0:
            return "close"
        if dist < 30.0:
            return "medium"
        return "far"

    @staticmethod
    def ttc_bucket(ttc: float) -> str:
        if ttc < 1.0:
            return "ttc_imminent"
        if ttc < 3.0:
            return "ttc_soon"
        if ttc < 6.0:
            return "ttc_later"
        return "ttc_far"

    @staticmethod
    def compute_closing_speed(
        ex, ey, evx, evy, ox, oy, ovx, ovy
    ) -> Tuple[float, float]:
        import numpy as np

        rel = np.array([ox - ex, oy - ey], dtype=np.float32)
        dist = float(np.linalg.norm(rel) + 1e-6)
        ego_v = np.array([evx, evy], dtype=np.float32)
        oth_v = np.array([ovx, ovy], dtype=np.float32)
        rel_v = oth_v - ego_v
        closing = float(-np.dot(rel, rel_v) / dist)
        return closing, dist

    @staticmethod
    def determine_action_state(vx: float, vy: float) -> str:
        """Classify the ego's observed motion into a descriptive kinematic state.

        This describes the **current motion** of the vehicle (an observation),
        NOT the action the policy has chosen.  Finer-grained buckets help
        downstream LLM narration avoid contradictions such as a ``stopped``
        vehicle that is about to accelerate.
        """
        speed = math.sqrt(vx**2 + vy**2)
        if speed < 0.1:
            return "stopped"
        if speed < 1.0:
            return "creeping"
        if speed < 5.0:
            return "slow"
        if speed < 15.0:
            return "cruising"
        return "fast"

    # --------------------------------------------------------------------- #
    # Core graph builder
    # --------------------------------------------------------------------- #

    def build(self, state_dict: Dict[str, Any], ego_idx: int) -> Dict[str, Any]:
        """
        Build the semantic graph from a state dictionary.

        Args:
            state_dict: Dictionary with ``current`` key containing arrays
                        for x, y, vx, vy, yaw, valid.
            ego_idx: Index of the ego vehicle.

        Returns:
            Dictionary with ``scenario_info``, ``context_nodes``,
            ``semantic_edges``, and ``context_categories``.
        """
        cur = state_dict["current"]
        N = len(cur["x"])

        # 1. Ego data
        ex, ey = float(cur["x"][ego_idx]), float(cur["y"][ego_idx])
        evx, evy = float(cur["vx"][ego_idx]), float(cur["vy"][ego_idx])
        ego_yaw = float(cur["yaw"][ego_idx])
        ego_speed = math.sqrt(evx**2 + evy**2)

        scenario_info = {
            "ego_velocity": round(ego_speed, 2),
            "ego_motion_state": self.determine_action_state(evx, evy),
        }

        context_nodes: List[Dict[str, Any]] = []
        semantic_edges: List[Dict[str, Any]] = []
        valid_indices = [i for i in range(N) if cur["valid"][i]]

        # 2. Build edges & nodes
        for j in valid_indices:
            if j == ego_idx:
                continue

            ox, oy = float(cur["x"][j]), float(cur["y"][j])
            ovx, ovy = float(cur["vx"][j]), float(cur["vy"][j])
            otype = "vehicle"

            closing, dist = self.compute_closing_speed(
                ex, ey, evx, evy, ox, oy, ovx, ovy
            )
            dx, dy = ox - ex, oy - ey
            x_local, y_local = self.ego_frame(dx, dy, ego_yaw)

            relations: List[str] = [self.distance_bucket(dist)]

            # Spatial (v2 style)
            relations.append("ahead_of" if x_local > 0 else "behind_of")
            relations.append("left_of" if y_local > 0 else "right_of")

            # Dynamic
            ttc_val = None
            if closing > 0.5:
                relations.append("approaching")
                ttc_val = dist / max(closing, 1e-6)
                relations.append(self.ttc_bucket(ttc_val))
            elif closing < -0.5:
                relations.append("moving_away")

            # Lane context (Leading/Following match v2)
            if abs(y_local) < self.lane_width_m:
                relations.append("in_ego_lane")
                if x_local > 0:
                    relations.append("leading")
                elif x_local < 0:
                    relations.append("following")

            semantic_edges.append(
                {
                    "target_id": int(j),
                    "target_type": otype,
                    "relation": relations,
                    "raw_distance": dist,
                    "ttc": ttc_val,
                }
            )
            context_nodes.append(
                {
                    "id": int(j),
                    "type": otype,
                    "distance_to_ego": round(dist, 1),
                    "speed": round(math.sqrt(ovx**2 + ovy**2), 1),
                }
            )

        # 3. Derive context categories
        context_categories = self._detect_context_categories(
            semantic_edges, context_nodes
        )

        return {
            "scenario_info": scenario_info,
            "context_nodes": context_nodes,
            "semantic_edges": semantic_edges,
            "context_categories": context_categories,
        }

    # --------------------------------------------------------------------- #
    # Context category detection
    # --------------------------------------------------------------------- #

    @staticmethod
    def _detect_context_categories(
        edges: List[Dict[str, Any]],
        nodes: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Classify the current scene into one or more context categories.

        Rules:
            - ``following``:          any agent in_ego_lane + ahead + close/very_close
            - ``threat_approaching``: any agent approaching + collision_likely/imminent
            - ``free_flow``:          no agent in_ego_lane within 30 m
            - ``near_boundary``:      (future) ego lateral offset > threshold
        """
        categories: List[str] = []

        has_in_lane_ahead_close = False
        has_threat = False
        has_in_lane_within_30 = False

        for edge in edges:
            rels = set(edge["relation"])

            # Following check
            if (
                "in_ego_lane" in rels
                and "ahead_of" in rels
                and ("close" in rels or "very_close" in rels)
            ):
                has_in_lane_ahead_close = True

            # Threat check
            if "approaching" in rels and (
                "ttc_soon" in rels or "ttc_imminent" in rels
            ):
                has_threat = True

            # Free-flow check
            if "in_ego_lane" in rels and edge["raw_distance"] < 30.0:
                has_in_lane_within_30 = True

        if has_in_lane_ahead_close:
            categories.append(CONTEXT_FOLLOWING)
        if has_threat:
            categories.append(CONTEXT_THREAT_APPROACHING)
        if not has_in_lane_within_30:
            categories.append(CONTEXT_FREE_FLOW)

        # Fallback: always have at least free_flow
        if not categories:
            categories.append(CONTEXT_FREE_FLOW)

        return categories
