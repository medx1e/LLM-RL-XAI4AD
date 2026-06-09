# Copyright 2025 - Report Builder for Attention-Grounded XRL Pipeline
"""
Module 8: ReportBuilder

Assembles outputs from all computation-layer modules into a structured
JSON report and persists it to disk.  This is the **output interface**
of the computation layer.
"""

import json
import os
from typing import Any, Dict, List, Optional


def build(
    step: int,
    timestamp_s: float,
    ego_state: Dict[str, Any],
    chosen_action: Dict[str, Any],
    context_categories: List[str],
    scene_edges: List[Dict[str, Any]],
    alternatives: List[Dict[str, Any]],
    necessity_score: float,
    attention_grounding: Dict[str, Any],
    decision_class: str,
    traffic_light: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assemble a structured report from all computation-layer outputs.

    Args:
        step: Simulation step index.
        timestamp_s: Timestamp in seconds.
        ego_state: ``{ego_velocity, ego_motion_state}`` from SemanticGraphBuilder.
        chosen_action: ``{label, accel, steer, outcome, ...}`` for the
            agent's actual action.
        context_categories: Active scene labels.
        scene_edges: List of semantic edges from SemanticGraphBuilder.
        alternatives: List of alternative rollout outcomes.
        necessity_score: Float in [0, 1].
        attention_grounding: ``{grounding_score, per_agent_breakdown}``
            from AttentionGrounder.
        decision_class: One of the four class labels.
        traffic_light: Optional traffic light state dict with keys
            ``state`` (red/yellow/green/none), ``distance_m``, and
            ``raw_states``.

    Returns:
        Validated report dictionary.

    Raises:
        ValueError: If any required field is missing or malformed.
    """
    # --- Validation ---
    if decision_class not in (
        "GROUNDED_CRITICAL",
        "UNGROUNDED_CRITICAL",
        "GROUNDED_ROUTINE",
        "ROUTINE",
    ):
        raise ValueError(f"Invalid decision_class: {decision_class}")

    if not isinstance(alternatives, list):
        raise ValueError("alternatives must be a list")

    # Keep only the 5 closest agents for the LLM prompt to avoid context bloating
    closest_edges = sorted(scene_edges, key=lambda e: e.get("raw_distance", 999.0))[:5]
    scene_description = []
    for e in closest_edges:
        scene_description.append({
            "agent_id": e["target_id"],
            "distance": round(e.get("raw_distance", 0.0), 1),
            "relations": e.get("relation", []),
            "ttc": round(e.get("ttc"), 2) if e.get("ttc") is not None else None
        })

    report = {
        "step": step,
        "timestamp_s": round(timestamp_s, 2),
        "ego_state": ego_state,
        "chosen_action": chosen_action,
        "context_categories": context_categories,
        "scene_description": scene_description,
        "alternatives": alternatives,
        "necessity_score": round(necessity_score, 4),
        "attention_grounding": attention_grounding,
        "decision_class": decision_class,
    }

    # Include traffic light state if available
    if traffic_light is not None:
        report["traffic_light"] = traffic_light

    return report


def save(
    report: Dict[str, Any],
    output_dir: str,
    scenario_id: str,
    step: int,
    model_name: Optional[str] = None,
) -> str:
    """
    Persist a report to disk as JSON.

    Args:
        report: Report dict from :func:`build`.
        output_dir: Directory to write into.
        scenario_id: Scenario identifier string.
        step: Step index (used in filename).
        model_name: Optional LLM model name for sub-directory.

    Returns:
        Absolute path to the written file.
    """
    if model_name:
        # Sanitize model name (replace slashes for directory safety)
        safe_model = model_name.replace("/", "_")
        reports_dir = os.path.join(output_dir, "reports", safe_model)
    else:
        reports_dir = os.path.join(output_dir, "reports")

    os.makedirs(reports_dir, exist_ok=True)

    filename = f"report_{scenario_id}_step_{step:04d}.json"
    path = os.path.join(reports_dir, filename)

    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return path
