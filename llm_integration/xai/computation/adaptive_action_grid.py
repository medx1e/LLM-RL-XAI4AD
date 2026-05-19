# Copyright 2025 - Adaptive Action Grid for Attention-Grounded XRL Pipeline
"""
Module 3: AdaptiveActionGrid

Given active context categories, construct the set of alternative actions
to evaluate via counterfactual rollouts. Pure data transformation — no
simulation happens here.
"""

from typing import Any, Dict, List


# Minimal fallback grid when no templates match
_DEFAULT_GRID = [
    {"label": "Maintain", "accel": 0.0, "steer": 0.0},
    {"label": "Gentle Brake", "accel": -1.5, "steer": 0.0},
    {"label": "Gentle Accelerate", "accel": 1.0, "steer": 0.0},
]


def build_action_grid(
    context_categories: List[str],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Build a deduplicated action grid from the union of all templates
    triggered by the active context categories.

    Args:
        context_categories: Active scene labels (e.g. ``["following", "threat_approaching"]``).
        config: Full pipeline config dict (must contain ``action_templates`` key).

    Returns:
        List of ``{label, accel, steer}`` dicts ready for the rollout engine.
    """
    templates = config.get("action_templates", {})

    seen: set = set()
    grid: List[Dict[str, Any]] = []

    for category in context_categories:
        category_actions = templates.get(category, [])
        for action in category_actions:
            key = (action["accel"], action["steer"])
            if key not in seen:
                seen.add(key)
                grid.append(
                    {
                        "label": action["label"],
                        "accel": action["accel"],
                        "steer": action["steer"],
                    }
                )

    # Fallback: ensure at least a minimal grid
    if not grid:
        grid = list(_DEFAULT_GRID)

    return grid
