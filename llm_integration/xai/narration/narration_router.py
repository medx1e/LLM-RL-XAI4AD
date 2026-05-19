# Copyright 2025 - Narration Router for Attention-Grounded XRL Pipeline
"""
Module 9: NarrationRouter

Reads a report JSON and selects the correct prompt template based on
the ``decision_class`` field.
"""

from typing import Any, Dict, Tuple


# Template key constants
DETAILED = "detailed"
DETAILED_CAVEAT = "detailed_caveat"
BRIEF = "brief"

# Dispatch map
_ROUTE_MAP = {
    "GROUNDED_CRITICAL": DETAILED,
    "UNGROUNDED_CRITICAL": DETAILED_CAVEAT,
    "GROUNDED_ROUTINE": BRIEF,
    "ROUTINE": BRIEF,
}


def route(report: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Select the prompt template for a given report.

    Args:
        report: Structured report dict (must contain ``decision_class``).

    Returns:
        Tuple of ``(template_key, report)``.
    """
    decision_class = report.get("decision_class", "ROUTINE")
    template_key = _ROUTE_MAP.get(decision_class, BRIEF)
    return template_key, report
