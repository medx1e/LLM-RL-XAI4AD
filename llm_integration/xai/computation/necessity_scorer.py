"""
Module 5: NecessityScorer

Takes counterfactual rollout outcomes and produces a necessity score
plus a deduplicated list of threat agents.
"""

from typing import Any, Dict, List


def compute(alternatives: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute the necessity score and extract threat agents.

    Args:
        alternatives: List of rollout outcome dicts, each with at least:
            - ``outcome``: one of ``"SAFE"``, ``"COLLISION"``, ``"OFFROAD"``
            - ``threat_agent_id``: int or None
            - ``min_ttc``: float or None (seconds)

    Returns:
        Dictionary with:
            - ``necessity_score``: float in [0, 1]
            - ``threat_agents``: list of ``{agent_id, min_ttc}`` dicts
    """
    if not alternatives:
        return {"necessity_score": 0.0, "threat_agents": []}

    failing = [a for a in alternatives if a.get("outcome") != "SAFE"]
    necessity_score = len(failing) / len(alternatives)

    # Deduplicate threat agents, keeping minimum TTC per agent
    threat_map: Dict[int, float] = {}
    for alt in failing:
        aid = alt.get("threat_agent_id")
        ttc = alt.get("min_ttc")
        if aid is not None and ttc is not None:
            if aid not in threat_map or ttc < threat_map[aid]:
                threat_map[aid] = ttc

    threat_agents = [
        {"agent_id": aid, "min_ttc": round(ttc, 3)}
        for aid, ttc in sorted(threat_map.items())
    ]

    return {
        "necessity_score": round(necessity_score, 4),
        "threat_agents": threat_agents,
    }
