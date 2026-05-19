"""
Module 10: PromptBuilder

Transforms a structured report into the final LLM prompt string,
using template variants selected by decision class.

System prompt constraints (from the specification):
    - Only reference information present in the report — no speculation
    - Never claim to know what the model "might" have been thinking
    - Cap at 3–5 sentences
    - Always cover: (1) what the vehicle did and why,
      (2) what the most dangerous alternative would have caused,
      (3) whether attention confirms the decision
"""

from typing import Any, Dict, List, Tuple

from xai.narration.narration_router import BRIEF, DETAILED, DETAILED_CAVEAT


# =============================================================================
# SHARED SYSTEM PROMPT
# =============================================================================

_SYSTEM_PROMPT = (
    "You are the explanatory module for an Autonomous Driving RL Agent. "
    "You will receive a structured driving report and must produce a "
    "factual narration of the agent's decision. Follow these rules strictly:\n\n"

    "FACTUAL ACCURACY:\n"
    "- Cite exact numeric values from the report (agent IDs, velocities, "
    "TTC values, distances, attention percentages). Do not round, "
    "paraphrase, or approximate — use the numbers as given.\n"
    "- Do not invent, infer, or speculate about any agents, road features, "
    "or events not explicitly listed in the report.\n"
    "- The necessity score represents the fraction of simulated alternative "
    "actions that resulted in failure (collision or off-road). It is NOT a "
    "probability of collision for the chosen action.\n\n"

    "REQUIRED STRUCTURE (3–5 sentences):\n"
    "Sentence 1 — ACTION & CAUSE: State the ego vehicle's speed, the exact "
    "chosen action label, and explain why it was necessary by referencing the "
    "necessity score and the specific threat agents or conditions.\n"
    "Sentence 2 — COUNTERFACTUAL CONTRAST: Name the single most dangerous "
    "alternative action by its exact label, state which agent it would have "
    "collided with (by ID), and cite the exact time-to-collision.\n"
    "Sentence 3 — ATTENTION CONFIRMATION: State the attention grounding score "
    "and identify which threat agents the model's attention was focused on "
    "(by ID and attention mass percentage). Explicitly state whether the "
    "attention distribution confirms or does not confirm awareness of the "
    "threat agents identified in the counterfactual analysis.\n"
    "Sentences 4–5 (optional): Additional context about safe alternatives or "
    "off-road outcomes, only if present in the report.\n\n"

    "PROHIBITED:\n"
    "- Do not claim to know what the model 'might', 'could', or 'probably' "
    "was thinking.\n"
    "- Do not reference road features, lane markings, traffic signals, or "
    "weather unless they appear in the report.\n"
    "- Do not use generic safety language — be specific to this scenario."
)


# =============================================================================
# TEMPLATE BUILDERS
# =============================================================================


def _describe_nearby_agents(report: Dict[str, Any]) -> str:
    """Convert the scene_description dictionary into readable sentences for the LLM."""
    desc_list = report.get("scene_description", [])
    if not desc_list:
        return "  No agents nearby."

    lines = []
    for d in desc_list:
        agent_id = d.get("agent_id")
        dist = d.get("distance", "?")
        rels = d.get("relations", [])
        ttc = d.get("ttc")

        rel_str = ", ".join(rels).replace("_", " ") if rels else "no specific relation"

        sentence = f"  - Agent {agent_id}: {dist}m away ({rel_str})"
        if ttc is not None:
            sentence += f", TTC = {ttc}s"
        lines.append(sentence)

    return "\n".join(lines)


def _describe_alternatives(alts: List[Dict[str, Any]]) -> str:
    """Format ALL alternatives into a clearly labeled list."""
    if not alts:
        return "  No alternatives simulated."

    lines = []
    for a in alts:
        label = a.get("label", "unknown")
        outcome = a.get("outcome", "UNKNOWN")
        entry = f"  - \"{label}\" → {outcome}"
        if outcome == "COLLISION":
            tid = a.get("threat_agent_id", "?")
            ttc = a.get("min_ttc", "?")
            entry += f" with Agent {tid} in {ttc}s"
        lines.append(entry)

    return "\n".join(lines)


def _describe_attention(report: Dict[str, Any]) -> str:
    """Format attention grounding data into a clearly labeled block."""
    grounding = report.get("attention_grounding", {})
    g_score = grounding.get("grounding_score")
    if g_score is None:
        return "  No attention grounding data available."

    breakdown = grounding.get("per_agent_breakdown", [])
    top_agents = sorted(
        breakdown, key=lambda x: x.get("attention_mass", 0), reverse=True
    )[:5]

    lines = [f"  Grounding score: {g_score:.2f}"]
    lines.append("  Per-agent attention breakdown (ranked):")
    for a in top_agents:
        aid = a.get("agent_id", "?")
        mass = a.get("attention_mass", 0)
        is_threat = a.get("is_threat", False)
        threat_tag = " [THREAT]" if is_threat else ""
        lines.append(f"    - Agent {aid}: {mass:.1%} attention mass{threat_tag}")

    return "\n".join(lines)


def _build_detailed(report: Dict[str, Any]) -> str:
    """Full causal explanation for GROUNDED_CRITICAL."""
    ego = report["ego_state"]
    chosen = report["chosen_action"]
    alts = report.get("alternatives", [])
    necessity = report.get("necessity_score", 0)

    scene_str = _describe_nearby_agents(report)
    alts_str = _describe_alternatives(alts)
    attn_str = _describe_attention(report)

    # Find worst alternative for explicit highlighting
    collisions = [a for a in alts if a.get("outcome") == "COLLISION"]
    worst_line = ""
    if collisions:
        worst = min(collisions, key=lambda a: a.get("min_ttc") or 999)
        worst_line = (
            f"  ⚠ MOST DANGEROUS: \"{worst.get('label')}\" → "
            f"COLLISION with Agent {worst.get('threat_agent_id')} "
            f"in {worst.get('min_ttc', '?')}s"
        )

    prompt = (
        f"=== DRIVING REPORT ===\n\n"

        f"[EGO STATE]\n"
        f"  Motion State: {ego.get('ego_motion_state', 'moving')}\n"
        f"  Velocity: {ego.get('ego_velocity', '?')} m/s\n\n"

        f"[NEARBY AGENTS]\n"
        f"{scene_str}\n\n"

        f"[CHOSEN ACTION]\n"
        f"  Label: {chosen.get('label', 'unknown')}\n"
        f"  Outcome: {chosen.get('outcome', 'SAFE')}\n\n"

        f"[NECESSITY SCORE]\n"
        f"  {necessity:.0%} of simulated alternative actions resulted in failure.\n\n"

        f"[COUNTERFACTUAL ALTERNATIVES]\n"
        f"{alts_str}\n"
    )

    if worst_line:
        prompt += f"\n{worst_line}\n"

    prompt += (
        f"\n[ATTENTION GROUNDING]\n"
        f"{attn_str}\n\n"
        f"=== END REPORT ===\n\n"
        f"Using ONLY the data above, narrate the agent's decision."
    )

    return prompt


def _build_detailed_with_caveat(report: Dict[str, Any]) -> str:
    """Full causal explanation + transparency caveat for UNGROUNDED_CRITICAL."""
    base = _build_detailed(report)

    caveat = (
        "\n\n⚠️ CRITICAL NOTE: The attention grounding score is LOW — the "
        "agent's attention was NOT focused on the identified threat agents. "
        "Your narration MUST explicitly state that the attention distribution "
        "does not confirm awareness of the threats, and that the safe outcome "
        "may be coincidental. Do not present the decision as intentionally safe."
    )
    return base + caveat


def _build_brief(report: Dict[str, Any]) -> str:
    """Brief narration for GROUNDED_ROUTINE / ROUTINE."""
    ego = report["ego_state"]
    chosen = report["chosen_action"]
    alts = report.get("alternatives", [])
    necessity = report.get("necessity_score", 0)
    decision_class = report.get("decision_class", "ROUTINE")

    scene_str = _describe_nearby_agents(report)
    alts_str = _describe_alternatives(alts)
    attn_str = _describe_attention(report)

    prompt = (
        f"=== DRIVING REPORT ===\n\n"

        f"[EGO STATE]\n"
        f"  Motion State: {ego.get('ego_motion_state', 'moving')}\n"
        f"  Velocity: {ego.get('ego_velocity', '?')} m/s\n\n"

        f"[NEARBY AGENTS]\n"
        f"{scene_str}\n\n"

        f"[CHOSEN ACTION]\n"
        f"  Label: {chosen.get('label', 'unknown')}\n"
        f"  Decision class: {decision_class}\n\n"

        f"[NECESSITY SCORE]\n"
        f"  {necessity:.0%} of simulated alternative actions resulted in failure.\n\n"

        f"[COUNTERFACTUAL ALTERNATIVES]\n"
        f"{alts_str}\n\n"

        f"[ATTENTION GROUNDING]\n"
        f"{attn_str}\n\n"

        f"=== END REPORT ===\n\n"
        f"Using ONLY the data above, narrate the agent's decision."
    )

    return prompt


# =============================================================================
# PUBLIC API
# =============================================================================

_TEMPLATE_MAP = {
    DETAILED: _build_detailed,
    DETAILED_CAVEAT: _build_detailed_with_caveat,
    BRIEF: _build_brief,
}


def build_prompt(
    template_key: str,
    report: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Build the ``(system_prompt, user_prompt)`` pair for the LLM.

    Args:
        template_key: One of ``"detailed"``, ``"detailed_caveat"``, ``"brief"``.
        report: Structured report dict.

    Returns:
        Tuple of ``(system_prompt, user_prompt)`` strings.
    """
    builder = _TEMPLATE_MAP.get(template_key, _build_brief)
    user_prompt = builder(report)
    return _SYSTEM_PROMPT, user_prompt

