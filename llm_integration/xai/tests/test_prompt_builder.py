# Tests for Module 10: PromptBuilder

import pytest

from xai.narration.prompt_builder import build_prompt


MOCK_GROUNDED_CRITICAL_REPORT = {
    "step": 40,
    "timestamp_s": 4.0,
    "ego_state": {"ego_velocity": 12.5, "ego_action": "moving"},
    "chosen_action": {"label": "Maintain", "accel": 0.0, "steer": 0.0, "outcome": "SAFE"},
    "context_categories": ["following"],
    "alternatives": [
        {
            "label": "Hard Brake",
            "accel": -4.0,
            "steer": 0.0,
            "outcome": "COLLISION",
            "min_ttc": 1.2,
            "threat_agent_id": 5,
        },
    ],
    "necessity_score": 0.75,
    "attention_grounding": {
        "grounding_score": 0.82,
        "per_agent_breakdown": [
            {"agent_id": 5, "severity": 0.45, "attention_mass": 0.38},
        ],
    },
    "decision_class": "GROUNDED_CRITICAL",
}

MOCK_UNGROUNDED_REPORT = {
    **MOCK_GROUNDED_CRITICAL_REPORT,
    "decision_class": "UNGROUNDED_CRITICAL",
    "attention_grounding": {
        "grounding_score": 0.12,
        "per_agent_breakdown": [
            {"agent_id": 5, "severity": 0.45, "attention_mass": 0.02},
        ],
    },
}

MOCK_ROUTINE_REPORT = {
    **MOCK_GROUNDED_CRITICAL_REPORT,
    "decision_class": "ROUTINE",
    "necessity_score": 0.1,
    "alternatives": [
        {"label": "Gentle Brake", "outcome": "SAFE", "min_ttc": None, "threat_agent_id": None},
    ],
    "attention_grounding": {"grounding_score": None, "per_agent_breakdown": []},
}


class TestPromptBuilder:
    def test_detailed_prompt(self):
        sys_p, user_p = build_prompt("detailed", MOCK_GROUNDED_CRITICAL_REPORT)
        assert "Autonomous Driving" in sys_p
        assert "12.5 m/s" in user_p
        assert "Maintain" in user_p
        assert "COLLISION" in user_p or "collision" in user_p.lower()

    def test_detailed_caveat_prompt(self):
        sys_p, user_p = build_prompt("detailed_caveat", MOCK_UNGROUNDED_REPORT)
        assert "TRANSPARENCY CAVEAT" in user_p
        assert "flagged" in user_p.lower()

    def test_brief_prompt(self):
        sys_p, user_p = build_prompt("brief", MOCK_ROUTINE_REPORT)
        assert "ROUTINE" in user_p
        assert len(user_p.split("\n")) <= 5

    def test_system_prompt_constraints(self):
        sys_p, _ = build_prompt("detailed", MOCK_GROUNDED_CRITICAL_REPORT)
        assert "do not speculate" in sys_p.lower()
        assert "3–5 sentences" in sys_p or "3-5 sentences" in sys_p

    def test_unknown_template_falls_back_to_brief(self):
        sys_p, user_p = build_prompt("nonexistent_template", MOCK_ROUTINE_REPORT)
        # Should not crash, falls back to brief
        assert len(user_p) > 0
