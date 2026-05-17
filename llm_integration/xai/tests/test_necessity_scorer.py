# Tests for Module 5: NecessityScorer

import pytest

from xai.computation import necessity_scorer


class TestNecessityScorer:
    def test_all_safe(self):
        alts = [
            {"outcome": "SAFE", "threat_agent_id": None, "min_ttc": None},
            {"outcome": "SAFE", "threat_agent_id": None, "min_ttc": None},
            {"outcome": "SAFE", "threat_agent_id": None, "min_ttc": None},
        ]
        result = necessity_scorer.compute(alts)
        assert result["necessity_score"] == 0.0
        assert result["threat_agents"] == []

    def test_all_collision(self):
        alts = [
            {"outcome": "COLLISION", "threat_agent_id": 5, "min_ttc": 1.2},
            {"outcome": "COLLISION", "threat_agent_id": 5, "min_ttc": 0.8},
            {"outcome": "COLLISION", "threat_agent_id": 12, "min_ttc": 2.0},
        ]
        result = necessity_scorer.compute(alts)
        assert result["necessity_score"] == 1.0
        # Deduplicate: agent 5 keeps min(1.2, 0.8) = 0.8
        agents = {a["agent_id"]: a["min_ttc"] for a in result["threat_agents"]}
        assert agents[5] == 0.8
        assert agents[12] == 2.0

    def test_mixed_outcomes(self):
        alts = [
            {"outcome": "SAFE", "threat_agent_id": None, "min_ttc": None},
            {"outcome": "COLLISION", "threat_agent_id": 3, "min_ttc": 1.5},
            {"outcome": "OFFROAD", "threat_agent_id": None, "min_ttc": None},
            {"outcome": "SAFE", "threat_agent_id": None, "min_ttc": None},
        ]
        result = necessity_scorer.compute(alts)
        # 2 out of 4 are non-SAFE
        assert result["necessity_score"] == 0.5
        # Only agent 3 has threat_agent_id
        assert len(result["threat_agents"]) == 1
        assert result["threat_agents"][0]["agent_id"] == 3

    def test_empty_alternatives(self):
        result = necessity_scorer.compute([])
        assert result["necessity_score"] == 0.0
        assert result["threat_agents"] == []

    def test_offroad_only(self):
        """OFFROAD counts as non-SAFE but has no threat_agent_id."""
        alts = [
            {"outcome": "OFFROAD", "threat_agent_id": None, "min_ttc": None},
            {"outcome": "OFFROAD", "threat_agent_id": None, "min_ttc": None},
        ]
        result = necessity_scorer.compute(alts)
        assert result["necessity_score"] == 1.0
        assert result["threat_agents"] == []  # No specific agent caused offroad
