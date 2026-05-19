# Tests for Module 3: AdaptiveActionGrid

import pytest

from xai.computation.adaptive_action_grid import build_action_grid

# Minimal config matching xai_config.yaml structure
MOCK_CONFIG = {
    "action_templates": {
        "following": [
            {"label": "Gentle Brake", "accel": -1.5, "steer": 0.0},
            {"label": "Moderate Brake", "accel": -3.0, "steer": 0.0},
            {"label": "Hard Brake", "accel": -4.0, "steer": 0.0},
            {"label": "Maintain", "accel": 0.0, "steer": 0.0},
            {"label": "Gentle Accelerate", "accel": 1.0, "steer": 0.0},
        ],
        "threat_approaching": [
            {"label": "Emergency Brake", "accel": -6.0, "steer": 0.0},
            {"label": "Hard Brake", "accel": -4.0, "steer": 0.0},
            {"label": "Evasive Left", "accel": -2.0, "steer": 0.10},
            {"label": "Evasive Right", "accel": -2.0, "steer": -0.10},
        ],
        "free_flow": [
            {"label": "Moderate Accelerate", "accel": 2.0, "steer": 0.0},
            {"label": "Strong Accelerate", "accel": 3.5, "steer": 0.0},
            {"label": "Maintain", "accel": 0.0, "steer": 0.0},
            {"label": "Lane Change Left", "accel": 0.0, "steer": 0.08},
            {"label": "Lane Change Right", "accel": 0.0, "steer": -0.08},
        ],
    }
}


class TestBuildActionGrid:
    def test_single_category_following(self):
        grid = build_action_grid(["following"], MOCK_CONFIG)
        assert len(grid) == 5
        labels = {a["label"] for a in grid}
        assert "Gentle Brake" in labels
        assert "Maintain" in labels

    def test_single_category_threat(self):
        grid = build_action_grid(["threat_approaching"], MOCK_CONFIG)
        assert len(grid) == 4
        labels = {a["label"] for a in grid}
        assert "Emergency Brake" in labels

    def test_union_deduplication(self):
        """following + threat_approaching share 'Hard Brake' (accel=-4.0, steer=0.0)."""
        grid = build_action_grid(["following", "threat_approaching"], MOCK_CONFIG)
        # following=5 + threat=4 = 9 total, minus 1 duplicate (Hard Brake) = 8
        assert len(grid) == 8
        keys = [(a["accel"], a["steer"]) for a in grid]
        assert len(keys) == len(set(keys)), "Duplicates should have been removed"

    def test_union_following_and_free_flow(self):
        """following + free_flow share 'Maintain' (accel=0.0, steer=0.0)."""
        grid = build_action_grid(["following", "free_flow"], MOCK_CONFIG)
        # following=5 + free_flow=5 = 10 total, minus 1 duplicate (Maintain) = 9
        assert len(grid) == 9

    def test_empty_categories_fallback(self):
        grid = build_action_grid([], MOCK_CONFIG)
        assert len(grid) >= 1, "Should return a fallback grid"

    def test_unknown_category_fallback(self):
        grid = build_action_grid(["nonexistent_category"], MOCK_CONFIG)
        assert len(grid) >= 1, "Should return a fallback grid for unknown categories"

    def test_grid_entries_have_required_keys(self):
        grid = build_action_grid(["following"], MOCK_CONFIG)
        for entry in grid:
            assert "label" in entry
            assert "accel" in entry
            assert "steer" in entry
