# Tests for Module 8: ReportBuilder

import json
import os
import tempfile

import pytest

from xai.computation import report_builder


MOCK_REPORT_ARGS = dict(
    step=40,
    timestamp_s=4.0,
    ego_state={"ego_velocity": 12.5, "ego_action": "moving"},
    chosen_action={
        "label": "Maintain",
        "accel": 0.0,
        "steer": 0.0,
        "outcome": "SAFE",
    },
    context_categories=["following", "threat_approaching"],
    alternatives=[
        {
            "label": "Hard Brake",
            "accel": -4.0,
            "steer": 0.0,
            "outcome": "COLLISION",
            "min_ttc": 1.2,
            "threat_agent_id": 5,
        },
        {
            "label": "Gentle Accelerate",
            "accel": 1.0,
            "steer": 0.0,
            "outcome": "SAFE",
            "min_ttc": None,
            "threat_agent_id": None,
        },
    ],
    necessity_score=0.5,
    attention_grounding={
        "grounding_score": 0.72,
        "per_agent_breakdown": [
            {"agent_id": 5, "severity": 0.45, "attention_mass": 0.38},
        ],
    },
    decision_class="GROUNDED_CRITICAL",
)


class TestReportBuilder:
    def test_build_valid(self):
        report = report_builder.build(**MOCK_REPORT_ARGS)
        assert report["step"] == 40
        assert report["timestamp_s"] == 4.0
        assert report["decision_class"] == "GROUNDED_CRITICAL"
        assert report["necessity_score"] == 0.5
        assert len(report["alternatives"]) == 2

    def test_build_invalid_decision_class(self):
        args = {**MOCK_REPORT_ARGS, "decision_class": "INVALID"}
        with pytest.raises(ValueError, match="Invalid decision_class"):
            report_builder.build(**args)

    def test_build_invalid_alternatives(self):
        args = {**MOCK_REPORT_ARGS, "alternatives": "not_a_list"}
        with pytest.raises(ValueError, match="alternatives must be a list"):
            report_builder.build(**args)

    def test_json_round_trip(self):
        report = report_builder.build(**MOCK_REPORT_ARGS)
        serialized = json.dumps(report, default=str)
        deserialized = json.loads(serialized)
        assert deserialized["decision_class"] == "GROUNDED_CRITICAL"
        assert deserialized["step"] == 40

    def test_save_creates_file(self):
        report = report_builder.build(**MOCK_REPORT_ARGS)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = report_builder.save(report, tmpdir, "scenario_0", 40)
            assert os.path.exists(path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["decision_class"] == "GROUNDED_CRITICAL"

    def test_required_fields_present(self):
        report = report_builder.build(**MOCK_REPORT_ARGS)
        required = [
            "step", "timestamp_s", "ego_state", "chosen_action",
            "context_categories", "alternatives", "necessity_score",
            "attention_grounding", "decision_class",
        ]
        for field in required:
            assert field in report, f"Missing required field: {field}"
