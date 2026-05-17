# Tests for Module 7: DecisionClassifier

import pytest

from xai.computation import decision_classifier

CONFIG = {
    "necessity_threshold": 0.5,
    "grounding_threshold": 0.5,
}


class TestDecisionClassifier:
    def test_grounded_critical(self):
        assert (
            decision_classifier.classify(0.8, 0.7, CONFIG) == "GROUNDED_CRITICAL"
        )

    def test_ungrounded_critical(self):
        assert (
            decision_classifier.classify(0.8, 0.2, CONFIG) == "UNGROUNDED_CRITICAL"
        )

    def test_grounded_routine(self):
        assert (
            decision_classifier.classify(0.2, 0.7, CONFIG) == "GROUNDED_ROUTINE"
        )

    def test_routine(self):
        assert decision_classifier.classify(0.2, 0.2, CONFIG) == "ROUTINE"

    def test_exact_thresholds_high(self):
        """On the threshold → counts as 'high'."""
        assert (
            decision_classifier.classify(0.5, 0.5, CONFIG) == "GROUNDED_CRITICAL"
        )

    def test_none_grounding_score(self):
        """None grounding → treated as 0."""
        assert (
            decision_classifier.classify(0.8, None, CONFIG) == "UNGROUNDED_CRITICAL"
        )
        assert decision_classifier.classify(0.2, None, CONFIG) == "ROUTINE"
