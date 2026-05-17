"""
Report loader — converts offline JSON reports into DeepEval test cases.

Reads ``report_*.json`` files from a directory, extracts the structured
report as the ``input`` and the LLM ``narration`` as the ``actual_output``,
and builds an ``EvaluationDataset`` for batch evaluation.
"""

from __future__ import annotations

import glob
import json
import os
from typing import List, Optional

from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase


def _report_to_test_case(report: dict, source_file: str) -> Optional[LLMTestCase]:
    """
    Convert a single report dict into an ``LLMTestCase``.

    The ``input`` is the full report JSON (with narration stripped) so the
    judge can cross-reference claims against ground-truth data.
    The ``actual_output`` is the narration text to be evaluated.

    Returns ``None`` if the report has no narration field.
    """
    narration = report.get("narration")
    if not narration or narration.startswith("[LLMNarrator] ERROR"):
        return None

    # Build context dict — everything except the narration itself
    context = {k: v for k, v in report.items()
               if k not in ("narration", "narration_response_time_s")}

    return LLMTestCase(
        input=json.dumps(context, indent=2),
        actual_output=narration,
        additional_metadata={
            "source_file": os.path.basename(source_file),
            "decision_class": report.get("decision_class", "UNKNOWN"),
            "scenario_step": report.get("step"),
            "response_time_s": report.get("narration_response_time_s", 0.0),
        },
    )


def load_reports(
    reports_dir: str,
    decision_class_filter: Optional[List[str]] = None,
    max_reports: Optional[int] = None,
) -> EvaluationDataset:
    """
    Load offline narration reports into a DeepEval ``EvaluationDataset``.

    Parameters
    ----------
    reports_dir
        Path to directory containing ``report_*.json`` files.
    decision_class_filter
        If given, only include reports whose ``decision_class`` is in this list.
        Example: ``["GROUNDED_CRITICAL", "UNGROUNDED_CRITICAL"]``.
    max_reports
        Cap the number of test cases (useful for quick smoke tests).

    Returns
    -------
    EvaluationDataset
        Dataset ready for ``deepeval.evaluate()``.
    """
    if not os.path.isdir(reports_dir):
        raise FileNotFoundError(f"Reports directory not found: {reports_dir}")

    report_files = sorted(glob.glob(os.path.join(reports_dir, "*.json")))
    if not report_files:
        raise FileNotFoundError(f"No JSON files found in: {reports_dir}")

    test_cases: List[LLMTestCase] = []
    skipped = 0

    for file_path in report_files:
        try:
            with open(file_path, "r") as f:
                report = json.load(f)
        except (json.JSONDecodeError, IOError):
            skipped += 1
            continue

        # Apply decision class filter
        if decision_class_filter:
            dc = report.get("decision_class", "UNKNOWN")
            if dc not in decision_class_filter:
                continue

        tc = _report_to_test_case(report, file_path)
        if tc is None:
            skipped += 1
            continue

        test_cases.append(tc)

        if max_reports and len(test_cases) >= max_reports:
            break

    print(f"📊 Loaded {len(test_cases)} test cases from {reports_dir} "
          f"(skipped {skipped})")

    dataset = EvaluationDataset()
    dataset.test_cases = test_cases
    return dataset


def get_decision_classes(dataset: EvaluationDataset) -> List[str]:
    """Extract the decision class for each test case in order."""
    classes = []
    for tc in dataset.test_cases:
        meta = tc.additional_metadata or {}
        classes.append(meta.get("decision_class", "UNKNOWN"))
    return classes
