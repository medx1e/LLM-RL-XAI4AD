"""
CLI entry-point for evaluating LLM narrations with the XAI rubric.

Usage
-----
Single model evaluation::

    python -m xai.llm_evaluation.evaluate_narrations \\
        --reports_dir xai/reports/llama-3.1-8b/ \\
        --model_name llama-3.1-8b \\
        --output results/eval_llama3.json

Multi-model comparison::

    python -m xai.llm_evaluation.evaluate_narrations \\
        --compare \\
        --reports_dirs xai/reports/llama-3.1-8b/ xai/reports/qwen3-4b/ \\
        --model_names llama-3.1-8b qwen3-4b \\
        --output results/comparison.json

Options::

    --judge_model       Judge model for G-Eval (default: gpt-4o)
    --threshold         Pass/fail threshold (default: 0.7)
    --filter_classes    Only evaluate specific decision classes
    --max_reports       Cap number of reports to evaluate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

from deepeval import evaluate

from xai.llm_evaluation.config import EvalConfig
from xai.llm_evaluation.loader import load_reports, get_decision_classes
from xai.llm_evaluation.metrics import get_all_metrics
from xai.llm_evaluation.cognitive_scores import (
    CognitiveScorecard,
    compute_cognitive_scores,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _collect_per_report_scores(
    test_cases,
    metrics,
) -> List[Dict[str, float]]:
    """
    After ``evaluate()`` has run, extract per-report scores from the
    metric objects.  DeepEval stores scores on each test case after
    evaluation, so we iterate through the results.
    """
    per_report: List[Dict[str, float]] = []

    for tc in test_cases:
        report_scores: Dict[str, float] = {}
        for metric in metrics:
            # After evaluate(), each metric has `score` for the last
            # measured test case.  For batch, we re-measure individually.
            metric.measure(tc)
            report_scores[metric.name] = metric.score or 0.0
        per_report.append(report_scores)

    return per_report


def _evaluate_single_model(
    reports_dir: str,
    model_name: str,
    config: EvalConfig,
    max_reports: Optional[int] = None,
) -> dict:
    """Run evaluation for a single model and return results dict."""

    dataset = load_reports(
        reports_dir=reports_dir,
        decision_class_filter=config.decision_class_filter,
        max_reports=max_reports,
    )

    if not dataset.test_cases:
        print(f"❌ No valid test cases found in {reports_dir}")
        return {}

    metrics = get_all_metrics(
        threshold=config.threshold,
        model=config.judge_model
    )
    decision_classes = get_decision_classes(dataset)

    # Run batch evaluation — this calls the judge model
    print(f"\n🔍 Evaluating {len(dataset.test_cases)} narrations "
          f"from '{model_name}' with judge='{config.judge_model}'...\n")

    from deepeval.evaluate import AsyncConfig
    from deepeval.evaluate import ErrorConfig

    eval_results = evaluate(
        test_cases=dataset.test_cases,
        metrics=metrics,
        error_config=ErrorConfig(
            ignore_errors=config.ignore_errors,
        ),
        async_config=AsyncConfig(
            run_async=True,
            max_concurrent=1,
            throttle_value=15, # Delay between test cases to respect 5 req/min.
        )
    )

    # Collect per-report scores from the evaluation results
    per_report_scores: List[Dict[str, float]] = []
    for result in eval_results.test_results:
        report_scores: Dict[str, float] = {}
        for metric_data in result.metrics_data:
            # DeepEval appends a suffix to the metric name, clean it up for exact matching
            clean_name = metric_data.name.replace(" [GEval]", "").replace(" (GEval)", "")
            report_scores[clean_name] = metric_data.score or 0.0
        per_report_scores.append(report_scores)

    # Calculate average response time
    response_times = [
        tc.additional_metadata.get("response_time_s", 0.0) 
        for tc in dataset.test_cases 
        if tc.additional_metadata and tc.additional_metadata.get("response_time_s", 0.0) > 0
    ]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0

    # Compute cognitive scorecard
    scorecard = compute_cognitive_scores(
        per_report_scores=per_report_scores,
        model_name=model_name,
        decision_classes=decision_classes,
        weights=config.cognitive_weights,
    )

    print(f"\n{scorecard.summary_table()}\n")

    return {
        "model_name": model_name,
        "reports_dir": reports_dir,
        "num_reports": len(dataset.test_cases),
        "avg_response_time_s": avg_response_time,
        "scorecard": scorecard.to_dict(),
        "per_report_scores": per_report_scores,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM narrations with the XAI rubric (DeepEval G-Eval).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Single-model mode
    parser.add_argument(
        "--reports_dir",
        type=str,
        help="Path to directory with report_*.json files (single-model mode).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="unknown",
        help="Label for the model being evaluated.",
    )

    # Multi-model comparison mode
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Enable multi-model comparison mode.",
    )
    parser.add_argument(
        "--reports_dirs",
        nargs="+",
        type=str,
        help="List of report directories (one per model) for comparison.",
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        type=str,
        help="Model names corresponding to --reports_dirs.",
    )

    # Common options
    parser.add_argument(
        "--judge_model",
        type=str,
        default="gpt-4o",
        help="Judge model for G-Eval (default: gpt-4o).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Pass/fail threshold for metrics (default: 0.7).",
    )
    parser.add_argument(
        "--filter_classes",
        nargs="+",
        type=str,
        default=None,
        help="Only evaluate reports with these decision classes.",
    )
    parser.add_argument(
        "--max_reports",
        type=int,
        default=None,
        help="Limit number of reports to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON results.",
    )
    parser.add_argument(
        "--ignore_errors",
        "-i",
        action="store_true",
        help="Skip failed test cases and continue the run.",
    )

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    config = EvalConfig(
        judge_model=args.judge_model,
        threshold=args.threshold,
        decision_class_filter=args.filter_classes,
        ignore_errors=args.ignore_errors,
    )

    if args.compare:
        # ── Multi-model comparison ───────────────────────────────────────
        if not args.reports_dirs or not args.model_names:
            parser.error("--compare requires --reports_dirs and --model_names")
        if len(args.reports_dirs) != len(args.model_names):
            parser.error("--reports_dirs and --model_names must have same length")

        all_results = []
        for rdir, mname in zip(args.reports_dirs, args.model_names):
            result = _evaluate_single_model(rdir, mname, config, max_reports=args.max_reports)
            if result:
                all_results.append(result)

        # Print comparison
        if all_results:
            print("\n" + "=" * 70)
            print("  MULTI-MODEL COMPARISON")
            print("=" * 70)
            header = f"{'Model':<20s} {'SitAware':>8s} {'Reason':>8s} {'Comm':>8s} {'Overall':>8s} {'RespTime':>8s}"
            print(header)
            print("-" * 70)
            for r in all_results:
                sc = r["scorecard"]["composite"]
                print(
                    f"{r['model_name']:<20s} "
                    f"{sc['situational_awareness']:>7.2%} "
                    f"{sc['reasoning_quality']:>7.2%} "
                    f"{sc['communication_quality']:>7.2%} "
                    f"{sc['overall_cognitive_score']:>7.2%} "
                    f"{r.get('avg_response_time_s', 0.0):>7.2f}s"
                )
            print("=" * 70)

        output = {"comparison": all_results}
    else:
        # ── Single-model evaluation ──────────────────────────────────────
        if not args.reports_dir:
            parser.error("--reports_dir is required in single-model mode")

        result = _evaluate_single_model(args.reports_dir, args.model_name, config, max_reports=args.max_reports)
        output = result

    # Save results
    if args.output and output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()
