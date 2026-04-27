"""Experiment pipeline: scan, analyze, report."""

from posthoc_xai.experiments.config import ExperimentConfig
from posthoc_xai.experiments.scanner import (
    scan_scenarios,
    select_scenarios,
    ScenarioInfo,
)
from posthoc_xai.experiments.analyzer import (
    analyze_timestep,
    analyze_scenarios,
    AnalysisResult,
)
from posthoc_xai.experiments.reporter import (
    summarize_model,
    compare_models,
    generate_temporal_plots_from_saved,
)
from posthoc_xai.experiments.runner import run_experiment, compare_experiments

__all__ = [
    "ExperimentConfig",
    "scan_scenarios",
    "select_scenarios",
    "ScenarioInfo",
    "analyze_timestep",
    "analyze_scenarios",
    "AnalysisResult",
    "summarize_model",
    "compare_models",
    "generate_temporal_plots_from_saved",
    "run_experiment",
    "compare_experiments",
]
