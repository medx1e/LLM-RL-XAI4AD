"""Experiment orchestration and CLI entry point."""

from __future__ import annotations

import argparse
import os
import sys
import time

from posthoc_xai.experiments.config import ExperimentConfig
from posthoc_xai.experiments.scanner import (
    scan_scenarios,
    select_scenarios,
    save_catalog,
    load_catalog,
)
from posthoc_xai.experiments.analyzer import analyze_scenarios
from posthoc_xai.experiments.reporter import (
    summarize_model,
    compare_models,
    generate_comparison_plots,
    print_summary,
)


def run_experiment(config: ExperimentConfig) -> dict:
    """Full pipeline for one model: scan -> select -> analyze -> summarize -> plot.

    Args:
        config: Experiment configuration.

    Returns:
        Summary dict for the model.
    """
    import posthoc_xai as xai

    print(f"\n{'='*70}")
    print(f"  Experiment: {config.experiment_name}")
    print(f"  Model: {config.model_name}")
    print(f"  Methods: {config.methods}")
    print(f"  Scenarios: {config.n_scenarios} scan, {config.max_selected} select")
    print(f"  Faithfulness: {config.compute_faithfulness}")
    print(f"{'='*70}\n")

    t0 = time.time()

    # Load model (without data generator — scanner creates its own to save memory)
    print("Step 1/5: Loading model...")
    model = xai.load_model(config.model_dir)
    print(f"  Model loaded: {model.name}")

    # Create output directories
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.model_results_dir, exist_ok=True)

    # 2. Scan (or load existing catalog)
    catalog_path = os.path.join(config.results_dir, "catalog.json")
    print("\nStep 2/5: Scanning scenarios...")
    if os.path.exists(catalog_path):
        catalog = load_catalog(catalog_path)
    else:
        catalog = scan_scenarios(model, config)
        save_catalog(catalog, catalog_path)
    print(f"  Catalog: {len(catalog)} scenarios")

    # 3. Select
    print("\nStep 3/5: Selecting scenarios...")
    selected = select_scenarios(catalog, config)

    # 4. Analyze
    n_points = sum(len(s.key_timesteps) for s in selected)
    print(f"\nStep 4/5: Analyzing {n_points} (scenario, timestep) pairs...")
    results = analyze_scenarios(model, selected, config)
    print(f"  Analysis complete: {len(results)} results")

    # 5. Summarize
    print("\nStep 5/5: Summarizing results...")
    summary = summarize_model(config.results_dir, config.model_name)

    elapsed = time.time() - t0
    print(f"\nExperiment complete in {elapsed:.0f}s")

    # Print summary
    print_summary(config.results_dir, config.model_name)

    return summary


def compare_experiments(
    output_dir: str,
    model_names: list[str] | None = None,
) -> dict:
    """Cross-model comparison from saved results. No model loading needed.

    Args:
        output_dir: Base results directory.
        model_names: Specific models to compare. None = auto-discover.

    Returns:
        Comparison dict.
    """
    print(f"\n{'='*70}")
    print(f"  Cross-Model Comparison")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*70}\n")

    comparison = compare_models(output_dir, model_names)
    generate_comparison_plots(output_dir, model_names)

    print_summary(output_dir)

    return comparison


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="posthoc_xai.experiments.runner",
        description="Post-Hoc XAI Experiment Pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- run ----
    run_parser = subparsers.add_parser("run", help="Run experiment for a single model")
    run_parser.add_argument(
        "--model", required=True,
        help="Path to model directory (e.g. runs_rlc/womd_sac_road_perceiver_minimal_42)",
    )
    run_parser.add_argument(
        "--data", default="data/training.tfrecord",
        help="Path to training.tfrecord",
    )
    run_parser.add_argument(
        "--preset", choices=["quick", "standard"], default=None,
        help="Use a preset configuration (overrides individual options)",
    )
    run_parser.add_argument(
        "--n-scenarios", type=int, default=None,
        help="Number of scenarios to scan",
    )
    run_parser.add_argument(
        "--max-selected", type=int, default=None,
        help="Max scenarios to analyze",
    )
    run_parser.add_argument(
        "--methods", nargs="+", default=None,
        help="XAI methods to run",
    )
    run_parser.add_argument(
        "--no-faithfulness", action="store_true",
        help="Disable faithfulness computation",
    )
    run_parser.add_argument(
        "--save-raw", action="store_true",
        help="Save raw attribution arrays (.npz)",
    )
    run_parser.add_argument(
        "--no-plots", action="store_true",
        help="Disable plot generation",
    )
    run_parser.add_argument(
        "--output", default="results",
        help="Output directory",
    )
    run_parser.add_argument(
        "--name", default=None,
        help="Experiment name (default: auto from model)",
    )
    run_parser.add_argument(
        "--timestep-strategy", choices=["key_moments", "fixed_interval", "all"],
        default=None,
        help="Timestep selection strategy",
    )

    # ---- compare ----
    compare_parser = subparsers.add_parser(
        "compare", help="Compare results across models"
    )
    compare_parser.add_argument(
        "--output", default="results",
        help="Output directory containing model results",
    )
    compare_parser.add_argument(
        "--models", nargs="+", default=None,
        help="Model names to compare (default: auto-discover)",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        # Build config
        if args.preset == "quick":
            config = ExperimentConfig.quick(args.model, data_path=args.data)
        elif args.preset == "standard":
            config = ExperimentConfig.standard(args.model, data_path=args.data)
        else:
            config = ExperimentConfig(model_dir=args.model, data_path=args.data)

        # Apply individual overrides
        if args.n_scenarios is not None:
            config.n_scenarios = args.n_scenarios
        if args.max_selected is not None:
            config.max_selected = args.max_selected
        if args.methods is not None:
            config.methods = args.methods
        if args.no_faithfulness:
            config.compute_faithfulness = False
        if args.save_raw:
            config.save_raw_arrays = True
        if args.no_plots:
            config.save_plots = False
        if args.output:
            config.output_dir = args.output
        if args.name:
            config.experiment_name = args.name
        if args.timestep_strategy:
            config.timestep_strategy = args.timestep_strategy

        # Re-run __post_init__ to update derived fields after overrides
        config.__post_init__()

        run_experiment(config)

    elif args.command == "compare":
        compare_experiments(args.output, args.models)


if __name__ == "__main__":
    main()
