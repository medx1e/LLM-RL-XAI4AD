"""Aggregation, summarization, and cross-model comparison."""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from posthoc_xai.experiments.analyzer import AnalysisResult, load_analysis


# ------------------------------------------------------------------
# Single-model summarization
# ------------------------------------------------------------------

def summarize_model(output_dir: str, model_name: str) -> dict:
    """Aggregate all analysis results for a single model.

    Computes mean/std of category importance, sparsity, and faithfulness
    across all analyzed (scenario, timestep) pairs.

    Args:
        output_dir: Base results directory (e.g. ``results/experiment_name``).
        model_name: Model name (subdirectory under output_dir).

    Returns:
        Summary dict, also saved to ``<model_name>/summary.json``.
    """
    analysis_dir = os.path.join(output_dir, model_name, "analysis")
    if not os.path.isdir(analysis_dir):
        print(f"  No analysis directory for {model_name}")
        return {}

    # Load all analysis results
    results: list[AnalysisResult] = []
    for fname in sorted(os.listdir(analysis_dir)):
        if fname.endswith(".json"):
            path = os.path.join(analysis_dir, fname)
            try:
                results.append(load_analysis(path))
            except Exception:
                continue

    if not results:
        print(f"  No analysis results for {model_name}")
        return {}

    print(f"  Summarizing {len(results)} analysis points for {model_name}")

    # Collect all methods seen
    all_methods = set()
    for r in results:
        all_methods.update(r.attributions.keys())
    all_methods = sorted(all_methods)

    # Aggregate category importance per method
    cat_importance: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        for method, attr_data in r.attributions.items():
            for cat, val in attr_data["category_importance"].items():
                cat_importance[method][cat].append(val)

    cat_summary: dict[str, dict[str, dict[str, float]]] = {}
    for method in all_methods:
        cat_summary[method] = {}
        for cat, vals in cat_importance[method].items():
            cat_summary[method][cat] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }

    # Aggregate sparsity per method
    sparsity_agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        for method, sp in r.sparsity.items():
            for metric, val in sp.items():
                sparsity_agg[method][metric].append(val)

    sparsity_summary: dict[str, dict[str, dict[str, float]]] = {}
    for method in all_methods:
        sparsity_summary[method] = {}
        for metric, vals in sparsity_agg[method].items():
            sparsity_summary[method][metric] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }

    # Aggregate faithfulness per method
    faith_agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        for method, fa in r.faithfulness.items():
            for metric, val in fa.items():
                if not np.isnan(val):
                    faith_agg[method][metric].append(val)

    faith_summary: dict[str, dict[str, dict[str, float]]] = {}
    for method in all_methods:
        faith_summary[method] = {}
        for metric, vals in faith_agg[method].items():
            if vals:
                faith_summary[method][metric] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }

    # Aggregate method agreement
    agreement_agg: dict[str, list[float]] = defaultdict(list)
    for r in results:
        if r.method_agreement:
            for pair, val in r.method_agreement.items():
                if not np.isnan(val):
                    agreement_agg[pair].append(val)

    agreement_summary: dict[str, dict[str, float]] = {}
    for pair, vals in agreement_agg.items():
        if vals:
            agreement_summary[pair] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }

    # Computation time
    time_agg: dict[str, list[float]] = defaultdict(list)
    for r in results:
        for method, attr_data in r.attributions.items():
            t = attr_data.get("computation_time_ms", 0)
            if t > 0:
                time_agg[method].append(t)

    time_summary: dict[str, dict[str, float]] = {}
    for method, vals in time_agg.items():
        time_summary[method] = {
            "mean_ms": float(np.mean(vals)),
            "std_ms": float(np.std(vals)),
        }

    summary = {
        "model_name": model_name,
        "n_analysis_points": len(results),
        "methods": all_methods,
        "category_importance": cat_summary,
        "sparsity": sparsity_summary,
        "faithfulness": faith_summary,
        "method_agreement": agreement_summary,
        "computation_time": time_summary,
    }

    # Save
    summary_path = os.path.join(output_dir, model_name, "summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {summary_path}")

    return summary


# ------------------------------------------------------------------
# Cross-model comparison
# ------------------------------------------------------------------

def compare_models(output_dir: str, model_names: list[str] | None = None) -> dict:
    """Compare summaries across multiple models.

    Args:
        output_dir: Base results directory.
        model_names: List of model names.  If None, auto-discovers from
            subdirectories that have a ``summary.json``.

    Returns:
        Comparison dict, also saved to ``comparison/summary.json``.
    """
    if model_names is None:
        model_names = _discover_models(output_dir)

    if not model_names:
        print("  No models found for comparison")
        return {}

    print(f"  Comparing models: {model_names}")

    summaries: dict[str, dict] = {}
    for name in model_names:
        summary_path = os.path.join(output_dir, name, "summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                summaries[name] = json.load(f)
        else:
            print(f"  Warning: no summary for {name}, running summarize_model...")
            summaries[name] = summarize_model(output_dir, name)

    # Build comparison tables
    all_methods = set()
    for s in summaries.values():
        all_methods.update(s.get("methods", []))
    all_methods = sorted(all_methods)

    # Category importance comparison: method -> category -> model -> mean
    cat_comparison: dict[str, dict[str, dict[str, float]]] = {}
    for method in all_methods:
        cat_comparison[method] = {}
        all_cats = set()
        for name, s in summaries.items():
            cats = s.get("category_importance", {}).get(method, {})
            all_cats.update(cats.keys())

        for cat in sorted(all_cats):
            cat_comparison[method][cat] = {}
            for name, s in summaries.items():
                val = (
                    s.get("category_importance", {})
                    .get(method, {})
                    .get(cat, {})
                    .get("mean", 0)
                )
                cat_comparison[method][cat][name] = val

    # Faithfulness ranking: method -> metric -> model -> mean
    faith_comparison: dict[str, dict[str, dict[str, float]]] = {}
    for method in all_methods:
        faith_comparison[method] = {}
        for metric in ["deletion_auc", "insertion_auc"]:
            faith_comparison[method][metric] = {}
            for name, s in summaries.items():
                val = (
                    s.get("faithfulness", {})
                    .get(method, {})
                    .get(metric, {})
                    .get("mean", 0)
                )
                faith_comparison[method][metric][name] = val

    # Sparsity comparison
    sparsity_comparison: dict[str, dict[str, dict[str, float]]] = {}
    for method in all_methods:
        sparsity_comparison[method] = {}
        for metric in ["gini", "entropy", "top_10_concentration", "top_50_concentration"]:
            sparsity_comparison[method][metric] = {}
            for name, s in summaries.items():
                val = (
                    s.get("sparsity", {})
                    .get(method, {})
                    .get(metric, {})
                    .get("mean", 0)
                )
                sparsity_comparison[method][metric][name] = val

    comparison = {
        "models": model_names,
        "methods": all_methods,
        "category_importance": cat_comparison,
        "faithfulness": faith_comparison,
        "sparsity": sparsity_comparison,
    }

    # Save
    comp_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comp_dir, exist_ok=True)
    comp_path = os.path.join(comp_dir, "summary.json")
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"  Comparison saved to {comp_path}")

    return comparison


def _discover_models(output_dir: str) -> list[str]:
    """Find model subdirectories that have a summary.json or analysis/ dir."""
    models = []
    if not os.path.isdir(output_dir):
        return models
    for name in sorted(os.listdir(output_dir)):
        subdir = os.path.join(output_dir, name)
        if not os.path.isdir(subdir):
            continue
        if name == "comparison" or name == "observations":
            continue
        if os.path.exists(os.path.join(subdir, "summary.json")) or os.path.isdir(
            os.path.join(subdir, "analysis")
        ):
            models.append(name)
    return models


# ------------------------------------------------------------------
# Cross-model plots
# ------------------------------------------------------------------

def generate_comparison_plots(
    output_dir: str, model_names: list[str] | None = None
) -> None:
    """Generate cross-model comparison plots.

    Creates:
    - Category importance grouped bar chart (per method)
    - Faithfulness ranking bar chart
    - Sparsity comparison (Gini per method per model)
    - Method agreement heatmap (per model)
    """
    comp_path = os.path.join(output_dir, "comparison", "summary.json")
    if not os.path.exists(comp_path):
        print("  No comparison summary found. Run compare_models first.")
        return

    with open(comp_path) as f:
        comparison = json.load(f)

    models = comparison["models"]
    methods = comparison["methods"]
    plot_dir = os.path.join(output_dir, "comparison", "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Short model names for readability
    short_names = [_short_model_name(m) for m in models]

    # 1. Category importance comparison (one plot per method)
    for method in methods:
        cat_data = comparison.get("category_importance", {}).get(method, {})
        if not cat_data:
            continue

        categories = sorted(cat_data.keys())
        n_cats = len(categories)
        n_models = len(models)

        fig, ax = plt.subplots(figsize=(max(10, n_cats * 2), 6))
        x = np.arange(n_cats)
        width = 0.8 / max(n_models, 1)
        colors = plt.cm.Set2(np.linspace(0, 1, n_models))

        for i, (model_name, short) in enumerate(zip(models, short_names)):
            values = [cat_data[c].get(model_name, 0) for c in categories]
            offset = (i - n_models / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=short, color=colors[i])

        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=15, ha="right")
        ax.set_ylabel("Mean Importance")
        ax.set_title(f"Category Importance — {method}")
        ax.legend()
        plt.tight_layout()
        fig.savefig(
            os.path.join(plot_dir, f"category_{method}.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

    # 2. Faithfulness ranking (deletion AUC)
    faith_data = comparison.get("faithfulness", {})
    if faith_data:
        for metric, label, better in [
            ("deletion_auc", "Deletion AUC (lower = better)", "lower"),
            ("insertion_auc", "Insertion AUC (higher = better)", "higher"),
        ]:
            fig, ax = plt.subplots(figsize=(max(10, len(methods) * 1.5), 6))
            n_methods = len(methods)
            n_models = len(models)
            x = np.arange(n_methods)
            width = 0.8 / max(n_models, 1)
            colors = plt.cm.Set2(np.linspace(0, 1, n_models))

            for i, (model_name, short) in enumerate(zip(models, short_names)):
                values = [
                    faith_data.get(m, {}).get(metric, {}).get(model_name, 0)
                    for m in methods
                ]
                offset = (i - n_models / 2 + 0.5) * width
                ax.bar(x + offset, values, width, label=short, color=colors[i])

            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=30, ha="right")
            ax.set_ylabel(label)
            ax.set_title(f"Faithfulness — {label}")
            ax.legend()
            plt.tight_layout()
            fig.savefig(
                os.path.join(plot_dir, f"faithfulness_{metric}.png"),
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)

    # 3. Sparsity comparison (Gini coefficient)
    sparsity_data = comparison.get("sparsity", {})
    if sparsity_data:
        fig, ax = plt.subplots(figsize=(max(10, len(methods) * 1.5), 6))
        n_methods = len(methods)
        n_models = len(models)
        x = np.arange(n_methods)
        width = 0.8 / max(n_models, 1)
        colors = plt.cm.Set2(np.linspace(0, 1, n_models))

        for i, (model_name, short) in enumerate(zip(models, short_names)):
            values = [
                sparsity_data.get(m, {}).get("gini", {}).get(model_name, 0)
                for m in methods
            ]
            offset = (i - n_models / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=short, color=colors[i])

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.set_ylabel("Gini Coefficient")
        ax.set_title("Sparsity — Gini Coefficient per Method")
        ax.legend()
        plt.tight_layout()
        fig.savefig(
            os.path.join(plot_dir, "sparsity_gini.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

    # 4. Method agreement heatmap (per model)
    for model_name, short in zip(models, short_names):
        summary_path = os.path.join(output_dir, model_name, "summary.json")
        if not os.path.exists(summary_path):
            continue

        with open(summary_path) as f:
            summary = json.load(f)

        agreement = summary.get("method_agreement", {})
        if not agreement:
            continue

        # Build a matrix of methods x methods
        method_set = set()
        for pair in agreement.keys():
            parts = pair.split("_vs_")
            if len(parts) == 2:
                method_set.update(parts)
        method_list = sorted(method_set)
        n = len(method_list)
        if n < 2:
            continue

        matrix = np.eye(n)
        for i, m_i in enumerate(method_list):
            for j, m_j in enumerate(method_list):
                if i == j:
                    continue
                key = f"{m_i}_vs_{m_j}"
                key_rev = f"{m_j}_vs_{m_i}"
                val = agreement.get(key, agreement.get(key_rev, {}))
                if isinstance(val, dict):
                    matrix[i, j] = val.get("mean", 0)
                else:
                    matrix[i, j] = float(val)

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(method_list, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(method_list, fontsize=8)
        ax.set_title(f"Method Agreement — {short}")

        # Annotate cells
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)

        fig.colorbar(im, ax=ax, label="Pearson r")
        plt.tight_layout()
        fig.savefig(
            os.path.join(plot_dir, f"agreement_{model_name}.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

    print(f"  Comparison plots saved to {plot_dir}")


def generate_temporal_plots_from_saved(
    output_dir: str,
    model_name: str,
    scenario_indices: list[int] | None = None,
) -> None:
    """Generate temporal plots from saved analysis JSON files.

    This is a standalone function for post-hoc temporal visualization without
    needing to re-run the analysis or load a model.

    Args:
        output_dir: Base results directory (e.g. ``results/default``).
        model_name: Model name (subdirectory under output_dir).
        scenario_indices: Specific scenario indices to plot. If None,
            auto-discovers all scenarios with 2+ timesteps.
    """
    from posthoc_xai.experiments.scanner import ScenarioInfo, load_catalog
    from posthoc_xai.experiments.analyzer import generate_temporal_plots
    from posthoc_xai.experiments.config import ExperimentConfig

    analysis_dir = os.path.join(output_dir, model_name, "analysis")
    if not os.path.isdir(analysis_dir):
        print(f"  No analysis directory for {model_name}")
        return

    # Load all analysis results and group by scenario
    by_scenario: dict[int, list[AnalysisResult]] = defaultdict(list)
    for fname in sorted(os.listdir(analysis_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(analysis_dir, fname)
        try:
            result = load_analysis(path)
            by_scenario[result.scenario_idx].append(result)
        except Exception:
            continue

    # Filter to scenarios with 2+ timesteps
    multi_ts = {
        idx: sorted(results, key=lambda r: r.timestep)
        for idx, results in by_scenario.items()
        if len(results) >= 2
    }

    if scenario_indices is not None:
        multi_ts = {idx: r for idx, r in multi_ts.items() if idx in scenario_indices}

    if not multi_ts:
        print(f"  No scenarios with 2+ timesteps found for {model_name}")
        return

    # Try to load catalog for scenario metadata
    catalog_path = os.path.join(output_dir, "catalog.json")
    catalog_map: dict[int, ScenarioInfo] = {}
    if os.path.exists(catalog_path):
        catalog = load_catalog(catalog_path)
        catalog_map = {s.index: s for s in catalog}

    # Build a minimal config for plot output paths
    config = ExperimentConfig(output_dir="", experiment_name="")
    # Override model_results_dir directly
    model_results = os.path.join(output_dir, model_name)

    print(f"  Generating temporal plots for {len(multi_ts)} scenarios...")

    for scenario_idx, results in sorted(multi_ts.items()):
        # Get or create ScenarioInfo
        if scenario_idx in catalog_map:
            scenario_info = catalog_map[scenario_idx]
        else:
            scenario_info = ScenarioInfo(
                index=scenario_idx,
                total_steps=max(r.timestep for r in results),
            )

        # Temporarily patch config paths
        config._model_results_override = model_results
        try:
            _generate_temporal_from_results(results, scenario_info, model_results)
            print(f"    scenario {scenario_idx:03d}: {len(results)} timesteps")
        except Exception as e:
            print(f"    scenario {scenario_idx:03d}: FAILED ({e})")


def _generate_temporal_from_results(
    scenario_results: list[AnalysisResult],
    scenario_info,
    model_results_dir: str,
) -> None:
    """Generate temporal plots for pre-loaded results (internal helper)."""
    from posthoc_xai.visualization.heatmaps import (
        plot_temporal_category,
        plot_temporal_category_stacked,
        plot_temporal_entity,
        plot_temporal_sparsity,
        plot_temporal_multi_method,
    )

    if len(scenario_results) < 2:
        return

    plot_dir = os.path.join(model_results_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    scenario_idx = scenario_info.index
    prefix = f"s{scenario_idx:03d}_temporal"

    scenario_results = sorted(scenario_results, key=lambda r: r.timestep)
    timesteps = [r.timestep for r in scenario_results]

    # Event markers
    events: dict[int, str] = {}
    if getattr(scenario_info, "collision", False):
        events[scenario_info.total_steps] = "collision"
    if getattr(scenario_info, "offroad", False):
        events[scenario_info.total_steps] = "offroad"

    all_methods = sorted({m for r in scenario_results for m in r.attributions})

    # Build category series per method
    method_cat_series: dict[str, dict[str, list[float]]] = {}
    for method in all_methods:
        cat_series: dict[str, list[float]] = {}
        for r in scenario_results:
            cat_imp = r.attributions.get(method, {}).get("category_importance", {})
            for cat, val in cat_imp.items():
                cat_series.setdefault(cat, []).append(val)
        if cat_series:
            method_cat_series[method] = cat_series

    # Build entity series per method
    method_entity_series: dict[str, dict[str, dict[str, list[float]]]] = {}
    for method in all_methods:
        ent_by_cat: dict[str, dict[str, list[float]]] = {}
        for r in scenario_results:
            ent_imp = r.attributions.get(method, {}).get("entity_importance", {})
            for cat, entities in ent_imp.items():
                ent_by_cat.setdefault(cat, {})
                for ent, val in entities.items():
                    ent_by_cat[cat].setdefault(ent, []).append(val)
        if ent_by_cat:
            method_entity_series[method] = ent_by_cat

    # Validity series
    validity_series: dict[str, dict[str, list[bool]]] = {}
    for r in scenario_results:
        for cat, entities in r.validity.items():
            validity_series.setdefault(cat, {})
            for ent, valid in entities.items():
                validity_series[cat].setdefault(ent, []).append(valid)

    # Sparsity series
    sparsity_series: dict[str, dict[str, list[float]]] = {}
    for method in all_methods:
        metric_series: dict[str, list[float]] = {}
        for r in scenario_results:
            for metric, val in r.sparsity.get(method, {}).items():
                metric_series.setdefault(metric, []).append(val)
        if metric_series:
            sparsity_series[method] = metric_series

    # --- Generate ---

    for method, cat_series in method_cat_series.items():
        try:
            fig = plot_temporal_category(timesteps, cat_series, method, scenario_idx, events)
            fig.savefig(os.path.join(plot_dir, f"{prefix}_{method}_categories.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            pass

        try:
            fig = plot_temporal_category_stacked(timesteps, cat_series, method, scenario_idx, events)
            fig.savefig(os.path.join(plot_dir, f"{prefix}_{method}_stacked.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            pass

    for method, ent_by_cat in method_entity_series.items():
        for cat in ["other_agents", "traffic_lights"]:
            if cat not in ent_by_cat:
                continue
            try:
                fig = plot_temporal_entity(
                    timesteps, ent_by_cat[cat], cat, method, scenario_idx,
                    validity_series.get(cat), top_n=5,
                )
                fig.savefig(os.path.join(plot_dir, f"{prefix}_{method}_{cat}.png"),
                            dpi=150, bbox_inches="tight")
                plt.close(fig)
            except Exception:
                pass

    gini_series = {m: metrics["gini"] for m, metrics in sparsity_series.items() if "gini" in metrics}
    if gini_series:
        try:
            fig = plot_temporal_sparsity(timesteps, gini_series, "gini", scenario_idx)
            fig.savefig(os.path.join(plot_dir, f"{prefix}_sparsity_gini.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            pass

    if len(method_cat_series) >= 2:
        try:
            fig = plot_temporal_multi_method(timesteps, method_cat_series, scenario_idx)
            fig.savefig(os.path.join(plot_dir, f"{prefix}_multi_method.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            pass


def _short_model_name(model_name: str) -> str:
    """Shorten model name for plot labels."""
    # womd_sac_road_perceiver_minimal_42 -> perceiver_42
    parts = model_name.split("_")
    # Find the encoder type (after 'road' or 'lane')
    encoder = model_name
    for i, p in enumerate(parts):
        if p in ("road", "lane") and i + 1 < len(parts):
            # Take encoder + seed
            remaining = parts[i + 1 :]
            # Remove 'minimal' if present
            remaining = [x for x in remaining if x != "minimal"]
            encoder = "_".join(remaining)
            break
    return encoder


# ------------------------------------------------------------------
# Pretty-print summary
# ------------------------------------------------------------------

def print_summary(output_dir: str, model_name: str | None = None) -> None:
    """Print a text summary table to console.

    If model_name is None, prints comparison across all models.
    """
    if model_name:
        _print_model_summary(output_dir, model_name)
    else:
        _print_comparison_summary(output_dir)


def _print_model_summary(output_dir: str, model_name: str) -> None:
    """Print summary for a single model."""
    summary_path = os.path.join(output_dir, model_name, "summary.json")
    if not os.path.exists(summary_path):
        print(f"No summary found for {model_name}")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    print(f"\n{'='*70}")
    print(f"  Model: {model_name}")
    print(f"  Analysis points: {summary.get('n_analysis_points', 0)}")
    print(f"{'='*70}")

    methods = summary.get("methods", [])

    # Category importance
    print(f"\n  Category Importance (mean +/- std):")
    print(f"  {'Method':<25}", end="")
    all_cats = set()
    for m in methods:
        for c in summary.get("category_importance", {}).get(m, {}):
            all_cats.add(c)
    cats = sorted(all_cats)
    for c in cats:
        print(f"  {c[:12]:>12}", end="")
    print()
    print(f"  {'-'*25}", end="")
    for _ in cats:
        print(f"  {'-'*12}", end="")
    print()

    for method in methods:
        print(f"  {method:<25}", end="")
        cat_data = summary.get("category_importance", {}).get(method, {})
        for c in cats:
            val = cat_data.get(c, {})
            mean = val.get("mean", 0) if isinstance(val, dict) else 0
            print(f"  {mean:>12.4f}", end="")
        print()

    # Sparsity
    print(f"\n  Sparsity Metrics (mean):")
    print(f"  {'Method':<25}  {'Gini':>8}  {'Entropy':>8}  {'Top-10':>8}  {'Top-50':>8}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for method in methods:
        sp = summary.get("sparsity", {}).get(method, {})
        gini = sp.get("gini", {}).get("mean", 0) if isinstance(sp.get("gini"), dict) else 0
        entropy = sp.get("entropy", {}).get("mean", 0) if isinstance(sp.get("entropy"), dict) else 0
        top10 = sp.get("top_10_concentration", {}).get("mean", 0) if isinstance(sp.get("top_10_concentration"), dict) else 0
        top50 = sp.get("top_50_concentration", {}).get("mean", 0) if isinstance(sp.get("top_50_concentration"), dict) else 0
        print(f"  {method:<25}  {gini:>8.4f}  {entropy:>8.4f}  {top10:>8.4f}  {top50:>8.4f}")

    # Faithfulness
    faith = summary.get("faithfulness", {})
    if any(faith.get(m) for m in methods):
        print(f"\n  Faithfulness (mean):")
        print(f"  {'Method':<25}  {'Del AUC':>10}  {'Ins AUC':>10}")
        print(f"  {'-'*25}  {'-'*10}  {'-'*10}")
        for method in methods:
            fd = faith.get(method, {})
            del_auc = fd.get("deletion_auc", {}).get("mean", 0) if isinstance(fd.get("deletion_auc"), dict) else 0
            ins_auc = fd.get("insertion_auc", {}).get("mean", 0) if isinstance(fd.get("insertion_auc"), dict) else 0
            print(f"  {method:<25}  {del_auc:>10.4f}  {ins_auc:>10.4f}")

    # Computation time
    time_data = summary.get("computation_time", {})
    if time_data:
        print(f"\n  Computation Time (mean ms):")
        for method in methods:
            td = time_data.get(method, {})
            mean_ms = td.get("mean_ms", 0)
            print(f"  {method:<25}  {mean_ms:>10.1f} ms")

    print()


def _print_comparison_summary(output_dir: str) -> None:
    """Print comparison across all models."""
    comp_path = os.path.join(output_dir, "comparison", "summary.json")
    if not os.path.exists(comp_path):
        print("No comparison summary found. Run compare_models first.")
        return

    with open(comp_path) as f:
        comparison = json.load(f)

    models = comparison["models"]
    methods = comparison["methods"]
    short_names = [_short_model_name(m) for m in models]

    print(f"\n{'='*70}")
    print(f"  Cross-Model Comparison")
    print(f"  Models: {', '.join(short_names)}")
    print(f"{'='*70}")

    # Faithfulness ranking
    faith = comparison.get("faithfulness", {})
    if faith:
        print(f"\n  Faithfulness — Deletion AUC (lower = better):")
        header = f"  {'Method':<25}"
        for short in short_names:
            header += f"  {short[:12]:>12}"
        print(header)
        print(f"  {'-'*25}" + f"  {'-'*12}" * len(models))

        for method in methods:
            line = f"  {method:<25}"
            for model_name in models:
                val = faith.get(method, {}).get("deletion_auc", {}).get(model_name, 0)
                line += f"  {val:>12.4f}"
            print(line)

    print()
