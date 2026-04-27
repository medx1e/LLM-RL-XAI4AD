"""XAI analysis on selected scenarios and timesteps."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from posthoc_xai.experiments.config import ExperimentConfig
from posthoc_xai.experiments.scanner import ScenarioInfo
from posthoc_xai.methods import METHOD_REGISTRY, Attribution
from posthoc_xai.metrics import faithfulness, sparsity
from posthoc_xai.models.base import ExplainableModel
from posthoc_xai.visualization import heatmaps


@dataclass
class AnalysisResult:
    """Result of XAI analysis for a single (scenario, timestep) pair."""

    scenario_idx: int
    timestep: int
    model_name: str

    # Per-method results: method -> {category_importance, entity_importance, computation_time_ms}
    attributions: dict[str, dict] = field(default_factory=dict)

    # Entity validity
    validity: dict[str, dict[str, bool]] = field(default_factory=dict)

    # Sparsity metrics: method -> {gini, entropy, top_10_concentration, top_50_concentration}
    sparsity: dict[str, dict[str, float]] = field(default_factory=dict)

    # Faithfulness metrics: method -> {deletion_auc, insertion_auc}
    faithfulness: dict[str, dict[str, float]] = field(default_factory=dict)

    # Cross-method agreement: pairwise category-level correlations
    method_agreement: dict[str, float] | None = None


def _pairwise_category_agreement(
    attr_results: dict[str, Attribution],
) -> dict[str, float] | None:
    """Compute pairwise Pearson correlation of category importance vectors."""
    methods = list(attr_results.keys())
    if len(methods) < 2:
        return None

    agreement: dict[str, float] = {}
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            a = attr_results[methods[i]]
            b = attr_results[methods[j]]
            cats = list(a.category_importance.keys())
            vec_a = [a.category_importance[c] for c in cats]
            vec_b = [b.category_importance[c] for c in cats]
            try:
                corr, _ = pearsonr(vec_a, vec_b)
            except Exception:
                corr = 0.0
            key = f"{methods[i]}_vs_{methods[j]}"
            agreement[key] = float(corr)

    return agreement


def analyze_timestep(
    model: ExplainableModel,
    obs: np.ndarray,
    config: ExperimentConfig,
    scenario_idx: int,
    timestep: int,
) -> AnalysisResult:
    """Run all configured XAI methods and metrics on a single observation.

    Args:
        model: Loaded ExplainableModel.
        obs: Flat observation array for this timestep.
        config: Experiment configuration.
        scenario_idx: Scenario index.
        timestep: Timestep within the episode.

    Returns:
        AnalysisResult with attributions, sparsity, and optionally faithfulness.
    """
    obs_jnp = jnp.array(obs)

    result = AnalysisResult(
        scenario_idx=scenario_idx,
        timestep=timestep,
        model_name=config.model_name,
    )

    # Entity validity
    try:
        result.validity = model.get_entity_validity(obs_jnp)
    except Exception:
        result.validity = {}

    # Run each XAI method
    attr_results: dict[str, Attribution] = {}
    for method_name in config.methods:
        if method_name not in METHOD_REGISTRY:
            print(f"    Warning: unknown method '{method_name}', skipping")
            continue

        method_cls = METHOD_REGISTRY[method_name]
        method = method_cls(model)
        attr = method(obs_jnp, config.target_action)
        attr_results[method_name] = attr

        result.attributions[method_name] = {
            "category_importance": attr.category_importance,
            "entity_importance": attr.entity_importance,
            "computation_time_ms": attr.computation_time_ms,
        }

        # Sparsity metrics
        result.sparsity[method_name] = sparsity.compute_all(attr)

        # Faithfulness metrics
        if config.compute_faithfulness:
            try:
                _pcts, del_outputs = faithfulness.deletion_curve(
                    model, obs_jnp, attr, n_steps=config.faithfulness_steps,
                    target_action=config.target_action,
                )
                _pcts, ins_outputs = faithfulness.insertion_curve(
                    model, obs_jnp, attr, n_steps=config.faithfulness_steps,
                    target_action=config.target_action,
                )
                result.faithfulness[method_name] = {
                    "deletion_auc": faithfulness.area_under_deletion_curve(del_outputs),
                    "insertion_auc": faithfulness.area_under_insertion_curve(ins_outputs),
                }
            except Exception as e:
                print(f"    Faithfulness failed for {method_name}: {e}")
                result.faithfulness[method_name] = {
                    "deletion_auc": float("nan"),
                    "insertion_auc": float("nan"),
                }

    # Cross-method agreement
    result.method_agreement = _pairwise_category_agreement(attr_results)

    return result


def analyze_scenarios(
    model: ExplainableModel,
    selected: list[ScenarioInfo],
    config: ExperimentConfig,
    progress_fn: Optional[Callable[[int, int, str], None]] = None,
) -> list[AnalysisResult]:
    """Run XAI analysis on all selected scenarios at their key timesteps.

    Resume-friendly: skips analysis points that already have saved JSON results.

    Args:
        model: Loaded ExplainableModel.
        selected: Scenarios to analyze (from select_scenarios).
        config: Experiment configuration.
        progress_fn: Optional callback ``(current, total, description)`` for progress.

    Returns:
        List of AnalysisResult for all analyzed points.
    """
    analysis_dir = os.path.join(config.model_results_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    total_points = sum(len(s.key_timesteps) for s in selected)
    current_point = 0
    results: list[AnalysisResult] = []

    for scenario in selected:
        if not scenario.saved_obs_path or not os.path.exists(scenario.saved_obs_path):
            print(f"  Warning: no saved observations for scenario {scenario.index}, skipping")
            continue

        obs_data = np.load(scenario.saved_obs_path)
        scenario_results: list[AnalysisResult] = []

        for ts in scenario.key_timesteps:
            current_point += 1
            json_path = os.path.join(analysis_dir, f"s{scenario.index:03d}_t{ts:03d}.json")

            # Resume: skip if already computed
            if os.path.exists(json_path):
                try:
                    result = load_analysis(json_path)
                    results.append(result)
                    scenario_results.append(result)
                    desc = f"scenario_{scenario.index:03d} step_{ts} (cached)"
                    if progress_fn:
                        progress_fn(current_point, total_points, desc)
                    else:
                        print(f"  [{current_point}/{total_points}] {desc}")
                    continue
                except Exception:
                    pass  # Recompute if loading fails

            obs_key = f"step_{ts}"
            if obs_key not in obs_data:
                print(f"    No observation for step {ts} in scenario {scenario.index}")
                continue

            obs = obs_data[obs_key]
            t0 = time.time()

            result = analyze_timestep(model, obs, config, scenario.index, ts)
            elapsed = time.time() - t0

            results.append(result)
            scenario_results.append(result)
            save_analysis(result, json_path)

            # Optionally save raw arrays
            if config.save_raw_arrays:
                npz_path = os.path.join(analysis_dir, f"s{scenario.index:03d}_t{ts:03d}.npz")
                np.savez_compressed(npz_path, observation=obs)

            desc = (
                f"scenario_{scenario.index:03d} step_{ts}: "
                f"{len(config.methods)} methods, "
                f"faithfulness={config.compute_faithfulness} "
                f"({elapsed:.1f}s)"
            )
            if progress_fn:
                progress_fn(current_point, total_points, desc)
            else:
                print(f"  [{current_point}/{total_points}] {desc}")

            # Generate per-timestep plots
            if config.save_plots:
                try:
                    generate_plots(result, model, obs, config)
                except Exception as e:
                    print(f"    Plot generation failed: {e}")

        # Generate temporal plots after all timesteps for this scenario
        if config.save_plots and len(scenario_results) >= 2:
            try:
                generate_temporal_plots(scenario_results, scenario, config)
            except Exception as e:
                print(f"    Temporal plot generation failed for scenario {scenario.index}: {e}")

    return results


# ------------------------------------------------------------------
# Serialization
# ------------------------------------------------------------------

def _make_serializable(obj):
    """Convert numpy/jax types to plain Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    return obj


def save_analysis(result: AnalysisResult, path: str) -> None:
    """Save an AnalysisResult to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "scenario_idx": result.scenario_idx,
        "timestep": result.timestep,
        "model_name": result.model_name,
        "attributions": _make_serializable(result.attributions),
        "validity": _make_serializable(result.validity),
        "sparsity": _make_serializable(result.sparsity),
        "faithfulness": _make_serializable(result.faithfulness),
        "method_agreement": _make_serializable(result.method_agreement),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_analysis(path: str) -> AnalysisResult:
    """Load an AnalysisResult from JSON."""
    with open(path) as f:
        data = json.load(f)
    return AnalysisResult(
        scenario_idx=data["scenario_idx"],
        timestep=data["timestep"],
        model_name=data["model_name"],
        attributions=data.get("attributions", {}),
        validity=data.get("validity", {}),
        sparsity=data.get("sparsity", {}),
        faithfulness=data.get("faithfulness", {}),
        method_agreement=data.get("method_agreement"),
    )


# ------------------------------------------------------------------
# Plot generation
# ------------------------------------------------------------------

def generate_plots(
    result: AnalysisResult,
    model: ExplainableModel,
    obs: np.ndarray,
    config: ExperimentConfig,
) -> None:
    """Generate per-analysis-point plots and save to disk.

    Creates: category importance bars, method comparison, entity importance,
    and faithfulness curves (if computed).
    """
    plot_dir = os.path.join(config.model_results_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    prefix = f"s{result.scenario_idx:03d}_t{result.timestep:03d}"
    obs_jnp = jnp.array(obs)

    # We need Attribution objects for the visualization functions.
    # Re-run methods (fast since JIT is warm) or reconstruct minimal objects.
    # For efficiency, reconstruct minimal Attribution objects from saved data.
    attr_list: list[Attribution] = []
    for method_name, attr_data in result.attributions.items():
        # Create a minimal Attribution with enough data for category-level plots
        dummy_raw = jnp.zeros(1)
        dummy_norm = jnp.zeros(1)
        attr = Attribution(
            raw=dummy_raw,
            normalized=dummy_norm,
            category_importance=attr_data["category_importance"],
            entity_importance=attr_data["entity_importance"],
            method_name=method_name,
            target_action=config.target_action,
            computation_time_ms=attr_data.get("computation_time_ms", 0),
        )
        attr_list.append(attr)

    # Method comparison (category importance)
    if len(attr_list) >= 1:
        try:
            fig = heatmaps.plot_method_comparison(
                attr_list,
                title=f"Method Comparison — scenario {result.scenario_idx}, step {result.timestep}",
            )
            fig.savefig(
                os.path.join(plot_dir, f"{prefix}_method_comparison.png"),
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)
        except Exception:
            pass

    # Per-method category importance
    for attr in attr_list:
        try:
            fig = heatmaps.plot_category_importance(
                attr,
                title=f"{attr.method_name} — scenario {result.scenario_idx}, step {result.timestep}",
            )
            fig.savefig(
                os.path.join(plot_dir, f"{prefix}_{attr.method_name}_categories.png"),
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)
        except Exception:
            pass

    # Entity importance (for each method)
    validity = result.validity or None
    for attr in attr_list:
        try:
            fig = heatmaps.plot_entity_importance(
                attr,
                validity=validity,
                title=f"{attr.method_name} — Entities — scenario {result.scenario_idx}, step {result.timestep}",
            )
            fig.savefig(
                os.path.join(plot_dir, f"{prefix}_{attr.method_name}_entities.png"),
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)
        except Exception:
            pass

    # Faithfulness curves
    if result.faithfulness and config.compute_faithfulness:
        try:
            # We need the actual deletion/insertion data, which requires
            # re-running the curves. Use saved AUC values to create a summary plot.
            _plot_faithfulness_summary(result, plot_dir, prefix)
        except Exception:
            pass


def generate_temporal_plots(
    scenario_results: list[AnalysisResult],
    scenario_info: ScenarioInfo,
    config: ExperimentConfig,
) -> None:
    """Generate temporal plots for a single scenario across its timesteps.

    Creates line plots and stacked area charts showing how category importance,
    entity importance, and sparsity evolve over the episode.

    Args:
        scenario_results: AnalysisResults for this scenario, sorted by timestep.
        scenario_info: ScenarioInfo with metadata (tags, events).
        config: Experiment configuration.
    """
    from posthoc_xai.visualization.heatmaps import (
        plot_temporal_category,
        plot_temporal_category_stacked,
        plot_temporal_entity,
        plot_temporal_sparsity,
        plot_temporal_multi_method,
    )

    if len(scenario_results) < 2:
        return

    plot_dir = os.path.join(config.model_results_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    scenario_idx = scenario_info.index
    prefix = f"s{scenario_idx:03d}_temporal"

    # Sort by timestep
    scenario_results = sorted(scenario_results, key=lambda r: r.timestep)
    timesteps = [r.timestep for r in scenario_results]

    # Build event markers from scenario tags
    events: dict[int, str] = {}
    if scenario_info.collision:
        events[scenario_info.total_steps] = "collision"
    if scenario_info.offroad:
        events[scenario_info.total_steps] = "offroad"

    # Collect all methods across results
    all_methods = set()
    for r in scenario_results:
        all_methods.update(r.attributions.keys())
    all_methods = sorted(all_methods)

    # --- Build time-series data ---

    # Category importance: {method: {category: [val_t0, val_t1, ...]}}
    method_cat_series: dict[str, dict[str, list[float]]] = {}
    for method in all_methods:
        cat_series: dict[str, list[float]] = {}
        for r in scenario_results:
            attr_data = r.attributions.get(method, {})
            cat_imp = attr_data.get("category_importance", {})
            for cat, val in cat_imp.items():
                cat_series.setdefault(cat, []).append(val)
        if cat_series:
            method_cat_series[method] = cat_series

    # Entity importance: {method: {category: {entity: [val_t0, ...]}}}
    method_entity_series: dict[str, dict[str, dict[str, list[float]]]] = {}
    for method in all_methods:
        ent_by_cat: dict[str, dict[str, list[float]]] = {}
        for r in scenario_results:
            attr_data = r.attributions.get(method, {})
            ent_imp = attr_data.get("entity_importance", {})
            for cat, entities in ent_imp.items():
                ent_by_cat.setdefault(cat, {})
                for ent, val in entities.items():
                    ent_by_cat[cat].setdefault(ent, []).append(val)
        if ent_by_cat:
            method_entity_series[method] = ent_by_cat

    # Entity validity over time: {category: {entity: [valid_t0, ...]}}
    validity_series: dict[str, dict[str, list[bool]]] = {}
    for r in scenario_results:
        for cat, entities in r.validity.items():
            validity_series.setdefault(cat, {})
            for ent, valid in entities.items():
                validity_series[cat].setdefault(ent, []).append(valid)

    # Sparsity: {method: {metric: [val_t0, val_t1, ...]}}
    sparsity_series: dict[str, dict[str, list[float]]] = {}
    for method in all_methods:
        metric_series: dict[str, list[float]] = {}
        for r in scenario_results:
            sp = r.sparsity.get(method, {})
            for metric, val in sp.items():
                metric_series.setdefault(metric, []).append(val)
        if metric_series:
            sparsity_series[method] = metric_series

    # --- Generate plots ---

    # 1. Per-method category importance over time (line + stacked)
    for method, cat_series in method_cat_series.items():
        try:
            fig = plot_temporal_category(
                timesteps, cat_series, method, scenario_idx, events
            )
            fig.savefig(
                os.path.join(plot_dir, f"{prefix}_{method}_categories.png"),
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)
        except Exception:
            pass

        try:
            fig = plot_temporal_category_stacked(
                timesteps, cat_series, method, scenario_idx, events
            )
            fig.savefig(
                os.path.join(plot_dir, f"{prefix}_{method}_stacked.png"),
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)
        except Exception:
            pass

    # 2. Per-method entity importance over time (agents + traffic lights)
    for method, ent_by_cat in method_entity_series.items():
        for cat in ["other_agents", "traffic_lights"]:
            if cat not in ent_by_cat:
                continue
            try:
                fig = plot_temporal_entity(
                    timesteps,
                    ent_by_cat[cat],
                    category_name=cat,
                    method_name=method,
                    scenario_idx=scenario_idx,
                    validity_series=validity_series.get(cat),
                    top_n=5,
                )
                fig.savefig(
                    os.path.join(plot_dir, f"{prefix}_{method}_{cat}.png"),
                    dpi=150, bbox_inches="tight",
                )
                plt.close(fig)
            except Exception:
                pass

    # 3. Sparsity over time (Gini, one line per method)
    gini_series = {}
    for method, metrics in sparsity_series.items():
        if "gini" in metrics:
            gini_series[method] = metrics["gini"]
    if gini_series:
        try:
            fig = plot_temporal_sparsity(
                timesteps, gini_series, "gini", scenario_idx
            )
            fig.savefig(
                os.path.join(plot_dir, f"{prefix}_sparsity_gini.png"),
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)
        except Exception:
            pass

    # 4. Multi-method comparison over time (grid: one subplot per category)
    if len(method_cat_series) >= 2:
        try:
            fig = plot_temporal_multi_method(
                timesteps, method_cat_series, scenario_idx
            )
            fig.savefig(
                os.path.join(plot_dir, f"{prefix}_multi_method.png"),
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)
        except Exception:
            pass


def _plot_faithfulness_summary(
    result: AnalysisResult,
    plot_dir: str,
    prefix: str,
) -> None:
    """Bar chart of deletion/insertion AUC per method."""
    methods = list(result.faithfulness.keys())
    if not methods:
        return

    del_aucs = [result.faithfulness[m].get("deletion_auc", 0) for m in methods]
    ins_aucs = [result.faithfulness[m].get("insertion_auc", 0) for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].barh(methods, del_aucs, color="salmon")
    axes[0].set_xlabel("Deletion AUC (lower = better)")
    axes[0].set_title(f"Deletion AUC — s{result.scenario_idx:03d} t{result.timestep:03d}")

    axes[1].barh(methods, ins_aucs, color="steelblue")
    axes[1].set_xlabel("Insertion AUC (higher = better)")
    axes[1].set_title(f"Insertion AUC — s{result.scenario_idx:03d} t{result.timestep:03d}")

    plt.tight_layout()
    fig.savefig(
        os.path.join(plot_dir, f"{prefix}_faithfulness.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)
