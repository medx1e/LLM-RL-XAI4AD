"""Attribution visualization: bar plots, method comparisons, curves."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from matplotlib.patches import Patch

from posthoc_xai.methods.base import Attribution


def plot_category_importance(
    attribution: Attribution,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Bar plot of per-category importance for a single method."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    categories = list(attribution.category_importance.keys())
    values = [attribution.category_importance[c] for c in categories]

    bars = ax.bar(categories, values, color="steelblue")
    ax.set_ylabel("Importance")
    ax.set_title(title or f"{attribution.method_name} — Category Importance")
    ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1.0)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    return fig


def plot_method_comparison(
    attributions: list[Attribution],
    title: str = "Method Comparison",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Side-by-side bar plot comparing multiple methods' category importance."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    categories = list(attributions[0].category_importance.keys())
    n_methods = len(attributions)
    n_categories = len(categories)

    x = np.arange(n_categories)
    width = 0.8 / n_methods
    colors = plt.cm.Set2(np.linspace(0, 1, n_methods))

    for i, attr in enumerate(attributions):
        values = [attr.category_importance[c] for c in categories]
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=attr.method_name, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    return fig


def plot_deletion_insertion_curves(
    deletion_data: list[tuple[str, np.ndarray, np.ndarray]],
    insertion_data: list[tuple[str, np.ndarray, np.ndarray]],
) -> plt.Figure:
    """Plot deletion and insertion curves for multiple methods.

    Each entry in the data lists is ``(method_name, percentages, outputs)``.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(deletion_data), 1)))

    # Deletion curve
    ax = axes[0]
    for i, (name, pcts, outputs) in enumerate(deletion_data):
        ax.plot(pcts, outputs, label=name, color=colors[i], linewidth=2)
    ax.set_xlabel("Fraction of Features Removed")
    ax.set_ylabel("Model Output")
    ax.set_title("Deletion Curve (lower is better)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Insertion curve
    ax = axes[1]
    for i, (name, pcts, outputs) in enumerate(insertion_data):
        ax.plot(pcts, outputs, label=name, color=colors[i], linewidth=2)
    ax.set_xlabel("Fraction of Features Added")
    ax.set_ylabel("Model Output")
    ax.set_title("Insertion Curve (higher is better)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_entity_importance(
    attribution: Attribution,
    validity: Optional[dict[str, dict[str, bool]]] = None,
    categories: Optional[list[str]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Bar plot of per-entity importance, with invalid entities grayed out.

    Args:
        attribution: Attribution result with ``entity_importance``.
        validity: Per-entity validity from ``model.get_entity_validity(obs)``.
            If ``None``, all entities are shown as valid.
        categories: Which categories to plot.  Defaults to
            ``["other_agents", "traffic_lights", "gps_path"]``.
        title: Figure title.

    Returns:
        Matplotlib figure.
    """
    if categories is None:
        categories = ["other_agents", "traffic_lights", "gps_path"]

    n_cats = len(categories)
    fig, axes = plt.subplots(1, n_cats, figsize=(5 * n_cats, 5))
    if n_cats == 1:
        axes = [axes]

    for ax, cat in zip(axes, categories):
        entities = attribution.entity_importance.get(cat, {})
        names = list(entities.keys())
        values = [entities[n] for n in names]

        # Color by validity
        colors = []
        for name in names:
            if validity and cat in validity:
                is_valid = validity[cat].get(name, True)
            else:
                is_valid = True
            colors.append("#2196F3" if is_valid else "#BDBDBD")

        bars = ax.barh(names, values, color=colors)
        ax.set_xlabel("Importance")
        ax.set_title(cat.replace("_", " ").title())
        ax.invert_yaxis()

        # Value labels
        for bar, val in zip(bars, values):
            if val > 0.001:
                ax.text(
                    bar.get_width() + 0.002,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}",
                    va="center",
                    fontsize=8,
                )

    # Legend
    if validity:
        legend_elements = [
            Patch(facecolor="#2196F3", label="Valid"),
            Patch(facecolor="#BDBDBD", label="Invalid"),
        ]
        fig.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=9,
        )

    fig.suptitle(
        title or f"{attribution.method_name} — Per-Entity Importance",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    return fig


def plot_temporal_category(
    timesteps: list[int],
    category_series: dict[str, list[float]],
    method_name: str = "",
    scenario_idx: Optional[int] = None,
    events: Optional[dict[int, str]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Line plot of category importance over timesteps.

    Args:
        timesteps: Sorted list of timestep indices.
        category_series: ``{category_name: [importance_t0, importance_t1, ...]}``.
        method_name: XAI method identifier (for title).
        scenario_idx: Scenario index (for title).
        events: Optional ``{timestep: label}`` to mark events (e.g. collision).
        ax: Optional existing axes.

    Returns:
        Matplotlib figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure

    colors = plt.cm.Set2(np.linspace(0, 1, len(category_series)))
    for (cat, values), color in zip(category_series.items(), colors):
        ax.plot(timesteps, values, marker="o", linewidth=2, label=cat, color=color)

    if events:
        for ts, label in events.items():
            ax.axvline(ts, color="red", linestyle="--", alpha=0.6)
            ax.text(ts, ax.get_ylim()[1] * 0.95, f" {label}", color="red",
                    fontsize=8, va="top")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Importance")
    scn = f" — scenario {scenario_idx}" if scenario_idx is not None else ""
    ax.set_title(f"{method_name} — Category Importance Over Time{scn}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_temporal_category_stacked(
    timesteps: list[int],
    category_series: dict[str, list[float]],
    method_name: str = "",
    scenario_idx: Optional[int] = None,
    events: Optional[dict[int, str]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Stacked area chart of category importance over timesteps.

    Shows how the relative composition of feature importance evolves.

    Args:
        timesteps: Sorted list of timestep indices.
        category_series: ``{category_name: [importance_t0, importance_t1, ...]}``.
        method_name: XAI method identifier.
        scenario_idx: Scenario index.
        events: Optional ``{timestep: label}`` for event markers.
        ax: Optional existing axes.

    Returns:
        Matplotlib figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure

    categories = list(category_series.keys())
    values = np.array([category_series[c] for c in categories])
    colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))

    ax.stackplot(timesteps, values, labels=categories, colors=colors, alpha=0.8)

    if events:
        for ts, label in events.items():
            ax.axvline(ts, color="red", linestyle="--", alpha=0.8, linewidth=1.5)
            ax.text(ts, ax.get_ylim()[1] * 0.95, f" {label}", color="red",
                    fontsize=8, va="top")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Importance")
    scn = f" — scenario {scenario_idx}" if scenario_idx is not None else ""
    ax.set_title(f"{method_name} — Category Importance (Stacked){scn}")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(timesteps[0], timesteps[-1])
    plt.tight_layout()
    return fig


def plot_temporal_entity(
    timesteps: list[int],
    entity_series: dict[str, list[float]],
    category_name: str = "other_agents",
    method_name: str = "",
    scenario_idx: Optional[int] = None,
    validity_series: Optional[dict[str, list[bool]]] = None,
    top_n: int = 5,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Line plot of top-N entity importance over timesteps.

    Args:
        timesteps: Sorted list of timestep indices.
        entity_series: ``{entity_name: [importance_t0, importance_t1, ...]}``.
        category_name: Category (for title).
        method_name: XAI method identifier.
        scenario_idx: Scenario index.
        validity_series: ``{entity_name: [valid_t0, valid_t1, ...]}`` for
            marking invalid regions.
        top_n: Show only the top N entities by max importance.
        ax: Optional existing axes.

    Returns:
        Matplotlib figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure

    # Rank entities by max importance across all timesteps
    max_imp = {e: max(vals) for e, vals in entity_series.items()}
    top_entities = sorted(max_imp, key=max_imp.get, reverse=True)[:top_n]

    colors = plt.cm.tab10(np.linspace(0, 1, len(top_entities)))
    for entity, color in zip(top_entities, colors):
        values = entity_series[entity]
        label = entity
        if validity_series and entity in validity_series:
            # Mark invalid regions
            valid = validity_series[entity]
            any_invalid = not all(valid)
            if any_invalid:
                label += " *"
        ax.plot(timesteps, values, marker="o", linewidth=2, label=label, color=color)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Importance")
    cat_label = category_name.replace("_", " ").title()
    scn = f" — scenario {scenario_idx}" if scenario_idx is not None else ""
    ax.set_title(f"{method_name} — {cat_label} Over Time (top {top_n}){scn}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_temporal_sparsity(
    timesteps: list[int],
    sparsity_series: dict[str, list[float]],
    metric_name: str = "gini",
    scenario_idx: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Line plot of sparsity over timesteps, one line per method.

    Args:
        timesteps: Sorted list of timestep indices.
        sparsity_series: ``{method_name: [metric_val_t0, metric_val_t1, ...]}``.
        metric_name: Which sparsity metric (for labels).
        scenario_idx: Scenario index.
        ax: Optional existing axes.

    Returns:
        Matplotlib figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure

    colors = plt.cm.Set1(np.linspace(0, 1, len(sparsity_series)))
    for (method, values), color in zip(sparsity_series.items(), colors):
        ax.plot(timesteps, values, marker="o", linewidth=2, label=method, color=color)

    ax.set_xlabel("Timestep")
    label_map = {"gini": "Gini Coefficient", "entropy": "Normalized Entropy",
                 "top_10_concentration": "Top-10 Concentration",
                 "top_50_concentration": "Top-50 Concentration"}
    ax.set_ylabel(label_map.get(metric_name, metric_name))
    scn = f" — scenario {scenario_idx}" if scenario_idx is not None else ""
    ax.set_title(f"Sparsity ({metric_name}) Over Time{scn}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_temporal_multi_method(
    timesteps: list[int],
    method_category_series: dict[str, dict[str, list[float]]],
    scenario_idx: Optional[int] = None,
) -> plt.Figure:
    """Grid of subplots: one per category, lines for different methods.

    Shows whether methods agree on temporal dynamics.

    Args:
        timesteps: Sorted list of timestep indices.
        method_category_series: ``{method_name: {category: [val_t0, val_t1, ...]}}``.
        scenario_idx: Scenario index.

    Returns:
        Matplotlib figure.
    """
    methods = list(method_category_series.keys())
    # Collect all categories
    all_cats = set()
    for cat_series in method_category_series.values():
        all_cats.update(cat_series.keys())
    categories = sorted(all_cats)
    n_cats = len(categories)

    cols = min(3, n_cats)
    rows = (n_cats + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    method_colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))

    for idx, cat in enumerate(categories):
        r, c = divmod(idx, cols)
        ax = axes[r][c]

        for method, color in zip(methods, method_colors):
            values = method_category_series[method].get(cat, [0] * len(timesteps))
            ax.plot(timesteps, values, marker="o", linewidth=1.5, label=method,
                    color=color, markersize=4)

        ax.set_title(cat.replace("_", " ").title(), fontsize=10)
        ax.set_xlabel("Timestep", fontsize=8)
        ax.set_ylabel("Importance", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    # Hide unused subplots
    for idx in range(n_cats, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    # Single legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(methods)),
               fontsize=9, bbox_to_anchor=(0.5, 1.02))

    scn = f" — scenario {scenario_idx}" if scenario_idx is not None else ""
    fig.suptitle(f"Method Comparison Over Time{scn}", fontsize=13, y=1.06)
    plt.tight_layout()
    return fig


def plot_agent_comparison(
    attributions: list[Attribution],
    validity: Optional[dict[str, dict[str, bool]]] = None,
    title: str = "Per-Agent Importance — Method Comparison",
) -> plt.Figure:
    """Compare per-agent importance across multiple XAI methods.

    Args:
        attributions: List of Attribution results from different methods.
        validity: Per-entity validity (used for background shading).
        title: Figure title.

    Returns:
        Matplotlib figure.
    """
    agent_names = list(attributions[0].entity_importance["other_agents"].keys())
    n_agents = len(agent_names)
    n_methods = len(attributions)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(n_agents)
    width = 0.8 / n_methods
    colors = plt.cm.Set2(np.linspace(0, 1, n_methods))

    for i, attr in enumerate(attributions):
        values = [attr.entity_importance["other_agents"][a] for a in agent_names]
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=attr.method_name, color=colors[i])

    # Shade invalid agents
    if validity and "other_agents" in validity:
        for j, name in enumerate(agent_names):
            if not validity["other_agents"].get(name, True):
                ax.axvspan(j - 0.45, j + 0.45, color="#EEEEEE", zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(agent_names, rotation=30, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig
