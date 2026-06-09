"""
Plotly-based interactive chart functions for the XAI4AD platform.

All functions return a ``plotly.graph_objects.Figure`` for use with
``st.plotly_chart(fig, width='stretch')``.

The dark theme (paper_bgcolor, plot_bgcolor, font colours, gridlines) is
applied automatically via the shared PLOTLY_LAYOUT dict from theme.py.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY_OK = True
except ImportError:  # pragma: no cover
    _PLOTLY_OK = False

from platform.shared.theme import (
    PLOTLY_LAYOUT, CAT_COLORS, METHOD_COLORS, AGENT_COLORS,
    CONCEPT_CATEGORIES, CONCEPT_CATEGORY_COLORS,
)


def _base() -> dict:
    """Return a fresh copy of PLOTLY_LAYOUT for update()."""
    import copy
    return copy.deepcopy(PLOTLY_LAYOUT)


# ─── Attribution charts ───────────────────────────────────────────────────────

def category_importance_chart(
    attribution,
    normalize_per_token: bool = False,
    title: str = "Feature Category Importance",
) -> "go.Figure":
    """Horizontal bar chart of attribution category importances."""
    _CAT_TOKENS = {
        "sdc_trajectory": 5, "sdc": 5,
        "other_agents": 40, "agents": 40,
        "roadgraph": 200,
        "traffic_lights": 25,
        "gps_path": 10, "gps": 10,
    }
    cat_imp = attribution.category_importance
    cats  = list(cat_imp.keys())
    vals  = [float(v) for v in cat_imp.values()]
    if normalize_per_token:
        vals = [
            v / _CAT_TOKENS.get(c, 1)
            for c, v in zip(cats, vals)
        ]
        title += " (per-token)"
    colors = [CAT_COLORS.get(c, "#8C8C8C") for c in cats]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=[c.replace("_", " ") for c in cats],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
            textfont=dict(color="#a7a8b3", size=10),
            hovertemplate="<b>%{y}</b><br>importance: %{x:.4f}<extra></extra>",
        )
    )
    layout = _base()
    layout.update(
        title=dict(text=title, font=dict(size=13, color="#f4f4f7"), x=0),
        height=220,
        yaxis=dict(**layout["yaxis"], categoryorder="total ascending"),
        xaxis=dict(**layout["xaxis"], title="Importance"),
    )
    fig.update_layout(**layout)
    return fig


def entity_importance_chart(
    attribution,
    top_n: int = 10,
    normalize: bool = False,
    title: str = "Top Entity Importances",
) -> "go.Figure":
    """Horizontal bar chart of top-N entity importances."""
    ent_imp = getattr(attribution, "entity_importance", {})
    if not ent_imp:
        return category_importance_chart(attribution, title=title)

    # Flatten nested dict if needed
    flat: dict[str, float] = {}
    for k, v in ent_imp.items():
        if isinstance(v, dict):
            flat.update({f"{k}/{kk}": float(vv) for kk, vv in v.items()})
        else:
            flat[k] = float(v)

    sorted_items = sorted(flat.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    names = [n for n, _ in sorted_items]
    vals  = [v for _, v in sorted_items]

    if normalize and sum(abs(v) for v in vals) > 0:
        total = sum(abs(v) for v in vals)
        vals  = [v / total for v in vals]
        title += " (relative)"

    def _color(name: str) -> str:
        n = name.lower()
        for cat, col in CAT_COLORS.items():
            if cat in n:
                return col
        if n.startswith("a") and len(n) <= 3:
            try:
                idx = int(n[1:])
                return AGENT_COLORS[idx % len(AGENT_COLORS)]
            except ValueError:
                pass
        return "#8C8C8C"

    colors = [_color(n) for n in names]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
            textfont=dict(color="#a7a8b3", size=9),
            hovertemplate="<b>%{y}</b><br>importance: %{x:.4f}<extra></extra>",
        )
    )
    layout = _base()
    layout.update(
        title=dict(text=title, font=dict(size=13, color="#f4f4f7"), x=0),
        height=max(200, top_n * 28),
        yaxis={**layout["yaxis"], "categoryorder": "total ascending", "tickfont": dict(color="#a7a8b3", size=9)},
        xaxis=dict(**layout["xaxis"], title="Importance"),
    )
    fig.update_layout(**layout)
    return fig


def attribution_timeline_chart(
    series: list,
    method_name: str = "",
    current_step: Optional[int] = None,
    color: str = "#6f6ae8",
) -> "go.Figure":
    """Line + filled area showing attribution magnitude across the episode."""
    valid = [(i, s) for i, s in enumerate(series) if s is not None]
    if not valid:
        return go.Figure()

    xs = [i for i, _ in valid]
    ys = []
    for _, s in valid:
        raw = np.array(getattr(s, "raw", []))
        ys.append(float(np.linalg.norm(raw)) if raw.size else 0.0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs, y=ys, mode="lines",
            fill="tozeroy",
            line=dict(color=color, width=2),
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)",
            name=method_name,
            hovertemplate="step %{x}<br>‖attr‖ = %{y:.4f}<extra></extra>",
        )
    )
    if current_step is not None:
        fig.add_vline(
            x=current_step,
            line=dict(color="#f4c25b", width=1.5, dash="dot"),
            annotation_text=f"t={current_step}",
            annotation_font=dict(color="#f4c25b", size=10),
        )
    layout = _base()
    layout.update(
        title=dict(text=f"Attribution magnitude — {method_name}", font=dict(size=12, color="#f4f4f7"), x=0),
        height=160,
        showlegend=False,
        yaxis=dict(**layout["yaxis"], title="‖attribution‖"),
        xaxis=dict(**layout["xaxis"], title="Timestep"),
    )
    fig.update_layout(**layout)
    return fig


# ─── Episode info ─────────────────────────────────────────────────────────────

def episode_info_chart(
    artifact,
    current_step: Optional[int] = None,
) -> "go.Figure":
    """Two-trace chart: per-step reward (area) + cumulative reward (line)."""
    rewards = np.array(artifact.scenario_data.rewards, dtype=float)
    T = len(rewards)
    xs = list(range(T))
    cumulative = np.cumsum(rewards).tolist()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=["Per-step reward", "Cumulative reward"],
    )
    # Per-step reward — split positive/negative
    pos = [r if r >= 0 else None for r in rewards]
    neg = [r if r < 0 else None  for r in rewards]

    fig.add_trace(
        go.Scatter(
            x=xs, y=list(rewards), mode="lines",
            line=dict(color="#a7a8b3", width=1),
            showlegend=False,
            hovertemplate="t=%{x}<br>reward=%{y:.3f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=xs, y=pos, mode="none", fill="tozeroy",
            fillcolor="rgba(47,194,138,0.20)", showlegend=False,
            hoverinfo="skip",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=xs, y=neg, mode="none", fill="tozeroy",
            fillcolor="rgba(224,82,74,0.20)", showlegend=False,
            hoverinfo="skip",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=xs, y=cumulative, mode="lines",
            line=dict(color="#6f6ae8", width=2),
            fill="tozeroy",
            fillcolor="rgba(111,106,232,0.10)",
            showlegend=False,
            hovertemplate="t=%{x}<br>cumulative=%{y:.2f}<extra></extra>",
        ),
        row=2, col=1,
    )
    if current_step is not None:
        for row in (1, 2):
            fig.add_vline(
                x=current_step,
                line=dict(color="#f4c25b", width=1.5, dash="dot"),
                row=row, col=1,
            )

    layout = _base()
    layout.update(
        height=260,
        showlegend=False,
        paper_bgcolor="#25283a",
        plot_bgcolor="#1e2030",
        margin=dict(l=12, r=12, t=36, b=12),
    )
    fig.update_layout(**layout)
    # Style subplot titles
    for ann in fig.layout.annotations:
        ann.update(font=dict(color="#a7a8b3", size=11))
    fig.update_xaxes(gridcolor="#3b3f55", zeroline=False, tickfont=dict(color="#a7a8b3", size=9))
    fig.update_yaxes(gridcolor="#3b3f55", zeroline=False, tickfont=dict(color="#a7a8b3", size=9))
    return fig


# ─── Attention chart ──────────────────────────────────────────────────────────

def attention_entity_chart(
    entity_weights: dict[str, float],
    title: str = "Attention by semantic entity",
) -> "go.Figure":
    """Horizontal bar chart of named entity attention weights."""
    if not entity_weights:
        return go.Figure()

    items = sorted(entity_weights.items(), key=lambda x: x[1], reverse=True)
    names = [n for n, _ in items]
    vals  = [v for _, v in items]

    def _agent_color(name: str) -> str:
        n = name.upper()
        for i, prefix in enumerate(["A0","A1","A2","A3","A4","A5","A6","A7"]):
            if n.startswith(prefix):
                return AGENT_COLORS[i]
        for cat, col in CAT_COLORS.items():
            if cat.lower().replace("_", "") in name.lower().replace(" ", ""):
                return col
        return "#8C8C8C"

    colors = [_agent_color(n) for n in names]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker=dict(color=colors, opacity=0.85, line=dict(width=0)),
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
            textfont=dict(color="#a7a8b3", size=9),
            hovertemplate="<b>%{y}</b><br>attention: %{x:.4f}<extra></extra>",
        )
    )
    layout = _base()
    layout.update(
        title=dict(text=title, font=dict(size=13, color="#f4f4f7"), x=0),
        height=max(180, len(names) * 30 + 60),
        yaxis=dict(**layout["yaxis"], categoryorder="total ascending"),
        xaxis=dict(**layout["xaxis"], title="Attention weight"),
        showlegend=False,
    )
    fig.update_layout(**layout)
    return fig


# ─── Category focus (multi-method grouped bar) ───────────────────────────────

def category_focus_chart(
    cat_data: dict[str, dict[str, float]],
    methods: list[str],
) -> "go.Figure":
    """Grouped horizontal bar chart comparing methods across categories."""
    all_cats = sorted({c for d in cat_data.values() for c in d})
    if not all_cats:
        return go.Figure()

    fig = go.Figure()
    for i, method in enumerate(methods):
        vals = [cat_data.get(method, {}).get(c, 0.0) for c in all_cats]
        color = METHOD_COLORS[i % len(METHOD_COLORS)]
        fig.add_trace(
            go.Bar(
                name=method.replace("_", " "),
                x=vals,
                y=[c.replace("_", " ") for c in all_cats],
                orientation="h",
                marker=dict(color=color, opacity=0.85, line=dict(width=0)),
                hovertemplate=f"<b>{method}</b><br>%{{y}}: %{{x:.3f}}<extra></extra>",
            )
        )
    layout = _base()
    layout.update(
        title=dict(text="Attribution by category", font=dict(size=13, color="#f4f4f7"), x=0),
        barmode="group",
        height=max(220, len(all_cats) * 40 + 80),
        yaxis=dict(**layout["yaxis"], categoryorder="total ascending"),
        xaxis=dict(**layout["xaxis"], title="Normalised importance"),
        legend=dict(**layout["legend"], orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_layout(**layout)
    return fig


# ─── Temporal stability ───────────────────────────────────────────────────────

def temporal_stability_chart(
    series_map: dict[str, list],
    methods: list[str],
    top_k: int = 20,
) -> "go.Figure":
    """Line chart of Jaccard similarity of top-K features across timesteps."""
    fig = go.Figure()
    for i, name in enumerate(methods):
        series = series_map.get(name)
        if not series:
            continue
        valid = [s for s in series if s is not None]
        if len(valid) < 2:
            continue

        def topk_set(attr, k=top_k):
            raw = np.abs(np.array(getattr(attr, "raw", [])).ravel())
            k_ = min(k, len(raw))
            idx = np.argpartition(raw, -k_)[-k_:] if k_ > 0 else np.array([], dtype=int)
            return set(idx.tolist())

        jaccs, xs = [], []
        for t, (a, b) in enumerate(zip(valid[:-1], valid[1:])):
            sa, sb = topk_set(a), topk_set(b)
            union = len(sa | sb)
            jaccs.append(len(sa & sb) / union if union else 0.0)
            xs.append(t + 1)

        if not jaccs:
            continue
        color = METHOD_COLORS[i % len(METHOD_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=xs, y=jaccs,
                mode="lines",
                name=name.replace("_", " "),
                line=dict(color=color, width=1.8),
                fill="tozeroy",
                fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)",
                hovertemplate=f"<b>{name}</b><br>t=%{{x}}<br>Jaccard=%{{y:.3f}}<extra></extra>",
            )
        )

    layout = _base()
    layout.update(
        title=dict(text=f"Top-{top_k} feature set stability", font=dict(size=13, color="#f4f4f7"), x=0),
        height=200,
        yaxis=dict(**layout["yaxis"], title="Jaccard similarity", range=[0, 1.05]),
        xaxis=dict(**layout["xaxis"], title="Timestep"),
        legend=dict(**layout["legend"], orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_layout(**layout)
    return fig


# ─── CBM concept timeline ─────────────────────────────────────────────────────

def concept_timeline_chart(
    pred: np.ndarray,
    true_: np.ndarray,
    valid: np.ndarray,
    cnames: list[str],
    selected: list[str],
    current_step: int,
    accent: str = "#6f6ae8",
    interesting_steps: list[int] | None = None,
    binary_concepts: set[str] | None = None,
) -> "go.Figure":
    """Multi-subplot Plotly figure: one row per selected concept.

    Renders ground-truth (blue solid) vs CBM prediction (orange dashed),
    shades invalid regions, marks interesting timesteps, and draws a
    vertical line at the current BEV step.
    """
    binary_concepts = binary_concepts or set()
    n = len(selected)
    if n == 0:
        return go.Figure()

    subplot_titles = [
        f"{c.replace('_', ' ').title()}{'  (binary)' if c in binary_concepts else ''}"
        for c in selected
    ]
    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06 / max(n, 1),
        subplot_titles=subplot_titles,
    )
    steps_arr = list(range(80))

    # Collect every band/line as a plain shape dict and apply them in ONE
    # update_layout(shapes=...) call at the end. Calling add_vrect/add_vline in
    # a loop is O(n²) (each call reprocesses the full layout) and dominates the
    # build time during BEV playback; batching keeps it ~instant.
    shapes: list[dict] = []
    ar, ag, ab = int(accent[1:3], 16), int(accent[3:5], 16), int(accent[5:7], 16)

    for row_idx, cname in enumerate(selected, start=1):
        if cname not in cnames:
            continue
        ci = cnames.index(cname)
        t_v = true_[:, ci].tolist()
        p_v = pred[:, ci].tolist()
        vm  = valid[:, ci].tolist()

        # Ground truth
        fig.add_trace(
            go.Scatter(
                x=steps_arr, y=t_v, mode="lines",
                name="Ground Truth" if row_idx == 1 else None,
                showlegend=(row_idx == 1),
                line=dict(color="#377EB8", width=2),
                legendgroup="truth",
                hovertemplate="t=%{x}<br>true=%{y:.3f}<extra></extra>",
            ),
            row=row_idx, col=1,
        )
        # Predicted
        fig.add_trace(
            go.Scatter(
                x=steps_arr, y=p_v, mode="lines",
                name="Predicted" if row_idx == 1 else None,
                showlegend=(row_idx == 1),
                line=dict(color="#DD8452", width=1.5, dash="dash"),
                legendgroup="pred",
                hovertemplate="t=%{x}<br>pred=%{y:.3f}<extra></extra>",
            ),
            row=row_idx, col=1,
        )
        # Per-subplot axis refs (row 1 → 'x'/'y', row 2 → 'x2'/'y2', …).
        xref = "x" if row_idx == 1 else f"x{row_idx}"
        yref = ("y" if row_idx == 1 else f"y{row_idx}") + " domain"

        # Invalid mask shading
        for t, is_valid in enumerate(vm):
            if not is_valid:
                shapes.append(dict(
                    type="rect", xref=xref, yref=yref,
                    x0=t - 0.5, x1=t + 0.5, y0=0, y1=1,
                    fillcolor="rgba(128,128,128,0.13)",
                    line=dict(width=0), layer="below",
                ))
        # Interesting timesteps
        for t in (interesting_steps or []):
            shapes.append(dict(
                type="rect", xref=xref, yref=yref,
                x0=t - 0.5, x1=t + 0.5, y0=0, y1=1,
                fillcolor=f"rgba({ar},{ag},{ab},0.18)",
                line=dict(width=0), layer="below",
            ))
        # Current step vertical line
        shapes.append(dict(
            type="line", xref=xref, yref=yref,
            x0=current_step, x1=current_step, y0=0, y1=1,
            line=dict(color=accent, width=1.2, dash="dot"),
        ))

        # Metric annotation
        valid_mask = np.array(vm, dtype=bool)
        if valid_mask.any():
            p_arr, t_arr = np.array(p_v)[valid_mask], np.array(t_v)[valid_mask]
            if cname in binary_concepts:
                acc = float(((p_arr > 0.5) == (t_arr > 0.5)).mean())
                ann_text = f"Acc={acc:.3f}"
            else:
                ss_res = np.sum((t_arr - p_arr) ** 2)
                ss_tot = np.sum((t_arr - t_arr.mean()) ** 2)
                r2 = (1 - ss_res / ss_tot) if ss_tot > 1e-10 else float("nan")
                ann_text = f"R²={r2:.3f}" if r2 == r2 else "R²=n/a"
            fig.add_annotation(
                x=78, y=0.95, text=ann_text,
                showarrow=False,
                font=dict(color="#a7a8b3", size=9),
                xref=f"x{row_idx}", yref=f"y{row_idx}",
                xanchor="right", yanchor="top",
            )

    per_row_h = 130
    layout = _base()
    layout.update(
        height=max(200, per_row_h * n + 60),
        # Constant uirevision → in-place update of the current-step line / data
        # instead of a full redraw (prevents flashing during BEV playback).
        uirevision="concept_timeline",
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            font=dict(color="#a7a8b3", size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="#25283a",
        plot_bgcolor="#1e2030",
        margin=dict(l=12, r=12, t=40, b=12),
        shapes=shapes,
    )
    fig.update_layout(**layout)
    for ann in fig.layout.annotations:
        ann.update(font=dict(color="#a7a8b3", size=10))
    fig.update_xaxes(gridcolor="#3b3f55", zeroline=False, tickfont=dict(color="#a7a8b3", size=9))
    fig.update_yaxes(
        gridcolor="#3b3f55", zeroline=False,
        tickfont=dict(color="#a7a8b3", size=9),
        range=[-0.05, 1.15],
    )
    return fig


def concept_snapshot_chart(
    pred: np.ndarray,
    valid: np.ndarray,
    cnames: list[str],
    current_step: int,
    categories: dict[str, list[str]] | None = None,
    category_colors: dict[str, str] | None = None,
    binary_concepts: set[str] | None = None,
) -> "go.Figure":
    """Horizontal bar chart of the CBM's predicted concept values at one step.

    One bar per concept, grouped and coloured by semantic category (one trace
    per category so the legend reads cleanly). Concepts whose ground-truth is
    not valid at this step are dimmed. Concept order is stable across steps so
    bars don't jump while the episode plays.
    """
    categories      = categories or CONCEPT_CATEGORIES
    category_colors = category_colors or CONCEPT_CATEGORY_COLORS
    binary_concepts = binary_concepts or set()

    fig = go.Figure()
    n_total = 0

    for cat, concept_list in categories.items():
        color = category_colors.get(cat, "#8C8C8C")
        names, vals, opac, texts = [], [], [], []
        for cn in concept_list:
            if cn not in cnames:
                continue
            ci = cnames.index(cn)
            v  = float(pred[current_step, ci])
            is_valid = bool(valid[current_step, ci])
            kind = "binary" if cn in binary_concepts else "continuous"
            names.append(cn.replace("_", " ").title())
            vals.append(v)
            opac.append(1.0 if is_valid else 0.28)
            texts.append(f"{v:.2f}" if is_valid else f"{v:.2f} (n/a)")
        if not names:
            continue
        n_total += len(names)
        fig.add_trace(
            go.Bar(
                x=vals, y=names, orientation="h",
                name=cat,
                marker=dict(color=color, opacity=opac, line=dict(width=0)),
                text=texts, textposition="outside",
                textfont=dict(color="#a7a8b3", size=10),
                hovertemplate="<b>%{y}</b><br>pred=%{x:.3f}<extra>" + cat + "</extra>",
            )
        )

    if n_total == 0:
        return go.Figure()

    layout = _base()
    layout.update(
        title=dict(text=f"Concept Values",
                   font=dict(size=13, color="#f4f4f7"), x=0),
        # Constant uirevision → Plotly updates the bars in place on each rerun
        # (no full redraw/flash) while the BEV player advances the step.
        uirevision="concept_snapshot",
        height=max(260, 26 * n_total + 130),
        barmode="overlay",
        bargap=0.35,
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, x=0,
            font=dict(color="#a7a8b3", size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="#25283a",
        plot_bgcolor="#1e2030",
        margin=dict(l=12, r=40, t=64, b=12),
        xaxis=dict(
            **{k: v for k, v in layout["xaxis"].items()
               if k not in ("zeroline",)},
            range=[-1.05, 1.05],
            zeroline=True, zerolinecolor="#4a4f6b", zerolinewidth=1,
            title="Predicted value",
        ),
        yaxis=dict(**layout["yaxis"], autorange="reversed"),
    )
    fig.update_layout(**layout)
    return fig


def episode_reward_chart(
    artifact,
    current_step: int = 0,
    accent: str = "#6f6ae8",
) -> "go.Figure":
    """Single-panel episode reward with positive/negative fill + step marker."""
    rewards = np.array(artifact.scenario_data.rewards, dtype=float)
    xs = list(range(len(rewards)))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs, y=rewards.tolist(), mode="lines",
            line=dict(color="#a7a8b3", width=1.2),
            showlegend=False,
            hovertemplate="t=%{x}<br>reward=%{y:.3f}<extra></extra>",
        )
    )
    # Positive fill
    pos = [r if r >= 0 else 0 for r in rewards]
    fig.add_trace(
        go.Scatter(
            x=xs, y=pos, mode="none",
            fill="tozeroy", fillcolor="rgba(47,194,138,0.22)",
            showlegend=True, name="Positive",
        )
    )
    # Negative fill
    neg = [r if r < 0 else 0 for r in rewards]
    fig.add_trace(
        go.Scatter(
            x=xs, y=neg, mode="none",
            fill="tozeroy", fillcolor="rgba(224,82,74,0.22)",
            showlegend=True, name="Negative",
        )
    )
    fig.add_vline(
        x=current_step,
        line=dict(color=accent, width=1.5, dash="dot"),
    )
    layout = _base()
    layout.update(
        title=dict(text="Episode Reward", font=dict(size=12, color="#f4f4f7"), x=0),
        height=180,
        yaxis=dict(**layout["yaxis"], title="Reward"),
        xaxis=dict(**layout["xaxis"], title="Timestep"),
        legend=dict(**layout["legend"], orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_layout(**layout)
    return fig
