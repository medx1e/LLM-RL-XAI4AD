"""Matplotlib visualisations for Post-hoc XAI tab."""

from __future__ import annotations

from typing import Optional

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np


# ---------------------------------------------------------------------------
# Shared palette & constants
# ---------------------------------------------------------------------------

_CAT_COLORS = {
    "sdc_trajectory": "#4C72B0",
    "sdc":            "#4C72B0",
    "other_agents":   "#DD8452",
    "agents":         "#DD8452",
    "roadgraph":      "#55A868",
    "traffic_lights": "#C44E52",
    "gps_path":       "#8172B2",
    "gps":            "#8172B2",
}
_DEFAULT_COLOR = "#999999"

# Attention token layout (280 tokens total)
_ATTN_SDC_END     = 5
_ATTN_AGENT_START = 5
_ATTN_N_AGENTS    = 8
_ATTN_ROAD_START  = 45
_ATTN_ROAD_END    = 245
_ATTN_TL_START    = 245
_ATTN_N_TLS       = 5
_ATTN_GPS_START   = 270
_ATTN_GPS_END     = 280

# Number of tokens per category (for per-token normalization)
_CAT_TOKEN_COUNTS = {
    "sdc_trajectory": 5,
    "sdc":            5,
    "other_agents":   40,
    "agents":         40,
    "roadgraph":      200,
    "traffic_lights": 25,
    "gps_path":       10,
    "gps":            10,
}

_WAYMAX_TYPE_LABELS = {0: "?", 1: "Veh", 2: "Ped", 3: "Cyc"}
_AGENT_TYPE_COLORS  = {
    "Veh": "#DD8452",
    "Ped": "#E377C2",
    "Cyc": "#BCBD22",
    "?":   _DEFAULT_COLOR,
}

# Categorical identity colours for the 8 observation agent slots.
# Same colour is used on the BEV vehicle AND its bar in the attention chart,
# so the user can visually match "red vehicle on map" ↔ "red bar in chart".
AGENT_ID_COLORS = [
    "#E41A1C",  # A0 red
    "#377EB8",  # A1 blue
    "#4DAF4A",  # A2 green
    "#984EA3",  # A3 purple
    "#FF7F00",  # A4 orange
    "#FFD92F",  # A5 yellow
    "#A65628",  # A6 brown
    "#F781BF",  # A7 pink
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _obs_slot_to_scene_idx(artifact, step: int) -> list[int]:
    """Return scene agent indices for observation slots 0-7 (closest-first)."""
    frame = artifact.scenario_data.frame_states[step]
    try:
        is_sdc = np.array(frame.object_metadata.is_sdc)          # (A,)
        curr_t = int(np.array(frame.timestep))
        valid  = np.array(frame.sim_trajectory.valid)[:, curr_t]  # (A,)
        x_arr  = np.array(frame.sim_trajectory.x)[:, curr_t]
        y_arr  = np.array(frame.sim_trajectory.y)[:, curr_t]
    except Exception:
        # Fallback: use precomputed arrays
        is_sdc     = np.zeros(len(artifact.scenario_data.agents_valid[step]), dtype=bool)
        valid      = artifact.scenario_data.agents_valid[step]
        x_arr, y_arr = artifact.scenario_data.agents_xy[step].T

    ego_xy = artifact.scenario_data.ego_xy[step]
    valid_non_sdc = valid.astype(bool) & ~is_sdc
    idx = np.where(valid_non_sdc)[0]
    if len(idx) == 0:
        return []
    dists = np.hypot(x_arr[idx] - ego_xy[0], y_arr[idx] - ego_xy[1])
    return list(idx[np.argsort(dists)])


def _agent_type_label(artifact, scene_idx: int) -> str:
    t = int(artifact.scenario_data.agents_types[scene_idx]) if scene_idx < len(artifact.scenario_data.agents_types) else 0
    return _WAYMAX_TYPE_LABELS.get(t, "?")


# ---------------------------------------------------------------------------
# 1. Category importance
# ---------------------------------------------------------------------------

def plot_category_importance(
    attribution,
    normalize_per_token: bool = False,
    title: str = "Feature Category Importance",
    figsize=(7, 3),
) -> plt.Figure:
    """Horizontal bar chart of ``attribution.category_importance``.

    ``normalize_per_token=True`` divides each bar by the number of tokens
    in that category so bars reflect average-per-token importance.
    """
    cat_imp: dict = attribution.category_importance
    if not cat_imp:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No category data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    labels = list(cat_imp.keys())
    values = [float(cat_imp[k]) for k in labels]

    if normalize_per_token:
        values = [
            v / _CAT_TOKEN_COUNTS.get(k, 1)
            for k, v in zip(labels, values)
        ]
        xlabel = "Importance per token"
    else:
        xlabel = "Importance (normalised)"

    colors = [_CAT_COLORS.get(k, _DEFAULT_COLOR) for k in labels]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(labels, values, color=colors)
    ax.set_xlabel(xlabel)
    ax.set_title(title + (" (per token)" if normalize_per_token else ""))
    ax.set_xlim(left=0)
    vmax = max(values) if values else 1
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_width() + vmax * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.4f}", va="center", fontsize=8,
        )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Entity importance
# ---------------------------------------------------------------------------

def _flatten_entity_importance(ent_imp: dict) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, val in ent_imp.items():
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                flat[sub_key] = float(sub_val)
        else:
            flat[key] = float(val)
    return flat


def _is_category_level(ent_imp: dict) -> bool:
    for val in ent_imp.values():
        if isinstance(val, dict):
            vals = list(val.values())
            if len(vals) > 1 and (max(vals) - min(vals)) > 1e-8:
                return False
    return True


def _entity_color(name: str) -> str:
    nl = name.lower()
    if nl.startswith("agent") or nl.startswith("veh") or nl.startswith("ped") or nl.startswith("cyc"):
        return _CAT_COLORS["other_agents"]
    if nl.startswith("road"):
        return _CAT_COLORS["roadgraph"]
    if nl.startswith("tl") or nl.startswith("traffic"):
        return _CAT_COLORS["traffic_lights"]
    if nl.startswith("sdc"):
        return _CAT_COLORS["sdc"]
    if nl.startswith("gps"):
        return _CAT_COLORS["gps"]
    return _DEFAULT_COLOR


def plot_entity_importance(
    attribution,
    top_n: int = 10,
    normalize: bool = False,
    title: str = "Top Entity Importances",
    figsize=(7, 4),
) -> plt.Figure:
    ent_imp: dict = attribution.entity_importance
    if not ent_imp:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No entity data", ha="center", va="center")
        ax.set_axis_off()
        return fig

    if _is_category_level(ent_imp):
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5,
            "This method provides category-level resolution only.\nSee the Category chart.",
            ha="center", va="center", fontsize=10, color="#555",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig

    flat = _flatten_entity_importance(ent_imp)

    if normalize:
        total = sum(abs(v) for v in flat.values()) or 1.0
        flat = {k: v / total for k, v in flat.items()}

    sorted_items = sorted(flat.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top = sorted_items[:top_n]
    labels = [k for k, _ in top]
    values = [v for _, v in top]
    colors = [_entity_color(k) for k in labels]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Relative importance" if normalize else "Importance")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Attribution timeline
# ---------------------------------------------------------------------------

def plot_attribution_timeline(
    series: list,
    method_name: str = "",
    current_step: Optional[int] = None,
    figsize=(9, 2),
) -> plt.Figure:
    magnitudes = []
    for attr in series:
        if attr is None:
            magnitudes.append(float("nan"))
        else:
            magnitudes.append(float(np.linalg.norm(np.array(attr.raw).ravel())))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(magnitudes, color="#4C72B0", linewidth=1.2)
    ax.fill_between(range(len(magnitudes)), magnitudes, alpha=0.15, color="#4C72B0")
    if current_step is not None:
        ax.axvline(current_step, color="red", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("‖attribution‖")
    ax.set_title(f"Attribution magnitude{f' — {method_name}' if method_name else ''}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Episode info
# ---------------------------------------------------------------------------

def plot_episode_info(
    artifact,
    current_step: Optional[int] = None,
    figsize=(9, 2.5),
) -> plt.Figure:
    rewards  = np.array(artifact.scenario_data.rewards)
    dones    = np.array(artifact.scenario_data.dones).astype(bool)
    cumulative = np.cumsum(rewards)
    steps = np.arange(len(rewards))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.plot(steps, rewards, color="#4C72B0", linewidth=1.2)
    ax1.fill_between(steps, rewards, alpha=0.15, color="#4C72B0")
    for cs in np.where(dones)[0]:
        ax1.axvline(cs, color="#C44E52", linewidth=1.5, linestyle=":", alpha=0.8)
    if current_step is not None:
        ax1.axvline(current_step, color="red", linewidth=1.2, linestyle="--")
    ax1.set_xlabel("Timestep"); ax1.set_ylabel("Reward"); ax1.set_title("Per-step reward")

    ax2.plot(steps, cumulative, color="#55A868", linewidth=1.2)
    ax2.fill_between(steps, cumulative, alpha=0.15, color="#55A868")
    if current_step is not None:
        ax2.axvline(current_step, color="red", linewidth=1.2, linestyle="--")
    ax2.set_xlabel("Timestep"); ax2.set_ylabel("Cumulative")
    ax2.set_title(f"Cumulative reward  (total {cumulative[-1]:.2f})")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Semantic attention bar chart
# ---------------------------------------------------------------------------

def aggregate_attention_by_entity(
    attn_dict: dict,
    key: str = "cross_attn_avg",
    artifact=None,
    step: Optional[int] = None,
) -> dict[str, float]:
    """Aggregate cross-attention into named semantic entities.

    If ``artifact`` and ``step`` are provided, agent slots are labelled by
    Waymax object type (Vehicle / Pedestrian / Cyclist).
    """
    if key not in attn_dict:
        return {}

    attn = np.array(attn_dict[key])
    if attn.ndim == 3:
        attn = attn[0]                   # drop batch → (latents, tokens)
    per_token = attn.mean(axis=0)        # average over latent queries → (280,)
    total = per_token.sum()
    if total > 0:
        per_token = per_token / total

    # Build scene-index → type label mapping if artifact provided
    type_labels: list[str] = []
    if artifact is not None and step is not None:
        scene_order = _obs_slot_to_scene_idx(artifact, step)
        for i in range(_ATTN_N_AGENTS):
            if i < len(scene_order):
                type_labels.append(_agent_type_label(artifact, scene_order[i]))
            else:
                type_labels.append("?")
    else:
        type_labels = ["?" for _ in range(_ATTN_N_AGENTS)]

    result: dict[str, float] = {}
    result["SDC (ego)"] = float(per_token[0:_ATTN_SDC_END].sum())
    for i in range(_ATTN_N_AGENTS):
        s = _ATTN_AGENT_START + i * 5
        lbl = f"A{i} · {type_labels[i]}"
        result[lbl] = float(per_token[s: s + 5].sum())
    result["Roadgraph (mean/tok)"] = float(per_token[_ATTN_ROAD_START:_ATTN_ROAD_END].mean())
    for i in range(_ATTN_N_TLS):
        s = _ATTN_TL_START + i * 5
        result[f"TL {i}"] = float(per_token[s: s + 5].sum())
    result["GPS path"] = float(per_token[_ATTN_GPS_START:_ATTN_GPS_END].sum())

    return result


def _entity_attn_color(name: str) -> str:
    if name.startswith("SDC"):
        return _CAT_COLORS["sdc"]
    if name.startswith("A") and len(name) >= 2 and name[1].isdigit():
        # "A3 · Veh" → identity colour for slot 3
        try:
            idx = int(name[1])
            return AGENT_ID_COLORS[idx]
        except (ValueError, IndexError):
            return _CAT_COLORS["other_agents"]
    if name.startswith("Road"):
        return _CAT_COLORS["roadgraph"]
    if name.startswith("TL"):
        return _CAT_COLORS["traffic_lights"]
    if name.startswith("GPS"):
        return _CAT_COLORS["gps"]
    return _DEFAULT_COLOR


def plot_attention_by_entity(
    attn_dict: dict,
    key: str = "cross_attn_avg",
    artifact=None,
    step: Optional[int] = None,
    title: str = "Attention by semantic entity",
    figsize=(8, 4),
) -> Optional[plt.Figure]:
    entity_attn = aggregate_attention_by_entity(attn_dict, key=key, artifact=artifact, step=step)
    if not entity_attn:
        return None

    labels = list(entity_attn.keys())
    values = [entity_attn[k] for k in labels]
    colors = [_entity_attn_color(k) for k in labels]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(labels, values, color=colors)
    ax.set_xlabel("Normalised attention weight (sum over tokens)")
    ax.set_title(title)
    ax.set_xlim(left=0)
    vmax = max(values) if values else 1
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_width() + vmax * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}", va="center", fontsize=7,
        )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Attention BEV overlay  (coloured vehicle rectangles)
# ---------------------------------------------------------------------------

def make_attention_overlay_fn(artifact, attn_series: list, selected_agents: set | None = None):
    """Return overlay_fn(ax, step) drawing identity-coloured vehicle boxes.

    Principle: **colour = identity, opacity = attention magnitude.**
    Each observation slot 0–7 has a fixed categorical colour (AGENT_ID_COLORS)
    shared with its bar in the attention chart — so the user matches
    "red vehicle on map" ↔ "red bar in chart" at a glance.
    Attention weight modulates alpha (0.35 → 0.95) and edge width.
    A small white-on-black digit in each box disambiguates identity even
    for colour-blind viewers.
    """
    import matplotlib.patheffects as pe

    def overlay_fn(ax: plt.Axes, step: int) -> None:
        if attn_series is None or step >= len(attn_series) or attn_series[step] is None:
            return

        entity_attn = aggregate_attention_by_entity(attn_series[step], artifact=artifact, step=step)
        if not entity_attn:
            return

        # Trajectory data at this step
        frame   = artifact.scenario_data.frame_states[step]
        curr_t  = int(np.array(frame.timestep))
        traj    = frame.sim_trajectory
        x_arr   = np.array(traj.x)[:, curr_t]
        y_arr   = np.array(traj.y)[:, curr_t]
        yaw_arr = np.array(traj.yaw)[:, curr_t]
        len_arr = np.array(traj.length)[:, curr_t]
        wid_arr = np.array(traj.width)[:, curr_t]

        scene_order = _obs_slot_to_scene_idx(artifact, step)

        def _lookup(i: int) -> float:
            for k, v in entity_attn.items():
                if k.startswith(f"A{i} "):
                    return v
            return 0.0

        agent_weights = np.array([_lookup(i) for i in range(_ATTN_N_AGENTS)])
        w_max = float(agent_weights.max()) if agent_weights.max() > 0 else 1.0

        for obs_idx, scene_idx in enumerate(scene_order[:_ATTN_N_AGENTS]):
            if selected_agents is not None and obs_idx not in selected_agents:
                continue
            w_norm = float(agent_weights[obs_idx]) / w_max  # 0..1
            alpha  = 0.35 + 0.60 * w_norm                   # identity always visible
            lw     = 1.2 + 2.8 * w_norm                     # border thickness ∝ weight

            color = AGENT_ID_COLORS[obs_idx]
            xi, yi     = x_arr[scene_idx], y_arr[scene_idx]
            li, wi_dim = max(float(len_arr[scene_idx]), 1.0), max(float(wid_arr[scene_idx]), 0.5)
            yawi       = float(yaw_arr[scene_idx])
            atype      = _agent_type_label(artifact, scene_idx)

            if atype == "Ped":
                # Circle for pedestrians (they are small, no meaningful yaw)
                r = max(li, wi_dim) / 2
                patch = mpatches.Circle((xi, yi), radius=r,
                                        linewidth=lw, edgecolor="black",
                                        facecolor=color, alpha=alpha)
                ax.add_patch(patch)
            elif atype == "Cyc":
                # Diamond (rotated square) for cyclists
                side = max(li, wi_dim)
                diamond = mpatches.RegularPolygon(
                    (0, 0), numVertices=4, radius=side * 0.65,
                    orientation=np.pi / 4,
                    linewidth=lw, edgecolor="black",
                    facecolor=color, alpha=alpha,
                )
                t = mtransforms.Affine2D().rotate(yawi).translate(xi, yi) + ax.transData
                diamond.set_transform(t)
                ax.add_patch(diamond)
            else:
                # Rectangle for vehicles (and unknown types)
                rect = mpatches.Rectangle(
                    (-li / 2, -wi_dim / 2), li, wi_dim,
                    linewidth=lw, edgecolor="black",
                    facecolor=color, alpha=alpha,
                )
                t = mtransforms.Affine2D().rotate(yawi).translate(xi, yi) + ax.transData
                rect.set_transform(t)
                ax.add_patch(rect)

            # Centred identity digit (white with black outline)
            txt = ax.text(
                xi, yi, str(obs_idx),
                ha="center", va="center",
                fontsize=9, fontweight="bold",
                color="white", zorder=10,
            )
            txt.set_path_effects([pe.withStroke(linewidth=2.0, foreground="black")])

    return overlay_fn
