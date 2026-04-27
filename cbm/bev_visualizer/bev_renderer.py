"""
BEV Renderer.

Wraps the existing vmax/Waymax Matplotlib visualization to produce numpy
images that can be embedded into Streamlit via st.image().

Public API
----------
render_frame(waymax_state, overlay_fn=None)  -> np.ndarray  (H, W, 3)
render_episode(frame_states, overlay_fn=None) -> list[np.ndarray]

Overlay Protocol
----------------
overlay_fn is an optional callable with signature:
    overlay_fn(ax: matplotlib.axes.Axes, step: int) -> None
It is called after the base BEV is drawn, before the image is captured.
Use it to add concept circles, attention heatmaps, or any annotation
layer — without touching the renderer core.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering — must be set before pyplot import
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "V-Max"))

from waymax import config as waymax_config, datatypes
from waymax.visualization import utils as waymax_utils
from vmax.simulator import overrides


# ── Core frame renderer ───────────────────────────────────────────────────────

def render_frame(
    waymax_state: datatypes.SimulatorState,
    overlay_fn: Callable | None = None,
    step: int = 0,
    fig_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Render a single Waymax SimulatorState as a BEV numpy image.

    Parameters
    ----------
    waymax_state : A scalar (unbatched) SimulatorState.
    overlay_fn   : Optional callable(ax, step) to add extra annotations.
    step         : Frame index — passed to overlay_fn for context.
    fig_size     : Matplotlib figure size in inches.

    Returns
    -------
    np.ndarray of shape (H, W, 3), dtype uint8.
    """
    viz_cfg = waymax_utils.VizConfig()
    fig, ax = waymax_utils.init_fig_ax(viz_cfg)
    fig.set_size_inches(*fig_size)

    # ── 1. Draw road graph ─────────────────────────────────────────────
    from waymax.visualization import viz as waymax_viz
    waymax_viz.plot_roadgraph_points(ax, waymax_state.roadgraph_points, verbose=False)

    # ── 2. Draw traffic lights ─────────────────────────────────────────
    t = int(np.array(waymax_state.timestep))
    waymax_viz.plot_traffic_light_signals_as_points(
        ax, waymax_state.log_traffic_light, t, verbose=False
    )

    # ── 3. Draw all agent bounding boxes (sim trajectory) ─────────────
    traj = waymax_state.sim_trajectory
    is_controlled = datatypes.get_control_mask(
        waymax_state.object_metadata, waymax_config.ObjectType.SDC
    )
    obj_types = waymax_state.object_metadata.object_types

    overrides.plot_trajectory(
        ax, traj, is_controlled, obj_types, time_idx=t, indices=None
    )

    # ── 4. Center the view on the SDC ─────────────────────────────────
    import numpy as _np
    current_xy = _np.array(traj.xy)[:, t, :]
    is_sdc = _np.array(waymax_state.object_metadata.is_sdc)
    origin_x, origin_y = current_xy[is_sdc][0, :2]
    ax.axis((
        origin_x - viz_cfg.back_x,
        origin_x + viz_cfg.front_x,
        origin_y - viz_cfg.back_y,
        origin_y + viz_cfg.front_y,
    ))

    # ── 5. Optional overlay (attention maps, concept annotations, …) ───
    if overlay_fn is not None:
        overlay_fn(ax, step)

    # ── 6. Convert to numpy image ──────────────────────────────────────
    img = waymax_utils.img_from_fig(fig)
    plt.close(fig)
    return img


# ── Episode renderer ──────────────────────────────────────────────────────────

def render_episode(
    frame_states: list,
    overlay_fn: Callable | None = None,
    fig_size: tuple[int, int] = (8, 8),
) -> list[np.ndarray]:
    """Render every frame in a closed-loop episode.

    Parameters
    ----------
    frame_states : List of scalar SimulatorState objects (from ScenarioData).
    overlay_fn   : Optional callable(ax, step) for extra annotations.
    fig_size     : Figure size in inches.

    Returns
    -------
    List of (H, W, 3) uint8 numpy images, one per frame.
    """
    frames = []
    for step, state in enumerate(frame_states):
        img = render_frame(state, overlay_fn=overlay_fn, step=step, fig_size=fig_size)
        frames.append(img)
    return frames
