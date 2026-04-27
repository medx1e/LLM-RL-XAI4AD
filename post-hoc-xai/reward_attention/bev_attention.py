"""BEV + attention visualization for the reward-conditioned attention experiment.

Two outputs:
  1. Scenario-colored scatter (no GPU) — reads saved timestep_data.pkl
  2. BEV attention video (needs GPU) — runs one episode, renders per-step attention

Usage:
    cd /home/med1e/post-hoc-xai
    eval "$(~/anaconda3/bin/conda shell.bash hook)" && conda activate vmax
    export PYTHONPATH=/home/med1e/post-hoc-xai/V-Max:$PYTHONPATH

    # Scatter only (fast, no GPU):
    python reward_attention/bev_attention.py --scatter-only

    # Full: scatter + BEV video for scenario 0
    python reward_attention/bev_attention.py --scenario-idx 0
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from functools import partial
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCENARIO_COLORS = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00"]
ATTN_CATEGORY_COLORS = {
    "attn_sdc":       "#2196F3",
    "attn_agents":    "#F44336",
    "attn_roadgraph": "#4CAF50",
    "attn_lights":    "#FF9800",
    "attn_gps":       "#9C27B0",
}
ATTN_LABELS = {
    "attn_sdc":       "Ego (SDC)",
    "attn_agents":    "Other Agents",
    "attn_roadgraph": "Road Graph",
    "attn_lights":    "Traffic Lights",
    "attn_gps":       "GPS Path",
}

PKL_PATH = Path("results/reward_attention/womd_sac_road_perceiver_complete_42/timestep_data.pkl")
EVENTS_PATH = Path("events/test_catalog.json")


# ---------------------------------------------------------------------------
# 1. Scenario-colored scatter
# ---------------------------------------------------------------------------


def plot_scenario_colored_scatter(
    pkl_path: Path = PKL_PATH,
    save_path: Path | None = None,
) -> None:
    """Scatter of collision_risk × attn_agents, colored by scenario.

    Shows within-episode regression lines for each scenario separately,
    making it visually clear why the pooled ρ=+0.02 masks within-episode ρ=+0.70.
    """
    with open(pkl_path, "rb") as f:
        records = pickle.load(f)

    from scipy import stats

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#111111")

    scenario_ids = sorted(set(r.scenario_id for r in records))

    all_rho_lines = []

    for i, sid in enumerate(scenario_ids):
        recs = [r for r in records if r.scenario_id == sid]
        x = np.array([r.collision_risk for r in recs])
        y = np.array([r.attn_agents for r in recs])
        color = SCENARIO_COLORS[i % len(SCENARIO_COLORS)]

        # Compute within-episode Spearman ρ
        rho, p = stats.spearmanr(x, y)
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        all_rho_lines.append((sid, rho, p, sig))

        # Scatter
        ax.scatter(x, y, color=color, alpha=0.55, s=28, zorder=3,
                   label=f"s{sid:03d}  ρ={rho:+.2f}{sig}  n={len(recs)}")

        # Within-episode OLS regression line
        if np.std(x) > 1e-4:
            slope, intercept, *_ = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, slope * x_line + intercept,
                    color=color, linewidth=2.2, alpha=0.9, zorder=4)

    # Pooled regression (dashed gray — shows the confounded result)
    all_x = np.array([r.collision_risk for r in records])
    all_y = np.array([r.attn_agents for r in records])
    pool_rho, pool_p = stats.spearmanr(all_x, all_y)
    slope, intercept, *_ = stats.linregress(all_x, all_y)
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, slope * x_line + intercept,
            color="white", linewidth=1.5, linestyle="--", alpha=0.5, zorder=5,
            label=f"Pooled  ρ={pool_rho:+.2f}  (confounded)")

    # Labels
    ax.set_xlabel("Collision Risk  (1 – TTC / 3s, clipped [0,1])", fontsize=12, color="#cccccc")
    ax.set_ylabel("Attention → Other Agents  (fraction of total)", fontsize=12, color="#cccccc")
    ax.set_title(
        "Within-Episode Attention Tracks Risk\n"
        "Pooled ρ=+0.02 masks within-episode ρ≈+0.70",
        fontsize=13, color="white", pad=10,
    )
    ax.tick_params(colors="#888888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    legend = ax.legend(
        fontsize=9, loc="upper left",
        facecolor="#1a1a1a", edgecolor="#444444", labelcolor="white",
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, max(all_y) + 0.03)

    # Annotation box
    ax.text(0.98, 0.04,
            "Within-episode regression lines (colored)\n"
            "vs pooled regression (white dashed)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="#aaaaaa",
            bbox=dict(facecolor="#1a1a1a", edgecolor="#333333", alpha=0.8, boxstyle="round"))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="#0d0d0d")
        print(f"Saved scenario-colored scatter to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. BEV attention video
# ---------------------------------------------------------------------------


def _load_events_for_scenario(scenario_id: int) -> list[dict]:
    """Load events for a given integer scenario_id from the catalog."""
    sid_str = f"s{scenario_id:03d}"
    with open(EVENTS_PATH) as f:
        cat = json.load(f)
    return [e for e in cat["events"] if e["scenario_id"] == sid_str]


def _attn_overlay_frame(
    base_img: np.ndarray,
    attn_dict: dict,
    timestep: int,
    total_steps: int,
    collision_risk: float,
    events: list[dict],
) -> np.ndarray:
    """Compose BEV frame + attention panel + timeline bar.

    Layout:
        Left  (75%): Waymax BEV frame
        Right (25%): Attention stacked bar + risk gauge
        Bottom:      Timeline bar with event markers
    """
    H, W = base_img.shape[:2]
    panel_w = W // 4           # right panel width
    timeline_h = 70            # bottom timeline bar height
    total_h = H + timeline_h
    total_w = W + panel_w

    dpi = 100
    fig = plt.figure(figsize=(total_w / dpi, total_h / dpi), dpi=dpi)
    fig.patch.set_facecolor("#0d0d0d")

    # --- BEV image ---
    ax_bev = fig.add_axes([0, timeline_h / total_h, W / total_w, H / total_h])
    ax_bev.imshow(base_img)
    ax_bev.set_axis_off()

    # Event border color
    active_events = [e for e in events if e["window"][0] <= timestep <= e["window"][1]]
    if active_events:
        most_severe = max(active_events, key=lambda e: {"critical": 3, "high": 2, "medium": 1, "low": 0}.get(e.get("severity", "low"), 0))
        sev = most_severe.get("severity", "low")
        border_color = {"critical": "#FF3300", "high": "#FF7700", "medium": "#FFCC00", "low": "#00CCFF"}.get(sev, "white")
        for spine in ax_bev.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_color)
            spine.set_linewidth(5)

    # Event text
    event_lines = [f"t={timestep}/{total_steps}"]
    for e in active_events:
        event_lines.append(f"► {e['event_type'].replace('_', ' ').upper()}")
        if "min_ttc" in e.get("metadata", {}):
            event_lines.append(f"  TTC={e['metadata']['min_ttc']:.2f}s")
    ax_bev.text(0.01, 0.99, "\n".join(event_lines),
                transform=ax_bev.transAxes, fontsize=7, va="top", ha="left",
                color="white", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))

    # --- Attention panel (right) ---
    ax_attn = fig.add_axes([W / total_w, (timeline_h + 30) / total_h,
                             panel_w / total_w, (H - 60) / total_h])
    ax_attn.set_facecolor("#111111")

    cats = ["attn_sdc", "attn_agents", "attn_roadgraph", "attn_lights", "attn_gps"]
    cat_labels = ["Ego", "Agents", "Road", "Lights", "GPS"]
    cat_colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]

    values = [attn_dict.get(c, 0.0) for c in cats]

    bars = ax_attn.barh(cat_labels[::-1], values[::-1],
                        color=cat_colors[::-1], height=0.6)
    ax_attn.set_xlim(0, max(max(values) * 1.2, 0.5))
    ax_attn.set_xlabel("Attention fraction", color="#888888", fontsize=7)
    ax_attn.tick_params(colors="#888888", labelsize=7)
    for spine in ax_attn.spines.values():
        spine.set_edgecolor("#333333")
    ax_attn.set_facecolor("#111111")

    # Value labels on bars
    for bar, val in zip(bars, values[::-1]):
        ax_attn.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", ha="left",
                     color="white", fontsize=6.5)

    # Risk gauge above bars
    ax_risk = fig.add_axes([W / total_w, (total_h - 50) / total_h,
                             panel_w / total_w, 40 / total_h])
    ax_risk.set_facecolor("#111111")
    cmap = cm.RdYlGn_r
    risk_color = cmap(collision_risk)
    ax_risk.barh(["Risk"], [collision_risk], color=[risk_color], height=0.6)
    ax_risk.barh(["Risk"], [1.0 - collision_risk], left=[collision_risk],
                 color="#222222", height=0.6)
    ax_risk.set_xlim(0, 1)
    ax_risk.set_axis_off()
    ax_risk.text(0.5, 0.5, f"Collision Risk: {collision_risk:.2f}",
                 transform=ax_risk.transAxes, ha="center", va="center",
                 fontsize=7.5, color="white", fontweight="bold")

    # --- Timeline bar (bottom) ---
    ax_tl = fig.add_axes([0, 0, (W + panel_w) / total_w, (timeline_h - 5) / total_h])
    ax_tl.set_facecolor("#111111")
    ax_tl.set_xlim(0, total_steps)
    ax_tl.set_ylim(0, 1)
    ax_tl.set_yticks([])
    ax_tl.tick_params(colors="#666666", labelsize=6)

    event_colors_map = {
        "hazard_onset": "#FF6600",
        "collision_imminent": "#FF0000",
        "near_miss": "#FFCC00",
        "evasive_steering": "#9900FF",
    }
    for ev in events:
        ec = event_colors_map.get(ev["event_type"], "#FFFFFF")
        ax_tl.axvspan(ev["onset"], ev.get("offset", ev["onset"]) + 1,
                      alpha=0.35, color=ec)
        ax_tl.axvline(ev["peak"], color=ec, linewidth=1.2, alpha=0.75)

    ax_tl.axvline(timestep, color="white", linewidth=2.0, zorder=10)
    ax_tl.set_xlabel("Timestep", fontsize=6, color="#888888")
    for spine in ax_tl.spines.values():
        spine.set_edgecolor("#333333")

    # Render to numpy
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0,
                facecolor="#0d0d0d")
    plt.close(fig)
    buf.seek(0)
    from PIL import Image
    result = np.array(Image.open(buf))[:, :, :3]
    buf.close()
    return result


def render_attention_bev(
    model,
    scenario_idx: int = 0,
    output_path: Path | None = None,
    fps: int = 8,
    attention_layer: str = "avg",
    save_frames_dir: Path | None = None,
) -> Path:
    """Run episode for scenario_idx, render BEV + attention overlay as GIF.

    Uses:
    - waymax.visualization.plot_simulator_state() for base BEV frames
    - model.get_attention(obs) for per-step attention weights
    - event catalog for timeline markers

    Args:
        model: ExplainableModel (PerceiverWrapper).
        scenario_idx: Which scenario in the data generator (0-based).
        output_path: Output path (.gif). Defaults to results/.../*.gif
        fps: Frames per second for GIF.
        attention_layer: "avg" or "0"-"3".
        save_frames_dir: If set, also save individual PNG frames here.

    Returns:
        Path to output GIF.
    """
    import jax
    import jax.numpy as jnp
    from functools import partial
    from vmax.agents import pipeline
    from waymax import visualization

    from reward_attention.config import TOKEN_RANGES

    if output_path is None:
        out_dir = Path("results/reward_attention/womd_sac_road_perceiver_complete_42")
        output_path = out_dir / f"fig5_bev_attention_s{scenario_idx:03d}.gif"

    events = _load_events_for_scenario(scenario_idx)
    print(f"  Loaded {len(events)} events for scenario {scenario_idx}")

    loaded = model._loaded
    step_fn = partial(pipeline.policy_step, env=loaded.env, policy_fn=loaded.policy_fn)
    jit_reset = jax.jit(loaded.env.reset)

    # Reset
    rng_key = jax.random.PRNGKey(0)
    rng_key, reset_key = jax.random.split(rng_key)
    env_transition = jit_reset(scenario_idx_to_scenario(model, scenario_idx),
                                jax.random.split(reset_key, 1))

    # Pre-load risk data from pkl if available (for collision_risk per step)
    risk_by_step: dict[int, float] = {}
    if PKL_PATH.exists():
        with open(PKL_PATH, "rb") as f:
            all_records = pickle.load(f)
        for r in all_records:
            if r.scenario_id == scenario_idx:
                risk_by_step[r.timestep] = r.collision_risk

    # Collect frames
    frames = []
    step = 0

    def _get_attention(obs_np: np.ndarray) -> dict:
        """Run forward pass and extract attention fractions."""
        obs_batch = jnp.array(obs_np[None])  # (1, obs_dim)
        attn_raw = model.get_attention(obs_batch)

        if attn_raw is None or len(attn_raw) == 0:
            return {k: 0.0 for k in ["attn_sdc", "attn_agents", "attn_roadgraph", "attn_lights", "attn_gps"]}

        # Choose layer
        if attention_layer == "avg":
            key = "cross_attn_avg"
        else:
            key = f"cross_attn_layer_{attention_layer}"
        attn = attn_raw.get(key, list(attn_raw.values())[0])
        # attn shape: (batch, n_queries, n_tokens) or (n_queries, n_tokens)
        attn = np.array(attn)
        if attn.ndim == 3:
            attn = attn[0]   # (n_queries, n_tokens)
        avg_over_queries = attn.mean(axis=0)  # (n_tokens,)
        # Normalize to fractions
        total = avg_over_queries.sum()
        if total > 1e-10:
            avg_over_queries = avg_over_queries / total

        result = {}
        for cat, (lo, hi) in TOKEN_RANGES.items():
            result[f"attn_{cat}"] = float(avg_over_queries[lo:hi].sum())
        # Rename to match our convention
        out = {
            "attn_sdc":       result.get("attn_sdc", 0.0),
            "attn_agents":    result.get("attn_other_agents", 0.0),
            "attn_roadgraph": result.get("attn_roadgraph", 0.0),
            "attn_lights":    result.get("attn_traffic_lights", 0.0),
            "attn_gps":       result.get("attn_gps_path", 0.0),
        }
        return out

    def _unbatch_state(state):
        import jax.tree_util as jtu
        return jtu.tree_map(
            lambda x: x[0] if hasattr(x, "shape") and len(x.shape) > 0 else x,
            state,
        )

    print("  Running episode loop ...")
    while not bool(env_transition.done):
        obs = env_transition.observation  # (1, obs_dim)
        obs_np = np.array(obs)

        # Attention
        attn_dict = _get_attention(obs_np)

        # Collision risk from preloaded records (faster than recomputing)
        collision_risk = risk_by_step.get(step, 0.0)

        # Base BEV frame from Waymax
        state = _unbatch_state(env_transition.state)
        base_img = np.array(visualization.plot_simulator_state(state, use_log_traj=False))
        if base_img.dtype != np.uint8:
            base_img = (np.clip(base_img, 0, 1) * 255).astype(np.uint8)

        # Compose frame
        frame = _attn_overlay_frame(
            base_img, attn_dict, step,
            total_steps=80,
            collision_risk=collision_risk,
            events=events,
        )
        frames.append(frame)

        if save_frames_dir:
            save_frames_dir.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            Image.fromarray(frame).save(save_frames_dir / f"frame_{step:03d}.png")

        # Step
        rng_key, step_key = jax.random.split(rng_key)
        env_transition, _ = step_fn(env_transition, key=jax.random.split(step_key, 1))
        step += 1

        if step % 10 == 0:
            print(f"    t={step:3d}  collision_risk={collision_risk:.3f}  "
                  f"attn_agents={attn_dict.get('attn_agents', 0):.3f}  "
                  f"attn_road={attn_dict.get('attn_roadgraph', 0):.3f}")

    print(f"  Episode done: {step} steps, {len(frames)} frames")

    # Write GIF
    from PIL import Image
    pil_frames = [Image.fromarray(f) for f in frames]
    if pil_frames:
        pil_frames[0].save(
            str(output_path),
            save_all=True,
            append_images=pil_frames[1:],
            duration=1000 // fps,
            loop=0,
            optimize=False,
        )
    print(f"  GIF saved to {output_path}  ({len(frames)} frames @ {fps} fps)")
    return output_path


def scenario_idx_to_scenario(model, idx: int):
    """Advance the model's data generator to scenario idx and return it."""
    # The data generator is already open; we advance it idx steps
    # Note: this is destructive — resets generator state
    # For a fresh generator, re-load the model or reset the generator
    loaded = model._loaded
    # Try to get a fresh generator
    if hasattr(loaded, "data_gen"):
        gen = loaded.data_gen
        scenario = None
        for i in range(idx + 1):
            try:
                scenario = next(gen)
            except StopIteration:
                raise ValueError(f"Data generator exhausted at index {i} (wanted {idx})")
        return scenario
    else:
        raise AttributeError("model._loaded has no data_gen attribute")


# ---------------------------------------------------------------------------
# Key-timestep panel figure (no video, just key moments)
# ---------------------------------------------------------------------------


def plot_key_timestep_panel(
    model,
    pkl_path: Path = PKL_PATH,
    scenario_idx: int = 0,
    key_timesteps: list[int] | None = None,
    save_path: Path | None = None,
) -> None:
    """Multi-panel: BEV at key timesteps + attention bar below each.

    Much faster than a full video — only renders a few frames.
    """
    import jax
    import jax.numpy as jnp
    from functools import partial
    from vmax.agents import pipeline
    from waymax import visualization
    from reward_attention.config import TOKEN_RANGES

    if key_timesteps is None:
        # Hazard onset (t=9), peak (t=35), and two other interesting timesteps
        key_timesteps = [5, 9, 20, 35, 50]

    events = _load_events_for_scenario(scenario_idx)

    # Load risk data
    risk_by_step: dict[int, float] = {}
    attn_by_step: dict[int, dict] = {}
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            all_records = pickle.load(f)
        for r in all_records:
            if r.scenario_id == scenario_idx:
                risk_by_step[r.timestep] = r.collision_risk
                attn_by_step[r.timestep] = {
                    "attn_sdc":       r.attn_sdc,
                    "attn_agents":    r.attn_agents,
                    "attn_roadgraph": r.attn_roadgraph,
                    "attn_lights":    r.attn_lights,
                    "attn_gps":       r.attn_gps,
                }

    print(f"  Attention data from pkl: {len(attn_by_step)} timesteps")

    # Run episode to collect BEV frames at key timesteps
    loaded = model._loaded
    step_fn = partial(pipeline.policy_step, env=loaded.env, policy_fn=loaded.policy_fn)
    jit_reset = jax.jit(loaded.env.reset)

    scenario = scenario_idx_to_scenario(model, scenario_idx)
    rng_key = jax.random.PRNGKey(0)
    rng_key, reset_key = jax.random.split(rng_key)
    env_transition = jit_reset(scenario, jax.random.split(reset_key, 1))

    def _unbatch_state(state):
        import jax.tree_util as jtu
        return jtu.tree_map(
            lambda x: x[0] if hasattr(x, "shape") and len(x.shape) > 0 else x,
            state,
        )

    bev_by_step: dict[int, np.ndarray] = {}
    step = 0
    print("  Collecting BEV frames ...")
    while not bool(env_transition.done) and step <= max(key_timesteps):
        if step in key_timesteps:
            state = _unbatch_state(env_transition.state)
            img = np.array(visualization.plot_simulator_state(state, use_log_traj=False))
            if img.dtype != np.uint8:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            bev_by_step[step] = img
            print(f"    Captured BEV at t={step}")

        rng_key, step_key = jax.random.split(rng_key)
        env_transition, _ = step_fn(env_transition, key=jax.random.split(step_key, 1))
        step += 1

    # Build multi-panel figure
    n_cols = len(key_timesteps)
    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(4.5 * n_cols, 7),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.patch.set_facecolor("#0d0d0d")

    cats = ["attn_sdc", "attn_agents", "attn_roadgraph", "attn_lights", "attn_gps"]
    cat_labels_short = ["Ego", "Agents", "Road", "Lights", "GPS"]
    cat_colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]

    active_event_times = set()
    for ev in events:
        for t in range(ev["window"][0], ev["window"][1] + 1):
            active_event_times.add(t)

    for col, t in enumerate(key_timesteps):
        ax_bev = axes[0, col] if n_cols > 1 else axes[0]
        ax_bar = axes[1, col] if n_cols > 1 else axes[1]

        # BEV
        ax_bev.set_facecolor("#111111")
        if t in bev_by_step:
            ax_bev.imshow(bev_by_step[t])
        ax_bev.set_axis_off()

        risk = risk_by_step.get(t, 0.0)
        in_event = t in active_event_times
        border_c = "#FF3300" if (risk > 0.7 or in_event) else ("#FF7700" if risk > 0.4 else "#444444")
        for spine in ax_bev.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_c)
            spine.set_linewidth(3)

        # Title
        active_ev_types = [ev["event_type"].replace("_", " ") for ev in events
                           if ev["window"][0] <= t <= ev["window"][1]]
        ev_label = f"\n[{active_ev_types[0]}]" if active_ev_types else ""
        ax_bev.set_title(f"t = {t}{ev_label}\nRisk = {risk:.2f}",
                         fontsize=9, color="white", pad=3)

        # Attention bar
        ax_bar.set_facecolor("#111111")
        if t in attn_by_step:
            vals = [attn_by_step[t].get(c, 0.0) for c in cats]
            bars = ax_bar.bar(cat_labels_short, vals, color=cat_colors, width=0.6)
            for bar, val in zip(bars, vals):
                ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{val:.2f}", ha="center", va="bottom",
                            color="white", fontsize=6.5)
        ax_bar.set_ylim(0, 0.7)
        ax_bar.set_ylabel("Attn fraction" if col == 0 else "", fontsize=7, color="#888888")
        ax_bar.tick_params(colors="#888888", labelsize=6.5)
        for spine in ax_bar.spines.values():
            spine.set_edgecolor("#333333")

    # Title row
    fig.suptitle(
        f"Perceiver Attention at Key Timesteps — Scenario s{scenario_idx:03d}\n"
        "Hazard onset: t=9, Peak: t=35  (red border = active event or risk > 0.7)",
        fontsize=11, color="white", y=1.01,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight",
                    facecolor="#0d0d0d")
        print(f"  Saved key-timestep panel to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Attention time-series overlay
# ---------------------------------------------------------------------------


def plot_attention_risk_timeseries(
    pkl_path: Path = PKL_PATH,
    scenario_idx: int = 0,
    save_path: Path | None = None,
) -> None:
    """Line plot: collision_risk + attn_agents + attn_roadgraph over time.

    Annotates event windows. Best companion to the BEV figure.
    """
    with open(pkl_path, "rb") as f:
        all_records = pickle.load(f)

    recs = sorted([r for r in all_records if r.scenario_id == scenario_idx],
                  key=lambda r: r.timestep)
    if not recs:
        print(f"No records for scenario {scenario_idx}")
        return

    events = _load_events_for_scenario(scenario_idx)
    ts = np.array([r.timestep for r in recs])
    risk = np.array([r.collision_risk for r in recs])
    attn_ag = np.array([r.attn_agents for r in recs])
    attn_rd = np.array([r.attn_roadgraph for r in recs])
    attn_sdc = np.array([r.attn_sdc for r in recs])

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.patch.set_facecolor("#0d0d0d")

    # Top: risk
    ax1 = axes[0]
    ax1.set_facecolor("#111111")
    ax1.fill_between(ts, risk, alpha=0.35, color="#F44336")
    ax1.plot(ts, risk, color="#F44336", linewidth=2, label="Collision Risk")
    ax1.set_ylabel("Collision Risk", color="#F44336", fontsize=10)
    ax1.set_ylim(-0.02, 1.05)
    ax1.tick_params(colors="#888888")
    ax1.yaxis.label.set_color("#F44336")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#333333")

    # Bottom: attention
    ax2 = axes[1]
    ax2.set_facecolor("#111111")
    ax2.plot(ts, attn_ag, color="#F44336", linewidth=2, label="Agents")
    ax2.plot(ts, attn_rd, color="#4CAF50", linewidth=2, label="Road Graph")
    ax2.plot(ts, attn_sdc, color="#2196F3", linewidth=2, label="Ego (SDC)")
    ax2.set_ylabel("Attention Fraction", color="#cccccc", fontsize=10)
    ax2.set_xlabel("Timestep", color="#cccccc", fontsize=10)
    ax2.tick_params(colors="#888888")
    ax2.legend(fontsize=9, facecolor="#1a1a1a", edgecolor="#444444", labelcolor="white", loc="upper right")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333333")

    # Event shading on both panels
    event_colors_map = {
        "hazard_onset": "#FF6600",
        "collision_imminent": "#FF0000",
        "near_miss": "#FFCC00",
        "evasive_steering": "#9900FF",
    }
    for ev in events:
        ec = event_colors_map.get(ev["event_type"], "#FFFFFF")
        for ax in axes:
            ax.axvspan(ev["onset"], ev.get("offset", ev["onset"]),
                       alpha=0.12, color=ec, zorder=0)
        ax1.axvline(ev["peak"], color=ec, linewidth=1.0, alpha=0.7, linestyle="--")

    # Annotation for hazard_onset
    for ev in events:
        if ev["event_type"] == "hazard_onset" and ev["peak"] == 35:
            ax1.annotate("Hazard\nPeak",
                         xy=(ev["peak"], risk[ts == ev["peak"]][0] if ev["peak"] in ts else 1.0),
                         xytext=(ev["peak"] + 3, 0.85),
                         fontsize=8, color="#FF6600",
                         arrowprops=dict(arrowstyle="->", color="#FF6600"))

    fig.suptitle(
        f"Collision Risk & Perceiver Attention over Time — Scenario s{scenario_idx:03d}\n"
        f"Within-episode Spearman ρ(risk, attn_agents) = +0.617**  |  ρ(risk, attn_road) = −0.535**",
        fontsize=10, color="white",
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
        print(f"  Saved risk+attention time series to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="BEV + attention visualization")
    parser.add_argument("--model", default="runs_rlc/womd_sac_road_perceiver_complete_42")
    parser.add_argument("--data", default="data/training.tfrecord")
    parser.add_argument("--scenario-idx", type=int, default=0,
                        help="Scenario index (0-based) for BEV video")
    parser.add_argument("--output", default="results/reward_attention/womd_sac_road_perceiver_complete_42")
    parser.add_argument("--scatter-only", action="store_true",
                        help="Only generate scenario-colored scatter (no GPU needed)")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip full GIF video (saves time), only render key-timestep panel")
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Scenario-colored scatter (always, no GPU) ---
    print("\n[1] Generating scenario-colored scatter ...")
    scatter_path = out_dir / "fig_scenario_scatter.png"
    plot_scenario_colored_scatter(save_path=scatter_path)

    # --- 2. Attention time series (no GPU) ---
    print(f"\n[2] Generating attention time series for scenario {args.scenario_idx} ...")
    ts_path = out_dir / f"fig_timeseries_s{args.scenario_idx:03d}.png"
    plot_attention_risk_timeseries(scenario_idx=args.scenario_idx, save_path=ts_path)

    if args.scatter_only:
        print("\nDone (scatter + timeseries only — skipping BEV).")
        return

    # --- 3. Load model (GPU) ---
    print("\n[3] Loading model ...")
    import posthoc_xai as xai
    model = xai.load_model(args.model, data_path=args.data)
    print(f"    Wrapper: {type(model).__name__}")

    # --- 4. Key-timestep panel ---
    print(f"\n[4] Generating key-timestep panel for scenario {args.scenario_idx} ...")
    panel_path = out_dir / f"fig_bev_panel_s{args.scenario_idx:03d}.png"
    plot_key_timestep_panel(
        model,
        scenario_idx=args.scenario_idx,
        key_timesteps=[5, 9, 20, 35, 50, 65],
        save_path=panel_path,
    )

    # --- 5. Full BEV GIF ---
    if not args.no_video:
        print(f"\n[5] Rendering full BEV attention GIF for scenario {args.scenario_idx} ...")
        gif_path = out_dir / f"fig_bev_attention_s{args.scenario_idx:03d}.gif"
        render_attention_bev(
            model,
            scenario_idx=args.scenario_idx,
            output_path=gif_path,
            fps=args.fps,
        )

    print(f"\nDone. All outputs in {out_dir}")


if __name__ == "__main__":
    main()
