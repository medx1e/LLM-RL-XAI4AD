"""BEV (Bird's Eye View) video renderer using Waymax native visualization.

Uses ``waymax.visualization.plot_simulator_state()`` for the base frame
(proper world coordinates, full road geometry, vehicle polygons) and overlays
event markers, info text, and a timeline bar on top.

Supports two modes:
- **Model rollout**: Runs an episode with the policy and renders each step.
- **Logged trajectory**: Replays the logged Waymo data (no model needed).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from event_mining.events.base import Event, EventType


# Event type → color mapping
EVENT_COLORS = {
    EventType.HAZARD_ONSET: "#FF6600",
    EventType.COLLISION_IMMINENT: "#FF0000",
    EventType.HARD_BRAKE: "#CC00CC",
    EventType.EVASIVE_STEERING: "#9900FF",
    EventType.NEAR_MISS: "#FFCC00",
    EventType.COLLISION: "#FF0000",
    EventType.OFF_ROAD: "#00CCFF",
}

# Matplotlib color names for overlay drawing
EVENT_COLORS_RGB = {
    EventType.HAZARD_ONSET: (1.0, 0.4, 0.0),
    EventType.COLLISION_IMMINENT: (1.0, 0.0, 0.0),
    EventType.HARD_BRAKE: (0.8, 0.0, 0.8),
    EventType.EVASIVE_STEERING: (0.6, 0.0, 1.0),
    EventType.NEAR_MISS: (1.0, 0.8, 0.0),
    EventType.COLLISION: (1.0, 0.0, 0.0),
    EventType.OFF_ROAD: (0.0, 0.8, 1.0),
}


def _unbatch_state(state):
    """Remove batch dimension from a Waymax SimulatorState."""
    import jax.tree_util as jtu
    return jtu.tree_map(
        lambda x: x[0] if hasattr(x, "shape") and len(x.shape) > 0 else x,
        state,
    )


def _get_active_events(events: list[Event], timestep: int) -> list[Event]:
    """Return events whose window contains the given timestep."""
    return [e for e in events if e.window[0] <= timestep <= e.window[1]]


def _overlay_event_info(
    img: np.ndarray,
    events: list[Event],
    timestep: int,
    total_steps: int,
) -> np.ndarray:
    """Add event info overlay and timeline bar to a Waymax frame.

    Takes a (H, W, 3) uint8 image and returns a new image with overlays.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from io import BytesIO

    active = _get_active_events(events, timestep)

    h, w = img.shape[:2]
    # Create figure matching image size + timeline bar
    dpi = 100
    fig_w = w / dpi
    fig_h = (h + 80) / dpi  # extra space for timeline
    fig = Figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="black")

    # Main BEV image
    ax_img = fig.add_axes([0, 80 / (h + 80), 1, h / (h + 80)])
    ax_img.imshow(img)
    ax_img.set_axis_off()

    # Event border
    if active:
        most_severe = max(active, key=lambda e: e.severity_score)
        color = EVENT_COLORS_RGB.get(most_severe.event_type, (1, 1, 1))
        for spine in ax_img.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(4)

    # Info text overlay
    info_lines = [f"Step {timestep}/{total_steps}"]
    for e in active:
        info_lines.append(f"EVENT: {e.event_type.value} ({e.severity.value})")
    info_text = "\n".join(info_lines)
    ax_img.text(
        0.01, 0.99, info_text,
        transform=ax_img.transAxes, fontsize=8, verticalalignment="top",
        color="white", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.75),
    )

    # Timeline bar
    ax_tl = fig.add_axes([0.05, 0.01, 0.9, 60 / (h + 80)])
    ax_tl.set_xlim(0, total_steps)
    ax_tl.set_ylim(0, 1)
    ax_tl.set_facecolor("#1a1a1a")
    ax_tl.set_yticks([])
    ax_tl.tick_params(colors="#888888", labelsize=6)

    for event in events:
        color = EVENT_COLORS.get(event.event_type, "#FFFFFF")
        ax_tl.axvspan(event.onset, event.offset + 1, alpha=0.5, color=color)
        ax_tl.axvline(event.peak, color=color, linewidth=1.5, alpha=0.8)

    ax_tl.axvline(timestep, color="white", linewidth=2, zorder=10)
    ax_tl.set_xlabel("Timestep", fontsize=7, color="#AAAAAA")
    for spine in ax_tl.spines.values():
        spine.set_edgecolor("#444444")

    # Render figure to numpy array
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    from PIL import Image
    result = np.array(Image.open(buf))[:, :, :3]  # drop alpha if present
    buf.close()
    return result


def render_model_video(
    model,
    scenario,
    events: list[Event] | None = None,
    output_path: str = "bev_model.mp4",
    fps: int = 10,
    use_log_traj: bool = False,
    scenario_id: str = "",
    rng_seed: int = 0,
) -> str:
    """Render a BEV video by running a model rollout on a scenario.

    Uses Waymax's native ``plot_simulator_state`` for base frames and
    overlays event markers on top.

    Args:
        model: An ExplainableModel (from ``posthoc_xai.load_model()``).
        scenario: A raw scenario from the data generator.
        events: Events to overlay (from event mining).
        output_path: Output file path (.mp4 or .gif).
        fps: Frames per second.
        use_log_traj: Show logged trajectory trace on the Waymax frame.
        scenario_id: Label for the scenario.
        rng_seed: Random seed for the episode rollout.

    Returns:
        Path to the output video file.
    """
    import jax
    from functools import partial
    from vmax.agents import pipeline
    from waymax import visualization

    events = events or []
    loaded = model._loaded

    step_fn = partial(pipeline.policy_step, env=loaded.env, policy_fn=loaded.policy_fn)
    jit_reset = jax.jit(loaded.env.reset)

    # Reset
    rng_key = jax.random.PRNGKey(rng_seed)
    rng_key, reset_key = jax.random.split(rng_key)
    env_transition = jit_reset(scenario, jax.random.split(reset_key, 1))

    # Collect frames
    frames = []
    step = 0

    while not bool(env_transition.done):
        state = _unbatch_state(env_transition.state)
        base_img = visualization.plot_simulator_state(state, use_log_traj=use_log_traj)
        base_img = np.array(base_img)
        if base_img.dtype != np.uint8:
            base_img = (base_img * 255).astype(np.uint8)

        frames.append((step, base_img))

        rng_key, step_key = jax.random.split(rng_key)
        env_transition, _ = step_fn(env_transition, key=jax.random.split(step_key, 1))
        step += 1

    # Capture final frame
    state = _unbatch_state(env_transition.state)
    base_img = np.array(visualization.plot_simulator_state(state, use_log_traj=use_log_traj))
    if base_img.dtype != np.uint8:
        base_img = (base_img * 255).astype(np.uint8)
    frames.append((step, base_img))

    total_steps = step

    # Add event overlays to each frame
    overlay_frames = []
    for t, img in frames:
        overlay_frames.append(_overlay_event_info(img, events, t, total_steps))

    # Write video
    _write_video(overlay_frames, output_path, fps)
    print(f"  Video saved to {output_path} ({len(overlay_frames)} frames)")
    return output_path


def render_logged_video(
    tfrecord_path: str,
    record_index: int = 0,
    events: list[Event] | None = None,
    output_path: str = "bev_logged.mp4",
    fps: int = 10,
    max_num_objects: int = 128,
    max_num_rg_points: int = 30000,
) -> str:
    """Render a BEV video from logged Waymo trajectory data (no model needed).

    Uses the same approach as ``render_waymax_record.py`` but adds event overlays.

    Args:
        tfrecord_path: Path to the .tfrecord file.
        record_index: Which scenario/record to render (0-based).
        events: Events to overlay.
        output_path: Output file path (.mp4 or .gif).
        fps: Frames per second.
        max_num_objects: Max objects for the dataloader.
        max_num_rg_points: Max road graph points (30k for full detail).

    Returns:
        Path to the output video file.
    """
    import dataclasses
    from waymax import config as waymax_config
    from waymax import dataloader, datatypes, visualization

    events = events or []

    # Configure dataloader
    base = (
        waymax_config.WOD_1_0_0_TESTING
        if hasattr(waymax_config, "WOD_1_0_0_TESTING")
        else waymax_config.WOD_1_1_0_TRAINING
    )
    cfg = dataclasses.replace(
        base,
        path=tfrecord_path,
        max_num_objects=max_num_objects,
        max_num_rg_points=max_num_rg_points,
        shuffle_seed=None,
        shuffle_buffer_size=1,
        num_shards=1,
        deterministic=True,
        repeat=None,
    )
    data_iter = dataloader.simulator_state_generator(config=cfg)

    # Skip to the desired record
    state = None
    for _ in range(record_index + 1):
        state = next(data_iter)

    # Render frames
    frames = []
    total_steps = int(state.remaining_timesteps)
    step = 0

    base_img = np.array(visualization.plot_simulator_state(state, use_log_traj=True))
    if base_img.dtype != np.uint8:
        base_img = (base_img * 255).astype(np.uint8)
    frames.append((step, base_img))

    for _ in range(total_steps):
        state = datatypes.update_state_by_log(state, num_steps=1)
        step += 1
        base_img = np.array(visualization.plot_simulator_state(state, use_log_traj=True))
        if base_img.dtype != np.uint8:
            base_img = (base_img * 255).astype(np.uint8)
        frames.append((step, base_img))

    # Add event overlays
    overlay_frames = []
    for t, img in frames:
        overlay_frames.append(_overlay_event_info(img, events, t, total_steps))

    _write_video(overlay_frames, output_path, fps)
    print(f"  Video saved to {output_path} ({len(overlay_frames)} frames)")
    return output_path


def render_event_clip(
    model,
    scenario,
    event: Event,
    output_path: str = "bev_event.mp4",
    fps: int = 10,
    padding: int = 5,
    rng_seed: int = 0,
) -> str:
    """Render a clip focused on a specific event from a model rollout.

    Runs the full episode but only saves frames around the event window.

    Args:
        model: An ExplainableModel.
        scenario: A raw scenario from the data generator.
        event: The event to focus on.
        output_path: Output file path.
        fps: Frames per second.
        padding: Extra frames before/after the event window.
        rng_seed: Random seed for the rollout.

    Returns:
        Path to the output video file.
    """
    import jax
    from functools import partial
    from vmax.agents import pipeline
    from waymax import visualization

    loaded = model._loaded

    clip_start = max(0, event.window[0] - padding)
    clip_end = event.window[1] + padding

    step_fn = partial(pipeline.policy_step, env=loaded.env, policy_fn=loaded.policy_fn)
    jit_reset = jax.jit(loaded.env.reset)

    rng_key = jax.random.PRNGKey(rng_seed)
    rng_key, reset_key = jax.random.split(rng_key)
    env_transition = jit_reset(scenario, jax.random.split(reset_key, 1))

    frames = []
    step = 0
    total_steps = 0

    while not bool(env_transition.done):
        if clip_start <= step <= clip_end:
            state = _unbatch_state(env_transition.state)
            base_img = np.array(visualization.plot_simulator_state(state, use_log_traj=False))
            if base_img.dtype != np.uint8:
                base_img = (base_img * 255).astype(np.uint8)
            frames.append((step, base_img))

        rng_key, step_key = jax.random.split(rng_key)
        env_transition, _ = step_fn(env_transition, key=jax.random.split(step_key, 1))
        step += 1

    total_steps = step

    # Overlay events on collected frames
    overlay_frames = []
    for t, img in frames:
        overlay_frames.append(_overlay_event_info(img, [event], t, total_steps))

    _write_video(overlay_frames, output_path, fps)
    print(f"  Event clip saved to {output_path} ({len(overlay_frames)} frames)")
    return output_path


def _write_video(frames: list[np.ndarray], output_path: str, fps: int) -> None:
    """Write frames to video file (MP4 via mediapy, or GIF via PIL)."""
    if not frames:
        print("  No frames to write!")
        return

    if output_path.endswith(".gif"):
        from PIL import Image
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=1000 // fps,
            loop=0,
        )
    else:
        # Try mediapy first (better quality), fall back to matplotlib
        try:
            import mediapy as media
            media.write_video(output_path, frames, fps=fps)
        except ImportError:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation

            h, w = frames[0].shape[:2]
            fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
            ax.set_axis_off()
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            im = ax.imshow(frames[0])

            def update(i):
                im.set_data(frames[i])
                return [im]

            anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 // fps, blit=True)
            anim.save(output_path, writer="ffmpeg", fps=fps)
            plt.close(fig)
