"""Render BEV videos for the scenarios where events were detected.

Produces GIFs with event overlays so we can visually verify whether
the mined events correspond to real critical driving moments.
"""

import os
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

VMAX_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "V-Max")
if VMAX_REPO not in sys.path:
    sys.path.insert(0, VMAX_REPO)

MODEL_DIR = "runs_rlc/womd_sac_road_perceiver_minimal_42"
DATA_PATH = "data/training.tfrecord"
OUTPUT_DIR = "experiments/event_xai_results/videos"
N_SCENARIOS = 2  # only need s000 and s001

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load model ───────────────────────────────────────────────
print("Loading model...")
import posthoc_xai as xai
model = xai.load_model(MODEL_DIR, data_path=DATA_PATH)
print(f"Model loaded: {model.name}")

# ── Load event catalog ───────────────────────────────────────
print("\nLoading event catalog...")
from event_mining.catalog import EventCatalog
catalog = EventCatalog.load("experiments/event_xai_results/catalog.json")
print(f"Catalog: {len(catalog)} events")

# ── Load scenarios ───────────────────────────────────────────
print("\nLoading scenarios from data generator...")
from vmax.simulator import make_data_generator

loaded = model._loaded
data_gen = make_data_generator(
    path=DATA_PATH,
    max_num_objects=loaded.config.get("max_num_objects", 64),
    include_sdc_paths=True,
    batch_dims=(1,),
    seed=42,
    repeat=1,
)

scenarios = {}
data_iter = iter(data_gen)
for i in range(N_SCENARIOS):
    sid = f"s{i:03d}"
    scenarios[sid] = next(data_iter)
    print(f"  Loaded {sid}")

# ── Render videos ────────────────────────────────────────────
from event_mining.visualization.bev_video import render_model_video, render_event_clip

for sid, scenario in scenarios.items():
    events_for_scenario = [e for e in catalog.events if e.scenario_id == sid]
    rng_seed = int(sid[1:])  # s000 -> 0, s001 -> 1

    print(f"\n{'='*50}")
    print(f"Rendering {sid}: {len(events_for_scenario)} events")
    for e in events_for_scenario:
        print(f"  {e.event_type.value} sev={e.severity.value} t={e.onset}-{e.offset}")

    # Full scenario video with all events overlaid
    out_path = os.path.join(OUTPUT_DIR, f"{sid}_full.gif")
    print(f"\n  Rendering full scenario video...")
    render_model_video(
        model, scenario, events_for_scenario,
        output_path=out_path, fps=10, rng_seed=rng_seed,
    )

    # Focused clips for the events we analyzed with XAI
    analyzed_events = [
        e for e in events_for_scenario
        if e.event_type.value in ("evasive_steering", "hazard_onset")
        and e.severity.value in ("medium", "low", "critical")
    ]
    for j, event in enumerate(analyzed_events[:3]):
        clip_path = os.path.join(
            OUTPUT_DIR,
            f"{sid}_clip_{event.event_type.value}_t{event.onset}.gif"
        )
        print(f"  Rendering clip: {event.event_type.value} t={event.onset}-{event.offset}...")
        render_event_clip(
            model, scenario, event,
            output_path=clip_path, fps=10, padding=8, rng_seed=rng_seed,
        )

print(f"\n\nAll videos saved to {OUTPUT_DIR}/")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"  {f:50s} {size/1024:.1f} KB")
