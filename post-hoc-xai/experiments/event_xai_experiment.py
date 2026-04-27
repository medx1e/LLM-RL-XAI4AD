"""End-to-end experiment: Event Mining → Temporal XAI Analysis.

1. Mine events from N scenarios to find critical moments
2. Pick the most interesting events (highest severity, diverse types)
3. Run multiple XAI methods at timesteps spanning each event window
4. Generate temporal plots showing how attributions shift during events
"""

import os
import sys
import json

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

VMAX_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "V-Max")
if VMAX_REPO not in sys.path:
    sys.path.insert(0, VMAX_REPO)

import numpy as np
import jax.numpy as jnp

# ── Config ──────────────────────────────────────────────────────
MODEL_DIR = "runs_rlc/womd_sac_road_perceiver_minimal_42"
DATA_PATH = "data/training.tfrecord"
N_SCENARIOS = 5            # mine this many scenarios
N_EVENTS_TO_ANALYZE = 3   # pick top N events for XAI
TIMESTEP_STRIDE = 3       # analyze every Nth step in event window
WINDOW_PADDING = 5        # extra steps before/after event
OUTPUT_DIR = "experiments/event_xai_results"
XAI_METHODS = ["vanilla_gradient", "integrated_gradients"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Step 1: Load model ─────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading model")
print("=" * 60)

import posthoc_xai as xai

model = xai.load_model(MODEL_DIR, data_path=DATA_PATH)
print(f"Model: {model.name}")
print(f"Observation structure: {model.observation_structure}")


# ── Step 2: Mine events ────────────────────────────────────────
print("\n" + "=" * 60)
print(f"STEP 2: Mining events from {N_SCENARIOS} scenarios")
print("=" * 60)

from event_mining import EventMiner, EventCatalog
from event_mining.integration.vmax_adapter import VMaxAdapter
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

adapter = VMaxAdapter(store_raw_obs=True)  # need obs for XAI
adapter.prepare(model)

miner = EventMiner()
catalog = EventCatalog()
scenario_data_map = {}  # scenario_id -> ScenarioData

data_iter = iter(data_gen)
for i in range(N_SCENARIOS):
    try:
        scenario = next(data_iter)
    except StopIteration:
        print(f"  Data exhausted after {i} scenarios")
        break

    sid = f"s{i:03d}"
    sd = adapter.extract_scenario_data(model, scenario, sid, rng_seed=i)
    scenario_data_map[sid] = sd

    events = miner.mine_scenario(sd)
    catalog.extend(events)

    n_ev = len(events)
    print(
        f"  [{i+1}/{N_SCENARIOS}] {sid}: {sd.total_steps} steps, "
        f"{n_ev} events, collision={sd.has_collision}, offroad={sd.has_offroad}"
    )

catalog.save(os.path.join(OUTPUT_DIR, "catalog.json"))
print(f"\nTotal events mined: {len(catalog)}")
print(f"Summary: {catalog.summary()}")


# ── Step 3: Select interesting events ──────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Selecting interesting events for XAI analysis")
print("=" * 60)

from event_mining.events.base import EventType

# Prioritize: collisions > offroad > evasive_steering > hard_brake > hazard_onset
PRIORITY = {
    EventType.COLLISION: 6,
    EventType.OFF_ROAD: 5,
    EventType.EVASIVE_STEERING: 4,
    EventType.HARD_BRAKE: 3,
    EventType.NEAR_MISS: 2,
    EventType.HAZARD_ONSET: 1,
}

# Sort by priority then severity
all_events = sorted(
    catalog.events,
    key=lambda e: (PRIORITY.get(e.event_type, 0), e.severity_score),
    reverse=True,
)

# Pick top N, preferring diverse scenarios
selected_events = []
seen_scenarios = set()
for e in all_events:
    # Skip near-miss events that span entire episode (noise)
    if e.event_type == EventType.NEAR_MISS and e.duration > 40:
        continue
    selected_events.append(e)
    seen_scenarios.add(e.scenario_id)
    if len(selected_events) >= N_EVENTS_TO_ANALYZE:
        break

print(f"Selected {len(selected_events)} events:")
for i, e in enumerate(selected_events):
    print(
        f"  {i+1}. {e.scenario_id} | {e.event_type.value:20s} "
        f"sev={e.severity.value:8s} t={e.onset}-{e.offset} "
        f"peak={e.peak} agent={e.causal_agent_id}"
    )


# ── Step 4: Run XAI at event timesteps ────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Running XAI methods on event windows")
print("=" * 60)

# Initialize XAI methods
methods = {}
for method_name in XAI_METHODS:
    methods[method_name] = xai.METHOD_REGISTRY[method_name](model)
    print(f"  Initialized: {method_name}")

results_all = []  # list of per-event result dicts

for event_idx, event in enumerate(selected_events):
    sid = event.scenario_id
    sd = scenario_data_map.get(sid)
    if sd is None or sd.raw_observations is None:
        print(f"  Skipping {sid} — no raw observations")
        continue

    # Determine timesteps to analyze
    win_start = max(0, event.onset - WINDOW_PADDING)
    win_end = min(sd.total_steps - 1, event.offset + WINDOW_PADDING)
    timesteps = list(range(win_start, win_end + 1, TIMESTEP_STRIDE))

    # Always include onset, peak, offset
    for key_t in [event.onset, event.peak, event.offset]:
        if key_t not in timesteps and 0 <= key_t < sd.total_steps:
            timesteps.append(key_t)
    timesteps = sorted(set(timesteps))

    print(
        f"\n  Event {event_idx+1}/{len(selected_events)}: "
        f"{sid} {event.event_type.value} (t={event.onset}-{event.offset})"
    )
    print(f"  Analyzing {len(timesteps)} timesteps: {timesteps}")

    event_results = {
        "event": event.to_dict(),
        "timesteps": timesteps,
        "methods": {},
    }

    for method_name, method in methods.items():
        print(f"    Running {method_name}...", end=" ", flush=True)
        category_series = {}   # {category: [importance_at_t0, ...]}
        entity_series = {}     # {entity: [importance_at_t0, ...]}
        timings = []

        for t in timesteps:
            obs = jnp.array(sd.raw_observations[t])
            attr = method(obs)

            # Collect category importance
            for cat, imp in attr.category_importance.items():
                category_series.setdefault(cat, []).append(float(imp))

            # Collect per-entity importance for other_agents
            if "other_agents" in attr.entity_importance:
                for ent, imp in attr.entity_importance["other_agents"].items():
                    entity_series.setdefault(ent, []).append(float(imp))

            timings.append(attr.computation_time_ms)

        avg_time = np.mean(timings)
        print(f"done ({avg_time:.0f}ms avg/step)")

        event_results["methods"][method_name] = {
            "category_series": category_series,
            "entity_series": entity_series,
            "avg_time_ms": float(avg_time),
        }

    results_all.append(event_results)

    # Save per-event results
    event_path = os.path.join(OUTPUT_DIR, f"event_{event_idx:02d}_{sid}.json")
    with open(event_path, "w") as f:
        json.dump(event_results, f, indent=2)


# ── Step 5: Generate temporal plots ───────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Generating temporal XAI plots")
print("=" * 60)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CATEGORY_COLORS = {
    "sdc_trajectory": "#4488FF",
    "other_agents": "#FF4444",
    "roadgraph": "#888888",
    "traffic_lights": "#FFCC00",
    "gps_path": "#00CC66",
}

for event_idx, result in enumerate(results_all):
    event = result["event"]
    timesteps = result["timesteps"]
    sid = event["scenario_id"]
    etype = event["event_type"]
    onset = event["onset"]
    peak = event["peak"]
    offset = event["offset"]

    print(f"\n  Plotting event {event_idx}: {sid} {etype}")

    # --- Plot 1: Category importance over time (one subplot per method) ---
    n_methods = len(result["methods"])
    fig, axes = plt.subplots(n_methods, 1, figsize=(12, 3.5 * n_methods), sharex=True)
    if n_methods == 1:
        axes = [axes]

    for ax, (method_name, mdata) in zip(axes, result["methods"].items()):
        cat_series = mdata["category_series"]
        for cat, values in cat_series.items():
            color = CATEGORY_COLORS.get(cat, "#AAAAAA")
            ax.plot(timesteps, values, "-o", label=cat, color=color, markersize=3, linewidth=1.5)

        # Mark event boundaries
        ax.axvline(onset, color="red", linestyle="--", alpha=0.7, label="onset")
        ax.axvline(peak, color="red", linestyle="-", alpha=0.9, linewidth=2, label="peak")
        if offset != onset:
            ax.axvline(offset, color="orange", linestyle="--", alpha=0.7, label="offset")

        # Shade event window
        ax.axvspan(onset, offset, alpha=0.1, color="red")

        ax.set_ylabel("Importance", fontsize=9)
        ax.set_title(f"{method_name}", fontsize=10)
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Timestep")
    fig.suptitle(
        f"Category Importance — {sid} | {etype} (t={onset}→{peak}→{offset})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"event_{event_idx:02d}_categories.png"), dpi=150)
    plt.close(fig)

    # --- Plot 2: Other agent importance over time (all methods, one plot) ---
    fig, axes = plt.subplots(n_methods, 1, figsize=(12, 3.5 * n_methods), sharex=True)
    if n_methods == 1:
        axes = [axes]

    for ax, (method_name, mdata) in zip(axes, result["methods"].items()):
        ent_series = mdata["entity_series"]
        if not ent_series:
            continue

        # Plot top 5 agents by peak importance
        agent_peaks = {a: max(vals) for a, vals in ent_series.items()}
        top_agents = sorted(agent_peaks, key=agent_peaks.get, reverse=True)[:5]

        causal = event.get("causal_agent_id")

        for agent in top_agents:
            values = ent_series[agent]
            lw = 2.5 if agent == f"agent_{causal}" else 1.2
            ls = "-" if agent == f"agent_{causal}" else "--"
            label = f"{agent} (CAUSAL)" if agent == f"agent_{causal}" else agent
            ax.plot(timesteps, values, ls, label=label, linewidth=lw, markersize=2)

        ax.axvline(onset, color="red", linestyle="--", alpha=0.7)
        ax.axvline(peak, color="red", linestyle="-", alpha=0.9, linewidth=2)
        ax.axvspan(onset, offset, alpha=0.1, color="red")

        ax.set_ylabel("Agent Importance", fontsize=9)
        ax.set_title(f"{method_name}", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Timestep")
    fig.suptitle(
        f"Agent Importance — {sid} | {etype} (causal=agent_{event.get('causal_agent_id')})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"event_{event_idx:02d}_agents.png"), dpi=150)
    plt.close(fig)

    # --- Plot 3: Stacked area chart (category composition over time) ---
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4))
    if n_methods == 1:
        axes = [axes]

    for ax, (method_name, mdata) in zip(axes, result["methods"].items()):
        cat_series = mdata["category_series"]
        cats = list(cat_series.keys())
        values = np.array([cat_series[c] for c in cats])  # (n_cats, n_timesteps)

        colors = [CATEGORY_COLORS.get(c, "#AAAAAA") for c in cats]
        ax.stackplot(timesteps, values, labels=cats, colors=colors, alpha=0.8)

        ax.axvline(onset, color="red", linestyle="--", alpha=0.8)
        ax.axvline(peak, color="red", linestyle="-", linewidth=2)
        ax.set_title(method_name, fontsize=9)
        ax.set_ylim(0, 1)

    axes[0].legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("Composition")
    fig.suptitle(f"Attribution Composition — {sid} | {etype}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"event_{event_idx:02d}_stacked.png"), dpi=150)
    plt.close(fig)

print(f"\nAll plots saved to {OUTPUT_DIR}/")


# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXPERIMENT COMPLETE")
print("=" * 60)

print(f"\nScenarios mined: {N_SCENARIOS}")
print(f"Total events found: {len(catalog)}")
print(f"Events analyzed with XAI: {len(selected_events)}")
print(f"XAI methods used: {XAI_METHODS}")
print(f"\nOutput files in {OUTPUT_DIR}/:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"  {f:45s} {size/1024:.1f} KB")
