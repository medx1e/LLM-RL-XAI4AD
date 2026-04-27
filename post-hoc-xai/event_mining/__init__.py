"""Event Mining for Autonomous Driving Scenarios.

Detects critical driving events (hazards, near-misses, collisions, hard brakes)
at frame-level granularity from V-Max model rollouts. Provides analysis windows
for targeted XAI investigation and BEV video visualization.

Quick start::

    from event_mining import EventMiner, mine_events

    # Mine events from a model
    catalog = mine_events(
        "runs_rlc/womd_sac_road_perceiver_minimal_42",
        n_scenarios=50,
        save_path="events/catalog.json",
    )
    print(catalog.summary())

    # Or use with an already-loaded model
    import posthoc_xai as xai
    model = xai.load_model("runs_rlc/womd_sac_road_perceiver_minimal_42",
                           data_path="data/training.tfrecord")
    catalog = mine_events(model, n_scenarios=50)
"""

from event_mining.events.base import Event, EventType, Severity, ScenarioData
from event_mining.catalog import EventCatalog
from event_mining.miner import EventMiner, mine_events

__all__ = [
    "Event",
    "EventType",
    "Severity",
    "ScenarioData",
    "EventCatalog",
    "EventMiner",
    "mine_events",
]
