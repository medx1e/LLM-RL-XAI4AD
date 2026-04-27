"""EventMiner: orchestrator that runs detectors on scenario data."""

from __future__ import annotations

from typing import Callable, Optional

from event_mining.catalog import EventCatalog
from event_mining.events.base import Event, EventDetector, ScenarioData
from event_mining.events import ALL_DETECTORS


class EventMiner:
    """Orchestrates event detection across scenarios.

    Holds a list of EventDetector instances and runs them on ScenarioData.
    Can also drive the full pipeline: load scenarios, run episodes, detect events.
    """

    def __init__(self, detectors: list[EventDetector] | None = None):
        """
        Args:
            detectors: List of detector instances. If None, uses all defaults.
        """
        if detectors is None:
            detectors = [cls() for cls in ALL_DETECTORS]
        self.detectors = detectors

    def mine_scenario(self, data: ScenarioData) -> list[Event]:
        """Run all detectors on a single scenario's data.

        Args:
            data: Extracted scenario data.

        Returns:
            List of all detected events.
        """
        events = []
        for detector in self.detectors:
            events.extend(detector.detect(data))
        return events

    def mine_from_model(
        self,
        model,
        n_scenarios: int = 50,
        save_path: str | None = None,
        progress_fn: Optional[Callable[[int, int, str], None]] = None,
        store_raw_obs: bool = True,
    ) -> EventCatalog:
        """Full mining pipeline: iterate scenarios, extract data, detect events.

        Args:
            model: An ExplainableModel (from ``posthoc_xai.load_model()``).
            n_scenarios: Number of scenarios to process.
            save_path: Optional path to save the resulting catalog as JSON.
            progress_fn: Optional callback ``(current, total, scenario_id)``.
            store_raw_obs: Whether to store raw observations in ScenarioData.

        Returns:
            EventCatalog with all detected events.
        """
        from vmax.simulator import make_data_generator
        from event_mining.integration.vmax_adapter import VMaxAdapter

        loaded = model._loaded

        # Create a fresh data generator for reproducibility
        data_gen = make_data_generator(
            path=str(loaded.config.get("data_path", "data/training.tfrecord")),
            max_num_objects=loaded.config.get("max_num_objects", 64),
            include_sdc_paths=True,
            batch_dims=(1,),
            seed=42,
            repeat=1,
        )

        adapter = VMaxAdapter(store_raw_obs=store_raw_obs)
        catalog = EventCatalog()
        data_iter = iter(data_gen)

        for i in range(n_scenarios):
            try:
                scenario = next(data_iter)
            except StopIteration:
                print(f"  Data exhausted after {i} scenarios")
                break

            scenario_id = f"s{i:03d}"

            # Extract per-step data
            scenario_data = adapter.extract_scenario_data(
                model, scenario, scenario_id, rng_seed=i
            )

            # Run all detectors
            events = self.mine_scenario(scenario_data)

            # Set scenario_id on events (in case detectors didn't)
            for event in events:
                if not event.scenario_id:
                    event.scenario_id = scenario_id

            catalog.extend(events)

            if progress_fn:
                progress_fn(i + 1, n_scenarios, scenario_id)
            else:
                n_events = len(events)
                print(
                    f"  [{i + 1}/{n_scenarios}] {scenario_id}: "
                    f"{scenario_data.total_steps} steps, "
                    f"{n_events} events, "
                    f"collision={scenario_data.has_collision}, "
                    f"offroad={scenario_data.has_offroad}"
                )

        if save_path:
            catalog.save(save_path)
            print(f"  Catalog saved to {save_path}")

        return catalog


def mine_events(
    model_or_dir,
    n_scenarios: int = 50,
    data_path: str | None = None,
    save_path: str | None = None,
    detectors: list[EventDetector] | None = None,
) -> EventCatalog:
    """Convenience function: load model if needed and mine events.

    Args:
        model_or_dir: An ExplainableModel or path to a model directory.
        n_scenarios: Number of scenarios to process.
        data_path: Path to .tfrecord (only used if model_or_dir is a path).
        save_path: Optional path to save catalog JSON.
        detectors: Optional list of detector instances.

    Returns:
        EventCatalog with all detected events.
    """
    from pathlib import Path

    if isinstance(model_or_dir, (str, Path)):
        import posthoc_xai as xai
        if data_path is None:
            data_path = "data/training.tfrecord"
        model = xai.load_model(model_or_dir, data_path=data_path)
    else:
        model = model_or_dir

    miner = EventMiner(detectors=detectors)
    return miner.mine_from_model(model, n_scenarios=n_scenarios, save_path=save_path)
