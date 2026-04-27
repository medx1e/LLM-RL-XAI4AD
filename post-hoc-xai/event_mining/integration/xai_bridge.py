"""Bridge between event mining results and the posthoc_xai framework.

Provides iterators and converters to feed mined events into XAI analysis.
"""

from __future__ import annotations

from typing import Any, Iterator

from event_mining.catalog import EventCatalog
from event_mining.events.base import Event, EventType


class XAIBridge:
    """Connects EventCatalog to the posthoc_xai experiment pipeline.

    Provides convenient iterators over events/timesteps for XAI analysis.
    """

    def __init__(self, catalog: EventCatalog):
        self.catalog = catalog

    def iter_analysis_windows(
        self,
        event_type: EventType | str | None = None,
        min_severity_score: float | None = None,
    ) -> Iterator[tuple[str, int, int, Event]]:
        """Yield (scenario_id, window_start, window_end, event) tuples.

        Useful for running XAI methods over entire event windows.
        """
        events = self.catalog.filter(
            event_type=event_type,
            min_severity_score=min_severity_score,
        )
        for event in events:
            yield event.scenario_id, event.window[0], event.window[1], event

    def iter_event_timesteps(
        self,
        event_type: EventType | str | None = None,
        min_severity_score: float | None = None,
        include_onset: bool = True,
        include_peak: bool = True,
        include_offset: bool = False,
    ) -> Iterator[tuple[str, int, dict[str, Any]]]:
        """Yield (scenario_id, timestep, metadata) for key event timesteps.

        By default yields onset and peak timesteps. These are the most
        interesting points for XAI analysis.

        Args:
            event_type: Filter by event type.
            min_severity_score: Minimum severity to include.
            include_onset: Yield onset timesteps.
            include_peak: Yield peak timesteps.
            include_offset: Yield offset timesteps.

        Yields:
            (scenario_id, timestep, metadata) where metadata includes
            event_type, severity, role ('onset'/'peak'/'offset'), and
            causal_agent_id.
        """
        events = self.catalog.filter(
            event_type=event_type,
            min_severity_score=min_severity_score,
        )

        seen = set()  # avoid duplicates when onset == peak

        for event in events:
            base_meta = {
                "event_type": event.event_type.value,
                "severity": event.severity.value,
                "severity_score": event.severity_score,
                "causal_agent_id": event.causal_agent_id,
                "event_onset": event.onset,
                "event_offset": event.offset,
            }

            timesteps = []
            if include_onset:
                timesteps.append((event.onset, "onset"))
            if include_peak:
                timesteps.append((event.peak, "peak"))
            if include_offset:
                timesteps.append((event.offset, "offset"))

            for ts, role in timesteps:
                key = (event.scenario_id, ts, event.event_type.value, role)
                if key in seen:
                    continue
                seen.add(key)

                meta = {**base_meta, "role": role}
                yield event.scenario_id, ts, meta

    def get_scenario_event_timeline(
        self, scenario_id: str
    ) -> list[dict[str, Any]]:
        """Get a chronological timeline of events for a scenario.

        Returns:
            List of dicts sorted by onset, each with event details.
        """
        events = self.catalog.filter(scenario_id=scenario_id)
        events.sort(key=lambda e: e.onset)

        return [
            {
                "onset": e.onset,
                "peak": e.peak,
                "offset": e.offset,
                "type": e.event_type.value,
                "severity": e.severity.value,
                "severity_score": e.severity_score,
                "causal_agent_id": e.causal_agent_id,
                "window": e.window,
                "metadata": e.metadata,
            }
            for e in events
        ]

    def to_dataframe(self):
        """Export all events as a pandas DataFrame.

        Returns:
            DataFrame with one row per event.
        """
        import pandas as pd

        rows = []
        for event in self.catalog:
            row = {
                "scenario_id": event.scenario_id,
                "event_type": event.event_type.value,
                "severity": event.severity.value,
                "severity_score": event.severity_score,
                "onset": event.onset,
                "peak": event.peak,
                "offset": event.offset,
                "duration": event.duration,
                "window_start": event.window[0],
                "window_end": event.window[1],
                "causal_agent_id": event.causal_agent_id,
            }
            # Flatten metadata
            for k, v in event.metadata.items():
                row[f"meta_{k}"] = v
            rows.append(row)

        return pd.DataFrame(rows)
