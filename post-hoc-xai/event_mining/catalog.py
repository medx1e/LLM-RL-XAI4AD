"""EventCatalog: queryable, serializable collection of detected events."""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Optional

from event_mining.events.base import Event, EventType, Severity


class EventCatalog:
    """Queryable collection of mining events with serialization support."""

    def __init__(self, events: list[Event] | None = None):
        self._events: list[Event] = list(events) if events else []

    def add(self, event: Event) -> None:
        """Add a single event to the catalog."""
        self._events.append(event)

    def extend(self, events: list[Event]) -> None:
        """Add multiple events."""
        self._events.extend(events)

    @property
    def events(self) -> list[Event]:
        return self._events

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self):
        return iter(self._events)

    def __getitem__(self, idx):
        return self._events[idx]

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter(
        self,
        event_type: EventType | str | None = None,
        min_severity_score: float | None = None,
        scenario_id: str | None = None,
        causal_agent_id: int | None = None,
    ) -> list[Event]:
        """Filter events by criteria.

        Args:
            event_type: Filter by event type (name string or EventType).
            min_severity_score: Only events with severity >= this score.
            scenario_id: Only events from this scenario.
            causal_agent_id: Only events involving this agent.

        Returns:
            Filtered list of events.
        """
        result = self._events

        if event_type is not None:
            if isinstance(event_type, str):
                event_type = EventType(event_type)
            result = [e for e in result if e.event_type == event_type]

        if min_severity_score is not None:
            result = [e for e in result if e.severity_score >= min_severity_score]

        if scenario_id is not None:
            result = [e for e in result if e.scenario_id == scenario_id]

        if causal_agent_id is not None:
            result = [e for e in result if e.causal_agent_id == causal_agent_id]

        return result

    def by_scenario(self) -> dict[str, list[Event]]:
        """Group events by scenario_id."""
        groups: dict[str, list[Event]] = {}
        for e in self._events:
            groups.setdefault(e.scenario_id, []).append(e)
        return groups

    def by_type(self) -> dict[EventType, list[Event]]:
        """Group events by event type."""
        groups: dict[EventType, list[Event]] = {}
        for e in self._events:
            groups.setdefault(e.event_type, []).append(e)
        return groups

    def get_windows(self) -> list[tuple[str, int, int, Event]]:
        """Get all analysis windows as (scenario_id, start, end, event) tuples."""
        return [
            (e.scenario_id, e.window[0], e.window[1], e)
            for e in self._events
        ]

    def get_analysis_points(self) -> list[tuple[str, int, Event]]:
        """Get all peak timesteps as (scenario_id, timestep, event) tuples."""
        return [
            (e.scenario_id, e.peak, e)
            for e in self._events
        ]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return summary statistics."""
        type_counts = Counter(e.event_type.value for e in self._events)
        severity_counts = Counter(e.severity.value for e in self._events)
        scenario_ids = set(e.scenario_id for e in self._events)

        return {
            "total_events": len(self._events),
            "unique_scenarios": len(scenario_ids),
            "by_type": dict(type_counts),
            "by_severity": dict(severity_counts),
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save catalog to JSON file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "events": [e.to_dict() for e in self._events],
            "summary": self.summary(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> EventCatalog:
        """Load catalog from JSON file."""
        with open(path) as f:
            data = json.load(f)
        events = [Event.from_dict(d) for d in data["events"]]
        return cls(events)
