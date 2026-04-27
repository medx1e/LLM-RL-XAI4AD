"""Safety-related event detectors: hazard onset and near-miss."""

from __future__ import annotations

from typing import List

import numpy as np

from event_mining.events.base import (
    Event,
    EventDetector,
    EventType,
    ScenarioData,
    Severity,
)


class HazardOnsetDetector(EventDetector):
    """Detect when TTC drops below a threshold for sustained duration.

    A hazard onset indicates the ego vehicle is on a collision course with
    another agent. The severity depends on how low the TTC gets.
    """

    def __init__(
        self,
        ttc_threshold: float = 3.0,
        min_duration: int = 3,
        critical_ttc: float = 1.0,
        padding: int = 10,
    ):
        super().__init__(
            ttc_threshold=ttc_threshold,
            min_duration=min_duration,
            critical_ttc=critical_ttc,
            padding=padding,
        )
        self.ttc_threshold = ttc_threshold
        self.min_duration = min_duration
        self.critical_ttc = critical_ttc
        self.padding = padding

    def detect(self, data: ScenarioData) -> List[Event]:
        if data.ttc is None or data.total_steps == 0:
            return []

        events = []
        n_agents = data.ttc.shape[1]

        for agent_idx in range(n_agents):
            ttc_col = data.ttc[:, agent_idx]
            valid_col = data.other_agents_valid[:, agent_idx]

            # Condition: TTC < threshold AND agent is valid
            condition = (ttc_col < self.ttc_threshold) & valid_col
            windows = self._find_continuous_windows(condition, self.min_duration)

            for onset, offset in windows:
                min_ttc = float(np.min(ttc_col[onset : offset + 1]))
                peak = self._find_event_peak(ttc_col, onset, offset, mode="min")

                if min_ttc < self.critical_ttc:
                    severity = Severity.CRITICAL
                elif min_ttc < self.ttc_threshold * 0.5:
                    severity = Severity.HIGH
                elif min_ttc < self.ttc_threshold * 0.75:
                    severity = Severity.MEDIUM
                else:
                    severity = Severity.LOW

                window = self._compute_window(onset, offset, data.total_steps, self.padding)

                events.append(Event(
                    event_type=EventType.HAZARD_ONSET,
                    severity=severity,
                    onset=onset,
                    peak=peak,
                    offset=offset,
                    window=window,
                    scenario_id=data.scenario_id,
                    causal_agent_id=agent_idx,
                    metadata={
                        "min_ttc": min_ttc,
                        "ttc_threshold": self.ttc_threshold,
                    },
                ))

        return events


class NearMissDetector(EventDetector):
    """Detect near-miss events: minimum distance drops very low without collision.

    A near-miss is when an agent comes dangerously close to ego but no
    collision actually occurs.
    """

    def __init__(
        self,
        distance_threshold: float = 3.0,
        min_duration: int = 1,
        padding: int = 10,
    ):
        super().__init__(
            distance_threshold=distance_threshold,
            min_duration=min_duration,
            padding=padding,
        )
        self.distance_threshold = distance_threshold
        self.min_duration = min_duration
        self.padding = padding

    def detect(self, data: ScenarioData) -> List[Event]:
        if data.min_distance is None or data.total_steps == 0:
            return []

        # Skip if there was a collision — that's a different event type
        if data.has_collision:
            return []

        events = []
        n_agents = data.min_distance.shape[1]

        for agent_idx in range(n_agents):
            dist_col = data.min_distance[:, agent_idx]
            valid_col = data.other_agents_valid[:, agent_idx]

            condition = (dist_col < self.distance_threshold) & valid_col
            windows = self._find_continuous_windows(condition, self.min_duration)

            for onset, offset in windows:
                min_dist = float(np.min(dist_col[onset : offset + 1]))
                peak = self._find_event_peak(dist_col, onset, offset, mode="min")

                if min_dist < 1.0:
                    severity = Severity.CRITICAL
                elif min_dist < 1.5:
                    severity = Severity.HIGH
                elif min_dist < 2.0:
                    severity = Severity.MEDIUM
                else:
                    severity = Severity.LOW

                window = self._compute_window(onset, offset, data.total_steps, self.padding)

                events.append(Event(
                    event_type=EventType.NEAR_MISS,
                    severity=severity,
                    onset=onset,
                    peak=peak,
                    offset=offset,
                    window=window,
                    scenario_id=data.scenario_id,
                    causal_agent_id=agent_idx,
                    metadata={
                        "min_distance": min_dist,
                        "distance_threshold": self.distance_threshold,
                    },
                ))

        return events
