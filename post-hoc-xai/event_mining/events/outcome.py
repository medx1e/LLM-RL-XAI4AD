"""Outcome-based event detectors: collision and off-road."""

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


class CollisionDetector(EventDetector):
    """Detect collision events using per-step collision flags.

    If a collision occurred, traces TTC backwards to find the onset of the
    dangerous situation that led to it.
    """

    def __init__(self, pre_collision_ttc_threshold: float = 5.0, padding: int = 15):
        super().__init__(
            pre_collision_ttc_threshold=pre_collision_ttc_threshold,
            padding=padding,
        )
        self.pre_collision_ttc_threshold = pre_collision_ttc_threshold
        self.padding = padding

    def detect(self, data: ScenarioData) -> List[Event]:
        if not data.has_collision or data.total_steps == 0:
            return []

        collision_time = data.collision_time
        if collision_time is None:
            collision_time = data.total_steps - 1

        # Find the onset by tracing TTC backwards from collision
        onset = collision_time
        if data.ttc is not None:
            min_ttc_per_step = np.min(data.ttc, axis=1)
            for t in range(collision_time - 1, -1, -1):
                if min_ttc_per_step[t] >= self.pre_collision_ttc_threshold:
                    onset = t + 1
                    break
            else:
                onset = 0

        # Find the causal agent (nearest at collision time)
        causal_agent = None
        if data.min_distance is not None:
            valid_at_coll = data.other_agents_valid[collision_time]
            dists_at_coll = data.min_distance[collision_time].copy()
            dists_at_coll[~valid_at_coll] = np.inf
            if np.any(valid_at_coll):
                causal_agent = int(np.argmin(dists_at_coll))

        window = self._compute_window(onset, collision_time, data.total_steps, self.padding)

        return [Event(
            event_type=EventType.COLLISION,
            severity=Severity.CRITICAL,
            onset=onset,
            peak=collision_time,
            offset=collision_time,
            window=window,
            scenario_id=data.scenario_id,
            causal_agent_id=causal_agent,
            metadata={
                "collision_timestep": collision_time,
                "onset_timestep": onset,
                "pre_collision_window": collision_time - onset,
            },
        )]


class OffRoadDetector(EventDetector):
    """Detect off-road events using per-step offroad flags."""

    def __init__(self, min_duration: int = 1, padding: int = 10):
        super().__init__(min_duration=min_duration, padding=padding)
        self.min_duration = min_duration
        self.padding = padding

    def detect(self, data: ScenarioData) -> List[Event]:
        if not data.has_offroad or data.total_steps == 0:
            return []

        if data.step_offroad is not None:
            condition = data.step_offroad.astype(bool)
        else:
            # Fallback: single event at offroad_time
            t = data.offroad_time if data.offroad_time is not None else data.total_steps - 1
            window = self._compute_window(t, t, data.total_steps, self.padding)
            return [Event(
                event_type=EventType.OFF_ROAD,
                severity=Severity.HIGH,
                onset=t,
                peak=t,
                offset=t,
                window=window,
                scenario_id=data.scenario_id,
                metadata={"offroad_timestep": t},
            )]

        windows = self._find_continuous_windows(condition, self.min_duration)
        events = []

        for onset, offset in windows:
            duration = offset - onset + 1
            if duration >= 10:
                severity = Severity.CRITICAL
            elif duration >= 5:
                severity = Severity.HIGH
            else:
                severity = Severity.MEDIUM

            window = self._compute_window(onset, offset, data.total_steps, self.padding)

            events.append(Event(
                event_type=EventType.OFF_ROAD,
                severity=severity,
                onset=onset,
                peak=onset,  # offroad onset is the most interesting point
                offset=offset,
                window=window,
                scenario_id=data.scenario_id,
                metadata={
                    "duration_steps": duration,
                    "offroad_start": onset,
                    "offroad_end": offset,
                },
            ))

        return events
