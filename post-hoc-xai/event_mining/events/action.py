"""Action-based event detectors: hard braking and evasive steering."""

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


class HardBrakeDetector(EventDetector):
    """Detect hard braking events (large negative acceleration).

    Hard braking indicates the ego vehicle is decelerating aggressively,
    often in response to a hazard.
    """

    def __init__(
        self,
        accel_threshold: float = -3.0,
        min_duration: int = 2,
        critical_accel: float = -6.0,
        padding: int = 10,
    ):
        super().__init__(
            accel_threshold=accel_threshold,
            min_duration=min_duration,
            critical_accel=critical_accel,
            padding=padding,
        )
        self.accel_threshold = accel_threshold
        self.min_duration = min_duration
        self.critical_accel = critical_accel
        self.padding = padding

    def detect(self, data: ScenarioData) -> List[Event]:
        if data.total_steps == 0:
            return []

        accel = data.ego_accel
        condition = accel < self.accel_threshold
        windows = self._find_continuous_windows(condition, self.min_duration)

        events = []
        for onset, offset in windows:
            min_accel = float(np.min(accel[onset : offset + 1]))
            peak = self._find_event_peak(accel, onset, offset, mode="min")

            if min_accel < self.critical_accel:
                severity = Severity.HIGH
            elif min_accel < self.accel_threshold * 1.5:
                severity = Severity.MEDIUM
            else:
                severity = Severity.LOW

            window = self._compute_window(onset, offset, data.total_steps, self.padding)

            # Find nearest agent at peak as potential cause
            causal_agent = None
            if data.nearest_agent_id is not None:
                causal_agent = int(data.nearest_agent_id[peak])

            events.append(Event(
                event_type=EventType.HARD_BRAKE,
                severity=severity,
                onset=onset,
                peak=peak,
                offset=offset,
                window=window,
                scenario_id=data.scenario_id,
                causal_agent_id=causal_agent,
                metadata={
                    "min_acceleration": min_accel,
                    "accel_threshold": self.accel_threshold,
                },
            ))

        return events


class EvasiveSteeringDetector(EventDetector):
    """Detect evasive steering events (large steering angle magnitude).

    Evasive steering indicates the ego is making a sharp turn, potentially
    to avoid a hazard.
    """

    def __init__(
        self,
        steering_threshold: float = 0.3,
        min_duration: int = 2,
        critical_steering: float = 0.6,
        padding: int = 10,
    ):
        super().__init__(
            steering_threshold=steering_threshold,
            min_duration=min_duration,
            critical_steering=critical_steering,
            padding=padding,
        )
        self.steering_threshold = steering_threshold
        self.min_duration = min_duration
        self.critical_steering = critical_steering
        self.padding = padding

    def detect(self, data: ScenarioData) -> List[Event]:
        if data.total_steps == 0:
            return []

        abs_steering = np.abs(data.ego_steering)
        condition = abs_steering > self.steering_threshold
        windows = self._find_continuous_windows(condition, self.min_duration)

        events = []
        for onset, offset in windows:
            max_steer = float(np.max(abs_steering[onset : offset + 1]))
            peak = self._find_event_peak(abs_steering, onset, offset, mode="max")

            if max_steer > self.critical_steering:
                severity = Severity.HIGH
            elif max_steer > self.steering_threshold * 1.5:
                severity = Severity.MEDIUM
            else:
                severity = Severity.LOW

            window = self._compute_window(onset, offset, data.total_steps, self.padding)

            causal_agent = None
            if data.nearest_agent_id is not None:
                causal_agent = int(data.nearest_agent_id[peak])

            events.append(Event(
                event_type=EventType.EVASIVE_STEERING,
                severity=severity,
                onset=onset,
                peak=peak,
                offset=offset,
                window=window,
                scenario_id=data.scenario_id,
                causal_agent_id=causal_agent,
                metadata={
                    "max_steering": max_steer,
                    "steering_threshold": self.steering_threshold,
                },
            ))

        return events
