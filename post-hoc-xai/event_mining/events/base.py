"""Core data structures for event mining."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

import numpy as np


class EventType(Enum):
    """Types of detectable driving events."""

    HAZARD_ONSET = "hazard_onset"
    COLLISION_IMMINENT = "collision_imminent"
    HARD_BRAKE = "hard_brake"
    EVASIVE_STEERING = "evasive_steering"
    NEAR_MISS = "near_miss"
    COLLISION = "collision"
    OFF_ROAD = "off_road"


class Severity(Enum):
    """Event severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def score(self) -> float:
        return {
            Severity.LOW: 0.25,
            Severity.MEDIUM: 0.5,
            Severity.HIGH: 0.75,
            Severity.CRITICAL: 1.0,
        }[self]


@dataclass
class Event:
    """A detected driving event with timing and context.

    Attributes:
        event_type: What kind of event was detected.
        severity: How severe the event is.
        onset: Timestep where the event begins.
        peak: Timestep of maximum intensity.
        offset: Timestep where the event ends.
        window: (start, end) analysis window around the event (may include padding).
        scenario_id: Identifier for the scenario this event belongs to.
        causal_agent_id: Index of the agent that caused / is involved in the event.
        metadata: Arbitrary extra information (thresholds, metric values, etc.).
    """

    event_type: EventType
    severity: Severity
    onset: int
    peak: int
    offset: int
    window: tuple[int, int]
    scenario_id: str = ""
    causal_agent_id: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> int:
        return self.offset - self.onset + 1

    @property
    def severity_score(self) -> float:
        return self.severity.score

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "onset": self.onset,
            "peak": self.peak,
            "offset": self.offset,
            "window": list(self.window),
            "scenario_id": self.scenario_id,
            "causal_agent_id": self.causal_agent_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Event:
        return cls(
            event_type=EventType(d["event_type"]),
            severity=Severity(d["severity"]),
            onset=d["onset"],
            peak=d["peak"],
            offset=d["offset"],
            window=tuple(d["window"]),
            scenario_id=d.get("scenario_id", ""),
            causal_agent_id=d.get("causal_agent_id"),
            metadata=d.get("metadata", {}),
        )


@dataclass
class ScenarioData:
    """All per-timestep data extracted from a V-Max episode rollout.

    All trajectory arrays have shape (T,) or (T, N_agents) where T is total_steps.
    Coordinates are ego-relative as provided by the V-Max observation unflattener.
    """

    scenario_id: str
    total_steps: int

    # Ego trajectory: (T,) arrays
    ego_x: np.ndarray  # ego x position (always 0 in ego-relative, but stored for world-frame use)
    ego_y: np.ndarray
    ego_vx: np.ndarray
    ego_vy: np.ndarray
    ego_yaw: np.ndarray
    ego_length: np.ndarray
    ego_width: np.ndarray

    # Ego actions: (T,) arrays — from policy output
    ego_accel: np.ndarray  # acceleration action
    ego_steering: np.ndarray  # steering action

    # Other agents: (T, N_agents, 7) — [x, y, vx, vy, yaw, length, width]
    other_agents: np.ndarray
    other_agents_valid: np.ndarray  # (T, N_agents) bool mask

    # Road graph: (N_rg_points, features) — static, same every step
    road_graph: Optional[np.ndarray] = None
    road_graph_valid: Optional[np.ndarray] = None

    # Traffic lights: (T, N_tl, features)
    traffic_lights: Optional[np.ndarray] = None
    traffic_lights_valid: Optional[np.ndarray] = None

    # GPS path: (N_gps, 2) — static route
    gps_path: Optional[np.ndarray] = None

    # Derived safety metrics: computed by metrics.py
    ttc: Optional[np.ndarray] = None  # (T, N_agents) time-to-collision
    min_distance: Optional[np.ndarray] = None  # (T, N_agents) distance to each agent
    nearest_agent_id: Optional[np.ndarray] = None  # (T,) index of nearest agent
    criticality: Optional[np.ndarray] = None  # (T,) composite criticality score

    # Per-step outcome flags (from env metrics)
    step_collision: Optional[np.ndarray] = None  # (T,) bool
    step_offroad: Optional[np.ndarray] = None  # (T,) bool

    # Episode-level outcomes
    has_collision: bool = False
    collision_time: Optional[int] = None
    has_offroad: bool = False
    offroad_time: Optional[int] = None
    route_completion: float = 0.0

    # Raw flat observations for XAI methods: (T, obs_dim)
    raw_observations: Optional[np.ndarray] = None


class EventDetector(ABC):
    """Abstract base class for event detectors."""

    def __init__(self, **kwargs):
        self.params = kwargs

    @abstractmethod
    def detect(self, data: ScenarioData) -> List[Event]:
        """Detect events in a scenario.

        Args:
            data: Extracted scenario data with all trajectories and metrics.

        Returns:
            List of detected events.
        """

    def _compute_window(
        self, onset: int, offset: int, total_steps: int, padding: int = 10
    ) -> tuple[int, int]:
        """Compute analysis window around an event with padding."""
        return (max(0, onset - padding), min(total_steps - 1, offset + padding))

    def _find_event_peak(
        self, signal: np.ndarray, onset: int, offset: int, mode: str = "max"
    ) -> int:
        """Find the peak timestep within an event window.

        Args:
            signal: 1D array of the metric being tracked.
            onset: Start of event.
            offset: End of event.
            mode: 'max' for highest value, 'min' for lowest value.
        """
        segment = signal[onset : offset + 1]
        if mode == "max":
            return onset + int(np.argmax(segment))
        return onset + int(np.argmin(segment))

    @staticmethod
    def _find_continuous_windows(
        condition: np.ndarray, min_duration: int = 1
    ) -> List[tuple[int, int]]:
        """Find continuous windows where a boolean condition holds.

        Args:
            condition: 1D boolean array.
            min_duration: Minimum number of consecutive True steps.

        Returns:
            List of (onset, offset) tuples.
        """
        if len(condition) == 0:
            return []

        windows = []
        in_window = False
        start = 0

        for i, val in enumerate(condition):
            if val and not in_window:
                start = i
                in_window = True
            elif not val and in_window:
                if i - start >= min_duration:
                    windows.append((start, i - 1))
                in_window = False

        # Handle window extending to end
        if in_window and len(condition) - start >= min_duration:
            windows.append((start, len(condition) - 1))

        return windows
