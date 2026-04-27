"""Event detection classes."""

from event_mining.events.base import (
    Event,
    EventType,
    Severity,
    ScenarioData,
    EventDetector,
)
from event_mining.events.safety import HazardOnsetDetector, NearMissDetector
from event_mining.events.action import HardBrakeDetector, EvasiveSteeringDetector
from event_mining.events.outcome import CollisionDetector, OffRoadDetector

ALL_DETECTORS = [
    HazardOnsetDetector,
    NearMissDetector,
    HardBrakeDetector,
    EvasiveSteeringDetector,
    CollisionDetector,
    OffRoadDetector,
]

__all__ = [
    "Event",
    "EventType",
    "Severity",
    "ScenarioData",
    "EventDetector",
    "HazardOnsetDetector",
    "NearMissDetector",
    "HardBrakeDetector",
    "EvasiveSteeringDetector",
    "CollisionDetector",
    "OffRoadDetector",
    "ALL_DETECTORS",
]
