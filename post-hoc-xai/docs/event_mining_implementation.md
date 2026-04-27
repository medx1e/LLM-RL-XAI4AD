# Event Mining Module — Implementation Plan

## For Critical Scenario Discovery in Autonomous Driving XAI

---

# 1. Overview

## 1.1 Purpose

Build a modular event mining system that:
1. Scans driving scenarios to detect **critical events** (hazards, hard brakes, collisions, etc.)
2. Extracts **time windows** around each event for focused analysis
3. Produces a **queryable catalog** of (scenario_id, timestep, event_type) tuples
4. Integrates seamlessly with the existing XAI framework

## 1.2 Core Insight

> "Don't filter scenarios. Filter windows. Interpretability in driving RL must be event-conditioned because nominal states dominate the distribution."

Most timesteps are boring (highway cruising, TTC > 10s). Event mining finds the 1% of timesteps worth explaining.

## 1.3 Design Principles

- **Modular:** Each event detector is a separate class
- **Extensible:** Easy to add new event types
- **Efficient:** Single pass through scenario data extracts all events
- **Queryable:** Filter events by type, severity, scenario, etc.
- **Integration-ready:** Output feeds directly into XAI analysis pipeline

---

# 2. Project Structure

```
event_mining/
├── __init__.py
├── events/
│   ├── __init__.py
│   ├── base.py                 # Event and EventDetector base classes
│   ├── safety.py               # Hazard, collision, near-miss detectors
│   ├── action.py               # Hard brake, evasive steering detectors
│   ├── traffic.py              # Traffic light interaction detectors
│   └── outcome.py              # Collision, off-road, violation detectors
│
├── catalog.py                  # EventCatalog class
├── miner.py                    # Main EventMiner orchestrator
├── metrics.py                  # TTC, criticality, distance calculations
├── windows.py                  # Window extraction utilities
│
├── integration/
│   ├── __init__.py
│   ├── vmax_adapter.py         # Extract data from V-Max scenarios
│   └── xai_bridge.py           # Feed events to XAI framework
│
└── cli.py                      # Command-line interface
```

---

# 3. Core Data Structures

## 3.1 Event Class

```python
# event_mining/events/base.py

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum


class EventType(Enum):
    """All supported event types."""
    # Safety-critical (Tier 1)
    HAZARD_ONSET = "hazard_onset"
    COLLISION_IMMINENT = "collision_imminent"
    HARD_BRAKE = "hard_brake"
    EVASIVE_STEERING = "evasive_steering"
    NEAR_MISS = "near_miss"
    
    # Decision events (Tier 2)
    TRAFFIC_LIGHT_APPROACH = "traffic_light_approach"
    TRAFFIC_LIGHT_CHANGE = "traffic_light_change"
    YIELD_DECISION = "yield_decision"
    LANE_CHANGE = "lane_change"
    
    # Outcome events (Tier 3)
    COLLISION = "collision"
    OFF_ROAD = "off_road"
    RED_LIGHT_VIOLATION = "red_light_violation"
    ROUTE_DEVIATION = "route_deviation"


class Severity(Enum):
    """Event severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Event:
    """
    A single detected event in a driving scenario.
    
    This is the core data structure that the XAI framework consumes.
    """
    # Identification
    scenario_id: str
    event_type: EventType
    
    # Timing
    event_time: int              # Timestep when event occurs (peak)
    onset_time: int              # When event condition first became true
    offset_time: int             # When event condition ended
    
    # Window (for analysis)
    window_start: int            # Suggested analysis start (onset - buffer)
    window_end: int              # Suggested analysis end (offset + buffer)
    
    # Severity
    severity: Severity
    severity_score: float        # Continuous 0-1 score
    
    # Event-specific metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Optional: related agent
    causal_agent_id: Optional[int] = None
    
    def __post_init__(self):
        """Validate event data."""
        assert self.window_start <= self.onset_time <= self.event_time
        assert self.event_time <= self.offset_time <= self.window_end
    
    @property
    def window_length(self) -> int:
        return self.window_end - self.window_start + 1
    
    @property
    def duration(self) -> int:
        """Event duration (onset to offset)."""
        return self.offset_time - self.onset_time + 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON storage."""
        return {
            'scenario_id': self.scenario_id,
            'event_type': self.event_type.value,
            'event_time': self.event_time,
            'onset_time': self.onset_time,
            'offset_time': self.offset_time,
            'window_start': self.window_start,
            'window_end': self.window_end,
            'severity': self.severity.value,
            'severity_score': self.severity_score,
            'metadata': self.metadata,
            'causal_agent_id': self.causal_agent_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Deserialize from JSON."""
        return cls(
            scenario_id=data['scenario_id'],
            event_type=EventType(data['event_type']),
            event_time=data['event_time'],
            onset_time=data['onset_time'],
            offset_time=data['offset_time'],
            window_start=data['window_start'],
            window_end=data['window_end'],
            severity=Severity(data['severity']),
            severity_score=data['severity_score'],
            metadata=data.get('metadata', {}),
            causal_agent_id=data.get('causal_agent_id'),
        )
```

## 3.2 Event Detector Base Class

```python
# event_mining/events/base.py (continued)

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Generator


@dataclass
class ScenarioData:
    """
    Pre-computed data for a scenario, used by all detectors.
    
    Computed once per scenario, shared across detectors for efficiency.
    """
    scenario_id: str
    num_timesteps: int
    
    # Ego vehicle state (T,)
    ego_x: np.ndarray
    ego_y: np.ndarray
    ego_vx: np.ndarray
    ego_vy: np.ndarray
    ego_heading: np.ndarray
    ego_speed: np.ndarray
    
    # Ego actions (T,)
    acceleration: np.ndarray
    steering: np.ndarray
    
    # Other agents (T, N_agents)
    other_x: np.ndarray
    other_y: np.ndarray
    other_vx: np.ndarray
    other_vy: np.ndarray
    other_valid: np.ndarray
    
    # Computed safety metrics (T,) or (T, N_agents)
    ttc_per_agent: np.ndarray          # (T, N_agents)
    min_ttc: np.ndarray                # (T,)
    min_distance: np.ndarray           # (T,)
    nearest_agent_id: np.ndarray       # (T,)
    
    # Traffic lights (T, N_lights)
    traffic_light_states: np.ndarray   # 0=unknown, 1=red, 2=yellow, 3=green
    traffic_light_distances: np.ndarray
    
    # Scenario outcome
    has_collision: bool
    collision_time: Optional[int]
    has_offroad: bool
    offroad_time: Optional[int]
    
    # Road geometry
    lane_deviation: np.ndarray         # (T,) lateral offset from lane center


class EventDetector(ABC):
    """
    Base class for all event detectors.
    
    Each detector looks for one type of event (or a family of related events).
    Detectors are stateless — all data comes from ScenarioData.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Args:
            config: Detector-specific thresholds and parameters
        """
        self.config = config or {}
        self._set_default_config()
    
    @abstractmethod
    def _set_default_config(self):
        """Set default threshold values. Override in subclasses."""
        pass
    
    @property
    @abstractmethod
    def event_types(self) -> List[EventType]:
        """List of event types this detector can produce."""
        pass
    
    @abstractmethod
    def detect(self, data: ScenarioData) -> List[Event]:
        """
        Detect all events of this type in the scenario.
        
        Args:
            data: Pre-computed scenario data
            
        Returns:
            List of detected events (may be empty)
        """
        pass
    
    def _compute_window(
        self, 
        onset: int, 
        offset: int, 
        num_timesteps: int,
        buffer_before: int = 10,
        buffer_after: int = 10,
    ) -> tuple:
        """
        Compute analysis window around an event.
        
        Args:
            onset: First timestep of event condition
            offset: Last timestep of event condition
            num_timesteps: Total timesteps in scenario
            buffer_before: Timesteps to include before onset
            buffer_after: Timesteps to include after offset
            
        Returns:
            (window_start, window_end)
        """
        window_start = max(0, onset - buffer_before)
        window_end = min(num_timesteps - 1, offset + buffer_after)
        return window_start, window_end
    
    def _find_event_peak(
        self, 
        signal: np.ndarray, 
        onset: int, 
        offset: int,
        mode: str = 'min'  # 'min' for TTC, 'max' for |acceleration|
    ) -> int:
        """Find the timestep with peak severity within event window."""
        window = signal[onset:offset+1]
        if mode == 'min':
            peak_idx = np.argmin(window)
        else:
            peak_idx = np.argmax(window)
        return onset + peak_idx
```

---

# 4. Event Detectors

## 4.1 Safety Event Detectors

```python
# event_mining/events/safety.py

import numpy as np
from typing import List, Optional
from .base import EventDetector, Event, EventType, Severity, ScenarioData


class HazardOnsetDetector(EventDetector):
    """
    Detects when TTC drops below threshold — the moment danger appears.
    
    This is the most important event type for XAI analysis.
    """
    
    def _set_default_config(self):
        self.config.setdefault('ttc_threshold', 3.0)       # seconds
        self.config.setdefault('min_duration', 3)          # timesteps
        self.config.setdefault('buffer_before', 15)        # ~1.5 seconds
        self.config.setdefault('buffer_after', 10)
    
    @property
    def event_types(self) -> List[EventType]:
        return [EventType.HAZARD_ONSET, EventType.COLLISION_IMMINENT]
    
    def detect(self, data: ScenarioData) -> List[Event]:
        events = []
        ttc_threshold = self.config['ttc_threshold']
        min_duration = self.config['min_duration']
        
        # Find all continuous windows where TTC < threshold
        below_threshold = data.min_ttc < ttc_threshold
        
        # Find onset/offset pairs
        windows = self._find_continuous_windows(below_threshold, min_duration)
        
        for onset, offset in windows:
            # Find peak (minimum TTC)
            peak_time = onset + np.argmin(data.min_ttc[onset:offset+1])
            peak_ttc = data.min_ttc[peak_time]
            
            # Determine severity
            if peak_ttc < 1.0:
                severity = Severity.CRITICAL
                event_type = EventType.COLLISION_IMMINENT
            elif peak_ttc < 1.5:
                severity = Severity.HIGH
                event_type = EventType.HAZARD_ONSET
            elif peak_ttc < 2.0:
                severity = Severity.MEDIUM
                event_type = EventType.HAZARD_ONSET
            else:
                severity = Severity.LOW
                event_type = EventType.HAZARD_ONSET
            
            # Compute severity score (0-1, higher = more dangerous)
            severity_score = max(0, 1 - peak_ttc / ttc_threshold)
            
            # Get causal agent
            causal_agent = int(data.nearest_agent_id[peak_time])
            
            # Compute window
            window_start, window_end = self._compute_window(
                onset, offset, data.num_timesteps,
                self.config['buffer_before'],
                self.config['buffer_after']
            )
            
            events.append(Event(
                scenario_id=data.scenario_id,
                event_type=event_type,
                event_time=peak_time,
                onset_time=onset,
                offset_time=offset,
                window_start=window_start,
                window_end=window_end,
                severity=severity,
                severity_score=severity_score,
                metadata={
                    'min_ttc': float(peak_ttc),
                    'ttc_at_onset': float(data.min_ttc[onset]),
                    'duration_timesteps': offset - onset + 1,
                },
                causal_agent_id=causal_agent,
            ))
        
        return events
    
    def _find_continuous_windows(
        self, 
        mask: np.ndarray, 
        min_duration: int
    ) -> List[tuple]:
        """Find continuous True regions in boolean mask."""
        windows = []
        in_window = False
        onset = 0
        
        for t in range(len(mask)):
            if mask[t] and not in_window:
                # Window starts
                in_window = True
                onset = t
            elif not mask[t] and in_window:
                # Window ends
                in_window = False
                if t - onset >= min_duration:
                    windows.append((onset, t - 1))
        
        # Handle window that extends to end
        if in_window and len(mask) - onset >= min_duration:
            windows.append((onset, len(mask) - 1))
        
        return windows


class NearMissDetector(EventDetector):
    """
    Detects near-miss events: minimum distance < threshold but no collision.
    """
    
    def _set_default_config(self):
        self.config.setdefault('distance_threshold', 2.0)  # meters
        self.config.setdefault('buffer_before', 15)
        self.config.setdefault('buffer_after', 10)
    
    @property
    def event_types(self) -> List[EventType]:
        return [EventType.NEAR_MISS]
    
    def detect(self, data: ScenarioData) -> List[Event]:
        if data.has_collision:
            return []  # Not a near-miss if there was a collision
        
        events = []
        threshold = self.config['distance_threshold']
        
        # Find minimum distance moment
        min_dist_time = np.argmin(data.min_distance)
        min_dist = data.min_distance[min_dist_time]
        
        if min_dist < threshold:
            # Compute window around the near-miss moment
            # Find when distance first dropped below threshold
            below_threshold = data.min_distance < threshold * 1.5
            onset = self._find_first_true(below_threshold, start=max(0, min_dist_time - 20))
            offset = self._find_last_true(below_threshold, end=min(len(below_threshold), min_dist_time + 20))
            
            window_start, window_end = self._compute_window(
                onset, offset, data.num_timesteps,
                self.config['buffer_before'],
                self.config['buffer_after']
            )
            
            severity_score = max(0, 1 - min_dist / threshold)
            
            events.append(Event(
                scenario_id=data.scenario_id,
                event_type=EventType.NEAR_MISS,
                event_time=min_dist_time,
                onset_time=onset,
                offset_time=offset,
                window_start=window_start,
                window_end=window_end,
                severity=Severity.HIGH if min_dist < 1.0 else Severity.MEDIUM,
                severity_score=severity_score,
                metadata={
                    'min_distance': float(min_dist),
                    'min_ttc_at_event': float(data.min_ttc[min_dist_time]),
                },
                causal_agent_id=int(data.nearest_agent_id[min_dist_time]),
            ))
        
        return events
    
    def _find_first_true(self, mask: np.ndarray, start: int = 0) -> int:
        for t in range(start, len(mask)):
            if mask[t]:
                return t
        return start
    
    def _find_last_true(self, mask: np.ndarray, end: int = None) -> int:
        end = end or len(mask)
        for t in range(end - 1, -1, -1):
            if mask[t]:
                return t
        return end - 1
```

## 4.2 Action Event Detectors

```python
# event_mining/events/action.py

import numpy as np
from typing import List
from .base import EventDetector, Event, EventType, Severity, ScenarioData


class HardBrakeDetector(EventDetector):
    """
    Detects hard braking events (high negative acceleration).
    """
    
    def _set_default_config(self):
        self.config.setdefault('accel_threshold', -3.0)    # m/s²
        self.config.setdefault('severe_threshold', -5.0)   # m/s²
        self.config.setdefault('min_duration', 2)          # timesteps
        self.config.setdefault('buffer_before', 10)
        self.config.setdefault('buffer_after', 5)
    
    @property
    def event_types(self) -> List[EventType]:
        return [EventType.HARD_BRAKE]
    
    def detect(self, data: ScenarioData) -> List[Event]:
        events = []
        threshold = self.config['accel_threshold']
        severe_threshold = self.config['severe_threshold']
        
        # Find hard braking windows
        hard_brake = data.acceleration < threshold
        windows = self._find_continuous_windows(hard_brake, self.config['min_duration'])
        
        for onset, offset in windows:
            # Find peak (minimum acceleration = hardest brake)
            peak_time = onset + np.argmin(data.acceleration[onset:offset+1])
            peak_accel = data.acceleration[peak_time]
            
            # Severity based on deceleration magnitude
            if peak_accel < severe_threshold:
                severity = Severity.HIGH
            elif peak_accel < threshold * 1.3:
                severity = Severity.MEDIUM
            else:
                severity = Severity.LOW
            
            severity_score = min(1.0, abs(peak_accel) / abs(severe_threshold))
            
            window_start, window_end = self._compute_window(
                onset, offset, data.num_timesteps,
                self.config['buffer_before'],
                self.config['buffer_after']
            )
            
            events.append(Event(
                scenario_id=data.scenario_id,
                event_type=EventType.HARD_BRAKE,
                event_time=peak_time,
                onset_time=onset,
                offset_time=offset,
                window_start=window_start,
                window_end=window_end,
                severity=severity,
                severity_score=severity_score,
                metadata={
                    'peak_acceleration': float(peak_accel),
                    'speed_at_onset': float(data.ego_speed[onset]),
                    'speed_at_offset': float(data.ego_speed[offset]),
                    'ttc_at_event': float(data.min_ttc[peak_time]),
                },
                causal_agent_id=int(data.nearest_agent_id[peak_time]) if data.min_ttc[peak_time] < 5 else None,
            ))
        
        return events
    
    def _find_continuous_windows(self, mask: np.ndarray, min_duration: int) -> List[tuple]:
        """Find continuous True regions in boolean mask."""
        windows = []
        in_window = False
        onset = 0
        
        for t in range(len(mask)):
            if mask[t] and not in_window:
                in_window = True
                onset = t
            elif not mask[t] and in_window:
                in_window = False
                if t - onset >= min_duration:
                    windows.append((onset, t - 1))
        
        if in_window and len(mask) - onset >= min_duration:
            windows.append((onset, len(mask) - 1))
        
        return windows


class EvasiveSteeringDetector(EventDetector):
    """
    Detects evasive steering maneuvers (high steering magnitude).
    """
    
    def _set_default_config(self):
        self.config.setdefault('steering_threshold', 0.3)  # radians
        self.config.setdefault('severe_threshold', 0.5)
        self.config.setdefault('min_duration', 2)
        self.config.setdefault('buffer_before', 10)
        self.config.setdefault('buffer_after', 5)
    
    @property
    def event_types(self) -> List[EventType]:
        return [EventType.EVASIVE_STEERING]
    
    def detect(self, data: ScenarioData) -> List[Event]:
        events = []
        threshold = self.config['steering_threshold']
        
        # Find high-magnitude steering
        evasive = np.abs(data.steering) > threshold
        windows = self._find_continuous_windows(evasive, self.config['min_duration'])
        
        for onset, offset in windows:
            peak_time = onset + np.argmax(np.abs(data.steering[onset:offset+1]))
            peak_steering = data.steering[peak_time]
            
            severity_score = min(1.0, abs(peak_steering) / self.config['severe_threshold'])
            severity = Severity.HIGH if abs(peak_steering) > self.config['severe_threshold'] else Severity.MEDIUM
            
            window_start, window_end = self._compute_window(
                onset, offset, data.num_timesteps,
                self.config['buffer_before'],
                self.config['buffer_after']
            )
            
            events.append(Event(
                scenario_id=data.scenario_id,
                event_type=EventType.EVASIVE_STEERING,
                event_time=peak_time,
                onset_time=onset,
                offset_time=offset,
                window_start=window_start,
                window_end=window_end,
                severity=severity,
                severity_score=severity_score,
                metadata={
                    'peak_steering': float(peak_steering),
                    'direction': 'left' if peak_steering > 0 else 'right',
                    'speed_at_event': float(data.ego_speed[peak_time]),
                    'ttc_at_event': float(data.min_ttc[peak_time]),
                },
                causal_agent_id=int(data.nearest_agent_id[peak_time]) if data.min_ttc[peak_time] < 5 else None,
            ))
        
        return events
    
    def _find_continuous_windows(self, mask: np.ndarray, min_duration: int) -> List[tuple]:
        windows = []
        in_window = False
        onset = 0
        
        for t in range(len(mask)):
            if mask[t] and not in_window:
                in_window = True
                onset = t
            elif not mask[t] and in_window:
                in_window = False
                if t - onset >= min_duration:
                    windows.append((onset, t - 1))
        
        if in_window and len(mask) - onset >= min_duration:
            windows.append((onset, len(mask) - 1))
        
        return windows
```

## 4.3 Outcome Event Detectors

```python
# event_mining/events/outcome.py

import numpy as np
from typing import List
from .base import EventDetector, Event, EventType, Severity, ScenarioData


class CollisionDetector(EventDetector):
    """
    Detects collision events — the most critical outcome.
    """
    
    def _set_default_config(self):
        self.config.setdefault('buffer_before', 20)  # 2 seconds before
        self.config.setdefault('buffer_after', 5)
    
    @property
    def event_types(self) -> List[EventType]:
        return [EventType.COLLISION]
    
    def detect(self, data: ScenarioData) -> List[Event]:
        if not data.has_collision:
            return []
        
        collision_time = data.collision_time
        
        # Find when danger started (TTC began dropping significantly)
        # Look back from collision to find onset
        onset = collision_time
        for t in range(collision_time - 1, max(0, collision_time - 30), -1):
            if data.min_ttc[t] > 5.0:
                onset = t + 1
                break
        
        window_start, window_end = self._compute_window(
            onset, collision_time, data.num_timesteps,
            self.config['buffer_before'],
            self.config['buffer_after']
        )
        
        return [Event(
            scenario_id=data.scenario_id,
            event_type=EventType.COLLISION,
            event_time=collision_time,
            onset_time=onset,
            offset_time=collision_time,
            window_start=window_start,
            window_end=window_end,
            severity=Severity.CRITICAL,
            severity_score=1.0,
            metadata={
                'ttc_at_onset': float(data.min_ttc[onset]),
                'speed_at_collision': float(data.ego_speed[collision_time]),
                'time_from_onset': collision_time - onset,
            },
            causal_agent_id=int(data.nearest_agent_id[collision_time]),
        )]


class OffRoadDetector(EventDetector):
    """
    Detects off-road events.
    """
    
    def _set_default_config(self):
        self.config.setdefault('buffer_before', 15)
        self.config.setdefault('buffer_after', 5)
    
    @property
    def event_types(self) -> List[EventType]:
        return [EventType.OFF_ROAD]
    
    def detect(self, data: ScenarioData) -> List[Event]:
        if not data.has_offroad:
            return []
        
        offroad_time = data.offroad_time
        
        # Find when lane deviation started increasing
        onset = offroad_time
        for t in range(offroad_time - 1, max(0, offroad_time - 20), -1):
            if abs(data.lane_deviation[t]) < 0.5:  # Within normal lane bounds
                onset = t + 1
                break
        
        window_start, window_end = self._compute_window(
            onset, offroad_time, data.num_timesteps,
            self.config['buffer_before'],
            self.config['buffer_after']
        )
        
        return [Event(
            scenario_id=data.scenario_id,
            event_type=EventType.OFF_ROAD,
            event_time=offroad_time,
            onset_time=onset,
            offset_time=offroad_time,
            window_start=window_start,
            window_end=window_end,
            severity=Severity.HIGH,
            severity_score=0.8,
            metadata={
                'lane_deviation_at_event': float(data.lane_deviation[offroad_time]),
                'speed_at_event': float(data.ego_speed[offroad_time]),
            },
        )]
```

---

# 5. Event Catalog

```python
# event_mining/catalog.py

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set
from collections import defaultdict

from .events.base import Event, EventType, Severity


@dataclass
class EventCatalog:
    """
    A queryable collection of detected events.
    
    Provides filtering, grouping, and serialization capabilities.
    """
    events: List[Event] = field(default_factory=list)
    scenarios_scanned: int = 0
    
    # Indexes for fast lookup
    _by_scenario: Dict[str, List[Event]] = field(default_factory=lambda: defaultdict(list))
    _by_type: Dict[EventType, List[Event]] = field(default_factory=lambda: defaultdict(list))
    _indexed: bool = False
    
    def add(self, event: Event):
        """Add a single event."""
        self.events.append(event)
        self._indexed = False
    
    def add_all(self, events: List[Event]):
        """Add multiple events."""
        self.events.extend(events)
        self._indexed = False
    
    def _build_indexes(self):
        """Build lookup indexes for fast filtering."""
        if self._indexed:
            return
        
        self._by_scenario = defaultdict(list)
        self._by_type = defaultdict(list)
        
        for event in self.events:
            self._by_scenario[event.scenario_id].append(event)
            self._by_type[event.event_type].append(event)
        
        self._indexed = True
    
    # ===== Filtering Methods =====
    
    def filter(
        self,
        event_types: Optional[List[EventType]] = None,
        min_severity: Optional[Severity] = None,
        min_severity_score: Optional[float] = None,
        scenario_ids: Optional[Set[str]] = None,
        has_causal_agent: Optional[bool] = None,
    ) -> 'EventCatalog':
        """
        Filter events by criteria.
        
        Returns a new EventCatalog with matching events.
        """
        filtered = []
        
        for event in self.events:
            # Type filter
            if event_types and event.event_type not in event_types:
                continue
            
            # Severity filter
            if min_severity and event.severity.value < min_severity.value:
                continue
            
            if min_severity_score and event.severity_score < min_severity_score:
                continue
            
            # Scenario filter
            if scenario_ids and event.scenario_id not in scenario_ids:
                continue
            
            # Causal agent filter
            if has_causal_agent is not None:
                if has_causal_agent and event.causal_agent_id is None:
                    continue
                if not has_causal_agent and event.causal_agent_id is not None:
                    continue
            
            filtered.append(event)
        
        return EventCatalog(
            events=filtered,
            scenarios_scanned=self.scenarios_scanned
        )
    
    def by_scenario(self, scenario_id: str) -> List[Event]:
        """Get all events for a specific scenario."""
        self._build_indexes()
        return self._by_scenario.get(scenario_id, [])
    
    def by_type(self, event_type: EventType) -> List[Event]:
        """Get all events of a specific type."""
        self._build_indexes()
        return self._by_type.get(event_type, [])
    
    # ===== Aggregation Methods =====
    
    def get_windows(self) -> List[tuple]:
        """
        Get all unique (scenario_id, window_start, window_end) tuples.
        
        This is the primary output for XAI analysis.
        """
        windows = set()
        for event in self.events:
            windows.add((event.scenario_id, event.window_start, event.window_end))
        return sorted(list(windows))
    
    def get_analysis_points(self) -> List[tuple]:
        """
        Get all (scenario_id, timestep) pairs for analysis.
        
        Returns event peak times, not full windows.
        """
        points = set()
        for event in self.events:
            points.add((event.scenario_id, event.event_time))
        return sorted(list(points))
    
    def get_scenario_ids(self) -> Set[str]:
        """Get all unique scenario IDs with events."""
        return set(event.scenario_id for event in self.events)
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        self._build_indexes()
        
        type_counts = {et.value: len(events) for et, events in self._by_type.items()}
        severity_counts = defaultdict(int)
        for event in self.events:
            severity_counts[event.severity.name] += 1
        
        return {
            'total_events': len(self.events),
            'scenarios_with_events': len(self._by_scenario),
            'scenarios_scanned': self.scenarios_scanned,
            'events_per_scenario': len(self.events) / max(1, len(self._by_scenario)),
            'by_type': type_counts,
            'by_severity': dict(severity_counts),
        }
    
    # ===== Serialization =====
    
    def save(self, path: str):
        """Save catalog to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'scenarios_scanned': self.scenarios_scanned,
            'events': [e.to_dict() for e in self.events],
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'EventCatalog':
        """Load catalog from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(
            scenarios_scanned=data['scenarios_scanned'],
            events=[Event.from_dict(e) for e in data['events']],
        )
    
    # ===== Iteration =====
    
    def __len__(self):
        return len(self.events)
    
    def __iter__(self):
        return iter(self.events)
    
    def __getitem__(self, idx):
        return self.events[idx]
```

---

# 6. Event Miner (Main Orchestrator)

```python
# event_mining/miner.py

import numpy as np
from typing import List, Optional, Dict, Any, Generator
from pathlib import Path
from tqdm import tqdm

from .events.base import EventDetector, ScenarioData
from .events.safety import HazardOnsetDetector, NearMissDetector
from .events.action import HardBrakeDetector, EvasiveSteeringDetector
from .events.outcome import CollisionDetector, OffRoadDetector
from .catalog import EventCatalog
from .metrics import compute_scenario_metrics


# Default detector configuration
DEFAULT_DETECTORS = [
    HazardOnsetDetector,
    NearMissDetector,
    HardBrakeDetector,
    EvasiveSteeringDetector,
    CollisionDetector,
    OffRoadDetector,
]


class EventMiner:
    """
    Main class for mining events from driving scenarios.
    
    Orchestrates multiple detectors and produces an EventCatalog.
    """
    
    def __init__(
        self,
        detectors: Optional[List[EventDetector]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            detectors: List of EventDetector instances. If None, uses defaults.
            config: Global configuration overrides.
        """
        self.config = config or {}
        
        if detectors is None:
            # Instantiate default detectors
            self.detectors = [D() for D in DEFAULT_DETECTORS]
        else:
            self.detectors = detectors
    
    def mine_scenario(self, data: ScenarioData) -> List:
        """
        Run all detectors on a single scenario.
        
        Args:
            data: Pre-computed scenario data
            
        Returns:
            List of detected events
        """
        all_events = []
        
        for detector in self.detectors:
            try:
                events = detector.detect(data)
                all_events.extend(events)
            except Exception as e:
                print(f"Detector {detector.__class__.__name__} failed on {data.scenario_id}: {e}")
        
        return all_events
    
    def mine_scenarios(
        self,
        scenario_iterator: Generator[ScenarioData, None, None],
        max_scenarios: Optional[int] = None,
        show_progress: bool = True,
    ) -> EventCatalog:
        """
        Mine events from multiple scenarios.
        
        Args:
            scenario_iterator: Generator yielding ScenarioData objects
            max_scenarios: Maximum scenarios to process (None = all)
            show_progress: Show progress bar
            
        Returns:
            EventCatalog with all detected events
        """
        catalog = EventCatalog()
        
        iterator = scenario_iterator
        if show_progress:
            iterator = tqdm(iterator, desc="Mining events", total=max_scenarios)
        
        for i, data in enumerate(iterator):
            if max_scenarios and i >= max_scenarios:
                break
            
            events = self.mine_scenario(data)
            catalog.add_all(events)
            catalog.scenarios_scanned += 1
        
        return catalog
    
    def mine_from_vmax(
        self,
        env,
        scenario_ids: List[str],
        model=None,
        show_progress: bool = True,
    ) -> EventCatalog:
        """
        Mine events directly from V-Max environment.
        
        Args:
            env: V-Max environment instance
            scenario_ids: List of scenario IDs to process
            model: Optional model for action-based events (uses log-replay if None)
            show_progress: Show progress bar
            
        Returns:
            EventCatalog
        """
        from .integration.vmax_adapter import VMaxAdapter
        
        adapter = VMaxAdapter(env)
        
        def scenario_generator():
            for scenario_id in scenario_ids:
                yield adapter.extract_scenario_data(scenario_id, model)
        
        return self.mine_scenarios(
            scenario_generator(),
            max_scenarios=len(scenario_ids),
            show_progress=show_progress,
        )


# ===== Convenience Functions =====

def mine_events(
    env,
    scenario_ids: List[str],
    model=None,
    detectors: Optional[List[EventDetector]] = None,
    save_path: Optional[str] = None,
) -> EventCatalog:
    """
    Convenience function for one-shot event mining.
    
    Args:
        env: V-Max environment
        scenario_ids: Scenarios to scan
        model: Model for rollouts (None = log-replay)
        detectors: Custom detectors (None = defaults)
        save_path: Optional path to save catalog
        
    Returns:
        EventCatalog
    """
    miner = EventMiner(detectors=detectors)
    catalog = miner.mine_from_vmax(env, scenario_ids, model)
    
    if save_path:
        catalog.save(save_path)
    
    return catalog
```

---

# 7. V-Max Integration

```python
# event_mining/integration/vmax_adapter.py

import numpy as np
from typing import Optional
import jax.numpy as jnp

from ..events.base import ScenarioData
from ..metrics import compute_ttc, compute_min_distance


class VMaxAdapter:
    """
    Extracts ScenarioData from V-Max environment and model.
    """
    
    def __init__(self, env):
        """
        Args:
            env: V-Max environment instance (with loaded scenario data)
        """
        self.env = env
    
    def extract_scenario_data(
        self, 
        scenario_id: str,
        model=None,
    ) -> ScenarioData:
        """
        Run a scenario and extract all data needed for event detection.
        
        Args:
            scenario_id: Scenario to load
            model: Model to use for rollout. If None, uses log-replay (expert).
            
        Returns:
            ScenarioData with all computed metrics
        """
        # Load scenario
        state = self.env.reset(scenario_id)
        
        # Storage
        ego_data = {'x': [], 'y': [], 'vx': [], 'vy': [], 'heading': [], 'speed': []}
        action_data = {'acceleration': [], 'steering': []}
        other_data = {'x': [], 'y': [], 'vx': [], 'vy': [], 'valid': []}
        tl_data = {'states': [], 'distances': []}
        
        collision_time = None
        offroad_time = None
        
        done = False
        t = 0
        
        while not done:
            # Get current state info
            ego_state = self._extract_ego_state(state)
            others_state = self._extract_others_state(state)
            tl_state = self._extract_traffic_lights(state)
            
            # Store
            for k, v in ego_state.items():
                ego_data[k].append(v)
            for k, v in others_state.items():
                other_data[k].append(v)
            for k, v in tl_state.items():
                tl_data[k].append(v)
            
            # Get action
            if model is not None:
                obs = self.env.get_observation(state)
                action = model.get_action(obs)
            else:
                action = self._get_log_action(state, t)
            
            action_data['acceleration'].append(float(action[0]))
            action_data['steering'].append(float(action[1]))
            
            # Step
            state, reward, done, info = self.env.step(state, action)
            
            # Check for outcomes
            if info.get('collision') and collision_time is None:
                collision_time = t
            if info.get('offroad') and offroad_time is None:
                offroad_time = t
            
            t += 1
        
        # Convert to arrays
        num_timesteps = t
        
        ego_x = np.array(ego_data['x'])
        ego_y = np.array(ego_data['y'])
        ego_vx = np.array(ego_data['vx'])
        ego_vy = np.array(ego_data['vy'])
        
        other_x = np.stack(other_data['x'])  # (T, N)
        other_y = np.stack(other_data['y'])
        other_vx = np.stack(other_data['vx'])
        other_vy = np.stack(other_data['vy'])
        other_valid = np.stack(other_data['valid'])
        
        # Compute derived metrics
        ttc_per_agent = compute_ttc(
            ego_x, ego_y, ego_vx, ego_vy,
            other_x, other_y, other_vx, other_vy,
            other_valid
        )
        
        min_ttc = np.min(np.where(other_valid, ttc_per_agent, 999), axis=1)
        min_ttc = np.clip(min_ttc, 0, 10)
        
        distances = np.sqrt((other_x - ego_x[:, None])**2 + (other_y - ego_y[:, None])**2)
        distances = np.where(other_valid, distances, 999)
        min_distance = np.min(distances, axis=1)
        nearest_agent_id = np.argmin(distances, axis=1)
        
        return ScenarioData(
            scenario_id=scenario_id,
            num_timesteps=num_timesteps,
            ego_x=ego_x,
            ego_y=ego_y,
            ego_vx=ego_vx,
            ego_vy=ego_vy,
            ego_heading=np.array(ego_data['heading']),
            ego_speed=np.array(ego_data['speed']),
            acceleration=np.array(action_data['acceleration']),
            steering=np.array(action_data['steering']),
            other_x=other_x,
            other_y=other_y,
            other_vx=other_vx,
            other_vy=other_vy,
            other_valid=other_valid,
            ttc_per_agent=ttc_per_agent,
            min_ttc=min_ttc,
            min_distance=min_distance,
            nearest_agent_id=nearest_agent_id,
            traffic_light_states=np.stack(tl_data['states']),
            traffic_light_distances=np.stack(tl_data['distances']),
            has_collision=collision_time is not None,
            collision_time=collision_time,
            has_offroad=offroad_time is not None,
            offroad_time=offroad_time,
            lane_deviation=self._compute_lane_deviation(ego_x, ego_y, state),
        )
    
    def _extract_ego_state(self, state) -> dict:
        """Extract ego vehicle state from simulator state."""
        # Implementation depends on V-Max state structure
        sim_state = state.sim_state
        ego_idx = 0  # Ego is typically index 0
        
        return {
            'x': float(sim_state.x[ego_idx]),
            'y': float(sim_state.y[ego_idx]),
            'vx': float(sim_state.vx[ego_idx]),
            'vy': float(sim_state.vy[ego_idx]),
            'heading': float(sim_state.yaw[ego_idx]),
            'speed': float(np.sqrt(sim_state.vx[ego_idx]**2 + sim_state.vy[ego_idx]**2)),
        }
    
    def _extract_others_state(self, state) -> dict:
        """Extract other agents' states."""
        sim_state = state.sim_state
        # Skip ego (index 0), get next N agents
        n_agents = min(8, sim_state.x.shape[0] - 1)
        
        return {
            'x': sim_state.x[1:n_agents+1].copy(),
            'y': sim_state.y[1:n_agents+1].copy(),
            'vx': sim_state.vx[1:n_agents+1].copy(),
            'vy': sim_state.vy[1:n_agents+1].copy(),
            'valid': sim_state.valid[1:n_agents+1].copy(),
        }
    
    def _extract_traffic_lights(self, state) -> dict:
        """Extract traffic light data."""
        # Implementation depends on V-Max state structure
        # Placeholder
        return {
            'states': np.zeros(5),
            'distances': np.ones(5) * 100,
        }
    
    def _get_log_action(self, state, t) -> np.ndarray:
        """Get logged expert action at timestep t."""
        # Extract from logged trajectory
        return np.array([0.0, 0.0])  # Placeholder
    
    def _compute_lane_deviation(self, ego_x, ego_y, state) -> np.ndarray:
        """Compute lateral deviation from lane center."""
        # Would require road geometry; placeholder
        return np.zeros(len(ego_x))
```

---

# 8. XAI Integration

```python
# event_mining/integration/xai_bridge.py

from typing import List, Dict, Any, Generator, Tuple
from ..catalog import EventCatalog
from ..events.base import Event


class XAIBridge:
    """
    Bridge between EventCatalog and XAI analysis framework.
    
    Provides iterators and data structures optimized for XAI experiments.
    """
    
    def __init__(self, catalog: EventCatalog):
        self.catalog = catalog
    
    def iter_analysis_windows(
        self,
        event_types: List = None,
        min_severity_score: float = 0.0,
    ) -> Generator[Tuple[str, int, int, Event], None, None]:
        """
        Iterate over analysis windows.
        
        Yields: (scenario_id, window_start, window_end, event)
        """
        filtered = self.catalog.filter(
            event_types=event_types,
            min_severity_score=min_severity_score,
        )
        
        for event in filtered:
            yield (
                event.scenario_id,
                event.window_start,
                event.window_end,
                event,
            )
    
    def iter_event_timesteps(
        self,
        event_types: List = None,
        include_context: int = 0,  # Timesteps before/after
    ) -> Generator[Tuple[str, int, Dict[str, Any]], None, None]:
        """
        Iterate over individual timesteps for analysis.
        
        Yields: (scenario_id, timestep, event_metadata)
        """
        filtered = self.catalog.filter(event_types=event_types)
        
        for event in filtered:
            # Yield event peak
            yield (
                event.scenario_id,
                event.event_time,
                {
                    'event_type': event.event_type.value,
                    'severity': event.severity_score,
                    'is_peak': True,
                    **event.metadata,
                }
            )
            
            # Optionally yield context timesteps
            if include_context > 0:
                for dt in range(-include_context, include_context + 1):
                    if dt == 0:
                        continue
                    t = event.event_time + dt
                    if event.window_start <= t <= event.window_end:
                        yield (
                            event.scenario_id,
                            t,
                            {
                                'event_type': event.event_type.value,
                                'severity': event.severity_score,
                                'is_peak': False,
                                'offset_from_peak': dt,
                            }
                        )
    
    def get_scenario_event_timeline(
        self, 
        scenario_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get timeline of all events in a scenario.
        
        Useful for temporal visualizations.
        """
        events = self.catalog.by_scenario(scenario_id)
        
        timeline = []
        for event in sorted(events, key=lambda e: e.event_time):
            timeline.append({
                'time': event.event_time,
                'type': event.event_type.value,
                'severity': event.severity_score,
                'window': (event.window_start, event.window_end),
                'metadata': event.metadata,
            })
        
        return timeline
    
    def to_dataframe(self):
        """Convert catalog to pandas DataFrame for analysis."""
        import pandas as pd
        
        records = []
        for event in self.catalog:
            records.append({
                'scenario_id': event.scenario_id,
                'event_type': event.event_type.value,
                'event_time': event.event_time,
                'onset_time': event.onset_time,
                'offset_time': event.offset_time,
                'window_start': event.window_start,
                'window_end': event.window_end,
                'severity': event.severity.name,
                'severity_score': event.severity_score,
                'causal_agent_id': event.causal_agent_id,
                **{f'meta_{k}': v for k, v in event.metadata.items()},
            })
        
        return pd.DataFrame(records)
```

---

# 9. Metrics Computation

```python
# event_mining/metrics.py

import numpy as np
from typing import Tuple


def compute_ttc(
    ego_x: np.ndarray,
    ego_y: np.ndarray, 
    ego_vx: np.ndarray,
    ego_vy: np.ndarray,
    other_x: np.ndarray,
    other_y: np.ndarray,
    other_vx: np.ndarray,
    other_vy: np.ndarray,
    other_valid: np.ndarray,
    max_ttc: float = 10.0,
) -> np.ndarray:
    """
    Compute Time-To-Collision for all agents at all timesteps.
    
    Args:
        ego_*: Ego state arrays (T,)
        other_*: Other agent arrays (T, N_agents)
        other_valid: Validity mask (T, N_agents)
        max_ttc: Maximum TTC value (clip)
        
    Returns:
        TTC array (T, N_agents)
    """
    T, N = other_x.shape
    ttc = np.full((T, N), max_ttc)
    
    for t in range(T):
        for i in range(N):
            if not other_valid[t, i]:
                continue
            
            # Relative position and velocity
            rel_x = other_x[t, i] - ego_x[t]
            rel_y = other_y[t, i] - ego_y[t]
            rel_vx = ego_vx[t] - other_vx[t, i]
            rel_vy = ego_vy[t] - other_vy[t, i]
            
            distance = np.sqrt(rel_x**2 + rel_y**2)
            
            # Closing speed (positive = approaching)
            if distance > 0.01:
                closing_speed = (rel_x * rel_vx + rel_y * rel_vy) / distance
            else:
                closing_speed = 0
            
            if closing_speed > 0.1:
                ttc[t, i] = min(distance / closing_speed, max_ttc)
            else:
                ttc[t, i] = max_ttc
    
    return ttc


def compute_criticality(
    ttc: np.ndarray,
    min_distance: np.ndarray,
    ego_speed: np.ndarray,
    ttc_weight: float = 0.4,
    distance_weight: float = 0.3,
    speed_weight: float = 0.3,
) -> np.ndarray:
    """
    Compute composite criticality score.
    
    Args:
        ttc: Minimum TTC array (T,)
        min_distance: Minimum distance to other agents (T,)
        ego_speed: Ego speed (T,)
        
    Returns:
        Criticality score array (T,), range [0, 1]
    """
    # TTC component (lower TTC = higher criticality)
    ttc_score = np.clip(1 - ttc / 5.0, 0, 1)
    
    # Distance component (closer = higher criticality)
    distance_score = np.clip(1 - min_distance / 20.0, 0, 1)
    
    # Speed component (faster = higher criticality in dangerous situations)
    speed_score = np.clip(ego_speed / 15.0, 0, 1)
    
    # Weighted combination
    criticality = (
        ttc_weight * ttc_score +
        distance_weight * distance_score +
        speed_weight * speed_score * ttc_score  # Speed only matters when TTC is low
    )
    
    return np.clip(criticality, 0, 1)
```

---

# 10. CLI Interface

```python
# event_mining/cli.py

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Event Mining for AD Scenarios')
    subparsers = parser.add_subparsers(dest='command')
    
    # Mine command
    mine_parser = subparsers.add_parser('mine', help='Mine events from scenarios')
    mine_parser.add_argument('--scenarios', type=int, default=1000, help='Number of scenarios')
    mine_parser.add_argument('--output', type=str, default='events/catalog.json', help='Output path')
    mine_parser.add_argument('--model', type=str, default=None, help='Model path (None=log-replay)')
    
    # Summary command
    summary_parser = subparsers.add_parser('summary', help='Summarize event catalog')
    summary_parser.add_argument('--catalog', type=str, required=True, help='Catalog path')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export events for XAI analysis')
    export_parser.add_argument('--catalog', type=str, required=True)
    export_parser.add_argument('--output', type=str, required=True)
    export_parser.add_argument('--types', nargs='+', default=None, help='Event types to export')
    export_parser.add_argument('--min-severity', type=float, default=0.0)
    
    args = parser.parse_args()
    
    if args.command == 'mine':
        from .miner import mine_events
        # Would need environment setup here
        print(f"Mining {args.scenarios} scenarios...")
        # catalog = mine_events(env, scenario_ids, save_path=args.output)
        
    elif args.command == 'summary':
        from .catalog import EventCatalog
        catalog = EventCatalog.load(args.catalog)
        summary = catalog.summary()
        print("\n=== Event Catalog Summary ===")
        for k, v in summary.items():
            print(f"{k}: {v}")
    
    elif args.command == 'export':
        from .catalog import EventCatalog
        from .events.base import EventType
        
        catalog = EventCatalog.load(args.catalog)
        
        if args.types:
            event_types = [EventType(t) for t in args.types]
            catalog = catalog.filter(event_types=event_types)
        
        if args.min_severity > 0:
            catalog = catalog.filter(min_severity_score=args.min_severity)
        
        # Export as CSV
        from .integration.xai_bridge import XAIBridge
        bridge = XAIBridge(catalog)
        df = bridge.to_dataframe()
        df.to_csv(args.output, index=False)
        print(f"Exported {len(df)} events to {args.output}")


if __name__ == '__main__':
    main()
```

---

# 11. Usage Examples

## 11.1 Basic Mining

```python
from event_mining import EventMiner, EventCatalog
from event_mining.integration.vmax_adapter import VMaxAdapter

# Setup V-Max environment
env = create_vmax_env()  # Your env setup

# Mine events
miner = EventMiner()
catalog = miner.mine_from_vmax(env, scenario_ids[:1000])

# Save
catalog.save('events/catalog.json')

# Summary
print(catalog.summary())
```

## 11.2 Filtering for XAI Analysis

```python
from event_mining import EventCatalog
from event_mining.events.base import EventType, Severity

# Load
catalog = EventCatalog.load('events/catalog.json')

# Get only high-severity hazard events
critical_events = catalog.filter(
    event_types=[EventType.HAZARD_ONSET, EventType.COLLISION_IMMINENT],
    min_severity=Severity.HIGH,
)

print(f"Found {len(critical_events)} critical hazard events")

# Get analysis windows
for scenario_id, start, end in critical_events.get_windows():
    print(f"Analyze {scenario_id} timesteps {start}-{end}")
```

## 11.3 Integration with XAI Framework

```python
from event_mining import EventCatalog
from event_mining.integration.xai_bridge import XAIBridge
from posthoc_xai import explain

# Load events
catalog = EventCatalog.load('events/catalog.json')
bridge = XAIBridge(catalog)

# Run XAI on all event timesteps
for scenario_id, timestep, metadata in bridge.iter_event_timesteps():
    # Load scenario and get observation at timestep
    obs = get_observation(scenario_id, timestep)
    
    # Run XAI
    attributions = explain(model, obs)
    
    # Store with event metadata
    results.append({
        'scenario_id': scenario_id,
        'timestep': timestep,
        'event_type': metadata['event_type'],
        'severity': metadata['severity'],
        'attributions': attributions,
    })
```

---

# 12. Implementation Priority

## Phase 1: Core (Day 1-2)

1. `events/base.py` — Event, ScenarioData, EventDetector classes
2. `events/safety.py` — HazardOnsetDetector
3. `catalog.py` — EventCatalog
4. `metrics.py` — TTC computation

## Phase 2: More Detectors (Day 2-3)

5. `events/action.py` — HardBrake, EvasiveSteering
6. `events/outcome.py` — Collision, OffRoad
7. `miner.py` — EventMiner orchestrator

## Phase 3: Integration (Day 3-4)

8. `integration/vmax_adapter.py` — V-Max data extraction
9. `integration/xai_bridge.py` — XAI framework bridge
10. `cli.py` — Command-line interface

## Phase 4: Testing (Day 4)

11. Test on real scenarios
12. Validate TTC computation
13. Tune thresholds based on event distribution

---

*This document provides complete specifications for implementing the Event Mining module. The modular design allows easy extension with new event types and integration with any XAI framework.*
