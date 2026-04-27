"""TemporalAnalyzer: attention trajectories over event windows.

Loads events/test_catalog.json and extracts attention trajectories around
critical events using already-collected TimestepRecord data.

Event catalog scenario_id format: "s000", "s001", etc.
TimestepRecord scenario_id format: integer 0, 1, ...

Matching: scenario 0 → "s000", scenario 1 → "s001", etc.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from event_mining.events.base import Event, EventType
from reward_attention.config import TimestepRecord, AnalysisConfig


# Default path to event catalog
DEFAULT_CATALOG = Path(__file__).parent.parent / "events" / "test_catalog.json"

# Padding around event window (timesteps)
DEFAULT_PADDING = 10

# Attention columns to track over time
ATTN_COLS = ["attn_sdc", "attn_agents", "attn_roadgraph", "attn_lights", "attn_gps"]


def _scenario_id_to_int(scenario_id: str) -> int:
    """Convert 's000' → 0, 's042' → 42, etc."""
    return int(scenario_id.lstrip("s"))


def _int_to_scenario_id(idx: int) -> str:
    """Convert 0 → 's000', 42 → 's042'."""
    return f"s{idx:03d}"


class TemporalAnalyzer:
    """Extract attention trajectories around critical events.

    Args:
        records: All TimestepRecords from the experiment.
        catalog_path: Path to event catalog JSON.
        padding: Extra timesteps before/after event window.
    """

    def __init__(
        self,
        records: List[TimestepRecord],
        catalog_path: Path | str | None = None,
        padding: int = DEFAULT_PADDING,
    ):
        self.padding = padding
        self._catalog_path = Path(catalog_path) if catalog_path else DEFAULT_CATALOG

        # Build lookup: (scenario_id_int, timestep) → TimestepRecord
        self._lookup: Dict[Tuple[int, int], TimestepRecord] = {}
        for rec in records:
            self._lookup[(rec.scenario_id, rec.timestep)] = rec

        # Available scenario integers
        self._available_scenarios = set(r.scenario_id for r in records)

        # Load events
        self._events = self._load_events()

    def _load_events(self) -> List[Event]:
        """Load and parse event catalog."""
        if not self._catalog_path.exists():
            return []
        with open(self._catalog_path) as f:
            data = json.load(f)
        events = []
        for e in data.get("events", []):
            try:
                events.append(Event.from_dict(e))
            except Exception:
                pass
        return events

    def get_matchable_events(self) -> List[Event]:
        """Return events whose scenarios are in our collected records."""
        result = []
        for ev in self._events:
            try:
                sid_int = _scenario_id_to_int(ev.scenario_id)
            except (ValueError, AttributeError):
                continue
            if sid_int in self._available_scenarios:
                result.append(ev)
        return result

    def analyze_event(self, event: Event) -> Optional[pd.DataFrame]:
        """Extract attention + risk trajectory over the event window.

        Args:
            event: Event from the catalog.

        Returns:
            DataFrame with columns: timestep, relative_t, attn_*, risk_*, event_phase.
            None if no records match.
        """
        try:
            sid_int = _scenario_id_to_int(event.scenario_id)
        except (ValueError, AttributeError):
            return None

        if sid_int not in self._available_scenarios:
            return None

        # Build window with padding
        w_start = max(0, event.window[0] - self.padding)
        w_end = event.window[1] + self.padding

        rows = []
        for t in range(w_start, w_end + 1):
            rec = self._lookup.get((sid_int, t))
            if rec is None:
                continue
            row = {
                "timestep": t,
                "relative_t": t - event.peak,
                "attn_sdc": rec.attn_sdc,
                "attn_agents": rec.attn_agents,
                "attn_roadgraph": rec.attn_roadgraph,
                "attn_lights": rec.attn_lights,
                "attn_gps": rec.attn_gps,
                "collision_risk": rec.collision_risk,
                "safety_risk": rec.safety_risk,
                "navigation_risk": rec.navigation_risk,
                "min_ttc": rec.min_ttc,
                "event_phase": self._classify_phase(t, event),
            }
            rows.append(row)

        return pd.DataFrame(rows) if rows else None

    def analyze_all_events(
        self,
        event_types: list[str] | None = None,
    ) -> Dict[str, pd.DataFrame]:
        """Analyze all matchable events.

        Args:
            event_types: Filter by event type strings (e.g. ['hazard_onset']).
                         None → all types.

        Returns:
            Dict mapping 'scenario_event_N' → trajectory DataFrame.
        """
        matched = self.get_matchable_events()
        if event_types:
            matched = [e for e in matched if e.event_type.value in event_types]

        results = {}
        for i, ev in enumerate(matched):
            key = f"s{_scenario_id_to_int(ev.scenario_id):03d}_{ev.event_type.value}_{i}"
            traj = self.analyze_event(ev)
            if traj is not None:
                results[key] = traj

        return results

    def summary(self) -> dict:
        """Compute summary statistics across all matchable events.

        Returns:
            Dict with:
                n_events_total: total events in catalog
                n_events_matched: events with collected records
                mean_agent_attn_increase: mean increase in attn_agents at event peak vs baseline
                pct_events_with_increase: % events where attn_agents increases at peak
        """
        matched = self.get_matchable_events()
        if not matched:
            return {
                "n_events_total": len(self._events),
                "n_events_matched": 0,
            }

        agent_attn_increases = []
        n_with_increase = 0

        for ev in matched:
            traj = self.analyze_event(ev)
            if traj is None or len(traj) < 3:
                continue

            # Baseline: mean attention before event onset
            pre = traj[traj["relative_t"] < -(self.padding // 2)]
            # Peak: timestep closest to event peak
            peak_row = traj[traj["timestep"] == ev.peak]

            if len(pre) == 0 or len(peak_row) == 0:
                continue

            baseline_agents = float(pre["attn_agents"].mean())
            peak_agents = float(peak_row["attn_agents"].iloc[0])
            delta = peak_agents - baseline_agents
            agent_attn_increases.append(delta)
            if delta > 0:
                n_with_increase += 1

        n = len(agent_attn_increases)
        return {
            "n_events_total": len(self._events),
            "n_events_matched": len(matched),
            "n_events_analyzed": n,
            "mean_agent_attn_increase": float(np.mean(agent_attn_increases)) if n > 0 else np.nan,
            "pct_events_with_increase": 100.0 * n_with_increase / n if n > 0 else np.nan,
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_phase(t: int, event: Event) -> str:
        """Classify timestep as pre / onset / peak / offset / post."""
        if t < event.onset:
            return "pre"
        elif t == event.peak:
            return "peak"
        elif t <= event.offset:
            return "during"
        else:
            return "post"
