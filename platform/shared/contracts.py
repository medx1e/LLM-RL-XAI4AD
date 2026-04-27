"""Platform-level shared data contracts.

This module defines the types that cross the boundary between the research
packages (bev_visualizer, posthoc_xai) and the Streamlit platform.

Importing this module does NOT trigger Waymax or JAX imports.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# Signature: overlay_fn(ax, step) -> None
# Called by bev_component after the base BEV is drawn.
OverlayFn = Callable[["matplotlib.axes.Axes", int], None]


# ---------------------------------------------------------------------------
# PlatformScenarioArtifact
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PlatformScenarioArtifact:
    """Platform-side wrapper around a cached rollout.

    Holds a bev_visualizer.ScenarioData object plus optional extra payloads
    that individual tabs may need.  The core ScenarioData is never mutated
    here; this wrapper only adds fields.

    Attributes
    ----------
    scenario_data
        A ``bev_visualizer.rollout_engine.ScenarioData`` instance.
        Contains ego/agent trajectories, frame_states, rewards, dones.
    model_key
        The PLATFORM_MODELS key that identifies the source model.
    scenario_idx
        Which scenario index (0-indexed) from the dataset.
    raw_observations
        Per-timestep flat observation vectors, shape ``(T, obs_size)``.
        Present only when the artifact was created by a curation script
        that captured observations alongside the rollout.  ``None`` when
        loaded from a legacy ScenarioData pickle.
    interesting_timesteps
        Zero-indexed steps flagged by the curator as worth highlighting
        (e.g. near-miss onset, hard brake).  The Post-hoc tab surfaces
        these as quick-jump anchors on the timeline.
    notes
        Short human-readable description of the scenario (e.g. "dense
        intersection, ego yields to left-turning vehicle").  Displayed
        next to the BEV player.
    metadata
        Free-form dict for module-specific extras.  Callers should use
        well-known keys documented in each tab's adapter.
    """

    scenario_data: Any  # bev_visualizer.rollout_engine.ScenarioData
    model_key: str
    scenario_idx: int
    raw_observations: Optional[np.ndarray] = None  # shape (T, obs_size)
    interesting_timesteps: Optional[list[int]] = None
    notes: str = ""
    metadata: dict = dataclasses.field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return len(self.scenario_data.frame_states)

    @property
    def has_raw_observations(self) -> bool:
        return self.raw_observations is not None


# ---------------------------------------------------------------------------
# XAI readiness sentinel
# ---------------------------------------------------------------------------

class XAINotReadyError(Exception):
    """Raised when an explanation cannot be produced for a scenario.

    The message should be user-displayable (shown in the Streamlit tab).
    """
