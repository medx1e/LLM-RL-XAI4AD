"""RiskComputer: ScenarioData → per-step continuous risk metrics in [0, 1].

All metrics use normalized coordinate space (V-Max ego-relative frame).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from event_mining.events.base import ScenarioData
from reward_attention.config import AnalysisConfig, TTC_THRESHOLD, ACCEL_THRESHOLD


@dataclass
class RiskArrays:
    """Per-step risk arrays computed from one scenario.

    All arrays have shape (T,) where T = sd.total_steps.
    """

    collision_risk: np.ndarray   # clip(1 - min_TTC / threshold, 0, 1)
    safety_risk: np.ndarray      # max(collision_risk, offroad_flag)
    navigation_risk: np.ndarray  # binary offroad proxy
    behavior_risk: np.ndarray    # |accel| / threshold, clipped to [0, 1]
    min_ttc: np.ndarray          # minimum TTC across valid agents, per step


class RiskComputer:
    """Compute continuous risk metrics from ScenarioData."""

    @staticmethod
    def from_scenario_data(
        sd: ScenarioData,
        config: AnalysisConfig | None = None,
    ) -> RiskArrays:
        """Compute risk arrays from a ScenarioData object.

        Args:
            sd: Populated ScenarioData from VMaxAdapter.
            config: AnalysisConfig with thresholds (uses module defaults if None).

        Returns:
            RiskArrays with per-step values.
        """
        ttc_threshold = config.ttc_threshold if config else TTC_THRESHOLD
        accel_threshold = config.accel_threshold if config else ACCEL_THRESHOLD

        T = sd.total_steps

        # ----------------------------------------------------------------
        # Minimum TTC per step (over valid agents)
        # ----------------------------------------------------------------
        if sd.ttc is not None and sd.ttc.shape[0] == T:
            ttc_arr = np.array(sd.ttc, dtype=float)  # (T, N_agents)

            # Mask invalid agents with large TTC
            if sd.other_agents_valid is not None:
                valid = np.array(sd.other_agents_valid, dtype=bool)  # (T, N_agents)
                ttc_arr = np.where(valid, ttc_arr, np.inf)

            min_ttc = np.min(ttc_arr, axis=1)  # (T,)
        else:
            min_ttc = np.full(T, np.inf)

        # ----------------------------------------------------------------
        # Collision risk: clip(1 - min_TTC / threshold, 0, 1)
        # When TTC=inf → risk=0; TTC=0 → risk=1; TTC=threshold → risk=0
        # ----------------------------------------------------------------
        finite_mask = np.isfinite(min_ttc)
        collision_risk = np.zeros(T, dtype=float)
        collision_risk[finite_mask] = np.clip(
            1.0 - min_ttc[finite_mask] / ttc_threshold, 0.0, 1.0
        )

        # ----------------------------------------------------------------
        # Navigation risk: binary offroad proxy
        # Uses step_offroad flag from environment metrics.
        # ----------------------------------------------------------------
        if sd.step_offroad is not None and len(sd.step_offroad) == T:
            navigation_risk = np.array(sd.step_offroad, dtype=float)
        else:
            navigation_risk = np.zeros(T, dtype=float)

        # ----------------------------------------------------------------
        # Safety risk: max(collision_risk, navigation_risk)
        # ----------------------------------------------------------------
        safety_risk = np.maximum(collision_risk, navigation_risk)

        # ----------------------------------------------------------------
        # Behavior risk: |accel| / threshold, clipped [0, 1]
        # ----------------------------------------------------------------
        if sd.ego_accel is not None and len(sd.ego_accel) == T:
            behavior_risk = np.clip(
                np.abs(np.array(sd.ego_accel, dtype=float)) / accel_threshold,
                0.0,
                1.0,
            )
        else:
            behavior_risk = np.zeros(T, dtype=float)

        return RiskArrays(
            collision_risk=collision_risk,
            safety_risk=safety_risk,
            navigation_risk=navigation_risk,
            behavior_risk=behavior_risk,
            min_ttc=min_ttc,
        )
