"""AttentionTimestepCollector: run episode, collect (obs, attention, state) per step.

Strategy:
  1. Run VMaxAdapter to get ScenarioData (episodes, states, actions, TTC, etc.)
     with store_raw_obs=True so we have all flat observations.
  2. Replay raw observations through model.get_attention() — no policy re-running,
     just pure forward passes through the encoder+attention heads.
  3. Aggregate attention by token category.
  4. Merge with risk metrics to build List[TimestepRecord].

This avoids modifying VMaxAdapter's episode loop while giving us full attention
data for every timestep.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from typing import List

from event_mining.integration.vmax_adapter import VMaxAdapter
from reward_attention.config import (
    TOKEN_RANGES,
    N_AGENTS,
    TOKENS_PER_ENTITY,
    AnalysisConfig,
    TimestepRecord,
    get_agent_token_range,
)
from reward_attention.risk_metrics import RiskComputer


class AttentionTimestepCollector:
    """Run scenarios and collect per-timestep (attention, risk) data.

    Args:
        config: AnalysisConfig with thresholds and layer selection.
    """

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self._adapter = VMaxAdapter(store_raw_obs=True)

    def collect(self, model, scenario, scenario_id: int) -> List[TimestepRecord]:
        """Collect TimestepRecords for one scenario.

        Args:
            model: PerceiverWrapper (from posthoc_xai.load_model).
            scenario: Raw scenario object from data_gen.
            scenario_id: Integer identifier for this scenario.

        Returns:
            List of TimestepRecord, one per timestep.
        """
        # 1. Run full episode, get ScenarioData (with raw observations)
        sd = self._adapter.extract_scenario_data(
            model, scenario, scenario_id=str(scenario_id)
        )

        if sd.total_steps == 0 or sd.raw_observations is None:
            return []

        # 2. Compute risk metrics from ScenarioData
        risk = RiskComputer.from_scenario_data(sd, self.config)

        # 3. Replay ALL observations in one batched forward pass (GPU-efficient)
        records = []
        T = sd.total_steps
        raw_obs = sd.raw_observations  # (T, obs_dim)

        # Determine layer key
        layer_key = (
            "cross_attn_avg"
            if self.config.attention_layer == "avg"
            else f"cross_attn_layer_{self.config.attention_layer}"
        )

        # Batched attention extraction: one pass for all T observations
        obs_batch = jnp.array(raw_obs)  # (T, obs_dim)
        # model.forward handles batched input; capture_intermediates works per-batch
        out = model.forward(obs_batch)
        attn_dict_batch = out.attention  # {'cross_attn_avg': (T, 16, 280), ...}

        if attn_dict_batch is None:
            return []

        attn_mat_batch = attn_dict_batch.get(layer_key, attn_dict_batch.get("cross_attn_avg"))
        if attn_mat_batch is None:
            return []

        attn_all = np.array(attn_mat_batch)  # (T, 16, 280)

        for t in range(T):
            # attn_np: (16, 280)
            attn_np = attn_all[t] if attn_all.ndim == 3 else attn_all

            # Average over queries → (280,) = importance per token
            token_importance = attn_np.mean(axis=0)  # (280,)

            # Aggregate by category (sum over tokens, then normalize)
            cat_sums = {}
            for cat, (s, e) in TOKEN_RANGES.items():
                cat_sums[cat] = float(token_importance[s:e].sum())
            total = sum(cat_sums.values()) + 1e-12
            attn_fracs = {cat: v / total for cat, v in cat_sums.items()}

            # Per-agent attention (sum over agent's 5 tokens, not normalized)
            per_agent = []
            for i in range(N_AGENTS):
                s, e = get_agent_token_range(i)
                per_agent.append(float(token_importance[s:e].sum()))
            # Normalize by total so values are comparable to attn_fracs
            per_agent_norm = [v / (total + 1e-12) for v in per_agent]

            # Attention to nearest agent
            nearest_id = int(sd.nearest_agent_id[t]) if sd.nearest_agent_id is not None else -1
            attn_nearest = per_agent_norm[nearest_id] if 0 <= nearest_id < N_AGENTS else 0.0

            # Attention to threat agent (lowest TTC)
            threat_id, attn_threat = self._get_threat_attention(
                per_agent_norm, sd.ttc[t] if sd.ttc is not None else None,
                np.array(sd.other_agents_valid[t]) if sd.other_agents_valid is not None else None
            )

            # Build record
            rec = TimestepRecord(
                scenario_id=scenario_id,
                timestep=t,
                attn_sdc=attn_fracs.get("sdc", 0.0),
                attn_agents=attn_fracs.get("other_agents", 0.0),
                attn_roadgraph=attn_fracs.get("roadgraph", 0.0),
                attn_lights=attn_fracs.get("traffic_lights", 0.0),
                attn_gps=attn_fracs.get("gps_path", 0.0),
                attn_per_agent=per_agent_norm,
                attn_to_nearest=attn_nearest,
                attn_to_threat=attn_threat,
                collision_risk=float(risk.collision_risk[t]),
                safety_risk=float(risk.safety_risk[t]),
                navigation_risk=float(risk.navigation_risk[t]),
                behavior_risk=float(risk.behavior_risk[t]),
                min_ttc=float(risk.min_ttc[t]),
                accel=float(sd.ego_accel[t]) if sd.ego_accel is not None else 0.0,
                steering=float(sd.ego_steering[t]) if sd.ego_steering is not None else 0.0,
                ego_speed=float(
                    np.sqrt(sd.ego_vx[t] ** 2 + sd.ego_vy[t] ** 2)
                ) if sd.ego_vx is not None else 0.0,
                num_valid_agents=int(
                    np.sum(sd.other_agents_valid[t])
                ) if sd.other_agents_valid is not None else 0,
                nearest_agent_id=nearest_id,
                threat_agent_id=threat_id,
                is_collision_step=bool(sd.step_collision[t]) if sd.step_collision is not None else False,
                is_offroad_step=bool(sd.step_offroad[t]) if sd.step_offroad is not None else False,
            )
            records.append(rec)

        return records

    @staticmethod
    def _get_threat_attention(
        per_agent_norm: list[float],
        ttc_row: np.ndarray | None,
        valid_row: np.ndarray | None,
    ) -> tuple[int, float]:
        """Return (agent_idx, attention) for the agent with lowest TTC."""
        if ttc_row is None or len(ttc_row) == 0:
            return -1, 0.0

        ttc_arr = np.array(ttc_row)
        if valid_row is not None:
            # Mask invalid agents with large TTC
            ttc_arr = np.where(valid_row, ttc_arr, np.inf)

        threat_id = int(np.argmin(ttc_arr))
        if np.isinf(ttc_arr[threat_id]):
            return -1, 0.0

        attn = per_agent_norm[threat_id] if threat_id < len(per_agent_norm) else 0.0
        return threat_id, float(attn)
