# Tests for Module 6: AttentionGrounder

import numpy as np
import pytest

from xai.computation import attention_grounder

# Default encoder layout matching xai_config.yaml
LAYOUT = {
    "n_sdc_timesteps": 11,
    "num_objects": 64,
    "timestep_agent": 11,
    "roadgraph_top_k": 1000,
    "num_traffic_lights": 16,
    "tl_timesteps": 1,
    "gps_path_len": 80,
}

CONFIG = {"attention_layer_key": "cross_attn_0"}


def _make_attention_weights(
    num_heads: int = 2,
    num_latents: int = 64,
    hot_agent_idx: int = -1,
) -> dict:
    """Create mock attention weights.

    If hot_agent_idx >= 0, concentrate most attention on that agent's tokens.
    """
    n_sdc = LAYOUT["n_sdc_timesteps"]
    n_other = LAYOUT["num_objects"] * LAYOUT["timestep_agent"]
    n_rg = LAYOUT["roadgraph_top_k"]
    n_tl = LAYOUT["num_traffic_lights"] * LAYOUT["tl_timesteps"]
    n_gps = LAYOUT["gps_path_len"]
    total_tokens = n_sdc + n_other + n_rg + n_tl + n_gps

    # Uniform small attention
    attn = np.ones((num_heads, num_latents, total_tokens), dtype=np.float32) * 0.001

    if hot_agent_idx >= 0:
        start = n_sdc + hot_agent_idx * LAYOUT["timestep_agent"]
        end = start + LAYOUT["timestep_agent"]
        attn[:, :, start:end] = 1.0  # High attention on this agent

    return {"cross_attn_0": attn}


class TestAttentionGrounder:
    def test_no_threat_agents(self):
        weights = _make_attention_weights()
        result = attention_grounder.compute(weights, LAYOUT, [], CONFIG)
        assert result["grounding_score"] is None
        assert result["per_agent_breakdown"] == []

    def test_single_threat_high_attention(self):
        """Agent 5 is a threat and has high attention → high grounding score."""
        weights = _make_attention_weights(hot_agent_idx=5)
        threats = [{"agent_id": 5, "min_ttc": 0.0}]  # Severity = 1.0

        result = attention_grounder.compute(weights, LAYOUT, threats, CONFIG)

        assert result["grounding_score"] is not None
        assert result["grounding_score"] > 0.5
        assert len(result["per_agent_breakdown"]) == 1
        assert result["per_agent_breakdown"][0]["agent_id"] == 5
        assert result["per_agent_breakdown"][0]["severity"] == 1.0

    def test_single_threat_low_attention(self):
        """Agent 5 is a threat but attention is on agent 10 → low grounding."""
        weights = _make_attention_weights(hot_agent_idx=10)
        threats = [{"agent_id": 5, "min_ttc": 0.0}]

        result = attention_grounder.compute(weights, LAYOUT, threats, CONFIG)

        assert result["grounding_score"] is not None
        assert result["grounding_score"] < 0.1

    def test_severity_computation(self):
        """Verify severity = 1/(1+min_ttc)."""
        weights = _make_attention_weights(hot_agent_idx=3)
        threats = [{"agent_id": 3, "min_ttc": 4.0}]  # Severity = 1/5 = 0.2

        result = attention_grounder.compute(weights, LAYOUT, threats, CONFIG)

        assert len(result["per_agent_breakdown"]) == 1
        assert result["per_agent_breakdown"][0]["severity"] == pytest.approx(0.2, abs=0.001)

    def test_missing_attention_key(self):
        """If the layer key doesn't exist, return None."""
        result = attention_grounder.compute(
            {"wrong_key": np.zeros((2, 64, 100))},
            LAYOUT,
            [{"agent_id": 1, "min_ttc": 1.0}],
            CONFIG,
        )
        assert result["grounding_score"] is None

    def test_batch_dimension_handled(self):
        """4D attention tensor (with batch dim) should be handled."""
        n_sdc = LAYOUT["n_sdc_timesteps"]
        n_other = LAYOUT["num_objects"] * LAYOUT["timestep_agent"]
        n_rg = LAYOUT["roadgraph_top_k"]
        n_tl = LAYOUT["num_traffic_lights"] * LAYOUT["tl_timesteps"]
        n_gps = LAYOUT["gps_path_len"]
        total = n_sdc + n_other + n_rg + n_tl + n_gps

        attn_4d = np.ones((1, 2, 64, total), dtype=np.float32) * 0.001
        weights = {"cross_attn_0": attn_4d}
        threats = [{"agent_id": 0, "min_ttc": 1.0}]

        result = attention_grounder.compute(weights, LAYOUT, threats, CONFIG)
        assert result["grounding_score"] is not None
