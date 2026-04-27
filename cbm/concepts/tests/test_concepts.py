"""Tests for the concept extraction module.

Covers: shapes, masks, numerical sanity, batch compatibility, JIT safety.
"""

import sys
sys.path.insert(0, "/home/med1e/cbm")

import jax
import jax.numpy as jnp
import pytest

from concepts.types import ConceptInput, ObservationConfig
from concepts.registry import extract_all_concepts, CONCEPT_REGISTRY
from concepts.adapters import structured_to_concept_input
from concepts import extractors as E
from concepts.geometry import l2_norm, wrap_angle, project_onto_path


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

DEFAULT_CFG = ObservationConfig()


def _make_dummy_input(
    batch_shape: tuple[int, ...] = (),
    cfg: ObservationConfig = DEFAULT_CFG,
    seed: int = 0,
) -> ConceptInput:
    """Create a synthetic ConceptInput with plausible values."""
    rng = jax.random.PRNGKey(seed)
    keys = jax.random.split(rng, 10)

    T = cfg.obs_past_num_steps
    N = cfg.num_closest_objects
    R = cfg.roadgraph_top_k
    L = cfg.num_closest_traffic_lights
    P = cfg.num_target_path_points

    def _shape(*s):
        return batch_shape + s

    # SDC: normalised xy near origin, small vel, yaw~0, size~0.07
    sdc_feat = jnp.concatenate([
        jax.random.uniform(keys[0], _shape(1, T, 2), minval=-0.05, maxval=0.05),  # xy
        jax.random.uniform(keys[1], _shape(1, T, 2), minval=0.3, maxval=0.5),     # vel
        jax.random.uniform(keys[2], _shape(1, T, 1), minval=-0.1, maxval=0.1),    # yaw
        jnp.full(_shape(1, T, 1), 0.07),                                           # length
        jnp.full(_shape(1, T, 1), 0.03),                                           # width
    ], axis=-1)
    sdc_mask = jnp.ones(_shape(1, T), dtype=bool)

    # Agents: some close, some far
    agent_xy = jax.random.uniform(keys[3], _shape(N, T, 2), minval=-0.5, maxval=0.5)
    agent_vel = jax.random.uniform(keys[4], _shape(N, T, 2), minval=0.0, maxval=0.5)
    agent_yaw = jax.random.uniform(keys[5], _shape(N, T, 1), minval=-0.3, maxval=0.3)
    agent_feat = jnp.concatenate([
        agent_xy, agent_vel, agent_yaw,
        jnp.full(_shape(N, T, 1), 0.07),
        jnp.full(_shape(N, T, 1), 0.03),
    ], axis=-1)
    agent_mask = jnp.ones(_shape(N, T), dtype=bool)
    # Mark last 2 agents as invalid
    if N >= 3:
        agent_mask = agent_mask.at[..., -2:, :].set(False)

    # Roadgraph
    rg_feat = jnp.concatenate([
        jax.random.uniform(keys[6], _shape(R, 2), minval=-1, maxval=1),   # xy
        jax.random.uniform(keys[7], _shape(R, 2), minval=-1, maxval=1),   # dir_xy
    ], axis=-1)
    rg_mask = jnp.ones(_shape(R,), dtype=bool)
    rg_mask = rg_mask.at[..., -50:].set(False)

    # Traffic lights: 5 lights, one red, one green
    tl_state = jnp.zeros(_shape(L, T, 8))
    tl_state = tl_state.at[..., 0, :, 3].set(1.0)  # first TL: STOP (red)
    tl_state = tl_state.at[..., 1, :, 5].set(1.0)  # second TL: GO (green)
    tl_feat = jnp.concatenate([
        jax.random.uniform(keys[8], _shape(L, T, 2), minval=-0.3, maxval=0.3),  # xy
        tl_state,
    ], axis=-1)
    tl_mask = jnp.ones(_shape(L, T), dtype=bool)
    tl_mask = tl_mask.at[..., 3:, :].set(False)  # only first 3 TLs valid

    # Path: 10 points going forward
    path_x = jnp.linspace(0.05, 0.8, P)
    path_y = jnp.zeros(P)
    path_feat = jnp.broadcast_to(
        jnp.stack([path_x, path_y], axis=-1),
        _shape(P, 2),
    )

    return ConceptInput(
        sdc_features=sdc_feat,
        sdc_mask=sdc_mask,
        agent_features=agent_feat,
        agent_mask=agent_mask,
        roadgraph_features=rg_feat,
        roadgraph_mask=rg_mask,
        tl_features=tl_feat,
        tl_mask=tl_mask,
        path_features=path_feat,
        config=cfg,
    )


# -----------------------------------------------------------------------
# Shape tests
# -----------------------------------------------------------------------

class TestShapes:

    def test_single_sample_output_shape(self):
        inp = _make_dummy_input()
        out = extract_all_concepts(inp)
        n_concepts = len(out.names)
        assert out.raw.shape == (n_concepts,)
        assert out.normalized.shape == (n_concepts,)
        assert out.valid.shape == (n_concepts,)

    def test_batched_output_shape(self):
        inp = _make_dummy_input(batch_shape=(4,))
        out = extract_all_concepts(inp)
        n_concepts = len(out.names)
        assert out.raw.shape == (4, n_concepts)
        assert out.valid.shape == (4, n_concepts)

    def test_concept_count_matches_registry(self):
        inp = _make_dummy_input()
        out = extract_all_concepts(inp, phases=(1, 2))
        assert len(out.names) == len(CONCEPT_REGISTRY)

    def test_phase1_only(self):
        inp = _make_dummy_input()
        out = extract_all_concepts(inp, phases=(1,))
        for schema in out.schemas:
            assert schema.phase == 1


# -----------------------------------------------------------------------
# Mask tests
# -----------------------------------------------------------------------

class TestMasks:

    def test_ego_speed_valid_when_sdc_valid(self):
        inp = _make_dummy_input()
        _, valid = E.ego_speed(inp)
        assert bool(valid)

    def test_ego_speed_invalid_when_sdc_masked(self):
        inp = _make_dummy_input()
        inp.sdc_mask = jnp.zeros_like(inp.sdc_mask, dtype=bool)
        _, valid = E.ego_speed(inp)
        assert not bool(valid)

    def test_dist_nearest_invalid_when_no_agents(self):
        inp = _make_dummy_input()
        inp.agent_mask = jnp.zeros_like(inp.agent_mask, dtype=bool)
        _, valid = E.dist_nearest_object(inp)
        assert not bool(valid)

    def test_tl_red_invalid_when_no_tl(self):
        inp = _make_dummy_input()
        inp.tl_mask = jnp.zeros_like(inp.tl_mask, dtype=bool)
        _, valid = E.traffic_light_red(inp)
        assert not bool(valid)


# -----------------------------------------------------------------------
# Numerical sanity tests
# -----------------------------------------------------------------------

class TestNumericalSanity:

    def test_ego_speed_non_negative(self):
        inp = _make_dummy_input()
        speed, _ = E.ego_speed(inp)
        assert float(speed) >= 0.0

    def test_ego_speed_reasonable_range(self):
        inp = _make_dummy_input()
        speed, _ = E.ego_speed(inp)
        assert float(speed) < 35.0  # max_speed is 30, but norm clamp allows slight overshoot

    def test_dist_nearest_positive(self):
        inp = _make_dummy_input()
        dist, valid = E.dist_nearest_object(inp)
        if bool(valid):
            assert float(dist) >= 0.0

    def test_heading_deviation_in_pi_range(self):
        inp = _make_dummy_input()
        dev, _ = E.heading_deviation(inp)
        assert float(dev) >= -jnp.pi - 0.01
        assert float(dev) <= jnp.pi + 0.01

    def test_progress_in_01(self):
        inp = _make_dummy_input()
        prog, _ = E.progress_along_route(inp)
        assert 0.0 <= float(prog) <= 1.0

    def test_traffic_light_red_detected(self):
        """Dummy input has TL index 0 set to STOP (red)."""
        inp = _make_dummy_input()
        red, valid = E.traffic_light_red(inp)
        assert bool(valid)
        assert float(red) == 1.0

    def test_num_objects_count_correct(self):
        inp = _make_dummy_input()
        # With default config, 6 out of 8 agents are valid (last 2 masked)
        # All 6 may or may not be within 10m depending on random positions
        count, valid = E.num_objects_within_radius(inp, radius_m=1000.0)
        assert bool(valid)
        assert float(count) == 6.0  # 6 valid agents, all within 1000m

    def test_ttc_capped_at_10(self):
        inp = _make_dummy_input()
        ttc, _ = E.ttc_lead_vehicle(inp)
        assert float(ttc) <= 10.0

    def test_normalized_output_in_01(self):
        inp = _make_dummy_input()
        out = extract_all_concepts(inp)
        assert jnp.all(out.normalized >= -0.01)
        assert jnp.all(out.normalized <= 1.01)


# -----------------------------------------------------------------------
# Batch compatibility
# -----------------------------------------------------------------------

class TestBatch:

    def test_batched_leading_dim(self):
        """extract_all_concepts works with a leading batch dimension."""
        inp = _make_dummy_input(batch_shape=(3,))
        out = extract_all_concepts(inp)
        assert out.raw.shape[0] == 3


# -----------------------------------------------------------------------
# JIT compatibility
# -----------------------------------------------------------------------

class TestJIT:

    def test_jit_extract_all(self):
        """JIT the JAX-array parts of extract_all_concepts."""
        inp = _make_dummy_input()

        @jax.jit
        def run(sdc_f, sdc_m, ag_f, ag_m, rg_f, rg_m, tl_f, tl_m, path_f):
            ci = ConceptInput(
                sdc_features=sdc_f, sdc_mask=sdc_m,
                agent_features=ag_f, agent_mask=ag_m,
                roadgraph_features=rg_f, roadgraph_mask=rg_m,
                tl_features=tl_f, tl_mask=tl_m,
                path_features=path_f,
                config=DEFAULT_CFG,
            )
            out = extract_all_concepts(ci)
            # Return only JAX arrays (names/schemas are static)
            return out.raw, out.normalized, out.valid

        raw, norm, valid = run(
            inp.sdc_features, inp.sdc_mask,
            inp.agent_features, inp.agent_mask,
            inp.roadgraph_features, inp.roadgraph_mask,
            inp.tl_features, inp.tl_mask,
            inp.path_features,
        )
        assert raw.shape[-1] == len(CONCEPT_REGISTRY)

    def test_jit_individual_concept(self):
        inp = _make_dummy_input()

        @jax.jit
        def run(sdc_f, sdc_m, ag_f, ag_m, rg_f, rg_m, tl_f, tl_m, path_f):
            ci = ConceptInput(
                sdc_features=sdc_f, sdc_mask=sdc_m,
                agent_features=ag_f, agent_mask=ag_m,
                roadgraph_features=rg_f, roadgraph_mask=rg_m,
                tl_features=tl_f, tl_mask=tl_m,
                path_features=path_f,
                config=DEFAULT_CFG,
            )
            return E.ego_speed(ci)

        speed, valid = run(
            inp.sdc_features, inp.sdc_mask,
            inp.agent_features, inp.agent_mask,
            inp.roadgraph_features, inp.roadgraph_mask,
            inp.tl_features, inp.tl_mask,
            inp.path_features,
        )
        assert speed.shape == ()


# -----------------------------------------------------------------------
# Geometry unit tests
# -----------------------------------------------------------------------

class TestGeometry:

    def test_l2_norm(self):
        v = jnp.array([3.0, 4.0])
        assert abs(float(l2_norm(v)) - 5.0) < 0.01

    def test_wrap_angle(self):
        assert abs(float(wrap_angle(jnp.array(3.5)))) < jnp.pi + 0.01
        assert abs(float(wrap_angle(jnp.array(-3.5)))) < jnp.pi + 0.01

    def test_project_onto_straight_path(self):
        path = jnp.array([[0.0, 0.0], [10.0, 0.0]])
        point = jnp.array([5.0, 2.0])
        lat, prog = project_onto_path(point, path)
        assert abs(float(lat) - 2.0) < 0.1
        assert abs(float(prog) - 0.5) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
