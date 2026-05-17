# Copyright 2025 Valeo.

"""Smoke tests for ActivationHarvester telemetry logic (no GPU / model required)."""

from __future__ import annotations

import numpy as np
import pytest

from sae_interpretability.config import SAEConfig


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_semantic(
    n_agents: int = 8,
    ego_speed: float = 10.0,
    agent_speeds: np.ndarray | None = None,
    is_ahead: np.ndarray | None = None,
    distances: np.ndarray | None = None,
    valid: np.ndarray | None = None,
    ttc: np.ndarray | None = None,
) -> dict:
    """Build a minimal semantic features dict for testing."""
    rng = np.random.default_rng(0)
    if agent_speeds is None:
        agent_speeds = rng.uniform(5, 20, n_agents)
    if is_ahead is None:
        is_ahead = np.zeros(n_agents)
        is_ahead[0] = 1.0
    if distances is None:
        distances = rng.uniform(5, 50, n_agents)
    if valid is None:
        valid = np.ones(n_agents, dtype=bool)
    if ttc is None:
        ttc = np.full(n_agents, 4.0)
    return {
        'ego_speed': ego_speed,
        'agent_speeds': agent_speeds,
        'is_ahead': is_ahead,
        'distance_to_ego': distances,
        'valid': valid,
        'ttc': ttc,
        'sdc_index': 0,
        'positions_x': np.zeros(n_agents),
        'positions_y': np.zeros(n_agents),
        'closing_speed': np.zeros(n_agents),
        'is_left': np.zeros(n_agents),
    }


@pytest.fixture
def cfg():
    return SAEConfig()


@pytest.fixture
def harvester(cfg):
    """ActivationHarvester with setup() skipped (no model / env needed)."""
    from sae_interpretability.harvester import ActivationHarvester
    h = object.__new__(ActivationHarvester)
    h.sae_cfg = cfg
    # Minimal attributes referenced by _build_telemetry_row / helpers
    h._n_vehicles = 8
    h._n_timesteps = 5
    h._n_roadgraph = 200
    h._n_traffic_lights = 5
    return h


# ------------------------------------------------------------------
# is_ttc_critical
# ------------------------------------------------------------------

def test_ttc_critical_when_below_threshold(harvester):
    sem = _make_semantic(ttc=np.full(8, 1.0))
    row = harvester._build_telemetry_row(sem, None)
    assert row['is_ttc_critical'] is True
    assert row['min_ttc'] == pytest.approx(1.0)


def test_ttc_not_critical_above_threshold(harvester):
    sem = _make_semantic(ttc=np.full(8, 3.0))
    row = harvester._build_telemetry_row(sem, None)
    assert row['is_ttc_critical'] is False


def test_ttc_critical_exactly_at_threshold_is_false(harvester, cfg):
    sem = _make_semantic(ttc=np.full(8, cfg.ttc_critical_threshold))
    row = harvester._build_telemetry_row(sem, None)
    # threshold is strict <, so equal is NOT critical
    assert row['is_ttc_critical'] is False


def test_ttc_critical_uses_only_valid_agents(harvester):
    ttc = np.array([0.5, 0.8, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
    valid = np.array([False, False, True, True, True, True, True, True])
    sem = _make_semantic(ttc=ttc, valid=valid)
    row = harvester._build_telemetry_row(sem, None)
    # Only valid agents: all TTC = 4.0 → not critical
    assert row['is_ttc_critical'] is False
    assert row['min_ttc'] == pytest.approx(4.0)


# ------------------------------------------------------------------
# is_lead_vehicle_hard_braking
# ------------------------------------------------------------------

def test_hard_braking_false_when_no_prev(harvester):
    sem = _make_semantic()
    row = harvester._build_telemetry_row(sem, None)
    assert row['is_lead_vehicle_hard_braking'] is False


def test_hard_braking_detected(harvester, cfg):
    G = 9.81
    dt = cfg.harvest_dt
    threshold = cfg.hard_braking_g_threshold * G * dt  # ~0.392 m/s per step

    prev_speeds = np.ones(8) * 20.0
    curr_speeds = prev_speeds.copy()
    curr_speeds[0] -= threshold + 0.05  # just above threshold

    is_ahead = np.zeros(8)
    is_ahead[0] = 1.0
    dists = np.array([10.0] + [50.0] * 7)

    prev_sem = _make_semantic(agent_speeds=prev_speeds, is_ahead=is_ahead, distances=dists)
    curr_sem = _make_semantic(agent_speeds=curr_speeds, is_ahead=is_ahead, distances=dists)

    row = harvester._build_telemetry_row(curr_sem, prev_sem)
    assert row['is_lead_vehicle_hard_braking'] is True


def test_no_hard_braking_gentle_decel(harvester, cfg):
    G = 9.81
    dt = cfg.harvest_dt
    threshold = cfg.hard_braking_g_threshold * G * dt

    prev_speeds = np.ones(8) * 20.0
    curr_speeds = prev_speeds.copy()
    curr_speeds[0] -= threshold * 0.5  # well below threshold

    is_ahead = np.zeros(8)
    is_ahead[0] = 1.0
    dists = np.array([10.0] + [50.0] * 7)

    prev_sem = _make_semantic(agent_speeds=prev_speeds, is_ahead=is_ahead, distances=dists)
    curr_sem = _make_semantic(agent_speeds=curr_speeds, is_ahead=is_ahead, distances=dists)

    row = harvester._build_telemetry_row(curr_sem, prev_sem)
    assert row['is_lead_vehicle_hard_braking'] is False


def test_hard_braking_only_lead_vehicle(harvester, cfg):
    """Hard braking on a behind-agent should NOT trigger the flag."""
    G = 9.81
    dt = cfg.harvest_dt
    threshold = cfg.hard_braking_g_threshold * G * dt + 1.0  # definitely over threshold

    prev_speeds = np.ones(8) * 20.0
    curr_speeds = prev_speeds.copy()
    curr_speeds[3] -= threshold  # agent 3 is NOT ahead

    is_ahead = np.zeros(8)
    is_ahead[0] = 1.0  # only agent 0 is ahead; agent 3 is not
    dists = np.array([10.0, 50.0, 40.0, 5.0, 20.0, 25.0, 15.0, 35.0])

    prev_sem = _make_semantic(agent_speeds=prev_speeds, is_ahead=is_ahead, distances=dists)
    curr_sem = _make_semantic(agent_speeds=curr_speeds, is_ahead=is_ahead, distances=dists)

    row = harvester._build_telemetry_row(curr_sem, prev_sem)
    assert row['is_lead_vehicle_hard_braking'] is False


def test_no_hard_braking_when_no_ahead_agents(harvester):
    is_ahead = np.zeros(8)  # nobody is ahead
    prev_sem = _make_semantic(is_ahead=is_ahead)
    curr_sem = _make_semantic(is_ahead=is_ahead)
    row = harvester._build_telemetry_row(curr_sem, prev_sem)
    assert row['is_lead_vehicle_hard_braking'] is False


# ------------------------------------------------------------------
# Telemetry row schema
# ------------------------------------------------------------------

def test_telemetry_row_has_all_expected_keys(harvester):
    sem = _make_semantic()
    row = harvester._build_telemetry_row(sem, None)
    required = {
        'ego_speed', 'ego_x', 'ego_y',
        'min_ttc', 'min_agent_dist',
        'is_ttc_critical', 'is_lead_vehicle_hard_braking',
        'agent_dists', 'agent_ttcs', 'agent_speeds',
        'agent_valid', 'agent_closing_speed',
        'agent_is_ahead', 'agent_is_left',
    }
    assert required.issubset(row.keys())


def test_telemetry_scalar_types(harvester):
    sem = _make_semantic()
    row = harvester._build_telemetry_row(sem, None)
    assert isinstance(row['ego_speed'], float)
    assert isinstance(row['min_ttc'], float)
    assert isinstance(row['is_ttc_critical'], bool)
    assert isinstance(row['is_lead_vehicle_hard_braking'], bool)
