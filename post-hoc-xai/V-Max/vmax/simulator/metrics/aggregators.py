# Copyright 2025 Valeo.


import numpy as np

from vmax.simulator import constants


def final(x):
    return x[-1]


def nuplan_speed_compliance(speed_violation, dt=constants.TIME_DELTA, max_overspeed_value_threshold=2.23):
    # Implementation of nuPlan speed compliance score, based on duration and depth of the speed limit violations
    scenario_duration = dt * (len(speed_violation) - 1)
    max_ov_threshold = max(max_overspeed_value_threshold, 1e-3)  # Safety if max_overspeed_value_threshold is set to 0
    if scenario_duration == 0:
        return 1.0
    violation_loss = dt * np.sum(speed_violation) / (max_ov_threshold * scenario_duration)

    return max(0.0, 1.0 - violation_loss)


def time_spent(x, dt=constants.TIME_DELTA):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return dt * np.sum(x > 0)


def multiple_lanes_aggregator(x, dt=constants.TIME_DELTA):
    t_spent = time_spent(x, dt)
    score = 1.0 * (t_spent <= 3.4) + 0.5 * (t_spent > 3.0) * (t_spent <= 5.7)
    return score


def nuplan_driving_direction_compliance(driving_direction_compliance, a=2.0, b=6.0):
    # Implementation of nuPlan's driving direction compliance score, based on distance drove into oncoming traffic
    distance = np.sum(driving_direction_compliance)
    score = 1.0 * (distance <= a) + 0.5 * (distance > a) * (distance <= b)
    return score


def all_within_bound(x, min_value=-np.inf, max_value=np.inf):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    res = np.logical_and(x >= min_value, x <= max_value)
    return np.float32(np.all(res))


def final_within_bound(x, min_value=-np.inf, max_value=np.inf):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    res = x[-1] >= min_value and x[-1] <= max_value
    return np.float32(res)


def nuplan_aggregate_score(metrics_dict: dict):
    avg_score = (
        5 * metrics_dict["ttc_within_bound"]
        + 5 * np.clip(metrics_dict["progress_ratio"], 0, 1)
        + 4 * metrics_dict["nuplan_speed_compliance"]
        + 2 * metrics_dict["comfort"]
    )

    avg_score /= 16

    mul_score = (
        (1 - metrics_dict["at_fault_collision"])
        * (1 - metrics_dict["offroad"])
        * metrics_dict["making_progress"]
        * metrics_dict["nuplan_driving_direction_compliance"]
    )

    return avg_score * mul_score


def vmax_aggregate_score(metrics_dict: dict):
    avg_score = (
        5 * metrics_dict["ttc_within_bound"]
        + 5 * np.clip(metrics_dict["progress_ratio"], 0, 1)
        + 4 * metrics_dict["nuplan_speed_compliance"]
        + 3 * metrics_dict["multiple_lanes_score"]
        + 2 * metrics_dict["comfort"]
    )

    avg_score /= 19

    mul_score = (
        (1 - metrics_dict["at_fault_collision"])
        * (1 - metrics_dict["offroad"])
        * (1 - metrics_dict["run_red_light"])
        * metrics_dict["making_progress"]
        * metrics_dict["nuplan_driving_direction_compliance"]
    )

    return avg_score * mul_score
