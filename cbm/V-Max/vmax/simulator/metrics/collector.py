# Copyright 2025 Valeo.


"""Module to collect and aggregate simulation metrics."""

from functools import partial

import numpy as np

from vmax.simulator.metrics import aggregators


_metrics_operands = {
    "ep_len_mean": aggregators.final,  # cumulative number of steps
    "ep_rew_mean": aggregators.final,  # cumulative reward at each timestep
    # Waymax metrics
    "log_divergence": np.mean,  # average L2 distance from expert log trajectory
    "offroad": np.max,  # 1.0 if offroad then stop as termination
    "overlap": np.max,  # 1.0 if collision then stop as termination
    "sdc_off_route": np.mean,  # average distance to the closest on-route path
    "sdc_progression": aggregators.final,
    "progress_ratio_nuplan": {
        "progress_ratio": aggregators.final,
        "making_progress": partial(aggregators.final_within_bound, min_value=0.2),
    },
    "sdc_wrongway": np.mean,  # average distance for wrong way driving
    # V-Max metrics
    "run_red_light": np.max,  # 1.0 if red light run then terminate
    "ttc": {
        "min_ttc": np.min,
        "average_ttc": np.mean,
        "ttc_within_bound": partial(aggregators.all_within_bound, min_value=0.95),
    },
    "at_fault_collision": np.max,  # 1.0 if collision is at fault
    "comfort": np.min,  # comfort metric value (0 if thresholds violated)
    "speed_limit": {
        "max_overspeed_m_per_s": np.max,
        "max_overspeed_km_per_h": lambda x: 3.6 * np.max(x),
        "nuplan_speed_compliance": aggregators.nuplan_speed_compliance,
    },
    "on_multiple_lanes": {
        "distance_on_multiple_lanes": np.sum,
        "time_on_multiple_lanes": aggregators.time_spent,
        "multiple_lanes_score": aggregators.multiple_lanes_aggregator,
    },
    "driving_direction_compliance": {
        "distance_into_oncoming_traffic": np.sum,
        "nuplan_driving_direction_compliance": aggregators.nuplan_driving_direction_compliance,
    },
}


def collect(metrics: dict, key_metric: str) -> dict:
    """Aggregate the episode metrics from the simulation.

    Args:
        metrics: The simulation metrics.
        key_metric: The key used to split episodes.

    Returns:
        Aggregated metrics per episode.

    """
    split_indices = _get_split_indices(metrics[key_metric])
    episode_metrics = {}

    for key, metric in metrics.items():
        _key = key.split("/")[-1]
        operand = _metrics_operands.get(_key, np.mean)

        list_metric = np.split(metric, split_indices)

        if not isinstance(operand, dict):
            episode_values = np.array([operand(l_metric) for l_metric in list_metric if len(l_metric) > 0])
            episode_metrics[key] = np.mean(episode_values)
        else:
            for sub_key, sub_operand in operand.items():
                episode_values = np.array([sub_operand(l_metric) for l_metric in list_metric if len(l_metric) > 0])
                episode_metrics[sub_key] = np.mean(episode_values)

    episode_metrics["nuplan_aggregate_score"] = aggregators.nuplan_aggregate_score(episode_metrics)
    episode_metrics["vmax_aggregate_score"] = aggregators.vmax_aggregate_score(episode_metrics)

    return episode_metrics


def _get_split_indices(steps: np.ndarray) -> np.ndarray:
    """Determine the indices at which to split the metric for each episode.

    Args:
        steps: An array representing the step metric over time.

    Returns:
        An array of indices for splitting episodes.

    """
    indices = np.argwhere(steps == 1) - 1
    indices = np.squeeze(indices) + 1

    return indices
