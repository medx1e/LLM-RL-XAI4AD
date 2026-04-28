# Copyright 2025 Valeo.


"""Module for added metrics in the simulator."""

from waymax.metrics import abstract_metric

from .at_fault_collision import AtFaultCollisionMetric
from .collector import collect
from .comfort import ComfortMetric
from .driving_direction_compliance import DrivingDirectionComplianceMetric
from .on_multiple_lanes import OnMultipleLanesMetric
from .progress_ratio import ProgressRatioMetric
from .red_light import RunRedLightMetric, get_id_red_for_sdc
from .route import OffRouteMetric
from .speed_limit import SpeedLimitViolationMetric, infer_speed_limit_from_simulator_state
from .ttc import TimeToCollisionMetric


__all__ = [
    "AtFaultCollisionMetric",
    "AtFaultCollisionMetric",
    "ComfortMetric",
    "ComfortMetric",
    "DesiredSpeedMetric",
    "DrivingDirectionComplianceMetric",
    "FollowLaneMetric",
    "FollowLaneMetric",
    "OffRouteMetric",
    "OnMultipleLanesMetric",
    "ProgressRatioMetric",
    "RunRedLightMetric",
    "RunRedLightMetric",
    "SpeedLimitViolationMetric",
    "TimeToCollisionMetric",
    "TimeToCollisionMetric",
    "collect",
    "compute_distance_lane_to_follow",
    "desired_speed_constant_ttc",
    "get_id_red_for_sdc",
    "infer_speed_limit_from_simulator_state",
]

_VMAX_METRICS_REGISTRY: dict[str, abstract_metric.AbstractMetric] = {
    "run_red_light": RunRedLightMetric(),
    "ttc": TimeToCollisionMetric(),
    "at_fault_collision": AtFaultCollisionMetric(),
    "comfort": ComfortMetric(),
    "speed_limit": SpeedLimitViolationMetric(),
    "on_multiple_lanes": OnMultipleLanesMetric(),
    "driving_direction_compliance": DrivingDirectionComplianceMetric(),
    "progress_ratio_nuplan": ProgressRatioMetric(),
}


def get_metrics(metric_name: str) -> abstract_metric.AbstractMetric:
    if metric_name in _VMAX_METRICS_REGISTRY:
        return _VMAX_METRICS_REGISTRY[metric_name]
    else:
        raise ValueError(f"Metric {metric_name} not registered.")
