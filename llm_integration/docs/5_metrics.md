# V-Max Metrics Documentation

This document provides an overview of the custom metrics developed in V-Max. Each metric is designed to evaluate a specific aspect of autonomous driving behavior in simulation. The explanations below are intended to help new users and developers understand what each metric measures and how it is computed.

---

## 1. AtFaultCollisionMetric
**Purpose:**
- Detects collisions that are attributable to the ego vehicle (SDC).

**How it works:**
- Checks for overlaps between the ego vehicle and other objects.
- Classifies collisions as rear, front, or lateral based on relative positions.
- Considers the ego vehicle at fault if:
  - It collides with a vulnerable road user (VRU),
  - It is moving and collides at the front or laterally while on multiple lanes,
  - Or if it collides with a stopped object.

**What it measures:**
- The number of at-fault collisions caused by the ego vehicle.

---

## 2. ComfortMetric
**Purpose:**
- Evaluates the comfort of the ego vehicle's ride according to nuPlan standards.

**How it works:**
- Computes lateral and longitudinal acceleration, jerk, yaw rate, and yaw acceleration over recent timesteps.
- Checks if these values stay within comfort thresholds.
- Returns 1 if all thresholds are respected, 0 otherwise.
- Also provides a continuous reward version for RL.

**What it measures:**
- Whether the ego vehicle's motion is comfortable for passengers.

---

## 3. DrivingDirectionComplianceMetric
**Purpose:**
- Detects if the ego vehicle is driving into oncoming traffic (wrong way).

**How it works:**
- Compares the ego vehicle's heading with the direction of nearby lane centers.
- Checks if the vehicle is closer to misaligned (opposite direction) lanes than aligned ones.

**What it measures:**
- The distance the ego vehicle travels in the wrong direction.

---

## 4. OnMultipleLanesMetric
**Purpose:**
- Detects if the ego vehicle is straddling or occupying multiple lanes.

**How it works:**
- Measures the distance from each corner of the ego vehicle to the closest lane center.
- If any corner is farther than half the lane width plus a margin, the vehicle is considered on multiple lanes.

**What it measures:**
- The time and distance the ego vehicle spends on multiple lanes.

---

## 5. ProgressRatioMetric
**Purpose:**
- Measures how much progress the ego vehicle has made compared to the expert (logged) trajectory.

**How it works:**
- For each point in the ego trajectory, finds the closest point on the expert trajectory.
- Computes the ratio of the maximum progress made to the total expert trajectory length.

**What it measures:**
- The normalized progress of the ego vehicle (1.0 = completed expert route).

---

## 6. RunRedLightMetric
**Purpose:**
- Detects if the ego vehicle has run a red light.

**How it works:**
- Checks if the ego vehicle crosses into a lane with a red traffic light between timesteps.
- Compares the lane IDs before and after the step and checks if the new lane is closed (red).

**What it measures:**
- Returns 1 if a red light was run, 0 otherwise.

---

## 7. OffRouteMetric
**Purpose:**
- Detects if the ego vehicle is off its planned route.

**How it works:**
- Measures the distance from the ego vehicle to the closest on-route and off-route paths.
- Considers the vehicle off-route if it is too far from the on-route path or closer to an off-route path.

**What it measures:**
- The distance to the closest valid on-route path (0 if on route).

---

## 8. SpeedLimitViolationMetric
**Purpose:**
- Measures if and by how much the ego vehicle exceeds the speed limit.

**How it works:**
- Infers the speed limit from the roadgraph and expert trajectory.
- Computes the difference between the ego speed and the speed limit.
- Returns the amount by which the speed limit is exceeded (0 if within limit).

**What it measures:**
- The magnitude of speed limit violations in meters per second.

---

## 9. TimeToCollisionMetric (TTC)
**Purpose:**
- Predicts the time until a potential collision with another object, assuming constant speed and heading.

**How it works:**
- Simulates future positions of all objects over a time horizon.
- Checks for overlaps (collisions) at each timestep.
- Returns the minimum time to collision for the ego vehicle.

**What it measures:**
- The shortest predicted time to collision with any object ahead.

---

## 10. Collector and Aggregators
**Purpose:**
- Collects and aggregates metrics over episodes for reporting and evaluation.

**How it works:**
- Splits metrics by episode boundaries.
- Applies custom aggregation functions (mean, max, sum, etc.) for each metric.
- Computes overall scores (e.g., nuPlan and V-Max aggregate scores) based on weighted combinations of metrics.

**What it measures:**
- Provides summary statistics and aggregate scores for simulation runs.

---

## Utility Functions
Several utility functions are used by the metrics to compute distances, angles, lane associations, and filtering (e.g., Savitzky-Golay smoothing, lane center detection, etc.). These are implemented in `metrics/utils.py` and are not metrics themselves, but support the calculations above.

---

## Adding New Metrics
To add a new metric:
1. Implement a new class inheriting from `abstract_metric.AbstractMetric`.
2. Register it in `metrics/__init__.py` and, if needed, in the collector.
3. Document its purpose and usage in this file.

---

For further details, refer to the code in the `vmax/simulator/metrics/` directory.
