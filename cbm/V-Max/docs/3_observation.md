# Observation Extractor Documentation

## Overview

The observation extractor in V-Max is responsible for transforming the simulator state into a structured observation vector suitable for a neural network. It supports flexible configuration to select which features to extract from objects, roadgraphs, traffic lights, and path targets.

## How the Extractor Works

- The extractor is initialized with configuration parameters specifying which features to extract and how many elements to include (e.g., number of closest objects, roadgraph points, etc.).
- For each simulation step, the extractor processes the simulator state and outputs a tuple of feature arrays:
  - SDC (Self-Driving Car) features
  - Other objects features
  - Roadgraph features
  - Traffic light features
  - Path target features
- These features are concatenated into a single observation vector (unless a non-flattened format is requested).
- The extractor supports normalization and masking, and can be extended to add noise or obstructions for robustness.

## How simulator_state is transformed into features

The transformation from `simulator_state` to features is performed by the observation extractor in several steps:

1. **SDC Observation Extraction**: The extractor first creates an SDC-centric observation from the full simulator state. This includes the SDC's own trajectory, other agents, roadgraph points, traffic lights, and path targets, all transformed into the SDC's local frame and clipped to the configured number of past steps and spatial region.

2. **Feature Selection and Filtering**:
   - **Objects**: The extractor computes the distance from the SDC to all other objects, selects the closest N objects (as configured), and extracts only the requested features (e.g., waypoints, velocity, yaw, etc.) for each. Features are normalized (e.g., positions by max meters, velocities by max speed) and padded if fewer than N objects are present.
   - **Roadgraph**: Roadgraph points are filtered by type (using integer IDs or meta-types), spatial region (meters_box or max_meters), and interval. The closest K points are selected, and only the requested features (e.g., waypoints, direction) are extracted and normalized.
   - **Traffic Lights**: The extractor finds the closest M traffic lights to the SDC, extracts the requested features (e.g., waypoints, state), and normalizes them.
   - **Path Target**: The extractor samples points along the planned path at a specified gap, up to a configured number of points, and extracts the requested features (e.g., waypoints).

3. **Stacking and Formatting**: For each feature group (SDC, objects, roadgraph, traffic lights, path target), the selected features are stacked into arrays of fixed shape, with masking applied for invalid or missing data.

4. **Concatenation and Output**: The arrays are either returned as a tuple (non-flattened) or concatenated into a single flat observation vector (flattened), ready for agent consumption.

5. **Optional Augmentations**: If enabled, the extractor can apply noise, masking, or obstructions to the features for robustness.

This process ensures that the agent receives a consistent, normalized, and configurable observation vector at each step, abstracting away the complexity of the raw simulator state.

## Configuration via YAML

The extractor is configured through the `observation_config` section in your config YAML file. Example:

```yaml
observation_type: vec
observation_config:
  obs_past_num_steps: 5
  objects:
    features:
      - waypoints
      - velocity
      - yaw
      - size
      - valid
    num_closest_objects: 8
  roadgraphs:
    features:
      - waypoints
      - direction
      # - types
      - valid
    element_types: [15, 16]  # or use meta-types like ["lane_center"]
    interval: 2
    max_meters: 70
    roadgraph_top_k: 200
    meters_box:
      front: 70
      back: 5
      left: 20
      right: 20
    max_num_lanes: 10
    max_num_points_per_lane: 20
  traffic_lights:
    features:
      - waypoints
      - state
      - valid
    num_closest_traffic_lights: 5
  path_target:
    features:
      - waypoints
    num_points: 3
    points_gap: 12
```

### Feature Options
- **objects.features**: Selects which features to extract for each object (e.g., waypoints, velocity, yaw, size, valid).
- **roadgraphs.features**: Features for roadgraph points (e.g., waypoints, direction, valid).
- **roadgraphs.element_types**: Filter roadgraph points by type (can use integer IDs or meta-types like "lane_center").
- **traffic_lights.features**: Features for traffic lights (e.g., waypoints, state, valid).
- **path_target.features**: Features for the path target (e.g., waypoints).

### Other Parameters
- **obs_past_num_steps**: Number of past steps to include in the observation.
- **num_closest_objects**: How many closest objects to include.
- **roadgraph_top_k**: Number of roadgraph points to include.
- **num_closest_traffic_lights**: Number of closest traffic lights to include.
- **num_points**: Number of points for the path target.
- **points_gap**: Gap between path target points.

## Usage

1. Set the `observation_type` and `observation_config` in your config YAML as shown above.
2. The environment will automatically use these settings to initialize the observation extractor.
3. The agent will receive observations as specified by your configuration.

## Extending the Extractor

To add new features or modify extraction logic, extend the extractor classes in `vmax/simulator/features/extractor/`. Update the config file to include your new features as needed.

## References
- See `vmax/simulator/features/extractor/vec_extractor.py` for implementation details.
- See `vmax/simulator/wrappers/observation.py` for how the extractor is used in the environment.
