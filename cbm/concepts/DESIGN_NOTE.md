# Design Note — Concept Extraction Module

## 1. V-Max Observation Contract

The V-Max pipeline transforms raw simulator state into a fixed-size flat
observation vector through this chain:

```
SimulatorState → sdc_observation_from_state() → VecFeaturesExtractor → flat vector
```

Key transformations:
- **SDC-centric frame**: all coordinates rotated so the ego vehicle is at (0,0) facing +x.
- **Closest-K selection**: only the K nearest objects/TLs/roadgraph points are retained.
- **Normalization**: xy / max_meters, vel / MAX_SPEED, sizes / max_meters, categorical → one-hot.
- **Validity mask**: last feature column per group is a boolean mask, extracted by `unflatten_features()`.

**Concrete shapes** (reference model `womd_sac_road_perceiver_minimal_42`):

| Group | Shape (after unflatten, no batch) | Feature dims |
|---|---|---|
| SDC trajectory | (1, 5, 7) | xy(2) vel_xy(2) yaw(1) length(1) width(1) |
| Other agents | (8, 5, 7) | same |
| Roadgraph | (200, 4) | xy(2) dir_xy(2) |
| Traffic lights | (5, 5, 10) | xy(2) state_onehot(8) |
| GPS path | (10, 2) | xy(2) |

Constants: `max_meters=70`, `MAX_SPEED=30`, `dt=0.1s`, `obs_past_num_steps=5`.

## 2. How Concept Extraction Aligns with the Contract

The concept module has a strict firewall:

1. **`adapters.py`** is the ONLY file that touches V-Max internals (the `unflatten_fn`).
2. **`ConceptInput`** is a plain dataclass of JAX arrays matching the shapes above.
3. **Extractors** receive `ConceptInput` only — they never access simulator state.

This means every concept is provably computable from what the encoder sees.
If a concept requires information not in the observation (e.g., roadgraph element types),
it is **not implemented** — it's listed as deferred with an explanation.

## 3. Safely Supported Concepts (11 total)

### Phase 1 (8 concepts)
| Concept | Status | Notes |
|---|---|---|
| ego_speed | SAFE | Direct from vel_xy norm |
| ego_acceleration | SAFE | Finite difference of speed; requires T≥2 |
| dist_nearest_object | SAFE | L2 norm of agent xy |
| num_objects_within_10m | SAFE | Thresholded count |
| traffic_light_red | SAFE | One-hot indices {0,3,6} map to red states |
| dist_to_traffic_light | SAFE | L2 norm of TL xy |
| heading_deviation | SAFE | Ego yaw vs path tangent angle |
| progress_along_route | SAFE | Projection onto path polyline |

### Phase 2 (3 concepts)
| Concept | Status | Notes |
|---|---|---|
| ttc_lead_vehicle | SAFE | Lead = closest ahead + in-lane agent; TTC from closing speed |
| lead_vehicle_decelerating | SAFE | Speed difference across timesteps for lead vehicle |
| at_intersection | SAFE (heuristic) | Proxy: any valid TL within 25m |

## 4. Deferred Concepts

| Concept | Reason |
|---|---|
| dist_to_road_edge | Roadgraph `types` feature is NOT in the observation config. Cannot distinguish road edges from lane centers. |
| lane_curvature_ahead | Same issue — requires lane center identification. |

These could be enabled if the model were retrained with `types` in roadgraph features,
or if `element_types: [15, 16]` filtering is verified AND the observation contains only road-edge points.

## 5. Blockers and Ambiguities

1. **Roadgraph type information**: The default observation config includes `[waypoints, direction, valid]`
   but NOT `types`. This prevents semantic roadgraph queries. This is the main blocker for
   `dist_to_road_edge` and `lane_curvature_ahead`.

2. **`progress_along_route` at t=0**: Always returns 0.0 at the initial timestep because the SDC
   is at the origin and path points start from `points_gap` steps ahead. This is expected and
   will produce meaningful values during rollouts (t > 0).

3. **`ttc_lead_vehicle` sensitivity**: The "in lane" heuristic (|y| < 2.5m) may miss lead vehicles
   on curved roads. This is a reasonable first approximation given the SDC-centric frame.

4. **`at_intersection`** is a proxy, not ground truth. The observation contract has no intersection
   label. Traffic-light proximity is the best available signal.
