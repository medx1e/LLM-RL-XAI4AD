# Concept Contract

Canonical reference for every driving concept in the extraction module.
Each concept is provably computable from the V-Max observation contract
(no privileged simulator information).

## V-Max Observation Contract (reference model: `womd_sac_road_perceiver_minimal_42`)

After `unflatten_features()`, batch dims omitted:

| Group | Shape | Feature layout (after mask removal) | Normalization |
|---|---|---|---|
| SDC trajectory | (1, 5, 7) | xy(0:2), vel_xy(2:4), yaw(4), length(5), width(6) | xy/70m, vel/30m·s⁻¹, size/70m, yaw raw |
| Other agents | (8, 5, 7) | same as SDC | same |
| Roadgraph | (200, 4) | xy(0:2), dir_xy(2:4) | xy/70m, dir raw |
| Traffic lights | (5, 5, 10) | xy(0:2), state_onehot(2:10) [8-dim] | xy/70m, one-hot |
| GPS path | (10, 2) | xy(0:2) | xy/70m |

Masks: last column of each group (except path) is extracted as a boolean
validity mask by `unflatten_features()`.

Coordinate frame: SDC-centric (ego at origin, facing +x).

---

## Phase 1 — Core Concepts

| Concept | Type | Formula | Source fields | Unit | Normalization (→[0,1]) | Validity mask | Obs-faithful? |
|---|---|---|---|---|---|---|---|
| `ego_speed` | continuous | `‖vel_xy[-1]‖ × 30` | sdc_features[vel_xy], sdc_mask | m/s | `/30` | sdc_mask[0,-1] | YES |
| `ego_acceleration` | continuous | `(speed[-1] − speed[-2]) / 0.1` | sdc_features[vel_xy], sdc_mask | m/s² | `(x+6)/12` | sdc_mask[0,-1] AND sdc_mask[0,-2] | YES |
| `dist_nearest_object` | continuous | `min_n ‖agent_xy[n,-1]‖ × 70` | agent_features[xy], agent_mask | m | `/70` | any(agent_mask[:,-1]) | YES |
| `num_objects_within_10m` | continuous | `Σ(‖agent_xy‖×70 < 10 ∧ valid)` | agent_features[xy], agent_mask | count | `/8` | always valid | YES |
| `traffic_light_red` | binary | `any(onehot[{0,3,6}] > 0.5 ∧ valid)` | tl_features[state], tl_mask | bool | identity | any(tl_mask[:,-1]) | YES |
| `dist_to_traffic_light` | continuous | `min_n ‖tl_xy[n,-1]‖ × 70` | tl_features[xy], tl_mask | m | `/70` | any(tl_mask[:,-1]) | YES |
| `heading_deviation` | continuous | `wrap(yaw − atan2(path_Δy, path_Δx))` | sdc_features[yaw], path_features[xy] | rad | `(x+π)/(2π)` | sdc_mask[0,-1] | YES |
| `progress_along_route` | continuous | `project((0,0) → path).arc_frac` | path_features[xy], sdc_mask | fraction | identity | sdc_mask[0,-1] | YES |

## Phase 2 — Extended Concepts

| Concept | Type | Formula | Source fields | Unit | Normalization (→[0,1]) | Validity mask | Obs-faithful? |
|---|---|---|---|---|---|---|---|
| `ttc_lead_vehicle` | continuous | `lead_Δx / max(ego_vx − lead_vx, ε)` capped 10 s | agent_features[xy,vel_xy], sdc_features[vel_xy], agent_mask | s | `/10` | lead vehicle exists (ahead, in lane, valid) | YES |
| `lead_vehicle_decelerating` | binary | `lead_speed[-2] − lead_speed[-1] > 0.5` | agent_features[xy,vel_xy], agent_mask | bool | identity | lead valid at t and t−1 | YES |
| `at_intersection` | binary | `any(‖tl_xy‖×70 < 25 ∧ valid)` | tl_features[xy], tl_mask | bool | identity | any(tl_mask[:,-1]) | YES (heuristic) |

## Deferred Concepts

| Concept | Reason deferred |
|---|---|
| `dist_to_road_edge` | Roadgraph features in default config do NOT include `types`; cannot distinguish road-edge points from lane-center or road-line points in the observation. Would require either adding `types` to the observation features or using a model trained with `element_types: [15, 16]` filtering AND verifying that filter applies. |
| `lane_curvature_ahead` | Requires lane-center polyline extraction; roadgraph `types` not available in observation. Same issue as above. |

## Notes

1. **One-hot traffic light state mapping** (after dropping unknown index 0):
   - Idx 0: ARROW_STOP (red), 1: ARROW_CAUTION (yellow), 2: ARROW_GO (green)
   - Idx 3: STOP (red), 4: CAUTION (yellow), 5: GO (green)
   - Idx 6: FLASHING_STOP (red), 7: FLASHING_CAUTION (yellow)
2. **Lead vehicle heuristic**: "ahead" = agent x > 0 in SDC frame; "in lane" = |agent y| < 2.5 m. These thresholds are conservative defaults.
3. **`at_intersection`** is a proxy: presence of nearby traffic light ≠ intersection geometry, but is the best available signal from the observation contract.
4. All coordinates are in the SDC-centric frame after `sdc_observation_from_state()` transformation.
5. Max normalization constants: `max_meters = 70`, `MAX_SPEED = 30`, `dt = 0.1`.
