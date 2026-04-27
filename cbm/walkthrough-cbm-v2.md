# CBM-V2 Pre-Training Implementation Walkthrough

## What Was Done

### 1. CBMConfig Refactor — Auto-Derived Index Lists

**File:** [config.py](file:///home/med1e/cbm/cbm_v1/config.py)

Replaced hardcoded `binary_concept_indices = (4, 9, 10)` and `continuous_concept_indices = (0,1,2,3,5,6,7,8)` with `@property` methods that derive these automatically from the concept registry based on `concept_phases`.

**Impact:** Adding new concepts now requires zero changes in `config.py`. The indices adapt automatically based on which phases are active. V1 backward compatibility verified — `concept_phases=(1,2)` produces the exact same indices as before.

---

### 2. Menger Curvature Geometry Helper

**File:** [geometry.py](file:///home/med1e/cbm/concepts/geometry.py#L93-L123)

Added `menger_curvature(path_xy, eps)` which computes curvature at each interior polyline point using the formula:
```
κ = 2|cross(P2−P1, P3−P2)| / (|P2−P1|·|P3−P2|·|P3−P1|)
```
Returns shape `(..., P-2)`. Pure JAX, JIT-safe, epsilon-guarded.

---

### 3. Four Phase 3 Extractors

**File:** [extractors.py](file:///home/med1e/cbm/concepts/extractors.py#L284-L378)

| Extractor | Type | Source | Output | Normalization |
|---|---|---|---|---|
| `path_curvature_max` | continuous | `path_features[xy]` | Max Menger curvature (1/m) | `/0.25`, clipped [0,1] |
| `path_net_heading_change` | continuous | `path_features[xy]` | Signed heading change, first→last segment (rad) | `(x+π)/(2π)` |
| `path_straightness` | continuous | `path_features[xy]` | chord/arc ratio [0,1] | identity |
| `heading_to_path_end` | continuous | `path_features[xy]` | `atan2(end_y, end_x)` in SDC frame (rad) | `(x+π)/(2π)` |

All are:
- Observation-faithful (only read `path_features`)
- JIT-safe (pure JAX, no Python control flow on values)
- Always valid (path always present; epsilon-guarded for degenerate cases)

---

### 4. Registry Registration + Normalization

**File:** [registry.py](file:///home/med1e/cbm/concepts/registry.py)

- 4 new `_reg(ConceptSchema(..., phase=3), E.fn)` entries
- 4 new normalization rules in `_normalize_concept()`:
  - `path_curvature_max`: `/0.25` (0.25 1/m ≈ 4m radius turn)
  - `path_net_heading_change` and `heading_to_path_end`: `(x+π)/(2π)`
  - `path_straightness`: identity (already [0,1])
- Default `phases` in `extract_all_concepts()` updated from `(1,2)` to `(1,2,3)`

---

### 5. Per-Concept Loss Logging

**File:** [cbm_sac_factory.py](file:///home/med1e/cbm/cbm_v1/cbm_sac_factory.py#L660-L706)

Added `_per_concept_losses()` which computes individual masked loss per concept and returns them as `concept_loss/<name>` entries in the training metrics dict. These will appear in TensorBoard as separate curves under `train/concept_loss/`.

---

## Smoke Test Results — 30/30 ✅

```
[A] Concept Registry
   1. Registry has 15 concepts                                 PASS
   2. Phase 3 has 4 concepts                                   PASS
   3. Phase 3 names correct                                    PASS

[B] CBMConfig Auto-Derivation
   4. V2 concept_names length = 15                             PASS
   5. V2 binary indices = (4, 9, 10)                           PASS
   6. V2 continuous indices = 12 items                         PASS
   7. V1 backward compat: 11 concepts                          PASS
   8. V1 backward compat: binary=(4,9,10)                      PASS

[C] Concept Extraction on Real Data
   9. Environment loads                                        PASS
  10. Concept output has 15 names                              PASS
  11. normalized shape ends with 15                            PASS
  12. raw shape ends with 15                                   PASS
  13. valid shape ends with 15                                 PASS
  14. All normalized values finite                             PASS
  15. All raw values finite (where valid)                      PASS
  16. Normalized in [0, 1]                                     PASS

[D] Phase 3 Concept Checks
  17. path_curvature_max >= 0                                  PASS   (0.000230)
  18. path_net_heading_change in [-pi, pi]                     PASS   (0.0008)
  19. path_straightness in [0, 1]                              PASS   (1.0000)
  20. heading_to_path_end in [-pi, pi]                         PASS   (-0.0036)
  21. Phase 3 all valid (no mask)                              PASS

[E] CBM Network Initialization (15 concepts)
  22. CBM network created (15 concepts)                        PASS
  23. Training state created                                   PASS
  24. Forward pass finite                                      PASS
  25. Concept predictions shape (15,)                          PASS
  26. Concepts in [0, 1]                                       PASS
  27. Concepts finite                                          PASS
  28. Concept loss finite                                      PASS   (0.0476)
  29. Per-concept losses: 15 entries                           PASS
  30. Per-concept losses all finite                            PASS
```

### Per-Concept Loss Values (initial, untrained)
```
concept_loss/ego_speed                   = 0.031234
concept_loss/ego_acceleration            = 0.000939
concept_loss/dist_nearest_object         = 0.036823
concept_loss/num_objects_within_10m      = 0.052651
concept_loss/traffic_light_red           = 0.000000  (masked)
concept_loss/dist_to_traffic_light       = 0.000000  (masked)
concept_loss/heading_deviation           = 0.004838
concept_loss/progress_along_route        = 0.100908
concept_loss/ttc_lead_vehicle            = 0.000000  (masked)
concept_loss/lead_vehicle_decelerating   = 0.000000  (masked)
concept_loss/at_intersection             = 0.000000  (masked)
concept_loss/path_curvature_max          = 0.116035
concept_loss/path_net_heading_change     = 0.001456
concept_loss/path_straightness           = 0.121823
concept_loss/heading_to_path_end         = 0.009275
```

## Files Changed

| File | Change | Lines |
|---|---|---|
| `cbm_v1/config.py` | Refactored to auto-derive indices from registry | Full rewrite |
| `concepts/geometry.py` | Added `menger_curvature()` | +30 |
| `concepts/extractors.py` | Added 4 Phase 3 extractors | +97 |
| `concepts/registry.py` | Registered Phase 3 + normalizations | +71 |
| `cbm_v1/cbm_sac_factory.py` | Added per-concept loss logging | +46 |
| `cbm_v1/smoke_test_v2.py` | New V2 verification test | New file |

**Zero changes to:**
- `cbm_trainer.py` — reads from config/registry dynamically
- `networks.py` — auto-sizes from tensor shape
- `concept_loss.py` — index-based, auto-adapts
- `adapters.py` — path_features already in ConceptInput
- V-Max source — firewall holds

## Blockers Before Short Frozen V2 Training

**None.** To launch a frozen V2 training run:

1. Set `num_concepts: 15` and `concept_phases: [1, 2, 3]` in the training YAML config
2. Everything else adapts automatically
3. Recommended: start with a 500-step smoke run on local 1GB data to confirm GPU memory fits
