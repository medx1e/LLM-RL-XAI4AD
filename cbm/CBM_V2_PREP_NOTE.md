# CBM-V2 Preparation Note

*Author: generated during CBM-V1→V2 transition discussion, 2026-04-05*
*Status: design/planning — no code has been written yet*

---

## Part A — How the `/concepts` Module Works

### Module Structure

```
concepts/
  schema.py        ConceptSchema + ConceptType enum
  types.py         ObservationConfig, ConceptInput, ConceptOutput dataclasses
  adapters.py      ONLY file that knows V-Max internals; builds ConceptInput from flat obs
  normalize.py     Slice constants (OBJ_XY, TL_STATE, PATH_XY, ...) + denorm helpers
  geometry.py      Pure JAX geometry: l2_norm, wrap_angle, project_onto_path, angle_between
  extractors.py    One function per concept; receives ConceptInput, returns (raw, valid)
  registry.py      _REGISTRY (OrderedDict) + extract_all_concepts(); _normalize_concept()
```

### Data Flow: V-Max Observation → Concept Values

```
flat obs (1655-d)
    │
    ▼ adapters.unflatten_fn()          ← adapters.py (ONLY file touching V-Max internals)
    │
    ▼ ConceptInput (structured JAX arrays)
    │   sdc_features  (1, T=5, F=7)     xy | vel_xy | yaw | length | width (all normalized)
    │   agent_features (8, T=5, F=7)    same layout
    │   roadgraph_features (200, 4)     xy | dir_xy
    │   tl_features (5, T=5, F=10)      xy | state_onehot(8)
    │   path_features (10, 2)           xy (normalized by max_meters=70)
    │   + sdc_mask, agent_mask, rg_mask, tl_mask (boolean validity)
    │
    ▼ extractors.py (one fn per concept)
    │   receives: ConceptInput
    │   returns: (raw, valid)           raw = denormalized real-world value
    │
    ▼ registry._normalize_concept()     concept-specific [0,1] normalization
    │
    ▼ ConceptOutput
        names: tuple of str
        raw:        (batch, C) — physical units (m, m/s, rad, bool)
        normalized: (batch, C) — all in [0,1]
        valid:      (batch, C) — boolean mask per-concept-per-sample
```

### Key Design Invariants

1. **Structural firewall**: `adapters.py` is the single crossing point into V-Max internals. Everything else works on `ConceptInput` only — guaranteed obs-faithful.
2. **Registry is ordered**: `CONCEPT_REGISTRY` is an `OrderedDict`; column order in `ConceptOutput` exactly matches insertion order. The CBM loss relies on this ordering via integer index lists in `CBMConfig`.
3. **`extract_all_concepts(inp, phases=(1,2))`**: the `phases` kwarg is already there to filter concepts by phase (1, 2, ...). Phase 3 is trivially added.
4. **JIT-safe**: all extractors use only JAX ops and `...` broadcasting — no Python control flow on data.
5. **Masks are validity-aware, not imputed**: invalid concepts (e.g. TTC when no lead vehicle exists) are masked out in the loss, not set to a dummy value.

### Normalization Pattern

Every concept follows the same pattern:
- Extractor returns `(raw, valid)` in physical units
- `registry._normalize_concept(raw, schema)` maps to [0,1] using a **hardcoded per-name rule**
- The normalized value is what goes into the bottleneck and into the BCE/Huber loss

The `_normalize_concept` function uses `if name == "..."` branching — it is the one place that needs updating when adding new concepts.

### `geometry.py` available utilities

| Function | What it does |
|---|---|
| `l2_norm(xy, axis)` | Euclidean norm with eps floor |
| `angle_between_vectors(v1, v2)` | Signed angle, shape (...,) |
| `wrap_angle(angle)` | Wrap to [-π, π] |
| `project_onto_path(point, path_xy)` | → (lateral_dist, progress_fraction) |

These are all reusable as-is for V2 path concepts.

---

## Part B — How CBM-V1 Consumes Concepts

### Where Concepts Enter Training

In `cbm_trainer.py`:
```python
def concept_targets_fn(observations):
    inp = observation_to_concept_input(observations, unflatten_fn, concept_config)
    out = extract_all_concepts(inp)
    return out.normalized, out.valid
```
This function is called **inside the SGD step** on every gradient update. The result is passed directly into `concept_loss(predicted, targets, valid, config)`.

### Concept Loss (concept_loss.py)

```python
L_concept = (BCE_masked_binary + Huber_masked_continuous) / total_valid
L_total = L_SAC_policy + lambda_concept * L_concept
```

- Binary concepts (indices in `config.binary_concept_indices = (4, 9, 10)`): BCE
- Continuous concepts (indices in `config.continuous_concept_indices = (0,1,2,3,5,6,7,8)`): Huber(δ=1.0)
- Both masked by `valid` — invalid entries contribute 0 to the loss
- Averaged over total valid entries (not total concepts), so concepts with low validity don't dominate

### How concept_dim is wired into the network (networks.py)

```
ConceptHead: Dense(128 → 64) + ReLU + Dense(64 → num_concepts) + Sigmoid → c (B, num_concepts)
Actor FC:    Dense(num_concepts → 64) + ReLU + Dense(64 → 32) + ReLU → then policy_output
Critic FC:   Dense(num_concepts + action_size → 64) + ... → Q-value
```

The actor and critic first layer size is **automatically equal to `num_concepts`** because `Flax Dense` infers `in_features` from the actual input tensor at initialization. There is **no hardcoded `11` anywhere in the network math**. Only `num_concepts` in `ConceptHead.num_concepts` matters.

### What changes if concept count grows

| Location | What changes | Effort |
|---|---|---|
| `concepts/extractors.py` | Add new extractor functions | Low |
| `concepts/registry.py` | Register new concepts (schema + fn) | Low |
| `concepts/registry.py` `_normalize_concept` | Add name-specific normalization rule | Low |
| `cbm_v1/config.py` `CBMConfig` | Bump `num_concepts`, add to `concept_names`, update `binary_concept_indices`, `continuous_concept_indices` | **Medium** (hardcoded index lists) |
| `cbm_v1/cbm_sac_factory.py` | Update `num_concepts` call passed to `make_networks` | Low (comes from config) |
| YAML configs | Bump `num_concepts: 11 → 15` | Trivial |
| Network architecture | Nothing — ConceptHead + Actor/Critic auto-size | Zero |
| Concept loss | Nothing — loss is index-list-based | Zero (indices auto-derived from config) |

**The single most brittle point**: `binary_concept_indices` and `continuous_concept_indices` in `CBMConfig` are hardcoded integer tuples. These must be updated manually and kept in sync with the registry insertion order. This is the main friction point for V2.

---

## Part C — CBM-V2 Concept Set Design

### Decision: First-Wave V2 Concepts (Phase 3)

Based on the `path_spatial_concepts_analysis.md` analysis:

| # | Concept | Keep for V2? | Reasoning |
|---|---------|-------------|-----------|
| `path_curvature_max` | **YES** | Best representative of curvature cluster (higher std than mean). Captures sharpest turn constraint. |
| `path_net_heading_change` | **YES** | Unique signed signal (left turn vs. right turn vs straight). NOT redundant with curvature_max (which is unsigned). |
| `path_straightness` | **YES** | Interpretable dimensionless ratio [0,1]. More stable than arc_length (low CoV). Captures global path complexity. |
| `heading_to_path_end` | **YES** | Angle from ego to the route endpoint. Varies at t=0 (std=0.57 rad) and is dynamic during rollout. Complementary to `heading_deviation` (first path segment vs. endpoint). |
| `dist_to_path` | **DEFER to V2b** | Mean ~5.83m at t=0 (systematic gap offset). Meaningful only during rollout. Add in a second wave once V2a is validated. |

**V2a (first wave, 4 concepts)**: `path_curvature_max`, `path_net_heading_change`, `path_straightness`, `heading_to_path_end`

**V2b (optional, second wave)**: `dist_to_path` — only after confirming the training loop is stable with dynamic concepts and the rollout signal is clean.

### Non-Redundancy Verification

| New concept | Existing concept | Overlap? |
|---|---|---|
| `path_curvature_max` | `heading_deviation` | No — heading_deviation measures current ego alignment; curvature_max describes road geometry ahead |
| `path_net_heading_change` | `heading_deviation` | Partial — deviation is ego-to-path angle error; net_heading_change is where the road itself is going. Diverge on curved roads exactly when it matters. |
| `path_straightness` | `progress_along_route` | No — progress is longitudinal position along the route; straightness measures geometric complexity |
| `heading_to_path_end` | `heading_deviation` | Complementary — deviation uses first path segment tangent; path_end uses the final destination angle |
| `dist_to_path` | `progress_along_route` | Complementary — progress is longitudinal, dist_to_path is lateral deviation |

### Schemas for V2 Concepts

| Concept | Type | Formula | Source | Unit | Normalization | Mask | Phase |
|---|---|---|---|---|---|---|---|
| `path_curvature_max` | continuous | max Menger curvature over interior path points | path_features[xy] | 1/m | `/0.25`, clamped [0,1] | always valid (path present) | 3 |
| `path_net_heading_change` | continuous | signed heading: last segment angle − first segment angle | path_features[xy] | rad | `(x + π) / (2π)` | always valid | 3 |
| `path_straightness` | continuous | chord_length / arc_length | path_features[xy] | ratio | identity, already [0,1] | always valid | 3 |
| `heading_to_path_end` | continuous | `atan2(end_y − 0, end_x − 0)` in SDC frame | path_features[xy] | rad | `(x + π) / (2π)` | always valid | 3 |

All four are:
- **Observation-faithful** — computed only from `path_features[xy]` which is in the encoder's input
- **JIT-safe** — pure JAX, no Python control flow on values
- **No validity mask needed** — path always present; epsilon guards handle degenerate arc_length=0

### Implementation of Menger Curvature for `path_curvature_max`

For three consecutive points P1, P2, P3:
```
curvature = 2 * |cross(P2-P1, P3-P2)| / (|P2-P1| * |P3-P2| * |P3-P1|)
```
Apply over interior points (indices 1 to P-2), take `jnp.max`. Epsilon guards on lengths prevent division by zero for degenerate (collinear or zero-length) paths.

---

## Part D — Safe Extension Plan

### Recommended Extension Order

```
Step 1: Add extractors (no CBM changes yet)
  → concepts/extractors.py: add path_curvature_max, path_net_heading_change,
                              path_straightness, heading_to_path_end
  → concepts/registry.py: register with phase=3 schemas
  → concepts/registry.py _normalize_concept: add 4 new name-specific normalizations

Step 2: Verify extractors standalone (no training needed)
  → run scripts/analyze_path_spatial_concepts.py or a quick unit test
  → confirm shapes, ranges, JIT compilation

Step 3: Bump CBM config for V2
  → cbm_v1/config.py: num_concepts=15, update concept_names, binary_concept_indices,
                        continuous_concept_indices (add indices 11,12,13,14 to continuous list)
  → YAML configs: num_concepts: 15, concept_phases: [1, 2, 3]

Step 4: Smoke test with new concept count
  → cbm_v1/smoke_test.py: will auto-detect 15 concepts via registry
  → verify: concept targets shape=(batch,15), loss is finite, no OOM

Step 5: Frozen training run at small scale (500 steps)
  → Use config_womd_frozen_short.yaml with num_concepts: 15
  → Verify concept_loss per-concept (add per-concept logging first)

Step 6: Full frozen training (2M steps)
  → Then joint mode, same as V1 progression
```

### Files Requiring Changes (with scope)

| File | Change | Scope |
|---|---|---|
| `concepts/extractors.py` | Add 4 new extractor functions | ~60 lines |
| `concepts/registry.py` | Register 4 new schemas + 4 normalization rules | ~60 lines |
| `cbm_v1/config.py` | CBMConfig defaults for V2 | ~10 lines |
| YAML configs | `num_concepts: 15`, `concept_phases: [1,2,3]` | trivial |
| `cbm_v1/smoke_test.py` | Update expected concept count | 1 line |
| `concepts/geometry.py` | Add `menger_curvature` helper | ~15 lines |

**Zero changes needed in:**
- `cbm_trainer.py` — reads `num_concepts` from config
- `networks.py` — auto-sizes from actual tensor
- `adapters.py` — path_features already in ConceptInput
- `concept_loss.py` — index-based, auto-adapts
- V-Max source — firewall holds

### Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| **Hardcoded index lists in `CBMConfig`** (`binary_concept_indices`, `continuous_concept_indices`) are error-prone | **Medium** | For V2, auto-derive these from the registry at CBMConfig construction time (read concept_type from schema). This eliminates the manual bookkeeping. |
| **`_normalize_concept` uses `if name == "..."` chains** — easy to forget | Low | Keep it, but add an assertion in `extract_all_concepts` that every registered name has a normalization rule |
| **Loss balance shifts with more concepts** | Medium | Adding 4 continuous concepts to 8 existing continuous ones increases the Huber denominator. `lambda_concept` may need retuning. Start with the same value and monitor `train/concept_loss` split |
| **`path_straightness` always [0,1]** — identity normalization correct | Low | Confirmed from analysis: range is [0.111, 1.0] but norm_range already [0,1] |
| **`path_curvature_max` = 0 for many straight scenarios** | Low | This is meaningful (straight road). Cap at 0.25 (1/m), clip to [0,1] |
| **`dist_to_path` systematic offset at t=0** | Medium if added | Defer to V2b. At t=0, mean=5.83m, encoder will learn this is a "reset artifact". Only add after validating rollout behavior. |
| **Concept head capacity** | Low | Current head is 128→64→`num_concepts`. Going 11→15 is minor — capacity is not a bottleneck |

### Recommendation: Auto-Derive Index Lists

The most important V2 prep change before implementing concepts is to refactor `CBMConfig` so that `binary_concept_indices` and `continuous_concept_indices` are **derived from the registry** at config construction time, not hardcoded. Proposed change:

```python
# In cbm_v1/config.py, replace hardcoded index lists with:
@property
def binary_concept_indices(self) -> tuple[int, ...]:
    from concepts.registry import CONCEPT_REGISTRY
    from concepts.schema import ConceptType
    active = [n for n, (s, _) in CONCEPT_REGISTRY.items()
              if s.phase in self.concept_phases]
    return tuple(i for i, n in enumerate(active)
                 if CONCEPT_REGISTRY[n][0].concept_type == ConceptType.BINARY)

@property
def continuous_concept_indices(self) -> tuple[int, ...]:
    from concepts.registry import CONCEPT_REGISTRY
    from concepts.schema import ConceptType
    active = [n for n, (s, _) in CONCEPT_REGISTRY.items()
              if s.phase in self.concept_phases]
    return tuple(i for i, n in enumerate(active)
                 if CONCEPT_REGISTRY[n][0].concept_type == ConceptType.CONTINUOUS)
```

This makes adding any number of new concepts zero-change in `config.py`. The `frozen=True` dataclass constraint means these become `@property` methods instead of fields, which is a minor structural change.

### Per-Concept Logging (add before V2 training)

Before launching a V2 training run, add per-concept concept loss to TensorBoard. Currently only the aggregated `train/concept_loss` is logged. Proposal: in `cbm_trainer.py`, after the SGD step compute per-concept losses (easy: re-run the loss per index) and log as `train/concept_loss/ego_speed`, etc. This makes V2 debugging much easier.

---

## Part E — Master Concept Table (V1 + V2 Plan)

| # | Concept | Keep for V2? | Type | Source Tensors | Normalization / Mask | Redundancy / Notes |
|---|---------|---|---|---|---|---|
| 0 | `ego_speed` | ✅ | continuous | sdc_features[vel_xy], sdc_mask | `/30`, clip [0,1]; valid=sdc_mask[0,-1] | Core signal |
| 1 | `ego_acceleration` | ✅ | continuous | sdc_features[vel_xy], sdc_mask | `(x+6)/12`, clip [0,1]; valid=sdc_mask[0,-1] AND [-2] | Core signal |
| 2 | `dist_nearest_object` | ✅ | continuous | agent_features[xy], agent_mask | `/70`, clip [0,1]; valid=any(agent_mask[:,-1]) | Core signal |
| 3 | `num_objects_within_10m` | ✅ | continuous | agent_features[xy], agent_mask | `/8`, clip [0,1]; always valid | Core signal |
| 4 | `traffic_light_red` | ✅ | **binary** | tl_features[state], tl_mask | identity {0,1}; valid=any(tl_mask[:,-1]) | High accuracy (V1: 99%+) |
| 5 | `dist_to_traffic_light` | ✅ | continuous | tl_features[xy], tl_mask | `/70`, clip [0,1]; valid=any(tl_mask[:,-1]) | Slightly correlated with at_intersection |
| 6 | `heading_deviation` | ✅ | continuous | sdc_features[yaw], path_features[xy] | `(x+π)/(2π)`; valid=sdc_mask[0,-1] | First-segment deviation; complement to heading_to_path_end |
| 7 | `progress_along_route` | ✅ | continuous | path_features[xy], sdc_mask | identity [0,1]; valid=sdc_mask[0,-1] | Poor R² in V1 frozen; expected, should improve in joint |
| 8 | `ttc_lead_vehicle` | ✅ | continuous | agent_features[xy,vel_xy], sdc_features[vel_xy] | `/10`, clip [0,1]; valid=lead exists | 33% valid at t=0; essential for safety |
| 9 | `lead_vehicle_decelerating` | ✅ | **binary** | agent_features[xy,vel_xy], agent_mask | identity; valid=lead valid at t and t-1 | 33% valid; acc=99.5% in V1 |
| 10 | `at_intersection` | ✅ | **binary** | tl_features[xy], tl_mask | identity; valid=any(tl_mask[:,-1]) | Heuristic proxy (TL proximity); acc=87.4% in V1 |
| 11 | `path_curvature_max` | **V2** | continuous | path_features[xy] | `/0.25`, clip [0,1]; always valid | New; ~path geometry cluster representative |
| 12 | `path_net_heading_change` | **V2** | continuous | path_features[xy] | `(x+π)/(2π)`; always valid | New; signed: L-turn vs R-turn vs straight |
| 13 | `path_straightness` | **V2** | continuous | path_features[xy] | identity [0,1]; always valid | New; chord/arc ratio |
| 14 | `heading_to_path_end` | **V2** | continuous | path_features[xy] | `(x+π)/(2π)`; always valid | New; dynamics during rollout; complement to heading_deviation |
| — | `dist_to_path` | **V2b** | continuous | path_features[xy] | `/10`, clip [0,1]; always valid | Defer — systematic offset at t=0; add after V2a validated |

**V2a total**: 15 concepts (11 V1 + 4 Phase 3)
**All Phase 3 concepts are continuous** — `binary_concept_indices` stays (4, 9, 10)

---

## Part F — Recommended Implementation Order

1. **Refactor CBMConfig index lists to auto-derive from registry** ← most impactful, do before anything else
2. **Add `menger_curvature` to `concepts/geometry.py`**
3. **Add 4 Phase 3 extractors to `concepts/extractors.py`**
4. **Register them with schemas in `concepts/registry.py` + add 4 normalization rules**
5. **Add per-concept loss logging in `cbm_trainer.py`**
6. **Bump `num_concepts: 15` in `cbm_v1/config.py` defaults and YAML configs**
7. **Smoke test (500 steps frozen mode) to verify phase 3 concepts are extracted correctly**
8. **Full frozen V2 training run on 1GB data**
9. **Evaluate: compare V1 vs V2 concept accuracy and task performance**
10. **Joint V2 training** (same pattern as V1)
11. **(V2b)** Add `dist_to_path` if rollout analysis confirms it's learning during RL

---

## Open Questions

1. **Should CBMConfig be unfrozen** (remove `frozen=True`) to allow `@property` for auto-derived index lists? This is a minor structural change but removes the biggest fragility point.
2. **Is `lambda_concept = 0.1` still the right weight with 15 concepts?** Adding 4 continuous concepts increases the Huber denominator slightly, which reduces effective `lambda` per concept. May want to re-tune or switch to per-concept lambda.
3. **How to compare V2 to V1 fairly?** V1 was trained on 1GB data; V2 will train on the same. The teammate's training on better GPUs with more data is the "real" comparison point.
4. **Should `at_intersection` be improved** (e.g. use roadgraph if types become available) before V2, or keep the heuristic?
5. **Shared encoder on better GPUs?** `CBMConfig.shared_encoder=False` means 2x encoder passes for critic. With 15 concepts and a better GPU, `shared_encoder=True` may be worth experimenting with to speed up training.
