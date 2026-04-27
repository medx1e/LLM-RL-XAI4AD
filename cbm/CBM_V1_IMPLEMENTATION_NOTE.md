# CBM-V1 Implementation Note

## Selected Backbone

**`womd_sac_road_perceiver_minimal_42`** — the reference SAC model with LQ (Perceiver) encoder.

- Encoder: LQ attention (dk=64, ff_mult=2, 16 latents, depth=4, tied weights)
- Encoder output: **128-d** latent vector
- Observation: 1655-d flat vector (SDC, agents, roadgraph, traffic lights, GPS path)
- Standard 4-dim roadgraph features (no speed_limit — avoids the zero-pad hack)
- Pretrained checkpoint: `runs_rlc/womd_sac_road_perceiver_minimal_42/model/model_final.pkl`

## Architecture

```
                          HARD BOTTLENECK
                               |
obs (1655) --> [LQ Encoder] --> z (128) --> [Concept Head] --> c (11) --> [Actor FC] --> action (2)
                 pretrained      |              MLP(64)         |         MLP(64,32)
                                 |                              |
                                 |                              +--> [Critic FC] --> Q (2 twins)
                                 |                                   MLP(64,32)
                                 |                                   input: c || action
                                 |
                          stop_gradient (frozen mode)
```

**Key design choices:**
- **HARD bottleneck**: actor and critic only see the 11-d concept vector `c`, never the 128-d latent `z`.
- **Concept head**: `Dense(128→64) + ReLU + Dense(64→11) + Sigmoid` outputs in [0, 1].
- **Twin critics**: 2 Q-networks (standard SAC twin-Q) each with independent encoder+concept_head in non-shared mode.
- **Frozen encoder**: `stop_gradient` is applied after the encoder in `__call__`, so neither SAC loss nor concept loss updates encoder weights.

## Integration Points in V-Max

The CBM-V1 implementation lives entirely under `~/cbm/cbm_v1/` with **zero modifications to V-Max source**.

| V-Max component | CBM-V1 equivalent | Relationship |
|---|---|---|
| `PolicyNetwork` | `CBMPolicyNetwork` | Replaces FC input with concept bottleneck |
| `ValueNetwork` | `CBMValueNetwork` | Same replacement |
| `sac_factory.make_networks` | `cbm_sac_factory.make_networks` | Builds CBM modules instead |
| `sac_factory.make_sgd_step` | `cbm_sac_factory.make_sgd_step` | Adds concept loss to policy loss |
| `sac_factory._make_loss_fn` | Inlined in `make_sgd_step` | Concept targets computed inline |
| `network_factory._build_encoder_layer` | Reused directly | Same encoder construction |

**Pretrained weight loading:** The function `_load_pretrained_encoder` copies the encoder subtree from the pretrained params into the CBM params, remapping `perceiver_attention` → `lq_attention` (the name change from the encoder type remap).

## Loss Definition

```
L_total = L_SAC_policy + lambda_concept * L_concept
```

Where:
- `L_SAC_policy` = standard SAC actor loss (`alpha * log_prob - min(Q1, Q2)`)
- `L_concept` = masked concept supervision loss
- `lambda_concept` = 0.1 (configurable via `CBMConfig`)

**Concept loss per type:**
- Binary concepts (traffic_light_red, lead_vehicle_decelerating, at_intersection): **BCE**
- Continuous concepts (all others): **Huber loss** (delta=1.0)
- **Validity masking**: invalid concepts (e.g., TTC when no lead vehicle) are excluded
- **Aggregation**: sum of masked losses / count of valid entries

Value loss is unchanged from standard SAC (MSE on TD target).

## Frozen vs Joint Mode

| Aspect | Frozen (`mode="frozen"`) | Joint (`mode="joint"`) |
|--------|--------------------------|------------------------|
| Encoder weights | Fixed (stop_gradient) | Trainable |
| Concept head | Trainable | Trainable |
| Actor FC | Trainable | Trainable |
| Critic FC | Trainable | Trainable |
| Encoder grad norm | 0.0 (verified) | 1.058 (verified) |
| Use case | Warm-up phase | Full fine-tuning |

**Implementation:** The `frozen_encoder` flag is a module attribute on `CBMPolicyNetwork` and `CBMValueNetwork`. When True, `jax.lax.stop_gradient(z)` is applied after the encoder in `__call__`. This cleanly blocks gradients from both the SAC loss and the concept loss.

## Files Added

```
cbm_v1/
  __init__.py           Module exports
  config.py             CBMConfig dataclass (all hyperparameters)
  networks.py           CBMPolicyNetwork, CBMValueNetwork, ConceptHead (Flax modules)
  concept_loss.py       Masked concept supervision loss (BCE + Huber)
  cbm_sac_factory.py    CBM-aware SAC factory (network construction, loss, SGD step)
  smoke_test.py         End-to-end smoke test on real data
```

## Smoke Test Results

All 22 checks pass:

```
1.  Environment loads                             PASS
2.  Observation shape                             PASS  (1, 1655)
3.  Observation finite                            PASS
4.  Concept count                                 PASS  (11)
5.  Concept targets finite                        PASS
6.  Pretrained params load                        PASS
7.  CBM config                                    PASS
8.  CBM networks created                          PASS
9.  Policy params > 0                             PASS  (612,855)
10. Value params > 0                              PASS  (612,917)
11. Pretrained encoder loaded (policy)            PASS
12. Pretrained encoder loaded (value)             PASS
13. Policy forward finite                         PASS  (1, 4)
14. Value forward finite                          PASS  (1, 2)
15. Concepts shape                                PASS  (1, 11)
16. Concepts in [0, 1]                            PASS
17. Concepts finite                               PASS
18. Concept loss finite                           PASS  (0.043)
19. Gradients finite                              PASS  (norm=2.19)
20. Encoder grads = 0 (frozen)                    PASS  (0.0)
21. Concept head grads > 0 (frozen)               PASS  (0.496)
22. Encoder grads > 0 (joint)                     PASS  (1.058)
```

## Known Risks / Considerations

1. **Information bottleneck at 11-d**: The policy now sees only 11 numbers instead of 128. If these concepts don't capture enough information for good driving, performance will degrade. This is expected and is the core CBM tradeoff.

2. **Concept quality at t=0**: Some concepts (progress_along_route, dist_to_path) are near-constant at t=0 in logged data but become meaningful during rollout. The concept head may learn to ignore them during warm-up.

3. **Single encoder for policy and value (non-shared mode)**: Each twin Q-network has its own encoder + concept_head copy. This means 2x encoder forward passes for the critic. Consider `shared_encoder=True` if memory is tight.

4. **Concept targets are computed inside the training step**: `concept_targets_fn(observations)` runs the full concept extraction pipeline (unflatten + 11 extractors) on every SGD step. This adds overhead but keeps concept targets observation-faithful.

5. **Module reference for concept extraction**: The factory stores a global `_cbm_policy_module` reference for concept extraction during training. This is a pragmatic choice — Flax's functional API makes it awkward to call sub-module methods otherwise. Works correctly under JIT.

## Extending to CBM-V2

The design supports easy extension:

- **Add concepts**: Increase `num_concepts` in CBMConfig, update `concept_phases` or `concept_names`. The concept head auto-sizes to the new count.
- **Path spatial concepts**: Add Phase 3 extractors to the registry, bump `num_concepts` to 16.
- **Soft bottleneck**: Add a residual stream `z_residual` alongside the concept vector. Change actor input from `c` to `[c || z_residual]`. The CBMPolicyNetwork is already structured to make this a minimal change.
- **Learnable concept weighting**: Replace fixed `lambda_concept` with per-concept weights or a learnable temperature.

## Running the Smoke Test

```bash
/home/med1e/anaconda3/envs/vmax/bin/python cbm_v1/smoke_test.py
```

Requires: `data/training.tfrecord` and `runs_rlc/womd_sac_road_perceiver_minimal_42/model/model_final.pkl`.
