# Norm-Weighted Attention Integration Plan

> Kobayashi et al. (EMNLP 2020) — "Attention is Not Only a Weight"
> Adding a fourth attention signal alongside `cross_attn_avg`, `cross_attn_rollout`, and `maxpool`

---

## Section A: Technical Feasibility

### Answer: **YES — Option 1 (Preferred) is fully feasible.**

The value projection vectors `v_j = x_j W_V` and the output projection matrix `W_O` are both accessible from the existing `capture_intermediates` setup and parameter tree. **No architectural changes to V-MAX are needed.**

### How `capture_intermediates` Works in This Architecture

The [AttentionLayer](file:///home/med1e/platform_fyp/cbm/V-Max/vmax/agents/networks/encoders/attention_utils.py#L96-L151) creates four `nn.Dense` sub-modules in `__call__`:

| Flax Name | Line | Operation | Shape (cross-attn) |
|---|---|---|---|
| `Dense_0` | L129 | Q = `Dense(dim, use_bias=False)(x)` | `(B, 16, 32)` — queries from latent |
| `Dense_1` | L130 | K = `Dense(dim, use_bias=False)(context)` | `(B, 280, 32)` — keys from input tokens |
| `Dense_2` | L131 | V = `Dense(dim, use_bias=False)(context)` | `(B, 280, 32)` — **values from input tokens** |
| `Dense_3` | L147 | O = `Dense(x.shape[-1])(out)` | `(B, 16, 256)` — output projection (post-attention) |

Where `dim = heads × head_features = 2 × 16 = 32` and `x.shape[-1] = latent_dim = 256`.

When `capture_intermediates=True` is set (as it is in [perceiver_wrapper.py L60-64](file:///home/med1e/platform_fyp/post-hoc-xai/posthoc_xai/models/perceiver_wrapper.py#L60-L64)), Flax captures the `__call__` return value of **every** `nn.Module` sub-module. The existing code already reads `Dense_0` and `Dense_1`:

```python
# perceiver_wrapper.py, L212-213 — ALREADY WORKING
cq_list = cross.get("Dense_0", {}).get("__call__", [])  # Q: (B, 16, 32)
ck_list = cross.get("Dense_1", {}).get("__call__", [])  # K: (B, 280, 32)
```

**Therefore, `Dense_2` (value vectors) is already captured and accessible via the same pattern:**

```python
cv_list = cross.get("Dense_2", {}).get("__call__", [])  # V: (B, 280, 32)
```

With `tie_layer_weights=True` and `encoder_depth=4`, this will be a list of 4 tensors. Since the input tokens `x` don't change across depth iterations (only latents change — see [lq.py L90-97](file:///home/med1e/platform_fyp/cbm/V-Max/vmax/agents/networks/encoders/lq.py#L90-L97)), all 4 entries are **identical**. We can safely use `cv_list[0]`.

### Accessing W_O (Output Projection Matrix)

The output projection `Dense_3` parameters live in the model's parameter tree at:

```python
params['params']['encoder_layer']['lq_attention']['cross_attn']['Dense_3']['kernel']  # (32, 256)
params['params']['encoder_layer']['lq_attention']['cross_attn']['Dense_3']['bias']    # (256,)
```

> [!IMPORTANT]
> The exact path should be verified at implementation time via `jax.tree_util.tree_map_with_path(lambda p, v: (p, v.shape), params)`. The path above follows Flax naming conventions and the module hierarchy `PolicyNetwork → encoder_layer (LQEncoder) → lq_attention (LQAttention) → cross_attn (AttentionLayer) → Dense_3`.

### The Full Kobayashi Computation

```
For each input token j (out of 280):
    v_j = Dense_2_output[:, j, :]          # (B, 32) — value projection
    f(x_j) = v_j @ W_O_kernel + W_O_bias  # (B, 256) — value after output projection
    norm_j = ‖f(x_j)‖₂                    # (B,) — effective contribution magnitude

For each query i and token j:
    norm_weighted[i, j] = α[i, j] × norm_j   # scalar × scalar ≥ 0
```

Since `α[i,j] ≥ 0` (softmax output), the norm factorizes cleanly: `‖α·f(x)‖ = α·‖f(x)‖`.

### Simplified Fallback (Option 1b)

If extracting W_O from the parameter tree proves fragile, a valid simplification is:

```python
# Use ‖v_j‖ directly, without the output projection
f_x_norms = jnp.linalg.norm(v_tokens, axis=-1)  # (B, 280)
```

This captures the value-transform effect (the key Kobayashi insight) but misses the output projection's effect on relative norms. Both versions should be compared in a mini-experiment.

> [!NOTE]
> Option 3 from the task description (`‖α_{i,j} · x_j‖` — using raw input tokens instead of value-transformed tokens) is NOT needed. Option 1 is fully achievable.

---

## Section B: Where to Add the Computation

### Location: [perceiver_wrapper.py `_extract_attention()`](file:///home/med1e/platform_fyp/post-hoc-xai/posthoc_xai/models/perceiver_wrapper.py#L177-L289)

Insert after the cross-attention loop (after L235 where `cross_avg` is computed) and before the self-attention section (L237):

```python
# ── Norm-weighted attention (Kobayashi et al. 2020) ───────────
# Value vectors for input tokens — captured from Dense_2 (V projection)
cv_list = cross.get("Dense_2", {}).get("__call__", [])

if cv_list:
    # All 4 layer calls produce identical V vectors (tied weights, same input x)
    v_tokens = cv_list[0]  # (B, 280, 32)

    # Extract W_O from the parameter tree (Dense_3 in cross_attn)
    # Path: params → encoder_layer → lq_attention → cross_attn → Dense_3
    try:
        cross_attn_params = (
            self._policy_params['params']
            ['encoder_layer']['lq_attention']['cross_attn']
        )
        w_o_kernel = cross_attn_params['Dense_3']['kernel']  # (32, 256)
        w_o_bias = cross_attn_params['Dense_3']['bias']      # (256,)

        # f(x_j) = v_j @ W_O + bias  →  (B, 280, 256)
        f_x = jnp.einsum("btd,dk->btk", v_tokens, w_o_kernel) + w_o_bias

        # Per-token value-vector norm  →  (B, 280)
        f_x_norms = jnp.linalg.norm(f_x, axis=-1)
    except (KeyError, TypeError):
        # Fallback: use value-vector norms without W_O projection
        f_x_norms = jnp.linalg.norm(v_tokens, axis=-1)  # (B, 280)

    # Norm-weighted attention per layer, then average
    # cross_attn_layer has shape (B, 16, 280), f_x_norms is (B, 280)
    # For each layer: norm_weighted[i,j] = α[i,j] * ‖f(x_j)‖
    norm_weighted_layers = []
    for attn_layer in per_cross_layer:
        nw = attn_layer * f_x_norms[:, None, :]  # (B, 16, 280)
        norm_weighted_layers.append(nw)

    # Average over 4 depth iterations
    nw_avg = jnp.stack(norm_weighted_layers, axis=0).mean(axis=0)  # (B, 16, 280)

    # Normalize each query's distribution to sum to 1
    nw_normalized = nw_avg / (nw_avg.sum(axis=-1, keepdims=True) + 1e-8)

    result["norm_weighted_attn"] = nw_normalized  # (B, 16, 280)
    result["f_x_norms"] = f_x_norms               # (B, 280)
```

### Why This Location

1. `cv_list` is populated from the same `cross` intermediates dict as `cq_list`/`ck_list` — same data source, same access pattern
2. `per_cross_layer` (the per-layer softmax attention) is already computed and available
3. The result dict is being populated — we just add two new keys
4. Falls back gracefully if Dense_2 or params are unavailable

---

## Section C: New Keys in the Attention Dict

Following the naming convention of existing keys (`cross_attn_avg`, `cross_attn_rollout`):

```python
attn_dict['norm_weighted_attn']     # (B, 16, 280) — norm-weighted, layer-averaged, row-normalized
attn_dict['f_x_norms']              # (B, 280) — raw value-vector norms (diagnostic)
```

### Key Properties

| Key | Shape | Normalized? | When Available |
|---|---|---|---|
| `cross_attn_avg` | `(B, 16, 280)` | Yes (softmax rows) | Always |
| `cross_attn_rollout` | `(B, 16, 280)` | Approx (rollout product) | When self-attn captured |
| `norm_weighted_attn` | `(B, 16, 280)` | Yes (explicit row normalization) | When Dense_2 captured |
| `f_x_norms` | `(B, 280)` | No (raw norms) | When Dense_2 captured |

---

## Section D: Changes to `attention_aggregation.py`

### Recommendation: **No changes needed.**

The `norm_weighted_attn` tensor has the **same shape** `(16, 280)` as `cross_attn_avg` and `cross_attn_rollout`. It can be passed directly to the existing `aggregate_attention()` function with `mode='mean'`:

```python
# In any experiment script:
nw_matrix = attn['norm_weighted_attn'][t]       # (16, 280)
cat_fractions = aggregate_attention(nw_matrix, 'mean')  # → {cat: float}
```

This works because:
1. `norm_weighted_attn` is already a `(16, 280)` matrix (same as all other attention signals)
2. `mode='mean'` averages over the 16 queries, then sums per category and normalizes — exactly what we want
3. The row normalization applied in `perceiver_wrapper.py` ensures each query distribution sums to 1, matching the softmax convention

Adding a separate `mode='norm_weighted'` would be redundant since the norm-weighting is applied **upstream** (in `_extract_attention`), not in the aggregation step.

---

## Section E: Changes to Phase 3 Script

### Overview of Required Changes

The current [phase3_scale_correlation.py](file:///home/med1e/platform_fyp/post-hoc-xai/experiments/phase3_scale_correlation.py) hardcodes the attention signal as rollout in [`compute_rollout_attention()`](file:///home/med1e/platform_fyp/post-hoc-xai/experiments/phase3_scale_correlation.py#L156-L170). The following changes add norm_weighted as a parallel attention signal.

### Change 1: Add `ATTENTION_SIGNALS` config constant (top of file, near L16)

```python
# Which attention signals to use as "ground truth" for correlation
ATTENTION_SIGNALS = ["rollout"]  # add "norm_weighted" when ready
```

### Change 2: Generalize attention extraction (replace `compute_rollout_attention`)

```python
def compute_attention_signals(model, raw_obs: np.ndarray) -> dict[str, np.ndarray]:
    """Batched forward pass → all attention signals as (T, 5) category fractions."""
    import jax.numpy as jnp

    obs_batch = jnp.array(raw_obs)
    out       = model.forward(obs_batch)
    attn      = out.attention

    signals = {}

    # Rollout (existing, canonical)
    key = "cross_attn_rollout" if "cross_attn_rollout" in attn else "cross_attn_avg"
    rollout = np.array(attn[key])  # (T, 16, 280)
    result_rollout = np.zeros((raw_obs.shape[0], len(CATS)))
    for t in range(raw_obs.shape[0]):
        d = _aggregate_attention(rollout[t], "rollout")
        result_rollout[t] = [d[c] for c in CATS]
    signals["rollout"] = result_rollout

    # Norm-weighted (new)
    if "norm_weighted_attn" in attn:
        nw = np.array(attn["norm_weighted_attn"])  # (T, 16, 280)
        result_nw = np.zeros((raw_obs.shape[0], len(CATS)))
        for t in range(raw_obs.shape[0]):
            d = _aggregate_attention(nw[t], "mean")  # same aggregation as rollout
            result_nw[t] = [d[c] for c in CATS]
        signals["norm_weighted"] = result_nw

    return signals
```

### Change 3: Modify the scenario processing loop (~L626-644)

```python
# Attention (all configured signals)
attn_signals = compute_attention_signals(model, raw_obs)

# Attribution methods (unchanged)
method_results = {}
for mname in METHODS:
    ...  # existing code

# Save — now per attention signal
for signal_name in ATTENTION_SIGNALS:
    if signal_name not in attn_signals:
        continue
    attn_arr = attn_signals[signal_name]
    save_scenario_result(
        scenario_id, attn_arr, method_results,
        collision_risk, ep["safety_risk"],
        ep["accel"], ep["steering"],
        out_dir, attention_signal=signal_name,  # NEW parameter
    )
```

### Change 4: Extend `save_scenario_result` to include attention signal name

Add `attention_signal` parameter and include it in the JSON filename and content:

```python
path = out_dir / f"scenario_{scenario_id:04d}_{attention_signal}.json"
```

### Impact on Existing Results

> [!WARNING]
> Existing Phase 3 JSONs are named `scenario_XXXX.json` (no attention signal suffix). These contain rollout-only correlations. Two options:
>
> **Option A (recommended):** Rename existing files to `scenario_XXXX_rollout.json` before running the updated script. The new script writes `scenario_XXXX_norm_weighted.json` alongside them. Aggregation loads both.
>
> **Option B:** Rerun the 2–3 existing scenarios (~2 min each with VG-only). Acceptable since the study hasn't scaled yet.

### Change 5: CLI flag (optional)

```python
parser.add_argument("--attention-signals", nargs="+", default=None,
                    help="Attention signals to correlate against. "
                         "Default: rollout. Add norm_weighted.")
```

---

## Section F: Backward Compatibility

### ✅ `reward_attention/` — Completely Untouched

The reward-attention study (already submitted paper) uses `cross_attn_avg` accessed via its own independent code path. It does NOT import from `perceiver_wrapper.py`'s `_extract_attention()`. The new `norm_weighted_attn` key is **additive** — it appears in the returned dict alongside existing keys, and no existing key is modified or removed.

### ✅ Phase 2 Pilot Results — Preserved

[phase2_correlation_pilot.py](file:///home/med1e/platform_fyp/post-hoc-xai/experiments/phase2_correlation_pilot.py) calls `model.forward()` and reads `cross_attn_rollout` from the attention dict. Adding `norm_weighted_attn` to the dict has zero effect on existing Phase 2 code — it simply ignores the new key. All Phase 2 results, figures, and JSONs are unchanged.

### ✅ Platform Cache — Compatible

The platform_cache stores pickled artifacts, NOT live attention dicts. Norm-weighted attention cannot be computed from cached data (it requires the Dense_2 capture from a live forward pass). However:
- Phase 2 and 3 both do **live forward passes** — they don't use cached attention
- The cache is only used for raw observations and cached attributions
- No cache modification is needed

### ✅ `attention_aggregation.py` — No Changes

As discussed in Section D, norm_weighted_attn uses the existing `aggregate_attention()` with `mode='mean'`.

---

## Section G: Testing Strategy

### Sanity Check 1 — Row Normalization

```python
# After _extract_attention() returns:
nw = attn['norm_weighted_attn']  # (B, 16, 280)
row_sums = nw.sum(axis=-1)       # (B, 16)
assert jnp.allclose(row_sums, 1.0, atol=1e-5), f"Row sums: {row_sums}"
```

**Expected:** Each query's distribution sums to 1.0 (we apply explicit normalization).

### Sanity Check 2 — Non-Identity with Raw Attention

```python
raw = attn['cross_attn_avg']          # (B, 16, 280)
nw  = attn['norm_weighted_attn']      # (B, 16, 280)
mad = jnp.abs(raw - nw).mean()
print(f"MAD(norm_weighted vs raw): {mad:.4f}")
assert mad > 0.001, "Norm-weighted and raw are identical — value norms are constant"
```

**Expected:** MAD > 0.001. If identical, value norms are constant across all 280 tokens (extremely unlikely given diverse input categories).

### Sanity Check 3 — Category-Level MAD Comparison

Run on the 3 platform_cache scenarios (same protocol as [Phase 1c](file:///home/med1e/platform_fyp/post-hoc-xai/experiments/phase1c_rollout_comparison.py)):

```python
for t in range(T):
    raw_cats     = aggregate_attention(raw_attn[t], 'mean')
    rollout_cats = aggregate_attention(rollout_attn[t], 'rollout')
    nw_cats      = aggregate_attention(nw_attn[t], 'mean')

    mad_nw_vs_raw     = mean_absolute_deviation(nw_cats, raw_cats)
    mad_nw_vs_rollout = mean_absolute_deviation(nw_cats, rollout_cats)
```

**Expected:** MAD > 0.01 (non-trivial difference). Report per-category breakdown as in Phase 1c findings.

### Sanity Check 4 — Category Sum = 1.0

```python
cats = aggregate_attention(nw_attn[0], 'mean')  # {cat: float}
total = sum(cats.values())
assert abs(total - 1.0) < 1e-6, f"Category sum: {total}"
```

**Expected:** Sum = 1.0 (guaranteed by `aggregate_attention`'s normalization).

### Diagnostic: Inspect `f_x_norms`

```python
norms = attn['f_x_norms']  # (B, 280)
print(f"f_x_norms — min: {norms.min():.4f}, max: {norms.max():.4f}, "
      f"std: {norms.std():.4f}, cv: {norms.std()/norms.mean():.2f}")
```

**Expected:** The coefficient of variation (CV) should be > 0.1. A CV near 0 means all tokens have equal value-vector norms, which would make norm-weighting meaningless.

---

## Section H: Effort Estimate

### Lines of Code

| File | Lines to Add | Lines to Modify |
|---|---|---|
| `perceiver_wrapper.py` | ~25 | 0 |
| `phase3_scale_correlation.py` | ~40 | ~20 (refactor `compute_rollout_attention` → `compute_attention_signals`, modify save/load) |
| Phase 1d mini-experiment script (NEW) | ~120 | N/A |
| **Total** | **~185** | **~20** |

### Estimated Implementation Time

| Task | Time |
|---|---|
| Implement norm-weighted in `perceiver_wrapper.py` + verify param path | 1.0 hr |
| Run sanity checks (4 checks on 1 scenario) | 0.5 hr |
| Phase 1d mini-experiment (3 scenarios, both models) | 1.0 hr |
| Integrate into Phase 3 script | 1.5 hr |
| Test Phase 3 with norm_weighted on 2 scenarios | 0.5 hr |
| **Total** | **4.5 hrs** |

### Should We Run Phase 1d First?

> [!IMPORTANT]
> **YES — run a mini Phase 1d experiment before integrating into Phase 3.** This follows the established pattern (Phase 1c validated rollout before it was promoted to Phase 2/3 canonical signal). Phase 1d should:
>
> 1. Compute norm_weighted category fractions for the 3 platform_cache scenarios
> 2. Report MAD(norm_weighted vs rollout) and MAD(norm_weighted vs raw) per category
> 3. Inspect the `f_x_norms` distribution (are value norms meaningfully heterogeneous?)
> 4. Compare both Option 1a (with W_O) and Option 1b (without W_O) to see if the output projection matters
>
> This takes ~1 hour and produces the diagnostic data needed to write the thesis methodology section on norm-weighted attention.

---

## Recommendation

**Proceed with implementation in this order:**

1. **Add norm-weighted computation to `perceiver_wrapper.py`** (~25 lines, Section B). This is the core change — purely additive, zero risk to existing code.

2. **Write and run Phase 1d** (new script `experiments/phase1d_norm_weighted_comparison.py`). Compare norm_weighted vs rollout vs raw on the 3 platform_cache scenarios. This validates the implementation and produces the MAD table for the thesis methodology section.

3. **Integrate into Phase 3** only after Phase 1d confirms the signal is meaningfully different from rollout. If MAD(norm_weighted vs rollout) < 0.01, the signal adds noise but not information — skip Phase 3 integration. If MAD > 0.02 (likely, given the heterogeneous token structure), integrate and run at scale.

4. **The key hypothesis** is that norm-weighted attention will agree more strongly with SARFA than rollout does, because both norm-weighting and SARFA capture the *magnitude* of each token's contribution to the action, not just the *relative allocation* of attention. This would strengthen the Kobayashi "attention is not only a weight" finding in the RL domain.
