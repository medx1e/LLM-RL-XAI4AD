# LQ vs Wayformer: Attention Architecture Differences

## Summary

**Yes, LQ (Perceiver) has attention heads**, but the architecture is fundamentally different from Wayformer.

## Architecture Comparison

### Wayformer Architecture

**Structure:** Entity-specific attention modules
- Separate cross-attention for each entity type:
  - `other_traj/cross_attn_0` - Attention to other vehicles
  - `roadgraph/cross_attn_0` - Attention to road elements
  - `traffic_lights/cross_attn_0` - Attention to traffic lights
  - etc.

**Attention Heads:**
- Each module has its own set of attention heads
- Heads specialize within each entity type

**Attention Weight Keys:**
```python
{
    'other_traj/cross_attn_0': [batch, n_latents, n_vehicle_tokens, n_heads],
    'roadgraph/cross_attn_0': [batch, n_latents, n_roadgraph_tokens, n_heads],
    ...
}
```

---

### LQ (Perceiver) Architecture

**Structure:** Unified cross-attention + self-attention
- **Cross-attention:** Latents attend to ALL input tokens (concatenated)
- **Self-attention:** Latents attend to each other

**Two Types of Heads:**
1. **Cross-attention heads** (`cross_num_heads`): For latent-to-input attention
2. **Self-attention heads** (`latent_num_heads`): For latent-to-latent attention

**Attention Weight Keys:**
```python
{
    'cross_attn_0': [batch, n_latents, n_all_tokens, cross_num_heads],
    'self_attn_0': [batch, n_latents, n_latents, latent_num_heads],
    'cross_attn_1': [batch, n_latents, n_all_tokens, cross_num_heads],
    'self_attn_1': [batch, n_latents, n_latents, latent_num_heads],
    ...
}
```

**Token Concatenation Order:**
```
[sdc_traj | other_traj | roadgraph | traffic_lights | gps_path]
```

---

## Key Differences

| Aspect | Wayformer | LQ (Perceiver) |
|--------|-----------|----------------|
| **Attention Structure** | Entity-specific modules | Unified cross-attention |
| **Token Organization** | Separate per entity type | Concatenated all together |
| **Head Types** | Single type (cross-attention) | Two types (cross + self) |
| **Specialization** | Heads specialize per entity | Heads attend to all entities |
| **Extraction** | Direct from `other_traj/cross_attn_0` | Extract vehicle tokens from `cross_attn_0` |

---

## Implementation Impact

### For Attention Aggregation

**Wayformer:**
```python
# Direct access to vehicle attention
attn = attention_weights['other_traj/cross_attn_0']
# Shape: (batch, n_latents, n_vehicle_tokens, n_heads)
```

**LQ:**
```python
# Need to extract vehicle tokens from concatenated input
attn_all = attention_weights['cross_attn_0']
# Shape: (batch, n_latents, n_all_tokens, n_heads)

# Extract vehicle token range
other_start, other_end = token_boundaries['other_traj']
attn_vehicles = attn_all[:, :, other_start:other_end, :]
# Shape: (batch, n_latents, n_vehicle_tokens, n_heads)
```

### For Concentration Metrics

**Both architectures:**
- After extraction, both produce `(n_heads, n_vehicles)` arrays
- Concentration metrics (Gini, entropy, top-k) work identically
- The `n_heads` dimension exists in both (just different types)

---

## Solution Implemented

Updated `aggregate_vehicle_attention()` to:
1. **Auto-detect architecture** from attention weight keys
2. **Route to appropriate handler:**
   - `_aggregate_wayformer_attention()` - Direct extraction
   - `_aggregate_lq_attention()` - Extract from concatenated tokens
3. **Return consistent format:** `(n_heads, n_vehicles)` for both

This ensures the rest of the pipeline (concentration metrics, calibration analysis) works identically for both architectures.
