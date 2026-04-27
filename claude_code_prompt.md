# Claude Code Task: Norm-Based Attention Integration — Inspect, Plan, Design

## Your Role
You are a research engineering agent. Your job is **not to implement anything yet** — it is to
**inspect the existing codebase, understand what is already built, and produce a concrete
integration plan** for adding norm-weighted attention (Kobayashi et al., EMNLP 2020) as a
fourth attention signal alongside the three already implemented.

Do not modify any files. Do not run experiments. Read, trace, plan, write the plan.

---

## Background: What Needs to Be Added

### The Scientific Goal
We want to compute **norm-weighted attention** for the Perceiver encoder — i.e., instead of
using the raw softmax attention weight `α_{i,j}` as the measure of how much query `i` attends
to token `j`, we use:

```
norm_weighted[i, j] = ‖ α_{i,j} · f(x_j) ‖
```

where `f(x_j) = (x_j W_V + b_V) W_O` is the value-transformed version of token `j` projected
through the output matrix. This is the "effective contribution" of token `j` to query output `i`
in Euclidean norm — the key quantity from Kobayashi et al. (2020) "Attention is Not Only a
Weight."

After computing this `(16, 280)` norm matrix, we aggregate it to the 5 input categories exactly
as we already do for other attention signals:
- SDC: tokens [0, 5)
- Other agents: tokens [5, 45)
- Roadgraph: tokens [45, 245)
- Traffic lights: tokens [245, 270)
- GPS path: tokens [270, 280)

The result is a 5-dim category vector comparable to `cross_attn_avg`, `cross_attn_rollout`,
and `maxpool_attn`.

### What Already Exists (the three attention signals in use)
1. `cross_attn_avg` — mean-pool of raw softmax weights over 16 queries → (16, 280) → mean → (280,) → cat sums
2. `cross_attn_rollout` — rollout through self-attention layers, currently the **canonical** signal for Phase 2/3
3. maxpool aggregation — max over 16 queries (available via `aggregate_attention(..., mode='maxpool')`)

We are adding a fourth: `norm_weighted_attn` — same shape path as the others, different computation.

### What This Is NOT
- This is NOT the Kobayashi 2021 "context-mixing ratio" (that requires decomposing the full attention block including RES and LN — skip)
- This is NOT the Kobayashi 2024 FF-block decomposition — skip entirely
- Just the EMNLP 2020 paper's core idea: multiply softmax weights by value-vector norms

---

## Codebase Location

```
/home/med1e/platform_fyp/post-hoc-xai/
├── posthoc_xai/
│   ├── models/
│   │   ├── perceiver_wrapper.py       ← MAIN TARGET — attention extraction lives here
│   │   ├── generic_wrapper.py
│   │   └── loader.py
│   ├── utils/
│   │   ├── normalization.py           ← size_correct_attribution, size_correct_attention
│   │   ├── attention_aggregation.py   ← aggregate_attention(attn, mode) — mean/maxpool/entropy
│   │   └── ig_baseline.py
│   └── visualization/
│       └── paper_figures.py
├── experiments/
│   ├── phase1b_aggregation_comparison.py   ← used attention_aggregation.py
│   ├── phase1c_rollout_comparison.py       ← added rollout extraction
│   ├── phase2_correlation_pilot.py         ← used rollout as canonical signal
│   └── phase3_scale_correlation.py         ← MAIN EXPERIMENT — where results are computed
└── reward_attention/                        ← ALREADY SUBMITTED PAPER — DO NOT MODIFY
```

---

## Your Step-by-Step Task

### Step 1 — Read `perceiver_wrapper.py` in full

Locate the file at the path above and read every line. You need to understand:

1. How `_extract_attention()` currently works — what intermediates it captures, what keys it
   returns in the attention dict.
2. **Specifically:** Does it already capture the **Value projection** matrices or the **post-value
   vectors** `(x_j W_V)` or `(x_j W_V) W_O`? Look for any capture of:
   - Dense layers after the key/query projections (typically Dense_2 for values in cross-attention)
   - Any `W_V`, `W_O`, or value-related intermediates
   - The raw input tokens `x_j` in their pre-attention form
3. What keys are currently returned: specifically check for `cross_attn_avg`, `cross_attn_rollout`,
   `self_attn_layer_*`, and whether value vectors are anywhere in the output.
4. How the architecture is structured: the Perceiver uses `tie_layer_weights=True` with
   `encoder_depth=4`, so there is ONE cross-attention module and ONE self-attention module,
   each CALLED 4 times. The intermediates will be lists of 4 tensors, not 4 separate modules.

### Step 2 — Read `attention_aggregation.py` in full

Understand:
1. The `aggregate_attention(attn_matrix, mode, n_tokens_per_category)` function signature
2. How the 5-category token ranges are passed in (hardcoded? from a config? passed as argument?)
3. Whether adding a new aggregation mode `'norm_weighted'` would be natural here, or whether
   norm-weighted attention should be computed upstream (in `perceiver_wrapper.py`) and stored
   as a new key alongside `cross_attn_avg` and `cross_attn_rollout`.

### Step 3 — Read `phase3_scale_correlation.py` in full

This is the main experiment. Understand:
1. How it loads and uses attention signals — which keys it reads from the attention dict
2. Where correlation with attribution methods is computed
3. How the results are stored (JSON/CSV structure) and figure generation is triggered
4. Whether adding a new attention signal requires changes only to the data-loading section,
   or deeper changes throughout the correlation computation pipeline
5. **Critically:** Does the script have a `--methods` or `--attention-signals` CLI flag, or is
   the set of attention signals hardcoded? If hardcoded, where?

### Step 4 — Read `phase2_correlation_pilot.py`

Same questions as Step 3 but for the pilot. Check whether it shares code with Phase 3 or is
independent. Understand if changes to one automatically propagate to the other.

### Step 5 — Read `phase1c_rollout_comparison.py` (briefly)

Understand the pattern used to add rollout as a new attention signal — this is the closest
precedent for what we are doing now. Note how rollout was: (a) extracted in
`perceiver_wrapper.py`, (b) stored in the attn dict under a new key, (c) used in downstream
experiments. We will follow the same pattern.

### Step 6 — Check the platform_cache data format

The platform_cache was used in Phase 2 and the rollout comparison. Check (if accessible)
what keys are stored per-timestep in the cached attention dicts:
- Are they pickled Python dicts? NumPy arrays? JSON?
- Would adding a new key `norm_weighted_attn` require re-running the cache, or can it be
  computed post-hoc from already-cached data?

---

## What to Produce: The Integration Plan

After reading the above files, write a **detailed integration plan** as a markdown document
covering these sections:

### Section A: Technical Feasibility

Answer: Can we extract the value vectors from the current `capture_intermediates` setup?

**Option 1 (Preferred if feasible):** The value projection `W_V` and output projection `W_O`
are accessible as parameters or intermediates. We can compute `f(x_j) = (x_j W_V) W_O`
directly for all 280 tokens at once, then compute `‖ α_{i,j} · f(x_j) ‖` for the full
(16, 280) attention matrix. This requires only one additional forward-pass capture.

**Option 2 (Fallback):** We cannot easily access `W_V` and `W_O` as separate tensors, but we
CAN access the full cross-attention output vectors (the 16 latent query vectors AFTER the
cross-attention block, before residual). In this case, we can decompose:
`output_i = Σ_j α_{i,j} f(x_j)` and the norm ‖f(x_j)‖ can be approximated from the
input embeddings and the weight matrices via the Flax model's parameter tree.

**Option 3 (Simplified norm approximation):** If value vectors are not accessible without
major changes, compute a simplified norm: `‖α_{i,j} · x_j‖` (norm of the input token
weighted by attention). This loses the value-transformation effect but is a valid first
approximation. Clearly flag if this is what we are doing.

State clearly which option is achievable given the current architecture and `capture_intermediates`
setup.

### Section B: Where to Add the Computation

Specify the exact location in `perceiver_wrapper.py` where the norm-weighted computation
should be added. Write pseudo-code showing:

```python
# In _extract_attention(), after computing cross_attn_avg and cross_attn_rollout:

# 1. Access value vectors for all 280 input tokens
value_vectors = ...  # shape: (B, 280, d_v) — how to get this

# 2. Project through output matrix
f_x = ...  # shape: (B, 280, d_model) — value vectors after W_O projection

# 3. Compute norm-weighted attention: for each query i and token j:
#    norm_weighted[i, j] = α[i, j] * ‖f(x_j)‖
f_x_norms = jnp.linalg.norm(f_x, axis=-1)  # shape: (B, 280)

# For each of the 16 queries and each of the 4 cross-attention layers:
# cross_attn has shape (B, num_layers, num_queries, num_tokens) = (B, 4, 16, 280)
# f_x_norms needs to broadcast to (B, 1, 1, 280) to multiply with attention

norm_weighted = cross_attn_all_layers * f_x_norms[:, None, None, :]  # (B, 4, 16, 280)
norm_weighted_avg = norm_weighted.mean(axis=1)  # (B, 16, 280) — average over 4 layers

# Then normalize so it can be compared as a distribution:
norm_weighted_normed = norm_weighted_avg / (norm_weighted_avg.sum(axis=-1, keepdims=True) + 1e-8)
```

Fill in the actual JAX code where `...` appears, based on what you find in the file.

### Section C: New Key in the Attention Dict

Specify exactly what new key(s) to add to the returned attention dictionary. Follow the naming
convention of existing keys. Recommended:

```python
attn_dict['norm_weighted_attn']  # (B, 16, 280) — norm-weighted, layer-averaged, normalized
```

Optionally also:
```python
attn_dict['f_x_norms']  # (B, 280) — just the value-vector norms, useful for analysis
```

### Section D: Changes to `attention_aggregation.py`

Specify whether:
1. `aggregate_attention()` needs a new `mode='norm_weighted'` option, OR
2. Norm-weighted attention should be aggregated by the same `mean` path (it's already a
   (16, 280) matrix, same shape as `cross_attn_avg`), just stored under a different key.

The second option requires zero changes to `attention_aggregation.py` — norm_weighted_attn
is stored in the dict and the caller passes it to `aggregate_attention` with `mode='mean'`.
This is cleaner. Confirm or reject this approach.

### Section E: Changes to Phase 3 Script

Specify the minimal changes needed to `phase3_scale_correlation.py` to:
1. Extract `norm_weighted_attn` from the attn dict (or compute it if not cached)
2. Run the same correlation pipeline against VG, IG, and SARFA
3. Include it in the output JSON/CSV under a new method key (e.g., `'norm_weighted'`)
4. Include it in the figure generation without removing any existing figures

**Key constraint:** The existing Phase 3 results (3 scenarios, VG+IG already run) must not
be invalidated. The new signal should be added as a parallel column in the results, not as
a replacement. Specify whether this requires rerunning the existing scenarios or whether
it can be computed post-hoc from cached data.

### Section F: Backward Compatibility

Confirm that:
1. `reward_attention/` is **completely untouched** — it uses `cross_attn_avg`, not the new signal
2. Phase 2 pilot results are **preserved** — new signal is additive, not a replacement
3. Platform cache can be extended with the new key OR the new key can be computed on-the-fly
   from already-cached raw attention data

### Section G: Testing Strategy

Specify a minimal test to confirm the norm-weighted attention is computed correctly before
running at scale:

1. **Sanity check 1:** On a single timestep, verify `norm_weighted_attn[i, :].sum() ≈ 1.0`
   (it should be normalized to a distribution over 280 tokens)
2. **Sanity check 2:** Verify that norm_weighted and raw cross_attn_avg are NOT identical
   (if they are identical, something is wrong — the value norms are constant, which would
   be unexpected)
3. **Sanity check 3:** On the platform_cache scenarios, compute norm_weighted category
   fractions and compare to rollout and raw. Report the MAD as we did in Phase 1c.
4. **Sanity check 4:** Check that `[SDC] + [agents] + [road] + [TL] + [GPS] = 1.0` after
   category aggregation.

### Section H: Effort Estimate

Given what you find in the code, estimate:
- Lines of code to add to `perceiver_wrapper.py`
- Lines of code to add to `phase3_scale_correlation.py`
- Estimated implementation time (hours)
- Whether a mini Phase 1d experiment (analogous to Phase 1c rollout comparison) should be
  run first, or whether norm-weighted can be added directly into Phase 3

---

## Constraints

1. **Do not modify `reward_attention/`** in any way. The submitted paper uses `cross_attn_avg`
   and must remain unchanged.
2. **Do not remove rollout.** It stays as the canonical Phase 2/3 signal. We are adding
   norm_weighted as a fourth signal to compare against, not replacing anything.
3. **Preserve all existing cached results.** The Phase 3 VG+IG results for 3 scenarios must
   not be invalidated. If adding norm_weighted requires rerunning those scenarios, flag it
   explicitly and estimate the cost.
4. **Do not implement anything not in this plan.** The Kobayashi 2021 context-mixing ratio
   and Kobayashi 2024 FF-block decomposition are explicitly out of scope.
5. **Follow existing code patterns.** If rollout was added by extending the return dict of
   `_extract_attention()`, the norm-weighted signal should follow the same pattern.

---

## Output Format

Produce a single markdown document structured as sections A through H above.
- Section A must give a clear YES/NO on feasibility and specify which option applies.
- Sections B and E must include actual code snippets (or pseudo-code with specific variable
  names from the actual codebase).
- Section F must explicitly confirm backward compatibility for `reward_attention/` and Phase 2.
- End with a single-paragraph **Recommendation** stating whether to proceed, and in what order.
