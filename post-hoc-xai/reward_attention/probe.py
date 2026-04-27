"""Phase 0: Verify attention extraction from the Perceiver/LQ encoder.

Run this FIRST before implementing anything else.

What this checks:
1. Model loads without errors
2. capture_intermediates returns a non-empty intermediates dict
3. _extract_attention finds actual attention tensors (not empty)
4. Attention tensor shapes make sense (batch, n_queries, n_tokens)
5. Token count matches expected (224 = 1+8+200+5+10)
6. Attention values are non-uniform (not all equal → real signal)

Usage:
    cd /home/med1e/post-hoc-xai
    eval "$(~/anaconda3/bin/conda shell.bash hook)" && conda activate vmax
    export PYTHONPATH=/home/med1e/post-hoc-xai/V-Max:$PYTHONPATH
    python reward_attention/probe.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import jax
import jax.numpy as jnp


MODEL_DIR = "runs_rlc/womd_sac_road_perceiver_complete_42"
DATA_PATH  = "data/training.tfrecord"

DIVIDER = "=" * 70


def print_tree(d, prefix="", max_depth=8, depth=0):
    """Recursively print a nested dict/list tree with shapes."""
    if depth > max_depth:
        print(f"{prefix}... (max depth)")
        return
    if isinstance(d, dict):
        for k, v in d.items():
            print_tree(v, prefix=f"{prefix}/{k}", depth=depth + 1)
    elif isinstance(d, (list, tuple)):
        for i, item in enumerate(d):
            print_tree(item, prefix=f"{prefix}[{i}]", depth=depth + 1)
    elif hasattr(d, "shape"):
        arr = np.array(d)
        print(f"{prefix}  shape={arr.shape}  dtype={arr.dtype}  "
              f"min={arr.min():.4f}  max={arr.max():.4f}  "
              f"mean={arr.mean():.4f}  std={arr.std():.4f}")
    else:
        print(f"{prefix}  <{type(d).__name__}>")


def main():
    print(DIVIDER)
    print("REWARD ATTENTION PROBE — Phase 0")
    print(DIVIDER)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print("\n[1] Loading model ...")
    import posthoc_xai as xai
    model = xai.load_model(MODEL_DIR, data_path=DATA_PATH)
    print(f"    Encoder type  : {model._loaded.encoder_type}")
    print(f"    Original type : {model._loaded.original_encoder_type}")
    print(f"    Obs size      : {model._loaded.obs_size}")
    print(f"    Action size   : {model._loaded.action_size}")
    print(f"    has_attention : {model.has_attention}")
    print(f"    Wrapper class : {type(model).__name__}")

    # ------------------------------------------------------------------
    # 2. Get one observation
    # ------------------------------------------------------------------
    print("\n[2] Getting scenario observation ...")
    scenario = next(model._loaded.data_gen)
    state = model._loaded.env.reset(scenario, jax.random.split(jax.random.PRNGKey(0), 1))
    obs = np.array(state.observation).reshape(-1)
    obs_jnp = jnp.array(obs)
    print(f"    obs shape: {obs.shape}  min={obs.min():.3f}  max={obs.max():.3f}")

    # ------------------------------------------------------------------
    # 3. Raw forward pass with capture_intermediates — print full tree
    # ------------------------------------------------------------------
    print("\n[3] Running forward pass with capture_intermediates ...")
    obs_batch = obs_jnp[None, :]  # (1, obs_dim)
    logits, state_out = model._loaded.policy_module.apply(
        model._loaded.policy_params,
        obs_batch,
        capture_intermediates=True,
        mutable=["intermediates"],
    )
    intermediates = state_out.get("intermediates", {})
    print(f"    logits shape: {logits.shape}")
    print(f"    intermediates top-level keys: {list(intermediates.keys())}")

    print("\n[3b] Full intermediates tree (all keys + shapes):")
    print_tree(intermediates)

    # ------------------------------------------------------------------
    # 4. Call the wrapper's _extract_attention
    # ------------------------------------------------------------------
    print(f"\n[4] Calling model._extract_attention(state_out) ...")
    attn_dict = model._extract_attention(state_out)
    if attn_dict is None or len(attn_dict) == 0:
        print("    *** ATTENTION DICT IS EMPTY — _extract_attention failed ***")
        print("    Review the tree above to find real attention keys.")
        print("    Look for tensors with shape (..., N_queries, N_tokens)")
    else:
        print(f"    Found {len(attn_dict)} attention tensor(s):")
        for key, val in attn_dict.items():
            arr = np.array(val)
            print(f"      '{key}'  shape={arr.shape}  "
                  f"min={arr.min():.4f}  max={arr.max():.4f}  "
                  f"std={arr.std():.4f}")

    # ------------------------------------------------------------------
    # 5. Call model.get_attention() (high-level API)
    # ------------------------------------------------------------------
    print(f"\n[5] Calling model.get_attention(obs) ...")
    attn_out = model.get_attention(obs_jnp)
    if attn_out is None or len(attn_out) == 0:
        print("    *** model.get_attention() returned None/empty ***")
    else:
        print(f"    Returned {len(attn_out)} tensor(s):")
        for key, val in attn_out.items():
            arr = np.array(val)
            print(f"      '{key}'  shape={arr.shape}  std={arr.std():.4f}")

    # ------------------------------------------------------------------
    # 6. Forward pass via model.forward() — full ModelOutput
    # ------------------------------------------------------------------
    print(f"\n[6] Calling model.forward(obs) ...")
    out = model.forward(obs_jnp)
    print(f"    action_mean shape : {out.action_mean.shape}")
    print(f"    action_std  shape : {out.action_std.shape}")
    print(f"    embedding   shape : {out.embedding.shape if out.embedding is not None else 'None'}")
    if out.attention is not None and len(out.attention) > 0:
        print(f"    attention keys    : {list(out.attention.keys())}")
        for k, v in out.attention.items():
            arr = np.array(v)
            print(f"      '{k}'  shape={arr.shape}")
    else:
        print("    attention: None/empty")

    # ------------------------------------------------------------------
    # 7. Observation structure
    # ------------------------------------------------------------------
    print(f"\n[7] Observation structure (category level):")
    for cat, (s, e) in model.observation_structure.items():
        print(f"    {cat:20s}  [{s:4d}, {e:4d})  size={e-s}")

    print(f"\n[7b] Detailed structure (entity level):")
    for cat, info in model.observation_structure_detailed.items():
        n = info["num_entities"]
        fpn = info["features_per_entity"]
        print(f"    {cat:20s}  {n} entities × {fpn} feats/entity")

    # ------------------------------------------------------------------
    # 8. Token count analysis
    # ------------------------------------------------------------------
    print(f"\n[8] Expected token counts (1 sdc + 8 agents + 200 road + 5 lights + 10 gps = 224):")
    detailed = model.observation_structure_detailed
    for cat, info in detailed.items():
        print(f"    {cat:20s}  {info['num_entities']} tokens")
    total_expected = sum(info["num_entities"] for info in detailed.values())
    print(f"    TOTAL TOKENS (expected): {total_expected}")

    # ------------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------------
    print(f"\n{DIVIDER}")
    print("PROBE SUMMARY")
    print(DIVIDER)

    attn_works = attn_out is not None and len(attn_out) > 0
    if attn_works:
        # Check if any tensor looks like cross-attention: (batch, n_queries, n_tokens)
        cross_attn_candidates = []
        for key, val in attn_out.items():
            arr = np.array(val)
            # Look for shape with 3+ dims and last dim could be n_tokens
            if arr.ndim >= 2 and arr.std() > 1e-4:
                cross_attn_candidates.append((key, arr.shape, arr.std()))

        print(f"  Attention extraction: {'OK' if attn_works else 'FAILED'}")
        print(f"  Non-uniform tensors : {len(cross_attn_candidates)}")
        for key, shape, std in cross_attn_candidates:
            print(f"    '{key}'  shape={shape}  std={std:.4f}")

        if cross_attn_candidates:
            # Try to identify n_tokens dimension
            for key, shape, std in cross_attn_candidates:
                if len(shape) >= 2:
                    print(f"\n  For '{key}'  shape={shape}:")
                    print(f"    If shape = (batch, n_queries, n_tokens): n_queries={shape[-2]}, n_tokens={shape[-1]}")
                    if len(shape) >= 3:
                        print(f"    If shape = (batch, n_heads, n_queries, n_tokens): n_heads={shape[-3]}, n_queries={shape[-2]}, n_tokens={shape[-1]}")
            print("\n  ACTION: Verify which dimension is n_tokens (expected ~224).")
            print("          Then hardcode TOKEN_RANGES in reward_attention/config.py.")
        else:
            print("\n  WARNING: All attention tensors are uniform (std ≈ 0).")
            print("           This may mean attention is not being extracted correctly.")
    else:
        print("  Attention extraction: FAILED")
        print("  ACTION: Review the intermediates tree in section [3b] above.")
        print("          Find tensors with softmax-normalized values (rows sum to ~1).")
        print("          Update _extract_attention in perceiver_wrapper.py with correct key path.")

    print(f"\n{DIVIDER}\n")


if __name__ == "__main__":
    main()
