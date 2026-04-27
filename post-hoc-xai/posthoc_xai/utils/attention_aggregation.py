"""Attention aggregation utilities.

Provides alternative strategies for collapsing the Perceiver's
(n_queries, n_tokens) cross-attention matrix into a per-category
importance vector for comparison with gradient attribution.

Token layout (280 tokens total):
  sdc_trajectory : [0,   5)   — 1 entity × 5 timesteps
  other_agents   : [5,  45)   — 8 agents × 5 timesteps
  roadgraph      : [45, 245)  — 200 road points
  traffic_lights : [245, 270) — 5 lights × 5 timesteps
  gps_path       : [270, 280) — 10 waypoints

Strategies implemented:
  mean     — current default: average across queries, sum per category
  maxpool  — max across queries (best-case attention per token), sum per category
  entropy  — weight each query by its sharpness (inverse entropy) before averaging;
             sharply attending queries contribute more than uniformly attending ones

Gradient-weighted (weight each query by |∂output/∂latent_k|) is deferred to
Phase 1c as it requires a separate architectural pass through the self-attention
layers and policy head.
"""

from typing import Dict, Literal, Tuple
import numpy as np

# ---------------------------------------------------------------------------
# Token layout
# ---------------------------------------------------------------------------

TOKEN_RANGES: Dict[str, Tuple[int, int]] = {
    "sdc_trajectory":  (0,   5),
    "other_agents":    (5,   45),
    "roadgraph":       (45,  245),
    "traffic_lights":  (245, 270),
    "gps_path":        (270, 280),
}

N_TOKENS  = 280
N_QUERIES = 16

AggMethod = Literal["mean", "maxpool", "entropy", "rollout"]


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def aggregate_attention(
    attn: np.ndarray,
    method: AggMethod = "mean",
) -> Dict[str, float]:
    """Collapse (n_queries, n_tokens) attention into per-category importance.

    Args:
        attn: Numpy array shape ``(n_queries, n_tokens)`` or
              ``(batch, n_queries, n_tokens)``. If batched, batch dim is
              squeezed (pass a single timestep).
        method: Aggregation strategy — see module docstring.

    Returns:
        Dict mapping category name to importance fraction (sums to 1).
    """
    attn = np.array(attn)
    if attn.ndim == 3:
        attn = attn[0]  # squeeze batch dim

    assert attn.ndim == 2, f"Expected (Q, T) or (1, Q, T), got {attn.shape}"
    assert attn.shape == (N_QUERIES, N_TOKENS), (
        f"Expected ({N_QUERIES}, {N_TOKENS}), got {attn.shape}"
    )

    if method == "mean":
        token_importance = attn.mean(axis=0)          # (280,)

    elif method == "maxpool":
        # For each token: maximum attention it received from any single query.
        # Captures the "at least one query cares about this token" signal.
        token_importance = attn.max(axis=0)           # (280,)

    elif method == "entropy":
        # Weight each query by its sharpness: queries that concentrate their
        # attention (low entropy) are more informative than uniform ones.
        # w_k = 1 / (H(attn[k, :]) + eps),  normalized to sum to 1.
        eps = 1e-10
        attn_safe = np.clip(attn, eps, 1.0)
        # Shannon entropy per query (bits, max = log2(280) ≈ 8.1)
        H = -np.sum(attn_safe * np.log2(attn_safe), axis=1)   # (16,)
        weights = 1.0 / (H + eps)
        weights = weights / weights.sum()                       # normalize
        token_importance = (attn * weights[:, None]).sum(axis=0)  # (280,)

    elif method == "rollout":
        # Rolled-out attention is already a (16, 280) matrix computed inside
        # _extract_attention(). Pass it directly as attn — it is used as-is
        # with mean aggregation over queries, same as 'mean'.
        # The rolled-out matrix already accounts for self-attention mixing, so
        # mean-pooling over queries here gives the effective per-token importance.
        token_importance = attn.mean(axis=0)

    else:
        raise ValueError(f"Unknown method '{method}'. Choose: mean, maxpool, entropy, rollout")

    # Aggregate per category
    cat_sums: Dict[str, float] = {}
    for cat, (s, e) in TOKEN_RANGES.items():
        cat_sums[cat] = float(token_importance[s:e].sum())

    total = sum(cat_sums.values()) + 1e-12
    return {cat: v / total for cat, v in cat_sums.items()}


def aggregate_attention_all(
    attn: np.ndarray,
    rollout_attn: np.ndarray = None,
) -> Dict[str, Dict[str, float]]:
    """Run all aggregation methods and return results for each.

    Args:
        attn: Shape ``(n_queries, n_tokens)`` or ``(1, n_queries, n_tokens)``.
            Used for mean, maxpool, and entropy methods.
        rollout_attn: Optional rolled-out attention, same shape as ``attn``.
            Required for the ``'rollout'`` method. If None, rollout is skipped.

    Returns:
        Dict ``{method_name: {category: importance}}``.
    """
    results = {
        method: aggregate_attention(attn, method)
        for method in ("mean", "maxpool", "entropy")
    }
    if rollout_attn is not None:
        results["rollout"] = aggregate_attention(rollout_attn, "rollout")
    return results


def per_agent_attention(
    attn: np.ndarray,
    method: AggMethod = "mean",
) -> np.ndarray:
    """Return attention fraction for each of the 8 other agents.

    Args:
        attn: Shape ``(n_queries, n_tokens)`` or ``(1, n_queries, n_tokens)``.
        method: Same aggregation strategy as :func:`aggregate_attention`.

    Returns:
        Numpy array of shape ``(8,)``, normalized so values sum to ~1 relative
        to the full token budget (not relative to agent tokens only).
    """
    attn = np.array(attn)
    if attn.ndim == 3:
        attn = attn[0]

    if method == "mean":
        token_importance = attn.mean(axis=0)
    elif method == "maxpool":
        token_importance = attn.max(axis=0)
    elif method == "entropy":
        eps = 1e-10
        attn_safe = np.clip(attn, eps, 1.0)
        H = -np.sum(attn_safe * np.log2(attn_safe), axis=1)
        weights = 1.0 / (H + eps)
        weights /= weights.sum()
        token_importance = (attn * weights[:, None]).sum(axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")

    total = token_importance.sum() + 1e-12
    agent_start = TOKEN_RANGES["other_agents"][0]  # 5
    tokens_per_agent = 5  # each agent: 5 timesteps × 1 token each

    per_agent = []
    for i in range(8):
        s = agent_start + i * tokens_per_agent
        e = s + tokens_per_agent
        per_agent.append(float(token_importance[s:e].sum() / total))
    return np.array(per_agent)


def query_entropy(attn: np.ndarray) -> np.ndarray:
    """Compute Shannon entropy for each of the 16 queries.

    Useful for diagnosing query specialization: low entropy = sharp/focused,
    high entropy = diffuse/uninformative.

    Args:
        attn: Shape ``(n_queries, n_tokens)`` or ``(1, n_queries, n_tokens)``.

    Returns:
        Numpy array shape ``(16,)`` with entropy in bits per query.
        Maximum possible entropy: log2(280) ≈ 8.13 bits.
    """
    attn = np.array(attn)
    if attn.ndim == 3:
        attn = attn[0]

    eps = 1e-10
    attn_safe = np.clip(attn, eps, 1.0)
    return -np.sum(attn_safe * np.log2(attn_safe), axis=1)  # (16,)
