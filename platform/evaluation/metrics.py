"""Attribution quality metrics — pure NumPy, no JAX/model calls required.

All functions work on lists of Attribution objects loaded from the cache.
They return plain dicts so the tab can render them without importing this module
into a JAX environment.
"""

from __future__ import annotations

from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Sparsity
# ---------------------------------------------------------------------------

def sparsity(attr_raw: np.ndarray, threshold: float = 0.01) -> float:
    """Fraction of attribution values whose |value| < threshold (relative to max).

    A score of 0.90 means 90% of features are effectively zero — the method
    is very focused.  Low sparsity (e.g. 0.30) means attribution is spread
    across many features.
    """
    flat = np.abs(attr_raw.ravel())
    max_val = flat.max()
    if max_val == 0:
        return 1.0
    return float((flat < threshold * max_val).mean())


# ---------------------------------------------------------------------------
# Concentration  (Gini coefficient of absolute attributions)
# ---------------------------------------------------------------------------

def gini_concentration(attr_raw: np.ndarray) -> float:
    """Gini coefficient of |attribution| values.

    0 = perfectly uniform (every feature gets equal weight).
    1 = perfectly concentrated (single feature gets everything).

    Higher = method is more focused / decisive.
    """
    flat = np.abs(attr_raw.ravel()).astype(float)
    if flat.sum() == 0:
        return 0.0
    flat = np.sort(flat)
    n = len(flat)
    cumulative = np.cumsum(flat)
    return float((2 * cumulative.sum()) / (n * flat.sum()) - (n + 1) / n)


# ---------------------------------------------------------------------------
# Top-K coverage
# ---------------------------------------------------------------------------

def topk_coverage(attr_raw: np.ndarray, k: float = 0.1) -> float:
    """Fraction of total attribution mass carried by the top-k fraction of features.

    ``k=0.1`` → "what % of mass do the top 10% features carry?"
    A high score (e.g. 0.80) means the method is selective.
    """
    flat = np.abs(attr_raw.ravel())
    if flat.sum() == 0:
        return 0.0
    n_top = max(1, int(len(flat) * k))
    idx = np.argpartition(flat, -n_top)[-n_top:]
    return float(flat[idx].sum() / flat.sum())


# ---------------------------------------------------------------------------
# Temporal stability
# ---------------------------------------------------------------------------

def temporal_rank_stability(series: list, top_k: int = 20) -> float:
    """Mean Jaccard similarity of top-K feature sets across consecutive timesteps.

    1.0 = identical top-K every step (perfectly stable).
    0.0 = completely different top-K every step.

    Useful for checking whether the method is consistent across timesteps
    rather than flipping important features arbitrarily.
    """
    valid = [s for s in series if s is not None]
    if len(valid) < 2:
        return float("nan")

    def topk_set(attr) -> set:
        flat = np.abs(np.array(attr.raw).ravel())
        idx = np.argpartition(flat, -min(top_k, len(flat)))[-min(top_k, len(flat)):]
        return set(idx.tolist())

    jaccs = []
    for a, b in zip(valid[:-1], valid[1:]):
        sa, sb = topk_set(a), topk_set(b)
        union = len(sa | sb)
        if union == 0:
            continue
        jaccs.append(len(sa & sb) / union)

    return float(np.mean(jaccs)) if jaccs else float("nan")


# ---------------------------------------------------------------------------
# Method agreement (pairwise)
# ---------------------------------------------------------------------------

def spearman_rho(a_raw: np.ndarray, b_raw: np.ndarray) -> float:
    """Spearman rank correlation between two attribution arrays.

    Values close to 1.0 mean the two methods rank features the same way.
    """
    from scipy.stats import spearmanr
    fa = np.abs(a_raw.ravel()).astype(float)
    fb = np.abs(b_raw.ravel()).astype(float)
    if len(fa) != len(fb) or len(fa) < 2:
        return float("nan")
    rho, _ = spearmanr(fa, fb)
    return float(rho)


def pairwise_agreement(series_map: dict[str, list], step: int) -> dict[tuple[str, str], float]:
    """Compute all pairwise Spearman rho values for the given timestep.

    ``series_map`` maps method name → list[Attribution].
    Returns a dict of (method_a, method_b) → rho.
    """
    names = list(series_map.keys())
    result: dict[tuple[str, str], float] = {}
    for i, na in enumerate(names):
        for j, nb in enumerate(names):
            if i >= j:
                continue
            sa = series_map[na]
            sb = series_map[nb]
            if (sa is None or step >= len(sa) or sa[step] is None or
                    sb is None or step >= len(sb) or sb[step] is None):
                result[(na, nb)] = float("nan")
                continue
            result[(na, nb)] = spearman_rho(
                np.array(sa[step].raw), np.array(sb[step].raw)
            )
    return result


# ---------------------------------------------------------------------------
# Category dominance
# ---------------------------------------------------------------------------

def category_dominance(attribution) -> tuple[str, float]:
    """Return (dominant_category, fraction) for an Attribution at one timestep."""
    cat_imp: dict = attribution.category_importance
    if not cat_imp:
        return ("unknown", float("nan"))
    total = sum(abs(float(v)) for v in cat_imp.values()) or 1.0
    best_k = max(cat_imp, key=lambda k: abs(float(cat_imp[k])))
    return (best_k, abs(float(cat_imp[best_k])) / total)


# ---------------------------------------------------------------------------
# Per-method profile  (used by the report tab)
# ---------------------------------------------------------------------------

def method_profile(series: list, step: int) -> dict:
    """Compute all fast metrics for one method at one timestep.

    Returns a dict with keys: sparsity, gini, topk10, top_cat, top_cat_frac,
    temporal_stability (computed over full series).
    Gracefully returns NaN entries if the step is missing.
    """
    if series is None or step >= len(series) or series[step] is None:
        return dict(sparsity=float("nan"), gini=float("nan"), topk10=float("nan"),
                    top_cat="n/a", top_cat_frac=float("nan"),
                    temporal_stability=float("nan"))

    attr = series[step]
    raw = np.array(attr.raw)
    sp  = sparsity(raw)
    gn  = gini_concentration(raw)
    tk  = topk_coverage(raw, k=0.1)
    tc, tf = category_dominance(attr)
    ts  = temporal_rank_stability(series)
    return dict(sparsity=sp, gini=gn, topk10=tk,
                top_cat=tc, top_cat_frac=tf,
                temporal_stability=ts)


# ---------------------------------------------------------------------------
# Attention–Attribution alignment
# ---------------------------------------------------------------------------

# Perceiver observation layout (mirrors viz.py constants)
_ATTN_AGENT_START = 5
_ATTN_N_AGENTS    = 8


def _agent_slot_importances_from_raw(attr_raw: np.ndarray, n_agents: int = 8) -> list[float]:
    """Sum absolute attribution over the 5 observation tokens per agent slot.

    Assumes the flat Perceiver observation layout where tokens 5–44 are
    8 agents × 5 timestep features each.
    """
    flat = np.abs(np.array(attr_raw).ravel())
    result = []
    for i in range(n_agents):
        s = _ATTN_AGENT_START + i * 5
        e = s + 5
        result.append(float(flat[s:e].sum()) if e <= len(flat) else float("nan"))
    return result


def attention_attribution_alignment(
    attn_series: list,
    attr_series_map: dict,
    artifact,
    n_agents: int = _ATTN_N_AGENTS,
) -> dict[str, list[float]]:
    """Spearman ρ between attention weight and attribution importance per agent slot.

    For each (method, agent slot), we collect T-length time series of:
      - attention weight on that slot (from aggregate_attention_by_entity)
      - attribution importance on that slot (summed raw attribution tokens)

    Then compute Spearman ρ.  ρ ≈ 1 → method and attention agree on this agent.
    ρ ≈ 0 → method sees no relation to attention on this agent.

    Returns
    -------
    dict mapping method_name → list of rho values (length n_agents).
    NaN if fewer than 3 valid timesteps.
    """
    from scipy.stats import spearmanr
    # Import here to avoid circular imports; viz has no dep on metrics
    from platform.posthoc.viz import aggregate_attention_by_entity

    # Build per-slot attention time series
    attn_by_slot: list[list[float]] = [[] for _ in range(n_agents)]
    valid_steps: list[int] = []

    for t, attn_dict in enumerate(attn_series):
        if attn_dict is None:
            continue
        entity_attn = aggregate_attention_by_entity(attn_dict, artifact=artifact, step=t)
        step_vals = []
        for i in range(n_agents):
            val = 0.0
            for k, v in entity_attn.items():
                if k.startswith(f"A{i} "):
                    val = float(v)
                    break
            step_vals.append(val)
        for i, v in enumerate(step_vals):
            attn_by_slot[i].append(v)
        valid_steps.append(t)

    if len(valid_steps) < 3:
        nan_row = [float("nan")] * n_agents
        return {m: nan_row[:] for m in attr_series_map}

    result: dict[str, list[float]] = {}
    for method, series in attr_series_map.items():
        rhos: list[float] = []
        for i in range(n_agents):
            attr_vals = []
            for t in valid_steps:
                if series is None or t >= len(series) or series[t] is None:
                    attr_vals.append(float("nan"))
                else:
                    slots = _agent_slot_importances_from_raw(series[t].raw, n_agents)
                    attr_vals.append(slots[i])

            a_arr = np.array(attn_by_slot[i])
            b_arr = np.array(attr_vals)
            mask = ~(np.isnan(a_arr) | np.isnan(b_arr))
            if mask.sum() < 3:
                rhos.append(float("nan"))
            else:
                rho, _ = spearmanr(a_arr[mask], b_arr[mask])
                rhos.append(float(rho))
        result[method] = rhos
    return result
