"""Attribution normalization utilities.

Provides alternatives to the default L1-normalization-of-absolutes pipeline,
and attention density correction for fair cross-category comparison.
"""

from typing import Dict

# --- Observation-space constants (feature counts per category) ---------------
# Flat observation: 1,655 features total
# sdc_trajectory : indices [0,   40)  →  40 features  (1 entity × 5ts × 8 feats)
# other_agents   : indices [40,  360) → 320 features  (8 agents × 5ts × 8 feats)
# roadgraph      : indices [360, 1360)→ 1000 features (200 points × 5 feats)
# traffic_lights : indices [1360,1635)→ 275 features  (5 lights × 5ts × 11 feats)
# gps_path       : indices [1635,1655)→  20 features  (10 waypoints × 2 feats)

CATEGORY_FEATURE_COUNTS: Dict[str, int] = {
    "sdc_trajectory": 40,
    "other_agents": 320,
    "roadgraph": 1000,
    "traffic_lights": 275,
    "gps_path": 20,
}

# --- Attention-space constants (token counts per category) -------------------
# Cross-attention operates on 280 encoded tokens (one token per entity/point)
# sdc_trajectory : 5  tokens  (1 entity × 5 timesteps)
# other_agents   : 40 tokens  (8 agents × 5 timesteps)
# roadgraph      : 200 tokens (200 road points, no temporal)
# traffic_lights : 25 tokens  (5 lights × 5 timesteps)
# gps_path       : 10 tokens  (10 waypoints, no temporal)

CATEGORY_TOKEN_COUNTS: Dict[str, int] = {
    "sdc_trajectory": 5,
    "other_agents": 40,
    "roadgraph": 200,
    "traffic_lights": 25,
    "gps_path": 10,
}


def size_correct_attribution(
    category_importance: Dict[str, float],
    feature_counts: Dict[str, int] = None,
) -> Dict[str, float]:
    """Apply per-feature size correction to category-level attribution.

    The default L1-aggregation sums abs(attribution) over all features in a
    category. Larger categories (roadgraph: 1000 features) accumulate more
    attribution than smaller ones (gps_path: 20 features) purely from size.

    This correction divides each category's raw sum by its feature count before
    renormalizing, giving importance *per feature* rather than importance *total*.

    Args:
        category_importance: Dict mapping category name to importance fraction
            (values should sum to ~1.0).
        feature_counts: Override feature counts. Defaults to CATEGORY_FEATURE_COUNTS.

    Returns:
        Size-corrected dict with the same keys, renormalized to sum to 1.

    Note:
        This answers "which category has the most influence per input dimension?"
        The uncorrected version answers "which category has the most total influence?"
        Both perspectives are valid and should be reported together.
    """
    counts = feature_counts or CATEGORY_FEATURE_COUNTS

    corrected = {}
    for cat, imp in category_importance.items():
        n = counts.get(cat, 1)
        corrected[cat] = imp / n

    total = sum(corrected.values()) + 1e-10
    return {cat: v / total for cat, v in corrected.items()}


def size_correct_attention(
    category_attention: Dict[str, float],
    token_counts: Dict[str, int] = None,
) -> Dict[str, float]:
    """Apply per-token size correction to category-level attention.

    Same idea as size_correct_attribution but for attention weights aggregated
    over tokens. Roadgraph has 200 tokens vs GPS's 10, so raw attention fractions
    are biased toward roadgraph by construction.

    Args:
        category_attention: Dict mapping category name to attention fraction.
        token_counts: Override token counts. Defaults to CATEGORY_TOKEN_COUNTS.

    Returns:
        Size-corrected dict renormalized to sum to 1.
    """
    counts = token_counts or CATEGORY_TOKEN_COUNTS

    corrected = {}
    for cat, attn in category_attention.items():
        n = counts.get(cat, 1)
        corrected[cat] = attn / n

    total = sum(corrected.values()) + 1e-10
    return {cat: v / total for cat, v in corrected.items()}


def correction_factors() -> Dict[str, float]:
    """Return the relative size-correction multiplier for each category.

    Useful for understanding how much each category is amplified or suppressed.
    Values > 1 mean the category gets amplified (it was penalized by small size).
    Values < 1 mean the category gets suppressed (it was inflated by large size).

    The reference is gps_path (20 features), the smallest category.
    """
    min_count = min(CATEGORY_FEATURE_COUNTS.values())  # 20 (gps_path)
    return {cat: min_count / n for cat, n in CATEGORY_FEATURE_COUNTS.items()}
