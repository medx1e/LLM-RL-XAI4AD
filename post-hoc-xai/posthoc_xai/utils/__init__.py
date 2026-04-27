"""Shared utilities."""

from posthoc_xai.utils.normalization import (
    size_correct_attribution,
    size_correct_attention,
    correction_factors,
    CATEGORY_FEATURE_COUNTS,
    CATEGORY_TOKEN_COUNTS,
)
from posthoc_xai.utils.ig_baseline import (
    detect_binary_features,
    compute_baseline,
    compute_baseline_stats,
    BaselineAccumulator,
)
from posthoc_xai.utils.attention_aggregation import (
    aggregate_attention,
    aggregate_attention_all,
    per_agent_attention,
    query_entropy,
    TOKEN_RANGES,
)

__all__ = [
    "detect_binary_features",
    "compute_baseline",
    "compute_baseline_stats",
    "BaselineAccumulator",
    "size_correct_attribution",
    "size_correct_attention",
    "correction_factors",
    "CATEGORY_FEATURE_COUNTS",
    "CATEGORY_TOKEN_COUNTS",
    "aggregate_attention",
    "aggregate_attention_all",
    "per_agent_attention",
    "query_entropy",
    "TOKEN_RANGES",
]
