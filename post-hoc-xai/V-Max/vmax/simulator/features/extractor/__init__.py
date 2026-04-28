# Copyright 2025 Valeo.


"""Feature extractors for various modalities."""

from .abstract_extractor import AbstractFeaturesExtractor  # noqa: I001
from .vec_extractor import VecFeaturesExtractor
from .gt_extractor import GTFeaturesExtractor
from .idm_extractor import IDMFeaturesExtractor
from .utils import OBJECT_MAPPING, RG_MAPPING, TL_MAPPING, get_feature_size, normalize_by_feature, normalize_path


__all__ = [
    "OBJECT_MAPPING",
    "RG_MAPPING",
    "TL_MAPPING",
    "AbstractFeaturesExtractor",
    "GTFeaturesExtractor",
    "IDMFeaturesExtractor",
    "VecFeaturesExtractor",
    "get_feature_size",
    "normalize_by_feature",
    "normalize_path",
]


def get_extractor(name: str) -> AbstractFeaturesExtractor:
    """Return the feature extractor based on the name.

    Args:
        name: Name of the feature extractor.

    Returns:
        The feature extractor.

    """
    mapping = {
        "vec": VecFeaturesExtractor,
        "gt": GTFeaturesExtractor,
        "idm": IDMFeaturesExtractor,
    }

    if name not in mapping:
        raise ValueError(f"Unknown feature extractor: {name}")

    return mapping[name]
