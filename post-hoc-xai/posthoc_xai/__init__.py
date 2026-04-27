"""Post-Hoc XAI Framework for Autonomous Driving.

A modular, JAX-based framework for applying post-hoc explainability
methods to V-MAX autonomous driving policies.

Quick start::

    import posthoc_xai as xai

    # Load a model (auto-selects wrapper based on encoder type)
    model = xai.load_model("runs_rlc/womd_sac_road_perceiver_minimal_42")

    # Get an observation and run vanilla gradient attribution
    method = xai.VanillaGradient(model)
    attribution = method(observation)

    # Inspect per-category importance
    print(attribution.category_importance)

    # Or run all methods at once
    results = xai.explain(model, observation)
"""

from __future__ import annotations

from pathlib import Path

from posthoc_xai.models.base import ExplainableModel, ModelOutput
from posthoc_xai.models.loader import load_vmax_model, LoadedVMAXModel
from posthoc_xai.models.perceiver_wrapper import PerceiverWrapper
from posthoc_xai.models.generic_wrapper import GenericWrapper
from posthoc_xai.methods.base import Attribution, AttributionMethod
from posthoc_xai.methods.vanilla_gradient import VanillaGradient
from posthoc_xai.methods.integrated_gradients import IntegratedGradients
from posthoc_xai.methods.smooth_grad import SmoothGrad
from posthoc_xai.methods.gradient_x_input import GradientXInput
from posthoc_xai.methods.perturbation import PerturbationAttribution
from posthoc_xai.methods.feature_ablation import FeatureAblation
from posthoc_xai.methods.sarfa import SARFA
from posthoc_xai.methods import METHOD_REGISTRY

# Encoder types that get the Perceiver wrapper (with attention extraction)
_PERCEIVER_TYPES = {"perceiver", "lq"}


def load_model(
    model_dir: str | Path,
    data_path: str | Path | None = None,
    max_num_objects: int = 64,
    vmax_repo: str | None = None,
) -> ExplainableModel:
    """Load a V-MAX model and return the appropriate ExplainableModel wrapper.

    This is the main entry point for the framework.

    Args:
        model_dir: Path to a model directory inside ``runs_rlc/``.
        data_path: Optional path to a ``.tfrecord`` for the data generator.
        max_num_objects: Max objects per scenario.
        vmax_repo: Optional explicit path to the V-Max repo.

    Returns:
        An ``ExplainableModel`` wrapper (PerceiverWrapper or GenericWrapper).
    """
    loaded = load_vmax_model(
        model_dir=model_dir,
        data_path=data_path,
        max_num_objects=max_num_objects,
        vmax_repo=vmax_repo,
    )

    if loaded.original_encoder_type in _PERCEIVER_TYPES:
        return PerceiverWrapper(loaded)
    else:
        return GenericWrapper(loaded)


def explain(
    model: ExplainableModel | str | Path,
    observation,
    methods: list[str] | None = None,
    target_action: int | None = None,
) -> dict[str, Attribution]:
    """Run multiple XAI methods on a single observation.

    Args:
        model: An ``ExplainableModel`` or path to a model directory.
        observation: Input observation array (unbatched).
        methods: List of method names (see ``METHOD_REGISTRY``).
            Defaults to ``['vanilla_gradient', 'integrated_gradients', 'perturbation']``.
        target_action: Which action dimension to explain (``None`` → all).

    Returns:
        Dict mapping method name to ``Attribution`` result.
    """
    if isinstance(model, (str, Path)):
        model = load_model(model)

    if methods is None:
        methods = ["vanilla_gradient", "integrated_gradients", "perturbation"]

    results: dict[str, Attribution] = {}
    for method_name in methods:
        if method_name not in METHOD_REGISTRY:
            raise ValueError(
                f"Unknown method '{method_name}'. "
                f"Available: {list(METHOD_REGISTRY.keys())}"
            )
        method = METHOD_REGISTRY[method_name](model)
        results[method_name] = method(observation, target_action)

    return results


__all__ = [
    # Loading
    "load_model",
    "load_vmax_model",
    "LoadedVMAXModel",
    # Model interface
    "ExplainableModel",
    "ModelOutput",
    "PerceiverWrapper",
    "GenericWrapper",
    # Methods
    "Attribution",
    "AttributionMethod",
    "VanillaGradient",
    "IntegratedGradients",
    "SmoothGrad",
    "GradientXInput",
    "PerturbationAttribution",
    "FeatureAblation",
    "SARFA",
    "METHOD_REGISTRY",
    # Convenience
    "explain",
]
