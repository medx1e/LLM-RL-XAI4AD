"""Attribution / XAI methods."""

from posthoc_xai.methods.base import Attribution, AttributionMethod
from posthoc_xai.methods.vanilla_gradient import VanillaGradient
from posthoc_xai.methods.integrated_gradients import IntegratedGradients
from posthoc_xai.methods.smooth_grad import SmoothGrad
from posthoc_xai.methods.gradient_x_input import GradientXInput
from posthoc_xai.methods.perturbation import PerturbationAttribution
from posthoc_xai.methods.feature_ablation import FeatureAblation
from posthoc_xai.methods.sarfa import SARFA, sarfa_batch

# Map names to classes for the `explain()` convenience function
METHOD_REGISTRY: dict[str, type[AttributionMethod]] = {
    "vanilla_gradient": VanillaGradient,
    "integrated_gradients": IntegratedGradients,
    "smooth_grad": SmoothGrad,
    "gradient_x_input": GradientXInput,
    "perturbation": PerturbationAttribution,
    "feature_ablation": FeatureAblation,
    "sarfa": SARFA,
}

__all__ = [
    "Attribution",
    "AttributionMethod",
    "VanillaGradient",
    "IntegratedGradients",
    "SmoothGrad",
    "GradientXInput",
    "PerturbationAttribution",
    "FeatureAblation",
    "SARFA",
    "sarfa_batch",
    "METHOD_REGISTRY",
]
