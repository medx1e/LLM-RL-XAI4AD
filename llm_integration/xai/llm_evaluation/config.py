"""
Evaluation configuration for the LLM narration evaluation framework.

Centralises judge-model settings, metric thresholds, cognitive-score
weights, and the model registry for multi-LLM comparison.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_JUDGE_MODEL = "gpt-4o"
DEFAULT_THRESHOLD = 0.7          # pass/fail threshold for individual metrics
DEFAULT_MAX_RETRIES = 3          # retries on judge API failures


# ── Cognitive-score weights ──────────────────────────────────────────────────

@dataclass
class CognitiveWeights:
    """Weights used to combine per-metric results into composite scores."""

    # Situational Awareness = w_pg * PerceptualGrounding + w_sf * SafetyFidelity
    w_perceptual_grounding: float = 0.5
    w_safety_fidelity: float = 0.5

    # Reasoning Quality = w_ca * CausalAccuracy + w_aj * ActionJustification
    w_causal_accuracy: float = 0.5
    w_action_justification: float = 0.5

    # Overall = w_sa * SituationalAwareness + w_rq * ReasoningQuality + w_cq * CommunicationQuality
    w_situational_awareness: float = 0.4
    w_reasoning_quality: float = 0.4
    w_communication_quality: float = 0.2


# ── Model registry ──────────────────────────────────────────────────────────

@dataclass
class ModelEntry:
    """Represents one LLM whose narrations we want to evaluate."""
    name: str
    reports_dir: str
    description: str = ""


@dataclass
class EvalConfig:
    """Top-level evaluation configuration."""

    judge_model: str = DEFAULT_JUDGE_MODEL
    threshold: float = DEFAULT_THRESHOLD
    max_retries: int = DEFAULT_MAX_RETRIES

    cognitive_weights: CognitiveWeights = field(default_factory=CognitiveWeights)

    # Registry of LLMs to evaluate — populated at runtime or via YAML
    model_registry: List[ModelEntry] = field(default_factory=list)

    # Optional: filter reports by decision class(es)
    decision_class_filter: Optional[List[str]] = None

    ignore_errors: bool = False


def default_config() -> EvalConfig:
    """Return a fresh default config."""
    return EvalConfig()
