"""
Cognitive Performance Scores for XAI narration evaluation.

Computes composite scores from per-metric G-Eval results to produce a
``CognitiveScorecard`` — a high-level summary of an LLM's explanation
quality across the dimensions that matter for autonomous driving XAI.

Composite Scores (updated with research-backed metrics)
-------------------------------------------------------
- **Situational Awareness**
    = avg(Perceptual Grounding, Safety Fidelity, Strict Context Grounding)
    Sources: core + DriveBench anti-hallucination

- **Reasoning Quality**
    = avg(Causal Accuracy, Action Justification, Progressive Alignment, Risk Coherence)
    Sources: core + AutoDriDM perception→decision chain + DriveLMM-o1 severity

- **Communication Quality**
    = avg(Conciseness, Logical Completeness)
    Sources: core + DriveLMM-o1 missing-step penalty

- **Overall Cognitive Score**
    = 0.4 × SitAwareness + 0.4 × Reasoning + 0.2 × Communication
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from xai.llm_evaluation.config import CognitiveWeights


# ── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class CognitiveScorecard:
    """Full cognitive assessment for a set of narrations."""

    # ── Per-metric averages (0-1) — Core ─────────────────────────────────
    safety_fidelity: float = 0.0
    perceptual_grounding: float = 0.0
    causal_accuracy: float = 0.0
    action_justification: float = 0.0
    conciseness: float = 0.0

    # ── Per-metric averages (0-1) — Research-backed ──────────────────────
    strict_context_grounding: float = 0.0   # DriveBench
    progressive_alignment: float = 0.0       # AutoDriDM
    logical_completeness: float = 0.0        # DriveLMM-o1
    risk_coherence: float = 0.0              # DriveLMM-o1

    # ── Composite scores (0-1) ───────────────────────────────────────────
    situational_awareness: float = 0.0
    reasoning_quality: float = 0.0
    communication_quality: float = 0.0
    overall_cognitive_score: float = 0.0

    # ── Metadata ─────────────────────────────────────────────────────────
    model_name: str = ""
    num_reports: int = 0
    decision_class_breakdown: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON export."""
        return {
            "model_name": self.model_name,
            "num_reports": self.num_reports,
            "decision_class_breakdown": self.decision_class_breakdown,
            "per_metric": {
                "safety_fidelity": round(self.safety_fidelity, 4),
                "perceptual_grounding": round(self.perceptual_grounding, 4),
                "causal_accuracy": round(self.causal_accuracy, 4),
                "action_justification": round(self.action_justification, 4),
                "conciseness": round(self.conciseness, 4),
                "strict_context_grounding": round(self.strict_context_grounding, 4),
                "progressive_alignment": round(self.progressive_alignment, 4),
                "logical_completeness": round(self.logical_completeness, 4),
                "risk_coherence": round(self.risk_coherence, 4),
            },
            "composite": {
                "situational_awareness": round(self.situational_awareness, 4),
                "reasoning_quality": round(self.reasoning_quality, 4),
                "communication_quality": round(self.communication_quality, 4),
                "overall_cognitive_score": round(self.overall_cognitive_score, 4),
            },
        }

    def summary_table(self) -> str:
        """Pretty-print a summary for terminal output."""
        lines = [
            f"╔══════════════════════════════════════════════════════════╗",
            f"║  Cognitive Scorecard — {self.model_name:<34s}║",
            f"╠══════════════════════════════════════════════════════════╣",
            f"║  Reports evaluated: {self.num_reports:<36d}║",
            f"╠══════════════════════════════════════════════════════════╣",
            f"║  Core Metrics                                           ║",
            f"║    Safety Fidelity         : {self.safety_fidelity:>6.2%}                    ║",
            f"║    Perceptual Grounding    : {self.perceptual_grounding:>6.2%}                    ║",
            f"║    Causal Accuracy         : {self.causal_accuracy:>6.2%}                    ║",
            f"║    Action Justification    : {self.action_justification:>6.2%}                    ║",
            f"║    Conciseness             : {self.conciseness:>6.2%}                    ║",
            f"║  Research-Backed Metrics                                ║",
            f"║    Context Grounding [DB]  : {self.strict_context_grounding:>6.2%}                    ║",
            f"║    Progressive Align [AD]  : {self.progressive_alignment:>6.2%}                    ║",
            f"║    Logical Complete  [DL]  : {self.logical_completeness:>6.2%}                    ║",
            f"║    Risk Coherence    [DL]  : {self.risk_coherence:>6.2%}                    ║",
            f"╠══════════════════════════════════════════════════════════╣",
            f"║  Composite Scores                                       ║",
            f"║    Situational Awareness   : {self.situational_awareness:>6.2%}                    ║",
            f"║    Reasoning Quality       : {self.reasoning_quality:>6.2%}                    ║",
            f"║    Communication Quality   : {self.communication_quality:>6.2%}                    ║",
            f"╠══════════════════════════════════════════════════════════╣",
            f"║  ★ Overall Cognitive Score : {self.overall_cognitive_score:>6.2%}                    ║",
            f"╚══════════════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)


# ── Computation ──────────────────────────────────────────────────────────────

# Map from GEval metric *name* → CognitiveScorecard field
_METRIC_NAME_MAP = {
    # Core
    "Safety Fidelity": "safety_fidelity",
    "Perceptual Grounding": "perceptual_grounding",
    "Causal Accuracy": "causal_accuracy",
    "Action Justification": "action_justification",
    "Conciseness and Constraint Compliance": "conciseness",
    # Research-backed
    "Strict Context Grounding": "strict_context_grounding",
    "Progressive Alignment": "progressive_alignment",
    "Logical Completeness": "logical_completeness",
    "Risk Coherence": "risk_coherence",
}

# Which metrics feed into each composite score
_COMPOSITE_GROUPS = {
    "situational_awareness": [
        "perceptual_grounding",
        "safety_fidelity",
        "strict_context_grounding",     # DriveBench
    ],
    "reasoning_quality": [
        "causal_accuracy",
        "action_justification",
        "progressive_alignment",        # AutoDriDM
        "risk_coherence",               # DriveLMM-o1
    ],
    "communication_quality": [
        "conciseness",
        "logical_completeness",         # DriveLMM-o1
    ],
}


def compute_cognitive_scores(
    per_report_scores: List[Dict[str, float]],
    model_name: str = "",
    decision_classes: Optional[List[str]] = None,
    weights: Optional[CognitiveWeights] = None,
) -> CognitiveScorecard:
    """
    Compute composite cognitive scores from per-report metric results.

    Parameters
    ----------
    per_report_scores
        List of dicts, one per report.  Each dict maps metric name
        (as used by ``GEval.name``) → score (0-1 float).
    model_name
        Label for the evaluated LLM.
    decision_classes
        Optional list of decision classes per-report (for breakdown).
    weights
        Override default cognitive weights.

    Returns
    -------
    CognitiveScorecard
    """
    w = weights or CognitiveWeights()
    n = len(per_report_scores)
    if n == 0:
        return CognitiveScorecard(model_name=model_name)

    # Aggregate per-metric averages
    sums: Dict[str, float] = {k: 0.0 for k in _METRIC_NAME_MAP.values()}
    for report_scores in per_report_scores:
        for metric_name, field_name in _METRIC_NAME_MAP.items():
            sums[field_name] += report_scores.get(metric_name, 0.0)

    avgs = {k: v / n for k, v in sums.items()}

    # Composite scores — equal-weight average within each group
    def _group_avg(group_key: str) -> float:
        fields = _COMPOSITE_GROUPS[group_key]
        return sum(avgs[f] for f in fields) / len(fields)

    sit_awareness = _group_avg("situational_awareness")
    reasoning = _group_avg("reasoning_quality")
    communication = _group_avg("communication_quality")

    overall = (
        w.w_situational_awareness * sit_awareness
        + w.w_reasoning_quality * reasoning
        + w.w_communication_quality * communication
    )

    # Decision class breakdown
    dc_breakdown: Dict[str, int] = {}
    if decision_classes:
        for dc in decision_classes:
            dc_breakdown[dc] = dc_breakdown.get(dc, 0) + 1

    return CognitiveScorecard(
        # Core metrics
        safety_fidelity=avgs["safety_fidelity"],
        perceptual_grounding=avgs["perceptual_grounding"],
        causal_accuracy=avgs["causal_accuracy"],
        action_justification=avgs["action_justification"],
        conciseness=avgs["conciseness"],
        # Research-backed
        strict_context_grounding=avgs["strict_context_grounding"],
        progressive_alignment=avgs["progressive_alignment"],
        logical_completeness=avgs["logical_completeness"],
        risk_coherence=avgs["risk_coherence"],
        # Composites
        situational_awareness=sit_awareness,
        reasoning_quality=reasoning,
        communication_quality=communication,
        overall_cognitive_score=overall,
        # Metadata
        model_name=model_name,
        num_reports=n,
        decision_class_breakdown=dc_breakdown,
    )
