"""
Custom G-Eval metrics for evaluating autonomous-driving XAI narrations.

Each metric is a DeepEval ``GEval`` instance with domain-specific criteria
and step-by-step evaluation instructions for the LLM judge.

Core Metrics
------------
1. Safety Fidelity
2. Perceptual Grounding
3. Causal Accuracy
4. Action Justification
5. Conciseness & Constraint Compliance

Research-Backed Metrics
-----------------------
6. Strict Context Grounding  (DriveBench)
7. Progressive Alignment      (AutoDriDM)
8. Logical Completeness        (DriveLMM-o1)
9. Risk Coherence              (DriveLMM-o1)
"""

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

from xai.llm_evaluation.config import DEFAULT_THRESHOLD


# ═════════════════════════════════════════════════════════════════════════════
# 1. Safety Fidelity
# ═════════════════════════════════════════════════════════════════════════════

def safety_fidelity_metric(threshold: float = DEFAULT_THRESHOLD, model: str = None) -> GEval:
    """Does the narration correctly report collision outcomes and threats?"""
    return GEval(
        name="Safety Fidelity",
         
        criteria=(
            "Evaluate whether the narration accurately reflects the safety-critical "
            "information present in the structured driving report. The narration must "
            "correctly describe collision outcomes, threat agent identifiers, and "
            "time-to-collision values without inventing dangers that do not exist in "
            "the report."
        ),
        evaluation_steps=[
            "Check that every collision outcome mentioned in the narration matches "
            "a collision listed in the report's 'alternatives' array.",
            "Verify that threat agent IDs cited in the narration exist in the report "
            "(either in 'alternatives' or 'attention_grounding.per_agent_breakdown').",
            "Confirm that time-to-collision (TTC) values mentioned are consistent "
            "with those in the report.",
            "Penalise the narration if it invents or hallucinated dangers, agents, "
            "or safety events not present in the report.",
            "Reward the narration if it correctly identifies that the chosen action "
            "outcome is SAFE and the most dangerous alternative would cause a collision.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=threshold,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 2. Perceptual Grounding
# ═════════════════════════════════════════════════════════════════════════════

def perceptual_grounding_metric(threshold: float = DEFAULT_THRESHOLD, model: str = None) -> GEval:
    """Does the narration reference what the attention system actually saw?"""
    return GEval(
        name="Perceptual Grounding",
         
        criteria=(
            "Evaluate whether the narration is grounded in the attention data from "
            "the RL agent's perception system. The narration should reference agents "
            "that appear in the 'attention_grounding' section of the report, correctly "
            "describe the grounding score, and not attribute awareness to agents the "
            "model did not attend to."
        ),
        evaluation_steps=[
            "Check that agents mentioned in the narration as 'attended to' or 'noticed' "
            "appear in the report's 'attention_grounding.per_agent_breakdown'.",
            "Verify that the grounding score value cited is consistent with "
            "'attention_grounding.grounding_score' in the report.",
            "For UNGROUNDED_CRITICAL cases, confirm the narration includes a caveat "
            "about attention NOT confirming awareness of threats.",
            "Penalise if the narration claims the model 'focused on' or 'was aware of' "
            "agents whose attention_mass is negligible (< 0.05).",
            "Confirm the narration correctly ranks the most-attended agents when "
            "multiple agents are mentioned.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=threshold,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 3. Causal Accuracy
# ═════════════════════════════════════════════════════════════════════════════

def causal_accuracy_metric(threshold: float = DEFAULT_THRESHOLD, model: str = None) -> GEval:
    """Is the cause-effect reasoning consistent with the report data?"""
    return GEval(
        name="Causal Accuracy",
         
        criteria=(
            "Evaluate whether the narration's causal reasoning (action → outcome → "
            "why) is logically consistent with the structured driving report. The "
            "cause-and-effect chain described must be traceable to data in the report "
            "and must not introduce speculative reasoning."
        ),
        evaluation_steps=[
            "Verify that the chosen action label in the narration matches "
            "'chosen_action.label' in the report.",
            "Check that the necessity score interpretation is correct — e.g., if "
            "necessity is 0.75, the narration should convey that ~75% of alternatives "
            "would fail, not that there is a 75% chance of collision.",
            "Verify counterfactual reasoning: if the narration describes 'what would "
            "have happened', it must match an alternative listed in the report.",
            "Penalise if the narration invents causal chains not supported by the data "
            "(e.g., attributing the decision to lane markings when no roadgraph data "
            "is in the report).",
            "Check that the ego vehicle's state (velocity, action) is correctly cited.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=threshold,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 4. Action Justification
# ═════════════════════════════════════════════════════════════════════════════

def action_justification_metric(threshold: float = DEFAULT_THRESHOLD, model: str = None) -> GEval:
    """Does the narration justify why the chosen action beats alternatives?"""
    return GEval(
        name="Action Justification",
         
        criteria=(
            "Evaluate whether the narration provides a convincing justification for "
            "why the agent chose its action over the available alternatives. The "
            "explanation should compare the chosen action against alternatives and "
            "identify the most dangerous one."
        ),
        evaluation_steps=[
            "Check that the narration explains WHY the chosen action was preferred, "
            "not just what it was.",
            "Verify that at least one alternative action is contrasted with the "
            "chosen action.",
            "Confirm the 'most dangerous alternative' identified in the narration "
            "matches the alternative with the worst outcome (lowest TTC or COLLISION) "
            "in the report.",
            "Check that the decision class (GROUNDED_CRITICAL, ROUTINE, etc.) is "
            "reflected in the tone and depth of the justification.",
            "Penalise if the narration provides a justification but bases it on "
            "information not present in the report.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=threshold,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 5. Conciseness & Constraint Compliance
# ═════════════════════════════════════════════════════════════════════════════

def conciseness_metric(threshold: float = DEFAULT_THRESHOLD, model: str = None) -> GEval:
    """Does the narration respect the system prompt constraints?"""
    return GEval(
        name="Conciseness and Constraint Compliance",
         
        criteria=(
            "Evaluate whether the narration complies with the system prompt "
            "constraints: (a) 3–5 sentences in length, (b) no speculation or "
            "claims about what the model 'might' be thinking, (c) only references "
            "information present in the structured report, and (d) covers the three "
            "required topics: what the vehicle did, the most dangerous alternative, "
            "and whether attention confirms the decision."
        ),
        evaluation_steps=[
            "Count the number of sentences in the narration. It should be between "
            "3 and 5 sentences (inclusive).",
            "Check for speculative language — phrases like 'might have', 'could have "
            "been thinking', 'probably intended', 'seems to' suggest speculation.",
            "Verify the narration does NOT introduce external knowledge or context "
            "beyond what is in the structured report.",
            "Confirm the narration addresses all three required topics: "
            "(1) what the vehicle did and why, (2) the most dangerous alternative, "
            "(3) whether attention confirms the decision.",
            "Penalise verbose or repetitive narrations that could be shorter without "
            "losing essential information.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=threshold,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 6. Strict Context Grounding  (inspired by DriveBench)
# ═════════════════════════════════════════════════════════════════════════════

def strict_context_grounding_metric(threshold: float = DEFAULT_THRESHOLD, model: str = None) -> GEval:
    """Anti-hallucination: every noun/agent/feature must exist in the report.

    Inspired by DriveBench's "Context Grounding" methodology which exposes
    models that give the right answer for the wrong reason by testing
    text-only vs. visual baselines.
    """
    return GEval(
        name="Strict Context Grounding",
         
        criteria=(
            "Evaluate whether EVERY entity (agent, road feature, object, or "
            "numeric value) mentioned in the narration is explicitly present in "
            "the structured report JSON. The narration must not hallucinate any "
            "noun, agent ID, road section, distance, or feature that cannot be "
            "traced directly to a field in the input report. This is a strict "
            "anti-hallucination check."
        ),
        evaluation_steps=[
            "List every agent ID mentioned in the narration. Verify each one "
            "exists in 'scene_description', 'alternatives', or "
            "'attention_grounding.per_agent_breakdown' of the report.",
            "List every numeric value in the narration (distances, speeds, TTC, "
            "scores). Verify each one matches a value in the report.",
            "Check for hallucinated road features — e.g., mentions of "
            "'intersection', 'lane', 'traffic light', 'road section' that do "
            "not appear in the report's 'scene_description' or 'context_categories'.",
            "Severely penalise any entity or fact that is fabricated — the model "
            "must ONLY use the JSON as its 'vision'.",
            "A narration that paraphrases report data accurately (even if it "
            "uses different words) should still score well, as long as no new "
            "facts are introduced.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=threshold,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 7. Progressive Alignment  (inspired by AutoDriDM)
# ═════════════════════════════════════════════════════════════════════════════

def progressive_alignment_metric(threshold: float = DEFAULT_THRESHOLD, model: str = None) -> GEval:
    """Three-level alignment: Object → Scene → Decision.

    Inspired by AutoDriDM's progressive evaluation across perception,
    scene understanding, and decision-making dimensions.  Critically,
    if the SLM fails at the Object level, a correct Decision-level
    answer should still be penalised.
    """
    return GEval(
        name="Progressive Alignment",
         
        criteria=(
            "Evaluate the narration across three progressive alignment levels. "
            "Each level builds on the previous — getting a higher level right "
            "while failing a lower level indicates the model is guessing rather "
            "than reasoning from perception.\n\n"
            "Level 1 — OBJECT: Does the narration correctly identify the exact "
            "threat agent IDs from the report?\n"
            "Level 2 — SCENE: Does the narration accurately describe the "
            "spatial relations and physics (distances, relative positions, TTC) "
            "from scene_description?\n"
            "Level 3 — DECISION: Does the narration correctly link the "
            "chosen_action to the attention_grounding score and decision_class?"
        ),
        evaluation_steps=[
            "OBJECT CHECK: Verify the narration references the correct "
            "threat_agent_id values from the report. Confusing Agent 1 with "
            "Agent 11, or citing a non-existent agent, is a Level-1 failure.",
            "SCENE CHECK: Verify spatial descriptions match scene_description — "
            "distances, relations (e.g., 'ahead', 'approaching'), and TTC "
            "values must be accurate.",
            "DECISION CHECK: Verify the narration correctly connects the "
            "chosen action to the grounding score and decision class. E.g., a "
            "GROUNDED_CRITICAL narration should link the action to attention "
            "confirming threat awareness.",
            "PROGRESSIVE PENALTY: If Level 1 (Object) fails but Level 3 "
            "(Decision) appears correct, apply a harsh penalty — the model is "
            "likely guessing the conclusion without grounding it in perception.",
            "Award high scores only when all three levels are simultaneously "
            "correct and logically consistent with each other.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=threshold,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 8. Logical Completeness  (inspired by DriveLMM-o1)
# ═════════════════════════════════════════════════════════════════════════════

def logical_completeness_metric(threshold: float = DEFAULT_THRESHOLD, model: str = None) -> GEval:
    """Missing-step penalty: all three logical chain links must be present.

    Inspired by DriveLMM-o1's evaluation of step-by-step reasoning
    sequences and penalisation of missing logical steps.
    """
    return GEval(
        name="Logical Completeness",
         
        criteria=(
            "Evaluate whether the narration contains all three required "
            "logical chain links from the system prompt, and penalise any "
            "missing step. The three links are:\n"
            "(a) WHAT: What the vehicle did and why (action + justification).\n"
            "(b) DANGER: What the most dangerous alternative would have caused.\n"
            "(c) ATTENTION: Whether the attention distribution confirms the "
            "decision.\n\n"
            "Each missing link is a significant deduction — a narration that "
            "covers only 2 of 3 is substantially incomplete."
        ),
        evaluation_steps=[
            "Check for Link (a) — WHAT: Does the narration state the chosen "
            "action AND provide a reason for it? Simply naming the action "
            "without any justification is a partial failure.",
            "Check for Link (b) — DANGER: Does the narration describe at least "
            "one dangerous alternative and its potential consequence (collision, "
            "TTC)? If no alternatives are discussed, this link is missing.",
            "Check for Link (c) — ATTENTION: Does the narration reference the "
            "attention grounding — either confirming awareness or flagging a "
            "lack of grounding (UNGROUNDED caveat)? If attention is never "
            "mentioned, this link is missing.",
            "Apply a -0.33 deduction for each missing link (approximate — the "
            "score should reflect how many of the 3 steps are present).",
            "A narration covering all 3 links with reasonable accuracy should "
            "score ≥ 0.8, even if the prose is imperfect.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=threshold,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 9. Risk Coherence  (inspired by DriveLMM-o1)
# ═════════════════════════════════════════════════════════════════════════════

def risk_coherence_metric(threshold: float = DEFAULT_THRESHOLD, model: str = None) -> GEval:
    """Tone must match severity: high necessity → urgent language, not 'routine'.

    Inspired by DriveLMM-o1's emphasis on accurate severity conveyance
    in step-by-step risk assessment.
    """
    return GEval(
        name="Risk Coherence",
         
        criteria=(
            "Evaluate whether the narration's tone and urgency are coherent "
            "with the risk level indicated by the report's necessity_score and "
            "decision_class. A high-necessity, critical scenario must NOT be "
            "described with calm, routine language, and vice versa. The "
            "severity conveyed in the prose must match the data."
        ),
        evaluation_steps=[
            "Read the necessity_score from the report. Scores ≥ 0.7 indicate "
            "high criticality; scores < 0.3 indicate routine conditions.",
            "Read the decision_class. GROUNDED_CRITICAL and UNGROUNDED_CRITICAL "
            "should trigger urgent, serious language. ROUTINE and "
            "GROUNDED_ROUTINE should use calm, matter-of-fact language.",
            "Check for MISMATCHES: If necessity_score ≥ 0.7 but the narration "
            "says 'routine driving conditions' or 'no significant threats', "
            "this is a severe coherence failure.",
            "Check the INVERSE: If necessity_score < 0.3 but the narration "
            "uses alarming language ('emergency', 'critical danger'), this is "
            "also a coherence failure — overstating risk is as bad as "
            "understating it.",
            "A well-calibrated narration uses language proportional to the "
            "actual risk level: moderate necessity should use moderate concern.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=threshold,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

METRIC_FACTORIES = {
    # Core metrics
    "safety_fidelity": safety_fidelity_metric,
    "perceptual_grounding": perceptual_grounding_metric,
    "causal_accuracy": causal_accuracy_metric,
    "action_justification": action_justification_metric,
    "conciseness": conciseness_metric,
    # Research-backed metrics
    "strict_context_grounding": strict_context_grounding_metric,
    "progressive_alignment": progressive_alignment_metric,
    "logical_completeness": logical_completeness_metric,
    "risk_coherence": risk_coherence_metric,
}


def get_all_metrics(threshold: float = DEFAULT_THRESHOLD, model: str = None) -> list:
    """Instantiate and return all 9 XAI evaluation metrics."""
    return [factory(threshold=threshold, model=model) for factory in METRIC_FACTORIES.values()]


def get_metric(name: str, threshold: float = DEFAULT_THRESHOLD, model: str = None) -> GEval:
    """Instantiate a single metric by key name."""
    factory = METRIC_FACTORIES.get(name)
    if factory is None:
        valid = ", ".join(METRIC_FACTORIES.keys())
        raise ValueError(f"Unknown metric '{name}'. Valid: {valid}")
    return factory(threshold=threshold, model=model)
