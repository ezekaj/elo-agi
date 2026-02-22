"""
Reasoning Orchestrator - Coordinates All Reasoning Types

Selects and combines appropriate reasoning systems based on task requirements.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum

from .perceptual import VisualFeatureExtractor, MultimodalIntegrator, ObjectRecognizer
from .dimensional import SpatialReasoner, TemporalReasoner, HierarchicalReasoner
from .logical import InductiveReasoner, DeductiveReasoner, AbductiveReasoner
from .interactive import FeedbackAdapter, TheoryOfMind, CollaborativeReasoner


class ReasoningType(Enum):
    PERCEPTUAL = "perceptual"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    HIERARCHICAL = "hierarchical"
    INDUCTIVE = "inductive"
    DEDUCTIVE = "deductive"
    ABDUCTIVE = "abductive"
    FEEDBACK = "feedback"
    SOCIAL = "social"
    COLLABORATIVE = "collaborative"


@dataclass
class TaskAnalysis:
    """Analysis of what reasoning types a task requires"""

    task_description: str
    required_types: Set[ReasoningType]
    primary_type: ReasoningType
    complexity: float
    suggested_sequence: List[ReasoningType]
    confidence: float


@dataclass
class ReasoningResult:
    """Result from a reasoning operation"""

    reasoning_type: ReasoningType
    result: Any
    confidence: float
    reasoning_chain: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReasoningOrchestrator:
    """
    Orchestrates all four types of reasoning.
    Selects appropriate reasoners and combines their outputs.
    """

    def __init__(self, agent_id: str = "orchestrator"):
        self.agent_id = agent_id

        self.visual_extractor = VisualFeatureExtractor()
        self.multimodal = MultimodalIntegrator()
        self.object_recognizer = ObjectRecognizer()

        self.spatial = SpatialReasoner()
        self.temporal = TemporalReasoner()
        self.hierarchical = HierarchicalReasoner()

        self.inductive = InductiveReasoner()
        self.deductive = DeductiveReasoner()
        self.abductive = AbductiveReasoner()

        self.feedback = FeedbackAdapter()
        self.theory_of_mind = TheoryOfMind(self_id=agent_id)
        self.collaborative = CollaborativeReasoner(self_id=agent_id)

        self.task_type_keywords = {
            ReasoningType.PERCEPTUAL: ["see", "look", "visual", "image", "pattern", "recognize"],
            ReasoningType.SPATIAL: ["where", "location", "rotate", "move", "position", "navigate"],
            ReasoningType.TEMPORAL: ["when", "sequence", "order", "before", "after", "duration"],
            ReasoningType.HIERARCHICAL: ["rule", "abstract", "category", "type", "level"],
            ReasoningType.INDUCTIVE: ["pattern", "generalize", "learn", "example", "similar"],
            ReasoningType.DEDUCTIVE: ["therefore", "conclude", "if-then", "logic", "prove"],
            ReasoningType.ABDUCTIVE: ["why", "cause", "explain", "reason", "because"],
            ReasoningType.FEEDBACK: ["try", "adapt", "improve", "feedback", "reward"],
            ReasoningType.SOCIAL: ["think", "believe", "want", "intend", "feel"],
            ReasoningType.COLLABORATIVE: ["together", "share", "coordinate", "team", "help"],
        }

    def analyze_task(self, task_description: str) -> TaskAnalysis:
        """Analyze what reasoning types a task requires"""
        task_lower = task_description.lower()
        required = set()
        scores = {}

        for rtype, keywords in self.task_type_keywords.items():
            score = sum(1 for kw in keywords if kw in task_lower)
            if score > 0:
                required.add(rtype)
                scores[rtype] = score

        if not required:
            required.add(ReasoningType.DEDUCTIVE)
            scores[ReasoningType.DEDUCTIVE] = 1

        primary = max(scores.keys(), key=lambda k: scores[k])

        sequence = self._determine_sequence(required, primary)

        complexity = len(required) / len(ReasoningType)
        confidence = max(scores.values()) / len(self.task_type_keywords[primary])

        return TaskAnalysis(
            task_description=task_description,
            required_types=required,
            primary_type=primary,
            complexity=complexity,
            suggested_sequence=sequence,
            confidence=min(1.0, confidence),
        )

    def _determine_sequence(
        self, required: Set[ReasoningType], primary: ReasoningType
    ) -> List[ReasoningType]:
        """Determine optimal sequence of reasoning types"""
        order = [
            ReasoningType.PERCEPTUAL,
            ReasoningType.SPATIAL,
            ReasoningType.TEMPORAL,
            ReasoningType.INDUCTIVE,
            ReasoningType.ABDUCTIVE,
            ReasoningType.DEDUCTIVE,
            ReasoningType.HIERARCHICAL,
            ReasoningType.SOCIAL,
            ReasoningType.COLLABORATIVE,
            ReasoningType.FEEDBACK,
        ]

        sequence = [r for r in order if r in required]

        if primary in sequence and sequence[0] != primary:
            sequence.remove(primary)
            sequence.insert(0, primary)

        return sequence

    def activate_reasoners(self, task_analysis: TaskAnalysis) -> Dict[ReasoningType, Any]:
        """Activate and return relevant reasoners"""
        active = {}

        for rtype in task_analysis.required_types:
            if rtype == ReasoningType.PERCEPTUAL:
                active[rtype] = {
                    "visual": self.visual_extractor,
                    "multimodal": self.multimodal,
                    "recognition": self.object_recognizer,
                }
            elif rtype == ReasoningType.SPATIAL:
                active[rtype] = self.spatial
            elif rtype == ReasoningType.TEMPORAL:
                active[rtype] = self.temporal
            elif rtype == ReasoningType.HIERARCHICAL:
                active[rtype] = self.hierarchical
            elif rtype == ReasoningType.INDUCTIVE:
                active[rtype] = self.inductive
            elif rtype == ReasoningType.DEDUCTIVE:
                active[rtype] = self.deductive
            elif rtype == ReasoningType.ABDUCTIVE:
                active[rtype] = self.abductive
            elif rtype == ReasoningType.FEEDBACK:
                active[rtype] = self.feedback
            elif rtype == ReasoningType.SOCIAL:
                active[rtype] = self.theory_of_mind
            elif rtype == ReasoningType.COLLABORATIVE:
                active[rtype] = self.collaborative

        return active

    def reason(
        self, task_description: str, input_data: Any = None, context: Dict[str, Any] = None
    ) -> List[ReasoningResult]:
        """
        Main reasoning interface.
        Analyzes task, activates reasoners, and returns results.
        """
        analysis = self.analyze_task(task_description)
        reasoners = self.activate_reasoners(analysis)
        results = []

        for rtype in analysis.suggested_sequence:
            if rtype not in reasoners:
                continue

            result = self._apply_reasoner(rtype, reasoners[rtype], input_data, context)
            if result:
                results.append(result)

                if isinstance(result.result, dict):
                    context = context or {}
                    context[rtype.value] = result.result

        return results

    def _apply_reasoner(
        self, rtype: ReasoningType, reasoner: Any, input_data: Any, context: Dict[str, Any]
    ) -> Optional[ReasoningResult]:
        """Apply a specific reasoner to input"""
        try:
            if rtype == ReasoningType.PERCEPTUAL:
                if isinstance(input_data, np.ndarray):
                    features = reasoner["visual"].extract_all(input_data)
                    return ReasoningResult(
                        reasoning_type=rtype,
                        result={"features": features},
                        confidence=0.8,
                        reasoning_chain=["extract_features"],
                    )

            elif rtype == ReasoningType.SPATIAL:
                if context and "objects" in context:
                    for obj in context["objects"]:
                        reasoner.add_object(obj)
                return ReasoningResult(
                    reasoning_type=rtype,
                    result={"spatial_state": reasoner.objects},
                    confidence=0.9,
                    reasoning_chain=["spatial_encoding"],
                )

            elif rtype == ReasoningType.TEMPORAL:
                if context and "events" in context:
                    for event in context["events"]:
                        reasoner.add_event(event)
                    ordered = reasoner.order_events([e.event_id for e in context["events"]])
                    return ReasoningResult(
                        reasoning_type=rtype,
                        result={"ordered_events": ordered},
                        confidence=0.9,
                        reasoning_chain=["temporal_ordering"],
                    )

            elif rtype == ReasoningType.INDUCTIVE:
                if context and "observations" in context:
                    for obs in context["observations"]:
                        reasoner.observe(obs)
                    hypotheses = reasoner.hypothesize()
                    return ReasoningResult(
                        reasoning_type=rtype,
                        result={"hypotheses": hypotheses},
                        confidence=hypotheses[0].confidence if hypotheses else 0.5,
                        reasoning_chain=["observe", "hypothesize"],
                    )

            elif rtype == ReasoningType.DEDUCTIVE:
                if context and "premises" in context:
                    for premise in context["premises"]:
                        reasoner.add_premise(premise)
                    inferences = reasoner.derive(context["premises"])
                    return ReasoningResult(
                        reasoning_type=rtype,
                        result={"inferences": inferences},
                        confidence=1.0 if inferences else 0.0,
                        reasoning_chain=["derive"],
                    )

            elif rtype == ReasoningType.ABDUCTIVE:
                if context and "effect" in context:
                    explanation = reasoner.explain(
                        context["effect"], context.get("effect_features", {})
                    )
                    return ReasoningResult(
                        reasoning_type=rtype,
                        result={"explanation": explanation},
                        confidence=explanation.probability if explanation else 0.0,
                        reasoning_chain=["explain"],
                    )

            elif rtype == ReasoningType.SOCIAL:
                if context and "agent" in context:
                    predicted = reasoner.predict_action(context["agent"])
                    return ReasoningResult(
                        reasoning_type=rtype,
                        result={"predicted_action": predicted},
                        confidence=predicted[1] if predicted else 0.0,
                        reasoning_chain=["predict_action"],
                    )

            return ReasoningResult(
                reasoning_type=rtype,
                result={"status": "activated"},
                confidence=0.5,
                reasoning_chain=["activate"],
            )

        except Exception as e:
            return ReasoningResult(
                reasoning_type=rtype,
                result={"error": str(e)},
                confidence=0.0,
                reasoning_chain=["error"],
            )

    def combine_outputs(self, results: List[ReasoningResult]) -> Dict[str, Any]:
        """Combine outputs from multiple reasoners"""
        combined = {
            "reasoning_types_used": [r.reasoning_type.value for r in results],
            "overall_confidence": np.mean([r.confidence for r in results]) if results else 0.0,
            "results": {},
        }

        for result in results:
            combined["results"][result.reasoning_type.value] = {
                "output": result.result,
                "confidence": result.confidence,
                "chain": result.reasoning_chain,
            }

        return combined

    def resolve_conflicts(self, results: List[ReasoningResult]) -> ReasoningResult:
        """Resolve conflicts when reasoners disagree"""
        if not results:
            return None

        if len(results) == 1:
            return results[0]

        priority_order = [
            ReasoningType.DEDUCTIVE,
            ReasoningType.PERCEPTUAL,
            ReasoningType.ABDUCTIVE,
            ReasoningType.INDUCTIVE,
            ReasoningType.SOCIAL,
        ]

        for priority_type in priority_order:
            for result in results:
                if result.reasoning_type == priority_type:
                    return result

        return max(results, key=lambda r: r.confidence)

    def meta_reason(
        self, task_description: str, previous_results: List[ReasoningResult] = None
    ) -> Dict[str, Any]:
        """Meta-level reasoning about the reasoning process itself"""
        analysis = self.analyze_task(task_description)

        meta_analysis = {
            "task_analysis": {
                "primary_type": analysis.primary_type.value,
                "required_types": [t.value for t in analysis.required_types],
                "complexity": analysis.complexity,
            },
            "strategy": analysis.suggested_sequence,
            "recommendations": [],
        }

        if analysis.complexity > 0.5:
            meta_analysis["recommendations"].append("Complex task: consider breaking into subtasks")

        if ReasoningType.SOCIAL in analysis.required_types:
            meta_analysis["recommendations"].append(
                "Social reasoning needed: consider perspective-taking"
            )

        if previous_results:
            avg_confidence = np.mean([r.confidence for r in previous_results])
            if avg_confidence < 0.5:
                meta_analysis["recommendations"].append(
                    "Low confidence: consider gathering more information"
                )

        return meta_analysis
