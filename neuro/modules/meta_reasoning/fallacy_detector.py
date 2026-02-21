"""
Fallacy Detector for Meta-Reasoning

Detects reasoning errors and biases:
- Confirmation bias detection
- Circular reasoning detection
- Various logical fallacies
- Correction suggestions
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import numpy as np


class FallacyType(Enum):
    """Types of reasoning fallacies."""
    CONFIRMATION_BIAS = "confirmation_bias"
    ANCHORING = "anchoring"
    AVAILABILITY = "availability"
    BASE_RATE_NEGLECT = "base_rate_neglect"
    CIRCULAR = "circular"
    FALSE_DICHOTOMY = "false_dichotomy"
    HASTY_GENERALIZATION = "hasty_generalization"
    SUNK_COST = "sunk_cost"
    OVERFITTING = "overfitting"
    PREMATURE_TERMINATION = "premature_termination"


@dataclass
class FallacyDetectorConfig:
    """Configuration for fallacy detection."""
    confirmation_bias_threshold: float = 0.7
    circular_depth_limit: int = 5
    overfitting_threshold: float = 0.9
    min_evidence_count: int = 3
    enable_all_detectors: bool = True


@dataclass
class FallacyDetection:
    """A detected fallacy."""
    fallacy_type: FallacyType
    confidence: float
    location: str
    evidence: List[str]
    severity: float


@dataclass
class ReasoningStep:
    """A step in a reasoning trace."""
    step_id: str
    content: str
    premises: List[str]
    conclusion: str
    evidence_used: List[str]
    confidence: float


class FallacyDetector:
    """
    Detects fallacies and biases in reasoning.

    Capabilities:
    - Detect confirmation bias
    - Detect circular reasoning
    - Detect various logical fallacies
    - Suggest corrections
    """

    def __init__(
        self,
        config: Optional[FallacyDetectorConfig] = None,
        random_seed: Optional[int] = None,
    ):
        self.config = config or FallacyDetectorConfig()
        self._rng = np.random.default_rng(random_seed)

        self._detection_history: List[FallacyDetection] = []
        self._total_detections = 0

    def detect_fallacies(
        self,
        reasoning_trace: List[ReasoningStep],
    ) -> List[FallacyDetection]:
        """
        Detect fallacies in a reasoning trace.

        Args:
            reasoning_trace: List of reasoning steps

        Returns:
            List of detected fallacies
        """
        fallacies = []

        if self.config.enable_all_detectors:
            circular = self._detect_circular_in_trace(reasoning_trace)
            if circular:
                fallacies.append(circular)

            hasty = self._detect_hasty_generalization(reasoning_trace)
            if hasty:
                fallacies.append(hasty)

            premature = self._detect_premature_termination(reasoning_trace)
            if premature:
                fallacies.append(premature)

        for detection in fallacies:
            self._detection_history.append(detection)
            self._total_detections += 1

        return fallacies

    def _detect_circular_in_trace(
        self,
        trace: List[ReasoningStep],
    ) -> Optional[FallacyDetection]:
        """Detect circular reasoning in trace."""
        conclusions = set()
        premises_map: Dict[str, Set[str]] = {}

        for step in trace:
            conclusions.add(step.conclusion)
            premises_map[step.step_id] = set(step.premises)

        for step in trace:
            if step.conclusion in step.premises:
                return FallacyDetection(
                    fallacy_type=FallacyType.CIRCULAR,
                    confidence=0.95,
                    location=step.step_id,
                    evidence=[f"Conclusion '{step.conclusion}' appears in premises"],
                    severity=0.8,
                )

        return None

    def _detect_hasty_generalization(
        self,
        trace: List[ReasoningStep],
    ) -> Optional[FallacyDetection]:
        """
        Detect hasty generalization using pattern matching and semantic analysis.

        Patterns checked:
        1. Universal quantifiers with insufficient evidence
        2. Strong claims with limited sample size indicators
        3. Generalizations from specific cases
        """
        # Patterns indicating universal claims (beyond just "all"/"always")
        universal_patterns = [
            # Universal quantifiers
            "all ", "every ", "always ", "never ", "none ", "no one ",
            "everyone ", "everything ", "everywhere ", "anybody ", "nobody ",
            # Strong generalizations
            " must ", " will always ", " can never ", " is always ", " are always ",
            # Categorical statements
            "without exception", "in every case", "100%", "guaranteed",
            # Implicit universals
            " is a ", " are the ", " defines ", " characterizes ",
        ]

        # Patterns indicating limited evidence
        limited_evidence_patterns = [
            "based on this example", "from this case", "this shows that",
            "therefore all", "so every", "hence always",
            "one ", "a single ", "this one ",
        ]

        for step in trace:
            conclusion_lower = step.conclusion.lower()
            evidence_count = len(step.evidence_used)

            # Check for universal patterns
            has_universal = any(
                pattern in conclusion_lower for pattern in universal_patterns
            )

            # Check for limited evidence indicators in the conclusion itself
            has_limited_indicator = any(
                pattern in conclusion_lower for pattern in limited_evidence_patterns
            )

            # Also check premises for limited scope
            premises_text = " ".join(step.premises).lower()
            premises_indicate_limited = any(
                pattern in premises_text for pattern in limited_evidence_patterns
            )

            # Detect hasty generalization if:
            # 1. Universal claim with insufficient evidence
            # 2. Limited evidence indicator with generalization
            if has_universal and evidence_count < self.config.min_evidence_count:
                return FallacyDetection(
                    fallacy_type=FallacyType.HASTY_GENERALIZATION,
                    confidence=0.75,
                    location=step.step_id,
                    evidence=[
                        f"Universal claim with only {evidence_count} evidence items",
                        "Pattern match: universal quantifier detected"
                    ],
                    severity=0.7,
                )

            if has_limited_indicator and has_universal:
                return FallacyDetection(
                    fallacy_type=FallacyType.HASTY_GENERALIZATION,
                    confidence=0.8,
                    location=step.step_id,
                    evidence=[
                        "Limited evidence indicator combined with universal claim",
                        "Pattern match: limited evidence + generalization"
                    ],
                    severity=0.7,
                )

            if premises_indicate_limited and has_universal and evidence_count < self.config.min_evidence_count * 2:
                return FallacyDetection(
                    fallacy_type=FallacyType.HASTY_GENERALIZATION,
                    confidence=0.65,
                    location=step.step_id,
                    evidence=[
                        "Premises indicate limited scope but conclusion generalizes",
                        f"Evidence count ({evidence_count}) insufficient for claim scope"
                    ],
                    severity=0.6,
                )

        return None

    def _detect_premature_termination(
        self,
        trace: List[ReasoningStep],
    ) -> Optional[FallacyDetection]:
        """Detect premature termination of reasoning."""
        if len(trace) < 2:
            return None

        confidences = [step.confidence for step in trace]

        if len(confidences) >= 3:
            last_conf = confidences[-1]
            prev_conf = confidences[-2]

            if prev_conf > last_conf + 0.2:
                return FallacyDetection(
                    fallacy_type=FallacyType.PREMATURE_TERMINATION,
                    confidence=0.6,
                    location=trace[-1].step_id,
                    evidence=["Confidence dropped before termination"],
                    severity=0.5,
                )

        return None

    def detect_confirmation_bias(
        self,
        hypotheses: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
    ) -> Optional[FallacyDetection]:
        """
        Detect confirmation bias.

        Args:
            hypotheses: List of hypotheses with support info
            evidence: List of evidence items

        Returns:
            FallacyDetection if bias detected
        """
        if not hypotheses or not evidence:
            return None

        main_hypothesis = hypotheses[0]
        supporting = sum(1 for e in evidence if e.get("supports", "") == main_hypothesis.get("id"))
        contradicting = sum(1 for e in evidence if e.get("contradicts", "") == main_hypothesis.get("id"))

        total = supporting + contradicting
        if total < self.config.min_evidence_count:
            return None

        support_ratio = supporting / total

        if support_ratio > self.config.confirmation_bias_threshold:
            detection = FallacyDetection(
                fallacy_type=FallacyType.CONFIRMATION_BIAS,
                confidence=support_ratio,
                location="evidence_collection",
                evidence=[
                    f"Support ratio: {support_ratio:.2f}",
                    f"Supporting: {supporting}, Contradicting: {contradicting}",
                ],
                severity=(support_ratio - 0.5) * 2,
            )
            self._detection_history.append(detection)
            self._total_detections += 1
            return detection

        return None

    def detect_circular_reasoning(
        self,
        inference_chain: List[Tuple[str, str]],
    ) -> Optional[FallacyDetection]:
        """
        Detect circular reasoning in inference chain.

        Args:
            inference_chain: List of (premise, conclusion) tuples

        Returns:
            FallacyDetection if circular reasoning detected
        """
        if not inference_chain:
            return None

        conclusion_to_premises: Dict[str, Set[str]] = {}

        for premise, conclusion in inference_chain:
            if conclusion not in conclusion_to_premises:
                conclusion_to_premises[conclusion] = set()
            conclusion_to_premises[conclusion].add(premise)

        def find_cycle(start: str, visited: Set[str], path: List[str]) -> Optional[List[str]]:
            if start in visited:
                cycle_start = path.index(start)
                return path[cycle_start:]

            if len(path) > self.config.circular_depth_limit:
                return None

            visited.add(start)
            path.append(start)

            if start in conclusion_to_premises:
                for premise in conclusion_to_premises[start]:
                    result = find_cycle(premise, visited.copy(), path.copy())
                    if result:
                        return result

            return None

        for conclusion in conclusion_to_premises:
            cycle = find_cycle(conclusion, set(), [])
            if cycle:
                detection = FallacyDetection(
                    fallacy_type=FallacyType.CIRCULAR,
                    confidence=0.9,
                    location=" -> ".join(cycle),
                    evidence=[f"Circular chain: {' -> '.join(cycle)}"],
                    severity=0.9,
                )
                self._detection_history.append(detection)
                self._total_detections += 1
                return detection

        return None

    def detect_anchoring(
        self,
        initial_estimate: float,
        final_estimate: float,
        evidence_range: Tuple[float, float],
    ) -> Optional[FallacyDetection]:
        """
        Detect anchoring bias.

        Args:
            initial_estimate: Initial estimate
            final_estimate: Final estimate
            evidence_range: Range suggested by evidence

        Returns:
            FallacyDetection if anchoring detected
        """
        evidence_midpoint = (evidence_range[0] + evidence_range[1]) / 2
        evidence_width = evidence_range[1] - evidence_range[0]

        dist_from_initial = abs(final_estimate - initial_estimate)
        dist_from_evidence = abs(final_estimate - evidence_midpoint)

        if dist_from_initial < dist_from_evidence and evidence_width > 0:
            anchoring_strength = 1 - (dist_from_initial / (evidence_width + 1e-8))
            anchoring_strength = np.clip(anchoring_strength, 0, 1)

            if anchoring_strength > 0.5:
                detection = FallacyDetection(
                    fallacy_type=FallacyType.ANCHORING,
                    confidence=float(anchoring_strength),
                    location="estimate_adjustment",
                    evidence=[
                        f"Final ({final_estimate:.2f}) closer to initial ({initial_estimate:.2f})",
                        f"than to evidence midpoint ({evidence_midpoint:.2f})",
                    ],
                    severity=float(anchoring_strength * 0.7),
                )
                self._detection_history.append(detection)
                self._total_detections += 1
                return detection

        return None

    def detect_base_rate_neglect(
        self,
        base_rate: float,
        likelihood_ratio: float,
        posterior: float,
    ) -> Optional[FallacyDetection]:
        """
        Detect base rate neglect.

        Args:
            base_rate: Prior probability
            likelihood_ratio: P(evidence|hypothesis) / P(evidence|not hypothesis)
            posterior: Estimated posterior probability

        Returns:
            FallacyDetection if base rate neglected
        """
        expected_posterior = (base_rate * likelihood_ratio) / (
            base_rate * likelihood_ratio + (1 - base_rate)
        )

        error = abs(posterior - expected_posterior)

        base_rate_influence = abs(posterior - likelihood_ratio / (1 + likelihood_ratio))
        evidence_influence = abs(posterior - base_rate)

        if evidence_influence > base_rate_influence * 2 and error > 0.1:
            detection = FallacyDetection(
                fallacy_type=FallacyType.BASE_RATE_NEGLECT,
                confidence=min(1.0, error * 2),
                location="probability_estimation",
                evidence=[
                    f"Expected posterior: {expected_posterior:.2f}",
                    f"Actual posterior: {posterior:.2f}",
                    f"Base rate: {base_rate:.2f}",
                ],
                severity=min(1.0, error),
            )
            self._detection_history.append(detection)
            self._total_detections += 1
            return detection

        return None

    def suggest_corrections(
        self,
        fallacies: List[FallacyDetection],
    ) -> List[str]:
        """
        Suggest corrections for detected fallacies.

        Args:
            fallacies: List of detected fallacies

        Returns:
            List of correction suggestions
        """
        suggestions = []

        corrections = {
            FallacyType.CONFIRMATION_BIAS: [
                "Actively seek disconfirming evidence",
                "Consider alternative hypotheses equally",
                "Use devil's advocate technique",
            ],
            FallacyType.ANCHORING: [
                "Reset and re-estimate from evidence alone",
                "Consider multiple independent starting points",
                "Focus on evidence range rather than initial value",
            ],
            FallacyType.CIRCULAR: [
                "Identify and break the circular dependency",
                "Find independent supporting evidence",
                "Restructure the argument chain",
            ],
            FallacyType.BASE_RATE_NEGLECT: [
                "Start with base rate as prior",
                "Apply Bayes' theorem correctly",
                "Consider population statistics",
            ],
            FallacyType.HASTY_GENERALIZATION: [
                "Gather more evidence before generalizing",
                "Use qualified language (some, often) instead of universals",
                "Look for counterexamples",
            ],
            FallacyType.PREMATURE_TERMINATION: [
                "Continue reasoning until confidence stabilizes",
                "Consider alternative solution paths",
                "Review completeness of analysis",
            ],
        }

        for fallacy in fallacies:
            specific = corrections.get(fallacy.fallacy_type, ["Review reasoning process"])
            for correction in specific:
                suggestions.append(f"[{fallacy.fallacy_type.value}] {correction}")

        return suggestions

    def get_detection_history(
        self,
        fallacy_type: Optional[FallacyType] = None,
        n: Optional[int] = None,
    ) -> List[FallacyDetection]:
        """Get detection history."""
        history = self._detection_history

        if fallacy_type:
            history = [d for d in history if d.fallacy_type == fallacy_type]

        if n:
            history = history[-n:]

        return history

    def statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        type_counts: Dict[str, int] = {}
        for detection in self._detection_history:
            t = detection.fallacy_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        severities = [d.severity for d in self._detection_history]
        confidences = [d.confidence for d in self._detection_history]

        return {
            "total_detections": self._total_detections,
            "type_distribution": type_counts,
            "avg_severity": float(np.mean(severities)) if severities else 0.0,
            "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "most_common": max(type_counts, key=type_counts.get) if type_counts else None,
        }
