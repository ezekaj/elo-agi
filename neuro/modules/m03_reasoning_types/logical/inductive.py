"""
Inductive Reasoning - Specific → General

Generalize from specific observations to form general rules.
Key properties: probabilistic, falsifiable by counterexample.
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class Observation:
    """A single observation/instance"""

    observation_id: str
    features: Dict[str, Any]
    label: Optional[Any] = None
    confidence: float = 1.0
    timestamp: float = 0.0


@dataclass
class Hypothesis:
    """A general hypothesis induced from observations"""

    hypothesis_id: str
    description: str
    predicate: Callable[[Dict[str, Any]], bool]
    confidence: float
    supporting_observations: List[str] = field(default_factory=list)
    counterexamples: List[str] = field(default_factory=list)
    n_observations: int = 0
    created_from_features: Dict[str, Any] = field(default_factory=dict)

    def test(self, features: Dict[str, Any]) -> bool:
        """Test if features satisfy the hypothesis"""
        try:
            return self.predicate(features)
        except Exception:
            return False

    def update_confidence(self, is_supporting: bool, observation_id: str):
        """Update confidence based on new evidence"""
        self.n_observations += 1

        if is_supporting:
            self.supporting_observations.append(observation_id)
            self.confidence = len(self.supporting_observations) / self.n_observations
        else:
            self.counterexamples.append(observation_id)
            self.confidence *= 0.5

    @property
    def is_refuted(self) -> bool:
        """Check if hypothesis has been refuted by counterexamples"""
        return len(self.counterexamples) > 0 and self.confidence < 0.1


class InductiveReasoner:
    """
    Generalize from specific instances to general rules.
    Implements inductive inference.

    "All observed swans are white" → "All swans are white"
    """

    def __init__(self, confidence_threshold: float = 0.7, min_observations: int = 3):
        self.observations: Dict[str, Observation] = {}
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.confidence_threshold = confidence_threshold
        self.min_observations = min_observations

    def observe(self, observation: Observation):
        """Record a new observation"""
        self.observations[observation.observation_id] = observation

        for hyp in list(self.hypotheses.values()):
            is_supporting = hyp.test(observation.features)
            hyp.update_confidence(is_supporting, observation.observation_id)

    def observe_batch(self, observations: List[Observation]):
        """Record multiple observations"""
        for obs in observations:
            self.observe(obs)

    def hypothesize(self, observation_ids: List[str] = None) -> List[Hypothesis]:
        """Form hypotheses from observations"""
        if observation_ids is None:
            observations = list(self.observations.values())
        else:
            observations = [
                self.observations[oid] for oid in observation_ids if oid in self.observations
            ]

        if len(observations) < self.min_observations:
            return []

        new_hypotheses = []

        common_features = self._find_common_features(observations)
        for feature, value in common_features.items():
            hyp = self._create_hypothesis(feature, value, observations)
            if hyp and hyp.hypothesis_id not in self.hypotheses:
                self.hypotheses[hyp.hypothesis_id] = hyp
                new_hypotheses.append(hyp)

        correlated = self._find_correlations(observations)
        for (feat1, val1), (feat2, val2), correlation in correlated:
            hyp = self._create_correlation_hypothesis(
                feat1, val1, feat2, val2, correlation, observations
            )
            if hyp and hyp.hypothesis_id not in self.hypotheses:
                self.hypotheses[hyp.hypothesis_id] = hyp
                new_hypotheses.append(hyp)

        return new_hypotheses

    def _find_common_features(self, observations: List[Observation]) -> Dict[str, Any]:
        """Find features that are constant across all observations"""
        if not observations:
            return {}

        first_features = observations[0].features
        common = {}

        for key, value in first_features.items():
            is_common = all(
                key in obs.features and obs.features[key] == value for obs in observations
            )
            if is_common:
                common[key] = value

        return common

    def _find_correlations(
        self, observations: List[Observation], min_correlation: float = 0.8
    ) -> List[Tuple[Tuple[str, Any], Tuple[str, Any], float]]:
        """Find correlated features across observations"""
        correlations = []

        all_features = set()
        for obs in observations:
            all_features.update(obs.features.keys())

        feature_values: Dict[str, List[Any]] = {f: [] for f in all_features}
        for obs in observations:
            for feat in all_features:
                feature_values[feat].append(obs.features.get(feat))

        features = list(all_features)
        for i, feat1 in enumerate(features):
            for feat2 in features[i + 1 :]:
                co_occurrence = Counter()
                total = 0

                for obs in observations:
                    if feat1 in obs.features and feat2 in obs.features:
                        key = (obs.features[feat1], obs.features[feat2])
                        co_occurrence[key] += 1
                        total += 1

                if total > 0:
                    for (v1, v2), count in co_occurrence.most_common(1):
                        correlation = count / total
                        if correlation >= min_correlation:
                            correlations.append(((feat1, v1), (feat2, v2), correlation))

        return correlations

    def _create_hypothesis(
        self, feature: str, value: Any, observations: List[Observation]
    ) -> Optional[Hypothesis]:
        """Create a hypothesis about a single feature"""
        hyp_id = f"hyp_{feature}_{value}"

        def predicate(features: Dict[str, Any]) -> bool:
            return features.get(feature) == value

        supporting = [obs.observation_id for obs in observations if predicate(obs.features)]

        if len(supporting) < self.min_observations:
            return None

        return Hypothesis(
            hypothesis_id=hyp_id,
            description=f"All instances have {feature} = {value}",
            predicate=predicate,
            confidence=len(supporting) / len(observations),
            supporting_observations=supporting,
            n_observations=len(observations),
            created_from_features={feature: value},
        )

    def _create_correlation_hypothesis(
        self,
        feat1: str,
        val1: Any,
        feat2: str,
        val2: Any,
        correlation: float,
        observations: List[Observation],
    ) -> Optional[Hypothesis]:
        """Create a hypothesis about correlation between features"""
        hyp_id = f"hyp_corr_{feat1}_{val1}_{feat2}_{val2}"

        def predicate(features: Dict[str, Any]) -> bool:
            if features.get(feat1) == val1:
                return features.get(feat2) == val2
            return True

        supporting = [obs.observation_id for obs in observations if predicate(obs.features)]

        return Hypothesis(
            hypothesis_id=hyp_id,
            description=f"If {feat1} = {val1}, then {feat2} = {val2}",
            predicate=predicate,
            confidence=correlation,
            supporting_observations=supporting,
            n_observations=len(observations),
            created_from_features={feat1: val1, feat2: val2},
        )

    def confidence(self, hypothesis_id: str, n_observations: int = None) -> float:
        """Get confidence in a hypothesis"""
        if hypothesis_id not in self.hypotheses:
            return 0.0

        hyp = self.hypotheses[hypothesis_id]

        if n_observations is not None:
            return len(hyp.supporting_observations) / max(1, n_observations)

        return hyp.confidence

    def counterexample_search(
        self, hypothesis_id: str, candidates: List[Observation] = None
    ) -> List[Observation]:
        """Search for counterexamples to a hypothesis"""
        if hypothesis_id not in self.hypotheses:
            return []

        hyp = self.hypotheses[hypothesis_id]

        if candidates is None:
            candidates = list(self.observations.values())

        counterexamples = []
        for obs in candidates:
            if not hyp.test(obs.features):
                counterexamples.append(obs)
                hyp.update_confidence(False, obs.observation_id)

        return counterexamples

    def get_best_hypotheses(self, min_confidence: float = None) -> List[Hypothesis]:
        """Get hypotheses sorted by confidence"""
        threshold = min_confidence or self.confidence_threshold

        valid = [
            h for h in self.hypotheses.values() if h.confidence >= threshold and not h.is_refuted
        ]

        return sorted(valid, key=lambda h: h.confidence, reverse=True)

    def generalize(
        self, observation_ids: List[str], target_feature: str
    ) -> Optional[Tuple[Any, float]]:
        """
        Predict the value of target_feature for new instances
        based on generalization from observations.
        """
        observations = [
            self.observations[oid] for oid in observation_ids if oid in self.observations
        ]

        if not observations:
            return None

        values = [
            obs.features.get(target_feature)
            for obs in observations
            if target_feature in obs.features
        ]

        if not values:
            return None

        value_counts = Counter(values)
        most_common_value, count = value_counts.most_common(1)[0]
        confidence = count / len(values)

        return most_common_value, confidence

    def strengthen(self, hypothesis_id: str, amount: float = 0.1):
        """Manually strengthen a hypothesis (learning signal)"""
        if hypothesis_id in self.hypotheses:
            hyp = self.hypotheses[hypothesis_id]
            hyp.confidence = min(1.0, hyp.confidence + amount)

    def weaken(self, hypothesis_id: str, amount: float = 0.1):
        """Manually weaken a hypothesis"""
        if hypothesis_id in self.hypotheses:
            hyp = self.hypotheses[hypothesis_id]
            hyp.confidence = max(0.0, hyp.confidence - amount)
