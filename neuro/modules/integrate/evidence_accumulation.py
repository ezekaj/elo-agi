"""
Evidence Accumulation: Combines evidence from multiple sources.

Implements drift-diffusion models and Bayesian evidence integration
for decision-making based on noisy, partial evidence from modules.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np

from .shared_space import SemanticEmbedding, ModalityType


class EvidenceType(Enum):
    """Types of evidence."""

    SENSORY = "sensory"  # Direct sensory evidence
    MEMORY = "memory"  # Retrieved from memory
    INFERENCE = "inference"  # Logical/probabilistic inference
    PRIOR = "prior"  # Prior belief
    CONTEXTUAL = "contextual"  # Context-dependent
    SOCIAL = "social"  # From social sources


@dataclass
class EvidenceSource:
    """A source of evidence."""

    name: str
    reliability: float = 1.0
    modality: Optional[ModalityType] = None
    bias: float = 0.0  # Systematic bias in evidence


@dataclass
class Evidence:
    """A piece of evidence for a hypothesis."""

    source: EvidenceSource
    evidence_type: EvidenceType
    value: np.ndarray  # Evidence vector
    strength: float  # How strong is this evidence
    timestamp: float = 0.0
    uncertainty: float = 0.1  # Measurement uncertainty
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def weight(self) -> float:
        """Compute evidence weight accounting for reliability and uncertainty."""
        return self.strength * self.source.reliability / (1 + self.uncertainty)


@dataclass
class AccumulatorConfig:
    """Configuration for evidence accumulator."""

    threshold: float = 1.0  # Decision threshold
    leak_rate: float = 0.0  # Evidence leak over time
    noise_std: float = 0.1  # Internal noise
    n_alternatives: int = 2  # Number of choice alternatives
    time_step: float = 0.01  # Integration time step
    max_time: float = 10.0  # Maximum accumulation time
    prior_strength: float = 0.0  # Strength of prior bias
    random_seed: Optional[int] = None


class DriftDiffusionAccumulator:
    """
    Drift-Diffusion Model for two-alternative decisions.

    Accumulates evidence until reaching a decision threshold.
    """

    def __init__(self, config: Optional[AccumulatorConfig] = None):
        self.config = config or AccumulatorConfig()
        self._rng = np.random.default_rng(self.config.random_seed)

        # Current evidence state
        self._state = 0.0
        self._time = 0.0

        # History
        self._state_history: List[float] = []
        self._evidence_history: List[Evidence] = []

    def reset(self, prior: float = 0.0) -> None:
        """Reset accumulator state."""
        self._state = prior * self.config.prior_strength
        self._time = 0.0
        self._state_history = [self._state]
        self._evidence_history = []

    def accumulate(self, evidence: Evidence) -> Optional[int]:
        """
        Accumulate evidence and check for threshold crossing.

        Returns:
            0 if lower threshold crossed
            1 if upper threshold crossed
            None if no threshold crossed
        """
        self._evidence_history.append(evidence)

        # Compute drift rate from evidence
        drift = evidence.weight * np.mean(evidence.value)

        # Add noise
        noise = self._rng.normal(0, self.config.noise_std)

        # Update state with leaky integration
        self._state = (1 - self.config.leak_rate) * self._state + self.config.time_step * (
            drift + noise
        )

        self._time += self.config.time_step
        self._state_history.append(self._state)

        # Check thresholds
        if self._state >= self.config.threshold:
            return 1
        elif self._state <= -self.config.threshold:
            return 0

        # Check timeout
        if self._time >= self.config.max_time:
            return 1 if self._state > 0 else 0

        return None

    def get_state(self) -> Tuple[float, float]:
        """Get current state and time."""
        return self._state, self._time

    def get_decision_probability(self) -> float:
        """Get probability of choosing option 1 given current state."""
        # Sigmoid function centered at 0
        return 1.0 / (1.0 + np.exp(-self._state / self.config.threshold))


class BayesianAccumulator:
    """
    Bayesian evidence accumulator for multiple hypotheses.

    Maintains posterior probabilities over hypotheses
    and updates with each piece of evidence.
    """

    def __init__(
        self,
        n_hypotheses: int,
        prior: Optional[np.ndarray] = None,
        config: Optional[AccumulatorConfig] = None,
    ):
        self.n_hypotheses = n_hypotheses
        self.config = config or AccumulatorConfig()
        self._rng = np.random.default_rng(self.config.random_seed)

        # Initialize prior (uniform if not specified)
        if prior is None:
            self._prior = np.ones(n_hypotheses) / n_hypotheses
        else:
            self._prior = prior / prior.sum()

        self._posterior = self._prior.copy()
        self._log_posterior = np.log(self._posterior + 1e-10)

        # Evidence history
        self._evidence_history: List[Evidence] = []

    def reset(self, prior: Optional[np.ndarray] = None) -> None:
        """Reset to prior."""
        if prior is not None:
            self._prior = prior / prior.sum()

        self._posterior = self._prior.copy()
        self._log_posterior = np.log(self._posterior + 1e-10)
        self._evidence_history = []

    def update(
        self,
        evidence: Evidence,
        likelihoods: np.ndarray,
    ) -> np.ndarray:
        """
        Update posterior with new evidence.

        Args:
            evidence: The evidence
            likelihoods: P(evidence | hypothesis) for each hypothesis

        Returns:
            Updated posterior probabilities
        """
        self._evidence_history.append(evidence)

        # Weight likelihoods by evidence reliability
        weighted_likelihoods = likelihoods**evidence.weight

        # Bayesian update in log space for numerical stability
        self._log_posterior += np.log(weighted_likelihoods + 1e-10)

        # Normalize
        log_sum = np.log(np.sum(np.exp(self._log_posterior - self._log_posterior.max())))
        self._log_posterior -= self._log_posterior.max() + log_sum

        self._posterior = np.exp(self._log_posterior)
        self._posterior /= self._posterior.sum()

        return self._posterior.copy()

    def get_posterior(self) -> np.ndarray:
        """Get current posterior probabilities."""
        return self._posterior.copy()

    def get_map_hypothesis(self) -> int:
        """Get maximum a posteriori hypothesis."""
        return int(np.argmax(self._posterior))

    def get_entropy(self) -> float:
        """Get entropy of posterior (uncertainty measure)."""
        return float(-np.sum(self._posterior * np.log(self._posterior + 1e-10)))

    def is_decided(self, threshold: float = 0.9) -> bool:
        """Check if posterior is concentrated enough for decision."""
        return float(np.max(self._posterior)) >= threshold


class EvidenceAccumulator:
    """
    Central evidence accumulator integrating multiple sources.

    Manages evidence streams from different modules and
    maintains integrated beliefs about hypotheses.
    """

    def __init__(self, config: Optional[AccumulatorConfig] = None):
        self.config = config or AccumulatorConfig()
        self._rng = np.random.default_rng(self.config.random_seed)

        # Registered evidence sources
        self._sources: Dict[str, EvidenceSource] = {}

        # Evidence buffers per hypothesis
        self._evidence_buffers: Dict[str, List[Evidence]] = {}

        # Accumulators for different decision types
        self._ddm: Optional[DriftDiffusionAccumulator] = None
        self._bayesian: Optional[BayesianAccumulator] = None

        # Accumulated evidence vectors
        self._accumulated: Dict[str, np.ndarray] = {}

        # Statistics
        self._total_evidence = 0
        self._decisions_made = 0

    def register_source(
        self,
        name: str,
        reliability: float = 1.0,
        modality: Optional[ModalityType] = None,
        bias: float = 0.0,
    ) -> EvidenceSource:
        """Register an evidence source."""
        source = EvidenceSource(
            name=name,
            reliability=reliability,
            modality=modality,
            bias=bias,
        )
        self._sources[name] = source
        return source

    def get_source(self, name: str) -> Optional[EvidenceSource]:
        """Get registered source by name."""
        return self._sources.get(name)

    def create_evidence(
        self,
        source_name: str,
        evidence_type: EvidenceType,
        value: np.ndarray,
        strength: float = 1.0,
        uncertainty: float = 0.1,
    ) -> Evidence:
        """Create evidence from a registered source."""
        source = self._sources.get(source_name)
        if source is None:
            source = EvidenceSource(name=source_name)
            self._sources[source_name] = source

        self._total_evidence += 1

        return Evidence(
            source=source,
            evidence_type=evidence_type,
            value=value,
            strength=strength,
            timestamp=float(self._total_evidence),
            uncertainty=uncertainty,
        )

    def accumulate(
        self,
        hypothesis: str,
        evidence: Evidence,
    ) -> np.ndarray:
        """Accumulate evidence for a hypothesis."""
        if hypothesis not in self._evidence_buffers:
            self._evidence_buffers[hypothesis] = []
            self._accumulated[hypothesis] = np.zeros_like(evidence.value)

        self._evidence_buffers[hypothesis].append(evidence)

        # Leaky integration
        leak = 1 - self.config.leak_rate
        self._accumulated[hypothesis] = (
            leak * self._accumulated[hypothesis] + evidence.weight * evidence.value
        )

        return self._accumulated[hypothesis].copy()

    def get_accumulated(self, hypothesis: str) -> Optional[np.ndarray]:
        """Get accumulated evidence for a hypothesis."""
        return self._accumulated.get(hypothesis)

    def compare_hypotheses(
        self,
        hypothesis1: str,
        hypothesis2: str,
    ) -> Tuple[str, float]:
        """Compare evidence for two hypotheses."""
        acc1 = self._accumulated.get(hypothesis1, np.zeros(1))
        acc2 = self._accumulated.get(hypothesis2, np.zeros(1))

        strength1 = float(np.linalg.norm(acc1))
        strength2 = float(np.linalg.norm(acc2))

        if strength1 > strength2:
            ratio = strength1 / (strength2 + 1e-8)
            return hypothesis1, ratio
        else:
            ratio = strength2 / (strength1 + 1e-8)
            return hypothesis2, ratio

    def decide_binary(
        self,
        evidence_stream: List[Evidence],
    ) -> Tuple[int, float, float]:
        """
        Make binary decision from evidence stream.

        Returns:
            (decision, confidence, reaction_time)
        """
        if self._ddm is None:
            self._ddm = DriftDiffusionAccumulator(self.config)
        else:
            self._ddm.reset()

        for evidence in evidence_stream:
            decision = self._ddm.accumulate(evidence)
            if decision is not None:
                self._decisions_made += 1
                state, time = self._ddm.get_state()
                confidence = abs(state) / self.config.threshold
                return decision, confidence, time

        # Forced decision at timeout
        self._decisions_made += 1
        state, time = self._ddm.get_state()
        decision = 1 if state > 0 else 0
        confidence = abs(state) / self.config.threshold
        return decision, min(1.0, confidence), time

    def decide_multi(
        self,
        evidence_stream: List[Tuple[Evidence, np.ndarray]],
        n_hypotheses: int,
        threshold: float = 0.9,
    ) -> Tuple[int, np.ndarray]:
        """
        Make multi-alternative decision.

        Args:
            evidence_stream: List of (evidence, likelihoods) pairs
            n_hypotheses: Number of hypotheses
            threshold: Decision threshold

        Returns:
            (chosen_hypothesis, posterior_probabilities)
        """
        if self._bayesian is None or self._bayesian.n_hypotheses != n_hypotheses:
            self._bayesian = BayesianAccumulator(n_hypotheses, config=self.config)
        else:
            self._bayesian.reset()

        for evidence, likelihoods in evidence_stream:
            self._bayesian.update(evidence, likelihoods)

            if self._bayesian.is_decided(threshold):
                self._decisions_made += 1
                return (self._bayesian.get_map_hypothesis(), self._bayesian.get_posterior())

        # Return best guess
        self._decisions_made += 1
        return (self._bayesian.get_map_hypothesis(), self._bayesian.get_posterior())

    def integrate_module_evidence(
        self,
        module_evidence: Dict[str, SemanticEmbedding],
        hypothesis: str,
    ) -> np.ndarray:
        """
        Integrate evidence from multiple modules for a hypothesis.

        Args:
            module_evidence: {module_name: embedding} from different modules
            hypothesis: The hypothesis being evaluated

        Returns:
            Integrated evidence vector
        """
        combined = np.zeros(512)  # Default embedding dim
        total_weight = 0.0

        for module, embedding in module_evidence.items():
            source = self._sources.get(module)
            if source is None:
                reliability = 1.0
            else:
                reliability = source.reliability

            weight = embedding.confidence * reliability
            combined += weight * embedding.vector
            total_weight += weight

        if total_weight > 1e-8:
            combined /= total_weight

        # Create and accumulate evidence
        evidence = self.create_evidence(
            source_name="integrated",
            evidence_type=EvidenceType.INFERENCE,
            value=combined,
            strength=total_weight,
        )

        return self.accumulate(hypothesis, evidence)

    def update_source_reliability(
        self,
        source_name: str,
        accuracy: float,
        learning_rate: float = 0.1,
    ) -> None:
        """Update source reliability based on accuracy."""
        source = self._sources.get(source_name)
        if source is None:
            return

        # Exponential moving average
        source.reliability = (1 - learning_rate) * source.reliability + learning_rate * accuracy

    def clear(self) -> None:
        """Clear all accumulated evidence."""
        self._evidence_buffers = {}
        self._accumulated = {}
        if self._ddm:
            self._ddm.reset()
        if self._bayesian:
            self._bayesian.reset()

    def statistics(self) -> Dict[str, Any]:
        """Get accumulator statistics."""
        return {
            "n_sources": len(self._sources),
            "n_hypotheses": len(self._evidence_buffers),
            "total_evidence": self._total_evidence,
            "decisions_made": self._decisions_made,
            "source_reliabilities": {
                name: source.reliability for name, source in self._sources.items()
            },
        }
