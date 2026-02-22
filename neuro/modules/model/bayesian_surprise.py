"""
Bayesian Surprise Engine

Detects significant/novel events using Bayesian inference.
Computes surprise as KL divergence between prior and posterior beliefs.
"""

import math
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Observation:
    """An observation/event to evaluate for surprise."""

    type: str
    value: Any
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SurpriseResult:
    """Result of surprise computation."""

    surprise: float
    prior: float
    posterior: float
    is_surprising: bool
    observation: Observation
    explanation: str


class BayesianSurprise:
    """
    Bayesian surprise detector using KL divergence.

    Maintains belief states about different observation types
    and computes how surprising new observations are.
    """

    def __init__(
        self,
        surprise_threshold: float = 0.5,
        learning_rate: float = 0.1,
        prior_pseudocount: float = 1.0,
    ):
        self.surprise_threshold = surprise_threshold
        self.learning_rate = learning_rate
        self.prior_pseudocount = prior_pseudocount

        # Belief states
        self.type_beliefs: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(lambda: self.prior_pseudocount)
        )
        self.type_counts: Dict[str, int] = defaultdict(int)

        # History
        self.history: List[SurpriseResult] = []
        self.surprise_history: List[float] = []

    def _kl_divergence(self, p: float, q: float) -> float:
        """
        Compute KL divergence D_KL(P || Q) for Bernoulli distributions.

        Uses symmetric KL divergence for stability.
        """
        # Clip to avoid log(0)
        p = max(min(p, 0.999), 0.001)
        q = max(min(q, 0.999), 0.001)

        # Symmetric KL divergence
        kl_pq = p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
        kl_qp = q * math.log(q / p) + (1 - q) * math.log((1 - q) / (1 - p))

        return (kl_pq + kl_qp) / 2

    def _get_prior(self, obs_type: str, obs_value: Any) -> float:
        """Get prior probability for an observation."""
        if obs_type not in self.type_beliefs:
            return 0.5  # Maximum uncertainty

        beliefs = self.type_beliefs[obs_type]
        total = sum(beliefs.values())

        if total == 0:
            return 0.5

        # Get belief for this value
        value_key = str(obs_value)[:100]  # Truncate long values
        return beliefs[value_key] / total

    def _update_belief(self, obs_type: str, obs_value: Any) -> float:
        """Update belief state and return posterior probability."""
        value_key = str(obs_value)[:100]

        # Update counts
        self.type_beliefs[obs_type][value_key] += 1
        self.type_counts[obs_type] += 1

        # Compute posterior
        total = sum(self.type_beliefs[obs_type].values())
        return self.type_beliefs[obs_type][value_key] / total

    def compute_surprise(self, observation: Observation) -> SurpriseResult:
        """
        Compute surprise for an observation.

        Returns:
            SurpriseResult with surprise value and metadata
        """
        obs_type = observation.type
        obs_value = observation.value

        # Get prior belief
        prior = self._get_prior(obs_type, obs_value)

        # Update belief and get posterior
        posterior = self._update_belief(obs_type, obs_value)

        # Compute surprise (KL divergence)
        surprise = self._kl_divergence(prior, posterior)

        # Determine if surprising
        is_surprising = surprise > self.surprise_threshold

        # Generate explanation
        if is_surprising:
            if prior < 0.1:
                explanation = (
                    f"Novel event: '{obs_value}' was rarely seen before (prior={prior:.2%})"
                )
            elif posterior > prior * 2:
                explanation = f"Frequency shift: '{obs_value}' is becoming more common"
            else:
                explanation = f"Unexpected: prior={prior:.2%}, posterior={posterior:.2%}"
        else:
            explanation = f"Expected: '{obs_value}' matches current beliefs (prior={prior:.2%})"

        result = SurpriseResult(
            surprise=surprise,
            prior=prior,
            posterior=posterior,
            is_surprising=is_surprising,
            observation=observation,
            explanation=explanation,
        )

        # Record history
        self.history.append(result)
        self.surprise_history.append(surprise)

        return result

    def is_surprising(self, observation: Observation) -> bool:
        """Quick check if an observation is surprising."""
        result = self.compute_surprise(observation)
        return result.is_surprising

    def get_most_surprising(self, k: int = 5) -> List[SurpriseResult]:
        """Get k most surprising events from history."""
        sorted_history = sorted(self.history, key=lambda x: x.surprise, reverse=True)
        return sorted_history[:k]

    def get_average_surprise(self, window: int = 100) -> float:
        """Get average surprise over recent history."""
        if not self.surprise_history:
            return 0.0

        recent = self.surprise_history[-window:]
        return sum(recent) / len(recent)

    def get_beliefs_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of current beliefs."""
        summary = {}
        for obs_type, beliefs in self.type_beliefs.items():
            total = sum(beliefs.values())
            if total > 0:
                # Get top 5 values for each type
                sorted_beliefs = sorted(beliefs.items(), key=lambda x: x[1], reverse=True)[:5]
                summary[obs_type] = {k: v / total for k, v in sorted_beliefs}
        return summary

    def reset_beliefs(self, obs_type: str = None) -> None:
        """Reset beliefs for a specific type or all types."""
        if obs_type:
            self.type_beliefs[obs_type] = defaultdict(lambda: self.prior_pseudocount)
            self.type_counts[obs_type] = 0
        else:
            self.type_beliefs = defaultdict(lambda: defaultdict(lambda: self.prior_pseudocount))
            self.type_counts = defaultdict(int)

    def get_stats(self) -> Dict[str, Any]:
        """Get surprise engine statistics."""
        return {
            "total_observations": sum(self.type_counts.values()),
            "observation_types": len(self.type_beliefs),
            "history_size": len(self.history),
            "average_surprise": self.get_average_surprise(),
            "surprising_events": sum(1 for r in self.history if r.is_surprising),
            "threshold": self.surprise_threshold,
        }


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("BAYESIAN SURPRISE TEST")
    print("=" * 60)

    surprise_engine = BayesianSurprise(surprise_threshold=0.3)

    # Simulate observations
    observations = [
        Observation(type="topic", value="AI"),
        Observation(type="topic", value="AI"),
        Observation(type="topic", value="AI"),
        Observation(type="topic", value="python"),
        Observation(type="topic", value="AI"),
        Observation(type="topic", value="quantum computing"),  # Novel!
        Observation(type="entity", value="Claude"),
        Observation(type="entity", value="GPT"),
        Observation(type="entity", value="Claude"),
        Observation(type="entity", value="LLaMA"),  # Novel!
    ]

    print("\nProcessing observations:")
    for obs in observations:
        result = surprise_engine.compute_surprise(obs)
        status = "SURPRISING!" if result.is_surprising else "expected"
        print(f"  [{obs.type}={obs.value}] surprise={result.surprise:.3f} ({status})")

    print("\nMost surprising events:")
    for result in surprise_engine.get_most_surprising(3):
        print(f"  [{result.surprise:.3f}] {result.observation.type}={result.observation.value}")
        print(f"    {result.explanation}")

    print(f"\nStats: {surprise_engine.get_stats()}")
    print(f"\nBeliefs summary: {surprise_engine.get_beliefs_summary()}")
