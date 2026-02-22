"""
Meta-Learning Controller for Adaptive Memory Consolidation.

Implements learning-to-learn mechanisms that adapt:
- Replay prioritization weights based on consolidation success
- Learning rates per memory type
- Optimal replay counts prediction
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np


class MemoryType(Enum):
    """Types of memories with different learning dynamics."""

    EPISODIC = "episodic"  # Specific events
    SEMANTIC = "semantic"  # Facts and concepts
    PROCEDURAL = "procedural"  # Skills and habits
    EMOTIONAL = "emotional"  # Emotionally salient


@dataclass
class LearningCurve:
    """Tracks learning progress for a specific memory."""

    memory_id: str
    memory_type: MemoryType
    consolidation_history: List[float] = field(default_factory=list)
    replay_counts: List[int] = field(default_factory=list)
    learning_rate: float = 0.1
    optimal_replay_count: int = 5
    total_replays: int = 0
    is_consolidated: bool = False
    created_at: float = 0.0

    def add_consolidation_step(self, strength: float, replays: int) -> None:
        """Record a consolidation step."""
        self.consolidation_history.append(strength)
        self.replay_counts.append(replays)
        self.total_replays += replays

    def get_learning_velocity(self, window: int = 3) -> float:
        """Compute recent learning velocity (improvement rate)."""
        if len(self.consolidation_history) < 2:
            return 0.0

        history = self.consolidation_history[-window:]
        if len(history) < 2:
            return history[-1] - history[0] if len(history) == 2 else 0.0

        # Linear regression slope
        x = np.arange(len(history))
        slope = np.polyfit(x, history, 1)[0]
        return float(slope)

    def get_efficiency(self) -> float:
        """Compute consolidation efficiency (progress per replay)."""
        if self.total_replays == 0:
            return 0.0

        progress = self.consolidation_history[-1] if self.consolidation_history else 0.0
        initial = self.consolidation_history[0] if self.consolidation_history else 0.0
        return (progress - initial) / self.total_replays


@dataclass
class ConsolidationOutcome:
    """Outcome of a consolidation attempt."""

    memory_id: str
    before_strength: float
    after_strength: float
    replays_used: int
    success: bool  # True if strength increased meaningfully
    efficiency: float  # Strength gain per replay

    @property
    def improvement(self) -> float:
        return self.after_strength - self.before_strength


@dataclass
class ReplayWeights:
    """Weights for replay prioritization."""

    recency: float = 0.4
    emotional_salience: float = 0.3
    incompleteness: float = 0.3
    interference_risk: float = 0.0  # New: prioritize memories at risk

    def normalize(self) -> "ReplayWeights":
        """Normalize weights to sum to 1."""
        total = (
            self.recency + self.emotional_salience + self.incompleteness + self.interference_risk
        )
        if total < 1e-8:
            return ReplayWeights()
        return ReplayWeights(
            recency=self.recency / total,
            emotional_salience=self.emotional_salience / total,
            incompleteness=self.incompleteness / total,
            interference_risk=self.interference_risk / total,
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "recency": self.recency,
            "emotional_salience": self.emotional_salience,
            "incompleteness": self.incompleteness,
            "interference_risk": self.interference_risk,
        }


class MetaLearningController:
    """
    Controls adaptive learning behavior across consolidation cycles.

    Implements:
    - Tracking consolidation success per memory
    - Adapting replay weights based on outcomes
    - Predicting optimal replay counts
    - Updating learning rates per memory type
    """

    def __init__(
        self,
        initial_weights: Optional[ReplayWeights] = None,
        adaptation_rate: float = 0.1,
        min_samples_for_adaptation: int = 10,
        random_seed: Optional[int] = None,
    ):
        self.weights = initial_weights or ReplayWeights()
        self.adaptation_rate = adaptation_rate
        self.min_samples_for_adaptation = min_samples_for_adaptation
        self._rng = np.random.default_rng(random_seed)

        # Learning curves per memory
        self._learning_curves: Dict[str, LearningCurve] = {}

        # Outcomes history
        self._outcomes: List[ConsolidationOutcome] = []

        # Type-specific learning rates
        self._type_learning_rates: Dict[MemoryType, float] = {
            MemoryType.EPISODIC: 0.15,
            MemoryType.SEMANTIC: 0.08,
            MemoryType.PROCEDURAL: 0.05,
            MemoryType.EMOTIONAL: 0.20,
        }

        # Type-specific success rates (for adaptation)
        self._type_success_rates: Dict[MemoryType, List[float]] = {mt: [] for mt in MemoryType}

        # Statistics
        self._n_adaptations = 0
        self._n_predictions = 0

    def register_memory(
        self,
        memory_id: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        initial_strength: float = 0.0,
        timestamp: float = 0.0,
    ) -> LearningCurve:
        """Register a new memory for tracking."""
        curve = LearningCurve(
            memory_id=memory_id,
            memory_type=memory_type,
            learning_rate=self._type_learning_rates[memory_type],
            created_at=timestamp,
        )
        if initial_strength > 0:
            curve.add_consolidation_step(initial_strength, 0)

        self._learning_curves[memory_id] = curve
        return curve

    def get_learning_curve(self, memory_id: str) -> Optional[LearningCurve]:
        """Get learning curve for a memory."""
        return self._learning_curves.get(memory_id)

    def track_consolidation_success(
        self,
        memory_id: str,
        before_strength: float,
        after_strength: float,
        replays_used: int = 1,
    ) -> ConsolidationOutcome:
        """
        Track the outcome of a consolidation attempt.

        Returns outcome with success metrics.
        """
        improvement = after_strength - before_strength
        efficiency = improvement / max(1, replays_used)
        success = improvement > 0.01  # Meaningful improvement threshold

        outcome = ConsolidationOutcome(
            memory_id=memory_id,
            before_strength=before_strength,
            after_strength=after_strength,
            replays_used=replays_used,
            success=success,
            efficiency=efficiency,
        )
        self._outcomes.append(outcome)

        # Update learning curve
        curve = self._learning_curves.get(memory_id)
        if curve:
            curve.add_consolidation_step(after_strength, replays_used)
            if after_strength >= 0.9:
                curve.is_consolidated = True

            # Track success rate by type
            self._type_success_rates[curve.memory_type].append(1.0 if success else 0.0)

        return outcome

    def adapt_replay_weights(
        self,
        recent_window: int = 20,
    ) -> ReplayWeights:
        """
        Adapt replay weights based on recent consolidation outcomes.

        Analyzes which memory characteristics led to successful consolidation.
        """
        if len(self._outcomes) < self.min_samples_for_adaptation:
            return self.weights

        recent = self._outcomes[-recent_window:]
        self._n_adaptations += 1

        # Analyze correlations between memory attributes and success
        successes = [o for o in recent if o.success]
        failures = [o for o in recent if not o.success]

        if not successes or not failures:
            return self.weights

        # Compute average characteristics of successful vs failed memories
        success_curves = [
            self._learning_curves.get(o.memory_id)
            for o in successes
            if o.memory_id in self._learning_curves
        ]
        failure_curves = [
            self._learning_curves.get(o.memory_id)
            for o in failures
            if o.memory_id in self._learning_curves
        ]

        success_curves = [c for c in success_curves if c is not None]
        failure_curves = [c for c in failure_curves if c is not None]

        if not success_curves or not failure_curves:
            return self.weights

        # Adjust weights based on what works
        # If incomplete memories consolidate better, increase incompleteness weight
        success_incomplete = np.mean(
            [
                1.0 - (c.consolidation_history[-1] if c.consolidation_history else 0.0)
                for c in success_curves
            ]
        )
        failure_incomplete = np.mean(
            [
                1.0 - (c.consolidation_history[-1] if c.consolidation_history else 0.0)
                for c in failure_curves
            ]
        )

        # Adjust incompleteness weight
        if success_incomplete > failure_incomplete + 0.1:
            # Incomplete memories are consolidating well
            new_incompleteness = self.weights.incompleteness + self.adaptation_rate * 0.1
        elif failure_incomplete > success_incomplete + 0.1:
            # Complete memories need more attention
            new_incompleteness = self.weights.incompleteness - self.adaptation_rate * 0.1
        else:
            new_incompleteness = self.weights.incompleteness

        # Emotional salience adaptation
        success_emotional = sum(
            1 for c in success_curves if c.memory_type == MemoryType.EMOTIONAL
        ) / max(1, len(success_curves))
        failure_emotional = sum(
            1 for c in failure_curves if c.memory_type == MemoryType.EMOTIONAL
        ) / max(1, len(failure_curves))

        if success_emotional > failure_emotional + 0.1:
            new_emotional = self.weights.emotional_salience + self.adaptation_rate * 0.1
        elif failure_emotional > success_emotional + 0.1:
            new_emotional = self.weights.emotional_salience - self.adaptation_rate * 0.1
        else:
            new_emotional = self.weights.emotional_salience

        # Bound weights
        new_incompleteness = max(0.1, min(0.5, new_incompleteness))
        new_emotional = max(0.1, min(0.5, new_emotional))
        new_recency = 1.0 - new_incompleteness - new_emotional - self.weights.interference_risk

        self.weights = ReplayWeights(
            recency=max(0.1, new_recency),
            emotional_salience=new_emotional,
            incompleteness=new_incompleteness,
            interference_risk=self.weights.interference_risk,
        ).normalize()

        return self.weights

    def predict_optimal_replays(
        self,
        memory_id: str,
        target_strength: float = 0.9,
    ) -> int:
        """
        Predict optimal number of replays needed for a memory.

        Uses learning curve history to estimate.
        """
        self._n_predictions += 1

        curve = self._learning_curves.get(memory_id)
        if curve is None:
            # Default based on memory type average
            return 5

        if curve.is_consolidated:
            return 0

        current = curve.consolidation_history[-1] if curve.consolidation_history else 0.0
        if current >= target_strength:
            return 0

        # Estimate from learning velocity
        velocity = curve.get_learning_velocity()
        if velocity <= 0:
            # No progress, use type default
            return self._get_type_default_replays(curve.memory_type)

        # Estimate replays needed
        gap = target_strength - current
        efficiency = curve.get_efficiency()

        if efficiency <= 0:
            return self._get_type_default_replays(curve.memory_type)

        estimated = int(np.ceil(gap / efficiency))
        # Bound by reasonable limits
        return max(1, min(50, estimated))

    def _get_type_default_replays(self, memory_type: MemoryType) -> int:
        """Get default replay count for a memory type."""
        defaults = {
            MemoryType.EPISODIC: 5,
            MemoryType.SEMANTIC: 8,
            MemoryType.PROCEDURAL: 15,
            MemoryType.EMOTIONAL: 3,
        }
        return defaults.get(memory_type, 5)

    def update_learning_rates(
        self,
        memory_type: MemoryType,
        success_rate: float,
    ) -> float:
        """
        Update learning rate for a memory type based on success rate.

        High success -> can use higher learning rate
        Low success -> need slower, more careful learning
        """
        current = self._type_learning_rates[memory_type]

        if success_rate > 0.8:
            # High success, can increase learning rate
            new_rate = current * (1 + self.adaptation_rate)
        elif success_rate < 0.4:
            # Low success, decrease learning rate
            new_rate = current * (1 - self.adaptation_rate)
        else:
            new_rate = current

        # Bound learning rate
        new_rate = max(0.01, min(0.5, new_rate))
        self._type_learning_rates[memory_type] = new_rate

        return new_rate

    def get_learning_rate(self, memory_type: MemoryType) -> float:
        """Get current learning rate for a memory type."""
        return self._type_learning_rates[memory_type]

    def get_type_success_rate(
        self,
        memory_type: MemoryType,
        window: int = 20,
    ) -> float:
        """Get recent success rate for a memory type."""
        rates = self._type_success_rates.get(memory_type, [])
        if not rates:
            return 0.5  # Default

        recent = rates[-window:]
        return float(np.mean(recent))

    def compute_priority_score(
        self,
        memory_id: str,
        recency_score: float,
        emotional_score: float,
        incompleteness_score: float,
        interference_score: float = 0.0,
    ) -> float:
        """Compute weighted priority score for a memory."""
        return (
            self.weights.recency * recency_score
            + self.weights.emotional_salience * emotional_score
            + self.weights.incompleteness * incompleteness_score
            + self.weights.interference_risk * interference_score
        )

    def get_memories_needing_replay(
        self,
        top_k: int = 10,
    ) -> List[Tuple[str, int]]:
        """
        Get memories that need replay, with recommended counts.

        Returns list of (memory_id, recommended_replays).
        """
        needs_replay = []

        for memory_id, curve in self._learning_curves.items():
            if curve.is_consolidated:
                continue

            current = curve.consolidation_history[-1] if curve.consolidation_history else 0.0
            if current < 0.9:
                replays = self.predict_optimal_replays(memory_id)
                needs_replay.append((memory_id, replays, current))

        # Sort by consolidation level (lowest first)
        needs_replay.sort(key=lambda x: x[2])

        return [(m_id, replays) for m_id, replays, _ in needs_replay[:top_k]]

    def reset_for_new_night(self) -> None:
        """Reset per-night counters while preserving learning."""
        # Keep learning curves and weights
        # Just mark a new night boundary
        pass

    def statistics(self) -> Dict[str, Any]:
        """Get meta-learning statistics."""
        total_memories = len(self._learning_curves)
        consolidated = sum(1 for c in self._learning_curves.values() if c.is_consolidated)

        # Type distribution
        type_counts = {mt.value: 0 for mt in MemoryType}
        for curve in self._learning_curves.values():
            type_counts[curve.memory_type.value] += 1

        # Recent success rate
        recent_outcomes = self._outcomes[-20:] if self._outcomes else []
        recent_success_rate = (
            sum(1 for o in recent_outcomes if o.success) / len(recent_outcomes)
            if recent_outcomes
            else 0.0
        )

        # Average efficiency
        avg_efficiency = np.mean([o.efficiency for o in self._outcomes]) if self._outcomes else 0.0

        return {
            "total_memories": total_memories,
            "consolidated_memories": consolidated,
            "consolidation_rate": consolidated / max(1, total_memories),
            "type_distribution": type_counts,
            "current_weights": self.weights.to_dict(),
            "type_learning_rates": {
                mt.value: rate for mt, rate in self._type_learning_rates.items()
            },
            "n_outcomes": len(self._outcomes),
            "n_adaptations": self._n_adaptations,
            "n_predictions": self._n_predictions,
            "recent_success_rate": recent_success_rate,
            "average_efficiency": float(avg_efficiency),
        }


__all__ = [
    "MemoryType",
    "LearningCurve",
    "ConsolidationOutcome",
    "ReplayWeights",
    "MetaLearningController",
]
