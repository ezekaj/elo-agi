"""
Surprise-Modulated Learning

Implements learning rate adaptation based on prediction surprise:
- High surprise = important event = higher learning rate
- Low surprise = expected event = lower learning rate
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np


class SurpriseType(Enum):
    """Types of surprise."""

    REWARD = "reward"
    STATE = "state"
    ACTION = "action"
    OUTCOME = "outcome"


@dataclass
class SurpriseConfig:
    """Configuration for surprise modulation."""

    base_learning_rate: float = 0.01
    min_learning_rate: float = 0.001
    max_learning_rate: float = 0.1
    surprise_scale: float = 1.0
    memory_decay: float = 0.99
    consolidation_threshold: float = 0.8


@dataclass
class SurpriseMetrics:
    """Metrics from a surprise computation."""

    predicted: Any
    actual: Any
    surprise_value: float
    surprise_type: SurpriseType
    modulated_lr: float
    should_consolidate: bool
    kl_divergence: float = 0.0
    entropy: float = 0.0


class SurpriseModulatedLearning:
    """
    Surprise-modulated learning rate adaptation.

    Key insight: Surprising events are informative and should
    drive larger learning updates.
    """

    def __init__(
        self,
        config: Optional[SurpriseConfig] = None,
        random_seed: Optional[int] = None,
    ):
        self.config = config or SurpriseConfig()
        self._rng = np.random.default_rng(random_seed)

        self._prediction_history: Dict[str, List[float]] = {}
        self._surprise_history: List[SurpriseMetrics] = []
        self._running_mean: Dict[str, float] = {}
        self._running_var: Dict[str, float] = {}

        self._total_surprises = 0
        self._consolidation_triggers = 0

    def compute_surprise(
        self,
        predicted: Any,
        actual: Any,
        surprise_type: SurpriseType = SurpriseType.OUTCOME,
        context_key: Optional[str] = None,
    ) -> SurpriseMetrics:
        """
        Compute surprise between prediction and actual outcome.

        Args:
            predicted: Predicted value
            actual: Actual observed value
            surprise_type: Type of surprise
            context_key: Optional key for context-dependent baselines

        Returns:
            SurpriseMetrics with computed values
        """
        predicted_arr = self._to_array(predicted)
        actual_arr = self._to_array(actual)

        raw_surprise = self._compute_raw_surprise(predicted_arr, actual_arr)

        if context_key:
            normalized_surprise = self._normalize_surprise(raw_surprise, context_key)
        else:
            normalized_surprise = np.clip(raw_surprise, 0.0, 5.0) / 5.0

        kl_div = self._compute_kl_divergence(predicted_arr, actual_arr)
        entropy = self._compute_entropy(actual_arr)

        modulated_lr = self.modulated_learning_rate(
            self.config.base_learning_rate, normalized_surprise
        )

        should_consolidate = self.should_consolidate(normalized_surprise)

        metrics = SurpriseMetrics(
            predicted=predicted,
            actual=actual,
            surprise_value=normalized_surprise,
            surprise_type=surprise_type,
            modulated_lr=modulated_lr,
            should_consolidate=should_consolidate,
            kl_divergence=kl_div,
            entropy=entropy,
        )

        self._surprise_history.append(metrics)
        self._total_surprises += 1
        if should_consolidate:
            self._consolidation_triggers += 1

        return metrics

    def modulated_learning_rate(
        self,
        base_rate: float,
        surprise: float,
    ) -> float:
        """
        Compute modulated learning rate based on surprise.

        Args:
            base_rate: Base learning rate
            surprise: Surprise value [0, 1]

        Returns:
            Modulated learning rate
        """
        modulation = 1.0 + self.config.surprise_scale * surprise

        lr = base_rate * modulation

        lr = np.clip(lr, self.config.min_learning_rate, self.config.max_learning_rate)

        return float(lr)

    def should_consolidate(self, surprise: float) -> bool:
        """
        Determine if event should trigger memory consolidation.

        High-surprise events are candidates for consolidation.

        Args:
            surprise: Surprise value [0, 1]

        Returns:
            True if consolidation is recommended
        """
        return surprise >= self.config.consolidation_threshold

    def get_adaptive_lr(
        self,
        module_id: str,
        predicted: Any,
        actual: Any,
    ) -> float:
        """
        Get adaptive learning rate for a module based on surprise.

        Args:
            module_id: Module identifier
            predicted: Module's prediction
            actual: Actual outcome

        Returns:
            Adaptive learning rate
        """
        metrics = self.compute_surprise(
            predicted,
            actual,
            surprise_type=SurpriseType.OUTCOME,
            context_key=module_id,
        )
        return metrics.modulated_lr

    def update_baseline(
        self,
        context_key: str,
        value: float,
    ) -> None:
        """
        Update running baseline for a context.

        Args:
            context_key: Context identifier
            value: New value to incorporate
        """
        if context_key not in self._running_mean:
            self._running_mean[context_key] = value
            self._running_var[context_key] = 0.0
        else:
            decay = self.config.memory_decay
            old_mean = self._running_mean[context_key]
            self._running_mean[context_key] = decay * old_mean + (1 - decay) * value

            diff = value - old_mean
            self._running_var[context_key] = (
                decay * self._running_var[context_key] + (1 - decay) * diff * diff
            )

    def get_recent_surprises(
        self,
        n: int = 10,
        surprise_type: Optional[SurpriseType] = None,
    ) -> List[SurpriseMetrics]:
        """Get most recent surprise events."""
        history = self._surprise_history
        if surprise_type:
            history = [s for s in history if s.surprise_type == surprise_type]
        return history[-n:]

    def get_average_surprise(
        self,
        window: int = 100,
        surprise_type: Optional[SurpriseType] = None,
    ) -> float:
        """Get average surprise over recent window."""
        recent = self.get_recent_surprises(window, surprise_type)
        if not recent:
            return 0.0
        return float(np.mean([s.surprise_value for s in recent]))

    def _to_array(self, value: Any) -> np.ndarray:
        """Convert value to numpy array."""
        if isinstance(value, np.ndarray):
            return value.flatten()
        elif isinstance(value, (list, tuple)):
            return np.array(value).flatten()
        elif isinstance(value, (int, float)):
            return np.array([value])
        else:
            return np.array([hash(str(value)) % 1000 / 1000.0])

    def _compute_raw_surprise(
        self,
        predicted: np.ndarray,
        actual: np.ndarray,
    ) -> float:
        """Compute raw surprise as prediction error."""
        if len(predicted) != len(actual):
            min_len = min(len(predicted), len(actual))
            predicted = predicted[:min_len]
            actual = actual[:min_len]

        if len(predicted) == 0:
            return 0.0

        mse = np.mean((predicted - actual) ** 2)
        surprise = np.sqrt(mse)

        return float(surprise)

    def _normalize_surprise(
        self,
        raw_surprise: float,
        context_key: str,
    ) -> float:
        """Normalize surprise using running statistics."""
        self.update_baseline(context_key, raw_surprise)

        mean = self._running_mean.get(context_key, raw_surprise)
        var = self._running_var.get(context_key, 1.0)
        std = max(np.sqrt(var), 1e-6)

        z_score = (raw_surprise - mean) / std

        normalized = 1.0 / (1.0 + np.exp(-z_score))

        return float(normalized)

    def _compute_kl_divergence(
        self,
        predicted: np.ndarray,
        actual: np.ndarray,
    ) -> float:
        """Compute KL divergence between predicted and actual distributions."""
        p = np.clip(predicted, 1e-8, 1.0)
        q = np.clip(actual, 1e-8, 1.0)

        if np.sum(p) > 1e-8:
            p = p / np.sum(p)
        if np.sum(q) > 1e-8:
            q = q / np.sum(q)

        kl = np.sum(p * np.log(p / q))
        return float(max(0.0, kl))

    def _compute_entropy(self, distribution: np.ndarray) -> float:
        """Compute entropy of a distribution."""
        p = np.clip(distribution, 1e-8, 1.0)
        if np.sum(p) > 1e-8:
            p = p / np.sum(p)
        entropy = -np.sum(p * np.log(p))
        return float(max(0.0, entropy))

    def statistics(self) -> Dict[str, Any]:
        """Get surprise modulation statistics."""
        return {
            "total_surprises": self._total_surprises,
            "consolidation_triggers": self._consolidation_triggers,
            "consolidation_rate": (self._consolidation_triggers / max(1, self._total_surprises)),
            "average_surprise": self.get_average_surprise(),
            "contexts_tracked": len(self._running_mean),
            "history_size": len(self._surprise_history),
        }
