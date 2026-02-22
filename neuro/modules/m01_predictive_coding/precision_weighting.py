"""
Precision-Weighted Prediction Error System

Implements: ξ = Π̃ε̃ (precision-weighted prediction error)

Precision represents confidence in predictions. High precision means
predictions are reliable; errors from these should be weighted more heavily.
Low precision indicates uncertainty; errors should be downweighted.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class PrecisionMode(Enum):
    """Different strategies for precision estimation"""

    FIXED = "fixed"
    VARIANCE = "variance"
    EXPONENTIAL = "exponential"
    BAYESIAN = "bayesian"


@dataclass
class PrecisionState:
    """Container for precision-related state"""

    value: float
    variance_estimate: float
    confidence: float
    n_samples: int


class PrecisionWeightedError:
    """Computes precision-weighted prediction errors.

    The core computation: ξ = Π * ε
    where ξ is weighted error, Π is precision, ε is raw error.
    """

    def __init__(
        self,
        dim: int,
        initial_precision: float = 1.0,
        min_precision: float = 0.01,
        max_precision: float = 100.0,
    ):
        self.dim = dim
        self.min_precision = min_precision
        self.max_precision = max_precision

        # Per-dimension precision (allows anisotropic precision)
        self.precision = np.full(dim, initial_precision)

        # Running statistics for variance estimation
        self.mean = np.zeros(dim)
        self.M2 = np.zeros(dim)  # For Welford's algorithm
        self.n_samples = 0

    def compute_error(self, predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
        """Compute raw prediction error.

        Args:
            predicted: Model's prediction
            actual: Observed value

        Returns:
            Raw prediction error ε = actual - predicted
        """
        return actual - predicted

    def weighted_error(
        self, predicted: np.ndarray, actual: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute precision-weighted prediction error.

        Args:
            predicted: Model's prediction
            actual: Observed value

        Returns:
            Tuple of (weighted_error ξ, raw_error ε)
        """
        raw_error = self.compute_error(predicted, actual)
        weighted = self.precision * raw_error
        return weighted, raw_error

    def update_statistics(self, error: np.ndarray) -> None:
        """Update running statistics with new error observation.

        Uses Welford's online algorithm for numerical stability.
        """
        self.n_samples += 1
        delta = error - self.mean
        self.mean += delta / self.n_samples
        delta2 = error - self.mean
        self.M2 += delta * delta2

    def estimate_precision_from_variance(self) -> np.ndarray:
        """Estimate precision as inverse variance.

        Π = 1 / σ² where σ² is estimated from error history.
        """
        if self.n_samples < 2:
            return self.precision

        variance = self.M2 / (self.n_samples - 1)
        variance = np.maximum(variance, 1e-8)  # Avoid division by zero

        precision = 1.0 / variance
        precision = np.clip(precision, self.min_precision, self.max_precision)

        return precision

    def update_precision(self, error: np.ndarray) -> np.ndarray:
        """Update precision based on new error observation.

        Args:
            error: New prediction error

        Returns:
            Updated precision values
        """
        self.update_statistics(error)
        self.precision = self.estimate_precision_from_variance()
        return self.precision

    def get_scalar_precision(self) -> float:
        """Get overall scalar precision (mean across dimensions)"""
        return float(np.mean(self.precision))

    def reset(self) -> None:
        """Reset statistics"""
        self.mean = np.zeros(self.dim)
        self.M2 = np.zeros(self.dim)
        self.n_samples = 0


class AdaptivePrecision:
    """Adaptive precision that responds to environmental volatility.

    - Precision increases when predictions are reliable (low error variance)
    - Precision decreases during uncertainty/volatility
    - Uses hierarchical estimation: precision of precision
    """

    def __init__(
        self,
        dim: int,
        learning_rate: float = 0.1,
        volatility_learning_rate: float = 0.01,
        initial_precision: float = 1.0,
        initial_volatility: float = 0.1,
    ):
        self.dim = dim
        self.learning_rate = learning_rate
        self.volatility_learning_rate = volatility_learning_rate

        # First-level precision (inverse variance of observations)
        self.precision = np.full(dim, initial_precision)

        # Second-level: volatility (how much precision itself changes)
        self.volatility = np.full(dim, initial_volatility)

        # Expected precision (prior)
        self.expected_precision = np.full(dim, initial_precision)

        # History for trend detection
        self.precision_history: List[np.ndarray] = []
        self.error_history: List[np.ndarray] = []
        self.max_history = 50

    def update(
        self, error: np.ndarray, surprise: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Update precision based on prediction error.

        Args:
            error: Current prediction error
            surprise: Optional explicit surprise signal

        Returns:
            Tuple of (updated precision, volatility)
        """
        # Store history
        self.error_history.append(error.copy())
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)

        # Compute local variance from recent errors
        if len(self.error_history) >= 2:
            recent_errors = np.array(self.error_history[-10:])
            local_variance = np.var(recent_errors, axis=0) + 1e-8
            observed_precision = 1.0 / local_variance
        else:
            observed_precision = self.precision

        # Precision update: move toward observed precision
        # Rate modulated by volatility (high volatility = faster adaptation)
        precision_error = observed_precision - self.precision
        self.precision += self.learning_rate * self.volatility * precision_error
        self.precision = np.clip(self.precision, 0.01, 100.0)

        # Volatility update: track how much precision is changing
        if len(self.precision_history) >= 2:
            precision_change = np.abs(self.precision - self.precision_history[-1])
            volatility_error = precision_change - self.volatility
            self.volatility += self.volatility_learning_rate * volatility_error
            self.volatility = np.clip(self.volatility, 0.001, 10.0)

        self.precision_history.append(self.precision.copy())
        if len(self.precision_history) > self.max_history:
            self.precision_history.pop(0)

        return self.precision, self.volatility

    def get_weighted_error(self, error: np.ndarray) -> np.ndarray:
        """Get precision-weighted error"""
        return self.precision * error

    def get_confidence(self) -> np.ndarray:
        """Get confidence level (normalized precision)"""
        return self.precision / (self.precision + 1)

    def is_volatile(self, threshold: float = 0.5) -> bool:
        """Check if environment is currently volatile"""
        return float(np.mean(self.volatility)) > threshold

    def detect_regime_change(self, window: int = 10) -> bool:
        """Detect if there's been a regime change (sudden volatility spike)"""
        if len(self.precision_history) < window * 2:
            return False

        recent = np.array(self.precision_history[-window:])
        earlier = np.array(self.precision_history[-window * 2 : -window])

        recent_var = np.var(recent, axis=0)
        earlier_var = np.var(earlier, axis=0)

        ratio = np.mean(recent_var) / (np.mean(earlier_var) + 1e-8)
        return ratio > 2.0  # Significant increase in precision variability

    def reset(self) -> None:
        """Reset to initial state"""
        self.precision = np.full(self.dim, 1.0)
        self.volatility = np.full(self.dim, 0.1)
        self.precision_history = []
        self.error_history = []


class HierarchicalPrecision:
    """Hierarchical precision for multi-level predictive coding.

    Different levels of the hierarchy have different precision dynamics:
    - Lower levels: faster precision adaptation (sensory)
    - Higher levels: slower precision changes (abstract beliefs)
    """

    def __init__(
        self, level_dims: List[int], base_learning_rate: float = 0.1, timescale_factor: float = 2.0
    ):
        self.n_levels = len(level_dims)

        # Create adaptive precision for each level
        self.levels: List[AdaptivePrecision] = []
        for i, dim in enumerate(level_dims):
            lr = base_learning_rate / (timescale_factor**i)
            vlr = lr * 0.1
            level = AdaptivePrecision(dim=dim, learning_rate=lr, volatility_learning_rate=vlr)
            self.levels.append(level)

    def update(self, errors: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Update precision at all levels.

        Args:
            errors: List of errors at each level

        Returns:
            Tuple of (precisions, volatilities) at each level
        """
        precisions = []
        volatilities = []

        for level, error in zip(self.levels, errors):
            p, v = level.update(error)
            precisions.append(p)
            volatilities.append(v)

        return precisions, volatilities

    def get_weighted_errors(self, errors: List[np.ndarray]) -> List[np.ndarray]:
        """Get precision-weighted errors at all levels"""
        return [level.get_weighted_error(error) for level, error in zip(self.levels, errors)]

    def get_overall_confidence(self) -> float:
        """Get overall system confidence (average across levels)"""
        confidences = [np.mean(level.get_confidence()) for level in self.levels]
        return float(np.mean(confidences))

    def reset(self) -> None:
        """Reset all levels"""
        for level in self.levels:
            level.reset()
