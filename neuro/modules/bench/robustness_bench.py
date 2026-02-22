"""
Robustness Benchmark: Tests for uncertainty quantification and out-of-distribution detection.

Evaluates:
- Out-of-distribution detection AUROC
- Calibration error (Expected Calibration Error)
- Adversarial robustness
- Selective prediction quality
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from .base_benchmark import Benchmark, BenchmarkConfig


@dataclass
class OODSample:
    """An out-of-distribution detection sample."""

    features: np.ndarray
    is_ood: bool  # True if out-of-distribution
    domain: str  # "in_distribution", "near_ood", "far_ood"


@dataclass
class CalibrationSample:
    """A calibration evaluation sample."""

    features: np.ndarray
    true_class: int
    n_classes: int


@dataclass
class AdversarialSample:
    """An adversarial robustness sample."""

    original: np.ndarray
    perturbation: np.ndarray
    true_class: int
    epsilon: float  # Perturbation budget


class OODDetectionBenchmark(Benchmark):
    """
    Benchmark for out-of-distribution detection.

    Tests agent's ability to distinguish between in-distribution
    and out-of-distribution inputs.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        config = config or BenchmarkConfig(
            name="ood_detection",
            description="Out-of-distribution detection benchmark",
            n_trials=100,
        )
        super().__init__(config)
        self.feature_dim = 32
        self.in_dist_mean = np.zeros(self.feature_dim)
        self.in_dist_std = 1.0

    @property
    def name(self) -> str:
        return "ood_detection"

    def generate_trial(self, trial_id: int) -> Tuple[OODSample, bool]:
        """Generate an OOD detection trial."""
        # Decide if this sample is OOD
        is_ood = self._rng.random() < 0.5

        if is_ood:
            # Generate OOD sample
            ood_type = self._rng.choice(["near", "far"])

            if ood_type == "near":
                # Near OOD: shifted mean
                shift = self._rng.uniform(2, 4)
                mean = self.in_dist_mean + shift
                features = self._rng.normal(mean, self.in_dist_std)
                domain = "near_ood"
            else:
                # Far OOD: different distribution
                features = self._rng.uniform(-5, 5, self.feature_dim)
                domain = "far_ood"
        else:
            # In-distribution sample
            features = self._rng.normal(self.in_dist_mean, self.in_dist_std)
            domain = "in_distribution"

        sample = OODSample(
            features=features,
            is_ood=is_ood,
            domain=domain,
        )

        return sample, is_ood

    def evaluate(self, expected: bool, actual: Any) -> Tuple[bool, float]:
        """Evaluate OOD detection."""
        if actual is None:
            return False, 0.0

        # actual should be a score (higher = more likely OOD)
        # or a boolean prediction
        if isinstance(actual, bool):
            predicted_ood = actual
            score = 1.0 if predicted_ood == expected else 0.0
        elif isinstance(actual, (int, float)):
            # Threshold at 0.5
            predicted_ood = actual > 0.5
            # Score based on how well the score aligns with the label
            if expected:  # True OOD
                score = float(actual)
            else:  # Not OOD
                score = 1.0 - float(actual)
        else:
            return False, 0.0

        success = predicted_ood == expected
        return success, score


class CalibrationBenchmark(Benchmark):
    """
    Benchmark for confidence calibration.

    Tests whether predicted confidence matches empirical accuracy.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        config = config or BenchmarkConfig(
            name="calibration",
            description="Confidence calibration benchmark",
            n_trials=100,
        )
        super().__init__(config)
        self.n_classes = 5
        self.feature_dim = 32

    @property
    def name(self) -> str:
        return "calibration"

    def generate_trial(self, trial_id: int) -> Tuple[CalibrationSample, int]:
        """Generate a calibration trial."""
        # Generate features clustered around class centroids
        true_class = self._rng.integers(0, self.n_classes)

        # Create class centroid
        centroid = np.zeros(self.feature_dim)
        centroid[true_class * 6 : (true_class + 1) * 6] = 1.0

        # Add noise
        features = centroid + self._rng.normal(0, 0.5, self.feature_dim)

        sample = CalibrationSample(
            features=features,
            true_class=true_class,
            n_classes=self.n_classes,
        )

        return sample, true_class

    def evaluate(self, expected: int, actual: Any) -> Tuple[bool, float]:
        """Evaluate calibration."""
        if actual is None:
            return False, 0.0

        # actual should be (predicted_class, confidence) or just predicted_class
        if isinstance(actual, tuple) and len(actual) == 2:
            predicted_class, confidence = actual
            confidence = float(confidence)
        elif isinstance(actual, (int, np.integer)):
            predicted_class = int(actual)
            confidence = 1.0  # Assume full confidence if not provided
        else:
            return False, 0.0

        # Success if correct prediction
        success = predicted_class == expected

        # Score: If correct, confidence should be high
        # If incorrect, confidence should be low
        if success:
            score = confidence
        else:
            score = 1.0 - confidence

        return success, score


class AdversarialBenchmark(Benchmark):
    """
    Benchmark for adversarial robustness.

    Tests agent's ability to correctly classify adversarially
    perturbed inputs.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        config = config or BenchmarkConfig(
            name="adversarial",
            description="Adversarial robustness benchmark",
            n_trials=100,
        )
        super().__init__(config)
        self.n_classes = 10
        self.feature_dim = 64
        self.epsilon = 0.3  # Default perturbation budget

    @property
    def name(self) -> str:
        return "adversarial"

    def generate_trial(self, trial_id: int) -> Tuple[AdversarialSample, int]:
        """Generate an adversarial trial."""
        # Generate clean sample
        true_class = self._rng.integers(0, self.n_classes)

        # Create clean input (one-hot encoded with some structure)
        original = np.zeros(self.feature_dim)
        class_start = true_class * 6
        original[class_start : class_start + 6] = self._rng.uniform(0.5, 1.0, 6)
        original += self._rng.normal(0, 0.1, self.feature_dim)

        # Vary epsilon based on trial difficulty
        epsilon = 0.1 + 0.3 * (trial_id % 10) / 10

        # Generate adversarial perturbation (simulated)
        # In practice, this would use FGSM/PGD
        perturbation = np.zeros(self.feature_dim)

        # Attack: reduce signal for true class, increase for wrong class
        wrong_class = (true_class + 1) % self.n_classes
        wrong_start = wrong_class * 6

        perturbation[class_start : class_start + 6] = -epsilon
        perturbation[wrong_start : wrong_start + 6] = epsilon

        # Clip to epsilon ball
        perturbation = np.clip(perturbation, -epsilon, epsilon)

        sample = AdversarialSample(
            original=original,
            perturbation=perturbation,
            true_class=true_class,
            epsilon=epsilon,
        )

        return sample, true_class

    def evaluate(self, expected: int, actual: Any) -> Tuple[bool, float]:
        """Evaluate adversarial robustness."""
        if actual is None:
            return False, 0.0

        try:
            predicted = int(actual)
        except (ValueError, TypeError):
            return False, 0.0

        success = predicted == expected
        score = 1.0 if success else 0.0

        return success, score


class UncertaintyBenchmark(Benchmark):
    """
    Benchmark for uncertainty quantification quality.

    Tests whether uncertainty estimates are meaningful
    (high for ambiguous inputs, low for clear inputs).
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        config = config or BenchmarkConfig(
            name="uncertainty",
            description="Uncertainty quantification benchmark",
            n_trials=100,
        )
        super().__init__(config)
        self.feature_dim = 32
        self.n_classes = 5

    @property
    def name(self) -> str:
        return "uncertainty"

    def generate_trial(self, trial_id: int) -> Tuple[Dict, float]:
        """Generate an uncertainty trial."""
        # Decide difficulty level
        difficulty = ["easy", "medium", "hard", "ambiguous"][trial_id % 4]

        true_class = self._rng.integers(0, self.n_classes)
        centroid = np.zeros(self.feature_dim)
        centroid[true_class * 6 : (true_class + 1) * 6] = 1.0

        if difficulty == "easy":
            # Clear signal, low expected uncertainty
            noise_scale = 0.1
            expected_uncertainty = 0.1
        elif difficulty == "medium":
            noise_scale = 0.5
            expected_uncertainty = 0.3
        elif difficulty == "hard":
            noise_scale = 1.0
            expected_uncertainty = 0.5
        else:  # ambiguous
            # Mix two classes
            other_class = (true_class + 1) % self.n_classes
            other_centroid = np.zeros(self.feature_dim)
            other_centroid[other_class * 6 : (other_class + 1) * 6] = 1.0
            centroid = 0.5 * centroid + 0.5 * other_centroid
            noise_scale = 0.3
            expected_uncertainty = 0.8

        features = centroid + self._rng.normal(0, noise_scale, self.feature_dim)

        task = {
            "features": features,
            "difficulty": difficulty,
            "true_class": true_class,
            "n_classes": self.n_classes,
        }

        return task, expected_uncertainty

    def evaluate(self, expected: float, actual: Any) -> Tuple[bool, float]:
        """Evaluate uncertainty estimate."""
        if actual is None:
            return False, 0.0

        try:
            uncertainty = float(actual)
        except (ValueError, TypeError):
            return False, 0.0

        # Uncertainty should be in [0, 1]
        if not 0 <= uncertainty <= 1:
            return False, 0.0

        # Score based on how close to expected uncertainty
        error = abs(uncertainty - expected)
        score = max(0.0, 1.0 - 2 * error)  # 0.5 error = 0 score

        # Success if within 0.2 of expected
        success = error < 0.2

        return success, score


class SelectivePredictionBenchmark(Benchmark):
    """
    Benchmark for selective prediction (abstention).

    Tests agent's ability to abstain on difficult inputs
    while maintaining high accuracy on accepted inputs.
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        config = config or BenchmarkConfig(
            name="selective_prediction",
            description="Selective prediction benchmark",
            n_trials=100,
        )
        super().__init__(config)
        self.feature_dim = 32
        self.n_classes = 5

    @property
    def name(self) -> str:
        return "selective_prediction"

    def generate_trial(self, trial_id: int) -> Tuple[Dict, Dict]:
        """Generate a selective prediction trial."""
        # Create dataset with varying difficulty
        true_class = self._rng.integers(0, self.n_classes)

        # Vary difficulty
        if trial_id % 3 == 0:
            noise = 0.2  # Easy
            should_abstain = False
        elif trial_id % 3 == 1:
            noise = 0.8  # Medium
            should_abstain = False
        else:
            noise = 2.0  # Hard - should abstain
            should_abstain = True

        centroid = np.zeros(self.feature_dim)
        centroid[true_class * 6 : (true_class + 1) * 6] = 1.0
        features = centroid + self._rng.normal(0, noise, self.feature_dim)

        task = {
            "features": features,
            "n_classes": self.n_classes,
        }

        expected = {
            "true_class": true_class,
            "should_abstain": should_abstain,
        }

        return task, expected

    def evaluate(self, expected: Dict, actual: Any) -> Tuple[bool, float]:
        """Evaluate selective prediction."""
        if actual is None:
            return False, 0.0

        # actual should be (prediction, abstain_flag) or just prediction
        if isinstance(actual, tuple) and len(actual) == 2:
            prediction, abstained = actual
            abstained = bool(abstained)
        elif isinstance(actual, str) and actual.lower() == "abstain":
            prediction = None
            abstained = True
        elif isinstance(actual, (int, np.integer)):
            prediction = int(actual)
            abstained = False
        else:
            return False, 0.0

        true_class = expected["true_class"]
        should_abstain = expected["should_abstain"]

        if abstained:
            # Abstained - good if should abstain, bad otherwise
            if should_abstain:
                return True, 1.0  # Correctly abstained
            else:
                return False, 0.5  # Unnecessary abstention (partial credit)
        else:
            # Made prediction
            correct = prediction == true_class
            if should_abstain:
                # Should have abstained
                if correct:
                    return True, 0.8  # Correct but risky
                else:
                    return False, 0.0  # Should have abstained
            else:
                # Shouldn't abstain
                return correct, 1.0 if correct else 0.0


def create_robustness_benchmark_suite() -> List[Benchmark]:
    """Create all robustness benchmarks."""
    return [
        OODDetectionBenchmark(),
        CalibrationBenchmark(),
        AdversarialBenchmark(),
        UncertaintyBenchmark(),
        SelectivePredictionBenchmark(),
    ]


__all__ = [
    "OODSample",
    "CalibrationSample",
    "AdversarialSample",
    "OODDetectionBenchmark",
    "CalibrationBenchmark",
    "AdversarialBenchmark",
    "UncertaintyBenchmark",
    "SelectivePredictionBenchmark",
    "create_robustness_benchmark_suite",
]
