"""
Confidence Calibration.

Implements:
- Temperature scaling
- Platt scaling
- Isotonic regression
- Expected Calibration Error (ECE)
- Reliability diagrams
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import numpy as np


class CalibrationMethod(Enum):
    """Calibration methods."""

    TEMPERATURE = "temperature"
    PLATT = "platt"
    ISOTONIC = "isotonic"
    HISTOGRAM = "histogram"
    BETA = "beta"


@dataclass
class CalibrationMetrics:
    """Calibration evaluation metrics."""

    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier_score: float  # Brier score
    reliability_diagram: Dict[str, np.ndarray]
    n_samples: int


@dataclass
class CalibrationResult:
    """Result of calibration."""

    calibrated_probs: np.ndarray
    method: CalibrationMethod
    parameters: Dict[str, Any]


class ConfidenceCalibrator:
    """
    Calibrate neural network confidence scores.

    Ensures predicted probabilities match empirical frequencies.
    """

    def __init__(
        self,
        method: CalibrationMethod = CalibrationMethod.TEMPERATURE,
        n_bins: int = 15,
    ):
        self.method = method
        self.n_bins = n_bins

        # Calibration parameters
        self.temperature: float = 1.0
        self.platt_a: float = 1.0
        self.platt_b: float = 0.0
        self._isotonic_calibrator: Optional[Dict] = None
        self._histogram_calibrator: Optional[Dict] = None

        # Statistics
        self._is_fitted = False
        self._n_calibrations = 0

    def temperature_scaling(
        self,
        logits: np.ndarray,
        T: Optional[float] = None,
    ) -> np.ndarray:
        """
        Apply temperature scaling.

        softmax(logits / T)
        Higher T = softer (more uncertain) predictions.

        Args:
            logits: Raw model outputs
            T: Temperature parameter

        Returns:
            Calibrated probabilities
        """
        T = T or self.temperature
        scaled_logits = logits / T
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        return exp_logits / np.sum(exp_logits)

    def platt_scaling(
        self,
        logits: np.ndarray,
        a: Optional[float] = None,
        b: Optional[float] = None,
    ) -> np.ndarray:
        """
        Apply Platt scaling.

        P = sigmoid(a * logit + b)

        Originally for binary classification, extended to multiclass
        via one-vs-all.

        Args:
            logits: Raw model outputs
            a: Scale parameter
            b: Bias parameter

        Returns:
            Calibrated probabilities
        """
        a = a or self.platt_a
        b = b or self.platt_b

        # Apply to each logit
        scaled = a * logits + b
        exp_scaled = np.exp(scaled - np.max(scaled))
        return exp_scaled / np.sum(exp_scaled)

    def isotonic_regression_calibrate(
        self,
        probs: np.ndarray,
    ) -> np.ndarray:
        """
        Apply isotonic regression calibration.

        Maps probabilities through a monotonic function learned from data.

        Args:
            probs: Uncalibrated probabilities

        Returns:
            Calibrated probabilities
        """
        if self._isotonic_calibrator is None:
            return probs

        calibrated = np.zeros_like(probs)
        for i in range(len(probs)):
            # Find nearest calibration point
            calibrated[i] = self._apply_isotonic(probs[i], i)

        # Renormalize
        calibrated = calibrated / np.sum(calibrated)
        return calibrated

    def _apply_isotonic(self, prob: float, class_idx: int) -> float:
        """Apply isotonic regression for a single probability."""
        if self._isotonic_calibrator is None or class_idx >= len(
            self._isotonic_calibrator["mappings"]
        ):
            return prob

        mapping = self._isotonic_calibrator["mappings"][class_idx]
        x_vals = mapping["x"]
        y_vals = mapping["y"]

        # Linear interpolation
        idx = np.searchsorted(x_vals, prob)
        if idx == 0:
            return y_vals[0]
        elif idx >= len(x_vals):
            return y_vals[-1]
        else:
            # Interpolate
            t = (prob - x_vals[idx - 1]) / (x_vals[idx] - x_vals[idx - 1] + 1e-8)
            return y_vals[idx - 1] + t * (y_vals[idx] - y_vals[idx - 1])

    def histogram_binning(
        self,
        probs: np.ndarray,
    ) -> np.ndarray:
        """
        Apply histogram binning calibration.

        Maps each probability to the average accuracy in its bin.
        """
        if self._histogram_calibrator is None:
            return probs

        calibrated = np.zeros_like(probs)
        for i in range(len(probs)):
            bin_idx = min(int(probs[i] * self.n_bins), self.n_bins - 1)
            if i < len(self._histogram_calibrator):
                calibrated[i] = self._histogram_calibrator[i][bin_idx]
            else:
                calibrated[i] = probs[i]

        # Renormalize
        if np.sum(calibrated) > 0:
            calibrated = calibrated / np.sum(calibrated)
        return calibrated

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        validation_split: float = 0.2,
    ) -> Dict[str, float]:
        """
        Fit calibration parameters.

        Args:
            logits: Model logits (n_samples, n_classes)
            labels: True labels (n_samples,)
            validation_split: Fraction for validation

        Returns:
            Dictionary of calibration metrics
        """
        n_samples = len(logits)
        n_val = int(n_samples * validation_split)

        # Split data
        indices = np.random.permutation(n_samples)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        train_logits = logits[train_idx]
        train_labels = labels[train_idx]
        val_logits = logits[val_idx]
        val_labels = labels[val_idx]

        if self.method == CalibrationMethod.TEMPERATURE:
            self._fit_temperature(train_logits, train_labels)
        elif self.method == CalibrationMethod.PLATT:
            self._fit_platt(train_logits, train_labels)
        elif self.method == CalibrationMethod.ISOTONIC:
            self._fit_isotonic(train_logits, train_labels)
        elif self.method == CalibrationMethod.HISTOGRAM:
            self._fit_histogram(train_logits, train_labels)

        self._is_fitted = True

        # Evaluate on validation
        val_probs = self.calibrate_batch(val_logits)
        metrics = self.compute_calibration_metrics(val_probs, val_labels)

        return {
            "ece": metrics.ece,
            "mce": metrics.mce,
            "brier_score": metrics.brier_score,
        }

    def _fit_temperature(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Fit temperature parameter."""
        # Grid search for optimal temperature
        best_ece = float("inf")
        best_temp = 1.0

        for temp in np.linspace(0.5, 5.0, 50):
            probs = np.array([self.temperature_scaling(l, temp) for l in logits])
            ece = self._compute_ece(probs, labels)

            if ece < best_ece:
                best_ece = ece
                best_temp = temp

        self.temperature = best_temp

    def _fit_platt(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Fit Platt scaling parameters."""
        # Simplified fitting using grid search
        best_ece = float("inf")
        best_a = 1.0
        best_b = 0.0

        for a in np.linspace(0.5, 2.0, 20):
            for b in np.linspace(-1.0, 1.0, 20):
                probs = np.array([self.platt_scaling(l, a, b) for l in logits])
                ece = self._compute_ece(probs, labels)

                if ece < best_ece:
                    best_ece = ece
                    best_a = a
                    best_b = b

        self.platt_a = best_a
        self.platt_b = best_b

    def _fit_isotonic(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Fit isotonic regression."""
        n_classes = logits.shape[1]
        probs = np.array([np.exp(l - np.max(l)) / np.sum(np.exp(l - np.max(l))) for l in logits])

        self._isotonic_calibrator = {"mappings": []}

        for c in range(n_classes):
            class_probs = probs[:, c]
            class_labels = (labels == c).astype(float)

            # Sort by probability
            sorted_idx = np.argsort(class_probs)
            sorted_probs = class_probs[sorted_idx]
            sorted_labels = class_labels[sorted_idx]

            # Pool Adjacent Violators algorithm (simplified)
            x_vals = []
            y_vals = []

            # Bin and average
            n_bins = 20
            for i in range(n_bins):
                start = i * len(sorted_probs) // n_bins
                end = (i + 1) * len(sorted_probs) // n_bins
                if end > start:
                    x_vals.append(np.mean(sorted_probs[start:end]))
                    y_vals.append(np.mean(sorted_labels[start:end]))

            self._isotonic_calibrator["mappings"].append(
                {
                    "x": np.array(x_vals),
                    "y": np.array(y_vals),
                }
            )

    def _fit_histogram(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Fit histogram binning."""
        n_classes = logits.shape[1]
        probs = np.array([np.exp(l - np.max(l)) / np.sum(np.exp(l - np.max(l))) for l in logits])

        self._histogram_calibrator = []

        for c in range(n_classes):
            class_probs = probs[:, c]
            class_labels = (labels == c).astype(float)

            bin_calibration = np.zeros(self.n_bins)
            for b in range(self.n_bins):
                bin_lower = b / self.n_bins
                bin_upper = (b + 1) / self.n_bins
                mask = (class_probs >= bin_lower) & (class_probs < bin_upper)

                if np.sum(mask) > 0:
                    bin_calibration[b] = np.mean(class_labels[mask])
                else:
                    bin_calibration[b] = (bin_lower + bin_upper) / 2

            self._histogram_calibrator.append(bin_calibration)

    def calibrate(
        self,
        logits: np.ndarray,
    ) -> CalibrationResult:
        """
        Calibrate model output.

        Args:
            logits: Raw model outputs

        Returns:
            CalibrationResult with calibrated probabilities
        """
        self._n_calibrations += 1

        if self.method == CalibrationMethod.TEMPERATURE:
            probs = self.temperature_scaling(logits)
            params = {"temperature": self.temperature}
        elif self.method == CalibrationMethod.PLATT:
            probs = self.platt_scaling(logits)
            params = {"a": self.platt_a, "b": self.platt_b}
        elif self.method == CalibrationMethod.ISOTONIC:
            # First get uncalibrated probs
            exp_logits = np.exp(logits - np.max(logits))
            uncal_probs = exp_logits / np.sum(exp_logits)
            probs = self.isotonic_regression_calibrate(uncal_probs)
            params = {"method": "isotonic"}
        elif self.method == CalibrationMethod.HISTOGRAM:
            exp_logits = np.exp(logits - np.max(logits))
            uncal_probs = exp_logits / np.sum(exp_logits)
            probs = self.histogram_binning(uncal_probs)
            params = {"n_bins": self.n_bins}
        else:
            probs = self.temperature_scaling(logits)
            params = {"temperature": self.temperature}

        return CalibrationResult(
            calibrated_probs=probs,
            method=self.method,
            parameters=params,
        )

    def calibrate_batch(
        self,
        logits: np.ndarray,
    ) -> np.ndarray:
        """Calibrate batch of logits."""
        return np.array([self.calibrate(l).calibrated_probs for l in logits])

    def expected_calibration_error(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        Compute Expected Calibration Error.

        ECE = sum_b (n_b / N) * |acc_b - conf_b|

        Args:
            probs: Predicted probabilities (n_samples, n_classes)
            labels: True labels (n_samples,)

        Returns:
            ECE value
        """
        return self._compute_ece(probs, labels)

    def _compute_ece(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Compute ECE."""
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)

        ece = 0.0
        for b in range(self.n_bins):
            bin_lower = b / self.n_bins
            bin_upper = (b + 1) / self.n_bins
            mask = (confidences > bin_lower) & (confidences <= bin_upper)

            if np.sum(mask) > 0:
                bin_accuracy = np.mean(accuracies[mask])
                bin_confidence = np.mean(confidences[mask])
                bin_weight = np.sum(mask) / len(labels)
                ece += bin_weight * abs(bin_accuracy - bin_confidence)

        return float(ece)

    def maximum_calibration_error(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Compute Maximum Calibration Error."""
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)

        mce = 0.0
        for b in range(self.n_bins):
            bin_lower = b / self.n_bins
            bin_upper = (b + 1) / self.n_bins
            mask = (confidences > bin_lower) & (confidences <= bin_upper)

            if np.sum(mask) > 0:
                bin_accuracy = np.mean(accuracies[mask])
                bin_confidence = np.mean(confidences[mask])
                mce = max(mce, abs(bin_accuracy - bin_confidence))

        return float(mce)

    def brier_score(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        Compute Brier score.

        BS = (1/N) * sum_i sum_c (p_ic - y_ic)^2
        """
        n_samples = len(labels)
        n_classes = probs.shape[1]

        one_hot = np.zeros((n_samples, n_classes))
        one_hot[np.arange(n_samples), labels] = 1

        return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    def reliability_diagram(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute reliability diagram data.

        Returns bin edges, accuracies, and confidences for plotting.
        """
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)

        bin_edges = np.linspace(0, 1, self.n_bins + 1)
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for b in range(self.n_bins):
            mask = (confidences > bin_edges[b]) & (confidences <= bin_edges[b + 1])
            count = np.sum(mask)
            bin_counts.append(count)

            if count > 0:
                bin_accuracies.append(np.mean(accuracies[mask]))
                bin_confidences.append(np.mean(confidences[mask]))
            else:
                bin_accuracies.append(0)
                bin_confidences.append((bin_edges[b] + bin_edges[b + 1]) / 2)

        return {
            "bin_edges": bin_edges,
            "accuracies": np.array(bin_accuracies),
            "confidences": np.array(bin_confidences),
            "counts": np.array(bin_counts),
        }

    def compute_calibration_metrics(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> CalibrationMetrics:
        """Compute all calibration metrics."""
        return CalibrationMetrics(
            ece=self._compute_ece(probs, labels),
            mce=self.maximum_calibration_error(probs, labels),
            brier_score=self.brier_score(probs, labels),
            reliability_diagram=self.reliability_diagram(probs, labels),
            n_samples=len(labels),
        )

    def statistics(self) -> Dict[str, Any]:
        """Get calibrator statistics."""
        return {
            "method": self.method.value,
            "is_fitted": self._is_fitted,
            "n_calibrations": self._n_calibrations,
            "temperature": self.temperature,
            "platt_a": self.platt_a,
            "platt_b": self.platt_b,
            "n_bins": self.n_bins,
        }
