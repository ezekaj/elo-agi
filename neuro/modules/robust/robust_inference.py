"""
Robust Inference Under Uncertainty.

Implements:
- Inference with uncertainty estimates
- Selective prediction (abstain when uncertain)
- Robust aggregation of multiple predictions
- Rejection based on confidence thresholds
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import numpy as np

from .uncertainty import UncertaintyQuantifier, UncertaintyEstimate, SimpleDropoutNN
from .ood_detection import OODDetector, OODResult
from .calibration import ConfidenceCalibrator, CalibrationResult


class RejectionPolicy(Enum):
    """Policies for rejecting uncertain predictions."""
    CONFIDENCE = "confidence"        # Reject low confidence
    UNCERTAINTY = "uncertainty"      # Reject high uncertainty
    OOD = "ood"                      # Reject OOD inputs
    COMBINED = "combined"            # Combine all criteria


@dataclass
class RobustPrediction:
    """Result of robust inference."""
    prediction: int
    probabilities: np.ndarray
    confidence: float
    uncertainty: float
    is_rejected: bool
    rejection_reason: Optional[str]
    is_ood: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectivePrediction:
    """Result of selective prediction."""
    prediction: Optional[int]       # None if abstained
    confidence: float
    abstained: bool
    abstention_reason: Optional[str]
    coverage: float                  # Fraction of predictions made
    risk: float                      # Error rate on predictions made


class RobustInference:
    """
    Robust inference system combining multiple robustness components.

    Integrates:
    - Uncertainty quantification
    - OOD detection
    - Confidence calibration
    - Selective prediction
    """

    def __init__(
        self,
        model: SimpleDropoutNN,
        uncertainty_quantifier: Optional[UncertaintyQuantifier] = None,
        ood_detector: Optional[OODDetector] = None,
        calibrator: Optional[ConfidenceCalibrator] = None,
        rejection_policy: RejectionPolicy = RejectionPolicy.COMBINED,
        confidence_threshold: float = 0.7,
        uncertainty_threshold: float = 0.5,
        ood_threshold: float = 0.5,
    ):
        self.model = model
        self.rejection_policy = rejection_policy
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.ood_threshold = ood_threshold

        # Components (create defaults if not provided)
        self.uncertainty = uncertainty_quantifier or UncertaintyQuantifier(model)
        self.ood_detector = ood_detector
        self.calibrator = calibrator

        # Statistics
        self._n_predictions = 0
        self._n_rejected = 0
        self._n_ood = 0

    def infer(
        self,
        x: np.ndarray,
        return_uncertainty: bool = True,
        check_ood: bool = True,
    ) -> RobustPrediction:
        """
        Make robust prediction with uncertainty and rejection.

        Args:
            x: Input
            return_uncertainty: Whether to compute uncertainty
            check_ood: Whether to check for OOD

        Returns:
            RobustPrediction with all information
        """
        self._n_predictions += 1

        # Get base prediction
        probs = self.model.predict_proba(x, apply_dropout=False)

        # Calibrate if available
        if self.calibrator is not None:
            logits = self.model.forward(x, apply_dropout=False)
            result = self.calibrator.calibrate(logits)
            probs = result.calibrated_probs

        prediction = int(np.argmax(probs))
        confidence = float(np.max(probs))

        # Get uncertainty
        uncertainty = 0.0
        if return_uncertainty:
            estimate = self.uncertainty.monte_carlo_dropout(x)
            uncertainty = estimate.total

        # Check OOD
        is_ood = False
        if check_ood and self.ood_detector is not None:
            ood_result = self.ood_detector.detect(x)
            is_ood = ood_result.is_ood
            if is_ood:
                self._n_ood += 1

        # Determine rejection
        is_rejected, rejection_reason = self._should_reject(
            confidence, uncertainty, is_ood
        )

        if is_rejected:
            self._n_rejected += 1

        return RobustPrediction(
            prediction=prediction,
            probabilities=probs,
            confidence=confidence,
            uncertainty=uncertainty,
            is_rejected=is_rejected,
            rejection_reason=rejection_reason,
            is_ood=is_ood,
            details={
                "policy": self.rejection_policy.value,
                "confidence_threshold": self.confidence_threshold,
                "uncertainty_threshold": self.uncertainty_threshold,
            },
        )

    def _should_reject(
        self,
        confidence: float,
        uncertainty: float,
        is_ood: bool,
    ) -> Tuple[bool, Optional[str]]:
        """Determine if prediction should be rejected."""
        if self.rejection_policy == RejectionPolicy.CONFIDENCE:
            if confidence < self.confidence_threshold:
                return True, f"Low confidence: {confidence:.3f} < {self.confidence_threshold}"
            return False, None

        elif self.rejection_policy == RejectionPolicy.UNCERTAINTY:
            if uncertainty > self.uncertainty_threshold:
                return True, f"High uncertainty: {uncertainty:.3f} > {self.uncertainty_threshold}"
            return False, None

        elif self.rejection_policy == RejectionPolicy.OOD:
            if is_ood:
                return True, "Out-of-distribution input detected"
            return False, None

        elif self.rejection_policy == RejectionPolicy.COMBINED:
            reasons = []
            if confidence < self.confidence_threshold:
                reasons.append(f"low confidence ({confidence:.3f})")
            if uncertainty > self.uncertainty_threshold:
                reasons.append(f"high uncertainty ({uncertainty:.3f})")
            if is_ood:
                reasons.append("OOD")

            if reasons:
                return True, "Rejected due to: " + ", ".join(reasons)
            return False, None

        return False, None

    def infer_with_uncertainty(
        self,
        x: np.ndarray,
    ) -> Tuple[int, float]:
        """
        Simple interface returning (prediction, uncertainty).

        Args:
            x: Input

        Returns:
            (predicted_class, uncertainty_score)
        """
        result = self.infer(x, return_uncertainty=True, check_ood=False)
        return result.prediction, result.uncertainty

    def reject_uncertain(
        self,
        x: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Tuple[Optional[int], bool]:
        """
        Return prediction or None if too uncertain.

        Args:
            x: Input
            threshold: Uncertainty threshold for rejection

        Returns:
            (prediction or None, was_rejected)
        """
        threshold = threshold or self.uncertainty_threshold
        result = self.infer(x)

        if result.uncertainty > threshold:
            return None, True
        return result.prediction, False

    def selective_predict(
        self,
        x: np.ndarray,
        target_coverage: Optional[float] = None,
    ) -> SelectivePrediction:
        """
        Make selective prediction (abstain when uncertain).

        Args:
            x: Input
            target_coverage: Target fraction of inputs to predict on

        Returns:
            SelectivePrediction result
        """
        result = self.infer(x)

        abstained = result.is_rejected
        prediction = None if abstained else result.prediction

        # Coverage is tracked over history
        coverage = 1.0 - (self._n_rejected / max(self._n_predictions, 1))

        return SelectivePrediction(
            prediction=prediction,
            confidence=result.confidence,
            abstained=abstained,
            abstention_reason=result.rejection_reason,
            coverage=coverage,
            risk=0.0,  # Would need ground truth to compute
        )

    def robust_aggregation(
        self,
        predictions: List[Tuple[int, float]],
        method: str = "weighted_vote",
    ) -> Tuple[int, float]:
        """
        Aggregate multiple predictions robustly.

        Args:
            predictions: List of (prediction, confidence) pairs
            method: Aggregation method

        Returns:
            (aggregated_prediction, aggregated_confidence)
        """
        if not predictions:
            return 0, 0.0

        if method == "majority_vote":
            votes = [p for p, _ in predictions]
            unique, counts = np.unique(votes, return_counts=True)
            winner = unique[np.argmax(counts)]
            confidence = np.max(counts) / len(predictions)
            return int(winner), float(confidence)

        elif method == "weighted_vote":
            # Weight by confidence
            n_classes = max(p for p, _ in predictions) + 1
            class_weights = np.zeros(n_classes)

            for pred, conf in predictions:
                if pred < n_classes:
                    class_weights[pred] += conf

            winner = int(np.argmax(class_weights))
            confidence = class_weights[winner] / (np.sum(class_weights) + 1e-8)
            return winner, float(confidence)

        elif method == "confidence_max":
            # Return prediction with highest confidence
            best_pred, best_conf = max(predictions, key=lambda x: x[1])
            return best_pred, best_conf

        elif method == "median":
            # Median confidence for winner
            from collections import Counter
            votes = [p for p, _ in predictions]
            winner = Counter(votes).most_common(1)[0][0]
            winner_confs = [c for p, c in predictions if p == winner]
            confidence = float(np.median(winner_confs))
            return winner, confidence

        else:
            return self.robust_aggregation(predictions, "weighted_vote")

    def batch_infer(
        self,
        batch: List[np.ndarray],
        return_rejected: bool = True,
    ) -> List[RobustPrediction]:
        """
        Make robust predictions on a batch.

        Args:
            batch: List of inputs
            return_rejected: Whether to include rejected predictions

        Returns:
            List of RobustPrediction objects
        """
        results = []
        for x in batch:
            result = self.infer(x)
            if return_rejected or not result.is_rejected:
                results.append(result)
        return results

    def calibrate_thresholds(
        self,
        validation_data: List[Tuple[np.ndarray, int]],
        target_coverage: float = 0.9,
        target_accuracy: float = 0.95,
    ) -> Dict[str, float]:
        """
        Calibrate rejection thresholds on validation data.

        Args:
            validation_data: (input, label) pairs
            target_coverage: Target fraction of inputs to not reject
            target_accuracy: Target accuracy on non-rejected inputs

        Returns:
            Calibrated thresholds
        """
        # Collect statistics
        confidences = []
        uncertainties = []
        correct = []

        for x, y in validation_data:
            result = self.infer(x)
            confidences.append(result.confidence)
            uncertainties.append(result.uncertainty)
            correct.append(result.prediction == y)

        confidences = np.array(confidences)
        uncertainties = np.array(uncertainties)
        correct = np.array(correct)

        # Find confidence threshold for target accuracy
        # Higher threshold = fewer predictions = higher accuracy
        sorted_conf_idx = np.argsort(confidences)[::-1]
        cumulative_accuracy = np.cumsum(correct[sorted_conf_idx]) / (np.arange(len(correct)) + 1)

        # Find threshold achieving target accuracy with max coverage
        meeting_accuracy = cumulative_accuracy >= target_accuracy
        if np.any(meeting_accuracy):
            best_idx = np.max(np.where(meeting_accuracy)[0])
            self.confidence_threshold = float(confidences[sorted_conf_idx[best_idx]])
        else:
            self.confidence_threshold = float(np.max(confidences))

        # Find uncertainty threshold for target coverage
        self.uncertainty_threshold = float(np.percentile(uncertainties, 100 * target_coverage))

        return {
            "confidence_threshold": self.confidence_threshold,
            "uncertainty_threshold": self.uncertainty_threshold,
            "achieved_coverage": 1.0 - self._n_rejected / len(validation_data),
        }

    def evaluate(
        self,
        test_data: List[Tuple[np.ndarray, int]],
    ) -> Dict[str, float]:
        """
        Evaluate robust inference system.

        Args:
            test_data: (input, label) pairs

        Returns:
            Evaluation metrics
        """
        predictions = []
        confidences = []
        rejected = []
        correct = []
        ood = []

        for x, y in test_data:
            result = self.infer(x)
            predictions.append(result.prediction)
            confidences.append(result.confidence)
            rejected.append(result.is_rejected)
            correct.append(result.prediction == y)
            ood.append(result.is_ood)

        predictions = np.array(predictions)
        rejected = np.array(rejected)
        correct = np.array(correct)

        # Coverage: fraction not rejected
        coverage = 1.0 - np.mean(rejected)

        # Accuracy on non-rejected
        non_rejected_mask = ~rejected
        if np.sum(non_rejected_mask) > 0:
            selective_accuracy = np.mean(correct[non_rejected_mask])
        else:
            selective_accuracy = 0.0

        # Overall accuracy
        overall_accuracy = np.mean(correct)

        # Risk (error rate on non-rejected)
        selective_risk = 1.0 - selective_accuracy

        return {
            "coverage": float(coverage),
            "selective_accuracy": float(selective_accuracy),
            "overall_accuracy": float(overall_accuracy),
            "selective_risk": float(selective_risk),
            "rejection_rate": float(1.0 - coverage),
            "ood_rate": float(np.mean(ood)),
            "mean_confidence": float(np.mean(confidences)),
        }

    def rejection_rate(self) -> float:
        """Get current rejection rate."""
        if self._n_predictions == 0:
            return 0.0
        return self._n_rejected / self._n_predictions

    def statistics(self) -> Dict[str, Any]:
        """Get inference statistics."""
        return {
            "n_predictions": self._n_predictions,
            "n_rejected": self._n_rejected,
            "n_ood": self._n_ood,
            "rejection_rate": self.rejection_rate(),
            "policy": self.rejection_policy.value,
            "confidence_threshold": self.confidence_threshold,
            "uncertainty_threshold": self.uncertainty_threshold,
            "ood_threshold": self.ood_threshold,
        }


class AdaptiveThresholdInference(RobustInference):
    """
    Robust inference with adaptive thresholds.

    Adjusts thresholds based on recent performance.
    """

    def __init__(
        self,
        model: SimpleDropoutNN,
        window_size: int = 100,
        target_coverage: float = 0.9,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.window_size = window_size
        self.target_coverage = target_coverage

        # History for adaptation
        self._recent_confidences: List[float] = []
        self._recent_uncertainties: List[float] = []
        self._recent_rejections: List[bool] = []

    def infer(
        self,
        x: np.ndarray,
        **kwargs,
    ) -> RobustPrediction:
        """Make prediction with adaptive thresholds."""
        result = super().infer(x, **kwargs)

        # Update history
        self._recent_confidences.append(result.confidence)
        self._recent_uncertainties.append(result.uncertainty)
        self._recent_rejections.append(result.is_rejected)

        # Trim to window size
        if len(self._recent_confidences) > self.window_size:
            self._recent_confidences = self._recent_confidences[-self.window_size:]
            self._recent_uncertainties = self._recent_uncertainties[-self.window_size:]
            self._recent_rejections = self._recent_rejections[-self.window_size:]

        # Adapt thresholds
        self._adapt_thresholds()

        return result

    def _adapt_thresholds(self) -> None:
        """Adapt thresholds based on recent history."""
        if len(self._recent_rejections) < 10:
            return

        current_coverage = 1.0 - np.mean(self._recent_rejections)

        if current_coverage < self.target_coverage - 0.05:
            # Too many rejections, relax thresholds
            self.confidence_threshold *= 0.98
            self.uncertainty_threshold *= 1.02
        elif current_coverage > self.target_coverage + 0.05:
            # Too few rejections, tighten thresholds
            self.confidence_threshold *= 1.02
            self.uncertainty_threshold *= 0.98

        # Clamp thresholds to reasonable ranges
        self.confidence_threshold = np.clip(self.confidence_threshold, 0.3, 0.99)
        self.uncertainty_threshold = np.clip(self.uncertainty_threshold, 0.1, 2.0)
