"""
Integration with neuro-integrate SharedSpace.

Provides robustness-aware embeddings and uncertainty filtering
for the shared semantic space.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import numpy as np

from .uncertainty import (
    SimpleDropoutNN, UncertaintyQuantifier, UncertaintyEstimate,
)
from .ood_detection import OODDetector, OODMethod, OODResult
from .calibration import ConfidenceCalibrator, CalibrationMethod
from .adversarial import AdversarialDefense, DefenseType


class RobustnessLevel(Enum):
    """Levels of robustness checking."""
    NONE = "none"
    BASIC = "basic"           # Just uncertainty
    STANDARD = "standard"     # Uncertainty + OOD
    FULL = "full"             # All checks including adversarial


@dataclass
class RobustnessMetadata:
    """Robustness metadata for embeddings."""
    uncertainty: float = 0.0
    epistemic: float = 0.0
    aleatoric: float = 0.0
    ood_score: float = 0.0
    is_ood: bool = False
    calibrated_confidence: float = 1.0
    is_calibrated: bool = False
    adversarial_score: float = 0.0
    is_adversarial: bool = False
    robustness_level: RobustnessLevel = RobustnessLevel.NONE


@dataclass
class RobustEmbedding:
    """
    Semantic embedding with robustness metadata.

    Extends the concept of SemanticEmbedding to include
    uncertainty quantification, OOD detection, and calibration.
    """
    vector: np.ndarray
    modality: str
    source_module: str
    confidence: float = 1.0
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    robustness: RobustnessMetadata = field(default_factory=RobustnessMetadata)

    def similarity(self, other: 'RobustEmbedding') -> float:
        """Compute cosine similarity with another embedding."""
        norm_self = np.linalg.norm(self.vector)
        norm_other = np.linalg.norm(other.vector)
        if norm_self < 1e-8 or norm_other < 1e-8:
            return 0.0
        return float(np.dot(self.vector, other.vector) / (norm_self * norm_other))

    def effective_confidence(self) -> float:
        """Get confidence adjusted for robustness."""
        if self.robustness.robustness_level == RobustnessLevel.NONE:
            return self.confidence

        # Reduce confidence based on uncertainty and OOD score
        adjusted = self.confidence
        adjusted *= (1.0 - self.robustness.uncertainty)
        adjusted *= (1.0 - self.robustness.ood_score)

        if self.robustness.is_calibrated:
            adjusted = self.robustness.calibrated_confidence

        return max(0.0, min(1.0, adjusted))

    def is_reliable(self, threshold: float = 0.5) -> bool:
        """Check if embedding is reliable based on robustness metrics."""
        if self.robustness.is_ood:
            return False
        if self.robustness.is_adversarial:
            return False
        return self.effective_confidence() >= threshold


class SharedSpaceRobustness:
    """
    Integrates robustness checking into SharedSpace operations.

    Provides:
    - Uncertainty quantification for embeddings
    - OOD detection for incoming data
    - Confidence calibration
    - Adversarial input detection
    - Filtering by reliability
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        n_classes: int = 10,
        uncertainty_samples: int = 30,
        ood_method: OODMethod = OODMethod.ENERGY,
        calibration_method: CalibrationMethod = CalibrationMethod.TEMPERATURE,
        robustness_level: RobustnessLevel = RobustnessLevel.STANDARD,
        random_seed: Optional[int] = None,
    ):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.robustness_level = robustness_level

        # Initialize robustness components
        self._uncertainty_model = SimpleDropoutNN(
            embedding_dim, hidden_dim, n_classes,
            dropout_rate=0.3, random_seed=random_seed,
        )
        self._uncertainty_quantifier = UncertaintyQuantifier(
            self._uncertainty_model, n_samples=uncertainty_samples,
        )

        self._ood_detector = OODDetector(
            model=self._uncertainty_model._to_classifier(),
            method=ood_method,
        )

        self._calibrator = ConfidenceCalibrator(method=calibration_method)

        self._defense = AdversarialDefense(
            model=self._uncertainty_model._to_simple_nn(),
            defense_type=DefenseType.DETECTION,
        )

        # Track statistics
        self._n_processed = 0
        self._n_rejected_ood = 0
        self._n_rejected_adversarial = 0
        self._n_low_confidence = 0

        # Reference embeddings for OOD detection
        self._reference_embeddings: List[np.ndarray] = []
        self._is_fitted = False

    def fit(
        self,
        reference_data: List[Tuple[np.ndarray, int]],
        calibration_logits: Optional[np.ndarray] = None,
        calibration_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Fit robustness components on reference data.

        Args:
            reference_data: List of (embedding, label) tuples for OOD fitting
            calibration_logits: Logits for calibration fitting
            calibration_labels: Labels for calibration fitting

        Returns:
            Fitting metrics
        """
        # Fit OOD detector
        self._ood_detector.fit(reference_data)

        # Fit calibrator if data provided
        if calibration_logits is not None and calibration_labels is not None:
            metrics = self._calibrator.fit(calibration_logits, calibration_labels)
        else:
            metrics = {}

        # Store reference embeddings
        self._reference_embeddings = [x for x, _ in reference_data]
        self._is_fitted = True

        return {
            "n_reference_samples": len(reference_data),
            "ood_fitted": True,
            "calibration_metrics": metrics,
        }

    def add_robustness(
        self,
        vector: np.ndarray,
        modality: str = "abstract",
        source_module: str = "unknown",
        base_confidence: float = 1.0,
    ) -> RobustEmbedding:
        """
        Add robustness metadata to an embedding vector.

        Args:
            vector: The embedding vector
            modality: Modality type string
            source_module: Source module name
            base_confidence: Initial confidence value

        Returns:
            RobustEmbedding with robustness metadata
        """
        self._n_processed += 1

        robustness = RobustnessMetadata(
            robustness_level=self.robustness_level,
        )

        # Compute uncertainty if enabled
        if self.robustness_level in [RobustnessLevel.BASIC,
                                      RobustnessLevel.STANDARD,
                                      RobustnessLevel.FULL]:
            estimate = self._uncertainty_quantifier.monte_carlo_dropout(vector)
            robustness.uncertainty = estimate.total
            robustness.epistemic = estimate.epistemic
            robustness.aleatoric = estimate.aleatoric

        # Compute OOD score if enabled
        if self.robustness_level in [RobustnessLevel.STANDARD,
                                      RobustnessLevel.FULL]:
            if self._is_fitted:
                ood_result = self._ood_detector.detect(vector)
                robustness.ood_score = ood_result.score
                robustness.is_ood = ood_result.is_ood
                if robustness.is_ood:
                    self._n_rejected_ood += 1

        # Check for adversarial if enabled
        if self.robustness_level == RobustnessLevel.FULL:
            defense_result = self._defense.detect_adversarial(vector)
            robustness.adversarial_score = defense_result.confidence
            robustness.is_adversarial = defense_result.detected_attack
            if robustness.is_adversarial:
                self._n_rejected_adversarial += 1

        # Calibrate confidence
        if self._calibrator._is_fitted:
            logits = self._uncertainty_model.forward(vector, apply_dropout=False)
            cal_result = self._calibrator.calibrate(logits)
            robustness.calibrated_confidence = float(np.max(cal_result.calibrated_probs))
            robustness.is_calibrated = True

        embedding = RobustEmbedding(
            vector=vector,
            modality=modality,
            source_module=source_module,
            confidence=base_confidence,
            timestamp=float(self._n_processed),
            robustness=robustness,
        )

        if embedding.effective_confidence() < 0.5:
            self._n_low_confidence += 1

        return embedding

    def filter_reliable(
        self,
        embeddings: List[RobustEmbedding],
        threshold: float = 0.5,
        exclude_ood: bool = True,
        exclude_adversarial: bool = True,
    ) -> List[RobustEmbedding]:
        """
        Filter embeddings to keep only reliable ones.

        Args:
            embeddings: List of robust embeddings
            threshold: Minimum effective confidence
            exclude_ood: Whether to exclude OOD samples
            exclude_adversarial: Whether to exclude adversarial samples

        Returns:
            Filtered list of reliable embeddings
        """
        reliable = []
        for emb in embeddings:
            if exclude_ood and emb.robustness.is_ood:
                continue
            if exclude_adversarial and emb.robustness.is_adversarial:
                continue
            if emb.effective_confidence() < threshold:
                continue
            reliable.append(emb)
        return reliable

    def rank_by_reliability(
        self,
        embeddings: List[RobustEmbedding],
    ) -> List[Tuple[RobustEmbedding, float]]:
        """
        Rank embeddings by reliability score.

        Args:
            embeddings: List of robust embeddings

        Returns:
            List of (embedding, reliability_score) sorted by score descending
        """
        scored = []
        for emb in embeddings:
            # Compute composite reliability score
            score = emb.effective_confidence()

            # Penalize high uncertainty
            score *= (1.0 - 0.5 * emb.robustness.uncertainty)

            # Penalize OOD
            if emb.robustness.is_ood:
                score *= 0.1

            # Penalize adversarial
            if emb.robustness.is_adversarial:
                score *= 0.0

            scored.append((emb, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def aggregate_with_uncertainty(
        self,
        embeddings: List[RobustEmbedding],
    ) -> RobustEmbedding:
        """
        Aggregate multiple embeddings with uncertainty weighting.

        Args:
            embeddings: List of embeddings to aggregate

        Returns:
            Aggregated embedding with combined uncertainty
        """
        if not embeddings:
            raise ValueError("Cannot aggregate empty list")

        # Weight by inverse uncertainty
        weights = []
        for emb in embeddings:
            w = 1.0 / (1.0 + emb.robustness.uncertainty)
            if emb.robustness.is_ood:
                w *= 0.1
            if emb.robustness.is_adversarial:
                w *= 0.0
            weights.append(w)

        total_weight = sum(weights)
        if total_weight < 1e-8:
            # All embeddings unreliable, use uniform weights
            weights = [1.0 / len(embeddings)] * len(embeddings)
            total_weight = 1.0

        weights = [w / total_weight for w in weights]

        # Weighted average of vectors
        agg_vector = np.zeros_like(embeddings[0].vector)
        for emb, w in zip(embeddings, weights):
            agg_vector += w * emb.vector

        # Normalize
        norm = np.linalg.norm(agg_vector)
        if norm > 1e-8:
            agg_vector = agg_vector / norm

        # Aggregate uncertainty (weighted average)
        agg_uncertainty = sum(
            w * emb.robustness.uncertainty
            for emb, w in zip(embeddings, weights)
        )
        agg_epistemic = sum(
            w * emb.robustness.epistemic
            for emb, w in zip(embeddings, weights)
        )
        agg_aleatoric = sum(
            w * emb.robustness.aleatoric
            for emb, w in zip(embeddings, weights)
        )
        agg_ood = sum(
            w * emb.robustness.ood_score
            for emb, w in zip(embeddings, weights)
        )

        robustness = RobustnessMetadata(
            uncertainty=agg_uncertainty,
            epistemic=agg_epistemic,
            aleatoric=agg_aleatoric,
            ood_score=agg_ood,
            is_ood=any(emb.robustness.is_ood for emb in embeddings),
            is_adversarial=any(emb.robustness.is_adversarial for emb in embeddings),
            robustness_level=self.robustness_level,
        )

        return RobustEmbedding(
            vector=agg_vector,
            modality="aggregated",
            source_module="robustness_aggregator",
            confidence=np.mean([emb.confidence for emb in embeddings]),
            timestamp=max(emb.timestamp for emb in embeddings),
            metadata={"n_aggregated": len(embeddings)},
            robustness=robustness,
        )

    def calibrate_threshold(
        self,
        embeddings: List[RobustEmbedding],
        target_reliability: float = 0.95,
    ) -> float:
        """
        Find confidence threshold for target reliability.

        Args:
            embeddings: Reference embeddings
            target_reliability: Desired fraction of reliable embeddings

        Returns:
            Confidence threshold
        """
        if not embeddings:
            return 0.5

        confidences = sorted([emb.effective_confidence() for emb in embeddings])
        idx = int((1 - target_reliability) * len(confidences))
        idx = max(0, min(len(confidences) - 1, idx))

        return confidences[idx]

    def statistics(self) -> Dict[str, Any]:
        """Get robustness processing statistics."""
        return {
            "n_processed": self._n_processed,
            "n_rejected_ood": self._n_rejected_ood,
            "n_rejected_adversarial": self._n_rejected_adversarial,
            "n_low_confidence": self._n_low_confidence,
            "rejection_rate_ood": (
                self._n_rejected_ood / max(1, self._n_processed)
            ),
            "rejection_rate_adversarial": (
                self._n_rejected_adversarial / max(1, self._n_processed)
            ),
            "low_confidence_rate": (
                self._n_low_confidence / max(1, self._n_processed)
            ),
            "is_fitted": self._is_fitted,
            "robustness_level": self.robustness_level.value,
            "embedding_dim": self.embedding_dim,
        }


# Helper to convert SimpleDropoutNN to the required interfaces
def _patch_simple_dropout_nn():
    """Add adapter methods to SimpleDropoutNN."""

    def _to_classifier(self):
        """Convert to SimpleClassifier interface."""
        from .ood_detection import SimpleClassifier
        classifier = SimpleClassifier(
            self.input_dim, self.hidden_dim, self.output_dim,
            random_seed=None,
        )
        classifier.W1 = self.W1.copy()
        classifier.W2 = self.W2.copy()
        classifier.b1 = self.b1.copy()
        classifier.b2 = self.b2.copy()
        return classifier

    def _to_simple_nn(self):
        """Convert to SimpleNN interface."""
        from .adversarial import SimpleNN
        nn = SimpleNN(
            self.input_dim, self.hidden_dim, self.output_dim,
            random_seed=None,
        )
        nn.W1 = self.W1.copy()
        nn.W2 = self.W2.copy()
        nn.b1 = self.b1.copy()
        nn.b2 = self.b2.copy()
        return nn

    SimpleDropoutNN._to_classifier = _to_classifier
    SimpleDropoutNN._to_simple_nn = _to_simple_nn

# Apply patch on import
_patch_simple_dropout_nn()


__all__ = [
    'RobustnessLevel',
    'RobustnessMetadata',
    'RobustEmbedding',
    'SharedSpaceRobustness',
]
