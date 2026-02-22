"""
Out-of-Distribution Detection.

Implements:
- Mahalanobis distance-based detection
- Energy-based OOD detection
- Likelihood ratio methods
- Isolation Forest
- Maximum Softmax Probability baseline
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np


class OODMethod(Enum):
    """OOD detection methods."""

    MSP = "msp"  # Maximum Softmax Probability
    ENERGY = "energy"  # Energy-based
    MAHALANOBIS = "mahalanobis"  # Mahalanobis distance
    LIKELIHOOD_RATIO = "likelihood_ratio"
    ISOLATION_FOREST = "isolation_forest"
    ENSEMBLE = "ensemble"  # Combine multiple methods


@dataclass
class OODResult:
    """Result of OOD detection."""

    is_ood: bool
    score: float  # Higher = more likely OOD
    confidence: float  # Confidence in detection
    method: OODMethod
    details: Dict[str, Any] = field(default_factory=dict)


class SimpleClassifier:
    """Simple neural network classifier for OOD detection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        random_seed: Optional[int] = None,
    ):
        rng = np.random.default_rng(random_seed)

        scale1 = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale2 = np.sqrt(2.0 / (hidden_dim + output_dim))

        self.W1 = rng.normal(0, scale1, (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, scale2, (hidden_dim, output_dim))
        self.b2 = np.zeros(output_dim)

        self._hidden = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass, returning logits."""
        self._hidden = np.maximum(0, x @ self.W1 + self.b1)
        return self._hidden @ self.W2 + self.b2

    def get_features(self, x: np.ndarray) -> np.ndarray:
        """Get hidden layer features."""
        self.forward(x)
        return self._hidden

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Get class probabilities."""
        logits = self.forward(x)
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)


class OODDetector:
    """
    Out-of-Distribution Detection.

    Detects whether inputs come from a different distribution
    than the training data.
    """

    def __init__(
        self,
        model: SimpleClassifier,
        method: OODMethod = OODMethod.ENERGY,
        threshold: Optional[float] = None,
    ):
        self.model = model
        self.method = method
        self.threshold = threshold

        # For Mahalanobis distance
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_cov_inv: Optional[np.ndarray] = None
        self._class_means: Dict[int, np.ndarray] = {}

        # For Isolation Forest
        self._isolation_trees: List[Any] = []

        # Statistics
        self._n_detections = 0
        self._n_ood = 0

    def fit(
        self,
        data: List[Tuple[np.ndarray, int]],
    ) -> None:
        """
        Fit OOD detector on in-distribution data.

        Args:
            data: In-distribution (input, label) pairs
        """
        # Collect features
        features = []
        labels = []
        for x, y in data:
            feat = self.model.get_features(x)
            features.append(feat)
            labels.append(y)

        features = np.array(features)
        labels = np.array(labels)

        # Compute global statistics
        self._feature_mean = np.mean(features, axis=0)
        cov = np.cov(features.T)
        # Add small regularization for invertibility
        cov += 1e-6 * np.eye(cov.shape[0])
        self._feature_cov_inv = np.linalg.inv(cov)

        # Compute per-class statistics
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            self._class_means[label] = np.mean(features[mask], axis=0)

        # Fit Isolation Forest
        self._fit_isolation_forest(features)

    def _fit_isolation_forest(
        self,
        features: np.ndarray,
        n_trees: int = 10,
        max_samples: int = 256,
    ) -> None:
        """Fit Isolation Forest for anomaly detection."""
        n_samples = len(features)
        self._isolation_trees = []

        for _ in range(n_trees):
            # Sample subset
            sample_size = min(max_samples, n_samples)
            indices = np.random.choice(n_samples, sample_size, replace=False)
            sample = features[indices]

            # Build tree
            tree = self._build_isolation_tree(sample, 0, int(np.ceil(np.log2(sample_size))))
            self._isolation_trees.append(tree)

    def _build_isolation_tree(
        self,
        data: np.ndarray,
        depth: int,
        max_depth: int,
    ) -> Dict[str, Any]:
        """Build a single isolation tree."""
        n_samples, n_features = data.shape

        if depth >= max_depth or n_samples <= 1:
            return {"type": "leaf", "size": n_samples}

        # Random feature and split
        feature_idx = np.random.randint(n_features)
        feature_values = data[:, feature_idx]
        min_val, max_val = np.min(feature_values), np.max(feature_values)

        if min_val == max_val:
            return {"type": "leaf", "size": n_samples}

        split_value = np.random.uniform(min_val, max_val)

        left_mask = feature_values < split_value
        right_mask = ~left_mask

        return {
            "type": "node",
            "feature": feature_idx,
            "split": split_value,
            "left": self._build_isolation_tree(data[left_mask], depth + 1, max_depth),
            "right": self._build_isolation_tree(data[right_mask], depth + 1, max_depth),
        }

    def _path_length(self, x: np.ndarray, tree: Dict, depth: int = 0) -> float:
        """Compute path length for a sample in isolation tree."""
        if tree["type"] == "leaf":
            # Adjustment for external nodes
            n = tree["size"]
            if n <= 1:
                return depth
            return depth + self._c(n)

        if x[tree["feature"]] < tree["split"]:
            return self._path_length(x, tree["left"], depth + 1)
        else:
            return self._path_length(x, tree["right"], depth + 1)

    def _c(self, n: int) -> float:
        """Average path length in unsuccessful search in BST."""
        if n <= 1:
            return 0
        return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n

    def msp_score(self, x: np.ndarray) -> float:
        """
        Maximum Softmax Probability score.

        Lower max probability = more likely OOD.
        """
        probs = self.model.predict_proba(x)
        return 1.0 - float(np.max(probs))  # Invert so higher = more OOD

    def energy_score(self, x: np.ndarray, temperature: float = 1.0) -> float:
        """
        Energy-based OOD score.

        E(x) = -T * log(sum(exp(f(x)/T)))
        Lower energy = in-distribution.
        """
        logits = self.model.forward(x)
        energy = -temperature * np.log(np.sum(np.exp(logits / temperature)))
        return float(-energy)  # Negate so higher = more OOD

    def mahalanobis_score(self, x: np.ndarray) -> float:
        """
        Mahalanobis distance-based OOD score.

        Measures distance from class-conditional Gaussians.
        Higher distance = more likely OOD.
        """
        if self._feature_cov_inv is None:
            raise RuntimeError("Must call fit() before using Mahalanobis score")

        features = self.model.get_features(x)

        # Find minimum Mahalanobis distance to any class
        min_distance = float("inf")
        for class_mean in self._class_means.values():
            diff = features - class_mean
            distance = float(diff @ self._feature_cov_inv @ diff)
            min_distance = min(min_distance, distance)

        return min_distance

    def likelihood_ratio_score(self, x: np.ndarray) -> float:
        """
        Likelihood ratio test for OOD.

        Compares likelihood under in-distribution vs background model.
        """
        if self._feature_mean is None:
            raise RuntimeError("Must call fit() before using likelihood ratio")

        features = self.model.get_features(x)

        # In-distribution: fitted Gaussian
        diff_in = features - self._feature_mean
        log_lik_in = -0.5 * float(diff_in @ self._feature_cov_inv @ diff_in)

        # Background: uniform/wide Gaussian
        log_lik_bg = -0.5 * float(np.sum(features**2) / 100)  # Wide variance

        # Log likelihood ratio (lower = more OOD)
        return float(log_lik_bg - log_lik_in)

    def isolation_forest_score(self, x: np.ndarray) -> float:
        """
        Isolation Forest anomaly score.

        Anomalies have shorter path lengths in isolation trees.
        Score in [0, 1] where higher = more anomalous.
        """
        if not self._isolation_trees:
            raise RuntimeError("Must call fit() before using Isolation Forest")

        features = self.model.get_features(x)

        # Average path length across trees
        path_lengths = [self._path_length(features, tree) for tree in self._isolation_trees]
        avg_path = np.mean(path_lengths)

        # Normalize to [0, 1]
        # s(x, n) = 2^(-E(h(x))/c(n))
        n = 256  # Assume standard sample size
        score = 2 ** (-avg_path / self._c(n))

        return float(score)

    def compute_score(self, x: np.ndarray) -> float:
        """Compute OOD score using configured method."""
        if self.method == OODMethod.MSP:
            return self.msp_score(x)
        elif self.method == OODMethod.ENERGY:
            return self.energy_score(x)
        elif self.method == OODMethod.MAHALANOBIS:
            return self.mahalanobis_score(x)
        elif self.method == OODMethod.LIKELIHOOD_RATIO:
            return self.likelihood_ratio_score(x)
        elif self.method == OODMethod.ISOLATION_FOREST:
            return self.isolation_forest_score(x)
        elif self.method == OODMethod.ENSEMBLE:
            return self._ensemble_score(x)
        else:
            return self.energy_score(x)

    def _ensemble_score(self, x: np.ndarray) -> float:
        """Combine multiple OOD detection methods."""
        scores = []

        # MSP
        scores.append(self.msp_score(x))

        # Energy
        scores.append(self.energy_score(x) / 10)  # Normalize

        # Mahalanobis (if fitted)
        if self._feature_cov_inv is not None:
            maha = self.mahalanobis_score(x)
            scores.append(min(maha / 100, 1.0))  # Normalize

        # Isolation Forest (if fitted)
        if self._isolation_trees:
            scores.append(self.isolation_forest_score(x))

        return float(np.mean(scores))

    def is_ood(self, x: np.ndarray, threshold: Optional[float] = None) -> bool:
        """Binary OOD decision."""
        threshold = threshold or self.threshold or 0.5
        score = self.compute_score(x)
        return score > threshold

    def detect(self, x: np.ndarray, threshold: Optional[float] = None) -> OODResult:
        """
        Full OOD detection with all information.

        Args:
            x: Input to check
            threshold: Detection threshold (higher score = more OOD)

        Returns:
            OODResult with detection information
        """
        self._n_detections += 1
        threshold = threshold or self.threshold or 0.5

        score = self.compute_score(x)
        is_ood = score > threshold

        if is_ood:
            self._n_ood += 1

        # Compute confidence based on distance from threshold
        distance_from_threshold = abs(score - threshold)
        confidence = min(1.0, distance_from_threshold / 0.3)

        # Collect additional details
        details = {
            "threshold": threshold,
            "margin": score - threshold,
        }

        # Add method-specific details
        if self.method == OODMethod.MSP:
            probs = self.model.predict_proba(x)
            details["max_prob"] = float(np.max(probs))
            details["predicted_class"] = int(np.argmax(probs))

        return OODResult(
            is_ood=is_ood,
            score=score,
            confidence=confidence,
            method=self.method,
            details=details,
        )

    def ood_confidence(self, x: np.ndarray) -> float:
        """Get soft OOD score (probability-like)."""
        score = self.compute_score(x)
        # Sigmoid transformation
        return float(1 / (1 + np.exp(-10 * (score - 0.5))))

    def calibrate_threshold(
        self,
        in_dist_data: List[np.ndarray],
        target_fpr: float = 0.05,
    ) -> float:
        """
        Calibrate threshold for target false positive rate.

        Args:
            in_dist_data: In-distribution samples
            target_fpr: Target false positive rate (fraction of ID marked as OOD)

        Returns:
            Calibrated threshold
        """
        scores = [self.compute_score(x) for x in in_dist_data]
        threshold = float(np.percentile(scores, 100 * (1 - target_fpr)))
        self.threshold = threshold
        return threshold

    def detection_rate(self) -> float:
        """Get rate of inputs detected as OOD."""
        if self._n_detections == 0:
            return 0.0
        return self._n_ood / self._n_detections

    def statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            "n_detections": self._n_detections,
            "n_ood": self._n_ood,
            "detection_rate": self.detection_rate(),
            "method": self.method.value,
            "threshold": self.threshold,
            "fitted": self._feature_mean is not None,
        }
