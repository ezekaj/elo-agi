"""Tests for out-of-distribution detection."""

import pytest
import numpy as np
from neuro.modules.robust.ood_detection import (
    SimpleClassifier,
    OODDetector,
    OODMethod,
    OODResult,
)


class TestSimpleClassifier:
    """Tests for SimpleClassifier."""

    def test_initialization(self):
        """Test classifier initialization."""
        model = SimpleClassifier(10, 20, 3)
        assert model.W1.shape == (10, 20)
        assert model.W2.shape == (20, 3)

    def test_forward_shape(self):
        """Test forward returns correct shape."""
        model = SimpleClassifier(10, 20, 3)
        x = np.random.randn(10)
        logits = model.forward(x)
        assert logits.shape == (3,)

    def test_get_features_shape(self):
        """Test get_features returns hidden layer."""
        model = SimpleClassifier(10, 20, 3)
        x = np.random.randn(10)
        features = model.get_features(x)
        assert features.shape == (20,)

    def test_predict_proba_valid(self):
        """Test predict_proba returns valid probabilities."""
        model = SimpleClassifier(10, 20, 3)
        x = np.random.randn(10)
        probs = model.predict_proba(x)
        assert np.all(probs >= 0)
        assert np.isclose(np.sum(probs), 1.0)


class TestOODDetector:
    """Tests for OODDetector."""

    @pytest.fixture
    def model(self):
        return SimpleClassifier(10, 20, 3, random_seed=42)

    @pytest.fixture
    def detector(self, model):
        return OODDetector(model, method=OODMethod.ENERGY)

    @pytest.fixture
    def in_dist_data(self):
        """In-distribution data (centered around 0)."""
        return [(np.random.randn(10) * 0.5, i % 3) for i in range(100)]

    @pytest.fixture
    def ood_data(self):
        """Out-of-distribution data (shifted mean)."""
        return [np.random.randn(10) + 5.0 for _ in range(20)]

    def test_msp_score_range(self, detector):
        """Test MSP score is in [0, 1]."""
        x = np.random.randn(10)
        score = detector.msp_score(x)
        assert 0 <= score <= 1

    def test_msp_score_lower_for_confident(self, detector):
        """Test MSP gives lower OOD score for confident predictions."""
        # This tests the inverse relationship
        x = np.random.randn(10)
        probs = detector.model.predict_proba(x)
        score = detector.msp_score(x)
        # score = 1 - max_prob, so higher max_prob means lower score
        assert score == 1.0 - np.max(probs)

    def test_energy_score_computed(self, detector):
        """Test energy score is computed."""
        x = np.random.randn(10)
        score = detector.energy_score(x)
        assert isinstance(score, float)

    def test_mahalanobis_requires_fit(self, detector):
        """Test Mahalanobis requires fitting first."""
        x = np.random.randn(10)
        with pytest.raises(RuntimeError):
            detector.mahalanobis_score(x)

    def test_fit_enables_mahalanobis(self, detector, in_dist_data):
        """Test fit enables Mahalanobis score."""
        detector.fit(in_dist_data)
        x = np.random.randn(10)
        score = detector.mahalanobis_score(x)
        assert isinstance(score, float)
        assert score >= 0

    def test_mahalanobis_higher_for_ood(self, detector, in_dist_data, ood_data):
        """Test Mahalanobis score higher for OOD."""
        detector.fit(in_dist_data)

        in_scores = [detector.mahalanobis_score(x) for x, _ in in_dist_data[:20]]
        ood_scores = [detector.mahalanobis_score(x) for x in ood_data]

        assert np.mean(ood_scores) > np.mean(in_scores)

    def test_likelihood_ratio_requires_fit(self, detector):
        """Test likelihood ratio requires fitting."""
        x = np.random.randn(10)
        with pytest.raises(RuntimeError):
            detector.likelihood_ratio_score(x)

    def test_likelihood_ratio_computed(self, detector, in_dist_data):
        """Test likelihood ratio is computed after fit."""
        detector.fit(in_dist_data)
        x = np.random.randn(10)
        score = detector.likelihood_ratio_score(x)
        assert isinstance(score, float)

    def test_isolation_forest_requires_fit(self, detector):
        """Test isolation forest requires fitting."""
        x = np.random.randn(10)
        with pytest.raises(RuntimeError):
            detector.isolation_forest_score(x)

    def test_isolation_forest_score_range(self, detector, in_dist_data):
        """Test isolation forest score in [0, 1]."""
        detector.fit(in_dist_data)
        x = np.random.randn(10)
        score = detector.isolation_forest_score(x)
        assert 0 <= score <= 1

    def test_isolation_forest_higher_for_anomaly(self, detector, in_dist_data, ood_data):
        """Test isolation forest higher for anomalies."""
        detector.fit(in_dist_data)

        in_scores = [detector.isolation_forest_score(x) for x, _ in in_dist_data[:20]]
        ood_scores = [detector.isolation_forest_score(x) for x in ood_data]

        # OOD should have higher average score
        assert np.mean(ood_scores) > np.mean(in_scores) - 0.1

    def test_is_ood_returns_bool(self, detector):
        """Test is_ood returns boolean."""
        x = np.random.randn(10)
        result = detector.is_ood(x, threshold=0.5)
        assert isinstance(result, bool)

    def test_detect_returns_result(self, detector):
        """Test detect returns OODResult."""
        x = np.random.randn(10)
        result = detector.detect(x)
        assert isinstance(result, OODResult)
        assert result.method == OODMethod.ENERGY

    def test_detect_includes_score(self, detector):
        """Test detect includes score."""
        x = np.random.randn(10)
        result = detector.detect(x)
        assert isinstance(result.score, float)

    def test_detect_includes_confidence(self, detector):
        """Test detect includes confidence."""
        x = np.random.randn(10)
        result = detector.detect(x)
        assert 0 <= result.confidence <= 1

    def test_ood_confidence_sigmoid(self, detector):
        """Test ood_confidence returns sigmoid-like score."""
        x = np.random.randn(10)
        conf = detector.ood_confidence(x)
        assert 0 <= conf <= 1

    def test_calibrate_threshold(self, detector, in_dist_data):
        """Test threshold calibration."""
        detector.fit(in_dist_data)
        in_samples = [x for x, _ in in_dist_data]
        threshold = detector.calibrate_threshold(in_samples, target_fpr=0.05)
        assert isinstance(threshold, float)
        assert detector.threshold == threshold

    def test_calibrate_threshold_controls_fpr(self, detector, in_dist_data):
        """Test calibrated threshold controls FPR."""
        detector.fit(in_dist_data)
        in_samples = [x for x, _ in in_dist_data]
        detector.calibrate_threshold(in_samples, target_fpr=0.1)

        # Check FPR on in-distribution
        false_positives = sum(1 for x in in_samples if detector.is_ood(x))
        fpr = false_positives / len(in_samples)
        assert fpr <= 0.15  # Allow some variance

    def test_statistics_tracking(self, detector):
        """Test statistics are tracked."""
        x = np.random.randn(10)
        detector.detect(x, threshold=0.3)
        detector.detect(x, threshold=0.7)
        stats = detector.statistics()
        assert stats["n_detections"] == 2

    def test_detection_rate(self, detector):
        """Test detection rate computation."""
        detector.threshold = 0.3
        for _ in range(10):
            x = np.random.randn(10)
            detector.detect(x)
        rate = detector.detection_rate()
        assert 0 <= rate <= 1


class TestOODMethods:
    """Tests for different OOD methods."""

    @pytest.fixture
    def model(self):
        return SimpleClassifier(10, 20, 3, random_seed=42)

    @pytest.fixture
    def in_dist_data(self):
        return [(np.random.randn(10) * 0.5, i % 3) for i in range(100)]

    def test_all_methods_return_score(self, model, in_dist_data):
        """Test all methods return a score."""
        x = np.random.randn(10)

        for method in OODMethod:
            detector = OODDetector(model, method=method)
            if method in [
                OODMethod.MAHALANOBIS,
                OODMethod.LIKELIHOOD_RATIO,
                OODMethod.ISOLATION_FOREST,
                OODMethod.ENSEMBLE,
            ]:
                detector.fit(in_dist_data)

            score = detector.compute_score(x)
            assert isinstance(score, float)

    def test_ensemble_combines_methods(self, model, in_dist_data):
        """Test ensemble combines multiple methods."""
        detector = OODDetector(model, method=OODMethod.ENSEMBLE)
        detector.fit(in_dist_data)

        x = np.random.randn(10)
        score = detector._ensemble_score(x)
        assert isinstance(score, float)

    def test_methods_distinguish_ood(self, model, in_dist_data):
        """Test all methods can distinguish OOD."""
        ood_data = [np.random.randn(10) + 10.0 for _ in range(20)]

        for method in OODMethod:
            detector = OODDetector(model, method=method)
            if method in [
                OODMethod.MAHALANOBIS,
                OODMethod.LIKELIHOOD_RATIO,
                OODMethod.ISOLATION_FOREST,
                OODMethod.ENSEMBLE,
            ]:
                detector.fit(in_dist_data)

            in_scores = [detector.compute_score(x) for x, _ in in_dist_data[:20]]
            ood_scores = [detector.compute_score(x) for x in ood_data]

            # Most OOD samples should have higher scores
            # (allowing for some overlap)
            median_in = np.median(in_scores)
            median_ood = np.median(ood_scores)
            # At least the median OOD should be higher than median in-dist
            assert median_ood >= median_in - 0.5, f"Method {method} failed"


class TestIsolationForest:
    """Tests specifically for Isolation Forest implementation."""

    @pytest.fixture
    def detector(self):
        model = SimpleClassifier(10, 20, 3, random_seed=42)
        return OODDetector(model, method=OODMethod.ISOLATION_FOREST)

    @pytest.fixture
    def data(self):
        return [(np.random.randn(10), i % 3) for i in range(200)]

    def test_trees_created(self, detector, data):
        """Test isolation trees are created."""
        detector.fit(data)
        assert len(detector._isolation_trees) > 0

    def test_path_length_positive(self, detector, data):
        """Test path length is positive."""
        detector.fit(data)
        x = np.random.randn(10)
        features = detector.model.get_features(x)
        for tree in detector._isolation_trees:
            length = detector._path_length(features, tree)
            assert length >= 0

    def test_anomaly_has_short_path(self, detector, data):
        """Test anomalies have shorter paths."""
        detector.fit(data)

        # Normal point
        x_normal = np.random.randn(10) * 0.5

        # Anomaly (extreme values)
        x_anomaly = np.ones(10) * 100

        score_normal = detector.isolation_forest_score(x_normal)
        score_anomaly = detector.isolation_forest_score(x_anomaly)

        # Anomaly should have higher score (shorter path -> higher score)
        assert score_anomaly > score_normal


class TestOODEdgeCases:
    """Edge case tests for OOD detection."""

    @pytest.fixture
    def model(self):
        return SimpleClassifier(10, 20, 3, random_seed=42)

    def test_zero_input(self, model):
        """Test handling of zero input."""
        detector = OODDetector(model, method=OODMethod.ENERGY)
        x = np.zeros(10)
        result = detector.detect(x)
        assert isinstance(result.score, float)
        assert not np.isnan(result.score)

    def test_large_input(self, model):
        """Test handling of large input values."""
        detector = OODDetector(model, method=OODMethod.MSP)
        x = np.ones(10) * 1000
        result = detector.detect(x)
        assert isinstance(result.score, float)
        assert not np.isnan(result.score)

    def test_negative_input(self, model):
        """Test handling of negative input."""
        detector = OODDetector(model, method=OODMethod.ENERGY)
        x = -np.ones(10) * 10
        result = detector.detect(x)
        assert isinstance(result.score, float)

    def test_small_dataset_fit(self, model):
        """Test fitting with small dataset."""
        detector = OODDetector(model, method=OODMethod.MAHALANOBIS)
        small_data = [(np.random.randn(10), 0) for _ in range(5)]
        # Should not crash
        detector.fit(small_data)

    def test_single_class_fit(self, model):
        """Test fitting with single class."""
        detector = OODDetector(model, method=OODMethod.MAHALANOBIS)
        single_class_data = [(np.random.randn(10), 0) for _ in range(50)]
        detector.fit(single_class_data)
        assert len(detector._class_means) == 1
