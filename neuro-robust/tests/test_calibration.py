"""Tests for confidence calibration."""

import pytest
import numpy as np
from src.calibration import (
    ConfidenceCalibrator, CalibrationMethod, CalibrationResult,
)
from src.uncertainty import SimpleDropoutNN


class TestConfidenceCalibrator:
    """Tests for ConfidenceCalibrator."""

    @pytest.fixture
    def model(self):
        """Simple model for generating logits."""
        return SimpleDropoutNN(10, 20, 3, dropout_rate=0.0, random_seed=42)

    @pytest.fixture
    def calibrator(self):
        return ConfidenceCalibrator(method=CalibrationMethod.TEMPERATURE)

    @pytest.fixture
    def logits_and_labels(self, model):
        """Generate logits and labels for calibration."""
        np.random.seed(42)
        n_samples = 100
        logits = []
        labels = []
        for i in range(n_samples):
            x = np.random.randn(10)
            log = model.forward(x, apply_dropout=False)
            logits.append(log)
            labels.append(i % 3)
        return np.array(logits), np.array(labels)

    def test_temperature_scaling_changes_probs(self, calibrator):
        """Test temperature scaling changes probabilities."""
        logits = np.array([1.0, 2.0, 3.0])
        probs_t1 = calibrator.temperature_scaling(logits, T=1.0)
        probs_t2 = calibrator.temperature_scaling(logits, T=2.0)
        assert not np.allclose(probs_t1, probs_t2)

    def test_temperature_scaling_softens(self, calibrator):
        """Test higher temperature softens distribution."""
        logits = np.array([1.0, 2.0, 3.0])
        probs_t1 = calibrator.temperature_scaling(logits, T=1.0)
        probs_t10 = calibrator.temperature_scaling(logits, T=10.0)
        # Higher T -> more uniform
        assert np.max(probs_t10) < np.max(probs_t1)

    def test_temperature_scaling_valid_probs(self, calibrator):
        """Test temperature scaling returns valid probabilities."""
        logits = np.array([1.0, 2.0, 3.0])
        for T in [0.5, 1.0, 2.0, 5.0]:
            probs = calibrator.temperature_scaling(logits, T=T)
            assert np.all(probs >= 0)
            assert np.isclose(np.sum(probs), 1.0)

    def test_fit_temperature_scaling(self, logits_and_labels):
        """Test fitting temperature scaling."""
        logits, labels = logits_and_labels
        calibrator = ConfidenceCalibrator(method=CalibrationMethod.TEMPERATURE)
        initial_T = calibrator.temperature
        calibrator.fit(logits, labels)
        assert calibrator._is_fitted
        assert isinstance(calibrator.temperature, float)

    def test_fit_platt_scaling(self, logits_and_labels):
        """Test Platt scaling fitting."""
        logits, labels = logits_and_labels
        calibrator = ConfidenceCalibrator(method=CalibrationMethod.PLATT)
        calibrator.fit(logits, labels)
        assert calibrator.platt_a is not None
        assert calibrator.platt_b is not None

    def test_platt_scaling_apply(self, logits_and_labels):
        """Test Platt scaling application."""
        logits, labels = logits_and_labels
        calibrator = ConfidenceCalibrator(method=CalibrationMethod.PLATT)
        calibrator.fit(logits, labels)
        probs = calibrator.platt_scaling(np.array([1.0, 2.0, 3.0]))
        assert np.all(probs >= 0)
        assert np.isclose(np.sum(probs), 1.0)

    def test_isotonic_fit(self, logits_and_labels):
        """Test isotonic regression fitting."""
        logits, labels = logits_and_labels
        calibrator = ConfidenceCalibrator(method=CalibrationMethod.ISOTONIC)
        calibrator.fit(logits, labels)
        assert calibrator._isotonic_calibrator is not None

    def test_histogram_binning_fit(self, logits_and_labels):
        """Test histogram binning fitting."""
        logits, labels = logits_and_labels
        calibrator = ConfidenceCalibrator(method=CalibrationMethod.HISTOGRAM)
        calibrator.fit(logits, labels)
        assert calibrator._histogram_calibrator is not None

    def test_calibrate_returns_result(self, calibrator):
        """Test calibrate returns CalibrationResult."""
        logits = np.array([1.0, 2.0, 3.0])
        result = calibrator.calibrate(logits)
        assert isinstance(result, CalibrationResult)

    def test_calibrate_valid_probs(self, calibrator):
        """Test calibrated probabilities are valid."""
        logits = np.array([1.0, 2.0, 3.0])
        result = calibrator.calibrate(logits)
        assert np.all(result.calibrated_probs >= 0)
        assert np.isclose(np.sum(result.calibrated_probs), 1.0)

    def test_expected_calibration_error(self):
        """Test ECE computation."""
        calibrator = ConfidenceCalibrator()
        probs = np.random.rand(100, 3)
        probs = probs / probs.sum(axis=1, keepdims=True)
        labels = np.random.randint(0, 3, 100)

        ece = calibrator.expected_calibration_error(probs, labels)
        assert 0 <= ece <= 1

    def test_maximum_calibration_error(self, calibrator):
        """Test MCE computation."""
        probs = np.random.rand(100, 3)
        probs = probs / probs.sum(axis=1, keepdims=True)
        labels = np.random.randint(0, 3, 100)

        mce = calibrator.maximum_calibration_error(probs, labels)
        assert 0 <= mce <= 1

    def test_brier_score(self, calibrator):
        """Test Brier score computation."""
        probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
        labels = np.array([0, 1, 2])
        brier = calibrator.brier_score(probs, labels)
        assert 0 <= brier <= 2  # Max Brier score is 2

    def test_brier_score_perfect(self, calibrator):
        """Test Brier score for perfect predictions."""
        probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        labels = np.array([0, 1, 2])
        brier = calibrator.brier_score(probs, labels)
        assert brier == 0.0

    def test_reliability_diagram(self, calibrator):
        """Test reliability diagram generation."""
        probs = np.random.rand(100, 3)
        probs = probs / probs.sum(axis=1, keepdims=True)
        labels = np.random.randint(0, 3, 100)

        diagram = calibrator.reliability_diagram(probs, labels)
        assert "accuracies" in diagram
        assert "confidences" in diagram
        assert "counts" in diagram

    def test_statistics_tracking(self, calibrator):
        """Test statistics are tracked."""
        logits = np.array([1.0, 2.0, 3.0])
        calibrator.calibrate(logits)
        calibrator.calibrate(logits)
        stats = calibrator.statistics()
        assert stats["n_calibrations"] == 2


class TestCalibrationMethods:
    """Tests comparing calibration methods."""

    @pytest.fixture
    def model(self):
        return SimpleDropoutNN(10, 20, 3, dropout_rate=0.0, random_seed=42)

    @pytest.fixture
    def logits_and_labels(self, model):
        np.random.seed(42)
        n_samples = 200
        logits = np.array([model.forward(np.random.randn(10), apply_dropout=False)
                          for _ in range(n_samples)])
        labels = np.array([i % 3 for i in range(n_samples)])
        return logits, labels

    def test_all_methods_produce_valid_probs(self, logits_and_labels):
        """Test all methods produce valid probabilities."""
        logits, labels = logits_and_labels

        for method in CalibrationMethod:
            calibrator = ConfidenceCalibrator(method=method)
            calibrator.fit(logits, labels)

            for log in logits[:10]:
                result = calibrator.calibrate(log)
                assert np.all(result.calibrated_probs >= 0)
                assert np.isclose(np.sum(result.calibrated_probs), 1.0)

    def test_temperature_preserves_ranking(self):
        """Test temperature scaling preserves prediction ranking."""
        calibrator = ConfidenceCalibrator()

        for _ in range(10):
            logits = np.random.randn(5)

            probs_t1 = calibrator.temperature_scaling(logits, T=1.0)
            probs_t3 = calibrator.temperature_scaling(logits, T=3.0)

            # Ranking should be preserved
            assert np.argmax(probs_t1) == np.argmax(probs_t3)


class TestCalibrationEdgeCases:
    """Edge case tests for calibration."""

    @pytest.fixture
    def calibrator(self):
        return ConfidenceCalibrator()

    def test_extreme_logits(self, calibrator):
        """Test handling of extreme logits."""
        logits = np.array([100.0, -100.0, 0.0])
        result = calibrator.calibrate(logits)
        assert np.all(result.calibrated_probs >= 0)
        assert np.isclose(np.sum(result.calibrated_probs), 1.0)
        assert not np.any(np.isnan(result.calibrated_probs))

    def test_equal_logits(self, calibrator):
        """Test handling of equal logits."""
        logits = np.array([1.0, 1.0, 1.0])
        result = calibrator.calibrate(logits)
        # Should be uniform
        assert np.allclose(result.calibrated_probs, [1/3, 1/3, 1/3])

    def test_small_calibration_set(self):
        """Test calibration with small dataset."""
        calibrator = ConfidenceCalibrator(method=CalibrationMethod.TEMPERATURE)
        logits = np.random.randn(10, 3)
        labels = np.array([i % 3 for i in range(10)])
        # Should not crash
        calibrator.fit(logits, labels)

    def test_single_class_data(self):
        """Test calibration with single class."""
        calibrator = ConfidenceCalibrator(method=CalibrationMethod.TEMPERATURE)
        logits = np.random.randn(50, 3)
        labels = np.zeros(50, dtype=int)
        # Should not crash
        calibrator.fit(logits, labels)

    def test_very_low_temperature(self, calibrator):
        """Test very low temperature."""
        logits = np.array([1.0, 2.0, 3.0])
        probs = calibrator.temperature_scaling(logits, T=0.1)
        # Should be nearly one-hot
        assert np.max(probs) > 0.99

    def test_very_high_temperature(self, calibrator):
        """Test very high temperature."""
        logits = np.array([1.0, 2.0, 3.0])
        probs = calibrator.temperature_scaling(logits, T=100.0)
        # Should be nearly uniform
        assert np.max(probs) < 0.4

    def test_zero_logits(self, calibrator):
        """Test zero logits."""
        logits = np.zeros(3)
        result = calibrator.calibrate(logits)
        assert np.allclose(result.calibrated_probs, [1/3, 1/3, 1/3])

    def test_single_class_logits(self, calibrator):
        """Test single element logits."""
        logits = np.array([1.0])
        result = calibrator.calibrate(logits)
        assert result.calibrated_probs[0] == 1.0


class TestECEComputation:
    """Detailed tests for ECE computation."""

    @pytest.fixture
    def calibrator(self):
        return ConfidenceCalibrator()

    def test_ece_perfect_calibration(self, calibrator):
        """Test ECE is low for well-calibrated model."""
        n = 1000
        probs = []
        labels = []

        np.random.seed(42)
        for conf in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for _ in range(n // 5):
                p = np.array([conf, (1 - conf) / 2, (1 - conf) / 2])
                probs.append(p)
                labels.append(0 if np.random.rand() < conf else 1)

        probs = np.array(probs)
        labels = np.array(labels)

        ece = calibrator.expected_calibration_error(probs, labels)
        assert ece < 0.2  # Allow some variance

    def test_ece_overconfident(self, calibrator):
        """Test ECE detects overconfidence."""
        n = 100
        probs = np.array([[0.9, 0.05, 0.05]] * n)
        labels = np.array([0 if i % 2 == 0 else 1 for i in range(n)])

        ece = calibrator.expected_calibration_error(probs, labels)
        assert ece > 0.3

    def test_ece_underconfident(self, calibrator):
        """Test ECE detects underconfidence."""
        n = 100
        probs = np.array([[0.5, 0.25, 0.25]] * n)
        labels = np.array([0] * 90 + [1] * 10)

        ece = calibrator.expected_calibration_error(probs, labels)
        assert ece > 0.3

    def test_mce_greater_than_ece(self, calibrator):
        """Test MCE >= ECE."""
        probs = np.random.rand(100, 3)
        probs = probs / probs.sum(axis=1, keepdims=True)
        labels = np.random.randint(0, 3, 100)

        ece = calibrator.expected_calibration_error(probs, labels)
        mce = calibrator.maximum_calibration_error(probs, labels)
        assert mce >= ece - 0.01  # Allow small numerical difference

    def test_brier_score_bounds(self, calibrator):
        """Test Brier score is bounded."""
        for _ in range(10):
            probs = np.random.rand(50, 5)
            probs = probs / probs.sum(axis=1, keepdims=True)
            labels = np.random.randint(0, 5, 50)

            brier = calibrator.brier_score(probs, labels)
            assert 0 <= brier <= 2


class TestFittingBehavior:
    """Tests for calibration fitting behavior."""

    def test_fit_sets_is_fitted(self):
        """Test fit sets _is_fitted flag."""
        calibrator = ConfidenceCalibrator(method=CalibrationMethod.TEMPERATURE)
        logits = np.random.randn(50, 3)
        labels = np.array([i % 3 for i in range(50)])
        assert not calibrator._is_fitted
        calibrator.fit(logits, labels)
        assert calibrator._is_fitted

    def test_temperature_stays_positive(self):
        """Test temperature parameter stays positive."""
        calibrator = ConfidenceCalibrator(method=CalibrationMethod.TEMPERATURE)
        logits = np.random.randn(100, 3)
        labels = np.array([i % 3 for i in range(100)])
        calibrator.fit(logits, labels)
        assert calibrator.temperature > 0

    def test_fit_returns_metrics(self):
        """Test fit returns metrics dict."""
        calibrator = ConfidenceCalibrator(method=CalibrationMethod.TEMPERATURE)
        logits = np.random.randn(100, 3)
        labels = np.array([i % 3 for i in range(100)])
        metrics = calibrator.fit(logits, labels)
        assert "ece" in metrics
        assert "mce" in metrics
        assert "brier_score" in metrics

    def test_calibrate_batch(self):
        """Test batch calibration."""
        calibrator = ConfidenceCalibrator(method=CalibrationMethod.TEMPERATURE)
        logits = np.random.randn(100, 3)
        labels = np.array([i % 3 for i in range(100)])
        calibrator.fit(logits, labels)

        batch_probs = calibrator.calibrate_batch(logits[:10])
        assert batch_probs.shape == (10, 3)
        assert np.allclose(batch_probs.sum(axis=1), 1.0)
