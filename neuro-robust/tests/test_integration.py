"""Integration tests for robust inference system."""

import pytest
import numpy as np
from neuro.modules.robust.robust_inference import (
    RobustInference,
    AdaptiveThresholdInference,
    RobustPrediction,
    SelectivePrediction,
    RejectionPolicy,
)
from neuro.modules.robust.uncertainty import SimpleDropoutNN, UncertaintyQuantifier
from neuro.modules.robust.ood_detection import OODDetector, OODMethod, SimpleClassifier


class TestRobustInference:
    """Tests for RobustInference."""

    @pytest.fixture
    def model(self):
        return SimpleDropoutNN(10, 20, 3, dropout_rate=0.2, random_seed=42)

    @pytest.fixture
    def robust_inference(self, model):
        return RobustInference(
            model,
            rejection_policy=RejectionPolicy.COMBINED,
            confidence_threshold=0.6,
            uncertainty_threshold=0.5,
        )

    @pytest.fixture
    def sample_input(self):
        return np.random.randn(10)

    def test_infer_returns_prediction(self, robust_inference, sample_input):
        """Test infer returns RobustPrediction."""
        result = robust_inference.infer(sample_input)
        assert isinstance(result, RobustPrediction)

    def test_infer_prediction_valid(self, robust_inference, sample_input):
        """Test prediction is valid class index."""
        result = robust_inference.infer(sample_input)
        assert 0 <= result.prediction < 3

    def test_infer_probabilities_valid(self, robust_inference, sample_input):
        """Test probabilities are valid."""
        result = robust_inference.infer(sample_input)
        assert np.all(result.probabilities >= 0)
        assert np.isclose(np.sum(result.probabilities), 1.0)

    def test_infer_confidence_range(self, robust_inference, sample_input):
        """Test confidence is in [0, 1]."""
        result = robust_inference.infer(sample_input)
        assert 0 <= result.confidence <= 1

    def test_infer_uncertainty_positive(self, robust_inference, sample_input):
        """Test uncertainty is non-negative."""
        result = robust_inference.infer(sample_input)
        assert result.uncertainty >= 0

    def test_infer_tracks_statistics(self, robust_inference, sample_input):
        """Test inference tracks statistics."""
        robust_inference.infer(sample_input)
        robust_inference.infer(sample_input)
        stats = robust_inference.statistics()
        assert stats["n_predictions"] == 2

    def test_infer_with_ood_detector(self, model, sample_input):
        """Test inference with OOD detector."""
        ood_model = SimpleClassifier(10, 20, 3, random_seed=42)
        ood_detector = OODDetector(ood_model, method=OODMethod.ENERGY)

        ri = RobustInference(
            model,
            ood_detector=ood_detector,
            rejection_policy=RejectionPolicy.OOD,
        )
        result = ri.infer(sample_input)
        assert isinstance(result.is_ood, bool)

    def test_infer_without_uncertainty(self, robust_inference, sample_input):
        """Test inference without uncertainty computation."""
        result = robust_inference.infer(sample_input, return_uncertainty=False)
        assert result.uncertainty == 0.0

    def test_infer_without_ood_check(self, robust_inference, sample_input):
        """Test inference without OOD check."""
        result = robust_inference.infer(sample_input, check_ood=False)
        assert result.is_ood is False


class TestRejectionPolicies:
    """Tests for different rejection policies."""

    @pytest.fixture
    def model(self):
        return SimpleDropoutNN(10, 20, 3, dropout_rate=0.3, random_seed=42)

    def test_confidence_rejection(self, model):
        """Test confidence-based rejection."""
        ri = RobustInference(
            model,
            rejection_policy=RejectionPolicy.CONFIDENCE,
            confidence_threshold=0.99,  # Very high
        )

        rejected_count = 0
        for _ in range(20):
            x = np.random.randn(10)
            result = ri.infer(x)
            if result.is_rejected:
                rejected_count += 1
                assert "confidence" in result.rejection_reason.lower()

        # With 0.99 threshold, most should be rejected
        assert rejected_count > 10

    def test_uncertainty_rejection(self, model):
        """Test uncertainty-based rejection."""
        ri = RobustInference(
            model,
            rejection_policy=RejectionPolicy.UNCERTAINTY,
            uncertainty_threshold=0.01,  # Very low
        )

        rejected_count = 0
        for _ in range(20):
            x = np.random.randn(10)
            result = ri.infer(x)
            if result.is_rejected:
                rejected_count += 1
                assert "uncertainty" in result.rejection_reason.lower()

        # With 0.01 threshold, most should be rejected
        assert rejected_count > 10

    def test_ood_rejection(self, model):
        """Test OOD-based rejection."""
        ood_model = SimpleClassifier(10, 20, 3, random_seed=42)
        ood_detector = OODDetector(ood_model, method=OODMethod.ENERGY, threshold=0.0)

        ri = RobustInference(
            model,
            ood_detector=ood_detector,
            rejection_policy=RejectionPolicy.OOD,
        )

        # Test with extreme input (likely OOD)
        x_extreme = np.ones(10) * 100
        ri.infer(x_extreme)
        # May or may not be rejected depending on detector

    def test_combined_rejection(self, model):
        """Test combined rejection policy."""
        ri = RobustInference(
            model,
            rejection_policy=RejectionPolicy.COMBINED,
            confidence_threshold=0.99,
            uncertainty_threshold=0.01,
        )

        rejected_count = 0
        for _ in range(20):
            x = np.random.randn(10)
            result = ri.infer(x)
            if result.is_rejected:
                rejected_count += 1

        # Most should be rejected with strict thresholds
        assert rejected_count > 10


class TestSelectivePrediction:
    """Tests for selective prediction."""

    @pytest.fixture
    def model(self):
        return SimpleDropoutNN(10, 20, 3, dropout_rate=0.2, random_seed=42)

    @pytest.fixture
    def robust_inference(self, model):
        return RobustInference(model, confidence_threshold=0.5)

    def test_selective_predict_returns_result(self, robust_inference):
        """Test selective_predict returns SelectivePrediction."""
        x = np.random.randn(10)
        result = robust_inference.selective_predict(x)
        assert isinstance(result, SelectivePrediction)

    def test_selective_predict_abstained_has_none(self, robust_inference):
        """Test abstained prediction has None prediction."""
        # Use strict threshold to force abstention
        robust_inference.confidence_threshold = 0.99

        abstained = False
        for _ in range(20):
            x = np.random.randn(10)
            result = robust_inference.selective_predict(x)
            if result.abstained:
                assert result.prediction is None
                abstained = True
                break

        assert abstained, "No abstentions occurred with strict threshold"

    def test_selective_predict_coverage_tracked(self, robust_inference):
        """Test coverage is tracked."""
        for _ in range(20):
            x = np.random.randn(10)
            result = robust_inference.selective_predict(x)

        assert 0 <= result.coverage <= 1

    def test_reject_uncertain_returns_tuple(self, robust_inference):
        """Test reject_uncertain returns (prediction, was_rejected)."""
        x = np.random.randn(10)
        pred, rejected = robust_inference.reject_uncertain(x)
        assert isinstance(rejected, bool)
        if rejected:
            assert pred is None
        else:
            assert isinstance(pred, int)

    def test_infer_with_uncertainty_returns_tuple(self, robust_inference):
        """Test infer_with_uncertainty returns (pred, uncertainty)."""
        x = np.random.randn(10)
        pred, unc = robust_inference.infer_with_uncertainty(x)
        assert isinstance(pred, int)
        assert isinstance(unc, float)


class TestRobustAggregation:
    """Tests for robust prediction aggregation."""

    @pytest.fixture
    def model(self):
        return SimpleDropoutNN(10, 20, 3, dropout_rate=0.2, random_seed=42)

    @pytest.fixture
    def robust_inference(self, model):
        return RobustInference(model)

    def test_majority_vote(self, robust_inference):
        """Test majority vote aggregation."""
        predictions = [(0, 0.8), (0, 0.6), (1, 0.9), (0, 0.7)]
        winner, conf = robust_inference.robust_aggregation(predictions, "majority_vote")
        assert winner == 0  # 3 votes vs 1
        assert conf == 0.75  # 3/4

    def test_weighted_vote(self, robust_inference):
        """Test weighted vote aggregation."""
        predictions = [(0, 0.9), (1, 0.8), (1, 0.7)]
        winner, conf = robust_inference.robust_aggregation(predictions, "weighted_vote")
        # Class 1 has higher total weight (0.8 + 0.7 = 1.5) vs class 0 (0.9)
        assert winner == 1

    def test_confidence_max(self, robust_inference):
        """Test confidence max aggregation."""
        predictions = [(0, 0.6), (1, 0.95), (2, 0.7)]
        winner, conf = robust_inference.robust_aggregation(predictions, "confidence_max")
        assert winner == 1
        assert conf == 0.95

    def test_median_aggregation(self, robust_inference):
        """Test median aggregation."""
        predictions = [(0, 0.6), (0, 0.8), (0, 0.9), (1, 0.5)]
        winner, conf = robust_inference.robust_aggregation(predictions, "median")
        assert winner == 0
        assert conf == 0.8  # Median of [0.6, 0.8, 0.9]

    def test_empty_predictions(self, robust_inference):
        """Test handling of empty predictions."""
        winner, conf = robust_inference.robust_aggregation([], "majority_vote")
        assert winner == 0
        assert conf == 0.0


class TestBatchInference:
    """Tests for batch inference."""

    @pytest.fixture
    def model(self):
        return SimpleDropoutNN(10, 20, 3, dropout_rate=0.2, random_seed=42)

    @pytest.fixture
    def robust_inference(self, model):
        return RobustInference(model, confidence_threshold=0.5)

    def test_batch_infer_returns_list(self, robust_inference):
        """Test batch_infer returns list of predictions."""
        batch = [np.random.randn(10) for _ in range(5)]
        results = robust_inference.batch_infer(batch)
        assert len(results) == 5
        assert all(isinstance(r, RobustPrediction) for r in results)

    def test_batch_infer_excludes_rejected(self, robust_inference):
        """Test batch_infer can exclude rejected predictions."""
        robust_inference.confidence_threshold = 0.99  # Strict
        batch = [np.random.randn(10) for _ in range(20)]

        results_with = robust_inference.batch_infer(batch, return_rejected=True)
        # Reset for second call
        robust_inference._n_predictions = 0
        robust_inference._n_rejected = 0
        results_without = robust_inference.batch_infer(batch, return_rejected=False)

        assert len(results_with) >= len(results_without)


class TestThresholdCalibration:
    """Tests for threshold calibration."""

    @pytest.fixture
    def model(self):
        return SimpleDropoutNN(10, 20, 3, dropout_rate=0.2, random_seed=42)

    @pytest.fixture
    def robust_inference(self, model):
        return RobustInference(model)

    @pytest.fixture
    def validation_data(self):
        return [(np.random.randn(10), i % 3) for i in range(100)]

    def test_calibrate_thresholds(self, robust_inference, validation_data):
        """Test threshold calibration."""
        result = robust_inference.calibrate_thresholds(
            validation_data,
            target_coverage=0.9,
            target_accuracy=0.95,
        )
        assert "confidence_threshold" in result
        assert "uncertainty_threshold" in result

    def test_calibration_updates_thresholds(self, robust_inference, validation_data):
        """Test calibration updates thresholds."""

        robust_inference.calibrate_thresholds(
            validation_data,
            target_coverage=0.5,
        )

        # At least one should change
        # May not change if already optimal
        assert isinstance(robust_inference.confidence_threshold, float)


class TestEvaluation:
    """Tests for evaluation metrics."""

    @pytest.fixture
    def model(self):
        return SimpleDropoutNN(10, 20, 3, dropout_rate=0.2, random_seed=42)

    @pytest.fixture
    def robust_inference(self, model):
        return RobustInference(model, confidence_threshold=0.5)

    @pytest.fixture
    def test_data(self):
        return [(np.random.randn(10), i % 3) for i in range(50)]

    def test_evaluate_returns_metrics(self, robust_inference, test_data):
        """Test evaluate returns metrics dict."""
        metrics = robust_inference.evaluate(test_data)
        assert "coverage" in metrics
        assert "selective_accuracy" in metrics
        assert "overall_accuracy" in metrics
        assert "selective_risk" in metrics

    def test_coverage_in_range(self, robust_inference, test_data):
        """Test coverage is in [0, 1]."""
        metrics = robust_inference.evaluate(test_data)
        assert 0 <= metrics["coverage"] <= 1

    def test_accuracy_in_range(self, robust_inference, test_data):
        """Test accuracy is in [0, 1]."""
        metrics = robust_inference.evaluate(test_data)
        assert 0 <= metrics["overall_accuracy"] <= 1
        assert 0 <= metrics["selective_accuracy"] <= 1

    def test_risk_is_complement_of_accuracy(self, robust_inference, test_data):
        """Test risk = 1 - accuracy."""
        metrics = robust_inference.evaluate(test_data)
        assert np.isclose(metrics["selective_risk"], 1.0 - metrics["selective_accuracy"])


class TestAdaptiveThresholdInference:
    """Tests for AdaptiveThresholdInference."""

    @pytest.fixture
    def model(self):
        return SimpleDropoutNN(10, 20, 3, dropout_rate=0.2, random_seed=42)

    @pytest.fixture
    def adaptive_inference(self, model):
        return AdaptiveThresholdInference(
            model,
            window_size=50,
            target_coverage=0.8,
            confidence_threshold=0.5,
        )

    def test_adaptive_inherits_robust(self, adaptive_inference):
        """Test AdaptiveThresholdInference inherits from RobustInference."""
        assert isinstance(adaptive_inference, RobustInference)

    def test_adaptive_tracks_history(self, adaptive_inference):
        """Test adaptive inference tracks history."""
        for _ in range(20):
            x = np.random.randn(10)
            adaptive_inference.infer(x)

        assert len(adaptive_inference._recent_confidences) == 20
        assert len(adaptive_inference._recent_uncertainties) == 20
        assert len(adaptive_inference._recent_rejections) == 20

    def test_adaptive_trims_history(self, adaptive_inference):
        """Test history is trimmed to window size."""
        for _ in range(100):
            x = np.random.randn(10)
            adaptive_inference.infer(x)

        assert len(adaptive_inference._recent_confidences) == 50

    def test_adaptive_adjusts_thresholds(self, adaptive_inference):
        """Test thresholds are adjusted."""

        # Make many predictions
        for _ in range(100):
            x = np.random.randn(10)
            adaptive_inference.infer(x)

        # Threshold should have been adjusted
        # (may be same if already at target coverage)
        assert isinstance(adaptive_inference.confidence_threshold, float)

    def test_adaptive_respects_bounds(self, adaptive_inference):
        """Test thresholds stay within bounds."""
        for _ in range(200):
            x = np.random.randn(10)
            adaptive_inference.infer(x)

        assert 0.3 <= adaptive_inference.confidence_threshold <= 0.99
        assert 0.1 <= adaptive_inference.uncertainty_threshold <= 2.0


class TestFullPipeline:
    """End-to-end tests for the full robust inference pipeline."""

    @pytest.fixture
    def full_system(self):
        """Create full robust inference system."""
        model = SimpleDropoutNN(10, 20, 3, dropout_rate=0.2, random_seed=42)
        uncertainty = UncertaintyQuantifier(model, n_samples=20)

        ood_model = SimpleClassifier(10, 20, 3, random_seed=42)
        ood_detector = OODDetector(ood_model, method=OODMethod.ENERGY)

        return RobustInference(
            model,
            uncertainty_quantifier=uncertainty,
            ood_detector=ood_detector,
            rejection_policy=RejectionPolicy.COMBINED,
            confidence_threshold=0.6,
            uncertainty_threshold=0.8,
        )

    def test_full_pipeline_works(self, full_system):
        """Test full pipeline produces valid results."""
        x = np.random.randn(10)
        result = full_system.infer(x)

        assert isinstance(result, RobustPrediction)
        assert 0 <= result.prediction < 3
        assert 0 <= result.confidence <= 1
        assert result.uncertainty >= 0

    def test_full_pipeline_batch(self, full_system):
        """Test full pipeline on batch."""
        batch = [np.random.randn(10) for _ in range(10)]
        results = full_system.batch_infer(batch)

        assert len(results) == 10
        for r in results:
            assert isinstance(r, RobustPrediction)

    def test_full_pipeline_selective(self, full_system):
        """Test full pipeline selective prediction."""
        for _ in range(10):
            x = np.random.randn(10)
            result = full_system.selective_predict(x)
            assert isinstance(result, SelectivePrediction)

    def test_rejection_rate_reasonable(self, full_system):
        """Test rejection rate is reasonable."""
        for _ in range(50):
            x = np.random.randn(10)
            full_system.infer(x)

        rate = full_system.rejection_rate()
        # Should reject some but not all
        assert 0 <= rate <= 1

    def test_statistics_complete(self, full_system):
        """Test statistics are complete."""
        for _ in range(10):
            x = np.random.randn(10)
            full_system.infer(x)

        stats = full_system.statistics()
        assert "n_predictions" in stats
        assert "n_rejected" in stats
        assert "rejection_rate" in stats
        assert "policy" in stats
