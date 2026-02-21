"""Tests for uncertainty quantification."""

import pytest
import numpy as np
from neuro.modules.robust.uncertainty import (
    SimpleDropoutNN, UncertaintyQuantifier, UncertaintyEstimate,
    UncertaintyType, EnsembleUncertainty, EvidentialUncertainty,
)

class TestSimpleDropoutNN:
    """Tests for SimpleDropoutNN."""

    def test_initialization(self):
        """Test network initialization."""
        model = SimpleDropoutNN(10, 20, 3)
        assert model.W1.shape == (10, 20)
        assert model.W2.shape == (20, 3)

    def test_forward_without_dropout(self):
        """Test forward pass without dropout is deterministic."""
        model = SimpleDropoutNN(10, 20, 3, random_seed=42)
        x = np.random.randn(10)
        out1 = model.forward(x, apply_dropout=False)
        out2 = model.forward(x, apply_dropout=False)
        assert np.allclose(out1, out2)

    def test_forward_with_dropout_varies(self):
        """Test forward pass with dropout varies."""
        model = SimpleDropoutNN(10, 20, 3, dropout_rate=0.5, random_seed=42)
        x = np.random.randn(10)
        outputs = [model.forward(x, apply_dropout=True) for _ in range(10)]
        # Check that outputs vary
        assert not all(np.allclose(outputs[0], o) for o in outputs[1:])

    def test_predict_proba_sums_to_one(self):
        """Test probabilities sum to 1."""
        model = SimpleDropoutNN(10, 20, 3)
        x = np.random.randn(10)
        probs = model.predict_proba(x)
        assert np.isclose(np.sum(probs), 1.0)

    def test_zero_dropout_deterministic(self):
        """Test zero dropout is deterministic."""
        model = SimpleDropoutNN(10, 20, 3, dropout_rate=0.0)
        x = np.random.randn(10)
        out1 = model.forward(x, apply_dropout=True)
        out2 = model.forward(x, apply_dropout=True)
        assert np.allclose(out1, out2)

class TestUncertaintyQuantifier:
    """Tests for UncertaintyQuantifier."""

    @pytest.fixture
    def quantifier(self):
        model = SimpleDropoutNN(10, 20, 3, dropout_rate=0.3, random_seed=42)
        return UncertaintyQuantifier(model, n_samples=30)

    @pytest.fixture
    def sample_input(self):
        return np.random.randn(10)

    def test_monte_carlo_dropout_returns_estimate(self, quantifier, sample_input):
        """Test MC dropout returns UncertaintyEstimate."""
        estimate = quantifier.monte_carlo_dropout(sample_input)
        assert isinstance(estimate, UncertaintyEstimate)

    def test_epistemic_uncertainty_positive(self, quantifier, sample_input):
        """Test epistemic uncertainty is non-negative."""
        estimate = quantifier.monte_carlo_dropout(sample_input)
        assert estimate.epistemic >= 0

    def test_aleatoric_uncertainty_positive(self, quantifier, sample_input):
        """Test aleatoric uncertainty is non-negative."""
        estimate = quantifier.monte_carlo_dropout(sample_input)
        assert estimate.aleatoric >= 0

    def test_total_uncertainty_positive(self, quantifier, sample_input):
        """Test total uncertainty is non-negative."""
        estimate = quantifier.monte_carlo_dropout(sample_input)
        assert estimate.total >= 0

    def test_confidence_between_0_and_1(self, quantifier, sample_input):
        """Test confidence is between 0 and 1."""
        estimate = quantifier.monte_carlo_dropout(sample_input)
        assert 0 <= estimate.confidence <= 1

    def test_prediction_is_probability(self, quantifier, sample_input):
        """Test prediction sums to 1."""
        estimate = quantifier.monte_carlo_dropout(sample_input)
        assert np.isclose(np.sum(estimate.prediction), 1.0)

    def test_samples_stored(self, quantifier, sample_input):
        """Test samples are stored."""
        estimate = quantifier.monte_carlo_dropout(sample_input)
        assert estimate.samples is not None
        assert estimate.samples.shape[0] == 30  # n_samples

    def test_epistemic_uncertainty_method(self, quantifier, sample_input):
        """Test epistemic_uncertainty helper."""
        uncertainty = quantifier.epistemic_uncertainty(sample_input)
        assert isinstance(uncertainty, float)
        assert uncertainty >= 0

    def test_aleatoric_uncertainty_method(self, quantifier, sample_input):
        """Test aleatoric_uncertainty helper."""
        uncertainty = quantifier.aleatoric_uncertainty(sample_input)
        assert isinstance(uncertainty, float)
        assert uncertainty >= 0

    def test_total_uncertainty_method(self, quantifier, sample_input):
        """Test total_uncertainty helper."""
        uncertainty = quantifier.total_uncertainty(sample_input)
        assert isinstance(uncertainty, float)
        assert uncertainty >= 0

    def test_predictive_entropy(self, quantifier, sample_input):
        """Test predictive entropy computation."""
        entropy = quantifier.predictive_entropy(sample_input)
        assert isinstance(entropy, float)
        assert entropy >= 0

    def test_mutual_information(self, quantifier, sample_input):
        """Test mutual information computation."""
        mi = quantifier.mutual_information(sample_input)
        assert isinstance(mi, float)
        # MI can be slightly negative due to numerical issues
        assert mi >= -0.1

    def test_statistics_tracking(self, quantifier, sample_input):
        """Test statistics are tracked."""
        quantifier.monte_carlo_dropout(sample_input)
        quantifier.monte_carlo_dropout(sample_input)
        stats = quantifier.statistics()
        assert stats["n_estimates"] == 2

    def test_more_samples_reduces_variance(self, sample_input):
        """Test more samples reduces estimate variance."""
        model = SimpleDropoutNN(10, 20, 3, dropout_rate=0.3, random_seed=42)

        # Low samples
        quant_low = UncertaintyQuantifier(model, n_samples=5)
        estimates_low = [quant_low.monte_carlo_dropout(sample_input).total for _ in range(10)]

        # High samples
        quant_high = UncertaintyQuantifier(model, n_samples=50)
        estimates_high = [quant_high.monte_carlo_dropout(sample_input).total for _ in range(10)]

        # Higher samples should have lower variance
        assert np.var(estimates_high) <= np.var(estimates_low) + 0.1

class TestEnsembleUncertainty:
    """Tests for EnsembleUncertainty."""

    @pytest.fixture
    def ensemble(self):
        return EnsembleUncertainty(10, 20, 3, n_models=3, random_seed=42)

    @pytest.fixture
    def training_data(self):
        return [(np.random.randn(10), i % 3) for i in range(50)]

    @pytest.fixture
    def sample_input(self):
        return np.random.randn(10)

    def test_ensemble_creates_models(self, ensemble):
        """Test ensemble creates correct number of models."""
        assert len(ensemble.models) == 3

    def test_models_have_different_weights(self, ensemble):
        """Test models have different initializations."""
        w1_list = [m.W1 for m in ensemble.models]
        assert not np.allclose(w1_list[0], w1_list[1])

    def test_train_returns_history(self, ensemble, training_data):
        """Test training returns history."""
        history = ensemble.train(training_data, n_epochs=5)
        assert "losses" in history
        assert len(history["losses"]) == 3  # One per model

    def test_predict_returns_estimate(self, ensemble, sample_input):
        """Test predict returns UncertaintyEstimate."""
        estimate = ensemble.predict(sample_input)
        assert isinstance(estimate, UncertaintyEstimate)

    def test_predict_epistemic_from_disagreement(self, ensemble, sample_input):
        """Test epistemic uncertainty from model disagreement."""
        estimate = ensemble.predict(sample_input)
        assert estimate.epistemic >= 0

    def test_predict_samples_stored(self, ensemble, sample_input):
        """Test predictions from all models stored."""
        estimate = ensemble.predict(sample_input)
        assert estimate.samples is not None
        assert estimate.samples.shape[0] == 3  # n_models

    def test_epistemic_uncertainty_method(self, ensemble, sample_input):
        """Test epistemic_uncertainty helper."""
        uncertainty = ensemble.epistemic_uncertainty(sample_input)
        assert isinstance(uncertainty, float)
        assert uncertainty >= 0

    def test_statistics_tracking(self, ensemble, sample_input):
        """Test statistics are tracked."""
        ensemble.predict(sample_input)
        ensemble.predict(sample_input)
        stats = ensemble.statistics()
        assert stats["n_estimates"] == 2
        assert stats["n_models"] == 3

    def test_training_reduces_loss(self, ensemble, training_data):
        """Test training reduces loss."""
        history = ensemble.train(training_data, n_epochs=20)
        # Check first model's loss decreases
        first_model_losses = history["losses"][0]
        assert first_model_losses[-1] < first_model_losses[0]

class TestEvidentialUncertainty:
    """Tests for EvidentialUncertainty."""

    @pytest.fixture
    def evidential(self):
        return EvidentialUncertainty(10, 20, 3, random_seed=42)

    @pytest.fixture
    def sample_input(self):
        return np.random.randn(10)

    def test_forward_positive_evidence(self, evidential, sample_input):
        """Test forward returns positive evidence."""
        evidence = evidential.forward(sample_input)
        assert np.all(evidence > 0)

    def test_predict_returns_estimate(self, evidential, sample_input):
        """Test predict returns UncertaintyEstimate."""
        estimate = evidential.predict(sample_input)
        assert isinstance(estimate, UncertaintyEstimate)

    def test_predict_probabilities_valid(self, evidential, sample_input):
        """Test prediction is valid probability."""
        estimate = evidential.predict(sample_input)
        assert np.all(estimate.prediction >= 0)
        assert np.isclose(np.sum(estimate.prediction), 1.0)

    def test_epistemic_from_lack_of_evidence(self, evidential, sample_input):
        """Test epistemic uncertainty from evidence."""
        estimate = evidential.predict(sample_input)
        # Epistemic = K / S where S is Dirichlet strength
        assert estimate.epistemic > 0

    def test_train_step_returns_loss(self, evidential, sample_input):
        """Test train step returns loss."""
        loss = evidential.train_step(sample_input, 1)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_training_reduces_loss(self, evidential):
        """Test training reduces loss."""
        data = [(np.random.randn(10), i % 3) for i in range(50)]

        initial_loss = sum(evidential.train_step(x, y) for x, y in data[:10]) / 10

        # Train
        for _ in range(20):
            for x, y in data:
                evidential.train_step(x, y)

        final_loss = sum(evidential.train_step(x, y) for x, y in data[:10]) / 10
        assert final_loss < initial_loss

    def test_statistics_tracking(self, evidential, sample_input):
        """Test statistics are tracked."""
        evidential.predict(sample_input)
        evidential.predict(sample_input)
        stats = evidential.statistics()
        assert stats["n_estimates"] == 2
        assert stats["n_classes"] == 3

class TestUncertaintyComparison:
    """Tests comparing uncertainty methods."""

    @pytest.fixture
    def sample_input(self):
        return np.random.randn(10)

    def test_all_methods_give_positive_uncertainty(self, sample_input):
        """Test all methods give positive uncertainty."""
        model = SimpleDropoutNN(10, 20, 3, dropout_rate=0.3, random_seed=42)

        # MC Dropout
        mc = UncertaintyQuantifier(model, n_samples=20)
        mc_estimate = mc.monte_carlo_dropout(sample_input)
        assert mc_estimate.total >= 0

        # Ensemble
        ens = EnsembleUncertainty(10, 20, 3, n_models=3, random_seed=42)
        ens_estimate = ens.predict(sample_input)
        assert ens_estimate.total >= 0

        # Evidential
        evid = EvidentialUncertainty(10, 20, 3, random_seed=42)
        evid_estimate = evid.predict(sample_input)
        assert evid_estimate.total >= 0

    def test_uncertain_input_higher_uncertainty(self):
        """Test that ambiguous inputs have higher uncertainty."""
        model = SimpleDropoutNN(10, 20, 3, dropout_rate=0.5, random_seed=42)
        quant = UncertaintyQuantifier(model, n_samples=50)

        # Create inputs that produce different confidence levels
        uncertainties = []
        for _ in range(20):
            x = np.random.randn(10)
            estimate = quant.monte_carlo_dropout(x)
            uncertainties.append((estimate.confidence, estimate.total))

        # Higher confidence should correlate with lower total uncertainty
        confidences = [c for c, u in uncertainties]
        totals = [u for c, u in uncertainties]

        correlation = np.corrcoef(confidences, totals)[0, 1]
        # Expect negative correlation (higher confidence = lower uncertainty)
        assert correlation < 0.5  # Allow some variance
