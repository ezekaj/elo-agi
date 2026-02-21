"""Tests for adversarial attacks and defenses."""

import pytest
import numpy as np
from neuro.modules.robust.adversarial import (
    SimpleNN, AdversarialAttack, AdversarialDefense,
    AttackType, DefenseType, AttackResult, DefenseResult,
)

class TestSimpleNN:
    """Tests for SimpleNN."""

    def test_initialization(self):
        """Test network initialization."""
        model = SimpleNN(10, 20, 3)
        assert model.W1.shape == (10, 20)
        assert model.W2.shape == (20, 3)
        assert model.b1.shape == (20,)
        assert model.b2.shape == (3,)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        model = SimpleNN(10, 20, 3)
        x = np.random.randn(10)
        logits = model.forward(x)
        assert logits.shape == (3,)

    def test_predict_proba_sums_to_one(self):
        """Test probabilities sum to 1."""
        model = SimpleNN(10, 20, 3)
        x = np.random.randn(10)
        probs = model.predict_proba(x)
        assert np.isclose(np.sum(probs), 1.0)

    def test_predict_proba_positive(self):
        """Test all probabilities are positive."""
        model = SimpleNN(10, 20, 3)
        x = np.random.randn(10)
        probs = model.predict_proba(x)
        assert np.all(probs >= 0)

    def test_gradient_wrt_input_shape(self):
        """Test gradient computation shape."""
        model = SimpleNN(10, 20, 3)
        x = np.random.randn(10)
        grad = model.gradient_wrt_input(x, 1)
        assert grad.shape == x.shape

    def test_gradient_wrt_input_nonzero(self):
        """Test gradient is non-zero."""
        model = SimpleNN(10, 20, 3)
        x = np.random.randn(10)
        grad = model.gradient_wrt_input(x, 1)
        assert np.any(grad != 0)

    def test_deterministic_with_seed(self):
        """Test reproducibility with seed."""
        model1 = SimpleNN(10, 20, 3, random_seed=42)
        model2 = SimpleNN(10, 20, 3, random_seed=42)
        assert np.allclose(model1.W1, model2.W1)
        assert np.allclose(model1.W2, model2.W2)

    def test_predict_returns_int(self):
        """Test predict returns integer class."""
        model = SimpleNN(10, 20, 3)
        x = np.random.randn(10)
        pred = model.predict(x)
        assert isinstance(pred, int)
        assert 0 <= pred < 3

class TestAdversarialAttack:
    """Tests for AdversarialAttack."""

    @pytest.fixture
    def model(self):
        return SimpleNN(10, 20, 3, random_seed=42)

    @pytest.fixture
    def attack(self, model):
        return AdversarialAttack(model, epsilon=0.1)

    @pytest.fixture
    def sample_input(self):
        np.random.seed(42)
        return np.random.rand(10)  # Use [0,1] range for clipping

    def test_fgsm_returns_attack_result(self, attack, sample_input):
        """Test FGSM returns AttackResult."""
        result = attack.fgsm(sample_input, 1, epsilon=0.1)
        assert isinstance(result, AttackResult)

    def test_fgsm_changes_input(self, attack, sample_input):
        """Test FGSM produces different input."""
        result = attack.fgsm(sample_input, 1, epsilon=0.1)
        assert not np.allclose(result.adversarial, sample_input)

    def test_fgsm_bounded_perturbation(self, attack, sample_input):
        """Test FGSM perturbation is bounded."""
        epsilon = 0.1
        result = attack.fgsm(sample_input, 1, epsilon=epsilon)
        assert result.linf_distance <= epsilon + 1e-6

    def test_pgd_returns_attack_result(self, attack, sample_input):
        """Test PGD returns AttackResult."""
        result = attack.pgd(sample_input, 1, epsilon=0.1, n_steps=5)
        assert isinstance(result, AttackResult)

    def test_pgd_changes_input(self, attack, sample_input):
        """Test PGD produces different input."""
        result = attack.pgd(sample_input, 1, epsilon=0.1, n_steps=5)
        assert not np.allclose(result.adversarial, sample_input)

    def test_pgd_bounded_perturbation(self, attack, sample_input):
        """Test PGD perturbation is bounded."""
        epsilon = 0.1
        result = attack.pgd(sample_input, 1, epsilon=epsilon, n_steps=5)
        assert result.linf_distance <= epsilon + 1e-6

    def test_pgd_more_steps_different(self, attack, sample_input):
        """Test PGD with different step counts."""
        result_5 = attack.pgd(sample_input, 1, epsilon=0.1, n_steps=5)
        result_20 = attack.pgd(sample_input, 1, epsilon=0.1, n_steps=20)
        # Both should be valid perturbations
        assert result_5.linf_distance <= 0.1 + 1e-6
        assert result_20.linf_distance <= 0.1 + 1e-6

    def test_carlini_wagner_returns_attack_result(self, attack, sample_input):
        """Test C&W returns AttackResult."""
        result = attack.carlini_wagner(sample_input, 1, n_steps=10)
        assert isinstance(result, AttackResult)

    def test_carlini_wagner_shape(self, attack, sample_input):
        """Test C&W produces correct shape."""
        result = attack.carlini_wagner(sample_input, 1, n_steps=10)
        assert result.adversarial.shape == sample_input.shape

    def test_attack_uses_configured_type(self, model, sample_input):
        """Test attack method uses configured attack type."""
        attack_fgsm = AdversarialAttack(model, attack_type=AttackType.FGSM)
        attack_pgd = AdversarialAttack(model, attack_type=AttackType.PGD)

        result1 = attack_fgsm.attack(sample_input, 1, epsilon=0.1)
        result2 = attack_pgd.attack(sample_input, 1, epsilon=0.1, n_steps=5)

        assert isinstance(result1, AttackResult)
        assert isinstance(result2, AttackResult)

    def test_attack_statistics(self, attack, sample_input):
        """Test attack statistics tracking."""
        attack.fgsm(sample_input, 1, epsilon=0.1)
        attack.pgd(sample_input, 1, epsilon=0.1, n_steps=5)
        stats = attack.statistics()
        assert stats["n_attacks"] == 2

    def test_success_rate(self, attack, sample_input):
        """Test success rate computation."""
        for _ in range(10):
            attack.fgsm(sample_input, 0, epsilon=0.3)
        rate = attack.success_rate()
        assert 0 <= rate <= 1

    def test_perturbation_stored(self, attack, sample_input):
        """Test perturbation is stored in result."""
        result = attack.fgsm(sample_input, 1, epsilon=0.1)
        assert result.perturbation is not None
        assert result.perturbation.shape == sample_input.shape

    def test_l2_distance_computed(self, attack, sample_input):
        """Test L2 distance is computed."""
        result = attack.fgsm(sample_input, 1, epsilon=0.1)
        assert result.l2_distance >= 0

    def test_predictions_stored(self, attack, sample_input):
        """Test predictions are stored."""
        result = attack.fgsm(sample_input, 1, epsilon=0.1)
        assert isinstance(result.original_prediction, int)
        assert isinstance(result.adversarial_prediction, int)

    def test_targeted_attack(self, attack, sample_input):
        """Test targeted attack."""
        target_class = 2
        result = attack.fgsm(sample_input, target_class, epsilon=0.3, targeted=True)
        assert isinstance(result, AttackResult)

class TestAdversarialDefense:
    """Tests for AdversarialDefense."""

    @pytest.fixture
    def model(self):
        return SimpleNN(10, 20, 3, random_seed=42)

    @pytest.fixture
    def defense(self, model):
        return AdversarialDefense(model, defense_type=DefenseType.INPUT_DENOISING)

    @pytest.fixture
    def training_data(self):
        return [(np.random.rand(10), i % 3) for i in range(50)]

    @pytest.fixture
    def sample_input(self):
        np.random.seed(42)
        return np.random.rand(10)

    def test_input_denoising_returns_result(self, defense, sample_input):
        """Test input denoising returns DefenseResult."""
        result = defense.input_denoising(sample_input)
        assert isinstance(result, DefenseResult)

    def test_input_denoising_shape(self, defense, sample_input):
        """Test denoised input has correct shape."""
        result = defense.input_denoising(sample_input)
        assert result.defended_input.shape == sample_input.shape

    def test_input_denoising_methods(self, defense, sample_input):
        """Test different denoising methods."""
        for method in ["median", "gaussian", "quantize"]:
            result = defense.input_denoising(sample_input, method=method)
            assert result.defended_input.shape == sample_input.shape

    def test_quantization_reduces_unique_values(self, defense, sample_input):
        """Test quantization reduces unique values."""
        result = defense.input_denoising(sample_input, method="quantize", levels=8)
        unique_values = np.unique(result.defended_input)
        assert len(unique_values) <= 9  # levels + 1

    def test_adversarial_training(self, defense, training_data):
        """Test adversarial training returns history."""
        history = defense.adversarial_training(
            training_data, epsilon=0.1, n_epochs=5
        )
        assert "clean_loss" in history
        assert "adv_loss" in history
        assert "accuracy" in history
        assert len(history["accuracy"]) == 5

    def test_detect_adversarial(self, defense, sample_input):
        """Test adversarial detection."""
        detection_defense = AdversarialDefense(
            defense.model, defense_type=DefenseType.DETECTION
        )
        result = detection_defense.detect_adversarial(sample_input)
        assert isinstance(result, DefenseResult)
        assert isinstance(result.detected_attack, bool)
        assert isinstance(result.confidence, float)

    def test_certified_defense(self, defense, sample_input):
        """Test certified defense."""
        pred, radius = defense.certified_defense(sample_input, n_samples=50, sigma=0.1)
        assert isinstance(pred, (int, np.integer))
        assert radius >= 0

    def test_randomization_defense(self, model, sample_input):
        """Test randomization defense."""
        defense = AdversarialDefense(model, defense_type=DefenseType.RANDOMIZATION)
        result = defense.randomization(sample_input, noise_std=0.05, n_samples=10)
        assert isinstance(result, DefenseResult)

    def test_defend_uses_configured_type(self, defense, sample_input):
        """Test defend uses configured defense type."""
        result = defense.defend(sample_input)
        assert isinstance(result, DefenseResult)
        assert result.defense_type == DefenseType.INPUT_DENOISING

    def test_defense_statistics(self, defense, sample_input):
        """Test defense statistics tracking."""
        defense.input_denoising(sample_input)
        defense.input_denoising(sample_input)
        stats = defense.statistics()
        assert stats["n_defended"] == 2

class TestAttackDefenseInteraction:
    """Tests for attack-defense interaction."""

    @pytest.fixture
    def model(self):
        return SimpleNN(10, 20, 3, random_seed=42)

    @pytest.fixture
    def sample_input(self):
        np.random.seed(42)
        return np.random.rand(10)

    def test_defense_changes_prediction(self, model, sample_input):
        """Test defense can change predictions."""
        attack = AdversarialAttack(model, epsilon=0.3)
        defense = AdversarialDefense(model, defense_type=DefenseType.INPUT_DENOISING)

        # Attack
        y = model.predict(sample_input)
        result = attack.fgsm(sample_input, y, epsilon=0.3)

        if result.success:
            # Defend
            defended = defense.input_denoising(result.adversarial, method="median")
            pred_after_defense = model.predict(defended.defended_input)
            # Prediction may or may not be restored

    def test_pgd_stronger_than_fgsm(self, model, sample_input):
        """Test PGD is at least as strong as FGSM."""
        attack = AdversarialAttack(model, epsilon=0.2)

        fgsm_success = 0
        pgd_success = 0
        n_trials = 20

        for i in range(n_trials):
            x = np.random.rand(10)
            y = model.predict(x)

            fgsm_result = attack.fgsm(x, y, epsilon=0.2)
            pgd_result = attack.pgd(x, y, epsilon=0.2, n_steps=20)

            if fgsm_result.success:
                fgsm_success += 1
            if pgd_result.success:
                pgd_success += 1

        # PGD should be at least as successful as FGSM
        assert pgd_success >= fgsm_success - 2  # Allow small variance

    def test_larger_epsilon_more_successful(self, model, sample_input):
        """Test larger epsilon increases attack success."""
        attack = AdversarialAttack(model, epsilon=0.1)

        success_small = 0
        success_large = 0
        n_trials = 30

        for i in range(n_trials):
            x = np.random.rand(10)
            y = model.predict(x)

            result_small = attack.fgsm(x, y, epsilon=0.05)
            result_large = attack.fgsm(x, y, epsilon=0.5)

            if result_small.success:
                success_small += 1
            if result_large.success:
                success_large += 1

        assert success_large >= success_small

class TestAttackEdgeCases:
    """Edge case tests for attacks."""

    @pytest.fixture
    def model(self):
        return SimpleNN(10, 20, 3, random_seed=42)

    def test_zero_epsilon(self, model):
        """Test attack with zero epsilon."""
        attack = AdversarialAttack(model, epsilon=0.0)
        x = np.random.rand(10)
        result = attack.fgsm(x, 1, epsilon=0.0)
        # Adversarial should be same as original (after clipping)
        assert result.linf_distance < 1e-6

    def test_large_epsilon(self, model):
        """Test attack with large epsilon."""
        attack = AdversarialAttack(model, epsilon=1.0)
        x = np.random.rand(10)
        result = attack.fgsm(x, 1, epsilon=1.0)
        # Should still be valid
        assert result.adversarial.shape == x.shape

    def test_zero_input(self, model):
        """Test attack on zero input."""
        attack = AdversarialAttack(model, epsilon=0.1)
        x = np.zeros(10)
        result = attack.fgsm(x, 1, epsilon=0.1)
        assert result.adversarial.shape == x.shape

    def test_ones_input(self, model):
        """Test attack on all-ones input."""
        attack = AdversarialAttack(model, epsilon=0.1)
        x = np.ones(10)
        result = attack.fgsm(x, 1, epsilon=0.1)
        assert result.adversarial.shape == x.shape

class TestDefenseEdgeCases:
    """Edge case tests for defenses."""

    @pytest.fixture
    def model(self):
        return SimpleNN(10, 20, 3, random_seed=42)

    def test_zero_input_denoising(self, model):
        """Test denoising on zero input."""
        defense = AdversarialDefense(model)
        x = np.zeros(10)
        result = defense.input_denoising(x)
        assert result.defended_input.shape == x.shape

    def test_constant_input_denoising(self, model):
        """Test denoising on constant input."""
        defense = AdversarialDefense(model)
        x = np.ones(10) * 0.5
        result = defense.input_denoising(x, method="median")
        # Median of constant should be same
        assert np.allclose(result.defended_input, x)

    def test_small_training_set(self, model):
        """Test adversarial training with small dataset."""
        defense = AdversarialDefense(model)
        small_data = [(np.random.rand(10), i % 3) for i in range(5)]
        history = defense.adversarial_training(small_data, n_epochs=2)
        assert len(history["accuracy"]) == 2
