"""Tests for Hebbian learning"""

import numpy as np
import pytest
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from src.hebbian import (
    HebbianLearning, OjaRule, BCMRule, HebbianNetwork, LearningParams
)


class TestHebbianLearning:
    """Tests for basic Hebbian learning"""

    def test_initialization(self):
        """Test Hebbian rule initialization"""
        rule = HebbianLearning()
        assert rule.params.learning_rate == 0.01
        assert rule.params.weight_max == 1.0

    def test_weight_change_shape(self):
        """Test weight change has correct shape"""
        rule = HebbianLearning()
        pre = np.random.rand(10)
        post = np.random.rand(5)

        dW = rule.compute_weight_change(pre, post)
        assert dW.shape == (5, 10)

    def test_weight_change_correlation(self):
        """Test that correlated activity increases weights"""
        rule = HebbianLearning(LearningParams(learning_rate=0.1))

        # Correlated activity
        pre = np.array([1.0, 0.0, 1.0])
        post = np.array([1.0, 0.0])

        dW = rule.compute_weight_change(pre, post)

        # Weight between active pre and post should increase
        assert dW[0, 0] > 0  # Both active
        assert dW[0, 1] == 0  # Pre inactive
        assert dW[1, 0] == 0  # Post inactive

    def test_weight_update_bounded(self):
        """Test weights stay within bounds"""
        params = LearningParams(weight_max=1.0, weight_min=0.0, learning_rate=0.5)
        rule = HebbianLearning(params)

        weights = np.ones((3, 3)) * 0.9
        pre = np.ones(3)
        post = np.ones(3)

        # Multiple updates should not exceed bounds
        for _ in range(100):
            weights = rule.update_weights(weights, pre, post)

        assert np.all(weights <= 1.0)
        assert np.all(weights >= 0.0)


class TestOjaRule:
    """Tests for Oja's normalized Hebbian rule"""

    def test_oja_prevents_unbounded_growth(self):
        """Test Oja's rule prevents weight explosion"""
        rule = OjaRule(LearningParams(learning_rate=0.1))

        weights = np.random.rand(3, 5) * 0.5
        pre = np.random.rand(5)
        post = np.random.rand(3)

        # Many updates
        for _ in range(1000):
            weights = rule.update_weights(weights, pre, post)

        # Weights should be bounded
        assert np.all(np.isfinite(weights))
        assert np.all(np.abs(weights) < 10)

    def test_oja_weight_normalization(self):
        """Test that Oja's rule leads to normalized weights"""
        rule = OjaRule(LearningParams(learning_rate=0.01))

        weights = np.random.rand(1, 10)
        pre = np.random.rand(10)

        for _ in range(500):
            post = np.tanh(weights @ pre)
            weights = rule.update_weights(weights, pre, post.flatten())

        # Weight vector should have reasonable magnitude
        assert np.linalg.norm(weights) < 10


class TestBCMRule:
    """Tests for BCM learning rule"""

    def test_bcm_threshold_update(self):
        """Test sliding threshold updates with activity"""
        rule = BCMRule()

        # High activity should raise threshold
        high_activity = np.ones(5) * 0.8
        for _ in range(50):
            rule.compute_weight_change(np.ones(5), high_activity)

        high_threshold = rule.threshold

        # Low activity should lower threshold
        rule2 = BCMRule()
        low_activity = np.ones(5) * 0.1
        for _ in range(50):
            rule2.compute_weight_change(np.ones(5), low_activity)

        low_threshold = rule2.threshold

        assert high_threshold > low_threshold

    def test_bcm_bidirectional(self):
        """Test BCM produces LTP and LTD"""
        rule = BCMRule()
        rule.threshold = 0.5

        pre = np.ones(3)

        # Above threshold = LTP
        high_post = np.ones(2) * 0.8
        dW_high = rule.compute_weight_change(pre, high_post)

        # Below threshold = LTD
        rule.threshold = 0.5  # Reset
        low_post = np.ones(2) * 0.3
        dW_low = rule.compute_weight_change(pre, low_post)

        assert np.mean(dW_high) > np.mean(dW_low)


class TestHebbianNetwork:
    """Tests for Hebbian network"""

    def test_network_initialization(self):
        """Test network initializes correctly"""
        net = HebbianNetwork([10, 5, 3])

        assert len(net.weights) == 2
        assert net.weights[0].shape == (5, 10)
        assert net.weights[1].shape == (3, 5)

    def test_forward_pass(self):
        """Test forward propagation"""
        net = HebbianNetwork([10, 5, 3])

        x = np.random.rand(10)
        output = net.forward(x)

        assert output.shape == (3,)
        assert len(net.activations) == 3

    def test_learning_modifies_weights(self):
        """Test that learning changes weights"""
        net = HebbianNetwork([10, 5], learning_rate=0.1)

        weights_before = net.weights[0].copy()

        x = np.random.rand(10)
        net.train_step(x)

        weights_after = net.weights[0]

        assert not np.allclose(weights_before, weights_after)

    def test_different_rules(self):
        """Test network works with different rules"""
        for rule in ['basic', 'oja', 'bcm']:
            net = HebbianNetwork([5, 3], learning_rule=rule)
            x = np.random.rand(5)
            output = net.train_step(x)
            assert output.shape == (3,)

    def test_training_produces_errors(self):
        """Test training returns error history"""
        net = HebbianNetwork([10, 5])
        data = np.random.rand(20, 10)

        errors = net.train(data, n_epochs=5)

        assert len(errors) == 5
        assert all(np.isfinite(e) for e in errors)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
