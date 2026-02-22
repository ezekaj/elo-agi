"""Tests for reward-modulated learning"""

import numpy as np
import pytest
from neuro.modules.m12_learning.reward_modulated import (
    DopamineSystem,
    RewardModulatedSTDP,
    ErrorBasedLearning,
)


class TestDopamineSystem:
    """Tests for dopamine system"""

    def test_initialization(self):
        """Test dopamine system initialization"""
        da = DopamineSystem()
        assert da.level == da.params.baseline
        assert da.expected_reward == 0.0

    def test_positive_prediction_error(self):
        """Test positive prediction error (unexpected reward)"""
        da = DopamineSystem()
        da.expected_reward = 0.5

        level = da.receive_reward(1.0)  # More than expected

        assert da.prediction_error > 0
        assert level > da.params.baseline

    def test_negative_prediction_error(self):
        """Test negative prediction error (less than expected)"""
        da = DopamineSystem()
        da.expected_reward = 1.0

        da.receive_reward(0.5)  # Less than expected

        assert da.prediction_error < 0

    def test_expected_reward_learning(self):
        """Test that expected reward updates"""
        da = DopamineSystem()

        # Repeated rewards should increase expectation
        for _ in range(20):
            da.receive_reward(1.0)

        assert da.expected_reward > 0.5

    def test_decay_to_baseline(self):
        """Test dopamine decays to baseline"""
        da = DopamineSystem()
        da.receive_reward(1.0)  # Spike

        initial_level = da.level

        for _ in range(100):
            da.update()

        # Should approach baseline
        assert abs(da.level - da.params.baseline) < abs(initial_level - da.params.baseline)

    def test_modulation_signal(self):
        """Test modulation signal for learning"""
        da = DopamineSystem()

        # Positive error = positive modulation
        da.receive_reward(1.0)
        assert da.get_modulation() > 0

        # Negative error = zero modulation (clipped)
        da.expected_reward = 2.0
        da.receive_reward(0.0)
        assert da.get_modulation() == 0


class TestRewardModulatedSTDP:
    """Tests for reward-modulated STDP"""

    def test_initialization(self):
        """Test initialization"""
        net = RewardModulatedSTDP(n_pre=10, n_post=5)

        assert net.weights.shape == (5, 10)
        assert net.eligibility_trace.shape == (5, 10)

    def test_eligibility_accumulation(self):
        """Test eligibility traces accumulate"""
        net = RewardModulatedSTDP(n_pre=3, n_post=2)

        # Initial eligibility
        assert np.allclose(net.eligibility_trace, 0)

        # Activity should build eligibility
        net.update(np.array([1, 0, 1]), np.array([1, 0]))

        assert not np.allclose(net.eligibility_trace, 0)

    def test_no_learning_without_reward(self):
        """Test weights don't change without reward"""
        net = RewardModulatedSTDP(n_pre=5, n_post=3)

        weights_before = net.weights.copy()

        # Activity without reward
        for _ in range(10):
            net.update(np.random.rand(5) > 0.5, np.random.rand(3) > 0.5)

        # Weights unchanged (only eligibility changed)
        assert np.allclose(weights_before, net.weights)

    def test_learning_with_reward(self):
        """Test weights change with reward"""
        net = RewardModulatedSTDP(n_pre=5, n_post=3)

        # Build eligibility
        for _ in range(10):
            net.update(np.random.rand(5) > 0.3, np.random.rand(3) > 0.3)

        weights_before = net.weights.copy()

        # Apply reward
        net.receive_reward(1.0)

        assert not np.allclose(weights_before, net.weights)

    def test_reward_direction(self):
        """Test positive reward strengthens eligible synapses"""
        net = RewardModulatedSTDP(n_pre=3, n_post=2)

        # Create specific eligibility pattern
        net.update(np.array([1, 0, 0]), np.array([0, 0]))
        net.update(np.array([0, 0, 0]), np.array([1, 0]))

        # This should create positive eligibility for [0,0]
        elig_before = net.eligibility_trace[0, 0]
        assert elig_before > 0  # Should have LTP eligibility

        weight_before = net.weights[0, 0]
        net.receive_reward(1.0)

        # Weight should increase
        assert net.weights[0, 0] >= weight_before

    def test_episode_training(self):
        """Test episode-based training"""
        net = RewardModulatedSTDP(n_pre=5, n_post=3)

        pre_seq = (np.random.rand(20, 5) > 0.5).astype(float)
        post_seq = (np.random.rand(20, 3) > 0.5).astype(float)

        w_change = net.train_episode(pre_seq, post_seq, reward=1.0)

        assert w_change > 0


class TestErrorBasedLearning:
    """Tests for error-based learning"""

    def test_initialization(self):
        """Test initialization"""
        net = ErrorBasedLearning(n_input=10, n_output=3)

        assert net.weights.shape == (3, 10)

    def test_forward_pass(self):
        """Test forward computation"""
        net = ErrorBasedLearning(n_input=5, n_output=2)

        x = np.random.rand(5)
        output = net.forward(x)

        assert output.shape == (2,)
        assert np.all(np.abs(output) <= 1)  # tanh bounds

    def test_weight_update(self):
        """Test weight update reduces error"""
        net = ErrorBasedLearning(n_input=5, n_output=2, learning_rate=0.1)

        x = np.random.rand(5)
        target = np.array([0.5, -0.5])

        error1 = net.update(x, target)
        error2 = net.update(x, target)

        # Error should decrease
        assert error2 <= error1 * 1.1  # Allow small variance

    def test_training(self):
        """Test training reduces error over time"""
        net = ErrorBasedLearning(n_input=5, n_output=2, learning_rate=0.1)

        inputs = np.random.rand(20, 5)
        targets = np.random.rand(20, 2) * 2 - 1  # [-1, 1]

        errors = net.train(inputs, targets, n_epochs=50)

        assert len(errors) == 50
        assert errors[-1] < errors[0]  # Error decreased


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
