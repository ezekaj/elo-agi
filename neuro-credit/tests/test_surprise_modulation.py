"""Tests for surprise-modulated learning."""

import pytest
import numpy as np

from neuro.modules.credit.surprise_modulation import (
    SurpriseMetrics,
    SurpriseConfig,
    SurpriseType,
    SurpriseModulatedLearning,
)

class TestSurpriseConfig:
    """Tests for SurpriseConfig class."""

    def test_default_config(self):
        config = SurpriseConfig()
        assert config.base_learning_rate == 0.01
        assert config.surprise_scale == 1.0

    def test_custom_config(self):
        config = SurpriseConfig(
            base_learning_rate=0.1,
            consolidation_threshold=0.9,
        )
        assert config.base_learning_rate == 0.1
        assert config.consolidation_threshold == 0.9

class TestSurpriseModulatedLearning:
    """Tests for SurpriseModulatedLearning class."""

    def test_creation(self):
        sml = SurpriseModulatedLearning(random_seed=42)
        assert sml is not None

    def test_compute_surprise_scalar(self):
        sml = SurpriseModulatedLearning(random_seed=42)

        metrics = sml.compute_surprise(1.0, 1.0)
        assert metrics.surprise_value < 0.5

        metrics_high = sml.compute_surprise(1.0, 5.0)
        assert metrics_high.surprise_value > metrics.surprise_value

    def test_compute_surprise_array(self):
        sml = SurpriseModulatedLearning(random_seed=42)

        predicted = np.array([1.0, 2.0, 3.0])
        actual = np.array([1.1, 2.1, 3.1])

        metrics = sml.compute_surprise(predicted, actual)
        assert isinstance(metrics, SurpriseMetrics)

    def test_compute_surprise_with_context(self):
        sml = SurpriseModulatedLearning(random_seed=42)

        for _ in range(10):
            sml.compute_surprise(1.0, 1.1, context_key="test")

        metrics = sml.compute_surprise(1.0, 5.0, context_key="test")
        assert metrics.surprise_value > 0.5

    def test_modulated_learning_rate(self):
        sml = SurpriseModulatedLearning(random_seed=42)

        lr_low = sml.modulated_learning_rate(0.01, 0.1)
        lr_high = sml.modulated_learning_rate(0.01, 0.9)

        assert lr_high > lr_low

    def test_modulated_lr_bounds(self):
        config = SurpriseConfig(
            base_learning_rate=0.01,
            min_learning_rate=0.001,
            max_learning_rate=0.1,
        )
        sml = SurpriseModulatedLearning(config=config, random_seed=42)

        lr = sml.modulated_learning_rate(0.01, 100.0)
        assert lr <= 0.1

        lr = sml.modulated_learning_rate(0.01, -100.0)
        assert lr >= 0.001

    def test_should_consolidate(self):
        config = SurpriseConfig(consolidation_threshold=0.8)
        sml = SurpriseModulatedLearning(config=config, random_seed=42)

        assert not sml.should_consolidate(0.5)
        assert sml.should_consolidate(0.9)

    def test_get_adaptive_lr(self):
        sml = SurpriseModulatedLearning(random_seed=42)

        lr = sml.get_adaptive_lr("module1", 1.0, 1.0)
        assert isinstance(lr, float)
        assert lr > 0

    def test_update_baseline(self):
        sml = SurpriseModulatedLearning(random_seed=42)

        sml.update_baseline("test", 1.0)
        sml.update_baseline("test", 2.0)
        sml.update_baseline("test", 3.0)

        assert "test" in sml._running_mean

    def test_get_recent_surprises(self):
        sml = SurpriseModulatedLearning(random_seed=42)

        for i in range(15):
            sml.compute_surprise(float(i), float(i + 1))

        recent = sml.get_recent_surprises(n=10)
        assert len(recent) == 10

    def test_get_recent_surprises_by_type(self):
        sml = SurpriseModulatedLearning(random_seed=42)

        sml.compute_surprise(1.0, 2.0, surprise_type=SurpriseType.REWARD)
        sml.compute_surprise(1.0, 2.0, surprise_type=SurpriseType.STATE)
        sml.compute_surprise(1.0, 2.0, surprise_type=SurpriseType.REWARD)

        reward_surprises = sml.get_recent_surprises(
            n=10, surprise_type=SurpriseType.REWARD
        )
        assert len(reward_surprises) == 2

    def test_get_average_surprise(self):
        sml = SurpriseModulatedLearning(random_seed=42)

        for _ in range(10):
            sml.compute_surprise(1.0, 2.0)

        avg = sml.get_average_surprise()
        assert isinstance(avg, float)
        assert avg > 0

    def test_surprise_metrics_fields(self):
        sml = SurpriseModulatedLearning(random_seed=42)

        metrics = sml.compute_surprise(
            np.array([0.5, 0.5]),
            np.array([0.6, 0.4]),
        )

        assert hasattr(metrics, "surprise_value")
        assert hasattr(metrics, "modulated_lr")
        assert hasattr(metrics, "should_consolidate")
        assert hasattr(metrics, "kl_divergence")
        assert hasattr(metrics, "entropy")

    def test_statistics(self):
        sml = SurpriseModulatedLearning(random_seed=42)

        for _ in range(5):
            sml.compute_surprise(1.0, 2.0)

        stats = sml.statistics()

        assert "total_surprises" in stats
        assert stats["total_surprises"] == 5
        assert "consolidation_rate" in stats

class TestSurpriseTypes:
    """Tests for different surprise types."""

    def test_reward_surprise(self):
        sml = SurpriseModulatedLearning(random_seed=42)

        metrics = sml.compute_surprise(
            0.0, 1.0, surprise_type=SurpriseType.REWARD
        )
        assert metrics.surprise_type == SurpriseType.REWARD

    def test_state_surprise(self):
        sml = SurpriseModulatedLearning(random_seed=42)

        metrics = sml.compute_surprise(
            np.array([0, 0, 0]),
            np.array([1, 1, 1]),
            surprise_type=SurpriseType.STATE,
        )
        assert metrics.surprise_type == SurpriseType.STATE

    def test_outcome_surprise(self):
        sml = SurpriseModulatedLearning(random_seed=42)

        metrics = sml.compute_surprise(
            "expected",
            "unexpected",
            surprise_type=SurpriseType.OUTCOME,
        )
        assert metrics.surprise_type == SurpriseType.OUTCOME
