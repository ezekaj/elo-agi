"""Tests for policy gradient computation."""

import pytest
import numpy as np

from neuro.modules.credit.policy_gradient import (
    GAEConfig,
    Advantage,
    CrossModulePolicyGradient,
    PolicyGradientResult,
)


class TestGAEConfig:
    """Tests for GAEConfig class."""

    def test_default_config(self):
        config = GAEConfig()
        assert config.gamma == 0.99
        assert config.lambda_param == 0.95
        assert config.normalize_advantages

    def test_custom_config(self):
        config = GAEConfig(gamma=0.9, normalize_advantages=False)
        assert config.gamma == 0.9
        assert not config.normalize_advantages


class TestCrossModulePolicyGradient:
    """Tests for CrossModulePolicyGradient class."""

    def test_creation(self):
        pg = CrossModulePolicyGradient(random_seed=42)
        assert pg is not None

    def test_compute_advantage_simple(self):
        pg = CrossModulePolicyGradient(random_seed=42)

        rewards = [1.0, 1.0, 1.0]
        values = [0.5, 0.5, 0.5]

        advantages, targets = pg.compute_advantage(rewards, values)

        assert len(advantages) == 3
        assert len(targets) == 3
        # Advantages are normalized, so we just check they exist
        assert all(isinstance(a, (float, np.floating)) for a in advantages)

    def test_compute_advantage_with_dones(self):
        pg = CrossModulePolicyGradient(random_seed=42)

        rewards = [1.0, 1.0, 1.0]
        values = [0.5, 0.5, 0.5]
        dones = [False, True, False]

        advantages, targets = pg.compute_advantage(rewards, values, dones)

        assert len(advantages) == 3

    def test_compute_advantage_normalized(self):
        config = GAEConfig(normalize_advantages=True)
        pg = CrossModulePolicyGradient(config=config, random_seed=42)

        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        values = [0.5, 1.0, 1.5, 2.0, 2.5]

        advantages, _ = pg.compute_advantage(rewards, values)

        assert np.abs(np.mean(advantages)) < 0.1
        assert np.abs(np.std(advantages) - 1.0) < 0.5

    def test_compute_advantage_unnormalized(self):
        config = GAEConfig(normalize_advantages=False)
        pg = CrossModulePolicyGradient(config=config, random_seed=42)

        rewards = [1.0, 2.0, 3.0]
        values = [0.0, 0.0, 0.0]

        advantages, _ = pg.compute_advantage(rewards, values)

        # Earlier advantages accumulate more future reward via GAE
        # Just check they're computed correctly
        assert len(advantages) == 3
        assert all(a > 0 for a in advantages)

    def test_compute_policy_loss_vanilla(self):
        pg = CrossModulePolicyGradient(random_seed=42)

        log_probs = np.array([-0.5, -0.3, -0.2])
        advantages = np.array([1.0, 0.5, 0.2])

        loss = pg.compute_policy_loss(log_probs, advantages)

        assert isinstance(loss, float)
        # Loss = -mean(log_probs * advantages), which is positive when
        # log_probs are negative and advantages are positive
        assert loss > 0

    def test_compute_policy_loss_ppo(self):
        pg = CrossModulePolicyGradient(random_seed=42)

        log_probs = np.array([-0.5, -0.3, -0.2])
        old_log_probs = np.array([-0.6, -0.4, -0.3])
        advantages = np.array([1.0, 0.5, 0.2])

        loss = pg.compute_policy_loss(log_probs, advantages, old_log_probs, clip_epsilon=0.2)

        assert isinstance(loss, float)

    def test_compute_value_loss(self):
        pg = CrossModulePolicyGradient(random_seed=42)

        values = np.array([1.0, 2.0, 3.0])
        targets = np.array([1.5, 2.5, 3.5])

        loss = pg.compute_value_loss(values, targets)

        assert loss > 0
        assert loss == pytest.approx(0.125)

    def test_compute_value_loss_clipped(self):
        pg = CrossModulePolicyGradient(random_seed=42)

        values = np.array([1.0, 2.0, 3.0])
        old_values = np.array([0.8, 1.8, 2.8])
        targets = np.array([1.5, 2.5, 3.5])

        loss = pg.compute_value_loss(values, targets, old_values, clip_range=0.1)

        assert isinstance(loss, float)

    def test_compute_entropy(self):
        pg = CrossModulePolicyGradient(random_seed=42)

        uniform = np.array([[0.25, 0.25, 0.25, 0.25]])
        peaked = np.array([[0.9, 0.05, 0.03, 0.02]])

        uniform_entropy = pg.compute_entropy(uniform)
        peaked_entropy = pg.compute_entropy(peaked)

        assert uniform_entropy > peaked_entropy

    def test_accumulate_gradients(self):
        pg = CrossModulePolicyGradient(random_seed=42)

        trajectory = {
            "rewards": [1.0, 1.0, 1.0],
            "values": [0.5, 0.5, 0.5],
            "log_probs": np.array([-0.5, -0.5, -0.5]),
        }

        gradients = pg.accumulate_gradients(trajectory, "module1")

        assert "policy" in gradients
        assert "value" in gradients

    def test_get_accumulated_gradients(self):
        pg = CrossModulePolicyGradient(random_seed=42)

        for _ in range(3):
            trajectory = {
                "rewards": [1.0, 1.0],
                "values": [0.5, 0.5],
                "log_probs": np.array([-0.5, -0.5]),
            }
            pg.accumulate_gradients(trajectory, "module1")

        gradients = pg.get_accumulated_gradients("module1", average=True)

        assert "policy" in gradients

    def test_clear_gradients(self):
        pg = CrossModulePolicyGradient(random_seed=42)

        trajectory = {
            "rewards": [1.0],
            "values": [0.5],
            "log_probs": np.array([-0.5]),
        }
        pg.accumulate_gradients(trajectory, "module1")

        pg.clear_gradients("module1")

        gradients = pg.get_accumulated_gradients("module1")
        assert len(gradients) == 0

    def test_compute_explained_variance(self):
        pg = CrossModulePolicyGradient(random_seed=42)

        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        returns = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        ev = pg.compute_explained_variance(values, returns)
        assert ev == pytest.approx(1.0)

        values_bad = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        ev_bad = pg.compute_explained_variance(values_bad, returns)
        assert ev_bad < 0.5

    def test_compute_full_result(self):
        pg = CrossModulePolicyGradient(random_seed=42)

        trajectory = {
            "rewards": [1.0, 1.0, 1.0],
            "values": [0.5, 0.5, 0.5],
            "log_probs": np.array([-0.5, -0.5, -0.5]),
            "action_probs": np.array(
                [
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                ]
            ),
        }

        result = pg.compute_full_result(trajectory, "module1")

        assert isinstance(result, PolicyGradientResult)
        assert isinstance(result.policy_loss, float)
        assert isinstance(result.value_loss, float)
        assert isinstance(result.entropy, float)
        assert len(result.advantages) == 3

    def test_statistics(self):
        pg = CrossModulePolicyGradient(random_seed=42)

        trajectory = {
            "rewards": [1.0],
            "values": [0.5],
            "log_probs": np.array([-0.5]),
        }
        pg.accumulate_gradients(trajectory, "module1")

        stats = pg.statistics()

        assert "total_trajectories" in stats
        assert stats["total_trajectories"] == 1


class TestAdvantage:
    """Tests for Advantage dataclass."""

    def test_advantage_creation(self):
        adv = Advantage(
            state_index=0,
            advantage=0.5,
            value_target=1.0,
            td_error=0.1,
            module_id="mod1",
        )
        assert adv.advantage == 0.5
        assert adv.module_id == "mod1"
