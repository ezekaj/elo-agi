"""Tests for contribution accounting with Shapley values."""

import pytest

from neuro.modules.credit.contribution_accounting import (
    Contribution,
    ShapleyConfig,
    ContributionAccountant,
)


class TestShapleyConfig:
    """Tests for ShapleyConfig class."""

    def test_default_config(self):
        config = ShapleyConfig()
        assert config.use_approximation
        assert config.num_samples == 100

    def test_custom_config(self):
        config = ShapleyConfig(use_approximation=False, num_samples=50)
        assert not config.use_approximation


class TestContribution:
    """Tests for Contribution class."""

    def test_contribution_creation(self):
        contrib = Contribution(
            module_id="mod1",
            shapley_value=0.5,
            marginal_contributions=[0.3, 0.5, 0.7],
            coalition_values={"A": 1.0, "B": 2.0},
            timestamp=10,
        )
        assert contrib.module_id == "mod1"
        assert contrib.shapley_value == 0.5


class TestContributionAccountant:
    """Tests for ContributionAccountant class."""

    def test_creation(self):
        accountant = ContributionAccountant(random_seed=42)
        assert accountant is not None

    def test_compute_shapley_simple(self):
        accountant = ContributionAccountant(random_seed=42)

        def value_fn(coalition):
            return len(coalition) * 1.0

        shapley = accountant.compute_shapley_values(
            outcome=3.0,
            active_modules=["m1", "m2", "m3"],
            value_function=value_fn,
        )

        assert len(shapley) == 3
        assert sum(shapley.values()) == pytest.approx(3.0, rel=0.1)

    def test_compute_shapley_asymmetric(self):
        config = ShapleyConfig(use_approximation=False)
        accountant = ContributionAccountant(config=config, random_seed=42)

        def value_fn(coalition):
            value = 0.0
            if "m1" in coalition:
                value += 2.0
            if "m2" in coalition:
                value += 1.0
            return value

        shapley = accountant.compute_shapley_values(
            outcome=3.0,
            active_modules=["m1", "m2"],
            value_function=value_fn,
        )

        assert shapley["m1"] > shapley["m2"]

    def test_compute_shapley_approximation(self):
        config = ShapleyConfig(use_approximation=True, num_samples=200)
        accountant = ContributionAccountant(config=config, random_seed=42)

        def value_fn(coalition):
            return len(coalition) ** 2

        shapley = accountant.compute_shapley_values(
            outcome=16.0,
            active_modules=["m1", "m2", "m3", "m4"],
            value_function=value_fn,
        )

        assert len(shapley) == 4

    def test_distribute_reward_equal(self):
        accountant = ContributionAccountant(random_seed=42)

        distribution = accountant.distribute_reward(
            reward=10.0,
            active_modules=["m1", "m2", "m3", "m4"],
        )

        assert len(distribution) == 4
        assert all(v == pytest.approx(2.5) for v in distribution.values())

    def test_distribute_reward_weighted(self):
        accountant = ContributionAccountant(random_seed=42)

        distribution = accountant.distribute_reward(
            reward=10.0,
            active_modules=["m1", "m2"],
            module_outputs={"m1": 3.0, "m2": 1.0},
        )

        assert distribution["m1"] > distribution["m2"]

    def test_record_contribution(self):
        accountant = ContributionAccountant(random_seed=42)

        contrib = accountant.record_contribution(
            module_id="m1",
            shapley_value=0.5,
            marginal_contributions=[0.3, 0.5],
        )

        assert isinstance(contrib, Contribution)
        assert len(accountant.get_contribution_history()) == 1

    def test_identify_underperforming(self):
        accountant = ContributionAccountant(random_seed=42)

        for _ in range(10):
            accountant.record_contribution("good", 1.0)
            accountant.record_contribution("bad", 0.1)

        underperforming = accountant.identify_underperforming_modules(threshold=0.5)

        assert "bad" in underperforming
        assert "good" not in underperforming

    def test_identify_top_contributors(self):
        accountant = ContributionAccountant(random_seed=42)

        for _ in range(10):
            accountant.record_contribution("top", 2.0)
            accountant.record_contribution("mid", 1.0)
            accountant.record_contribution("low", 0.5)

        top = accountant.identify_top_contributors(n=2)

        assert len(top) == 2
        assert top[0][0] == "top"
        assert top[1][0] == "mid"

    def test_get_cumulative_contributions(self):
        accountant = ContributionAccountant(random_seed=42)

        accountant.distribute_reward(10.0, ["m1", "m2"])
        accountant.distribute_reward(20.0, ["m1", "m2"])

        cumulative = accountant.get_cumulative_contributions()

        assert cumulative["m1"] == pytest.approx(15.0)
        assert cumulative["m2"] == pytest.approx(15.0)

    def test_get_contribution_history_filtered(self):
        accountant = ContributionAccountant(random_seed=42)

        for i in range(5):
            accountant.record_contribution("m1", float(i))
            accountant.record_contribution("m2", float(i) * 2)

        m1_history = accountant.get_contribution_history(module_id="m1")
        assert len(m1_history) == 5

        recent = accountant.get_contribution_history(n=3)
        assert len(recent) == 3

    def test_compute_contribution_variance(self):
        accountant = ContributionAccountant(random_seed=42)

        for v in [1.0, 1.0, 1.0]:
            accountant.record_contribution("stable", v)
        for v in [0.0, 1.0, 2.0]:
            accountant.record_contribution("variable", v)

        stable_var = accountant.compute_contribution_variance("stable")
        variable_var = accountant.compute_contribution_variance("variable")

        assert stable_var < variable_var

    def test_reset(self):
        accountant = ContributionAccountant(random_seed=42)

        accountant.record_contribution("m1", 1.0)
        accountant.distribute_reward(10.0, ["m1"])

        accountant.reset()

        assert len(accountant.get_contribution_history()) == 0
        assert len(accountant.get_cumulative_contributions()) == 0

    def test_statistics(self):
        accountant = ContributionAccountant(random_seed=42)

        accountant.record_contribution("m1", 1.0)
        accountant.record_contribution("m2", 2.0)

        def value_fn(c):
            return len(c)

        accountant.compute_shapley_values(1.0, ["m1"], value_fn)

        stats = accountant.statistics()

        assert "total_modules" in stats
        assert "total_computations" in stats
        assert stats["total_modules"] == 2


class TestShapleyProperties:
    """Tests for Shapley value properties."""

    def test_efficiency(self):
        config = ShapleyConfig(use_approximation=False)
        accountant = ContributionAccountant(config=config, random_seed=42)

        def value_fn(coalition):
            return len(coalition) * 2.0

        shapley = accountant.compute_shapley_values(
            outcome=6.0,
            active_modules=["a", "b", "c"],
            value_function=value_fn,
        )

        assert sum(shapley.values()) == pytest.approx(6.0, rel=0.1)

    def test_symmetry(self):
        config = ShapleyConfig(use_approximation=False)
        accountant = ContributionAccountant(config=config, random_seed=42)

        def value_fn(coalition):
            return len(coalition)

        shapley = accountant.compute_shapley_values(
            outcome=2.0,
            active_modules=["a", "b"],
            value_function=value_fn,
        )

        assert shapley["a"] == pytest.approx(shapley["b"], rel=0.1)

    def test_null_player(self):
        config = ShapleyConfig(use_approximation=False)
        accountant = ContributionAccountant(config=config, random_seed=42)

        def value_fn(coalition):
            value = 0.0
            if "a" in coalition:
                value += 1.0
            return value

        shapley = accountant.compute_shapley_values(
            outcome=1.0,
            active_modules=["a", "null"],
            value_function=value_fn,
        )

        assert shapley["null"] == pytest.approx(0.0, abs=0.01)
        assert shapley["a"] == pytest.approx(1.0, rel=0.1)
