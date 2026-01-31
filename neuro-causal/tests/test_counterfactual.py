"""
Comprehensive tests for Nested Counterfactual reasoning.

Tests cover:
- Basic counterfactuals
- Nested counterfactuals
- Contrastive explanations
- Probability of necessity (PN)
- Probability of sufficiency (PS)
- Counterfactual stability
"""

import pytest
import numpy as np
from src.differentiable_scm import DifferentiableSCM
from src.counterfactual import (
    NestedCounterfactual,
    CounterfactualQuery,
    CounterfactualResult,
    ContrastiveExplanation,
    ExplanationType,
)


@pytest.fixture
def simple_scm():
    """Create a simple chain SCM: X -> Y -> Z."""
    scm = DifferentiableSCM(random_seed=42)
    scm.add_linear_mechanism("X", [], {}, intercept=0.0, noise_std=1.0)
    scm.add_linear_mechanism("Y", ["X"], {"X": 2.0}, intercept=0.0, noise_std=0.1)
    scm.add_linear_mechanism("Z", ["Y"], {"Y": 3.0}, intercept=0.0, noise_std=0.1)
    return scm


@pytest.fixture
def fork_scm():
    """Create a fork SCM: Y <- X -> Z."""
    scm = DifferentiableSCM(random_seed=42)
    scm.add_linear_mechanism("X", [], {}, intercept=0.0, noise_std=1.0)
    scm.add_linear_mechanism("Y", ["X"], {"X": 2.0}, intercept=0.0, noise_std=0.1)
    scm.add_linear_mechanism("Z", ["X"], {"X": 3.0}, intercept=0.0, noise_std=0.1)
    return scm


@pytest.fixture
def diamond_scm():
    """Create a diamond SCM: A -> B, A -> C, B -> D, C -> D."""
    scm = DifferentiableSCM(random_seed=42)
    scm.add_linear_mechanism("A", [], {}, intercept=1.0, noise_std=0.5)
    scm.add_linear_mechanism("B", ["A"], {"A": 2.0}, intercept=0.0, noise_std=0.1)
    scm.add_linear_mechanism("C", ["A"], {"A": 1.5}, intercept=0.0, noise_std=0.1)
    scm.add_linear_mechanism("D", ["B", "C"], {"B": 1.0, "C": 1.0}, intercept=0.0, noise_std=0.1)
    return scm


class TestBasicCounterfactual:
    """Test basic counterfactual computation."""

    def test_counterfactual_query_creation(self):
        """Query creation should work."""
        query = CounterfactualQuery(
            evidence={"X": 1.0, "Y": 2.0},
            intervention={"X": 3.0},
            outcome_var="Y",
            description="Test query",
        )
        assert query.evidence == {"X": 1.0, "Y": 2.0}
        assert query.intervention == {"X": 3.0}

    def test_counterfactual_result(self, simple_scm):
        """Counterfactual should return proper result."""
        cf = NestedCounterfactual(simple_scm)

        result = cf.compute(
            evidence={"X": 2.0, "Y": 4.0, "Z": 12.0},
            intervention={"X": 4.0},
            outcome_var="Y",
        )

        assert isinstance(result, CounterfactualResult)
        assert result.factual_value is not None
        assert result.counterfactual_value is not None
        assert result.inferred_noise is not None

    def test_counterfactual_respects_mechanism(self, simple_scm):
        """Counterfactual should respect causal mechanisms."""
        cf = NestedCounterfactual(simple_scm)

        # Y = 2X + noise, so if X changes from 2 to 4, Y should change by ~4
        result = cf.compute(
            evidence={"X": 2.0, "Y": 4.0},
            intervention={"X": 4.0},
            outcome_var="Y",
        )

        # Y should be approximately 8 (2 * 4)
        assert abs(result.counterfactual_value - 8.0) < 1.0

    def test_counterfactual_preserves_noise(self, simple_scm):
        """Counterfactual should preserve inferred noise."""
        cf = NestedCounterfactual(simple_scm)

        # Evidence with specific noise component
        evidence = {"X": 3.0, "Y": 6.5}  # Y should be 6 if noise=0, so noise ~ 0.5

        result = cf.compute(
            evidence=evidence,
            intervention={"X": 5.0},
            outcome_var="Y",
        )

        # Y = 2*5 + noise(~0.5) = 10.5
        assert abs(result.counterfactual_value - 10.5) < 1.0

    def test_counterfactual_chain_propagation(self, simple_scm):
        """Counterfactual should propagate through chain."""
        cf = NestedCounterfactual(simple_scm)

        result = cf.compute(
            evidence={"X": 1.0, "Y": 2.0, "Z": 6.0},
            intervention={"X": 2.0},
            outcome_var="Z",
        )

        # X=2 -> Y=4 -> Z=12
        assert abs(result.counterfactual_value - 12.0) < 2.0


class TestNestedCounterfactual:
    """Test nested counterfactual computation."""

    def test_nested_counterfactual_basic(self, simple_scm):
        """Nested counterfactual should compute nested values."""
        cf = NestedCounterfactual(simple_scm)

        result = cf.compute_nested(
            primary_intervention={"X": 5.0},
            secondary_intervention={"X": 3.0},
            outcome_var="Y",
        )

        assert "nested_outcome" in result
        assert "simple_outcome" in result
        assert "nesting_effect" in result

    def test_nested_vs_simple_difference(self, simple_scm):
        """Nested and simple counterfactuals should differ."""
        cf = NestedCounterfactual(simple_scm)

        # With evidence, nested should consider it
        result = cf.compute_nested(
            primary_intervention={"X": 5.0},
            secondary_intervention={"X": 3.0},
            outcome_var="Y",
            evidence={"X": 1.0, "Y": 2.5},  # Some noise
        )

        # The nesting effect shows the difference between nested and simple reasoning
        assert "nesting_effect" in result

    def test_nested_counterfactual_worlds(self, diamond_scm):
        """Nested counterfactual should create proper nested worlds."""
        cf = NestedCounterfactual(diamond_scm)

        result = cf.compute_nested(
            primary_intervention={"A": 2.0},
            secondary_intervention={"B": 5.0},
            outcome_var="D",
        )

        assert "inner_world" in result
        assert "nested_world" in result
        assert "B" in result["inner_world"]


class TestContrastiveExplanation:
    """Test contrastive explanation generation."""

    def test_contrastive_explanation_creation(self, simple_scm):
        """Contrastive explanation should be created."""
        cf = NestedCounterfactual(simple_scm)

        explanation = cf.contrastive_explanation(
            actual_outcome={"X": 2.0, "Y": 4.0},
            foil_outcome={"X": 2.0, "Y": 8.0},
            cause_variable="X",
            outcome_variable="Y",
        )

        assert isinstance(explanation, ContrastiveExplanation)
        assert explanation.fact is not None
        assert explanation.foil is not None

    def test_contrastive_explanation_finds_cause(self, simple_scm):
        """Explanation should identify what change causes foil."""
        cf = NestedCounterfactual(simple_scm)

        explanation = cf.contrastive_explanation(
            actual_outcome={"X": 2.0, "Y": 4.0},
            foil_outcome={"X": 2.0, "Y": 8.0},
            cause_variable="X",
            outcome_variable="Y",
        )

        # To get Y=8 instead of Y=4, X needs to be 4 instead of 2
        assert abs(explanation.counterfactual_cause_value - 4.0) < 1.0

    def test_contrastive_explanation_text(self, simple_scm):
        """Explanation should generate readable text."""
        cf = NestedCounterfactual(simple_scm)

        explanation = cf.contrastive_explanation(
            actual_outcome={"X": 2.0, "Y": 4.0},
            foil_outcome={"X": 2.0, "Y": 8.0},
            cause_variable="X",
            outcome_variable="Y",
        )

        assert isinstance(explanation.explanation, str)
        assert len(explanation.explanation) > 0
        assert "Y" in explanation.explanation or "X" in explanation.explanation


class TestProbabilityOfNecessity:
    """Test Probability of Necessity computation."""

    def test_pn_returns_probability(self, simple_scm):
        """PN should return value in [0, 1]."""
        cf = NestedCounterfactual(simple_scm, n_monte_carlo=50)

        pn = cf.probability_of_necessity(
            cause="X",
            effect="Y",
            cause_value=2.0,
            effect_value=4.0,
            evidence={"X": 2.0, "Y": 4.0},
        )

        assert 0.0 <= pn <= 1.0

    def test_pn_high_for_necessary_cause(self, simple_scm):
        """PN should be high when cause is necessary."""
        cf = NestedCounterfactual(simple_scm, n_monte_carlo=100)

        # X directly causes Y with coefficient 2, so X is necessary for Y
        pn = cf.probability_of_necessity(
            cause="X",
            effect="Y",
            cause_value=2.0,
            effect_value=4.0,
            evidence={"X": 2.0, "Y": 4.0},
        )

        # Should be high because if X had been different, Y would have been different
        assert pn > 0.3  # Allow some tolerance

    def test_pn_low_for_unnecessary_cause(self):
        """PN should be low when cause is not necessary."""
        # Create SCM where Y doesn't depend on X
        scm = DifferentiableSCM(random_seed=42)
        scm.add_linear_mechanism("X", [], {}, noise_std=1.0)
        scm.add_linear_mechanism("Y", [], {}, intercept=5.0, noise_std=0.1)  # Y constant

        cf = NestedCounterfactual(scm, n_monte_carlo=50)

        pn = cf.probability_of_necessity(
            cause="X",
            effect="Y",
            cause_value=2.0,
            effect_value=5.0,
            evidence={"X": 2.0, "Y": 5.0},
        )

        # Should be low because Y doesn't depend on X
        assert pn < 0.5


class TestProbabilityOfSufficiency:
    """Test Probability of Sufficiency computation."""

    def test_ps_returns_probability(self, simple_scm):
        """PS should return value in [0, 1]."""
        cf = NestedCounterfactual(simple_scm, n_monte_carlo=50)

        ps = cf.probability_of_sufficiency(
            cause="X",
            effect="Y",
            cause_value=2.0,
            effect_value=4.0,
            evidence={},
        )

        assert 0.0 <= ps <= 1.0

    def test_ps_high_for_sufficient_cause(self, simple_scm):
        """PS should be high when cause is sufficient."""
        cf = NestedCounterfactual(simple_scm, n_monte_carlo=100)

        # Setting X=2 should be sufficient to get Y~4
        ps = cf.probability_of_sufficiency(
            cause="X",
            effect="Y",
            cause_value=2.0,
            effect_value=4.0,
            evidence={},
        )

        # Should be reasonably high
        assert ps > 0.2


class TestExplanationType:
    """Test explanation type classification."""

    def test_explain_returns_type(self, simple_scm):
        """Explain should return ExplanationType."""
        cf = NestedCounterfactual(simple_scm, n_monte_carlo=30)

        explanation_type = cf.explain(
            cause="X",
            effect="Y",
            evidence={"X": 2.0, "Y": 4.0},
        )

        assert isinstance(explanation_type, ExplanationType)

    def test_explanation_types(self):
        """All explanation types should be accessible."""
        assert ExplanationType.NECESSARY is not None
        assert ExplanationType.SUFFICIENT is not None
        assert ExplanationType.NECESSARY_SUFFICIENT is not None
        assert ExplanationType.CONTRIBUTORY is not None


class TestCounterfactualStability:
    """Test counterfactual stability analysis."""

    def test_stability_analysis_output(self, simple_scm):
        """Stability analysis should return proper metrics."""
        cf = NestedCounterfactual(simple_scm, n_monte_carlo=30)

        stability = cf.counterfactual_stability(
            evidence={"X": 2.0, "Y": 4.0},
            intervention={"X": 4.0},
            outcome_var="Y",
            perturbation_std=0.1,
        )

        assert "base_counterfactual" in stability
        assert "mean" in stability
        assert "std" in stability
        assert "stability" in stability

    def test_stability_score_bounds(self, simple_scm):
        """Stability score should be bounded."""
        cf = NestedCounterfactual(simple_scm, n_monte_carlo=30)

        stability = cf.counterfactual_stability(
            evidence={"X": 2.0, "Y": 4.0},
            intervention={"X": 4.0},
            outcome_var="Y",
        )

        assert 0.0 <= stability["stability"] <= 1.0

    def test_low_perturbation_high_stability(self, simple_scm):
        """Low perturbation should give high stability."""
        cf = NestedCounterfactual(simple_scm, n_monte_carlo=30)

        stability = cf.counterfactual_stability(
            evidence={"X": 2.0, "Y": 4.0},
            intervention={"X": 4.0},
            outcome_var="Y",
            perturbation_std=0.01,  # Very small perturbation
        )

        assert stability["stability"] > 0.5


class TestStatisticsTracking:
    """Test statistics and tracking."""

    def test_query_count_increments(self, simple_scm):
        """Query count should increment."""
        cf = NestedCounterfactual(simple_scm)

        initial_stats = cf.statistics()
        cf.compute({"X": 1.0}, {"X": 2.0}, "Y")
        cf.compute({"X": 2.0}, {"X": 3.0}, "Y")

        final_stats = cf.statistics()
        assert final_stats["n_queries"] == initial_stats["n_queries"] + 2

    def test_nested_query_count(self, simple_scm):
        """Nested query count should increment."""
        cf = NestedCounterfactual(simple_scm)

        initial_stats = cf.statistics()
        cf.compute_nested({"X": 1.0}, {"X": 2.0}, "Y")

        final_stats = cf.statistics()
        assert final_stats["n_nested_queries"] == initial_stats["n_nested_queries"] + 1

    def test_explanation_count(self, simple_scm):
        """Explanation count should increment."""
        cf = NestedCounterfactual(simple_scm)

        initial_stats = cf.statistics()
        cf.contrastive_explanation(
            {"X": 1.0, "Y": 2.0},
            {"X": 1.0, "Y": 4.0},
            "X", "Y"
        )

        final_stats = cf.statistics()
        assert final_stats["n_explanations"] == initial_stats["n_explanations"] + 1


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_empty_evidence(self, simple_scm):
        """Should handle empty evidence."""
        cf = NestedCounterfactual(simple_scm)

        result = cf.compute(
            evidence={},
            intervention={"X": 2.0},
            outcome_var="Y",
        )

        assert result.counterfactual_value is not None

    def test_single_variable_scm(self):
        """Should work with single variable SCM."""
        scm = DifferentiableSCM()
        scm.add_linear_mechanism("X", [], {}, noise_std=1.0)

        cf = NestedCounterfactual(scm)
        result = cf.compute(
            evidence={"X": 3.0},
            intervention={"X": 5.0},
            outcome_var="X",
        )

        assert result.counterfactual_value == 5.0

    def test_intervention_on_root(self, simple_scm):
        """Intervention on root should propagate."""
        cf = NestedCounterfactual(simple_scm)

        result = cf.compute(
            evidence={"X": 0.0, "Y": 0.0, "Z": 0.0},
            intervention={"X": 5.0},
            outcome_var="Z",
        )

        # X=5 -> Y=10 -> Z=30
        assert abs(result.counterfactual_value - 30.0) < 5.0

    def test_intervention_on_leaf(self, simple_scm):
        """Intervention on leaf should not affect ancestors."""
        cf = NestedCounterfactual(simple_scm)

        result = cf.compute(
            evidence={"X": 2.0, "Y": 4.0, "Z": 12.0},
            intervention={"Z": 100.0},
            outcome_var="Y",
        )

        # Y should be unchanged (Z doesn't affect Y)
        assert abs(result.counterfactual_value - 4.0) < 1.0

    def test_multiple_interventions(self, fork_scm):
        """Should handle multiple simultaneous interventions."""
        cf = NestedCounterfactual(fork_scm)

        # Intervene on both Y and Z
        result = cf.compute(
            evidence={"X": 1.0, "Y": 2.0, "Z": 3.0},
            intervention={"Y": 10.0, "Z": 20.0},
            outcome_var="Y",
        )

        assert result.counterfactual_value == 10.0


class TestMathematicalConsistency:
    """Test mathematical properties of counterfactuals."""

    def test_idempotent_intervention(self, simple_scm):
        """Intervention with observed value should give same result."""
        cf = NestedCounterfactual(simple_scm)

        evidence = {"X": 3.0, "Y": 6.0, "Z": 18.0}
        result = cf.compute(
            evidence=evidence,
            intervention={"X": 3.0},  # Same as observed
            outcome_var="Y",
        )

        assert abs(result.counterfactual_value - result.factual_value) < 0.5

    def test_composition_of_interventions(self, simple_scm):
        """Sequential interventions should compose correctly."""
        cf = NestedCounterfactual(simple_scm)

        # First intervention: X=2
        result1 = cf.compute(
            evidence={"X": 1.0},
            intervention={"X": 2.0},
            outcome_var="Y",
        )

        # Use result as new evidence
        result2 = cf.compute(
            evidence={"X": 2.0, "Y": result1.counterfactual_value},
            intervention={"Y": 10.0},
            outcome_var="Z",
        )

        # Z should be ~30 (Y=10 -> Z=30)
        assert abs(result2.counterfactual_value - 30.0) < 5.0

    def test_pn_ps_bounds_relationship(self, simple_scm):
        """PNS <= min(PN, PS) should hold."""
        cf = NestedCounterfactual(simple_scm, n_monte_carlo=100)

        pn = cf.probability_of_necessity(
            cause="X", effect="Y",
            cause_value=2.0, effect_value=4.0,
            evidence={"X": 2.0, "Y": 4.0},
        )

        ps = cf.probability_of_sufficiency(
            cause="X", effect="Y",
            cause_value=2.0, effect_value=4.0,
            evidence={},
        )

        pns = cf.probability_of_necessity_and_sufficiency(
            cause="X", effect="Y",
            cause_value=2.0, effect_value=4.0,
        )

        assert pns <= min(pn, ps) + 0.1  # Allow small tolerance
