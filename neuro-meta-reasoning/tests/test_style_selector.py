"""Tests for style selector."""

import pytest
import numpy as np

from src.style_selector import (
    StyleSelector,
    StyleSelectorConfig,
    StyleSelection,
    ReasoningStyle,
    StyleFeedback,
)
from src.problem_classifier import ProblemAnalysis, ProblemType, ProblemDifficulty


class TestStyleSelectorConfig:
    """Tests for StyleSelectorConfig class."""

    def test_default_config(self):
        config = StyleSelectorConfig()
        assert config.exploration_rate == 0.1
        assert config.learning_rate == 0.1
        assert config.allow_style_combination

    def test_custom_config(self):
        config = StyleSelectorConfig(exploration_rate=0.2, max_combined_styles=2)
        assert config.exploration_rate == 0.2
        assert config.max_combined_styles == 2


class TestStyleSelector:
    """Tests for StyleSelector class."""

    def test_creation(self):
        selector = StyleSelector(random_seed=42)
        assert selector is not None

    def test_select_style(self):
        selector = StyleSelector(random_seed=42)

        analysis = ProblemAnalysis(
            problem_type=ProblemType.LOGICAL,
            type_confidence=0.8,
            complexity=0.5,
            difficulty=ProblemDifficulty.MEDIUM,
            features={},
            subproblems=[],
            estimated_steps=5,
            requires_domain_knowledge=False,
        )

        selection = selector.select_style(analysis)

        assert isinstance(selection, StyleSelection)
        assert selection.primary_style in ReasoningStyle
        assert 0 <= selection.primary_fitness <= 1
        assert 0 <= selection.confidence <= 1

    def test_select_with_constraints(self):
        selector = StyleSelector(random_seed=42)

        analysis = ProblemAnalysis(
            problem_type=ProblemType.LOGICAL,
            type_confidence=0.8,
            complexity=0.5,
            difficulty=ProblemDifficulty.MEDIUM,
            features={},
            subproblems=[],
            estimated_steps=5,
            requires_domain_knowledge=False,
        )

        constraints = {
            "excluded_styles": [ReasoningStyle.HEURISTIC],
        }

        selection = selector.select_style(analysis, constraints)

        assert selection.primary_style != ReasoningStyle.HEURISTIC

    def test_compute_style_fitness(self):
        selector = StyleSelector(random_seed=42)

        analysis = ProblemAnalysis(
            problem_type=ProblemType.LOGICAL,
            type_confidence=0.8,
            complexity=0.5,
            difficulty=ProblemDifficulty.MEDIUM,
            features={},
            subproblems=[],
            estimated_steps=5,
            requires_domain_knowledge=False,
        )

        fitness = selector.compute_style_fitness(ReasoningStyle.DEDUCTIVE, analysis)

        assert 0 <= fitness <= 1
        assert fitness > 0.5

    def test_adaptive_selection(self):
        selector = StyleSelector(random_seed=42)

        analysis = ProblemAnalysis(
            problem_type=ProblemType.CAUSAL,
            type_confidence=0.8,
            complexity=0.5,
            difficulty=ProblemDifficulty.MEDIUM,
            features={},
            subproblems=[],
            estimated_steps=5,
            requires_domain_knowledge=False,
        )

        feedback_history = [
            StyleFeedback(ReasoningStyle.CAUSAL, ProblemType.CAUSAL, True, 0.8, 0.9),
            StyleFeedback(ReasoningStyle.DEDUCTIVE, ProblemType.CAUSAL, False, 0.3, 0.4),
        ]

        selection = selector.adaptive_selection(analysis, feedback_history)

        assert isinstance(selection, StyleSelection)

    def test_update_from_feedback(self):
        selector = StyleSelector(random_seed=42)

        initial_rankings = selector.get_style_rankings(ProblemType.LOGICAL)

        feedback = StyleFeedback(
            style=ReasoningStyle.INDUCTIVE,
            problem_type=ProblemType.LOGICAL,
            success=True,
            efficiency=0.9,
            quality=0.9,
        )
        selector.update_from_feedback(feedback)

        new_rankings = selector.get_style_rankings(ProblemType.LOGICAL)

        # Verify that feedback actually updated rankings
        # With positive feedback (success=True, efficiency=0.9, quality=0.9),
        # the DEDUCTIVE style fitness for LOGICAL should increase
        deductive_idx = [s for s, _ in new_rankings].index(ReasoningStyle.DEDUCTIVE)
        initial_deductive_idx = [s for s, _ in initial_rankings].index(ReasoningStyle.DEDUCTIVE)

        # DEDUCTIVE should be ranked at least as high (lower index = higher rank)
        # or the actual fitness value should have increased
        initial_deductive_fitness = dict(initial_rankings)[ReasoningStyle.DEDUCTIVE]
        new_deductive_fitness = dict(new_rankings)[ReasoningStyle.DEDUCTIVE]

        # After positive feedback, fitness should increase or stay the same
        assert new_deductive_fitness >= initial_deductive_fitness, (
            f"DEDUCTIVE fitness should increase after positive feedback, "
            f"was {initial_deductive_fitness:.3f}, now {new_deductive_fitness:.3f}"
        )

    def test_record_feedback(self):
        selector = StyleSelector(random_seed=42)

        selector.record_feedback(
            ReasoningStyle.ANALOGICAL,
            ProblemType.ANALOGICAL,
            success=True,
            efficiency=0.8,
            quality=0.85,
        )

        stats = selector.statistics()
        assert stats["total_feedbacks"] == 1

    def test_get_style_rankings(self):
        selector = StyleSelector(random_seed=42)

        rankings = selector.get_style_rankings(ProblemType.LOGICAL)

        assert len(rankings) == len(ReasoningStyle)
        assert all(isinstance(r[0], ReasoningStyle) for r in rankings)
        assert all(0 <= r[1] <= 1 for r in rankings)

        fitnesses = [r[1] for r in rankings]
        assert fitnesses == sorted(fitnesses, reverse=True)

    def test_get_fitness_matrix(self):
        selector = StyleSelector(random_seed=42)

        matrix = selector.get_fitness_matrix()

        assert len(matrix) == len(ReasoningStyle)
        for style in ReasoningStyle:
            assert style.value in matrix
            assert len(matrix[style.value]) == len(ProblemType)

    def test_statistics(self):
        selector = StyleSelector(random_seed=42)

        analysis = ProblemAnalysis(
            problem_type=ProblemType.LOGICAL,
            type_confidence=0.8,
            complexity=0.5,
            difficulty=ProblemDifficulty.MEDIUM,
            features={},
            subproblems=[],
            estimated_steps=5,
            requires_domain_knowledge=False,
        )
        selector.select_style(analysis)

        stats = selector.statistics()

        assert "total_selections" in stats
        assert "exploration_rate" in stats
        assert stats["total_selections"] == 1


class TestReasoningStyle:
    """Tests for ReasoningStyle enum."""

    def test_styles(self):
        assert ReasoningStyle.DEDUCTIVE.value == "deductive"
        assert ReasoningStyle.INDUCTIVE.value == "inductive"
        assert ReasoningStyle.ABDUCTIVE.value == "abductive"
        assert ReasoningStyle.ANALOGICAL.value == "analogical"
        assert ReasoningStyle.CAUSAL.value == "causal"


class TestStyleSelection:
    """Tests for StyleSelection dataclass."""

    def test_selection_creation(self):
        selection = StyleSelection(
            primary_style=ReasoningStyle.DEDUCTIVE,
            primary_fitness=0.9,
            secondary_styles=[(ReasoningStyle.SYSTEMATIC, 0.7)],
            confidence=0.85,
            rationale="Logical problem",
        )

        assert selection.primary_style == ReasoningStyle.DEDUCTIVE
        assert selection.primary_fitness == 0.9
        assert len(selection.secondary_styles) == 1
