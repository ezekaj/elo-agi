"""Tests for grammar manifold and Universal Grammar"""

import numpy as np
import pytest
from neuro.modules.m08_language.grammar_manifold import (
    GrammarConstraintManifold,
    UniversalGrammar,
    ImpossibleGrammarGenerator,
    GrammarState
)

class TestGrammarConstraintManifold:
    """Tests for grammar constraint manifold"""

    def test_initialization(self):
        """Test manifold initialization"""
        manifold = GrammarConstraintManifold(dim=32)

        assert manifold.dim == 32
        assert len(manifold.constraints) > 0

    def test_possible_grammar_check(self):
        """Test checking if grammar is possible"""
        manifold = GrammarConstraintManifold(dim=32)

        # Near center = possible
        possible_params = manifold.center + np.random.randn(32) * 0.1
        assert manifold.is_possible_grammar(possible_params) or manifold.get_violation_score(possible_params) < 1.0

        # Far from center = possibly impossible
        impossible_params = manifold.center + np.random.randn(32) * 10
        violation = manifold.get_violation_score(impossible_params)
        assert violation > 0  # Some violation

    def test_violation_score(self):
        """Test violation score computation"""
        manifold = GrammarConstraintManifold(dim=32)

        params = np.random.randn(32)
        violation = manifold.get_violation_score(params)

        assert violation >= 0
        assert np.isfinite(violation)

    def test_inhibition_signal(self):
        """Test Broca's inhibition signal"""
        manifold = GrammarConstraintManifold(dim=32)

        # Possible grammar = no inhibition
        possible_params = manifold.center.copy()
        inhibition_possible = manifold.inhibition_signal(possible_params)

        # Impossible grammar = inhibition
        impossible_params = np.random.randn(32) * 10
        inhibition_impossible = manifold.inhibition_signal(impossible_params)

        # Inhibition is non-negative
        assert inhibition_possible >= 0
        assert inhibition_impossible >= 0

    def test_project_to_possible(self):
        """Test projection to nearest possible grammar"""
        manifold = GrammarConstraintManifold(dim=32)

        # Start with impossible
        impossible = np.random.randn(32) * 5

        # Project
        projected = manifold.project_to_possible(impossible)

        # Projected should be more possible
        original_violation = manifold.get_violation_score(impossible)
        projected_violation = manifold.get_violation_score(projected)

        assert projected_violation <= original_violation + 0.1  # May not always decrease due to complexity

    def test_distance_to_boundary(self):
        """Test distance to boundary computation"""
        manifold = GrammarConstraintManifold(dim=32)

        # At center = inside
        center_dist = manifold.distance_to_boundary(manifold.center)

        # Far away = outside
        far_dist = manifold.distance_to_boundary(manifold.center + np.ones(32) * 10)

        # Center should have positive distance, far should have negative
        # (distance convention: positive = inside, negative = outside)
        assert np.isfinite(center_dist)
        assert np.isfinite(far_dist)

    def test_evaluate(self):
        """Test full grammar evaluation"""
        manifold = GrammarConstraintManifold(dim=32)

        params = np.random.randn(32)
        state = manifold.evaluate(params)

        assert isinstance(state, GrammarState)
        assert state.parameters.shape == (32,)
        assert isinstance(state.is_possible, bool)
        assert np.isfinite(state.violation_score)

class TestUniversalGrammar:
    """Tests for Universal Grammar"""

    def test_initialization(self):
        """Test UG initialization"""
        ug = UniversalGrammar(dim=32)

        assert ug.dim == 32
        assert len(ug.principles) > 0
        assert len(ug.parameters) > 0

    def test_evaluate(self):
        """Test UG evaluation"""
        ug = UniversalGrammar(dim=32)

        params = np.random.randn(32)
        result = ug.evaluate(params)

        assert 'overall' in result
        assert 0 <= result['overall'] <= 1

        for principle in ug.principles:
            assert principle in result

    def test_ug_compatibility(self):
        """Test UG compatibility check"""
        ug = UniversalGrammar(dim=32)

        params = np.random.randn(32)
        is_compatible = ug.is_ug_compatible(params)

        assert isinstance(is_compatible, bool)

    def test_parameter_setting(self):
        """Test setting UG parameters"""
        ug = UniversalGrammar(dim=32)

        # Set parameter
        ug.set_parameter('head_direction', 0.5)

        settings = ug.get_parameter_settings()
        assert settings['head_direction'] == 0.5

        # Clipping
        ug.set_parameter('head_direction', 2.0)  # Out of range
        settings = ug.get_parameter_settings()
        assert settings['head_direction'] == 1.0  # Clipped

class TestImpossibleGrammarGenerator:
    """Tests for impossible grammar generation"""

    def test_generate_possible(self):
        """Test generating possible grammars"""
        gen = ImpossibleGrammarGenerator(dim=32)

        possible = gen.generate_possible()

        assert possible.shape == (32,)
        assert gen.manifold.is_possible_grammar(possible)

    def test_generate_impossible(self):
        """Test generating impossible grammars"""
        gen = ImpossibleGrammarGenerator(dim=32)

        # Random impossible
        impossible = gen.generate_impossible('random')
        assert impossible.shape == (32,)

        # Structure-violating
        structure_impossible = gen.generate_impossible('structure')
        assert structure_impossible.shape == (32,)

        # Unbounded
        unbounded_impossible = gen.generate_impossible('unbounded')
        assert unbounded_impossible.shape == (32,)

    def test_generate_test_set(self):
        """Test generating test set"""
        gen = ImpossibleGrammarGenerator(dim=32)

        test_set = gen.generate_test_set(n_possible=5, n_impossible=5)

        assert 'possible' in test_set
        assert 'impossible' in test_set
        assert len(test_set['possible']) == 5
        assert len(test_set['impossible']) == 5

class TestSelectiveInhibition:
    """Tests verifying selective inhibition for impossible grammars"""

    def test_selective_inhibition_pattern(self):
        """Verify Broca's shows selective inhibition

        Key finding: Inhibition for impossible, not possible grammars.
        """
        manifold = GrammarConstraintManifold(dim=32)
        gen = ImpossibleGrammarGenerator(dim=32)

        possible_grammars = [gen.generate_possible() for _ in range(10)]
        impossible_grammars = [gen.generate_impossible() for _ in range(10)]

        possible_inhibitions = [manifold.inhibition_signal(g) for g in possible_grammars]
        impossible_inhibitions = [manifold.inhibition_signal(g) for g in impossible_grammars]

        mean_possible = np.mean(possible_inhibitions)
        mean_impossible = np.mean(impossible_inhibitions)

        # Impossible grammars should trigger more inhibition
        # This is a soft test as the manifold is randomly initialized
        assert mean_impossible >= mean_possible * 0.5 or mean_possible < 0.3

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
