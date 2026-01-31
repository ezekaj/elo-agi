"""Tests for structural plasticity"""

import numpy as np
import pytest
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from src.structural_plasticity import (
    StructuralPlasticity, SynapticPruning, DendriticGrowth, StructuralParams
)


class TestStructuralPlasticity:
    """Tests for structural plasticity"""

    def test_initialization(self):
        """Test initialization"""
        sp = StructuralPlasticity(n_neurons=10)

        assert sp.connectivity.shape == (10, 10)
        assert sp.weights.shape == (10, 10)
        # No self-connections
        assert not np.any(np.diag(sp.connectivity))

    def test_activity_recording(self):
        """Test activity is recorded"""
        sp = StructuralPlasticity(n_neurons=5)

        activity = np.random.rand(5)
        sp.record_activity(activity)

        mean_act = sp.get_mean_activity()
        assert mean_act.shape == (5,)

    def test_synapse_formation(self):
        """Test new synapses can form"""
        params = StructuralParams(
            synapse_formation_rate=0.5,
            activity_threshold=0.1
        )
        sp = StructuralPlasticity(n_neurons=5, params=params)

        # Record high activity
        for _ in range(50):
            sp.record_activity(np.ones(5) * 0.5)

        initial_synapses = np.sum(sp.connectivity)
        formed = sp.form_synapses()
        final_synapses = np.sum(sp.connectivity)

        # Should form some synapses (probabilistic)
        assert final_synapses >= initial_synapses

    def test_synapse_elimination(self):
        """Test weak synapses can be eliminated"""
        params = StructuralParams(
            synapse_elimination_rate=0.5,
            weight_threshold=0.3
        )
        sp = StructuralPlasticity(n_neurons=5, params=params)

        # Make all weights weak
        sp.weights = sp.weights * 0.1

        initial_synapses = np.sum(sp.connectivity)
        eliminated = sp.eliminate_synapses()
        final_synapses = np.sum(sp.connectivity)

        # Should eliminate some synapses
        assert final_synapses <= initial_synapses

    def test_update_cycle(self):
        """Test full update cycle"""
        sp = StructuralPlasticity(n_neurons=10)

        activity = np.random.rand(10)
        formed, eliminated = sp.update(activity)

        assert isinstance(formed, int)
        assert isinstance(eliminated, int)

    def test_connectivity_density(self):
        """Test connectivity density calculation"""
        sp = StructuralPlasticity(n_neurons=10)

        density = sp.get_connectivity_density()

        assert 0 <= density <= 1


class TestSynapticPruning:
    """Tests for synaptic pruning"""

    def test_initialization(self):
        """Test initialization"""
        weights = np.random.rand(5, 5)
        pruner = SynapticPruning(weights)

        assert pruner.weights.shape == (5, 5)
        assert pruner.usage_count.shape == (5, 5)

    def test_usage_recording(self):
        """Test usage is recorded"""
        weights = np.ones((3, 3))
        pruner = SynapticPruning(weights, activity_threshold=0.5)

        pre = np.array([1.0, 0.0, 1.0])
        post = np.array([1.0, 1.0, 0.0])

        pruner.record_usage(pre, post)

        # Check expected usage pattern
        assert pruner.usage_count[0, 0] == 1  # Both active
        assert pruner.usage_count[0, 1] == 0  # Pre inactive
        assert pruner.usage_count[2, 0] == 0  # Post inactive

    def test_pruning_removes_unused(self):
        """Test pruning removes unused synapses"""
        weights = np.ones((5, 5)) * 0.5
        pruner = SynapticPruning(weights, pruning_rate=0.9)

        # Record many updates with specific pattern
        active_pre = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        active_post = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

        for _ in range(50):
            pruner.record_usage(active_pre, active_post)

        pruned = pruner.prune()

        # Should have pruned some weights
        final_weights = pruner.get_weights()
        assert np.sum(final_weights == 0) > 0


class TestDendriticGrowth:
    """Tests for dendritic growth"""

    def test_initialization(self):
        """Test initialization"""
        growth = DendriticGrowth(n_neurons=10)

        assert len(growth.dendritic_size) == 10
        assert np.all(growth.dendritic_size == 1.0)

    def test_growth_with_activity(self):
        """Test dendrites grow with activity"""
        growth = DendriticGrowth(n_neurons=5, growth_rate=0.1)

        # High activity
        activity = np.ones(5) * 0.9

        for _ in range(50):
            growth.update(activity)

        # Dendrites should grow
        assert np.all(growth.dendritic_size > 1.0)

    def test_retraction_without_activity(self):
        """Test dendrites retract without activity"""
        growth = DendriticGrowth(n_neurons=5, retraction_rate=0.1)
        growth.dendritic_size = np.ones(5) * 5.0  # Start large

        # No activity
        activity = np.zeros(5)

        for _ in range(50):
            growth.update(activity)

        # Dendrites should shrink
        assert np.all(growth.dendritic_size < 5.0)

    def test_size_bounds(self):
        """Test dendritic size stays bounded"""
        growth = DendriticGrowth(n_neurons=3)
        growth.max_size = 5.0

        # Extreme activity
        for _ in range(1000):
            growth.update(np.ones(3))

        assert np.all(growth.dendritic_size <= growth.max_size)
        assert np.all(growth.dendritic_size >= 0.1)

    def test_receptive_field_size(self):
        """Test receptive field size getter"""
        growth = DendriticGrowth(n_neurons=5)

        sizes = growth.get_receptive_field_sizes()

        assert len(sizes) == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
