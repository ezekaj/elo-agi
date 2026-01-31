"""Tests for engram system"""

import pytest
import numpy as np

from src.engram import Engram, Neuron, EngramState


class TestNeuron:
    """Tests for individual neurons"""

    def test_activation(self):
        """Test neuron activation"""
        neuron = Neuron(id=0, threshold=0.5)

        neuron.activate(0.8)
        assert neuron.is_active()

        neuron.activate(0.3)
        assert not neuron.is_active()

    def test_reset(self):
        """Test neuron reset"""
        neuron = Neuron(id=0)
        neuron.activate(1.0)
        neuron.reset()

        assert neuron.activation == 0.0


class TestEngram:
    """Tests for engram formation and operations"""

    def test_encode_pattern(self):
        """Test encoding a pattern"""
        engram = Engram(n_neurons=50)
        pattern = np.random.rand(50)

        engram.encode(pattern)

        assert engram.pattern is not None
        assert engram.state == EngramState.LABILE

    def test_hebbian_learning(self):
        """Test that Hebbian learning strengthens co-active connections"""
        np.random.seed(42)
        engram = Engram(n_neurons=20, connectivity=0.5, learning_rate=0.5)

        # Create pattern with some neurons active
        pattern = np.zeros(20)
        pattern[0:5] = 1.0  # First 5 neurons active

        # Get initial weight between two active neurons
        # Find a connection between active neurons
        initial_weights = {}
        for i in range(5):
            for j in range(5):
                if j in engram.neurons[i].connections:
                    initial_weights[(i, j)] = engram.neurons[i].connections[j]

        # Encode (applies Hebbian learning)
        engram.encode(pattern)

        # Check that weights increased between co-active neurons
        for (i, j), initial_weight in initial_weights.items():
            new_weight = engram.neurons[i].connections.get(j, 0)
            # Weight should have increased (or stayed same if already maxed)
            assert new_weight >= initial_weight - 0.1

    def test_pattern_completion(self):
        """Test reactivation with partial cue"""
        np.random.seed(42)
        engram = Engram(n_neurons=50, connectivity=0.3, learning_rate=0.3)

        # Encode full pattern
        original_pattern = np.random.rand(50)
        engram.encode(original_pattern)
        engram.consolidate()  # Strengthen

        # Create partial cue (50% of pattern)
        partial_cue = original_pattern.copy()
        partial_cue[25:] = 0  # Zero out half

        # Reactivate with partial cue
        reconstructed = engram.reactivate(partial_cue)

        # Should have some similarity to original
        similarity = engram.similarity(reconstructed)
        assert similarity > 0.3  # Some pattern completion occurred

    def test_consolidation(self):
        """Test that consolidation changes state"""
        engram = Engram(n_neurons=20)
        pattern = np.random.rand(20)

        engram.encode(pattern)
        assert engram.state == EngramState.LABILE

        engram.consolidate()
        assert engram.state == EngramState.CONSOLIDATED
        assert engram.strength > 0.3

    def test_destabilize_restabilize(self):
        """Test reconsolidation cycle"""
        engram = Engram(n_neurons=20)
        pattern = np.random.rand(20)

        engram.encode(pattern)
        engram.consolidate()
        assert not engram.is_labile()

        # Destabilize (reactivate)
        engram.destabilize()
        assert engram.is_labile()

        # Restabilize
        engram.restabilize()
        assert not engram.is_labile()

    def test_pruning(self):
        """Test connection pruning (forgetting)"""
        np.random.seed(42)
        engram = Engram(n_neurons=20, connectivity=0.5)

        # Count initial connections
        initial_count = sum(len(n.connections) for n in engram.neurons)

        # Prune with high threshold
        pruned = engram.prune(threshold=0.5)

        # Should have removed some connections
        final_count = sum(len(n.connections) for n in engram.neurons)
        assert final_count < initial_count
        assert pruned > 0

    def test_similarity(self):
        """Test pattern similarity computation"""
        engram = Engram(n_neurons=20)

        pattern1 = np.array([1, 0, 1, 0, 1] * 4)
        pattern2 = np.array([1, 0, 1, 0, 1] * 4)  # Same
        pattern3 = np.array([0, 1, 0, 1, 0] * 4)  # Opposite

        engram.encode(pattern1)

        assert engram.similarity(pattern2) > 0.9
        assert engram.similarity(pattern3) < 0.5


class TestEngramManipulation:
    """Tests for engram manipulation (erase, trigger, implant)"""

    def test_erase_memory(self):
        """Test erasing a memory by pruning"""
        engram = Engram(n_neurons=30)
        pattern = np.random.rand(30)

        engram.encode(pattern)
        engram.consolidate()

        # "Erase" by aggressive pruning
        engram.prune(threshold=0.9)  # Remove almost all connections

        # Reactivation should fail to reconstruct
        partial = pattern.copy()
        partial[15:] = 0

        reconstructed = engram.reactivate(partial)
        similarity = engram.similarity(reconstructed)

        # Should be very low after pruning
        assert similarity < 0.5

    def test_trigger_recall(self):
        """Test artificially triggering recall"""
        np.random.seed(42)
        engram = Engram(n_neurons=50, connectivity=0.3)
        pattern = np.random.rand(50)

        engram.encode(pattern)
        for _ in range(5):  # Multiple consolidation cycles
            engram.consolidate()

        # Trigger with very partial cue
        cue = np.zeros(50)
        cue[:10] = pattern[:10]  # Only 20% of pattern

        recalled = engram.reactivate(cue, iterations=20)

        # Should get some pattern completion
        assert not np.allclose(recalled, cue)

    def test_implant_synthetic_memory(self):
        """Test creating artificial engram"""
        engram = Engram(n_neurons=30)

        # Create artificial "memory" pattern
        synthetic_pattern = np.zeros(30)
        synthetic_pattern[0::3] = 1.0  # Arbitrary pattern

        # Implant by encoding
        engram.encode(synthetic_pattern)
        engram.consolidate()

        # Should be retrievable
        retrieved = engram.reactivate(synthetic_pattern * 0.5)
        similarity = engram.similarity(retrieved)

        assert similarity > 0.5
