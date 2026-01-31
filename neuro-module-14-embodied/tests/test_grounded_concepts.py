"""Tests for grounded concepts"""

import numpy as np
import pytest
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from src.grounded_concepts import (
    ModalityBindings, GroundedConcept, ConceptGrounding, GroundingParams
)


class TestModalityBindings:
    """Tests for modality bindings"""

    def test_initialization(self):
        """Test bindings initialization"""
        bindings = ModalityBindings(n_modalities=5, features_per_modality=10)

        assert bindings.n_modalities == 5
        assert len(bindings.bindings) == 0

    def test_add_binding(self):
        """Test adding modality binding"""
        bindings = ModalityBindings(features_per_modality=10)

        repr = np.random.rand(10)
        bindings.add_binding("apple", "visual", repr)

        assert "apple" in bindings.bindings
        assert "visual" in bindings.bindings["apple"]

    def test_get_binding(self):
        """Test retrieving binding"""
        bindings = ModalityBindings(features_per_modality=10)

        repr = np.random.rand(10)
        bindings.add_binding("apple", "visual", repr)

        retrieved = bindings.get_binding("apple", "visual")

        assert retrieved is not None
        assert len(retrieved) == 10

    def test_compute_similarity(self):
        """Test similarity computation"""
        bindings = ModalityBindings(features_per_modality=10)

        # Similar concepts
        repr1 = np.random.rand(10)
        bindings.add_binding("apple", "visual", repr1)
        bindings.add_binding("pear", "visual", repr1 + np.random.rand(10) * 0.1)

        similarity = bindings.compute_similarity("apple", "pear")

        assert 0 <= similarity <= 1


class TestGroundedConcept:
    """Tests for grounded concept"""

    def test_initialization(self):
        """Test concept initialization"""
        concept = GroundedConcept("apple")

        assert concept.name == "apple"
        assert concept.activation == 0.0

    def test_ground_in_experience(self):
        """Test grounding in sensory experience"""
        concept = GroundedConcept("apple")

        experience = np.random.rand(10)
        concept.ground_in_experience(experience, "visual")

        assert len(concept.experiences) == 1

    def test_activation(self):
        """Test concept activation"""
        concept = GroundedConcept("apple")

        concept.activate(0.5)

        assert concept.activation == 0.5

    def test_simulate_modality(self):
        """Test modal simulation"""
        concept = GroundedConcept("apple")
        concept.ground_in_experience(np.random.rand(10), "visual")
        concept.activate(1.0)

        simulation = concept.simulate_modality("visual")

        assert len(simulation) > 0

    def test_decay(self):
        """Test activation decay"""
        concept = GroundedConcept("apple")
        concept.activate(1.0)

        concept.decay(dt=10.0)

        assert concept.activation < 1.0

    def test_similarity(self):
        """Test concept similarity"""
        concept1 = GroundedConcept("apple")
        concept2 = GroundedConcept("pear")

        # Ground similarly
        exp = np.random.rand(10)
        concept1.ground_in_experience(exp, "visual")
        concept2.ground_in_experience(exp + np.random.rand(10) * 0.1, "visual")

        similarity = concept1.get_similarity(concept2)

        assert -1 <= similarity <= 1


class TestConceptGrounding:
    """Tests for concept grounding system"""

    def test_initialization(self):
        """Test system initialization"""
        system = ConceptGrounding()

        assert len(system.concepts) == 0

    def test_create_concept(self):
        """Test concept creation"""
        system = ConceptGrounding()

        concept = system.create_concept("apple")

        assert "apple" in system.concepts
        assert concept.name == "apple"

    def test_ground_concept(self):
        """Test grounding a concept"""
        system = ConceptGrounding()

        system.ground_concept("apple", np.random.rand(50), "visual")

        assert "apple" in system.concepts

    def test_activate_concept(self):
        """Test concept activation"""
        system = ConceptGrounding()
        system.ground_concept("apple", np.random.rand(50), "visual")

        simulations = system.activate_concept("apple", strength=0.8)

        assert "visual" in simulations
        assert "apple" in system.active_concepts

    def test_find_similar(self):
        """Test finding similar concepts"""
        system = ConceptGrounding()

        # Create similar concepts
        base = np.random.rand(50)
        system.ground_concept("apple", base, "visual")
        system.ground_concept("pear", base + np.random.rand(50) * 0.1, "visual")
        system.ground_concept("car", np.random.rand(50), "visual")  # Different

        similar = system.find_similar("apple", n=2)

        assert len(similar) <= 2
        # Pear should be more similar than car
        if len(similar) >= 1:
            assert isinstance(similar[0], tuple)

    def test_update(self):
        """Test system update"""
        system = ConceptGrounding()
        system.ground_concept("apple", np.random.rand(50), "visual")
        system.activate_concept("apple", 0.5)

        system.update(dt=10.0)

        # Activation should decay
        assert system.concepts["apple"].activation < 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
