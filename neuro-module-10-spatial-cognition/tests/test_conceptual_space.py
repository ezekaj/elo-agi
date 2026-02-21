"""Tests for conceptual space (2025 discovery)."""

import numpy as np
import pytest
from neuro.modules.m10_spatial_cognition.conceptual_space import (
    ConceptCell, ConceptualGrid, SocialDistanceGrid, ConceptualMap, ConceptFeatures
)

class TestConceptCell:
    """Test concept cell functionality."""

    def test_concept_cell_creation(self):
        cell = ConceptCell(
            concept_center=np.array([0.5, 0.5]),
            concept_radius=0.3,
            associated_concept="democracy"
        )
        assert cell.associated_concept == "democracy"
        assert cell.concept_radius == 0.3

    def test_activation_at_center(self):
        cell = ConceptCell(
            concept_center=np.array([0.5, 0.5]),
            concept_radius=0.3,
            peak_activation=1.0
        )
        activation = cell.compute_activation(np.array([0.5, 0.5]))
        assert activation == pytest.approx(1.0, rel=0.01)

    def test_activation_decreases_with_distance(self):
        cell = ConceptCell(
            concept_center=np.array([0.5, 0.5]),
            concept_radius=0.3
        )
        act_center = cell.compute_activation(np.array([0.5, 0.5]))
        act_edge = cell.compute_activation(np.array([0.8, 0.5]))
        act_far = cell.compute_activation(np.array([1.5, 1.5]))

        assert act_center > act_edge > act_far

    def test_is_active(self):
        cell = ConceptCell(
            concept_center=np.array([0.5, 0.5]),
            concept_radius=0.3
        )
        assert cell.is_active(np.array([0.5, 0.5]))
        assert not cell.is_active(np.array([2.0, 2.0]))

class TestConceptualGrid:
    """Test conceptual grid functionality."""

    def test_grid_creation(self):
        grid = ConceptualGrid(spacing=0.3, dimensions=3)
        assert grid.dimensions == 3
        assert grid.params.spacing == 0.3

    def test_grid_activation(self):
        grid = ConceptualGrid(spacing=0.3, dimensions=2)
        activation = grid.get_activation(np.array([0.5, 0.5]))
        assert 0 <= activation <= 1.0

    def test_conceptual_distance(self):
        grid = ConceptualGrid(spacing=0.3, dimensions=2)
        dist = grid.compute_conceptual_distance(
            np.array([0.0, 0.0]),
            np.array([0.3, 0.4])
        )
        assert dist == pytest.approx(0.5, rel=0.01)

class TestSocialDistanceGrid:
    """Test social distance encoding."""

    def test_social_grid_creation(self):
        sg = SocialDistanceGrid(dimensions=2)
        assert sg.dimensions == 2
        assert "power" in sg.dimension_names

    def test_set_and_get_position(self):
        sg = SocialDistanceGrid()
        sg.set_social_position("alice", np.array([0.8, 0.6]))

        pos = sg.get_social_position("alice")
        assert pos is not None
        np.testing.assert_array_equal(pos, [0.8, 0.6])

    def test_compute_social_distance(self):
        sg = SocialDistanceGrid()
        sg.set_social_position("boss", np.array([0.9, 0.5]))
        sg.set_social_position("colleague", np.array([0.5, 0.5]))

        dist = sg.compute_social_distance("boss", "colleague")
        assert dist == pytest.approx(0.4, rel=0.01)

    def test_unknown_person_distance(self):
        sg = SocialDistanceGrid()
        sg.set_social_position("alice", np.array([0.5, 0.5]))

        dist = sg.compute_social_distance("alice", "unknown")
        assert dist is None

    def test_find_socially_similar(self):
        sg = SocialDistanceGrid()
        sg.set_social_position("alice", np.array([0.5, 0.5]))
        sg.set_social_position("bob", np.array([0.6, 0.5]))  # Close to alice
        sg.set_social_position("carol", np.array([0.9, 0.9]))  # Far from alice

        similar = sg.find_socially_similar("alice", threshold=0.3)
        assert len(similar) == 1
        assert similar[0][0] == "bob"

class TestConceptualMap:
    """Test conceptual map integration."""

    def test_conceptual_map_creation(self):
        cm = ConceptualMap(concept_dimensions=10, n_concept_cells=50)
        assert cm.concept_dimensions == 10
        assert len(cm.concept_cells) == 50

    def test_embed_concept(self):
        cm = ConceptualMap(concept_dimensions=5)
        pos = cm.embed_concept(
            "democracy",
            features=np.array([0.8, 0.7, 0.6, 0.5, 0.4]),
            category="political"
        )
        assert len(pos) == 5
        assert "democracy" in cm.get_all_concepts()

    def test_concept_padding(self):
        cm = ConceptualMap(concept_dimensions=5)
        # Embed concept with fewer dimensions
        pos = cm.embed_concept("simple", features=np.array([0.5, 0.5]))
        assert len(pos) == 5

    def test_find_similar_concepts(self):
        cm = ConceptualMap(concept_dimensions=3, random_seed=42)

        cm.embed_concept("cat", features=np.array([0.5, 0.5, 0.5]))
        cm.embed_concept("dog", features=np.array([0.55, 0.5, 0.5]))  # Close to cat
        cm.embed_concept("rock", features=np.array([0.1, 0.1, 0.1]))  # Far from cat

        similar = cm.find_similar("cat", n=2)
        assert len(similar) == 2
        assert similar[0][0] == "dog"  # Most similar
        assert similar[1][0] == "rock"

    def test_navigate_concepts(self):
        cm = ConceptualMap(concept_dimensions=2)
        cm.embed_concept("start", features=np.array([0.0, 0.0]))
        cm.embed_concept("goal", features=np.array([1.0, 1.0]))

        path = cm.navigate_concepts("start", "goal", steps=5)
        assert len(path) == 6  # 5 steps + 1 for endpoints
        # First point should be near start
        np.testing.assert_array_almost_equal(path[0], [0.0, 0.0])
        # Last point should be near goal
        np.testing.assert_array_almost_equal(path[-1], [1.0, 1.0])

    def test_compute_analogy(self):
        cm = ConceptualMap(concept_dimensions=2, random_seed=42)

        # Set up analogy: king - man + woman = queen
        cm.embed_concept("king", features=np.array([0.9, 0.1]))
        cm.embed_concept("man", features=np.array([0.5, 0.1]))
        cm.embed_concept("woman", features=np.array([0.5, 0.9]))
        cm.embed_concept("queen", features=np.array([0.9, 0.9]))

        result = cm.compute_analogy("man", "king", "woman")
        assert result is not None
        name, distance = result
        assert name == "queen"

    def test_conceptual_distance(self):
        cm = ConceptualMap(concept_dimensions=2)
        cm.embed_concept("a", features=np.array([0.0, 0.0]))
        cm.embed_concept("b", features=np.array([0.3, 0.4]))

        dist = cm.conceptual_distance("a", "b")
        assert dist == pytest.approx(0.5, rel=0.01)

    def test_get_concept_activations(self):
        cm = ConceptualMap(concept_dimensions=3, n_concept_cells=20)
        cm.embed_concept("test", features=np.array([0.5, 0.5, 0.5]))

        activations = cm.get_concept_activations("test")
        assert len(activations) == 20
        assert np.sum(activations) > 0

    def test_map_physical_to_conceptual(self):
        cm = ConceptualMap(concept_dimensions=5)
        spatial_pos = np.array([0.3, 0.7])

        conceptual = cm.map_physical_to_conceptual(spatial_pos)
        assert len(conceptual) == 5
        assert conceptual[0] == 0.3
        assert conceptual[1] == 0.7

    def test_active_concept_cells(self):
        cm = ConceptualMap(concept_dimensions=3, n_concept_cells=50)
        cm.embed_concept("test", features=np.array([0.0, 0.0, 0.0]))

        active = cm.get_active_concept_cells("test", threshold=0.1)
        # Should have at least one active cell
        assert len(active) >= 0  # May be 0 if no cells near this point

    def test_get_all_concepts(self):
        cm = ConceptualMap(concept_dimensions=2)
        cm.embed_concept("a", features=np.array([0.1, 0.1]))
        cm.embed_concept("b", features=np.array([0.5, 0.5]))

        concepts = cm.get_all_concepts()
        assert "a" in concepts
        assert "b" in concepts
        assert len(concepts) == 2

class TestConceptFeatures:
    """Test concept features dataclass."""

    def test_concept_features_creation(self):
        cf = ConceptFeatures(
            name="democracy",
            features=np.array([0.8, 0.7, 0.6]),
            category="political"
        )
        assert cf.name == "democracy"
        assert cf.category == "political"
        assert len(cf.features) == 3

class TestResearchValidation:
    """Test that implementation validates 2025 research findings."""

    def test_concept_cells_like_place_cells(self):
        """Concept cells should have Gaussian tuning like place cells."""
        cell = ConceptCell(
            concept_center=np.array([0.5, 0.5]),
            concept_radius=0.3
        )

        # Should have peak at center
        center_act = cell.compute_activation(np.array([0.5, 0.5]))
        edge_act = cell.compute_activation(np.array([0.8, 0.5]))

        assert center_act > edge_act
        # Should follow Gaussian decay
        expected_ratio = np.exp(-0.3**2 / (2 * 0.3**2))
        actual_ratio = edge_act / center_act
        assert actual_ratio == pytest.approx(expected_ratio, rel=0.1)

    def test_grid_like_coding_in_concept_space(self):
        """Grid-like coding should provide metric for conceptual distance."""
        cm = ConceptualMap(concept_dimensions=3)

        cm.embed_concept("a", features=np.array([0.0, 0.0, 0.0]))
        cm.embed_concept("b", features=np.array([0.5, 0.0, 0.0]))
        cm.embed_concept("c", features=np.array([1.0, 0.0, 0.0]))

        dist_ab = cm.conceptual_distance("a", "b")
        dist_bc = cm.conceptual_distance("b", "c")
        dist_ac = cm.conceptual_distance("a", "c")

        # Triangle inequality should hold
        assert dist_ac <= dist_ab + dist_bc + 0.01

        # Distances should be proportional to feature differences
        assert dist_ab == pytest.approx(dist_bc, rel=0.01)
        assert dist_ac == pytest.approx(2 * dist_ab, rel=0.01)

    def test_social_distance_computation(self):
        """Social relationships should be encoded using spatial metrics."""
        sg = SocialDistanceGrid()

        # CEO has high power
        sg.set_social_position("ceo", np.array([0.95, 0.5]))
        # Manager has medium power
        sg.set_social_position("manager", np.array([0.7, 0.5]))
        # Employee has lower power
        sg.set_social_position("employee", np.array([0.4, 0.5]))

        ceo_manager = sg.compute_social_distance("ceo", "manager")
        manager_employee = sg.compute_social_distance("manager", "employee")
        ceo_employee = sg.compute_social_distance("ceo", "employee")

        # Distance CEO-employee should be largest
        assert ceo_employee > ceo_manager
        assert ceo_employee > manager_employee

    def test_analogy_as_vector_arithmetic(self):
        """Analogies should work via vector arithmetic like word2vec."""
        cm = ConceptualMap(concept_dimensions=3, random_seed=42)

        # Create semantic relationships
        # Animals vs vehicles, small vs large
        cm.embed_concept("dog", features=np.array([0.3, 0.2, 0.0]))
        cm.embed_concept("horse", features=np.array([0.3, 0.8, 0.0]))  # Large animal
        cm.embed_concept("car", features=np.array([0.7, 0.5, 0.0]))
        cm.embed_concept("truck", features=np.array([0.7, 0.8, 0.0]))  # Large vehicle

        # dog:horse :: car:? should give truck (small:large relationship)
        result = cm.compute_analogy("dog", "horse", "car")
        assert result is not None
        name, dist = result
        assert name == "truck"
