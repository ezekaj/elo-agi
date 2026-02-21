"""Tests for place cells."""

import numpy as np
import pytest
from neuro.modules.m10_spatial_cognition.place_cells import PlaceCell, PlaceCellPopulation, PlaceField

class TestPlaceCell:
    """Test individual place cell functionality."""

    def test_place_cell_creation(self):
        cell = PlaceCell(center=np.array([0.5, 0.5]), radius=0.2)
        assert cell.place_field.center[0] == 0.5
        assert cell.place_field.center[1] == 0.5
        assert cell.place_field.radius == 0.2

    def test_firing_at_center(self):
        cell = PlaceCell(center=np.array([0.5, 0.5]), radius=0.2, peak_rate=20.0)
        rate = cell.compute_firing(np.array([0.5, 0.5]))
        assert rate == pytest.approx(20.0, rel=0.01)

    def test_firing_decreases_with_distance(self):
        cell = PlaceCell(center=np.array([0.5, 0.5]), radius=0.2)
        rate_center = cell.compute_firing(np.array([0.5, 0.5]))
        rate_edge = cell.compute_firing(np.array([0.7, 0.5]))
        rate_far = cell.compute_firing(np.array([1.0, 1.0]))

        assert rate_center > rate_edge > rate_far

    def test_is_in_field(self):
        cell = PlaceCell(center=np.array([0.5, 0.5]), radius=0.2)
        assert cell.is_in_field(np.array([0.5, 0.5]))
        assert not cell.is_in_field(np.array([0.0, 0.0]))

class TestPlaceCellPopulation:
    """Test place cell population functionality."""

    def test_population_creation(self):
        pop = PlaceCellPopulation(n_cells=50, environment_size=(1.0, 1.0))
        assert len(pop) == 50

    def test_get_population_activity(self):
        pop = PlaceCellPopulation(n_cells=50, environment_size=(1.0, 1.0))
        activity = pop.get_population_activity(np.array([0.5, 0.5]))
        assert len(activity) == 50
        assert np.all(activity >= 0)

    def test_decode_position(self):
        pop = PlaceCellPopulation(n_cells=100, environment_size=(1.0, 1.0), random_seed=42)
        true_pos = np.array([0.5, 0.5])
        activity = pop.get_population_activity(true_pos)
        decoded = pop.decode_position(activity)

        # Decoded should be close to true position
        error = np.linalg.norm(decoded - true_pos)
        assert error < 0.3  # Within 30% of environment

    def test_get_active_cells(self):
        pop = PlaceCellPopulation(n_cells=100, environment_size=(1.0, 1.0))
        active = pop.get_active_cells(np.array([0.5, 0.5]), threshold=0.1)
        assert len(active) > 0

    def test_remapping(self):
        pop = PlaceCellPopulation(n_cells=50, environment_size=(1.0, 1.0), random_seed=42)

        # Get activity before remapping
        activity_before = pop.get_population_activity(np.array([0.5, 0.5]))

        # Remap (full remapping)
        pop.remap(preserve_fraction=0.0)

        # Get activity after remapping
        activity_after = pop.get_population_activity(np.array([0.5, 0.5]))

        # Activity patterns should be different
        correlation = np.corrcoef(activity_before, activity_after)[0, 1]
        assert abs(correlation) < 0.8  # Significant change

    def test_activity_map_generation(self):
        pop = PlaceCellPopulation(n_cells=50, environment_size=(1.0, 1.0))
        X, Y, activity_map = pop.get_activity_map(resolution=20)
        assert X.shape == (20, 20)
        assert Y.shape == (20, 20)
        assert activity_map.shape == (20, 20)

class TestPlaceField:
    """Test place field dataclass."""

    def test_place_field_creation(self):
        pf = PlaceField(center=np.array([0.3, 0.7]), radius=0.15, peak_rate=25.0)
        assert pf.radius == 0.15
        assert pf.peak_rate == 25.0
        assert pf.cell_id  # Should have generated an ID
