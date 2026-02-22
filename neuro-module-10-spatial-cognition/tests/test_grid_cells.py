"""Tests for grid cells."""

import numpy as np
import pytest
from neuro.modules.m10_spatial_cognition.grid_cells import (
    GridCell,
    GridCellModule,
    GridCellPopulation,
    GridParameters,
)


class TestGridCell:
    """Test individual grid cell functionality."""

    def test_grid_cell_creation(self):
        cell = GridCell(spacing=0.3, orientation=0.0)
        assert cell.params.spacing == 0.3
        assert cell.params.orientation == 0.0

    def test_hexagonal_firing_pattern(self):
        cell = GridCell(spacing=0.3, orientation=0.0, peak_rate=15.0)

        # Grid cells should fire periodically
        rates = []
        for x in np.linspace(0, 1, 50):
            rate = cell.compute_firing(np.array([x, 0.0]))
            rates.append(rate)

        rates = np.array(rates)
        # Should have multiple peaks (periodic)
        peaks = np.where((rates[1:-1] > rates[:-2]) & (rates[1:-1] > rates[2:]))[0]
        assert len(peaks) >= 2  # At least 2 peaks in this range

    def test_grid_nodes_detection(self):
        cell = GridCell(spacing=0.3, orientation=0.0)
        nodes = cell.get_grid_nodes(environment_bounds=(0, 1, 0, 1), threshold=0.5)
        assert len(nodes) > 0


class TestGridCellModule:
    """Test grid cell module functionality."""

    def test_module_creation(self):
        module = GridCellModule(n_cells=20, spacing=0.3, orientation=0.0)
        assert len(module.cells) == 20
        for cell in module.cells:
            assert cell.params.spacing == 0.3

    def test_same_spacing_different_phases(self):
        module = GridCellModule(n_cells=10, spacing=0.3, random_seed=42)

        phases = [cell.params.phase for cell in module.cells]
        # Phases should be different
        for i in range(len(phases)):
            for j in range(i + 1, len(phases)):
                assert not np.allclose(phases[i], phases[j])

    def test_module_activity(self):
        module = GridCellModule(n_cells=20, spacing=0.3)
        activity = module.get_module_activity(np.array([0.5, 0.5]))
        assert len(activity) == 20
        assert np.all(activity >= 0)


class TestGridCellPopulation:
    """Test multi-scale grid cell population."""

    def test_population_creation(self):
        pop = GridCellPopulation(n_modules=4, cells_per_module=20, base_spacing=0.2)
        assert len(pop.modules) == 4
        assert pop.total_cells == 80

    def test_increasing_spacing(self):
        pop = GridCellPopulation(n_modules=4, base_spacing=0.2, scale_ratio=1.5)

        spacings = [m.spacing for m in pop.modules]
        # Each module should have larger spacing
        for i in range(len(spacings) - 1):
            assert spacings[i + 1] > spacings[i]

    def test_population_activity(self):
        pop = GridCellPopulation(n_modules=3, cells_per_module=10)
        activity = pop.get_population_activity(np.array([0.5, 0.5]))
        assert len(activity) == 30

    def test_path_integration(self):
        pop = GridCellPopulation(n_modules=3, cells_per_module=10)
        pop.reset_position(np.array([0.0, 0.0]))

        # Integrate some movement
        for _ in range(10):
            pop.path_integrate(velocity=np.array([0.1, 0.0]), dt=0.1)

        position = pop.get_position_estimate()
        expected = np.array([0.1, 0.0])
        error = np.linalg.norm(position - expected)
        assert error < 0.15  # Reasonable accuracy

    def test_firing_map_generation(self):
        pop = GridCellPopulation(n_modules=2, cells_per_module=5)
        X, Y, firing_map = pop.get_firing_map(
            cell_idx=0, environment_bounds=(0, 1, 0, 1), resolution=20
        )
        assert X.shape == (20, 20)
        assert firing_map.shape == (20, 20)


class TestGridParameters:
    """Test grid parameters dataclass."""

    def test_parameters_creation(self):
        params = GridParameters(
            spacing=0.3, orientation=np.pi / 6, phase=np.array([0.1, 0.1]), scale=1
        )
        assert params.spacing == 0.3
        assert params.orientation == pytest.approx(np.pi / 6)
        assert params.cell_id  # Should have ID
