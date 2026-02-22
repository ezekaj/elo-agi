"""
Grid Cells: Entorhinal cortex neurons with hexagonal firing patterns

Based on Moser & Moser (2005) Nobel Prize winning discovery.
Grid cells fire in a regular hexagonal lattice pattern as an animal
moves through space, providing a metric for distance and displacement.

Brain region: Medial Entorhinal Cortex (MEC)
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import hashlib


@dataclass
class GridParameters:
    """Parameters defining a grid cell's firing pattern"""

    spacing: float  # Distance between grid nodes (cm or normalized)
    orientation: float  # Rotation angle of grid (radians)
    phase: np.ndarray  # (x, y) offset of grid pattern
    scale: int = 1  # Grid scale level (1=finest, larger=coarser)
    cell_id: str = field(default="")

    def __post_init__(self):
        if not self.cell_id:
            content_str = f"{self.spacing}{self.orientation}{self.phase.tobytes()}"
            self.cell_id = hashlib.md5(content_str.encode()).hexdigest()[:8]


class GridCell:
    """
    Single entorhinal grid cell with hexagonal firing pattern.

    The cell fires whenever the animal is at any node of a
    regular hexagonal lattice tiling the environment.
    """

    def __init__(
        self,
        spacing: float = 0.3,
        orientation: float = 0.0,
        phase: Optional[np.ndarray] = None,
        scale: int = 1,
        peak_rate: float = 15.0,
    ):
        if phase is None:
            phase = np.zeros(2)

        self.params = GridParameters(
            spacing=spacing, orientation=orientation, phase=np.array(phase), scale=scale
        )
        self.peak_rate = peak_rate
        self.firing_rate = 0.0

    def compute_firing(self, position: np.ndarray) -> float:
        """
        Compute firing rate using hexagonal grid pattern.

        Uses sum of three cosine gratings at 60-degree angles
        to create hexagonal periodicity.
        """
        position = np.array(position) - self.params.phase

        # Rotate position by grid orientation
        cos_theta = np.cos(self.params.orientation)
        sin_theta = np.sin(self.params.orientation)
        rotated = np.array(
            [
                cos_theta * position[0] + sin_theta * position[1],
                -sin_theta * position[0] + cos_theta * position[1],
            ]
        )

        # Spatial frequency
        k = 2 * np.pi / self.params.spacing

        # Three gratings at 0, 60, 120 degrees create hexagonal pattern
        # Using the formula from Solstad et al. (2006)
        angles = [0, np.pi / 3, 2 * np.pi / 3]

        response = 0.0
        for angle in angles:
            direction = np.array([np.cos(angle), np.sin(angle)])
            response += np.cos(k * np.dot(rotated, direction))

        # Normalize to [0, 1] range (sum of 3 cosines ranges from -3 to 3)
        response = (response + 3) / 6

        # Apply nonlinearity to sharpen peaks
        response = response**2

        self.firing_rate = self.peak_rate * response
        return self.firing_rate

    def get_grid_nodes(
        self, environment_bounds: Tuple[float, float, float, float], threshold: float = 0.5
    ) -> List[np.ndarray]:
        """Get all grid node positions within environment bounds"""
        x_min, x_max, y_min, y_max = environment_bounds
        nodes = []

        # Sample environment and find peaks
        resolution = 100
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)

        for xi in x:
            for yi in y:
                pos = np.array([xi, yi])
                rate = self.compute_firing(pos)
                if rate > threshold * self.peak_rate:
                    nodes.append(pos)

        return nodes

    @property
    def cell_id(self) -> str:
        return self.params.cell_id


class GridCellModule:
    """
    Module of grid cells sharing the same spacing and orientation.

    Grid cells are organized into discrete modules where cells
    within a module have the same spacing and orientation but
    different phases.
    """

    def __init__(
        self,
        n_cells: int = 20,
        spacing: float = 0.3,
        orientation: float = 0.0,
        random_seed: Optional[int] = None,
    ):
        self.n_cells = n_cells
        self.spacing = spacing
        self.orientation = orientation
        self.cells: List[GridCell] = []

        if random_seed is not None:
            np.random.seed(random_seed)

        self._create_cells()

    def _create_cells(self) -> None:
        """Create cells with same spacing/orientation but different phases"""
        self.cells = []

        for _ in range(self.n_cells):
            # Random phase offset within one grid period
            phase = np.array(
                [np.random.uniform(0, self.spacing), np.random.uniform(0, self.spacing)]
            )

            cell = GridCell(spacing=self.spacing, orientation=self.orientation, phase=phase)
            self.cells.append(cell)

    def get_module_activity(self, position: np.ndarray) -> np.ndarray:
        """Get firing rates of all cells in module at position"""
        return np.array([cell.compute_firing(position) for cell in self.cells])


class GridCellPopulation:
    """
    Multi-scale grid cell system with multiple modules.

    Different modules have different spacings, creating a
    multi-resolution representation that supports efficient
    path integration and position coding.
    """

    def __init__(
        self,
        n_modules: int = 4,
        cells_per_module: int = 20,
        base_spacing: float = 0.2,
        scale_ratio: float = 1.4,
        random_seed: Optional[int] = None,
    ):
        self.n_modules = n_modules
        self.cells_per_module = cells_per_module
        self.base_spacing = base_spacing
        self.scale_ratio = scale_ratio
        self.modules: List[GridCellModule] = []

        if random_seed is not None:
            np.random.seed(random_seed)

        self._create_modules()

        # State for path integration
        self._position_estimate = np.zeros(2)
        self._last_activity = None

    def _create_modules(self) -> None:
        """Create modules with geometrically increasing spacings"""
        self.modules = []

        for i in range(self.n_modules):
            spacing = self.base_spacing * (self.scale_ratio**i)
            orientation = np.random.uniform(0, np.pi / 3)  # Random within 60 deg

            module = GridCellModule(
                n_cells=self.cells_per_module, spacing=spacing, orientation=orientation
            )
            self.modules.append(module)

    def get_population_activity(self, position: np.ndarray) -> np.ndarray:
        """Get firing rates of all cells across all modules"""
        activities = []
        for module in self.modules:
            activities.extend(module.get_module_activity(position))
        return np.array(activities)

    def compute_displacement(
        self, start_activity: np.ndarray, end_activity: np.ndarray
    ) -> np.ndarray:
        """
        Estimate displacement vector from change in population activity.

        Uses phase differences across modules to compute movement.
        """
        # Simplified: use center of mass of activity differences
        # In reality, this involves complex phase unwrapping

        total_cells = sum(len(m.cells) for m in self.modules)
        if len(start_activity) != total_cells or len(end_activity) != total_cells:
            raise ValueError("Activity vectors must match population size")

        # Compute weighted displacement estimate from each module
        displacement = np.zeros(2)
        total_weight = 0

        idx = 0
        for module in self.modules:
            module_start = start_activity[idx : idx + len(module.cells)]
            module_end = end_activity[idx : idx + len(module.cells)]
            idx += len(module.cells)

            # Weight by spacing (finer scales more precise for small movements)
            weight = 1.0 / module.spacing

            # Estimate displacement from activity change
            # This is a simplification - real decoding is more complex
            delta = np.sum(module_end - module_start)
            displacement += weight * delta * np.array([1, 1]) * module.spacing * 0.1

            total_weight += weight

        if total_weight > 0:
            displacement /= total_weight

        return displacement

    def path_integrate(self, velocity: np.ndarray, dt: float) -> np.ndarray:
        """
        Update position estimate using velocity (dead reckoning).

        This is how grid cells support navigation without landmarks.
        """
        velocity = np.array(velocity)
        self._position_estimate += velocity * dt
        return self._position_estimate.copy()

    def reset_position(self, position: np.ndarray) -> None:
        """Reset position estimate (e.g., when landmark is recognized)"""
        self._position_estimate = np.array(position)

    def get_position_estimate(self) -> np.ndarray:
        """Get current position estimate from path integration"""
        return self._position_estimate.copy()

    def get_firing_map(
        self,
        cell_idx: int,
        environment_bounds: Tuple[float, float, float, float] = (0, 1, 0, 1),
        resolution: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate 2D firing rate map for a specific cell.

        Returns X, Y meshgrid and firing rate map for visualization.
        """
        x_min, x_max, y_min, y_max = environment_bounds
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)

        # Find the cell
        all_cells = [cell for module in self.modules for cell in module.cells]
        if cell_idx >= len(all_cells):
            raise ValueError(f"Cell index {cell_idx} out of range")

        cell = all_cells[cell_idx]
        firing_map = np.zeros((resolution, resolution))

        for i in range(resolution):
            for j in range(resolution):
                pos = np.array([X[i, j], Y[i, j]])
                firing_map[i, j] = cell.compute_firing(pos)

        return X, Y, firing_map

    @property
    def total_cells(self) -> int:
        return sum(len(m.cells) for m in self.modules)

    def __len__(self) -> int:
        return self.total_cells
