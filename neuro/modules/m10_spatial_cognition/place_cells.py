"""
Place Cells: Hippocampal neurons encoding spatial location

Based on O'Keefe & Dostrovsky (1971) discovery and subsequent research.
Place cells fire when an animal is at a specific location in the environment.

Brain region: Hippocampus (CA1, CA3)
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import hashlib


@dataclass
class PlaceField:
    """Receptive field of a place cell - defines where it fires"""

    center: np.ndarray  # (x, y) center of firing field
    radius: float  # Field size (standard deviation of Gaussian)
    peak_rate: float = 20.0  # Maximum firing rate (Hz)
    cell_id: str = field(default="")

    def __post_init__(self):
        if not self.cell_id:
            content_str = f"{self.center.tobytes()}{self.radius}"
            self.cell_id = hashlib.md5(content_str.encode()).hexdigest()[:8]


class PlaceCell:
    """
    Single hippocampal place cell.

    Fires maximally when animal is at the center of its place field,
    with Gaussian falloff as distance increases.
    """

    def __init__(self, center: np.ndarray, radius: float = 0.3, peak_rate: float = 20.0):
        self.place_field = PlaceField(center=np.array(center), radius=radius, peak_rate=peak_rate)
        self.firing_rate = 0.0
        self._last_position = None

    def compute_firing(self, position: np.ndarray) -> float:
        """
        Compute firing rate based on distance to place field center.

        Uses Gaussian tuning: rate = peak * exp(-dist^2 / (2*sigma^2))
        """
        position = np.array(position)
        distance = np.linalg.norm(position - self.place_field.center)

        # Gaussian firing profile
        self.firing_rate = self.place_field.peak_rate * np.exp(
            -(distance**2) / (2 * self.place_field.radius**2)
        )
        self._last_position = position
        return self.firing_rate

    def is_in_field(self, position: np.ndarray, threshold: float = 0.1) -> bool:
        """Check if position is within place field (firing > threshold * peak)"""
        rate = self.compute_firing(position)
        return rate > threshold * self.place_field.peak_rate

    @property
    def cell_id(self) -> str:
        return self.place_field.cell_id


class PlaceCellPopulation:
    """
    Population of place cells covering an environment.

    Provides population-level decoding of position from
    distributed activity patterns.
    """

    def __init__(
        self,
        n_cells: int = 100,
        environment_size: Tuple[float, float] = (1.0, 1.0),
        field_radius: float = 0.15,
        random_seed: Optional[int] = None,
    ):
        self.n_cells = n_cells
        self.environment_size = environment_size
        self.field_radius = field_radius
        self.cells: List[PlaceCell] = []

        if random_seed is not None:
            np.random.seed(random_seed)

        self._create_random_fields()

    def _create_random_fields(self) -> None:
        """Initialize population with randomly placed fields"""
        self.cells = []

        for _ in range(self.n_cells):
            # Random center within environment
            center = np.array(
                [
                    np.random.uniform(0, self.environment_size[0]),
                    np.random.uniform(0, self.environment_size[1]),
                ]
            )

            # Some variation in field size
            radius = self.field_radius * np.random.uniform(0.8, 1.2)

            cell = PlaceCell(center=center, radius=radius)
            self.cells.append(cell)

    def get_population_activity(self, position: np.ndarray) -> np.ndarray:
        """Get firing rates of all cells at given position"""
        position = np.array(position)
        return np.array([cell.compute_firing(position) for cell in self.cells])

    def decode_position(self, activity: np.ndarray) -> np.ndarray:
        """
        Population vector decode: estimate position from activity pattern.

        Uses center-of-mass weighted by firing rates.
        """
        if len(activity) != len(self.cells):
            raise ValueError(f"Activity length {len(activity)} != {len(self.cells)} cells")

        total_activity = np.sum(activity)
        if total_activity < 1e-6:
            # No activity - return center of environment
            return np.array([self.environment_size[0] / 2, self.environment_size[1] / 2])

        # Weighted average of place field centers
        weighted_pos = np.zeros(2)
        for i, cell in enumerate(self.cells):
            weighted_pos += activity[i] * cell.place_field.center

        return weighted_pos / total_activity

    def get_active_cells(self, position: np.ndarray, threshold: float = 0.1) -> List[PlaceCell]:
        """Get cells firing above threshold at position"""
        activity = self.get_population_activity(position)
        max_rate = max(cell.place_field.peak_rate for cell in self.cells)

        return [cell for cell, rate in zip(self.cells, activity) if rate > threshold * max_rate]

    def remap(self, preserve_fraction: float = 0.0) -> None:
        """
        Global remapping: rearrange place fields for new environment.

        In new environments, place cells form completely new maps.
        preserve_fraction: fraction of cells to keep (partial remapping)
        """
        n_remap = int(self.n_cells * (1 - preserve_fraction))
        indices = np.random.choice(self.n_cells, n_remap, replace=False)

        for idx in indices:
            new_center = np.array(
                [
                    np.random.uniform(0, self.environment_size[0]),
                    np.random.uniform(0, self.environment_size[1]),
                ]
            )
            self.cells[idx].place_field.center = new_center

    def get_activity_map(self, resolution: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate 2D activity map for visualization.

        Returns x_grid, y_grid, activity_map
        """
        x = np.linspace(0, self.environment_size[0], resolution)
        y = np.linspace(0, self.environment_size[1], resolution)
        X, Y = np.meshgrid(x, y)

        activity_map = np.zeros((resolution, resolution))

        for i in range(resolution):
            for j in range(resolution):
                pos = np.array([X[i, j], Y[i, j]])
                activity = self.get_population_activity(pos)
                activity_map[i, j] = np.max(activity)  # Peak activity at location

        return X, Y, activity_map

    def __len__(self) -> int:
        return len(self.cells)
