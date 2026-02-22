"""
Border Cells: Neurons encoding proximity to environmental boundaries

Based on Solstad et al. (2008) discovery of border cells.
These cells fire when the animal is near specific walls or edges
of the environment, anchoring the spatial map to boundaries.

Brain region: Medial Entorhinal Cortex (MEC), Subiculum
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class WallDirection(Enum):
    """Cardinal directions for walls"""

    NORTH = "north"  # y = y_max
    SOUTH = "south"  # y = y_min
    EAST = "east"  # x = x_max
    WEST = "west"  # x = x_min


@dataclass
class Wall:
    """A wall segment in the environment"""

    start: np.ndarray  # (x, y) start point
    end: np.ndarray  # (x, y) end point
    direction: WallDirection

    def distance_to(self, position: np.ndarray) -> float:
        """Compute perpendicular distance from position to wall"""
        # For axis-aligned walls (simplification)
        if self.direction in [WallDirection.NORTH, WallDirection.SOUTH]:
            return abs(position[1] - self.start[1])
        else:
            return abs(position[0] - self.start[0])


class BorderCell:
    """
    Single border cell that fires near a specific wall.

    Uses exponential decay from wall distance.
    """

    def __init__(
        self,
        preferred_wall: WallDirection,
        distance_tuning: float = 0.15,
        peak_rate: float = 25.0,
        firing_threshold: float = 0.3,
    ):
        self.preferred_wall = preferred_wall
        self.distance_tuning = distance_tuning  # Distance constant
        self.peak_rate = peak_rate
        self.firing_threshold = firing_threshold  # Max distance for firing
        self.firing_rate = 0.0

    def compute_firing(
        self, position: np.ndarray, environment_bounds: Tuple[float, float, float, float]
    ) -> float:
        """
        Compute firing rate based on distance to preferred wall.

        Uses exponential decay: rate = peak * exp(-distance / tuning)
        """
        x_min, x_max, y_min, y_max = environment_bounds
        position = np.array(position)

        # Compute distance to preferred wall
        if self.preferred_wall == WallDirection.NORTH:
            distance = y_max - position[1]
        elif self.preferred_wall == WallDirection.SOUTH:
            distance = position[1] - y_min
        elif self.preferred_wall == WallDirection.EAST:
            distance = x_max - position[0]
        elif self.preferred_wall == WallDirection.WEST:
            distance = position[0] - x_min
        else:
            distance = float("inf")

        # Only fire if within threshold distance
        if distance > self.firing_threshold:
            self.firing_rate = 0.0
        else:
            # Exponential decay from wall
            self.firing_rate = self.peak_rate * np.exp(-distance / self.distance_tuning)

        return self.firing_rate

    def is_near_wall(
        self, position: np.ndarray, environment_bounds: Tuple[float, float, float, float]
    ) -> bool:
        """Check if position is near the preferred wall"""
        rate = self.compute_firing(position, environment_bounds)
        return rate > 0.1 * self.peak_rate


class BorderCellPopulation:
    """
    Population of border cells encoding all environmental boundaries.

    Provides information about which walls are nearby and their distances.
    """

    def __init__(
        self,
        cells_per_wall: int = 5,
        environment_bounds: Tuple[float, float, float, float] = (0, 1, 0, 1),
        distance_tuning: float = 0.1,
    ):
        self.cells_per_wall = cells_per_wall
        self.environment_bounds = environment_bounds
        self.distance_tuning = distance_tuning
        self.cells: List[BorderCell] = []

        self._create_cells()

    def _create_cells(self) -> None:
        """Create border cells for all four walls"""
        self.cells = []

        for direction in WallDirection:
            for i in range(self.cells_per_wall):
                # Vary tuning distance slightly
                tuning = self.distance_tuning * (0.8 + 0.4 * i / self.cells_per_wall)

                cell = BorderCell(preferred_wall=direction, distance_tuning=tuning)
                self.cells.append(cell)

    def get_population_activity(self, position: np.ndarray) -> np.ndarray:
        """Get firing rates of all border cells"""
        return np.array(
            [cell.compute_firing(position, self.environment_bounds) for cell in self.cells]
        )

    def detect_boundary(self, position: np.ndarray, threshold: float = 0.3) -> List[WallDirection]:
        """Detect which boundaries are nearby"""
        activity = self.get_population_activity(position)

        nearby_walls = set()
        for cell, rate in zip(self.cells, activity):
            if rate > threshold * cell.peak_rate:
                nearby_walls.add(cell.preferred_wall)

        return list(nearby_walls)

    def get_distance_to_walls(self, position: np.ndarray) -> Dict[WallDirection, float]:
        """Get distances to all walls"""
        x_min, x_max, y_min, y_max = self.environment_bounds
        position = np.array(position)

        return {
            WallDirection.NORTH: y_max - position[1],
            WallDirection.SOUTH: position[1] - y_min,
            WallDirection.EAST: x_max - position[0],
            WallDirection.WEST: position[0] - x_min,
        }

    def get_nearest_wall(self, position: np.ndarray) -> Tuple[WallDirection, float]:
        """Get the nearest wall and its distance"""
        distances = self.get_distance_to_walls(position)
        nearest = min(distances, key=distances.get)
        return nearest, distances[nearest]

    def get_wall_activity_map(
        self, wall: WallDirection, resolution: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate activity map for cells preferring a specific wall.

        Returns X, Y meshgrid and combined activity map.
        """
        x_min, x_max, y_min, y_max = self.environment_bounds
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)

        activity_map = np.zeros((resolution, resolution))

        # Get cells for this wall
        wall_cells = [c for c in self.cells if c.preferred_wall == wall]

        for i in range(resolution):
            for j in range(resolution):
                pos = np.array([X[i, j], Y[i, j]])
                total_activity = sum(
                    c.compute_firing(pos, self.environment_bounds) for c in wall_cells
                )
                activity_map[i, j] = total_activity / len(wall_cells)

        return X, Y, activity_map

    def update_environment(self, new_bounds: Tuple[float, float, float, float]) -> None:
        """Update environment bounds (e.g., when environment changes)"""
        self.environment_bounds = new_bounds

    def __len__(self) -> int:
        return len(self.cells)
