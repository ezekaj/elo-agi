"""
Head Direction Cells: Neurons encoding compass-like facing direction

Based on Taube et al. (1990) discovery of head direction cells.
These neurons fire maximally when the animal faces a particular direction,
acting as an internal compass.

Brain regions: Postsubiculum, anterior thalamus, retrosplenial cortex
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class HeadDirectionTuning:
    """Tuning curve parameters for head direction cell"""

    preferred_direction: float  # Angle in radians (0 to 2*pi)
    tuning_width: float  # Concentration parameter (kappa) for von Mises
    peak_rate: float = 30.0  # Maximum firing rate (Hz)


class HeadDirectionCell:
    """
    Single head direction cell with von Mises tuning curve.

    Fires maximally when animal faces preferred direction,
    with cosine-like falloff for other directions.
    """

    def __init__(
        self, preferred_direction: float, tuning_width: float = 2.0, peak_rate: float = 30.0
    ):
        self.tuning = HeadDirectionTuning(
            preferred_direction=preferred_direction % (2 * np.pi),
            tuning_width=tuning_width,
            peak_rate=peak_rate,
        )
        self.firing_rate = 0.0

    def compute_firing(self, current_heading: float) -> float:
        """
        Compute firing rate based on current heading direction.

        Uses von Mises distribution (circular Gaussian).
        """
        current_heading = current_heading % (2 * np.pi)

        # Von Mises: exp(kappa * cos(theta - preferred))
        # Normalized to give peak_rate at preferred direction
        angle_diff = current_heading - self.tuning.preferred_direction
        kappa = self.tuning.tuning_width

        # von Mises response
        response = np.exp(kappa * (np.cos(angle_diff) - 1))

        self.firing_rate = self.tuning.peak_rate * response
        return self.firing_rate

    @property
    def preferred_direction_degrees(self) -> float:
        return np.degrees(self.tuning.preferred_direction)


class HeadDirectionSystem:
    """
    Ring attractor network for head direction representation.

    Models the head direction system as a ring of cells with
    bump attractor dynamics, maintaining a stable heading estimate.
    """

    def __init__(self, n_cells: int = 60, tuning_width: float = 2.0, peak_rate: float = 30.0):
        self.n_cells = n_cells
        self.tuning_width = tuning_width
        self.peak_rate = peak_rate
        self.cells: List[HeadDirectionCell] = []

        self._current_heading = 0.0
        self._heading_estimate = 0.0

        self._create_cells()

    def _create_cells(self) -> None:
        """Create ring of cells with evenly spaced preferred directions"""
        self.cells = []

        for i in range(self.n_cells):
            preferred = (2 * np.pi * i) / self.n_cells
            cell = HeadDirectionCell(
                preferred_direction=preferred,
                tuning_width=self.tuning_width,
                peak_rate=self.peak_rate,
            )
            self.cells.append(cell)

    def get_population_activity(self, heading: float) -> np.ndarray:
        """Get firing rates of all cells for given heading"""
        return np.array([cell.compute_firing(heading) for cell in self.cells])

    def decode_heading(self, activity: np.ndarray) -> float:
        """
        Population vector decode: estimate heading from activity.

        Uses circular mean weighted by firing rates.
        """
        if len(activity) != len(self.cells):
            raise ValueError(f"Activity length {len(activity)} != {len(self.cells)} cells")

        # Compute weighted circular mean
        sin_sum = 0.0
        cos_sum = 0.0

        for i, cell in enumerate(self.cells):
            angle = cell.tuning.preferred_direction
            sin_sum += activity[i] * np.sin(angle)
            cos_sum += activity[i] * np.cos(angle)

        self._heading_estimate = np.arctan2(sin_sum, cos_sum) % (2 * np.pi)
        return self._heading_estimate

    def update_heading(self, angular_velocity: float, dt: float) -> float:
        """
        Integrate angular velocity to update heading estimate.

        This is path integration for direction (angular integration).
        """
        self._current_heading += angular_velocity * dt
        self._current_heading = self._current_heading % (2 * np.pi)
        return self._current_heading

    def anchor_to_landmark(self, landmark_direction: float) -> None:
        """
        Reset heading based on visual landmark.

        Corrects drift accumulated during path integration.
        """
        self._current_heading = landmark_direction % (2 * np.pi)
        self._heading_estimate = self._current_heading

    def get_current_heading(self) -> float:
        """Get current heading in radians"""
        return self._current_heading

    def get_heading_degrees(self) -> float:
        """Get current heading in degrees"""
        return np.degrees(self._current_heading)

    def get_tuning_curves(self, resolution: int = 360) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get tuning curves for all cells.

        Returns angles and firing rates matrix for visualization.
        """
        angles = np.linspace(0, 2 * np.pi, resolution)
        rates = np.zeros((self.n_cells, resolution))

        for i, cell in enumerate(self.cells):
            for j, angle in enumerate(angles):
                rates[i, j] = cell.compute_firing(angle)

        return angles, rates

    def get_population_vector(self) -> Tuple[float, float]:
        """
        Get population vector representation.

        Returns (direction, magnitude) of summed activity.
        """
        activity = self.get_population_activity(self._current_heading)

        sin_sum = 0.0
        cos_sum = 0.0

        for i, cell in enumerate(self.cells):
            angle = cell.tuning.preferred_direction
            sin_sum += activity[i] * np.sin(angle)
            cos_sum += activity[i] * np.cos(angle)

        direction = np.arctan2(sin_sum, cos_sum) % (2 * np.pi)
        magnitude = np.sqrt(sin_sum**2 + cos_sum**2)

        return direction, magnitude

    def __len__(self) -> int:
        return len(self.cells)
