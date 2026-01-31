"""
Path Integration: Dead reckoning navigation using grid cells

Path integration allows animals to keep track of their position
by integrating self-motion signals (velocity, rotation) over time.
Grid cells are believed to be the neural substrate for this computation.

Also known as: Dead reckoning, odometry
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

from .grid_cells import GridCellPopulation
from .head_direction_cells import HeadDirectionSystem


@dataclass
class PathIntegrationState:
    """Current state of path integration system"""
    position: np.ndarray
    heading: float
    uncertainty: float
    time: float


class PathIntegrator:
    """
    Velocity-based position updating using grid cells.

    Maintains a position estimate that drifts without
    external correction from landmarks.
    """

    def __init__(
        self,
        grid_cells: Optional[GridCellPopulation] = None,
        head_direction: Optional[HeadDirectionSystem] = None,
        initial_position: Optional[np.ndarray] = None,
        initial_heading: float = 0.0,
        noise_scale: float = 0.01,
        uncertainty_growth: float = 0.001
    ):
        self.grid_cells = grid_cells or GridCellPopulation()
        self.head_direction = head_direction or HeadDirectionSystem()

        self.position_estimate = np.array(initial_position) if initial_position is not None else np.zeros(2)
        self.heading_estimate = initial_heading
        self.uncertainty = 0.0

        self.noise_scale = noise_scale
        self.uncertainty_growth = uncertainty_growth
        self.time = 0.0

        # History for trajectory
        self._trajectory: List[np.ndarray] = [self.position_estimate.copy()]
        self._heading_history: List[float] = [self.heading_estimate]

    def integrate(self, velocity: np.ndarray, dt: float) -> np.ndarray:
        """
        Update position estimate from velocity.

        velocity: (vx, vy) or speed in heading direction
        dt: time step
        """
        velocity = np.array(velocity)

        # Add integration noise (path integration is not perfect)
        noise = np.random.randn(2) * self.noise_scale * np.sqrt(dt)

        # Update position
        self.position_estimate += velocity * dt + noise

        # Update uncertainty
        self.uncertainty += self.uncertainty_growth * dt

        # Update time
        self.time += dt

        # Record trajectory
        self._trajectory.append(self.position_estimate.copy())

        # Update grid cells
        self.grid_cells.path_integrate(velocity, dt)

        return self.position_estimate.copy()

    def integrate_with_heading(
        self,
        speed: float,
        angular_velocity: float,
        dt: float
    ) -> Tuple[np.ndarray, float]:
        """
        Update position using speed and heading.

        Converts speed + heading to velocity vector.
        """
        # Update heading first
        self.heading_estimate += angular_velocity * dt
        self.heading_estimate = self.heading_estimate % (2 * np.pi)
        self._heading_history.append(self.heading_estimate)

        # Update head direction system
        self.head_direction.update_heading(angular_velocity, dt)

        # Convert speed + heading to velocity
        velocity = speed * np.array([
            np.cos(self.heading_estimate),
            np.sin(self.heading_estimate)
        ])

        position = self.integrate(velocity, dt)
        return position, self.heading_estimate

    def reset(self, position: np.ndarray, heading: Optional[float] = None) -> None:
        """
        Reset position estimate (anchor to known location).

        Called when recognizing a landmark or starting position.
        """
        self.position_estimate = np.array(position)
        self.uncertainty = 0.0

        if heading is not None:
            self.heading_estimate = heading
            self.head_direction.anchor_to_landmark(heading)

        self.grid_cells.reset_position(position)

        # Reset trajectory
        self._trajectory = [self.position_estimate.copy()]
        if heading is not None:
            self._heading_history = [heading]

    def get_displacement(
        self,
        start_position: np.ndarray,
        end_position: np.ndarray
    ) -> np.ndarray:
        """Compute displacement vector between two positions"""
        return np.array(end_position) - np.array(start_position)

    def correct_with_landmark(
        self,
        landmark_position: np.ndarray,
        observed_distance: float,
        observed_bearing: float
    ) -> None:
        """
        Correct position estimate using landmark observation.

        landmark_position: known position of landmark
        observed_distance: measured distance to landmark
        observed_bearing: measured bearing to landmark (radians)
        """
        # Compute expected position based on landmark
        expected_position = landmark_position - observed_distance * np.array([
            np.cos(self.heading_estimate + observed_bearing),
            np.sin(self.heading_estimate + observed_bearing)
        ])

        # Blend current estimate with landmark-based estimate
        # Weight by uncertainty (high uncertainty = trust landmark more)
        alpha = min(0.8, self.uncertainty * 10)
        self.position_estimate = (
            (1 - alpha) * self.position_estimate +
            alpha * expected_position
        )

        # Reduce uncertainty after landmark correction
        self.uncertainty *= (1 - alpha)

    def get_state(self) -> PathIntegrationState:
        """Get current path integration state"""
        return PathIntegrationState(
            position=self.position_estimate.copy(),
            heading=self.heading_estimate,
            uncertainty=self.uncertainty,
            time=self.time
        )

    def get_trajectory(self) -> np.ndarray:
        """Get recorded trajectory as array"""
        return np.array(self._trajectory)

    def get_total_distance(self) -> float:
        """Compute total distance traveled"""
        trajectory = np.array(self._trajectory)
        if len(trajectory) < 2:
            return 0.0

        distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        return np.sum(distances)

    def get_displacement_from_start(self) -> np.ndarray:
        """Get current displacement from starting position"""
        if len(self._trajectory) < 1:
            return np.zeros(2)
        return self.position_estimate - self._trajectory[0]


class VectorBasedNavigator:
    """
    Vector-based navigation for goal-directed movement.

    Uses path integration to maintain position and compute
    vectors to remembered goal locations.
    """

    def __init__(
        self,
        path_integrator: Optional[PathIntegrator] = None
    ):
        self.integrator = path_integrator or PathIntegrator()
        self.goal_positions: dict = {}  # Named goal locations

    def set_goal(self, name: str, position: np.ndarray) -> None:
        """Remember a goal location"""
        self.goal_positions[name] = np.array(position)

    def get_vector_to_goal(self, goal_name: str) -> Tuple[float, float]:
        """
        Get distance and direction to a remembered goal.

        Returns (distance, bearing) from current position.
        """
        if goal_name not in self.goal_positions:
            raise ValueError(f"Unknown goal: {goal_name}")

        goal = self.goal_positions[goal_name]
        current = self.integrator.position_estimate

        displacement = goal - current
        distance = np.linalg.norm(displacement)

        if distance < 1e-6:
            bearing = 0.0
        else:
            # Bearing relative to current heading
            absolute_bearing = np.arctan2(displacement[1], displacement[0])
            bearing = (absolute_bearing - self.integrator.heading_estimate) % (2 * np.pi)

        return distance, bearing

    def navigate_step(
        self,
        goal_name: str,
        speed: float,
        dt: float
    ) -> Tuple[np.ndarray, bool]:
        """
        Take one step toward goal.

        Returns (new_position, arrived)
        """
        distance, bearing = self.get_vector_to_goal(goal_name)

        if distance < 0.05:  # Arrived threshold
            return self.integrator.position_estimate.copy(), True

        # Turn toward goal
        turn_rate = np.clip(bearing, -np.pi/4, np.pi/4) / dt

        # Move forward
        position, _ = self.integrator.integrate_with_heading(speed, turn_rate, dt)

        return position, False

    def update(self, velocity: np.ndarray, dt: float) -> np.ndarray:
        """Update position with movement"""
        return self.integrator.integrate(velocity, dt)
