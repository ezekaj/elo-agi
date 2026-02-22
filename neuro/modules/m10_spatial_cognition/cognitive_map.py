"""
Cognitive Map: Unified hippocampal spatial representation

Based on Tolman's (1948) cognitive map theory and O'Keefe & Nadel's (1978)
hippocampal spatial mapping framework.

The cognitive map integrates:
- Place cells (where am I?)
- Grid cells (how far have I moved?)
- Head direction cells (which way am I facing?)
- Border cells (where are the boundaries?)

Brain region: Hippocampus (integrates inputs from entorhinal cortex)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

from .place_cells import PlaceCellPopulation
from .grid_cells import GridCellPopulation
from .head_direction_cells import HeadDirectionSystem
from .border_cells import BorderCellPopulation, WallDirection
from .path_integration import PathIntegrator


@dataclass
class Landmark:
    """A recognizable landmark in the environment"""

    name: str
    position: np.ndarray
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Environment:
    """Spatial environment representation"""

    bounds: Tuple[float, float, float, float]  # x_min, x_max, y_min, y_max
    landmarks: List[Landmark] = field(default_factory=list)
    name: str = "default"

    @property
    def width(self) -> float:
        return self.bounds[1] - self.bounds[0]

    @property
    def height(self) -> float:
        return self.bounds[3] - self.bounds[2]

    @property
    def center(self) -> np.ndarray:
        return np.array(
            [(self.bounds[0] + self.bounds[1]) / 2, (self.bounds[2] + self.bounds[3]) / 2]
        )

    def is_valid_position(self, position: np.ndarray) -> bool:
        """Check if position is within bounds"""
        x, y = position
        return self.bounds[0] <= x <= self.bounds[1] and self.bounds[2] <= y <= self.bounds[3]

    def get_visible_landmarks(
        self, position: np.ndarray, heading: float, fov: float = np.pi
    ) -> List[Tuple[Landmark, float, float]]:
        """
        Get landmarks visible from position.

        Returns list of (landmark, distance, bearing) tuples.
        """
        visible = []

        for landmark in self.landmarks:
            displacement = landmark.position - position
            distance = np.linalg.norm(displacement)

            if distance < 1e-6:
                continue

            # Compute bearing relative to heading
            absolute_bearing = np.arctan2(displacement[1], displacement[0])
            relative_bearing = (absolute_bearing - heading) % (2 * np.pi)

            # Normalize to [-pi, pi]
            if relative_bearing > np.pi:
                relative_bearing -= 2 * np.pi

            # Check if within field of view
            if abs(relative_bearing) <= fov / 2:
                visible.append((landmark, distance, relative_bearing))

        return visible

    def add_landmark(self, name: str, position: np.ndarray, **features) -> Landmark:
        """Add a landmark to the environment"""
        landmark = Landmark(name=name, position=np.array(position), features=features)
        self.landmarks.append(landmark)
        return landmark


@dataclass
class CognitiveMapState:
    """Current state of the cognitive map"""

    position: np.ndarray
    heading: float
    place_activity: np.ndarray
    grid_activity: np.ndarray
    head_direction_activity: np.ndarray
    border_activity: np.ndarray
    nearby_walls: List[WallDirection]


class CognitiveMap:
    """
    Integrated spatial representation system.

    Combines all spatial cell types into a unified map
    that supports navigation, memory, and planning.
    """

    def __init__(
        self,
        environment: Optional[Environment] = None,
        n_place_cells: int = 200,
        n_grid_modules: int = 4,
        n_head_direction_cells: int = 60,
        random_seed: Optional[int] = None,
    ):
        # Default environment
        if environment is None:
            environment = Environment(bounds=(0, 1, 0, 1))

        self.environment = environment
        self.n_place_cells = n_place_cells

        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize cell populations
        env_size = (environment.width, environment.height)

        self.place_cells = PlaceCellPopulation(
            n_cells=n_place_cells, environment_size=env_size, random_seed=random_seed
        )

        self.grid_cells = GridCellPopulation(n_modules=n_grid_modules, random_seed=random_seed)

        self.head_direction = HeadDirectionSystem(n_cells=n_head_direction_cells)

        self.border_cells = BorderCellPopulation(environment_bounds=environment.bounds)

        # Path integrator for navigation
        self.path_integrator = PathIntegrator(
            grid_cells=self.grid_cells,
            head_direction=self.head_direction,
            initial_position=environment.center,
        )

        # Current state
        self._position = environment.center.copy()
        self._heading = 0.0

        # Memory of encoded locations
        self._encoded_locations: Dict[str, np.ndarray] = {}

    def update(
        self, velocity: np.ndarray, angular_velocity: float = 0.0, dt: float = 0.1
    ) -> CognitiveMapState:
        """
        Update cognitive map with movement.

        Uses path integration to update position estimate.
        """
        # Path integration
        if np.linalg.norm(velocity) > 1e-6 or abs(angular_velocity) > 1e-6:
            speed = np.linalg.norm(velocity)
            self._position, self._heading = self.path_integrator.integrate_with_heading(
                speed, angular_velocity, dt
            )
        else:
            self._position = self.path_integrator.position_estimate.copy()
            self._heading = self.path_integrator.heading_estimate

        # Clamp to environment bounds
        self._position = np.clip(
            self._position,
            [self.environment.bounds[0], self.environment.bounds[2]],
            [self.environment.bounds[1], self.environment.bounds[3]],
        )

        return self.get_state()

    def localize(self, sensory_input: Optional[Dict] = None) -> np.ndarray:
        """
        Estimate position from cell activities.

        Can incorporate sensory input for landmark correction.
        """
        # Get current activities
        place_activity = self.place_cells.get_population_activity(self._position)

        # Decode position from place cells
        decoded_position = self.place_cells.decode_position(place_activity)

        # If landmarks visible, correct estimate
        if sensory_input and "landmarks" in sensory_input:
            for landmark_name, observation in sensory_input["landmarks"].items():
                if landmark_name in [lm.name for lm in self.environment.landmarks]:
                    landmark = next(
                        lm for lm in self.environment.landmarks if lm.name == landmark_name
                    )
                    self.path_integrator.correct_with_landmark(
                        landmark.position, observation["distance"], observation["bearing"]
                    )
                    decoded_position = self.path_integrator.position_estimate

        return decoded_position

    def navigate_to(self, goal: np.ndarray, speed: float = 0.1) -> Tuple[np.ndarray, float]:
        """
        Compute velocity vector to navigate toward goal.

        Returns (velocity, distance_to_goal)
        """
        goal = np.array(goal)
        displacement = goal - self._position
        distance = np.linalg.norm(displacement)

        if distance < 1e-6:
            return np.zeros(2), 0.0

        # Direction to goal
        direction = displacement / distance

        # Velocity toward goal
        velocity = direction * min(speed, distance)

        return velocity, distance

    def encode_location(self, name: str, position: Optional[np.ndarray] = None) -> None:
        """
        Store current or specified location in memory.

        Creates a place-cell-based memory of the location.
        """
        if position is None:
            position = self._position.copy()

        self._encoded_locations[name] = np.array(position)

    def recall_location(self, name: str) -> Optional[np.ndarray]:
        """Retrieve a stored location"""
        return self._encoded_locations.get(name)

    def remap(self, new_environment: Environment) -> None:
        """
        Global remapping for new environment.

        Place cells completely reorganize their fields.
        """
        self.environment = new_environment

        # Update border cells
        self.border_cells.update_environment(new_environment.bounds)

        # Remap place cells (new random fields)
        self.place_cells.remap(preserve_fraction=0.0)

        # Reset position to center
        self._position = new_environment.center.copy()
        self._heading = 0.0

        self.path_integrator.reset(self._position, self._heading)

        # Clear encoded locations
        self._encoded_locations.clear()

    def get_state(self) -> CognitiveMapState:
        """Get current cognitive map state"""
        place_activity = self.place_cells.get_population_activity(self._position)
        grid_activity = self.grid_cells.get_population_activity(self._position)
        head_activity = self.head_direction.get_population_activity(self._heading)
        border_activity = self.border_cells.get_population_activity(self._position)
        nearby_walls = self.border_cells.detect_boundary(self._position)

        return CognitiveMapState(
            position=self._position.copy(),
            heading=self._heading,
            place_activity=place_activity,
            grid_activity=grid_activity,
            head_direction_activity=head_activity,
            border_activity=border_activity,
            nearby_walls=nearby_walls,
        )

    def get_position(self) -> np.ndarray:
        """Get current position"""
        return self._position.copy()

    def get_heading(self) -> float:
        """Get current heading in radians"""
        return self._heading

    def get_heading_degrees(self) -> float:
        """Get current heading in degrees"""
        return np.degrees(self._heading)

    def set_position(self, position: np.ndarray) -> None:
        """Directly set position (e.g., for teleportation/testing)"""
        self._position = np.array(position)
        self.path_integrator.reset(self._position, self._heading)

    def set_heading(self, heading: float) -> None:
        """Directly set heading"""
        self._heading = heading % (2 * np.pi)
        self.path_integrator.heading_estimate = self._heading
        self.head_direction.anchor_to_landmark(self._heading)

    def get_trajectory(self) -> np.ndarray:
        """Get recorded trajectory from path integrator"""
        return self.path_integrator.get_trajectory()

    def get_active_place_cells(self, threshold: float = 0.1) -> List:
        """Get place cells currently active above threshold"""
        return self.place_cells.get_active_cells(self._position, threshold)

    def distance_to_location(self, name: str) -> Optional[float]:
        """Get distance to a stored location"""
        loc = self.recall_location(name)
        if loc is None:
            return None
        return np.linalg.norm(loc - self._position)

    def bearing_to_location(self, name: str) -> Optional[float]:
        """Get bearing to a stored location (relative to current heading)"""
        loc = self.recall_location(name)
        if loc is None:
            return None

        displacement = loc - self._position
        if np.linalg.norm(displacement) < 1e-6:
            return 0.0

        absolute_bearing = np.arctan2(displacement[1], displacement[0])
        relative_bearing = (absolute_bearing - self._heading) % (2 * np.pi)

        if relative_bearing > np.pi:
            relative_bearing -= 2 * np.pi

        return relative_bearing
