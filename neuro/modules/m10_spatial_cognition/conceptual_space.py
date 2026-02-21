"""
Conceptual Space: Abstract concept mapping using spatial cell machinery

Based on 2025 research showing that hippocampal spatial cells also encode
abstract concepts. The same neural machinery used for physical navigation
supports navigation through conceptual space.

Key findings:
- "Concept cells" function like place cells for abstract ideas
- Grid-like coding measures conceptual distance
- Social relationships encoded using spatial metrics

Brain region: Hippocampus (extends beyond physical space)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import hashlib

from .grid_cells import GridParameters


@dataclass
class ConceptFeatures:
    """Feature vector representing a concept"""
    name: str
    features: np.ndarray  # High-dimensional feature vector
    category: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConceptCell:
    """
    Place cell analog for concepts.

    Fires maximally when "thinking about" a specific concept,
    with Gaussian falloff for related concepts.
    """

    def __init__(
        self,
        concept_center: np.ndarray,
        concept_radius: float = 0.3,
        associated_concept: str = "",
        peak_activation: float = 1.0
    ):
        self.concept_center = np.array(concept_center)
        self.concept_radius = concept_radius
        self.associated_concept = associated_concept
        self.peak_activation = peak_activation
        self.activation = 0.0

        # Generate ID
        content_str = f"{concept_center.tobytes()}{associated_concept}"
        self.cell_id = hashlib.md5(content_str.encode()).hexdigest()[:8]

    def compute_activation(self, concept_position: np.ndarray) -> float:
        """
        Compute activation based on distance in concept space.

        Like place cells but in abstract space.
        """
        concept_position = np.array(concept_position)
        distance = np.linalg.norm(concept_position - self.concept_center)

        # Gaussian activation
        self.activation = self.peak_activation * np.exp(
            -distance**2 / (2 * self.concept_radius**2)
        )
        return self.activation

    def is_active(self, concept_position: np.ndarray, threshold: float = 0.1) -> bool:
        """Check if cell is active for this concept"""
        return self.compute_activation(concept_position) > threshold * self.peak_activation


class ConceptualGrid:
    """
    Grid cell analog for conceptual distance.

    Provides a metric in concept space, allowing measurement
    of "conceptual distance" between ideas.
    """

    def __init__(
        self,
        spacing: float = 0.3,
        orientation: float = 0.0,
        dimensions: int = 2,
        peak_activation: float = 1.0
    ):
        self.params = GridParameters(
            spacing=spacing,
            orientation=orientation,
            phase=np.zeros(dimensions),
            scale=1
        )
        self.dimensions = dimensions
        self.peak_activation = peak_activation
        self.activation = 0.0

    def get_activation(self, concept_position: np.ndarray) -> float:
        """
        Compute grid-like activation in concept space.

        Generalizes hexagonal grid to arbitrary dimensions.
        """
        concept_position = np.array(concept_position)

        if len(concept_position) < 2:
            concept_position = np.pad(concept_position, (0, 2 - len(concept_position)))

        # Use first 2 dimensions for grid computation
        pos_2d = concept_position[:2] - self.params.phase[:2]

        # Rotate
        cos_theta = np.cos(self.params.orientation)
        sin_theta = np.sin(self.params.orientation)
        rotated = np.array([
            cos_theta * pos_2d[0] + sin_theta * pos_2d[1],
            -sin_theta * pos_2d[0] + cos_theta * pos_2d[1]
        ])

        # Spatial frequency
        k = 2 * np.pi / self.params.spacing

        # Three gratings for hexagonal pattern
        angles = [0, np.pi/3, 2*np.pi/3]
        response = 0.0
        for angle in angles:
            direction = np.array([np.cos(angle), np.sin(angle)])
            response += np.cos(k * np.dot(rotated, direction))

        # Normalize
        response = (response + 3) / 6
        response = response ** 2

        self.activation = self.peak_activation * response
        return self.activation

    def compute_conceptual_distance(
        self,
        concept_a: np.ndarray,
        concept_b: np.ndarray
    ) -> float:
        """Compute distance between concepts in this grid's metric"""
        return np.linalg.norm(np.array(concept_a) - np.array(concept_b))


class SocialDistanceGrid:
    """
    Grid coding for social relationships.

    Encodes social distance along dimensions like:
    - Power/hierarchy
    - Affiliation/closeness
    - Trust
    """

    def __init__(
        self,
        dimensions: int = 2,
        dimension_names: Optional[List[str]] = None
    ):
        self.dimensions = dimensions
        self.dimension_names = dimension_names or ["power", "affiliation"]

        # Social positions of known entities
        self._social_positions: Dict[str, np.ndarray] = {}

        # Grid for measuring social distance
        self._grid = ConceptualGrid(
            spacing=0.25,
            dimensions=dimensions
        )

    def set_social_position(
        self,
        person: str,
        position: np.ndarray
    ) -> None:
        """Set someone's position in social space"""
        self._social_positions[person] = np.array(position)

    def get_social_position(self, person: str) -> Optional[np.ndarray]:
        """Get someone's position in social space"""
        return self._social_positions.get(person)

    def compute_social_distance(
        self,
        person_a: str,
        person_b: str
    ) -> Optional[float]:
        """Compute social distance between two people"""
        pos_a = self._social_positions.get(person_a)
        pos_b = self._social_positions.get(person_b)

        if pos_a is None or pos_b is None:
            return None

        return np.linalg.norm(pos_a - pos_b)

    def get_social_activation(self, person: str) -> float:
        """Get grid-like activation for a person's social position"""
        pos = self._social_positions.get(person)
        if pos is None:
            return 0.0
        return self._grid.get_activation(pos)

    def find_socially_similar(
        self,
        person: str,
        threshold: float = 0.3
    ) -> List[Tuple[str, float]]:
        """Find people who are socially similar"""
        pos = self._social_positions.get(person)
        if pos is None:
            return []

        similar = []
        for other, other_pos in self._social_positions.items():
            if other == person:
                continue
            distance = np.linalg.norm(pos - other_pos)
            if distance < threshold:
                similar.append((other, distance))

        return sorted(similar, key=lambda x: x[1])


class ConceptualMap:
    """
    Hippocampal mapping of abstract knowledge.

    Uses the same spatial machinery as physical navigation
    to organize and navigate through concept space.
    """

    def __init__(
        self,
        concept_dimensions: int = 10,
        n_concept_cells: int = 100,
        n_grids: int = 3,
        random_seed: Optional[int] = None
    ):
        self.concept_dimensions = concept_dimensions
        self.n_concept_cells = n_concept_cells

        if random_seed is not None:
            np.random.seed(random_seed)

        # Concept cells covering concept space
        self.concept_cells: List[ConceptCell] = []
        self._create_concept_cells()

        # Conceptual grids for distance measurement
        self.conceptual_grids: List[ConceptualGrid] = []
        self._create_grids(n_grids)

        # Social distance grid
        self.social_grid = SocialDistanceGrid(dimensions=2)

        # Stored concepts
        self._concepts: Dict[str, ConceptFeatures] = {}
        self._concept_positions: Dict[str, np.ndarray] = {}

    def _create_concept_cells(self) -> None:
        """Create concept cells with random centers in concept space"""
        self.concept_cells = []

        for _ in range(self.n_concept_cells):
            center = np.random.randn(self.concept_dimensions) * 0.5

            cell = ConceptCell(
                concept_center=center,
                concept_radius=0.3
            )
            self.concept_cells.append(cell)

    def _create_grids(self, n_grids: int) -> None:
        """Create conceptual grids at different scales"""
        self.conceptual_grids = []

        for i in range(n_grids):
            spacing = 0.2 * (1.4 ** i)  # Increasing spacing
            orientation = np.random.uniform(0, np.pi / 3)

            grid = ConceptualGrid(
                spacing=spacing,
                orientation=orientation,
                dimensions=self.concept_dimensions
            )
            self.conceptual_grids.append(grid)

    def embed_concept(
        self,
        name: str,
        features: np.ndarray,
        category: str = "",
        **metadata
    ) -> np.ndarray:
        """
        Embed a concept into conceptual space.

        Returns position in concept space.
        """
        features = np.array(features)

        # Pad or truncate to concept_dimensions
        if len(features) < self.concept_dimensions:
            features = np.pad(features, (0, self.concept_dimensions - len(features)))
        else:
            features = features[:self.concept_dimensions]

        # Store concept
        concept = ConceptFeatures(
            name=name,
            features=features,
            category=category,
            metadata=metadata
        )
        self._concepts[name] = concept

        # Position is the feature vector (could be transformed)
        position = features.copy()
        self._concept_positions[name] = position

        # Update concept cells if any match this concept
        for cell in self.concept_cells:
            if cell.is_active(position, threshold=0.5):
                if not cell.associated_concept:
                    cell.associated_concept = name

        return position

    def get_concept_position(self, name: str) -> Optional[np.ndarray]:
        """Get a concept's position in concept space"""
        return self._concept_positions.get(name)

    def find_similar(
        self,
        concept_name: str,
        n: int = 5
    ) -> List[Tuple[str, float]]:
        """Find n most similar concepts"""
        position = self._concept_positions.get(concept_name)
        if position is None:
            return []

        distances = []
        for name, pos in self._concept_positions.items():
            if name == concept_name:
                continue
            dist = np.linalg.norm(position - pos)
            distances.append((name, dist))

        distances.sort(key=lambda x: x[1])
        return distances[:n]

    def navigate_concepts(
        self,
        start: str,
        goal: str,
        steps: int = 10
    ) -> List[np.ndarray]:
        """
        Compute path through concept space from start to goal.

        Returns list of intermediate positions.
        """
        start_pos = self._concept_positions.get(start)
        goal_pos = self._concept_positions.get(goal)

        if start_pos is None or goal_pos is None:
            return []

        # Linear interpolation through concept space
        path = []
        for i in range(steps + 1):
            t = i / steps
            pos = start_pos * (1 - t) + goal_pos * t
            path.append(pos)

        return path

    def compute_analogy(
        self,
        a: str,
        b: str,
        c: str
    ) -> Optional[Tuple[str, float]]:
        """
        Compute analogy: a:b :: c:?

        Uses spatial relations: ? = c + (b - a)
        """
        pos_a = self._concept_positions.get(a)
        pos_b = self._concept_positions.get(b)
        pos_c = self._concept_positions.get(c)

        if pos_a is None or pos_b is None or pos_c is None:
            return None

        # Analogy vector
        target_pos = pos_c + (pos_b - pos_a)

        # Find closest concept to target
        best_match = None
        best_distance = float('inf')

        for name, pos in self._concept_positions.items():
            if name in [a, b, c]:
                continue
            dist = np.linalg.norm(pos - target_pos)
            if dist < best_distance:
                best_distance = dist
                best_match = name

        if best_match:
            return (best_match, best_distance)
        return None

    def get_concept_activations(
        self,
        concept_name: str
    ) -> np.ndarray:
        """Get concept cell activations for a concept"""
        position = self._concept_positions.get(concept_name)
        if position is None:
            return np.zeros(len(self.concept_cells))

        return np.array([
            cell.compute_activation(position)
            for cell in self.concept_cells
        ])

    def get_active_concept_cells(
        self,
        concept_name: str,
        threshold: float = 0.1
    ) -> List[ConceptCell]:
        """Get concept cells active for a concept"""
        position = self._concept_positions.get(concept_name)
        if position is None:
            return []

        return [
            cell for cell in self.concept_cells
            if cell.is_active(position, threshold)
        ]

    def conceptual_distance(self, concept_a: str, concept_b: str) -> Optional[float]:
        """Compute conceptual distance between two concepts"""
        pos_a = self._concept_positions.get(concept_a)
        pos_b = self._concept_positions.get(concept_b)

        if pos_a is None or pos_b is None:
            return None

        return np.linalg.norm(pos_a - pos_b)

    def map_physical_to_conceptual(
        self,
        spatial_position: np.ndarray,
        scaling: float = 1.0
    ) -> np.ndarray:
        """
        Transfer from physical space to concept space.

        Demonstrates shared neural machinery.
        """
        # Pad spatial position to concept dimensions
        spatial_position = np.array(spatial_position)
        if len(spatial_position) < self.concept_dimensions:
            conceptual = np.zeros(self.concept_dimensions)
            conceptual[:len(spatial_position)] = spatial_position * scaling
        else:
            conceptual = spatial_position[:self.concept_dimensions] * scaling

        return conceptual

    def get_all_concepts(self) -> List[str]:
        """Get list of all stored concept names"""
        return list(self._concepts.keys())

    def __len__(self) -> int:
        return len(self._concepts)
