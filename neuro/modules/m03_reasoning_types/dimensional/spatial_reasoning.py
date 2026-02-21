"""
Spatial Reasoning - Parietal Cortex Simulation

Mental manipulation of spatial representations including rotation,
transformation, navigation, and perspective-taking.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import heapq


class SpatialRelationType(Enum):
    ABOVE = "above"
    BELOW = "below"
    LEFT = "left"
    RIGHT = "right"
    FRONT = "front"
    BEHIND = "behind"
    INSIDE = "inside"
    OUTSIDE = "outside"
    NEAR = "near"
    FAR = "far"
    TOUCHING = "touching"
    OVERLAPPING = "overlapping"


@dataclass
class SpatialObject:
    """An object with spatial properties"""
    object_id: str
    position: np.ndarray
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    scale: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def __post_init__(self):
        self.position = np.array(self.position, dtype=float)
        if self.bounds is None:
            half_scale = self.scale / 2
            self.bounds = (self.position - half_scale, self.position + half_scale)

    def get_center(self) -> np.ndarray:
        return self.position.copy()

    def transformed_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box after rotation"""
        corners = []
        min_b, max_b = self.bounds
        for x in [min_b[0], max_b[0]]:
            for y in [min_b[1], max_b[1]]:
                for z in [min_b[2], max_b[2]]:
                    corner = np.array([x, y, z]) - self.position
                    rotated = self.rotation @ corner + self.position
                    corners.append(rotated)
        corners = np.array(corners)
        return corners.min(axis=0), corners.max(axis=0)


@dataclass
class SpatialRelation:
    """A relation between two spatial objects"""
    relation_type: SpatialRelationType
    object1_id: str
    object2_id: str
    strength: float = 1.0
    parameters: Dict = field(default_factory=dict)


class SpatialRelations:
    """Compute and represent spatial relationships"""

    def __init__(self, near_threshold: float = 2.0, touch_threshold: float = 0.1):
        self.near_threshold = near_threshold
        self.touch_threshold = touch_threshold

    def compute_relation(self,
                         obj1: SpatialObject,
                         obj2: SpatialObject
                         ) -> List[SpatialRelation]:
        """Compute all applicable relations between two objects"""
        relations = []
        diff = obj2.position - obj1.position
        distance = np.linalg.norm(diff)

        if distance < self.touch_threshold:
            relations.append(SpatialRelation(
                SpatialRelationType.TOUCHING, obj1.object_id, obj2.object_id,
                strength=1.0 - distance / self.touch_threshold
            ))
        elif distance < self.near_threshold:
            relations.append(SpatialRelation(
                SpatialRelationType.NEAR, obj1.object_id, obj2.object_id,
                strength=1.0 - distance / self.near_threshold
            ))
        else:
            relations.append(SpatialRelation(
                SpatialRelationType.FAR, obj1.object_id, obj2.object_id,
                strength=distance / self.near_threshold
            ))

        if abs(diff[0]) > abs(diff[1]) and abs(diff[0]) > abs(diff[2]):
            if diff[0] > 0:
                relations.append(SpatialRelation(
                    SpatialRelationType.RIGHT, obj1.object_id, obj2.object_id,
                    strength=abs(diff[0]) / (distance + 0.001)
                ))
            else:
                relations.append(SpatialRelation(
                    SpatialRelationType.LEFT, obj1.object_id, obj2.object_id,
                    strength=abs(diff[0]) / (distance + 0.001)
                ))

        if abs(diff[1]) > abs(diff[0]) and abs(diff[1]) > abs(diff[2]):
            if diff[1] > 0:
                relations.append(SpatialRelation(
                    SpatialRelationType.ABOVE, obj1.object_id, obj2.object_id,
                    strength=abs(diff[1]) / (distance + 0.001)
                ))
            else:
                relations.append(SpatialRelation(
                    SpatialRelationType.BELOW, obj1.object_id, obj2.object_id,
                    strength=abs(diff[1]) / (distance + 0.001)
                ))

        if abs(diff[2]) > abs(diff[0]) and abs(diff[2]) > abs(diff[1]):
            if diff[2] > 0:
                relations.append(SpatialRelation(
                    SpatialRelationType.FRONT, obj1.object_id, obj2.object_id,
                    strength=abs(diff[2]) / (distance + 0.001)
                ))
            else:
                relations.append(SpatialRelation(
                    SpatialRelationType.BEHIND, obj1.object_id, obj2.object_id,
                    strength=abs(diff[2]) / (distance + 0.001)
                ))

        min1, max1 = obj1.transformed_bounds()
        min2, max2 = obj2.transformed_bounds()

        overlap = np.all(max1 >= min2) and np.all(max2 >= min1)
        if overlap:
            relations.append(SpatialRelation(
                SpatialRelationType.OVERLAPPING, obj1.object_id, obj2.object_id
            ))

            if (np.all(min2 >= min1) and np.all(max2 <= max1)):
                relations.append(SpatialRelation(
                    SpatialRelationType.INSIDE, obj2.object_id, obj1.object_id
                ))
            elif (np.all(min1 >= min2) and np.all(max1 <= max2)):
                relations.append(SpatialRelation(
                    SpatialRelationType.INSIDE, obj1.object_id, obj2.object_id
                ))

        return relations

    def satisfy_constraints(self,
                            objects: Dict[str, SpatialObject],
                            constraints: List[SpatialRelation],
                            max_iterations: int = 100
                            ) -> Dict[str, SpatialObject]:
        """Arrange objects to satisfy spatial constraints"""
        objects = {k: SpatialObject(
            v.object_id,
            v.position.copy(),
            v.rotation.copy(),
            v.scale.copy()
        ) for k, v in objects.items()}

        for iteration in range(max_iterations):
            total_adjustment = 0

            for constraint in constraints:
                if constraint.object1_id not in objects or constraint.object2_id not in objects:
                    continue

                obj1 = objects[constraint.object1_id]
                obj2 = objects[constraint.object2_id]
                adjustment = self._compute_adjustment(obj1, obj2, constraint)

                obj2.position += adjustment
                total_adjustment += np.linalg.norm(adjustment)

            if total_adjustment < 0.001:
                break

        return objects

    def _compute_adjustment(self,
                            obj1: SpatialObject,
                            obj2: SpatialObject,
                            constraint: SpatialRelation
                            ) -> np.ndarray:
        """Compute position adjustment to satisfy a constraint"""
        adjustment = np.zeros(3)
        diff = obj2.position - obj1.position
        distance = np.linalg.norm(diff)

        strength = 0.1 * constraint.strength

        if constraint.relation_type == SpatialRelationType.ABOVE:
            if diff[1] <= 0:
                adjustment[1] = strength * (1.0 - diff[1])
        elif constraint.relation_type == SpatialRelationType.BELOW:
            if diff[1] >= 0:
                adjustment[1] = -strength * (1.0 + diff[1])
        elif constraint.relation_type == SpatialRelationType.LEFT:
            if diff[0] >= 0:
                adjustment[0] = -strength * (1.0 + diff[0])
        elif constraint.relation_type == SpatialRelationType.RIGHT:
            if diff[0] <= 0:
                adjustment[0] = strength * (1.0 - diff[0])
        elif constraint.relation_type == SpatialRelationType.NEAR:
            if distance > self.near_threshold:
                direction = diff / (distance + 0.001)
                adjustment = -direction * strength * (distance - self.near_threshold)
        elif constraint.relation_type == SpatialRelationType.FAR:
            if distance < self.near_threshold:
                direction = diff / (distance + 0.001) if distance > 0 else np.array([1, 0, 0])
                adjustment = direction * strength * (self.near_threshold - distance)

        return adjustment


class SpatialReasoner:
    """
    Mental manipulation of spatial representations.
    Simulates parietal cortex spatial processing.
    """

    def __init__(self):
        self.objects: Dict[str, SpatialObject] = {}
        self.relations = SpatialRelations()
        self.mental_rotation_rate = 1.0

    def add_object(self, obj: SpatialObject):
        """Add an object to the spatial representation"""
        self.objects[obj.object_id] = obj

    def mental_rotation(self,
                        object_id: str,
                        angle: float,
                        axis: np.ndarray = None
                        ) -> Tuple[SpatialObject, float]:
        """
        Mentally rotate an object.
        Returns rotated object and simulated "mental rotation time".

        Key insight: Mental rotation time is proportional to angle (Shepard & Metzler)
        """
        if object_id not in self.objects:
            raise ValueError(f"Object {object_id} not found")

        if axis is None:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis / np.linalg.norm(axis)

        rotation_time = abs(angle) / self.mental_rotation_rate

        c = np.cos(angle)
        s = np.sin(angle)
        x, y, z = axis

        rotation_matrix = np.array([
            [c + x*x*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s],
            [y*x*(1-c) + z*s, c + y*y*(1-c), y*z*(1-c) - x*s],
            [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)]
        ])

        obj = self.objects[object_id]
        rotated = SpatialObject(
            object_id=obj.object_id + "_rotated",
            position=obj.position.copy(),
            rotation=rotation_matrix @ obj.rotation,
            scale=obj.scale.copy()
        )

        return rotated, rotation_time

    def spatial_transformation(self,
                               object_id: str,
                               translation: np.ndarray = None,
                               rotation: np.ndarray = None,
                               scale: np.ndarray = None
                               ) -> SpatialObject:
        """Apply a general spatial transformation"""
        if object_id not in self.objects:
            raise ValueError(f"Object {object_id} not found")

        obj = self.objects[object_id]

        new_position = obj.position.copy()
        new_rotation = obj.rotation.copy()
        new_scale = obj.scale.copy()

        if translation is not None:
            new_position += np.array(translation)
        if rotation is not None:
            new_rotation = np.array(rotation) @ new_rotation
        if scale is not None:
            new_scale *= np.array(scale)

        return SpatialObject(
            object_id=obj.object_id + "_transformed",
            position=new_position,
            rotation=new_rotation,
            scale=new_scale
        )

    def navigation(self,
                   start: Tuple[float, float, float],
                   goal: Tuple[float, float, float],
                   obstacles: List[SpatialObject] = None,
                   grid_resolution: float = 0.5
                   ) -> List[np.ndarray]:
        """
        Plan a path from start to goal avoiding obstacles.
        Uses A* algorithm on a discretized grid.
        """
        start = np.array(start)
        goal = np.array(goal)

        if obstacles is None:
            obstacles = []

        def is_blocked(pos):
            for obs in obstacles:
                min_b, max_b = obs.transformed_bounds()
                if np.all(pos >= min_b) and np.all(pos <= max_b):
                    return True
            return False

        def heuristic(pos):
            return np.linalg.norm(pos - goal)

        def get_neighbors(pos):
            neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        neighbor = pos + np.array([dx, dy, dz]) * grid_resolution
                        if not is_blocked(neighbor):
                            neighbors.append(neighbor)
            return neighbors

        def pos_to_key(pos):
            return tuple(np.round(pos / grid_resolution).astype(int))

        open_set = [(heuristic(start), 0, start.tolist())]
        came_from = {}
        g_score = {pos_to_key(start): 0}

        while open_set:
            _, _, current_list = heapq.heappop(open_set)
            current = np.array(current_list)
            current_key = pos_to_key(current)

            if np.linalg.norm(current - goal) < grid_resolution:
                path = [current]
                while current_key in came_from:
                    current = came_from[current_key]
                    current_key = pos_to_key(current)
                    path.append(current)
                return list(reversed(path))

            for neighbor in get_neighbors(current):
                neighbor_key = pos_to_key(neighbor)
                tentative_g = g_score[current_key] + np.linalg.norm(neighbor - current)

                if neighbor_key not in g_score or tentative_g < g_score[neighbor_key]:
                    came_from[neighbor_key] = current
                    g_score[neighbor_key] = tentative_g
                    f_score = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor.tolist()))

        return [start, goal]

    def perspective_taking(self,
                           viewpoint: Tuple[float, float, float],
                           look_at: Tuple[float, float, float]
                           ) -> Dict[str, np.ndarray]:
        """
        Compute how objects appear from a different viewpoint.
        Returns transformed positions relative to the viewpoint.
        """
        viewpoint = np.array(viewpoint)
        look_at = np.array(look_at)

        forward = look_at - viewpoint
        forward = forward / np.linalg.norm(forward)

        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 0.001:
            world_up = np.array([0.0, 0.0, 1.0])
            right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        view_matrix = np.array([right, up, -forward])

        transformed = {}
        for obj_id, obj in self.objects.items():
            relative_pos = obj.position - viewpoint
            view_pos = view_matrix @ relative_pos
            transformed[obj_id] = view_pos

        return transformed

    def compare_objects(self,
                        obj1_id: str,
                        obj2_id: str,
                        allow_rotation: bool = True
                        ) -> Tuple[bool, float, float]:
        """
        Compare two objects for shape similarity.
        Returns (are_same, similarity_score, rotation_time).
        """
        if obj1_id not in self.objects or obj2_id not in self.objects:
            return False, 0.0, 0.0

        obj1 = self.objects[obj1_id]
        obj2 = self.objects[obj2_id]

        if not allow_rotation:
            similarity = 1.0 - np.linalg.norm(obj1.scale - obj2.scale) / np.linalg.norm(obj1.scale)
            return similarity > 0.9, max(0, similarity), 0.0

        best_similarity = 0.0
        total_rotation = 0.0

        for angle in np.linspace(0, 2 * np.pi, 36):
            rotated, rot_time = self.mental_rotation(obj1_id, angle)
            total_rotation += rot_time

            similarity = 1.0 - np.linalg.norm(rotated.scale - obj2.scale) / np.linalg.norm(obj1.scale)

            if similarity > best_similarity:
                best_similarity = similarity

            if similarity > 0.95:
                return True, similarity, total_rotation

        return best_similarity > 0.9, best_similarity, total_rotation
