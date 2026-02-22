"""
Visual Feature Extraction - Occipital Lobe Simulation

Hierarchical visual feature extraction without linguistic involvement.
Processes: edges → textures → shapes → objects
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class FeatureType(Enum):
    EDGE = "edge"
    TEXTURE = "texture"
    SHAPE = "shape"
    MOTION = "motion"
    COLOR = "color"


@dataclass
class Feature:
    """A detected visual feature"""

    feature_type: FeatureType
    location: Tuple[int, int]
    orientation: float = 0.0
    magnitude: float = 1.0
    scale: float = 1.0
    properties: Dict = field(default_factory=dict)

    def distance_to(self, other: "Feature") -> float:
        """Euclidean distance to another feature"""
        return np.sqrt(
            (self.location[0] - other.location[0]) ** 2
            + (self.location[1] - other.location[1]) ** 2
        )


@dataclass
class FeatureMap:
    """Spatial arrangement of detected features"""

    width: int
    height: int
    features: Dict[Tuple[int, int], List[Feature]] = field(default_factory=dict)

    def add_feature(self, feature: Feature):
        """Add a feature at its location"""
        loc = feature.location
        if loc not in self.features:
            self.features[loc] = []
        self.features[loc].append(feature)

    def query_region(self, x1: int, y1: int, x2: int, y2: int) -> List[Feature]:
        """Get all features within a bounding box"""
        result = []
        for (x, y), features in self.features.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                result.extend(features)
        return result

    def query_type(self, feature_type: FeatureType) -> List[Feature]:
        """Get all features of a specific type"""
        result = []
        for features in self.features.values():
            for f in features:
                if f.feature_type == feature_type:
                    result.append(f)
        return result

    def find_pattern(
        self, template: List[Feature], tolerance: float = 0.1
    ) -> List[Tuple[int, int]]:
        """Find locations where a pattern of features matches"""
        if not template:
            return []

        matches = []
        all_features = [f for features in self.features.values() for f in features]

        for anchor in all_features:
            if anchor.feature_type != template[0].feature_type:
                continue

            offset = (
                anchor.location[0] - template[0].location[0],
                anchor.location[1] - template[0].location[1],
            )

            matched = True
            for t_feature in template[1:]:
                expected_loc = (
                    t_feature.location[0] + offset[0],
                    t_feature.location[1] + offset[1],
                )

                found = False
                for f in all_features:
                    if (
                        f.feature_type == t_feature.feature_type
                        and abs(f.location[0] - expected_loc[0]) <= tolerance * self.width
                        and abs(f.location[1] - expected_loc[1]) <= tolerance * self.height
                    ):
                        found = True
                        break

                if not found:
                    matched = False
                    break

            if matched:
                matches.append(offset)

        return matches


class VisualFeatureExtractor:
    """
    Hierarchical visual feature extraction simulating occipital lobe.

    Processing hierarchy:
    1. Edge detection (V1)
    2. Texture analysis (V2)
    3. Shape recognition (V4)
    4. Motion detection (MT/V5)
    """

    def __init__(
        self, edge_threshold: float = 0.1, texture_window: int = 5, shape_min_edges: int = 3
    ):
        self.edge_threshold = edge_threshold
        self.texture_window = texture_window
        self.shape_min_edges = shape_min_edges

        self.sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    def extract_all(self, image: np.ndarray) -> FeatureMap:
        """Extract all feature types from an image"""
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        feature_map = FeatureMap(width=image.shape[1], height=image.shape[0])

        edges = self.extract_edges(gray)
        for edge in edges:
            feature_map.add_feature(edge)

        textures = self.extract_textures(gray)
        for texture in textures:
            feature_map.add_feature(texture)

        shapes = self.extract_shapes(edges)
        for shape in shapes:
            feature_map.add_feature(shape)

        return feature_map

    def extract_edges(self, image: np.ndarray) -> List[Feature]:
        """
        Extract edge features using gradient detection.
        Simulates V1 simple cells responding to oriented edges.
        """
        edges = []
        h, w = image.shape

        padded = np.pad(image, 1, mode="edge")

        gx = np.zeros_like(image)
        gy = np.zeros_like(image)

        for i in range(h):
            for j in range(w):
                region = padded[i : i + 3, j : j + 3]
                gx[i, j] = np.sum(region * self.sobel_x)
                gy[i, j] = np.sum(region * self.sobel_y)

        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx)

        max_mag = magnitude.max() if magnitude.max() > 0 else 1
        magnitude = magnitude / max_mag

        for i in range(h):
            for j in range(w):
                if magnitude[i, j] > self.edge_threshold:
                    edges.append(
                        Feature(
                            feature_type=FeatureType.EDGE,
                            location=(j, i),
                            orientation=orientation[i, j],
                            magnitude=magnitude[i, j],
                        )
                    )

        return edges

    def extract_textures(self, image: np.ndarray) -> List[Feature]:
        """
        Extract texture features using local statistics.
        Simulates V2 texture-selective neurons.
        """
        textures = []
        h, w = image.shape
        win = self.texture_window

        for i in range(0, h - win, win):
            for j in range(0, w - win, win):
                region = image[i : i + win, j : j + win]

                variance = np.var(region)
                mean_val = np.mean(region)

                dx = np.diff(region, axis=1)
                dy = np.diff(region, axis=0)
                gradient_energy = np.mean(dx**2) + np.mean(dy**2)

                textures.append(
                    Feature(
                        feature_type=FeatureType.TEXTURE,
                        location=(j + win // 2, i + win // 2),
                        magnitude=variance,
                        scale=float(win),
                        properties={
                            "variance": float(variance),
                            "mean": float(mean_val),
                            "gradient_energy": float(gradient_energy),
                            "smoothness": 1.0 / (1.0 + variance),
                        },
                    )
                )

        return textures

    def extract_shapes(self, edges: List[Feature]) -> List[Feature]:
        """
        Extract shape features by grouping edges.
        Simulates V4 shape-selective processing.
        """
        shapes = []

        if len(edges) < self.shape_min_edges:
            return shapes

        edge_locs = np.array([e.location for e in edges])
        edge_oris = np.array([e.orientation for e in edges])

        visited = set()

        for i, edge in enumerate(edges):
            if i in visited:
                continue

            cluster = [i]
            visited.add(i)

            for j, other in enumerate(edges):
                if j in visited:
                    continue

                dist = edge.distance_to(other)
                ori_diff = abs(edge.orientation - other.orientation)
                ori_diff = min(ori_diff, np.pi - ori_diff)

                if dist < 10 and ori_diff < np.pi / 4:
                    cluster.append(j)
                    visited.add(j)

            if len(cluster) >= self.shape_min_edges:
                cluster_edges = [edges[idx] for idx in cluster]
                center_x = np.mean([e.location[0] for e in cluster_edges])
                center_y = np.mean([e.location[1] for e in cluster_edges])

                orientations = [e.orientation for e in cluster_edges]
                ori_variance = np.var(orientations)

                if ori_variance < 0.1:
                    shape_type = "line"
                elif ori_variance < 0.5:
                    shape_type = "curve"
                else:
                    shape_type = "corner"

                shapes.append(
                    Feature(
                        feature_type=FeatureType.SHAPE,
                        location=(int(center_x), int(center_y)),
                        magnitude=len(cluster_edges),
                        properties={
                            "shape_type": shape_type,
                            "n_edges": len(cluster_edges),
                            "orientation_variance": float(ori_variance),
                        },
                    )
                )

        return shapes

    def extract_motion(self, image_sequence: List[np.ndarray]) -> List[Feature]:
        """
        Extract motion features from image sequence.
        Simulates MT/V5 motion-selective neurons.
        """
        if len(image_sequence) < 2:
            return []

        motion_features = []

        for t in range(1, len(image_sequence)):
            prev = image_sequence[t - 1].astype(float)
            curr = image_sequence[t].astype(float)

            if len(prev.shape) == 3:
                prev = np.mean(prev, axis=2)
                curr = np.mean(curr, axis=2)

            diff = curr - prev

            h, w = diff.shape
            block_size = 16

            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = diff[i : i + block_size, j : j + block_size]
                    motion_mag = np.sqrt(np.mean(block**2))

                    if motion_mag > 0.05:
                        gy, gx = np.gradient(block)
                        motion_dir = np.arctan2(np.mean(gy), np.mean(gx))

                        motion_features.append(
                            Feature(
                                feature_type=FeatureType.MOTION,
                                location=(j + block_size // 2, i + block_size // 2),
                                orientation=motion_dir,
                                magnitude=motion_mag,
                                properties={"frame": t, "velocity": float(motion_mag)},
                            )
                        )

        return motion_features
