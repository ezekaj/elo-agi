"""
V4 and IT: Higher visual areas for shape and object recognition.

Implements shape selectivity, view-invariant object representation,
and categorical coding.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import numpy as np


@dataclass
class ShapeDescriptor:
    """Description of a visual shape."""

    curvature: np.ndarray  # Curvature values around contour
    angles: np.ndarray  # Angles at key points
    aspect_ratio: float
    convexity: float  # 0-1, how convex
    symmetry: float  # 0-1, bilateral symmetry
    complexity: float  # Number of inflection points
    embedding: Optional[np.ndarray] = None


@dataclass
class ObjectRepresentation:
    """Representation of a recognized object."""

    identity: str
    confidence: float
    features: Dict[str, float]
    embedding: np.ndarray
    parts: List["ObjectRepresentation"] = field(default_factory=list)
    viewpoint: Optional[Tuple[float, float, float]] = None  # yaw, pitch, roll


@dataclass
class V4Output:
    """Output from V4 processing."""

    shape_map: np.ndarray  # Shape-selective responses
    curvature_map: np.ndarray  # Detailed curvature
    color_shape: np.ndarray  # Color-form conjunctions
    intermediate_forms: List[ShapeDescriptor] = field(default_factory=list)
    size: Tuple[int, int] = (0, 0)


@dataclass
class ITOutput:
    """Output from IT cortex processing."""

    object_responses: np.ndarray  # Responses to object categories
    identity_embeddings: np.ndarray  # View-invariant embeddings
    detected_objects: List[ObjectRepresentation] = field(default_factory=list)
    size: Tuple[int, int] = (0, 0)


class V4Processor:
    """
    V4 visual area processing.

    Implements:
    - Curvature selectivity
    - Shape selectivity
    - Color-form conjunctions
    - Intermediate complexity features
    """

    def __init__(
        self,
        n_curvature_cells: int = 8,
        n_shape_cells: int = 16,
        embedding_dim: int = 64,
    ):
        self.n_curvature_cells = n_curvature_cells
        self.n_shape_cells = n_shape_cells
        self.embedding_dim = embedding_dim

        # Curvature tuning
        self._curvature_prefs = np.linspace(-1, 1, n_curvature_cells)

        # Shape templates (simplified)
        self._shape_templates = self._create_shape_templates()

    def _create_shape_templates(self) -> List[np.ndarray]:
        """Create template shapes for matching."""
        templates = []

        # Basic shapes as 16x16 binary masks
        size = 16

        # Circle
        y, x = np.ogrid[-size // 2 : size // 2, -size // 2 : size // 2]
        circle = (x**2 + y**2) < (size // 3) ** 2
        templates.append(circle.astype(float))

        # Square
        square = np.ones((size, size))
        square[:3, :] = 0
        square[-3:, :] = 0
        square[:, :3] = 0
        square[:, -3:] = 0
        templates.append(square)

        # Triangle
        triangle = np.zeros((size, size))
        for i in range(size):
            width = int((i / size) * size * 0.8)
            start = (size - width) // 2
            triangle[i, start : start + max(1, width)] = 1
        templates.append(triangle)

        # Add more basic shapes
        # Horizontal line
        h_line = np.zeros((size, size))
        h_line[size // 2 - 1 : size // 2 + 1, 2:-2] = 1
        templates.append(h_line)

        # Vertical line
        v_line = np.zeros((size, size))
        v_line[2:-2, size // 2 - 1 : size // 2 + 1] = 1
        templates.append(v_line)

        return templates

    def process(self, v2_output) -> V4Output:
        """
        Process V2 output through V4.

        Args:
            v2_output: V2Output from V2 processing

        Returns:
            V4Output with shape-selective responses
        """
        h, w = v2_output.size

        # Curvature map from V2
        curvature_map = v2_output.curvature_map

        # Enhanced curvature selectivity
        curvature_selective = self._curvature_selectivity(curvature_map)

        # Shape selectivity
        shape_map = self._shape_selectivity(v2_output.contour_map)

        # Color-shape (simplified - just edge info for now)
        color_shape = v2_output.contour_map * 0.5 + shape_map * 0.5

        # Extract intermediate forms
        intermediate_forms = self._extract_shapes(
            v2_output.contour_map,
            curvature_map,
        )

        return V4Output(
            shape_map=shape_map,
            curvature_map=curvature_selective,
            color_shape=color_shape,
            intermediate_forms=intermediate_forms,
            size=(h, w),
        )

    def _curvature_selectivity(
        self,
        curvature_map: np.ndarray,
    ) -> np.ndarray:
        """
        Compute curvature-selective cell responses.
        """
        h, w = curvature_map.shape
        responses = np.zeros((h, w, self.n_curvature_cells))

        # Normalize curvature to [-1, 1]
        curv_norm = curvature_map / (curvature_map.max() + 1e-8) * 2 - 1

        for i, pref in enumerate(self._curvature_prefs):
            # Gaussian tuning around preferred curvature
            responses[:, :, i] = np.exp(-((curv_norm - pref) ** 2) / 0.5)

        return np.max(responses, axis=2)

    def _shape_selectivity(
        self,
        contour_map: np.ndarray,
    ) -> np.ndarray:
        """
        Compute shape-selective responses using template matching.
        """
        from scipy.ndimage import convolve

        h, w = contour_map.shape
        responses = np.zeros((h, w, len(self._shape_templates)))

        for i, template in enumerate(self._shape_templates):
            # Normalize template
            template_norm = template - template.mean()
            template_norm = template_norm / (np.sqrt(np.sum(template_norm**2)) + 1e-8)

            # Cross-correlation (simple template matching)
            response = convolve(contour_map, template_norm, mode="constant")
            responses[:, :, i] = np.maximum(0, response)

        return np.max(responses, axis=2)

    def _extract_shapes(
        self,
        contour_map: np.ndarray,
        curvature_map: np.ndarray,
    ) -> List[ShapeDescriptor]:
        """
        Extract shape descriptors from contour map.
        """
        shapes = []

        # Find contour regions using thresholding
        threshold = contour_map.max() * 0.3
        regions = contour_map > threshold

        # Get connected components (simplified)
        from scipy.ndimage import label

        labeled, n_regions = label(regions)

        for region_id in range(1, min(n_regions + 1, 10)):  # Limit to 10 regions
            mask = labeled == region_id
            if mask.sum() < 10:  # Skip tiny regions
                continue

            # Compute shape properties
            y_coords, x_coords = np.where(mask)
            if len(y_coords) == 0:
                continue

            # Bounding box
            y_min, y_max = y_coords.min(), y_coords.max()
            x_min, x_max = x_coords.min(), x_coords.max()

            height = y_max - y_min + 1
            width = x_max - x_min + 1

            aspect_ratio = width / max(height, 1)
            convexity = mask.sum() / max((height * width), 1)

            # Curvature along region
            region_curv = curvature_map[mask]

            # Symmetry (simplified - compare left/right halves)
            center_x = (x_min + x_max) // 2
            left = mask[:, :center_x] if center_x > 0 else np.zeros((1, 1))
            right = mask[:, center_x:] if center_x < mask.shape[1] else np.zeros((1, 1))
            right_flipped = np.fliplr(right)

            # Resize to match
            min_w = min(left.shape[1], right_flipped.shape[1])
            if min_w > 0:
                symmetry = (
                    1.0
                    - np.abs(
                        left[:, :min_w].astype(float) - right_flipped[:, :min_w].astype(float)
                    ).mean()
                )
            else:
                symmetry = 0.5

            # Complexity (number of curvature peaks)
            complexity = np.sum(np.abs(np.diff(region_curv)) > 0.1) / len(region_curv)

            shape = ShapeDescriptor(
                curvature=region_curv[:20] if len(region_curv) > 20 else region_curv,
                angles=np.array([0.0]),  # Simplified
                aspect_ratio=aspect_ratio,
                convexity=convexity,
                symmetry=symmetry,
                complexity=complexity,
            )
            shapes.append(shape)

        return shapes

    def statistics(self) -> Dict[str, Any]:
        """Get V4 statistics."""
        return {
            "n_curvature_cells": self.n_curvature_cells,
            "n_shape_cells": self.n_shape_cells,
            "n_templates": len(self._shape_templates),
        }


class ITProcessor:
    """
    Inferotemporal (IT) cortex processing.

    Implements:
    - View-invariant object representation
    - Categorical coding
    - Part-based recognition
    """

    def __init__(
        self,
        n_categories: int = 20,
        embedding_dim: int = 128,
        invariance_pool_size: int = 3,
    ):
        self.n_categories = n_categories
        self.embedding_dim = embedding_dim
        self.invariance_pool_size = invariance_pool_size

        # Category prototypes (learned)
        self._prototypes: Dict[str, np.ndarray] = {}
        self._prototype_names = []

        # Initialize with basic categories
        self._initialize_categories()

    def _initialize_categories(self):
        """Initialize basic object category prototypes."""
        np.random.seed(42)  # For reproducibility

        categories = [
            "face",
            "car",
            "house",
            "animal",
            "tool",
            "plant",
            "body",
            "food",
            "furniture",
            "vehicle",
        ]

        for cat in categories:
            # Random prototype embedding
            proto = np.random.randn(self.embedding_dim)
            proto = proto / np.linalg.norm(proto)
            self._prototypes[cat] = proto
            self._prototype_names.append(cat)

    def process(self, v4_output: V4Output) -> ITOutput:
        """
        Process V4 output through IT cortex.

        Args:
            v4_output: V4Output from V4 processing

        Returns:
            ITOutput with object-level representations
        """
        h, w = v4_output.size

        # Create pooled representation (view invariance via pooling)
        pooled_shape = self._invariance_pooling(v4_output.shape_map)
        pooled_curvature = self._invariance_pooling(v4_output.curvature_map)

        # Generate embedding from intermediate forms
        if v4_output.intermediate_forms:
            embedding = self._forms_to_embedding(v4_output.intermediate_forms)
        else:
            # Fallback: use pooled features
            embedding = np.concatenate(
                [
                    pooled_shape.flatten()[: self.embedding_dim // 2],
                    pooled_curvature.flatten()[: self.embedding_dim // 2],
                ]
            )
            if len(embedding) < self.embedding_dim:
                embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
            elif len(embedding) > self.embedding_dim:
                embedding = embedding[: self.embedding_dim]
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        # Compute category responses
        object_responses = self._category_responses(embedding)

        # Detect objects
        detected_objects = self._detect_objects(
            embedding,
            object_responses,
            v4_output.intermediate_forms,
        )

        # Create identity embeddings map
        identity_embeddings = np.zeros((h, w, self.embedding_dim))
        # (Simplified: same embedding everywhere objects detected)
        if detected_objects:
            for obj in detected_objects:
                identity_embeddings += obj.embedding

        return ITOutput(
            object_responses=object_responses,
            identity_embeddings=identity_embeddings,
            detected_objects=detected_objects,
            size=(h, w),
        )

    def _invariance_pooling(self, feature_map: np.ndarray) -> np.ndarray:
        """
        Pool features for view/position invariance.

        Uses max pooling over spatial neighborhoods.
        """
        from scipy.ndimage import maximum_filter

        pooled = maximum_filter(feature_map, size=self.invariance_pool_size)
        return pooled

    def _forms_to_embedding(
        self,
        forms: List[ShapeDescriptor],
    ) -> np.ndarray:
        """
        Convert shape descriptors to embedding vector.
        """
        embedding = np.zeros(self.embedding_dim)

        for i, form in enumerate(forms[:5]):  # Use up to 5 forms
            offset = i * (self.embedding_dim // 5)

            # Encode shape properties
            props = [
                form.aspect_ratio,
                form.convexity,
                form.symmetry,
                form.complexity,
            ]

            for j, prop in enumerate(props):
                if offset + j < self.embedding_dim:
                    embedding[offset + j] = prop

            # Encode curvature (if available)
            if form.curvature is not None and len(form.curvature) > 0:
                curv_summary = [
                    form.curvature.mean(),
                    form.curvature.std(),
                    form.curvature.min(),
                    form.curvature.max(),
                ]
                for j, c in enumerate(curv_summary):
                    idx = offset + 4 + j
                    if idx < self.embedding_dim:
                        embedding[idx] = c

        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding

    def _category_responses(self, embedding: np.ndarray) -> np.ndarray:
        """
        Compute responses to object categories.
        """
        responses = np.zeros(len(self._prototypes))

        for i, (name, proto) in enumerate(self._prototypes.items()):
            # Cosine similarity
            sim = np.dot(embedding, proto)
            responses[i] = max(0, sim)

        return responses

    def _detect_objects(
        self,
        embedding: np.ndarray,
        category_responses: np.ndarray,
        forms: List[ShapeDescriptor],
    ) -> List[ObjectRepresentation]:
        """
        Detect and identify objects.
        """
        detected = []

        # Get top category
        if len(category_responses) > 0:
            best_idx = np.argmax(category_responses)
            best_response = category_responses[best_idx]

            if best_response > 0.3:  # Threshold
                obj = ObjectRepresentation(
                    identity=self._prototype_names[best_idx],
                    confidence=float(best_response),
                    features={
                        "n_forms": len(forms),
                        "category_idx": int(best_idx),
                    },
                    embedding=embedding,
                )
                detected.append(obj)

        return detected

    def learn_category(self, name: str, examples: List[np.ndarray]) -> None:
        """
        Learn a new object category from examples.
        """
        if not examples:
            return

        # Average embedding
        mean_embedding = np.mean(examples, axis=0)
        mean_embedding = mean_embedding / (np.linalg.norm(mean_embedding) + 1e-8)

        self._prototypes[name] = mean_embedding
        if name not in self._prototype_names:
            self._prototype_names.append(name)

    def statistics(self) -> Dict[str, Any]:
        """Get IT statistics."""
        return {
            "n_categories": len(self._prototypes),
            "embedding_dim": self.embedding_dim,
            "categories": self._prototype_names,
        }
