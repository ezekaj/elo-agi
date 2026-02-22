"""
Binding: Cross-modal integration and feature binding.

Implements temporal binding, spatial binding, and unified
percept formation across sensory modalities.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import numpy as np


class Modality(Enum):
    """Sensory modalities."""

    VISUAL = "visual"
    AUDITORY = "auditory"
    TACTILE = "tactile"
    PROPRIOCEPTIVE = "proprioceptive"


@dataclass
class ModalityInput:
    """Input from a single modality."""

    modality: Modality
    features: np.ndarray  # Feature vector
    spatial_location: Optional[Tuple[float, float, float]] = None
    timestamp: float = 0.0
    confidence: float = 1.0


@dataclass
class BoundPercept:
    """A unified percept binding multiple modalities."""

    id: str
    modalities: Set[Modality]
    unified_features: np.ndarray
    spatial_location: Optional[Tuple[float, float, float]]
    temporal_window: Tuple[float, float]
    binding_strength: float
    component_inputs: List[ModalityInput] = field(default_factory=list)


@dataclass
class BindingOutput:
    """Output from binding process."""

    percepts: List[BoundPercept]
    binding_map: np.ndarray  # (n_inputs, n_inputs) binding strength
    coherence: float  # Overall binding coherence


class TemporalBinder:
    """
    Bind events based on temporal proximity.

    Implements the temporal binding window hypothesis.
    """

    def __init__(
        self,
        binding_window: float = 0.1,  # 100ms temporal binding window
        decay_rate: float = 10.0,
    ):
        self.binding_window = binding_window
        self.decay_rate = decay_rate

    def compute_temporal_binding(
        self,
        inputs: List[ModalityInput],
    ) -> np.ndarray:
        """
        Compute temporal binding strength between inputs.

        Returns:
            (n_inputs, n_inputs) matrix of binding strengths
        """
        n = len(inputs)
        binding = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                dt = abs(inputs[i].timestamp - inputs[j].timestamp)

                # Exponential decay with window
                if dt < self.binding_window:
                    binding[i, j] = np.exp(-self.decay_rate * dt)
                else:
                    binding[i, j] = 0.0

        return binding


class SpatialBinder:
    """
    Bind events based on spatial proximity.

    Maps different modality coordinate systems to common space.
    """

    def __init__(
        self,
        spatial_sigma: float = 0.1,  # Spatial binding width
    ):
        self.spatial_sigma = spatial_sigma

        # Coordinate system transforms (modality -> common space)
        self._transforms: Dict[Modality, np.ndarray] = {}

    def set_transform(self, modality: Modality, transform: np.ndarray) -> None:
        """Set coordinate transform for a modality."""
        self._transforms[modality] = transform

    def to_common_space(
        self,
        location: Tuple[float, float, float],
        modality: Modality,
    ) -> np.ndarray:
        """Transform modality-specific location to common space."""
        loc = np.array(location)

        if modality in self._transforms:
            transform = self._transforms[modality]
            # Apply affine transform
            loc = transform[:3, :3] @ loc + transform[:3, 3]

        return loc

    def compute_spatial_binding(
        self,
        inputs: List[ModalityInput],
    ) -> np.ndarray:
        """
        Compute spatial binding strength between inputs.

        Returns:
            (n_inputs, n_inputs) matrix of binding strengths
        """
        n = len(inputs)
        binding = np.ones((n, n))

        for i in range(n):
            if inputs[i].spatial_location is None:
                continue

            loc_i = self.to_common_space(
                inputs[i].spatial_location,
                inputs[i].modality,
            )

            for j in range(n):
                if inputs[j].spatial_location is None:
                    continue

                loc_j = self.to_common_space(
                    inputs[j].spatial_location,
                    inputs[j].modality,
                )

                # Euclidean distance
                dist = np.linalg.norm(loc_i - loc_j)

                # Gaussian binding
                binding[i, j] = np.exp(-(dist**2) / (2 * self.spatial_sigma**2))

        return binding


class FeatureBinder:
    """
    Bind features across modalities into unified representations.
    """

    def __init__(
        self,
        unified_dim: int = 256,
    ):
        self.unified_dim = unified_dim

        # Projection matrices for each modality
        self._projections: Dict[Modality, np.ndarray] = {}

    def set_projection(self, modality: Modality, projection: np.ndarray) -> None:
        """Set feature projection for a modality."""
        self._projections[modality] = projection

    def project_to_unified(
        self,
        features: np.ndarray,
        modality: Modality,
    ) -> np.ndarray:
        """Project modality features to unified space."""
        if modality in self._projections:
            proj = self._projections[modality]
            # Handle dimension mismatch
            if features.shape[0] != proj.shape[1]:
                # Pad or truncate
                if features.shape[0] < proj.shape[1]:
                    features = np.pad(features, (0, proj.shape[1] - features.shape[0]))
                else:
                    features = features[: proj.shape[1]]
            unified = proj @ features
        else:
            # Default: pad/truncate to unified_dim
            if len(features) < self.unified_dim:
                unified = np.pad(features, (0, self.unified_dim - len(features)))
            else:
                unified = features[: self.unified_dim]

        # Normalize
        unified = unified / (np.linalg.norm(unified) + 1e-8)

        return unified

    def combine_features(
        self,
        inputs: List[ModalityInput],
        binding_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Combine features from multiple inputs using binding weights.

        Args:
            inputs: List of modality inputs
            binding_weights: (n_inputs,) weights for combination

        Returns:
            Unified feature vector
        """
        unified = np.zeros(self.unified_dim)

        total_weight = 0.0
        for inp, weight in zip(inputs, binding_weights):
            proj = self.project_to_unified(inp.features, inp.modality)
            unified += weight * inp.confidence * proj
            total_weight += weight * inp.confidence

        if total_weight > 0:
            unified = unified / total_weight

        return unified


class CrossModalBinder:
    """
    Complete cross-modal binding system.

    Integrates temporal, spatial, and feature binding to create
    unified percepts from multimodal input.
    """

    def __init__(
        self,
        binding_window: float = 0.1,
        spatial_sigma: float = 0.1,
        unified_dim: int = 256,
        binding_threshold: float = 0.3,
    ):
        self.binding_threshold = binding_threshold

        self.temporal_binder = TemporalBinder(binding_window=binding_window)
        self.spatial_binder = SpatialBinder(spatial_sigma=spatial_sigma)
        self.feature_binder = FeatureBinder(unified_dim=unified_dim)

        self._percept_counter = 0

    def bind(self, inputs: List[ModalityInput]) -> BindingOutput:
        """
        Bind multimodal inputs into unified percepts.

        Args:
            inputs: List of inputs from various modalities

        Returns:
            BindingOutput with bound percepts
        """
        if not inputs:
            return BindingOutput(percepts=[], binding_map=np.array([]), coherence=0.0)

        len(inputs)

        # Compute binding strengths
        temporal_binding = self.temporal_binder.compute_temporal_binding(inputs)
        spatial_binding = self.spatial_binder.compute_spatial_binding(inputs)

        # Combined binding (product of temporal and spatial)
        binding_map = temporal_binding * spatial_binding

        # Find clusters of bound inputs
        percepts = self._cluster_percepts(inputs, binding_map)

        # Compute overall coherence
        coherence = self._compute_coherence(binding_map)

        return BindingOutput(
            percepts=percepts,
            binding_map=binding_map,
            coherence=coherence,
        )

    def _cluster_percepts(
        self,
        inputs: List[ModalityInput],
        binding_map: np.ndarray,
    ) -> List[BoundPercept]:
        """Cluster inputs into bound percepts."""
        n = len(inputs)
        assigned = [False] * n
        percepts = []

        for i in range(n):
            if assigned[i]:
                continue

            # Find all inputs bound to this one
            cluster_indices = [i]
            assigned[i] = True

            for j in range(n):
                if not assigned[j] and binding_map[i, j] > self.binding_threshold:
                    cluster_indices.append(j)
                    assigned[j] = True

            # Create percept from cluster
            cluster_inputs = [inputs[k] for k in cluster_indices]
            binding_weights = np.array([binding_map[i, k] for k in cluster_indices])

            percept = self._create_percept(cluster_inputs, binding_weights)
            percepts.append(percept)

        return percepts

    def _create_percept(
        self,
        inputs: List[ModalityInput],
        binding_weights: np.ndarray,
    ) -> BoundPercept:
        """Create a bound percept from clustered inputs."""
        self._percept_counter += 1

        # Collect modalities
        modalities = {inp.modality for inp in inputs}

        # Combine features
        unified_features = self.feature_binder.combine_features(inputs, binding_weights)

        # Compute spatial location (weighted average)
        spatial_location = self._compute_location(inputs, binding_weights)

        # Temporal window
        timestamps = [inp.timestamp for inp in inputs]
        temporal_window = (min(timestamps), max(timestamps))

        # Binding strength (average pairwise binding)
        binding_strength = float(binding_weights.mean())

        return BoundPercept(
            id=f"percept_{self._percept_counter}",
            modalities=modalities,
            unified_features=unified_features,
            spatial_location=spatial_location,
            temporal_window=temporal_window,
            binding_strength=binding_strength,
            component_inputs=inputs,
        )

    def _compute_location(
        self,
        inputs: List[ModalityInput],
        weights: np.ndarray,
    ) -> Optional[Tuple[float, float, float]]:
        """Compute weighted average location."""
        locations = []
        loc_weights = []

        for inp, w in zip(inputs, weights):
            if inp.spatial_location is not None:
                loc = self.spatial_binder.to_common_space(
                    inp.spatial_location,
                    inp.modality,
                )
                locations.append(loc)
                loc_weights.append(w)

        if not locations:
            return None

        locations = np.array(locations)
        loc_weights = np.array(loc_weights)
        loc_weights = loc_weights / loc_weights.sum()

        avg_loc = (locations.T @ loc_weights).flatten()

        return tuple(avg_loc)

    def _compute_coherence(self, binding_map: np.ndarray) -> float:
        """Compute overall binding coherence."""
        if binding_map.size == 0:
            return 0.0

        # Coherence = average binding strength
        n = binding_map.shape[0]
        if n <= 1:
            return 1.0

        # Exclude diagonal
        mask = ~np.eye(n, dtype=bool)
        coherence = binding_map[mask].mean()

        return float(coherence)

    def statistics(self) -> Dict[str, Any]:
        """Get binder statistics."""
        return {
            "n_percepts_created": self._percept_counter,
            "binding_threshold": self.binding_threshold,
            "temporal_window": self.temporal_binder.binding_window,
            "spatial_sigma": self.spatial_binder.spatial_sigma,
        }
