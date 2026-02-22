"""
V1 and V2: Primary and secondary visual cortex processing.

Implements orientation selectivity, spatial frequency filtering,
and basic feature detection.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


@dataclass
class OrientationColumn:
    """A column of orientation-selective cells."""

    preferred_angle: float  # In radians
    bandwidth: float = 0.5  # Tuning width in radians
    spatial_frequency: float = 0.1
    phase: float = 0.0
    response: float = 0.0


@dataclass
class SpatialFrequencyFilter:
    """A spatial frequency filter (like a Gabor)."""

    frequency: float
    orientation: float
    sigma: float
    phase: float = 0.0
    kernel: Optional[np.ndarray] = None


@dataclass
class V1Output:
    """Output from V1 processing."""

    orientation_map: np.ndarray  # (H, W, n_orientations)
    frequency_map: np.ndarray  # (H, W, n_frequencies)
    edge_map: np.ndarray  # Combined edge detection
    complex_cells: np.ndarray  # Phase-invariant responses
    size: Tuple[int, int] = (0, 0)


@dataclass
class V2Output:
    """Output from V2 processing."""

    contour_map: np.ndarray  # Illusory contours, border ownership
    texture_map: np.ndarray  # Texture segregation
    corner_map: np.ndarray  # Corner/junction detection
    curvature_map: np.ndarray  # Curvature estimation
    depth_cues: np.ndarray  # Monocular depth cues
    size: Tuple[int, int] = (0, 0)


class V1Processor:
    """
    Primary visual cortex (V1) processing.

    Implements:
    - Gabor filter bank for orientation/frequency detection
    - Simple cells (phase-sensitive)
    - Complex cells (phase-invariant)
    - Orientation columns
    """

    def __init__(
        self,
        n_orientations: int = 8,
        n_frequencies: int = 4,
        min_frequency: float = 0.05,
        max_frequency: float = 0.4,
        gabor_sigma: float = 3.0,
    ):
        self.n_orientations = n_orientations
        self.n_frequencies = n_frequencies
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.gabor_sigma = gabor_sigma

        # Create filter bank
        self._filters = self._create_gabor_bank()
        self._kernels = None  # Lazy initialization

    def _create_gabor_bank(self) -> List[SpatialFrequencyFilter]:
        """Create bank of Gabor filters."""
        filters = []

        orientations = np.linspace(0, np.pi, self.n_orientations, endpoint=False)
        frequencies = np.logspace(
            np.log10(self.min_frequency), np.log10(self.max_frequency), self.n_frequencies
        )

        for freq in frequencies:
            for theta in orientations:
                filters.append(
                    SpatialFrequencyFilter(
                        frequency=freq,
                        orientation=theta,
                        sigma=self.gabor_sigma / freq,  # Scale sigma with frequency
                        phase=0.0,
                    )
                )
                # Add quadrature pair (90° phase shift)
                filters.append(
                    SpatialFrequencyFilter(
                        frequency=freq,
                        orientation=theta,
                        sigma=self.gabor_sigma / freq,
                        phase=np.pi / 2,
                    )
                )

        return filters

    def _create_gabor_kernel(
        self,
        filt: SpatialFrequencyFilter,
        size: int = 31,
    ) -> np.ndarray:
        """Create a Gabor kernel."""
        half = size // 2
        y, x = np.mgrid[-half : half + 1, -half : half + 1]

        # Rotation
        x_theta = x * np.cos(filt.orientation) + y * np.sin(filt.orientation)
        y_theta = -x * np.sin(filt.orientation) + y * np.cos(filt.orientation)

        # Gabor function
        gaussian = np.exp(-(x_theta**2 + y_theta**2) / (2 * filt.sigma**2))
        sinusoid = np.cos(2 * np.pi * filt.frequency * x_theta + filt.phase)

        kernel = gaussian * sinusoid

        # Normalize
        kernel = kernel - kernel.mean()
        kernel = kernel / (np.sqrt(np.sum(kernel**2)) + 1e-8)

        return kernel

    def _get_kernels(self, size: int = 31) -> List[np.ndarray]:
        """Get or create Gabor kernels."""
        if self._kernels is None:
            self._kernels = [self._create_gabor_kernel(f, size) for f in self._filters]
        return self._kernels

    def process(
        self,
        retina_output,
        kernel_size: int = 31,
    ) -> V1Output:
        """
        Process retina output through V1.

        Args:
            retina_output: RetinaOutput from retina processing
            kernel_size: Size of Gabor kernels

        Returns:
            V1Output with orientation and frequency maps
        """
        from scipy.ndimage import convolve

        # Get input (use luminance or combined response)
        if hasattr(retina_output, "luminance"):
            image = retina_output.luminance
        else:
            image = retina_output

        h, w = image.shape[:2]
        kernels = self._get_kernels(kernel_size)

        # Apply all filters
        len(self._filters) // 2  # Each orientation has 2 phases

        orientation_responses = np.zeros((h, w, self.n_orientations))
        frequency_responses = np.zeros((h, w, self.n_frequencies))

        # Simple cell responses
        simple_responses = []
        for kernel in kernels:
            response = convolve(image, kernel, mode="constant")
            simple_responses.append(response)

        # Complex cell responses (energy model - sum of squared quadrature pair)
        complex_responses = []
        for i in range(0, len(simple_responses), 2):
            energy = np.sqrt(simple_responses[i] ** 2 + simple_responses[i + 1] ** 2)
            complex_responses.append(energy)

        # Organize by orientation and frequency
        idx = 0
        for fi in range(self.n_frequencies):
            for oi in range(self.n_orientations):
                response = complex_responses[idx]
                orientation_responses[:, :, oi] = np.maximum(
                    orientation_responses[:, :, oi], response
                )
                frequency_responses[:, :, fi] = np.maximum(frequency_responses[:, :, fi], response)
                idx += 1

        # Edge map (max across orientations and frequencies)
        edge_map = np.max(orientation_responses, axis=2)

        # Stack complex responses
        complex_cells = np.stack(complex_responses, axis=-1)

        return V1Output(
            orientation_map=orientation_responses,
            frequency_map=frequency_responses,
            edge_map=edge_map,
            complex_cells=complex_cells,
            size=(h, w),
        )

    def get_dominant_orientation(self, v1_output: V1Output) -> np.ndarray:
        """Get dominant orientation at each location."""
        return np.argmax(v1_output.orientation_map, axis=2) * np.pi / self.n_orientations

    def statistics(self) -> Dict[str, Any]:
        """Get V1 statistics."""
        return {
            "n_orientations": self.n_orientations,
            "n_frequencies": self.n_frequencies,
            "n_filters": len(self._filters),
        }


class V2Processor:
    """
    Secondary visual cortex (V2) processing.

    Implements:
    - Border ownership
    - Contour integration
    - Texture segmentation
    - Corner/junction detection
    """

    def __init__(
        self,
        contour_length: int = 5,
        curvature_radius: int = 3,
    ):
        self.contour_length = contour_length
        self.curvature_radius = curvature_radius

    def process(self, v1_output: V1Output) -> V2Output:
        """
        Process V1 output through V2.

        Args:
            v1_output: Output from V1 processing

        Returns:
            V2Output with higher-level features
        """
        h, w = v1_output.size

        # Contour integration
        contour_map = self._contour_integration(v1_output.orientation_map)

        # Texture segregation
        texture_map = self._texture_segregation(v1_output.frequency_map)

        # Corner detection
        corner_map = self._detect_corners(v1_output.orientation_map)

        # Curvature estimation
        curvature_map = self._estimate_curvature(v1_output.orientation_map)

        # Monocular depth cues
        depth_cues = self._extract_depth_cues(
            v1_output.frequency_map,
            v1_output.edge_map,
        )

        return V2Output(
            contour_map=contour_map,
            texture_map=texture_map,
            corner_map=corner_map,
            curvature_map=curvature_map,
            depth_cues=depth_cues,
            size=(h, w),
        )

    def _contour_integration(
        self,
        orientation_map: np.ndarray,
    ) -> np.ndarray:
        """
        Integrate local orientations into contours.

        Uses co-circularity constraint.
        """

        h, w, n_ori = orientation_map.shape

        # Get edge strength and orientation
        edge_strength = np.max(orientation_map, axis=2)
        dominant_ori = np.argmax(orientation_map, axis=2)

        # Contour integration via orientation consistency
        contour_map = np.zeros((h, w))

        # Simple approach: boost edges that have consistent neighbors
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue

                # Shift and compare orientations
                shifted_ori = np.roll(np.roll(dominant_ori, dy, axis=0), dx, axis=1)
                ori_diff = np.abs(dominant_ori - shifted_ori)
                consistency = 1.0 - ori_diff / (n_ori / 2)
                consistency = np.clip(consistency, 0, 1)

                contour_map += edge_strength * consistency

        return contour_map / 8.0  # Normalize by number of neighbors

    def _texture_segregation(
        self,
        frequency_map: np.ndarray,
    ) -> np.ndarray:
        """
        Segregate regions by texture differences.

        Uses frequency content differences.
        """
        from scipy.ndimage import uniform_filter

        h, w, n_freq = frequency_map.shape

        # Local texture descriptor (frequency histogram)
        window_size = 15
        local_texture = np.zeros((h, w, n_freq))

        for i in range(n_freq):
            local_texture[:, :, i] = uniform_filter(frequency_map[:, :, i], size=window_size)

        # Texture gradient (change in texture)
        texture_gradient = np.zeros((h, w))
        for i in range(n_freq):
            gx = np.abs(np.diff(local_texture[:, :, i], axis=1, prepend=0))
            gy = np.abs(np.diff(local_texture[:, :, i], axis=0, prepend=0))
            texture_gradient += gx + gy

        return texture_gradient

    def _detect_corners(
        self,
        orientation_map: np.ndarray,
    ) -> np.ndarray:
        """
        Detect corners and junctions.

        Locations where multiple orientations meet.
        """
        h, w, n_ori = orientation_map.shape

        # Count number of significant orientations at each location
        threshold = np.max(orientation_map) * 0.3
        significant = (orientation_map > threshold).astype(float)
        n_significant = np.sum(significant, axis=2)

        # Corners have multiple orientations (2+)
        corner_map = (n_significant > 1.5).astype(float)

        # Weight by total orientation energy
        corner_map *= np.sum(orientation_map, axis=2)

        return corner_map

    def _estimate_curvature(
        self,
        orientation_map: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate local curvature from orientation changes.
        """
        dominant_ori = np.argmax(orientation_map, axis=2).astype(float)

        # Orientation gradient (rate of change)
        gx = np.abs(np.diff(dominant_ori, axis=1, prepend=0))
        gy = np.abs(np.diff(dominant_ori, axis=0, prepend=0))

        # Handle wrap-around
        n_ori = orientation_map.shape[2]
        gx = np.minimum(gx, n_ori - gx)
        gy = np.minimum(gy, n_ori - gy)

        curvature = np.sqrt(gx**2 + gy**2)

        return curvature

    def _extract_depth_cues(
        self,
        frequency_map: np.ndarray,
        edge_map: np.ndarray,
    ) -> np.ndarray:
        """
        Extract monocular depth cues.

        Uses texture gradient and edge density.
        """
        from scipy.ndimage import uniform_filter

        # Texture gradient cue (finer texture = closer)
        high_freq = frequency_map[:, :, -1]  # Highest frequency
        texture_cue = 1.0 - uniform_filter(high_freq, size=15)

        # Edge density cue
        edge_density = uniform_filter(edge_map, size=15)
        edge_cue = edge_density / (edge_density.max() + 1e-8)

        # Combine cues
        depth_cues = 0.5 * texture_cue + 0.5 * edge_cue

        return depth_cues

    def statistics(self) -> Dict[str, Any]:
        """Get V2 statistics."""
        return {
            "contour_length": self.contour_length,
            "curvature_radius": self.curvature_radius,
        }
