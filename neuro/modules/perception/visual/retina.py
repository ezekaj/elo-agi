"""
Retina: Early visual processing.

Implements photoreceptor responses, center-surround processing,
and ganglion cell encoding.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple
from enum import Enum
import numpy as np


class PhotoreceptorType(Enum):
    """Types of photoreceptors."""

    ROD = "rod"  # Low-light, achromatic
    CONE_L = "cone_l"  # Long wavelength (red)
    CONE_M = "cone_m"  # Medium wavelength (green)
    CONE_S = "cone_s"  # Short wavelength (blue)


class GanglionCellType(Enum):
    """Types of retinal ganglion cells."""

    MAGNO = "magno"  # Motion, luminance (M pathway)
    PARVO = "parvo"  # Color, fine detail (P pathway)
    KONIO = "konio"  # Blue-yellow (K pathway)


@dataclass
class PhotoreceptorResponse:
    """Response from photoreceptors."""

    luminance: np.ndarray  # Overall brightness
    l_cone: np.ndarray  # Long wavelength response
    m_cone: np.ndarray  # Medium wavelength response
    s_cone: np.ndarray  # Short wavelength response
    rod: Optional[np.ndarray] = None  # Rod response (low light)


@dataclass
class RetinaOutput:
    """Output from retinal processing."""

    on_center: np.ndarray  # ON-center ganglion cell responses
    off_center: np.ndarray  # OFF-center ganglion cell responses
    magno: np.ndarray  # Magnocellular pathway (motion)
    parvo: np.ndarray  # Parvocellular pathway (detail)
    red_green: np.ndarray  # Red-green opponent channel
    blue_yellow: np.ndarray  # Blue-yellow opponent channel
    luminance: np.ndarray  # Achromatic luminance
    size: Tuple[int, int] = (0, 0)


class Retina:
    """
    Retinal processing model.

    Implements:
    - Photoreceptor responses
    - Horizontal cell lateral inhibition
    - Center-surround receptive fields
    - Ganglion cell encoding
    - Color opponent channels
    """

    def __init__(
        self,
        center_size: float = 1.0,
        surround_size: float = 3.0,
        adaptation_rate: float = 0.1,
    ):
        self.center_size = center_size
        self.surround_size = surround_size
        self.adaptation_rate = adaptation_rate

        # Adaptation state
        self._adaptation_level = 1.0
        self._previous_input = None

        # Ganglion cell parameters
        self._magno_temporal_filter = self._create_temporal_filter(fast=True)
        self._parvo_temporal_filter = self._create_temporal_filter(fast=False)

    def _create_temporal_filter(self, fast: bool = True) -> np.ndarray:
        """Create temporal filter for ganglion cells."""
        if fast:
            # Fast, transient response (magno)
            t = np.linspace(0, 0.1, 10)
            kernel = t * np.exp(-t / 0.02)
        else:
            # Slow, sustained response (parvo)
            t = np.linspace(0, 0.2, 20)
            kernel = np.exp(-t / 0.05)

        return kernel / kernel.sum()

    def process(self, image: np.ndarray) -> RetinaOutput:
        """
        Process an image through the retina.

        Args:
            image: Input image, shape (H, W) for grayscale or (H, W, 3) for RGB

        Returns:
            RetinaOutput with ganglion cell activations
        """
        # Ensure 3-channel input
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)

        # Normalize to 0-1
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0

        h, w = image.shape[:2]

        # Step 1: Photoreceptor responses
        photo_response = self._photoreceptor_response(image)

        # Step 2: Horizontal cell processing (lateral inhibition)
        adapted = self._horizontal_cell_processing(photo_response)

        # Step 3: Center-surround processing
        on_center, off_center = self._center_surround(adapted.luminance)

        # Step 4: Color opponent channels
        red_green = self._opponent_channel(adapted.l_cone, adapted.m_cone)
        blue_yellow = self._opponent_channel(adapted.s_cone, (adapted.l_cone + adapted.m_cone) / 2)

        # Step 5: Ganglion cell pathways
        magno = self._magno_pathway(on_center, off_center)
        parvo = self._parvo_pathway(on_center, off_center, red_green)

        # Update adaptation
        self._update_adaptation(photo_response.luminance)

        return RetinaOutput(
            on_center=on_center,
            off_center=off_center,
            magno=magno,
            parvo=parvo,
            red_green=red_green,
            blue_yellow=blue_yellow,
            luminance=adapted.luminance,
            size=(h, w),
        )

    def _photoreceptor_response(self, image: np.ndarray) -> PhotoreceptorResponse:
        """Compute photoreceptor responses."""
        # Extract RGB channels
        r, g, b = image[..., 0], image[..., 1], image[..., 2]

        # Cone responses (simplified spectral sensitivity)
        l_cone = 0.7 * r + 0.3 * g
        m_cone = 0.3 * r + 0.7 * g
        s_cone = b

        # Luminance
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

        # Log transformation (Weber-Fechner law)
        epsilon = 1e-6
        l_cone = np.log(l_cone + epsilon) - np.log(epsilon)
        m_cone = np.log(m_cone + epsilon) - np.log(epsilon)
        s_cone = np.log(s_cone + epsilon) - np.log(epsilon)
        luminance = np.log(luminance + epsilon) - np.log(epsilon)

        return PhotoreceptorResponse(
            luminance=luminance,
            l_cone=l_cone,
            m_cone=m_cone,
            s_cone=s_cone,
        )

    def _horizontal_cell_processing(
        self,
        photo_response: PhotoreceptorResponse,
    ) -> PhotoreceptorResponse:
        """Apply horizontal cell lateral inhibition."""
        # Gaussian blur for lateral inhibition
        from scipy.ndimage import gaussian_filter

        sigma = self.surround_size

        # Subtract local mean (adaptation)
        def adapt(channel):
            local_mean = gaussian_filter(channel, sigma)
            return channel - self.adaptation_rate * local_mean

        return PhotoreceptorResponse(
            luminance=adapt(photo_response.luminance),
            l_cone=adapt(photo_response.l_cone),
            m_cone=adapt(photo_response.m_cone),
            s_cone=adapt(photo_response.s_cone),
        )

    def _center_surround(
        self,
        luminance: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute center-surround receptive field responses.

        Returns ON-center and OFF-center responses.
        """
        from scipy.ndimage import gaussian_filter

        # Center (small Gaussian)
        center = gaussian_filter(luminance, self.center_size)

        # Surround (larger Gaussian)
        surround = gaussian_filter(luminance, self.surround_size)

        # Difference of Gaussians
        on_center = np.maximum(0, center - surround)  # ON-center, OFF-surround
        off_center = np.maximum(0, surround - center)  # OFF-center, ON-surround

        return on_center, off_center

    def _opponent_channel(
        self,
        excitatory: np.ndarray,
        inhibitory: np.ndarray,
    ) -> np.ndarray:
        """Compute color opponent channel."""
        return excitatory - inhibitory

    def _magno_pathway(
        self,
        on_center: np.ndarray,
        off_center: np.ndarray,
    ) -> np.ndarray:
        """
        Magnocellular pathway processing.

        Combines ON and OFF responses, sensitive to motion/change.
        """
        # Transient response - difference from previous
        current = on_center + off_center

        if self._previous_input is not None:
            temporal_diff = np.abs(current - self._previous_input)
        else:
            temporal_diff = current

        # Store for next frame
        self._previous_input = current.copy()

        return temporal_diff

    def _parvo_pathway(
        self,
        on_center: np.ndarray,
        off_center: np.ndarray,
        color_opponent: np.ndarray,
    ) -> np.ndarray:
        """
        Parvocellular pathway processing.

        High spatial frequency, color sensitive.
        """
        # Combine luminance edges with color
        luminance_edge = on_center + off_center
        return luminance_edge + 0.5 * np.abs(color_opponent)

    def _update_adaptation(self, luminance: np.ndarray) -> None:
        """Update light adaptation level."""
        mean_lum = luminance.mean()
        self._adaptation_level = (
            1 - self.adaptation_rate
        ) * self._adaptation_level + self.adaptation_rate * mean_lum

    def get_receptive_field_map(
        self,
        size: Tuple[int, int] = (64, 64),
    ) -> np.ndarray:
        """Generate a visualization of center-surround receptive fields."""
        from scipy.ndimage import gaussian_filter

        np.zeros(size)
        center = (size[0] // 2, size[1] // 2)

        # Create impulse
        impulse = np.zeros(size)
        impulse[center] = 1.0

        # Apply DoG
        center_response = gaussian_filter(impulse, self.center_size)
        surround_response = gaussian_filter(impulse, self.surround_size)

        return center_response - surround_response

    def statistics(self) -> Dict[str, Any]:
        """Get retina statistics."""
        return {
            "center_size": self.center_size,
            "surround_size": self.surround_size,
            "adaptation_level": self._adaptation_level,
        }
