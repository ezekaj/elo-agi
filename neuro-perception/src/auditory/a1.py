"""
A1: Primary Auditory Cortex processing.

Implements tonotopic organization, spectrotemporal receptive fields,
and basic auditory feature extraction.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


@dataclass
class SpectrotemporalRF:
    """Spectrotemporal receptive field (STRF)."""
    kernel: np.ndarray  # (n_freq, n_time) filter
    best_frequency: float
    best_rate: float  # Preferred modulation rate
    bandwidth: float
    direction: str  # 'up', 'down', or 'flat'


@dataclass
class A1Output:
    """Output from A1 processing."""
    tonotopic_map: np.ndarray      # Frequency-organized responses
    rate_map: np.ndarray           # Temporal modulation responses
    scale_map: np.ndarray          # Spectral modulation responses
    onset_map: np.ndarray          # Onset/offset detection
    strf_responses: np.ndarray     # Full STRF responses
    size: Tuple[int, int] = (0, 0)  # (n_freq, n_time)


class A1Processor:
    """
    Primary Auditory Cortex (A1) processing.

    Implements:
    - Tonotopic organization
    - Spectrotemporal receptive fields (STRFs)
    - Rate (temporal modulation) analysis
    - Scale (spectral modulation) analysis
    - Onset detection
    """

    def __init__(
        self,
        n_rates: int = 8,
        n_scales: int = 8,
        max_rate: float = 32.0,  # Hz
        max_scale: float = 8.0,  # cycles/octave
    ):
        self.n_rates = n_rates
        self.n_scales = n_scales
        self.max_rate = max_rate
        self.max_scale = max_scale

        # Create STRF bank
        self._strfs = self._create_strf_bank()

        # Onset detection kernel
        self._onset_kernel = self._create_onset_kernel()

    def _create_strf_bank(self) -> List[SpectrotemporalRF]:
        """Create bank of STRFs with different rate/scale tuning."""
        strfs = []

        rates = np.logspace(0, np.log10(self.max_rate), self.n_rates)
        scales = np.logspace(0, np.log10(self.max_scale), self.n_scales)

        for rate in rates:
            for scale in scales:
                for direction in ['up', 'down']:
                    strf = self._create_strf(rate, scale, direction)
                    strfs.append(strf)

        return strfs

    def _create_strf(
        self,
        rate: float,
        scale: float,
        direction: str,
        kernel_size: Tuple[int, int] = (11, 21),
    ) -> SpectrotemporalRF:
        """
        Create a single STRF.

        Uses a Gabor-like function in spectrotemporal domain.
        """
        n_freq, n_time = kernel_size

        # Create meshgrid
        t = np.linspace(-1, 1, n_time)
        f = np.linspace(-1, 1, n_freq)
        T, F = np.meshgrid(t, f)

        # Temporal modulation (rate)
        sigma_t = 1.0 / (rate + 1)

        # Spectral modulation (scale)
        sigma_f = 1.0 / (scale + 1)

        # Direction (upward vs downward sweep)
        if direction == 'up':
            angle = np.pi / 4
        else:
            angle = -np.pi / 4

        # Rotated coordinates
        T_rot = T * np.cos(angle) - F * np.sin(angle)
        F_rot = T * np.sin(angle) + F * np.cos(angle)

        # Gabor function
        gaussian = np.exp(-(T_rot**2 / (2 * sigma_t**2) + F_rot**2 / (2 * sigma_f**2)))
        sinusoid = np.cos(2 * np.pi * (rate * T + scale * F))

        kernel = gaussian * sinusoid
        kernel = kernel - kernel.mean()
        kernel = kernel / (np.sqrt(np.sum(kernel**2)) + 1e-8)

        return SpectrotemporalRF(
            kernel=kernel,
            best_frequency=0.0,  # Will be set during processing
            best_rate=rate,
            bandwidth=sigma_f,
            direction=direction,
        )

    def _create_onset_kernel(self, size: int = 11) -> np.ndarray:
        """Create onset detection kernel."""
        # Difference of Gaussians in time
        t = np.linspace(-2, 2, size)

        on_kernel = np.exp(-t**2 / 0.5)
        off_kernel = np.exp(-(t - 0.5)**2 / 0.5)

        kernel = on_kernel - 0.5 * off_kernel
        kernel = kernel - kernel.mean()

        return kernel

    def process(self, cochlea_output) -> A1Output:
        """
        Process cochlear output through A1.

        Args:
            cochlea_output: CochleaOutput from cochlear processing

        Returns:
            A1Output with cortical responses
        """
        from scipy.ndimage import convolve, convolve1d

        # Get auditory nerve representation
        an_response = cochlea_output.auditory_nerve
        n_freq, n_time = an_response.shape

        # Tonotopic map (just the input, organized by frequency)
        tonotopic_map = an_response.copy()

        # Apply STRFs
        n_strfs = len(self._strfs)
        strf_responses = np.zeros((n_strfs, n_freq, n_time))

        for i, strf in enumerate(self._strfs):
            # 2D convolution with STRF kernel
            response = convolve(an_response, strf.kernel, mode='constant')
            strf_responses[i] = np.maximum(0, response)

        # Rate map (max response across scales for each rate)
        rate_map = np.zeros((self.n_rates, n_freq, n_time))
        for ri in range(self.n_rates):
            # Each rate has n_scales * 2 (up/down) STRFs
            start_idx = ri * self.n_scales * 2
            end_idx = start_idx + self.n_scales * 2
            rate_map[ri] = np.max(strf_responses[start_idx:end_idx], axis=0)

        # Scale map (max response across rates for each scale)
        scale_map = np.zeros((self.n_scales, n_freq, n_time))
        for si in range(self.n_scales):
            indices = []
            for ri in range(self.n_rates):
                base = ri * self.n_scales * 2
                indices.extend([base + si * 2, base + si * 2 + 1])
            scale_map[si] = np.max(strf_responses[indices], axis=0)

        # Onset detection
        onset_map = np.zeros((n_freq, n_time))
        for fi in range(n_freq):
            onset_response = convolve1d(an_response[fi], self._onset_kernel, mode='constant')
            onset_map[fi] = np.maximum(0, onset_response)

        return A1Output(
            tonotopic_map=tonotopic_map,
            rate_map=np.max(rate_map, axis=0),  # Collapse rate dimension
            scale_map=np.max(scale_map, axis=0),  # Collapse scale dimension
            onset_map=onset_map,
            strf_responses=strf_responses,
            size=(n_freq, n_time),
        )

    def get_modulation_spectrum(
        self,
        a1_output: A1Output,
    ) -> Dict[str, np.ndarray]:
        """
        Compute modulation power spectrum from STRF responses.
        """
        # Average STRF responses over time and frequency
        n_strfs = a1_output.strf_responses.shape[0]

        rate_responses = np.zeros(self.n_rates)
        scale_responses = np.zeros(self.n_scales)

        for ri in range(self.n_rates):
            start_idx = ri * self.n_scales * 2
            end_idx = start_idx + self.n_scales * 2
            rate_responses[ri] = np.mean(a1_output.strf_responses[start_idx:end_idx])

        for si in range(self.n_scales):
            indices = []
            for ri in range(self.n_rates):
                base = ri * self.n_scales * 2
                indices.extend([base + si * 2, base + si * 2 + 1])
            scale_responses[si] = np.mean(a1_output.strf_responses[indices])

        return {
            "rate_spectrum": rate_responses,
            "scale_spectrum": scale_responses,
            "rates": np.logspace(0, np.log10(self.max_rate), self.n_rates),
            "scales": np.logspace(0, np.log10(self.max_scale), self.n_scales),
        }

    def statistics(self) -> Dict[str, Any]:
        """Get A1 statistics."""
        return {
            "n_strfs": len(self._strfs),
            "n_rates": self.n_rates,
            "n_scales": self.n_scales,
            "max_rate": self.max_rate,
            "max_scale": self.max_scale,
        }
