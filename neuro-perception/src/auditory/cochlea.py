"""
Cochlea: Early auditory processing.

Implements frequency decomposition via basilar membrane simulation,
hair cell transduction, and auditory nerve encoding.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np


@dataclass
class HairCellResponse:
    """Response from inner hair cells."""
    firing_rate: np.ndarray  # (n_channels,) firing rates
    phase_locked: np.ndarray  # Phase-locked timing information
    adaptation_state: np.ndarray  # Current adaptation level


@dataclass
class CochleaOutput:
    """Output from cochlear processing."""
    basilar_membrane: np.ndarray  # (n_channels, n_time) displacement
    inner_hair_cells: np.ndarray  # IHC responses
    auditory_nerve: np.ndarray    # AN fiber firing rates
    center_frequencies: np.ndarray  # CF for each channel
    time_axis: np.ndarray         # Time points


class GammatoneFilterbank:
    """
    Gammatone filterbank for cochlear frequency analysis.

    Approximates the frequency selectivity of the basilar membrane.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_channels: int = 64,
        low_freq: float = 80.0,
        high_freq: float = 8000.0,
    ):
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self.low_freq = low_freq
        self.high_freq = high_freq

        # Compute center frequencies (ERB scale)
        self.center_frequencies = self._erb_space(low_freq, high_freq, n_channels)

        # Compute filter coefficients
        self._coefficients = self._compute_coefficients()

    def _erb_space(
        self,
        low: float,
        high: float,
        n: int,
    ) -> np.ndarray:
        """
        Generate frequencies on ERB (Equivalent Rectangular Bandwidth) scale.
        """
        # ERB formula constants
        ear_q = 9.26449
        min_bw = 24.7

        # Convert to ERB scale
        low_erb = ear_q * np.log(1 + low / (min_bw * ear_q))
        high_erb = ear_q * np.log(1 + high / (min_bw * ear_q))

        # Linear spacing in ERB
        erb_points = np.linspace(low_erb, high_erb, n)

        # Convert back to Hz
        frequencies = (np.exp(erb_points / ear_q) - 1) * min_bw * ear_q

        return frequencies

    def _compute_coefficients(self) -> List[Dict[str, np.ndarray]]:
        """Compute gammatone filter coefficients."""
        coefficients = []

        for cf in self.center_frequencies:
            # ERB at this frequency
            erb = 24.7 * (4.37 * cf / 1000 + 1)

            # Gammatone parameters
            b = 1.019 * 2 * np.pi * erb

            coefficients.append({
                'cf': cf,
                'b': b,
                'erb': erb,
            })

        return coefficients

    def process(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply gammatone filterbank to signal.

        Args:
            signal: Input audio signal (n_samples,)

        Returns:
            Filtered output (n_channels, n_samples)
        """
        n_samples = len(signal)
        output = np.zeros((self.n_channels, n_samples))

        # Time vector
        t = np.arange(n_samples) / self.sample_rate

        for i, coef in enumerate(self._coefficients):
            cf = coef['cf']
            b = coef['b']

            # Gammatone impulse response (simplified)
            # h(t) = t^(n-1) * exp(-2*pi*b*t) * cos(2*pi*cf*t)
            n_order = 4
            t_ir = np.arange(int(0.05 * self.sample_rate)) / self.sample_rate

            ir = (t_ir ** (n_order - 1)) * np.exp(-2 * np.pi * b * t_ir)
            ir = ir * np.cos(2 * np.pi * cf * t_ir)
            ir = ir / (np.sum(ir ** 2) + 1e-8) ** 0.5  # Normalize

            # Convolve
            filtered = np.convolve(signal, ir, mode='same')
            output[i, :] = filtered

        return output


class Cochlea:
    """
    Cochlear processing model.

    Implements:
    - Basilar membrane frequency decomposition
    - Inner hair cell transduction
    - Auditory nerve encoding
    - Adaptation and compression
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_channels: int = 64,
        low_freq: float = 80.0,
        high_freq: float = 8000.0,
        compression_power: float = 0.3,
    ):
        self.sample_rate = sample_rate
        self.n_channels = n_channels
        self.compression_power = compression_power

        # Gammatone filterbank
        self.filterbank = GammatoneFilterbank(
            sample_rate=sample_rate,
            n_channels=n_channels,
            low_freq=low_freq,
            high_freq=high_freq,
        )

        # Adaptation state
        self._adaptation = np.ones(n_channels)
        self._adaptation_rate = 0.01

    def process(self, signal: np.ndarray) -> CochleaOutput:
        """
        Process audio signal through cochlea.

        Args:
            signal: Input audio (n_samples,) or (n_samples, n_channels)

        Returns:
            CochleaOutput with basilar membrane and neural responses
        """
        # Handle stereo input
        if signal.ndim == 2:
            signal = signal.mean(axis=1)

        # Normalize
        if signal.max() > 1.0:
            signal = signal / 32768.0  # Assume 16-bit

        n_samples = len(signal)
        time_axis = np.arange(n_samples) / self.sample_rate

        # Step 1: Basilar membrane (filterbank)
        basilar_membrane = self.filterbank.process(signal)

        # Step 2: Half-wave rectification (hair cell transduction)
        rectified = np.maximum(0, basilar_membrane)

        # Step 3: Compression (outer hair cell gain control)
        compressed = np.sign(rectified) * np.abs(rectified) ** self.compression_power

        # Step 4: Inner hair cell response (low-pass + adaptation)
        ihc_response = self._inner_hair_cell(compressed)

        # Step 5: Auditory nerve encoding
        auditory_nerve = self._auditory_nerve_encoding(ihc_response)

        return CochleaOutput(
            basilar_membrane=basilar_membrane,
            inner_hair_cells=ihc_response,
            auditory_nerve=auditory_nerve,
            center_frequencies=self.filterbank.center_frequencies,
            time_axis=time_axis,
        )

    def _inner_hair_cell(self, bm_response: np.ndarray) -> np.ndarray:
        """
        Inner hair cell transduction with adaptation.
        """
        # Low-pass filter (simplified)
        from scipy.ndimage import uniform_filter1d

        # Temporal integration
        window = int(0.002 * self.sample_rate)  # 2ms window
        smoothed = uniform_filter1d(bm_response, size=max(1, window), axis=1)

        # Apply adaptation
        adapted = smoothed * self._adaptation[:, None]

        # Update adaptation state
        mean_response = np.mean(np.abs(smoothed), axis=1)
        self._adaptation = (
            (1 - self._adaptation_rate) * self._adaptation +
            self._adaptation_rate / (1 + mean_response)
        )

        return adapted

    def _auditory_nerve_encoding(self, ihc_response: np.ndarray) -> np.ndarray:
        """
        Encode IHC response as auditory nerve firing rates.
        """
        # Sigmoid nonlinearity for rate-level function
        max_rate = 200.0  # Max firing rate (spikes/s)
        threshold = 0.1

        # Rate = max_rate * sigmoid(input)
        rates = max_rate / (1 + np.exp(-10 * (ihc_response - threshold)))

        return rates

    def get_spectrogram(
        self,
        cochlea_output: CochleaOutput,
        frame_size: int = 256,
        hop_size: int = 128,
    ) -> np.ndarray:
        """
        Convert cochlear output to spectrogram-like representation.
        """
        n_channels, n_samples = cochlea_output.auditory_nerve.shape
        n_frames = (n_samples - frame_size) // hop_size + 1

        spectrogram = np.zeros((n_channels, n_frames))

        for i in range(n_frames):
            start = i * hop_size
            end = start + frame_size
            spectrogram[:, i] = np.mean(
                cochlea_output.auditory_nerve[:, start:end], axis=1
            )

        return spectrogram

    def statistics(self) -> Dict[str, Any]:
        """Get cochlear statistics."""
        return {
            "sample_rate": self.sample_rate,
            "n_channels": self.n_channels,
            "freq_range": (
                self.filterbank.low_freq,
                self.filterbank.high_freq,
            ),
            "compression": self.compression_power,
        }
